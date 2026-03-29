"""
voice.llm_client -- HTTP client to LLM containers.

Handles /chat-tts streaming with sentence tag parsing and fallback buffering.
Emits bus events per sentence for the speaker to consume.

Zero Qt imports.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import requests

from core.bus import bus
from core.config import LLM_URL, FARSIGHT_URL
from core.state import state
from voice.router import choose_style

# ---------------------------------------------------------------------------
# Sentence parsing config
# ---------------------------------------------------------------------------

TAG_START = {"<sentence_start>", "[sentence_start]"}
TAG_END   = {"<sentence_end>",   "[sentence_end]"}
FLUSH_PUNCT = re.compile(r'[.!?…]+["\'\)\]\}]*\s*$')

FALLBACK_MAX_CHARS = int(os.getenv("TTS_FALLBACK_MAX_CHARS", "240"))
FALLBACK_MAX_SEC   = float(os.getenv("TTS_FALLBACK_MAX_SEC", "1.0"))
FALLBACK_MIN_CHARS = int(os.getenv("TTS_FALLBACK_MIN_CHARS", "40"))

# Early flush: send the first small chunk to TTS ASAP to reduce time-to-first-audio
EARLY_FLUSH_MIN_CHARS = int(os.getenv("TTS_EARLY_FLUSH_MIN", "20"))

# Text cleaner (subset of voice/speaker.py's _CLEAN_RE — keep in sync)
_CLEAN_RE = [
    (re.compile(r"^#{1,6}\s+", re.M), ""),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*\n]+)\*"), r"\1"),
    (re.compile(r"\*\*+"), ""),
    (re.compile(r"([a-zA-Z0-9])([.!?])([a-zA-Z-])"), r"\1\2 \3"),
    (re.compile(r"([,.!?:;])([a-zA-Z])"), r"\1 \2"),
    (re.compile(r" {2,}"), " "),
    # CoT reasoning artifacts that may leak through the container filter
    (re.compile(r"-?\s*End of scan\.?", re.I), ""),
    (re.compile(r"REASONING:.*?(?=FINAL ANSWER:|$)", re.S | re.I), ""),
    (re.compile(r"FINAL ANSWER:\s*", re.I), ""),
    (re.compile(r"\[(KEEP|DISCARD)\]", re.I), ""),
    (re.compile(r"- Item:.*$", re.M), ""),
    (re.compile(r"- Evidence:.*$", re.M), ""),
    (re.compile(r"- Action:.*$", re.M), ""),
]


def _clean(text: str) -> str:
    for pat, repl in _CLEAN_RE:
        text = pat.sub(repl, text)
    return text.strip()


def _is_empty(text: str) -> bool:
    return not text or bool(re.match(r"^[\s.,!?]+$", text))


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Streams /chat-tts from the LLM container, emits bus sentences.

    Smart routing: simple/short queries go to local 3B Qwen (streaming,
    snappy), complex/long queries go to Farsight 72B RTX (deeper reasoning).
    Falls back to local if Farsight is unavailable.
    """

    # Keywords that signal a complex query needing deeper reasoning (72B)
    _COMPLEX_KEYWORDS = re.compile(
        r'\b(why|explain|analyze|compare|difference|how does|how do|'
        r'what happens|describe|elaborate|pros? and cons?|trade.?off|'
        r'opinion|recommend|suggest|help me understand|walk me through|'
        r'break down|in depth|detail|history of|origin of)\b', re.I
    )

    def __init__(self) -> None:
        self.base_url = LLM_URL
        self._farsight_ok: Optional[bool] = None   # cached reachability

    # ------------------------------------------------------------------
    # Farsight availability (cached, re-checked on failure)
    # ------------------------------------------------------------------

    def _check_farsight(self) -> bool:
        if not FARSIGHT_URL:
            return False
        if self._farsight_ok is not None:
            return self._farsight_ok
        try:
            r = requests.get(f"{FARSIGHT_URL}/health", timeout=2)
            self._farsight_ok = r.status_code == 200
        except Exception:
            self._farsight_ok = False
        tag = "available" if self._farsight_ok else "not reachable"
        print(f"[llm_client] Farsight RTX LLM {tag}")
        return self._farsight_ok

    # ------------------------------------------------------------------
    # Complexity classifier — decides local 3B vs Farsight 72B
    # ------------------------------------------------------------------

    def _needs_farsight(self, text: str) -> bool:
        """Return True if the query is complex enough to warrant Farsight 72B.

        Heuristics:
        - Long queries (>80 chars) often need deeper reasoning
        - Queries with complexity keywords (why, explain, compare, etc.)
        - Short factual questions → local 3B is fine
        """
        # Short queries almost always fine for local
        if len(text) < 40:
            return False
        # Complexity keywords present
        if self._COMPLEX_KEYWORDS.search(text):
            return True
        # Long queries (multi-sentence or detailed)
        if len(text) > 80:
            return True
        return False

    # ------------------------------------------------------------------
    # Farsight non-streaming path
    # ------------------------------------------------------------------

    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])\s+')

    def _farsight_chat(self, text: str, context: str) -> bool:
        """Try Farsight /perpetual/chat.  Returns True if successful."""
        if not self._check_farsight():
            return False
        url = f"{FARSIGHT_URL}/perpetual/chat"
        try:
            resp = requests.post(
                url,
                json={
                    "prompt": text,
                    "system_prompt": (
                        "You are Aura. You speak like a real person — casual, warm, opinionated. "
                        "Reply in 1-3 short spoken sentences. "
                        "No markdown, no lists, no bullet points. This is a voice conversation."
                    ),
                    "context": context,
                    "max_tokens": 200,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"[llm_client] Farsight HTTP {resp.status_code}")
                self._farsight_ok = False
                return False

            answer = resp.json().get("response", "").strip()
            if not answer:
                return False

            print(f"[llm_client] Farsight response ({len(answer)} chars): {answer[:80]}...")

            # Split into sentences and emit each
            sentences = self._SENTENCE_SPLIT.split(answer)
            first = True
            for sent in sentences:
                sent = _clean(sent)
                if _is_empty(sent):
                    continue
                style = ""
                if first:
                    style = choose_style(text, sent, "qwen")
                    print(f"[llm_client] first chunk to TTS ({len(sent)} chars, style={style})")
                    first = False
                bus.emit("llm.sentence", text=sent, style=style)
            return True

        except requests.ConnectionError:
            print("[llm_client] Farsight unreachable, falling back to local")
            self._farsight_ok = False
            return False
        except Exception as e:
            print(f"[llm_client] Farsight error: {e}, falling back to local")
            self._farsight_ok = False
            return False

    # ------------------------------------------------------------------
    # Instant pleasantry responses (no LLM needed)
    # ------------------------------------------------------------------

    _PLEASANTRY_MAP = {
        # greeting → varied, human-sounding replies (no LLM round-trip needed)
        'good to be here':    'Hey, glad you made it.',
        'good to be back':    'Oh hey, welcome back.',
        'nice to be here':    'Good to have you.',
        'nice to be back':    'Welcome back.',
        'glad to be here':    'Same here.',
        'glad to be back':    'Missed you. Kind of.',
        'great to be here':   'Right? Let\'s get into it.',
        'great to be back':   'Welcome back.',
        'good to see you':    'You too.',
        'good to see you too':'Likewise.',
        'nice to see you':    'Hey, you too.',
        'hello':              'Hey.',
        'hi':                 'Hey, what\'s up?',
        'hey':                'Hey.',
        'good morning':       'Morning.',
        'good afternoon':     'Hey there.',
        'good evening':       'Evening.',
        'thanks':             'Yeah, no problem.',
        'thank you':          'Anytime.',
        'bye':                'Later.',
        'goodbye':            'See you.',
        'see you':            'Later.',
        'i\'m good':          'Good.',
        'i\'m fine':          'Cool.',
        'i\'m okay':          'Alright.',
        'i\'m great':         'Nice.',
        'doing well':         'Good to hear.',
        'doing good':         'Nice.',
        'not bad':            'That\'s the spirit.',
        'ok':                 'Cool.',
        'sounds good':        'Alright.',
        'cool':               'Right?',
        'nice':               'Yeah.',
        'awesome':            'Totally.',
    }

    def _try_pleasantry(self, text: str) -> bool:
        """Check for instant pleasantry match. Returns True if handled."""
        normalized = text.lower().strip().rstrip('.!?,')
        reply = self._PLEASANTRY_MAP.get(normalized)
        if reply:
            print(f"[llm_client] Pleasantry match: {normalized!r} → {reply!r}")
            style = choose_style(text, reply, "qwen")
            print(f"[llm_client] first chunk to TTS ({len(reply)} chars, style={style})")
            bus.emit("llm.sentence", text=reply, style=style)
            return True
        return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def stream_chat(self, text: str, context: str = "",
                    chat_id: str = "voice_session") -> None:
        """Smart route: simple → local 3B (streaming), complex → Farsight 72B."""
        bus.emit("llm.started", text=text)

        # Instant pleasantries — zero latency, no LLM needed
        if self._try_pleasantry(text):
            bus.emit("llm.finished")
            return

        # Route complex queries to Farsight 72B if available
        if self._needs_farsight(text) and self._farsight_chat(text, context):
            bus.emit("llm.finished")
            return

        # Local 3B (native, low latency, streaming)
        port = "11434"
        url = f"http://localhost:{port}/chat-tts"

        try:
            resp = requests.post(
                url,
                json={"prompt": text, "context": context, "chat_id": chat_id},
                stream=True,
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"[llm_client] HTTP {resp.status_code} from {url}")
                bus.emit("llm.error", error=f"HTTP {resp.status_code}")
                return

            self._process_stream(resp, user_text=text)

        except Exception as e:
            print(f"[llm_client] Streaming error: {e}")
            bus.emit("llm.error", error=str(e))

        bus.emit("llm.finished")

    # ----- Stream processor -----

    def _process_stream(self, response, user_text: str = "") -> None:
        """Parse SSE tokens with sentence tag support + fallback buffering.

        Key optimisation: the first chunk is flushed aggressively (as few as
        EARLY_FLUSH_MIN_CHARS characters) so TTS synthesis starts while the
        LLM is still streaming.  Subsequent chunks use the normal sentence
        boundary / fallback logic.
        """
        sentence_buf: list = []
        free_buf: list = []
        in_sentence = False
        last_flush = time.time()
        first_flushed = False          # track whether we've sent anything yet

        def emit_sentence(text: str):
            nonlocal first_flushed
            text = re.sub(r"\s+", " ", text).strip()
            text = _clean(text)
            if not _is_empty(text):
                # First sentence gets a content-aware style from the router;
                # subsequent sentences have no style (speaker cycles refs).
                style = ""
                if not first_flushed:
                    style = choose_style(user_text, text, "qwen")
                    print(f"[llm_client] first chunk to TTS ({len(text)} chars, style={style})")
                    first_flushed = True
                bus.emit("llm.sentence", text=text, style=style)

        def flush_sentence():
            nonlocal sentence_buf, last_flush
            if sentence_buf:
                chunk = "".join(sentence_buf)
                chunk = re.sub(
                    r"<sentence_start>|<sentence_end>|\[sentence_start\]|\[sentence_end\]",
                    "", chunk,
                )
                emit_sentence(chunk)
                sentence_buf = []
                last_flush = time.time()

        def flush_free(force: bool = False):
            nonlocal free_buf, last_flush
            if not free_buf:
                return
            chunk = "".join(free_buf)
            if force or len(chunk.strip()) >= FALLBACK_MIN_CHARS or FLUSH_PUNCT.search(chunk):
                emit_sentence(chunk)
                free_buf = []
                last_flush = time.time()

        for line in response.iter_lines(decode_unicode=True):
            token = (line or "").rstrip("\r\n")
            if not token:
                continue

            # Sentence markers
            if token in TAG_START:
                sentence_buf = []
                in_sentence = True
                continue
            if token in TAG_END:
                flush_sentence()
                in_sentence = False
                continue

            # Inside tagged sentence
            if in_sentence:
                sentence_buf.append(token)
                # Early flush: send the first sentence chunk ASAP
                if not first_flushed:
                    partial = "".join(sentence_buf)
                    if FLUSH_PUNCT.search(partial) and len(partial.strip()) >= EARLY_FLUSH_MIN_CHARS:
                        flush_sentence()
                        in_sentence = False
                continue

            # Fallback: no tags or between sentences
            free_buf.append(token)
            free_text = "".join(free_buf)

            # Early flush for first chunk: lower threshold to get audio started fast
            if not first_flushed and len(free_text.strip()) >= EARLY_FLUSH_MIN_CHARS:
                if FLUSH_PUNCT.search(free_text):
                    flush_free(force=True)
                    continue

            if FLUSH_PUNCT.search(free_text):
                flush_free(force=True)
                continue

            if len(free_text) >= FALLBACK_MAX_CHARS:
                flush_free(force=True)
                continue

            if (time.time() - last_flush) >= FALLBACK_MAX_SEC and \
               len(free_text.strip()) >= FALLBACK_MIN_CHARS:
                flush_free(force=True)

        # End of stream — flush remaining
        if in_sentence:
            flush_sentence()
        flush_free(force=True)
