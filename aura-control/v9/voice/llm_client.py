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
from core.config import LLM_URL
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
    """Streams /chat-tts from the LLM container, emits bus sentences."""

    def __init__(self) -> None:
        self.base_url = LLM_URL

    def stream_chat(self, text: str, context: str = "",
                    chat_id: str = "voice_session") -> None:
        """POST to /chat-tts and emit bus events per sentence.

        Preferred path: LLM returns <sentence_start>/<sentence_end> markers.
        Fallback: punctuation + buffer size + timeout flushing.
        """
        # Determine LLM port from mode
        port = "11434"  # both medical & generic use same port
        url = f"http://localhost:{port}/chat-tts"

        bus.emit("llm.started", text=text)

        try:
            resp = requests.post(
                url,
                json={"prompt": text, "context": context, "chat_id": chat_id},
                stream=True,
                timeout=30,
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
