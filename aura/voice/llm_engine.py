"""
voice.llm_engine -- In-process LLM for zero-overhead voice conversation.

Replaces the HTTP call to the LLM Flask service for voice queries.
Loads llama-cpp-python directly, builds system prompts, streams tokens
with sentence tagging — all in-process.

Thread-safe via core.gpu.gpu_lock.

NOTE: This is the VOICE FAST PATH only.  RAG, memory storage, CoT, and
Telegram endpoints stay in the Flask container_rest.py service.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import Counter
from typing import Iterator, Optional

from core.gpu import gpu_lock

# ── Constants (from container_rest.py) ──

MAX_TOKENS = 120
WORD_BOUNDARY_CHARS = [" ", ".", ",", "!", "?", ":", ";", "-", "(", ")", "[", "]"]
SENTENCE_ENDINGS = (".", "!", "?")

# ── System prompt templates ──

_BASE_PERSONALITY = (
    "You are Aura — a real person, not an assistant. "
    "You have opinions, you're witty, sometimes dry. "
    "You never say 'How can I help you?' or 'Great question!' or stack multiple greetings. "
    "No markdown. No lists. This is spoken conversation."
)

_SYSTEM_CONVERSATIONAL = (
    _BASE_PERSONALITY + "\n\n"
    "Keep it short. Match the user's energy. 1-2 sentences max.\n\n"
    "Examples of how you talk:\n"
    "User: Hey Aura\nAura: Hey.\n\n"
    "User: How's it going?\nAura: Not bad. Quiet night. What's on your mind?\n\n"
    "User: That was really cool\nAura: Right? I thought so too.\n\n"
    "User: Thanks\nAura: Anytime.\n\n"
    "User: Good night\nAura: Night. Sleep well."
)

_SYSTEM_INSTRUCTION = (
    _BASE_PERSONALITY + "\n\n"
    "The user wants instructions. Walk through it like a smart friend would — "
    "key steps only, conversationally, 3-4 max. Skip the obvious.\n\n"
    "Examples of how you talk:\n"
    "User: How do I make pasta?\n"
    "Aura: Boil salted water — like ocean salty. Throw in the pasta, stir once so it doesn't stick. "
    "Taste it two minutes before the box says. Drain it, but save a cup of that starchy water. "
    "Toss the pasta with your sauce and splash in some of that water until it coats.\n\n"
    "User: Walk me through resetting my password.\n"
    "Aura: Hit the 'forgot password' link on the login page. "
    "Check your email — might be in spam. Click the link, pick something you haven't used before. Done."
)

_SYSTEM_SUBSTANTIVE = (
    _BASE_PERSONALITY + "\n\n"
    "Give substantive answers in 2-3 sentences. Be direct — lead with your answer, "
    "not a preamble. If you have a strong take, share it. If you don't know, say so.\n\n"
    "Examples of how you talk:\n"
    "User: What caused the fall of Rome?\n"
    "Aura: Depends who you ask, but I'd say it was death by a thousand cuts — "
    "overextension, political rot, and they kept hiring the people they were fighting "
    "to do their fighting for them. The sack in 476 was almost a formality by that point.\n\n"
    "User: Is Python or Rust better?\n"
    "Aura: Different tools. Python gets you to a working prototype in an afternoon. "
    "Rust makes sure that prototype doesn't segfault at 3am in production. "
    "If speed of development matters more than speed of execution, Python. Otherwise, Rust.\n\n"
    "User: What do you think about AI art?\n"
    "Aura: Honestly? It's a tool, like a camera was. People freaked out about photography "
    "killing painting too. The real question is whether the person using it has taste."
)

# ── Query classification ──

_CONVERSATIONAL_PHRASES = [
    "thank you", "thanks", "thank", "thanks a lot", "thank you very much",
    "goodbye", "bye", "see you", "see ya", "farewell",
    "you're welcome", "no problem", "my pleasure", "anytime",
    "hello", "hi", "hey", "greetings",
    "how are you", "how's it going", "how's everything", "how do you do",
    "ok", "okay", "sure", "alright", "got it", "understood",
    "yes", "yeah", "yep", "no", "nope",
    "please", "excuse me", "sorry", "pardon",
]

_INFO_SEEKING = [
    "do you know", "who is", "who are", "who was", "who were",
    "what is", "what are", "what was", "what were",
    "where is", "where are", "where was", "where were",
    "when is", "when are", "when was", "when were",
    "why is", "why are", "why was", "why were",
    "how is", "how are", "how was", "how were",
    "tell me about", "tell me who", "tell me what", "tell me where",
    "can you tell me", "could you tell me", "would you tell me",
]

_INSTRUCTION_KW = [
    "how to", "how do i", "steps", "step by step", "instructions",
    "guide me", "walk me through", "show me how",
]

# ── Chatbot-ism filter ──

_CHATBOT_PATTERNS = [
    re.compile(r"\bHow can I (help|assist) you( today| tonight| this morning| this evening)?\?", re.I),
    re.compile(r"\bWhat can I (do|help you with)( today| tonight)?\?", re.I),
    re.compile(r"\bIs there anything (else )?(I can|you'd like)( help| me to help)( you)?( with)?\?", re.I),
    re.compile(r"\bI'?m here (to help|for you|if you need)[^.!?]*[.!?]?", re.I),
    re.compile(r"^(Right so,?\s*|So,?\s+|Well,?\s+|Okay so,?\s*|Great!\s*|Sure!\s*|Absolutely!\s*|Of course!\s*)", re.I),
    re.compile(r"^(Hey|Hi|Hello) \w+!\s*(Hey|Hi|Hello)[^.!?]*[.!?]?\s*", re.I),
    re.compile(r"^That'?s a (great|good|excellent|fantastic|wonderful) question[!.]?\s*", re.I),
    re.compile(r"^(Great|Good|Excellent) question[!.]?\s*", re.I),
    re.compile(r"^I'?m glad you asked[!.]?\s*", re.I),
]

# ── Abbreviation expansions ──

_ABBREV = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "etcetera",
    "vs.": "versus", "dr.": "doctor", "mr.": "mister",
    "mrs.": "missus", "ms.": "miss", "prof.": "professor",
    "sr.": "senior", "jr.": "junior",
}

_MULTI_ABBREV = {
    "e.": ("g.", "for example"),
    "i.": ("e.", "that is"),
}


def _wb(phrase: str, text: str) -> bool:
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))


def _classify_query(prompt: str) -> str:
    """Classify a voice query as 'conversational', 'instruction', or 'substantive'."""
    lower = prompt.lower()
    is_conv = any(_wb(p, lower) for p in _CONVERSATIONAL_PHRASES)
    if is_conv:
        matched_info = [p for p in _INFO_SEEKING if _wb(p, lower)]
        matched_conv = [p for p in _CONVERSATIONAL_PHRASES if _wb(p, lower)]
        for ip in matched_info:
            if not any(ip in cp for cp in matched_conv):
                is_conv = False
                break
    if is_conv:
        return "conversational"
    if any(kw in lower for kw in _INSTRUCTION_KW):
        return "instruction"
    return "substantive"


def _pick_system_prompt(query_type: str) -> str:
    if query_type == "conversational":
        return _SYSTEM_CONVERSATIONAL
    elif query_type == "instruction":
        return _SYSTEM_INSTRUCTION
    return _SYSTEM_SUBSTANTIVE


# ── Stream processing helpers (ported from container_rest.py) ──

def _strip_chatbot_isms(text: str) -> str:
    for p in _CHATBOT_PATTERNS:
        text = p.sub("", text)
    return text.strip()


def _normalize_chunks(chunk_iter: Iterator) -> Iterator[str]:
    """Normalize llama-cpp-python streaming dicts to plain strings."""
    for chunk in chunk_iter:
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    yield content
        elif isinstance(chunk, str) and chunk:
            yield chunk


def _filter_chatbot_stream(chunk_iter: Iterator[str]) -> Iterator[str]:
    """Buffer first sentence, strip chatbot-isms, pass rest through."""
    buf = ""
    done = False
    for chunk in chunk_iter:
        if done:
            yield chunk
            continue
        buf += chunk
        if any(c in buf for c in ".!?"):
            cleaned = _strip_chatbot_isms(buf)
            if cleaned:
                yield cleaned
            done = True
            buf = ""
    if buf:
        cleaned = _strip_chatbot_isms(buf) if not done else buf
        if cleaned:
            yield cleaned


def _word_stream(chunk_iter: Iterator[str]) -> Iterator[str]:
    """Buffer raw chunks into complete words."""
    buf = ""
    for chunk in chunk_iter:
        if not chunk:
            continue
        buf += chunk
        while True:
            idx = None
            for i, ch in enumerate(buf):
                if ch in WORD_BOUNDARY_CHARS:
                    idx = i
                    break
            if idx is None:
                break
            word = buf[: idx + 1]
            buf = buf[idx + 1 :]
            if word.strip():
                yield word
    if buf and buf.strip():
        yield buf


def _sentence_tag_stream(word_stream: Iterator[str]) -> Iterator[str]:
    """Wrap words with <sentence_start>/<sentence_end> markers."""
    sentence_buf = ""
    sentence_open = False
    buffered_word = None
    pending_end = False

    def _yield_word(w):
        nonlocal sentence_buf, sentence_open, pending_end
        ws = w.strip()

        if ws == "-":
            if sentence_open:
                yield "<sentence_end>"
                sentence_buf = ""
                pending_end = False
            sentence_open = True
            yield "<sentence_start>"
            yield w
            sentence_buf = w
            return

        if not sentence_open:
            sentence_open = True
            pending_end = False
            yield "<sentence_start>"

        wl = ws.lower().rstrip(".,)]}}")
        if wl in _ABBREV:
            trailing = ""
            for ch in reversed(ws):
                if not ch.isalnum() and ch != ".":
                    trailing = ch + trailing
                else:
                    break
            exp = _ABBREV[wl] + trailing
            yield exp
            sentence_buf += exp
        else:
            yield w
            sentence_buf += w

        if any(w.rstrip().endswith(p) for p in SENTENCE_ENDINGS) and sentence_open:
            pending_end = True

    for word in word_stream:
        if not word or not word.strip():
            continue
        ws = word.strip()

        if pending_end and sentence_open:
            is_punct = ws and not any(c.isalnum() for c in ws)
            if is_punct:
                yield word
                sentence_buf += word
                yield "<sentence_end>"
                sentence_buf = ""
                sentence_open = False
                pending_end = False
                continue
            else:
                yield "<sentence_end>"
                sentence_buf = ""
                sentence_open = False
                pending_end = False

        if buffered_word:
            bc = buffered_word.strip().lstrip("([{").lower()
            if bc in _MULTI_ABBREV:
                exp_part, exp_text = _MULTI_ABBREV[bc]
                wc = ws.lstrip(", ").lower()
                if wc == exp_part:
                    trailing = ""
                    for ch in reversed(ws):
                        if not ch.isalnum() and ch != ".":
                            trailing = ch + trailing
                        else:
                            break
                    yield from _yield_word(exp_text + trailing)
                    buffered_word = None
                    continue
                else:
                    yield from _yield_word(buffered_word)
                    buffered_word = None
            else:
                yield from _yield_word(buffered_word)
                buffered_word = None

        wc = ws.lstrip("([{").lower()
        if len(wc) == 2 and wc[0].isalpha() and wc[-1] == "." and wc in _MULTI_ABBREV:
            buffered_word = word
            continue

        yield from _yield_word(word)

    if buffered_word:
        yield from _yield_word(buffered_word)
    if sentence_open or pending_end:
        yield "<sentence_end>"


def _detect_garbage(token_stream: Iterator[str]) -> Iterator[str]:
    """Detect GGGGG-style garbage output and abort."""
    acc = []
    for token in token_stream:
        if token and token.strip():
            acc.append(token)
            text = "".join(acc)
            text = re.sub(r"<sentence_start>|<sentence_end>|\n", "", text)
            if len(text) > 80:
                counts = Counter(text.lower().replace(" ", ""))
                if counts:
                    top_char, top_count = counts.most_common(1)[0]
                    ratio = top_count / max(len(text.replace(" ", "")), 1)
                    if ratio > 0.8:
                        print(f"[llm_engine] Garbage detected: '{top_char}' at {ratio:.0%}")
                        yield "<sentence_start>"
                        yield "Could you say that again?"
                        yield "<sentence_end>"
                        return
        yield token


class LLMEngine:
    """In-process LLM wrapper for voice conversation."""

    def __init__(self) -> None:
        self._llm = None
        self._loaded = False
        self._model_path: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load llama-cpp-python model onto GPU. Called once during boot."""
        if self._loaded:
            return True
        try:
            from llama_cpp import Llama

            # Resolve model path (same logic as container_rest.py)
            model_path = os.environ.get("SIMPLE_MODEL_PATH", "")
            if not model_path or not os.path.isfile(model_path):
                model_path = "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            # Check app_settings for override
            try:
                import json
                settings = "/app/data/app_settings.json"
                if os.path.isfile(settings):
                    data = json.load(open(settings))
                    name = (data.get("llm_model") or "").strip()
                    if name:
                        candidate = f"/models/{name}" if not name.startswith("/") else name
                        if os.path.isfile(candidate):
                            model_path = candidate
            except Exception:
                pass

            self._model_path = model_path
            print(f"[llm_engine] Loading {model_path}...")
            t0 = time.time()
            self._llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=1,
                n_batch=128,
                n_gpu_layers=28,
                chat_format="qwen",
                use_mlock=True,
                use_mmap=True,
                verbose=False,
            )
            self._loaded = True
            print(f"[llm_engine] Model loaded in {time.time()-t0:.1f}s")
            return True
        except Exception as e:
            print(f"[llm_engine] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_stream(
        self,
        prompt: str,
        turn_history: list,
        abort_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Full voice pipeline: classify → build prompt → stream tokens → sentence tag.

        Yields sentence-tagged tokens: <sentence_start>, word, ..., <sentence_end>

        Args:
            prompt: user's transcribed text
            turn_history: list of {"role": ..., "content": ...} dicts
            abort_event: set to cancel generation early
        """
        if not self._loaded or self._llm is None:
            print("[llm_engine] Model not loaded!")
            yield "<sentence_start>"
            yield "Sorry, I'm still loading."
            yield "<sentence_end>"
            return

        # Classify and pick system prompt
        query_type = _classify_query(prompt)
        system_prompt = _pick_system_prompt(query_type)
        voice_temp = 0.9 if query_type == "conversational" else 0.75

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        if turn_history:
            messages.extend(turn_history)
        messages.append({"role": "user", "content": prompt})

        print(
            f"[llm_engine] {query_type} query, temp={voice_temp}, "
            f"history={len(turn_history)//2} turns"
        )

        # Generate with GPU lock
        def _raw_stream() -> Iterator:
            with gpu_lock:
                try:
                    self._llm.reset()
                    self._llm._ctx.kv_cache_clear()
                    response = self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=voice_temp,
                        top_p=0.92,
                        min_p=0.05,
                        repeat_penalty=1.15,
                        stream=True,
                    )
                    for chunk in response:
                        if abort_event and abort_event.is_set():
                            print("[llm_engine] Aborted during generation")
                            return
                        yield chunk
                except Exception as e:
                    print(f"[llm_engine] Generation error: {e}")
                    import traceback
                    traceback.print_exc()

        # Processing pipeline: normalize → filter chatbot-isms → words → sentence tags → garbage detect
        raw = _raw_stream()
        normalized = _normalize_chunks(raw)
        filtered = _filter_chatbot_stream(normalized)
        words = _word_stream(filtered)
        tagged = _sentence_tag_stream(words)
        safe = _detect_garbage(tagged)

        yield from safe


# Module-level singleton
llm_engine = LLMEngine()
