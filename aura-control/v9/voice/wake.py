"""
voice.wake -- Wake word detection (fuzzy matching + OpenWakeWord).

Extracted from carbon_demo.py's heard_wake / human_should_respond logic.
"""

from __future__ import annotations

import re
import time

from core.config import WAKE_WORDS

_WAKE_RE = re.compile(r"(?i)\b(hey\s+)?(aura|ora|oura|laura)\b[:,]?\s*")


def heard_wake(text: str) -> bool:
    low = (text or "").lower()
    if any(w in low for w in WAKE_WORDS):
        return True
    norm = "".join(ch for ch in low if ch.isalpha())
    return "aura" in norm or "ora" in norm or "oura" in norm or norm.startswith("aur")


def strip_wake(text: str) -> str:
    return _WAKE_RE.sub("", (text or "").strip()).strip()


def should_respond(text: str, last_active_ts: float, window_s: float = 14.0) -> bool:
    """Fuzzy human-like decision: should Aura respond to this utterance?"""
    if not text:
        return False
    t = text.strip().lower()

    # Recent conversation — allow quick follow-ups
    if (time.time() - last_active_ts) < window_s:
        return True

    if heard_wake(t):
        return True

    # Ignore very short noise
    if len(t.split()) <= 2 and not t.endswith("?"):
        return False

    # Question / request patterns
    if t.endswith("?"):
        return True
    if any(k in t for k in [
        "can you", "could you", "would you", "please", "help",
        "how do i", "what is", "why", "explain", "show me",
    ]):
        return True

    # Emotional cues
    if any(k in t for k in [
        "i'm stressed", "im stressed", "overwhelmed",
        "anxious", "hard day", "tired", "i feel",
    ]):
        return True

    return False
