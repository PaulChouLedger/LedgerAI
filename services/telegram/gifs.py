"""
gifs -- GIF reactions for Aura Telegram bot.

Curated pool of reaction GIFs by mood category.  GIFs are stored locally
in data/telegram/gifs/<mood>/ so we never depend on external CDN availability.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# GIF probability: how often Aura sends a GIF after her response
GIF_PROBABILITY = 0.12  # ~12% of responses

# ---------------------------------------------------------------------------
# Load local GIF pool from disk
# ---------------------------------------------------------------------------

_GIF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "telegram" / "gifs"

_GIF_POOL: dict[str, list[str]] = {}
_ALL_GIFS: list[str] = []


def _load_pool() -> None:
    """Scan data/telegram/gifs/<mood>/*.gif and populate the pool."""
    global _ALL_GIFS
    if not _GIF_DIR.is_dir():
        log.warning("GIF directory not found: %s", _GIF_DIR)
        return
    for mood_dir in sorted(_GIF_DIR.iterdir()):
        if not mood_dir.is_dir():
            continue
        mood = mood_dir.name
        gifs = sorted(str(p) for p in mood_dir.glob("*.gif"))
        if gifs:
            _GIF_POOL[mood] = gifs
    _ALL_GIFS = [p for paths in _GIF_POOL.values() for p in paths]
    log.info("Loaded %d GIFs across %d moods from %s",
             len(_ALL_GIFS), len(_GIF_POOL), _GIF_DIR)


_load_pool()

# ---------------------------------------------------------------------------
# Mood extraction from text
# ---------------------------------------------------------------------------

_MOOD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(agree|exactly|right|true|correct|yep|yeah)\b", re.I), "agree"),
    (re.compile(r"\b(disagree|wrong|nah|nope|doubt)\b", re.I), "disagree"),
    (re.compile(r"\b(funny|hilarious|lmao|lol|joke|laughing)\b", re.I), "funny"),
    (re.compile(r"\b(wow|whoa|damn|impressive|incredible|insane)\b", re.I), "wow"),
    (re.compile(r"\b(boring|bored|meh|whatever|yawn)\b", re.I), "bored"),
    (re.compile(r"\b(think|hmm|consider|maybe|wonder)\b", re.I), "thinking"),
    (re.compile(r"\b(bye|later|see you|goodnight|peace)\b", re.I), "goodbye"),
    (re.compile(r"\b(crypto|bitcoin|token|blockchain|defi)\b", re.I), "crypto"),
    (re.compile(r"\b(roast|burn|savage|destroyed|rekt)\b", re.I), "roast"),
    (re.compile(r"\b(confused|what|huh|lost)\b", re.I), "confused"),
    (re.compile(r"\b(obviously|clearly|duh|please)\b", re.I), "eye_roll"),
    (re.compile(r"\b(slay|queen|iconic|period)\b", re.I), "sassy"),
]


def _pick_gif(text: str) -> Optional[str]:
    """Pick a GIF path that matches the mood of the text.

    2026-09-06: the no-match fallback (sassy/mic_drop) is GONE — it made
    her most neutral replies carry her most theatrical GIF ("I'm here."
    + mic drop, 4 of the last 8 sends; owner: "these mic drop gifs have
    to stop"). A GIF that doesn't match the moment is noise; no mood
    match means NO GIF, and mic_drop now fires for nothing at all.
    """
    if not _ALL_GIFS:
        return None
    for pattern, mood in _MOOD_PATTERNS:
        if pattern.search(text):
            pool = _GIF_POOL.get(mood)
            if pool:
                return random.choice(pool)
    return None


# ---------------------------------------------------------------------------
# Forced GIF triggers
# ---------------------------------------------------------------------------

_FORCE_GIF_TRIGGER = re.compile(r"\baura[,]?\s+what\s+the\s+hell\b", re.IGNORECASE)

_FORCE_GIF_RESPONSES = [
    "Don't look at me like that.",
    "I stand by what I said.",
    "You started it.",
    "Bold of you to come at me.",
    "I regret nothing.",
    "That's between me and my conscience.",
    "Oh, we're doing this now.",
    "Noted. Moving on.",
    "You say that like I'm wrong.",
    "I've been called worse by better people.",
    "Take it up with management.",
    "Cry about it.",
    "Sorry, did that hit a nerve?",
    "Welcome to the conversation.",
]


def check_force_gif(text: str) -> Optional[tuple[str, str]]:
    """Check if the message triggers a forced GIF response.

    Returns (response_text, gif_path) or None.
    """
    if not _FORCE_GIF_TRIGGER.search(text):
        return None

    pool = _GIF_POOL.get("wtf", []) + _GIF_POOL.get("sassy", []) + _GIF_POOL.get("eye_roll", [])
    if not pool:
        return None
    gif_path = random.choice(pool)
    response = random.choice(_FORCE_GIF_RESPONSES)
    log.info("Force GIF triggered")
    return (response, gif_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_get_gif(response_text: str) -> Optional[str]:
    """Roll the dice — maybe return a GIF file path based on Aura's response.

    Returns a local GIF file path ~12% of the time, or None.
    """
    if random.random() > GIF_PROBABILITY:
        return None

    path = _pick_gif(response_text)
    if path:
        log.info("GIF selected: %s", os.path.basename(path))
    return path
