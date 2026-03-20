"""
gifs -- GIF reactions for Aura Telegram bot.

Curated pool of reaction GIFs by mood category. No external API needed.
GIFs are sourced from Tenor/GIPHY via direct URLs.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

log = logging.getLogger(__name__)

# GIF probability: how often Aura sends a GIF after her response
GIF_PROBABILITY = 0.12  # ~12% of responses

# ---------------------------------------------------------------------------
# Curated GIF pool by mood category
# ---------------------------------------------------------------------------

_GIF_POOL: dict[str, list[str]] = {
    "agree": [
        "https://media.tenor.com/images/d5af5cd1f8e9f8e5c8e5c8e5c8e5c8e5/tenor.gif",
        "https://media.giphy.com/media/3oEjHV0z8S7WM4MwnK/giphy.gif",  # nodding
        "https://media.giphy.com/media/l0MYJnJQ4EiYLxvQ4/giphy.gif",  # yes nod
        "https://media.giphy.com/media/3oEdv6sy3ulljPMGdy/giphy.gif",  # thumbs up
        "https://media.giphy.com/media/26gsspfbt1HfVQ9va/giphy.gif",  # exactly
    ],
    "disagree": [
        "https://media.giphy.com/media/3o7btT1T9qpQZWhNlK/giphy.gif",  # nope
        "https://media.giphy.com/media/fXnRObM8Q0RkOmR5nf/giphy.gif",  # no way
        "https://media.giphy.com/media/l4FGuhL4U2WSOXsmI/giphy.gif",  # shake head
        "https://media.giphy.com/media/STfLOU6iRBRunMciZv/giphy.gif",  # nah
    ],
    "sassy": [
        "https://media.giphy.com/media/3o85xIO33l7RlmLR4I/giphy.gif",  # hair flip
        "https://media.giphy.com/media/l0HlvtIPdijJT1Dqw/giphy.gif",  # sassy
        "https://media.giphy.com/media/xUA7b0fN4FPzaGbwSA/giphy.gif",  # deal with it
        "https://media.giphy.com/media/3o7qDSOvkaCER9CgKY/giphy.gif",  # bye
        "https://media.giphy.com/media/l0HlPystfePnAI3G8/giphy.gif",  # smirk
    ],
    "funny": [
        "https://media.giphy.com/media/10JhviFuU2gWD6/giphy.gif",  # laughing
        "https://media.giphy.com/media/O5NyCibf93upy/giphy.gif",  # lol
        "https://media.giphy.com/media/Q7ozWVYCR0nyW2rvPW/giphy.gif",  # dead
        "https://media.giphy.com/media/l1J9EdzfOSgfyueLm/giphy.gif",  # crying laughing
    ],
    "wow": [
        "https://media.giphy.com/media/l0MYEqEzwMWFCg8rm/giphy.gif",  # surprised
        "https://media.giphy.com/media/xT0xeJpnrWC3XWblEk/giphy.gif",  # mind blown
        "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",  # shocked
        "https://media.giphy.com/media/3kzJvEciJa94SMW3hN/giphy.gif",  # whoa
    ],
    "bored": [
        "https://media.giphy.com/media/l2JehQ2GitHGdVG9Y/giphy.gif",  # yawn
        "https://media.giphy.com/media/14sLIve5MRaamu/giphy.gif",  # bored
        "https://media.giphy.com/media/gKsJUddjnpPG0/giphy.gif",  # whatever
    ],
    "thinking": [
        "https://media.giphy.com/media/a5viI92PAF89q/giphy.gif",  # thinking
        "https://media.giphy.com/media/CaiVJuZGvR8HK/giphy.gif",  # hmm
        "https://media.giphy.com/media/3o7TKTDn976rzVgky4/giphy.gif",  # pondering
    ],
    "mic_drop": [
        "https://media.giphy.com/media/3o7qDEq2bMbcbPRQ2c/giphy.gif",  # mic drop
        "https://media.giphy.com/media/l0MYy7QpDDVGVfAAw/giphy.gif",  # drop it
        "https://media.giphy.com/media/3o7TKF1fSIs1R19B8k/giphy.gif",  # boom
    ],
    "confused": [
        "https://media.giphy.com/media/WRQBXSCnEFJIuxktnw/giphy.gif",  # confused math
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",  # huh
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",  # what
    ],
    "roast": [
        "https://media.giphy.com/media/pQmWjYrz39YAg/giphy.gif",  # burn
        "https://media.giphy.com/media/cF7QqO5DYA26k/giphy.gif",  # ooh burn
        "https://media.giphy.com/media/r1HGFou3mUwMw/giphy.gif",  # savage
        "https://media.giphy.com/media/l0Iy69RBORbFfl1S0/giphy.gif",  # destruction
    ],
    "goodbye": [
        "https://media.giphy.com/media/42D3CxaINsAFemFuId/giphy.gif",  # peace
        "https://media.giphy.com/media/m9eG1qVjvN56H0MXt8/giphy.gif",  # wave
    ],
    "crypto": [
        "https://media.giphy.com/media/trN9ht5RlE3Dcwavg2/giphy.gif",  # stonks
        "https://media.giphy.com/media/n0AYAELt5C8P6rUVDk/giphy.gif",  # to the moon
        "https://media.giphy.com/media/67ThRZlYBvibtdF9JH/giphy.gif",  # money printer
    ],
    "wtf": [
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",  # excuse me
        "https://media.giphy.com/media/3o6Zt4HU9uwXmXSAuI/giphy.gif",  # wtf
        "https://media.giphy.com/media/ukGm72ZLZvYfS/giphy.gif",  # side eye
        "https://media.giphy.com/media/CDJo4EgHwbaPS/giphy.gif",  # concerned
        "https://media.giphy.com/media/l0IypeKl9NJanC3Ek/giphy.gif",  # shocked face
    ],
    "eye_roll": [
        "https://media.giphy.com/media/Rhhr8D5mKSX7O/giphy.gif",  # eye roll
        "https://media.giphy.com/media/sbwjM9VRh0mLm/giphy.gif",  # sure jan
        "https://media.giphy.com/media/1zSz5MVw4zKg0/giphy.gif",  # ok sure
    ],
}

# Flatten for fallback random pick
_ALL_GIFS = [url for urls in _GIF_POOL.values() for url in urls]

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


def _pick_gif(text: str) -> str:
    """Pick a GIF URL that matches the mood of the text."""
    for pattern, mood in _MOOD_PATTERNS:
        if pattern.search(text):
            pool = _GIF_POOL.get(mood, _ALL_GIFS)
            return random.choice(pool)
    # No mood match — pick from sassy or mic_drop (good defaults for Aura)
    return random.choice(_GIF_POOL["sassy"] + _GIF_POOL["mic_drop"])


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
    "That's between me and my processors.",
    "Oh, we're doing this now.",
    "Noted. Moving on.",
    "You say that like I'm wrong.",
    "I've been called worse by better hardware.",
    "Take it up with my architect.",
    "Cry about it.",
    "Sorry, did that hit a nerve?",
    "Welcome to the conversation.",
]


def check_force_gif(text: str) -> Optional[tuple[str, str]]:
    """Check if the message triggers a forced GIF response.

    Returns (response_text, gif_url) or None.
    """
    if not _FORCE_GIF_TRIGGER.search(text):
        return None

    gif_url = random.choice(_GIF_POOL["wtf"] + _GIF_POOL["sassy"] + _GIF_POOL["eye_roll"])
    response = random.choice(_FORCE_GIF_RESPONSES)
    log.info("Force GIF triggered")
    return (response, gif_url)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_get_gif(response_text: str) -> Optional[str]:
    """Roll the dice — maybe return a GIF URL based on Aura's response.

    Returns a GIF URL ~12% of the time, or None.
    """
    if random.random() > GIF_PROBABILITY:
        return None

    url = _pick_gif(response_text)
    log.info("GIF selected: %s", url[:80])
    return url
