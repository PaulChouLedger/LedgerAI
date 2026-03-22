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
        "https://media.giphy.com/media/gVoBC0SuaHStq/giphy.gif",  # robert downey jr nod
        "https://media.giphy.com/media/NEvPzZ8bd1V4Y/giphy.gif",  # obama not bad
        "https://media.giphy.com/media/l1J9wXoC8W4JFmREY/giphy.gif",  # clapping
        "https://media.giphy.com/media/10Jpr9KSaXLchW/giphy.gif",  # fist pump yes
    ],
    "disagree": [
        "https://media.giphy.com/media/3o7btT1T9qpQZWhNlK/giphy.gif",  # nope
        "https://media.giphy.com/media/fXnRObM8Q0RkOmR5nf/giphy.gif",  # no way
        "https://media.giphy.com/media/l4FGuhL4U2WSOXsmI/giphy.gif",  # shake head
        "https://media.giphy.com/media/STfLOU6iRBRunMciZv/giphy.gif",  # nah
        "https://media.giphy.com/media/VcWnY3R6YWVtC/giphy.gif",  # no no no
        "https://media.giphy.com/media/26tOXgoz0WNQhwb04/giphy.gif",  # hard no
        "https://media.giphy.com/media/xn8xD14FtUGQgXxST5/giphy.gif",  # denied
        "https://media.giphy.com/media/VHC5FCjK49dZe/giphy.gif",  # absolutely not
    ],
    "sassy": [
        "https://media.giphy.com/media/3o85xIO33l7RlmLR4I/giphy.gif",  # hair flip
        "https://media.giphy.com/media/l0HlvtIPdijJT1Dqw/giphy.gif",  # sassy
        "https://media.giphy.com/media/xUA7b0fN4FPzaGbwSA/giphy.gif",  # deal with it
        "https://media.giphy.com/media/3o7qDSOvkaCER9CgKY/giphy.gif",  # bye
        "https://media.giphy.com/media/l0HlPystfePnAI3G8/giphy.gif",  # smirk
        "https://media.giphy.com/media/31wVvW0sOur7O/giphy.gif",  # beyonce hair flip
        "https://media.giphy.com/media/GeX7gAlaR947S/giphy.gif",  # snapping fingers
        "https://media.giphy.com/media/xThuWhoaNyNBjTGERa/giphy.gif",  # attitude walk
        "https://media.giphy.com/media/gZ8emTQmTrWQE/giphy.gif",  # too cool
    ],
    "funny": [
        "https://media.giphy.com/media/10JhviFuU2gWD6/giphy.gif",  # laughing
        "https://media.giphy.com/media/O5NyCibf93upy/giphy.gif",  # lol
        "https://media.giphy.com/media/Q7ozWVYCR0nyW2rvPW/giphy.gif",  # dead
        "https://media.giphy.com/media/l1J9EdzfOSgfyueLm/giphy.gif",  # crying laughing
        "https://media.giphy.com/media/GpyS1lJXJYupG/giphy.gif",  # wheeze laugh
        "https://media.giphy.com/media/l0ExayQDzrI2xOb8A/giphy.gif",  # can't stop laughing
        "https://media.giphy.com/media/XHeLeuirRbwptHhSWd/giphy.gif",  # losing it
    ],
    "wow": [
        "https://media.giphy.com/media/l0MYEqEzwMWFCg8rm/giphy.gif",  # surprised
        "https://media.giphy.com/media/xT0xeJpnrWC3XWblEk/giphy.gif",  # mind blown
        "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",  # shocked
        "https://media.giphy.com/media/3kzJvEciJa94SMW3hN/giphy.gif",  # whoa
        "https://media.giphy.com/media/75ZaxapnyMp2w/giphy.gif",  # jaw drop
        "https://media.giphy.com/media/OK27wINdQS5YQ/giphy.gif",  # blown away
        "https://media.giphy.com/media/5aLrlDiJPMPFS/giphy.gif",  # speechless
        "https://media.giphy.com/media/5VKbvrjxpVJCM/giphy.gif",  # holy cow
    ],
    "bored": [
        "https://media.giphy.com/media/l2JehQ2GitHGdVG9Y/giphy.gif",  # yawn
        "https://media.giphy.com/media/14sLIve5MRaamu/giphy.gif",  # bored
        "https://media.giphy.com/media/gKsJUddjnpPG0/giphy.gif",  # whatever
        "https://media.giphy.com/media/h41uUhjop8NJCI7CQX/giphy.gif",  # over it
        "https://media.giphy.com/media/rq6c5xD7leHW8/giphy.gif",  # crickets
        "https://media.giphy.com/media/tXL4FHPSnVJ0A/giphy.gif",  # meh
    ],
    "thinking": [
        "https://media.giphy.com/media/a5viI92PAF89q/giphy.gif",  # thinking
        "https://media.giphy.com/media/CaiVJuZGvR8HK/giphy.gif",  # hmm
        "https://media.giphy.com/media/3o7TKTDn976rzVgky4/giphy.gif",  # pondering
        "https://media.giphy.com/media/d3mlE7uhX8KFgEmY/giphy.gif",  # chin stroke
        "https://media.giphy.com/media/DfSXiR60W9MVq/giphy.gif",  # calculating
        "https://media.giphy.com/media/y3QOvy7xxMwKI/giphy.gif",  # processing
    ],
    "mic_drop": [
        "https://media.giphy.com/media/3o7qDEq2bMbcbPRQ2c/giphy.gif",  # mic drop
        "https://media.giphy.com/media/l0MYy7QpDDVGVfAAw/giphy.gif",  # drop it
        "https://media.giphy.com/media/3o7TKF1fSIs1R19B8k/giphy.gif",  # boom
        "https://media.giphy.com/media/15BuyagtKucHm/giphy.gif",  # obama mic drop
        "https://media.giphy.com/media/3o7qDSOvfaCO9b3MlO/giphy.gif",  # walk away
        "https://media.giphy.com/media/DfbpTbQ9TvSX6/giphy.gif",  # drop the bass
    ],
    "confused": [
        "https://media.giphy.com/media/WRQBXSCnEFJIuxktnw/giphy.gif",  # confused math
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",  # huh
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",  # what
        "https://media.giphy.com/media/ji6zzUZwNIuLS/giphy.gif",  # confused travolta
        "https://media.giphy.com/media/lkdH8FmImcGoylv3t3/giphy.gif",  # wait what
        "https://media.giphy.com/media/hzrvwvnbgIV6E/giphy.gif",  # head scratch
    ],
    "roast": [
        "https://media.giphy.com/media/pQmWjYrz39YAg/giphy.gif",  # burn
        "https://media.giphy.com/media/cF7QqO5DYA26k/giphy.gif",  # ooh burn
        "https://media.giphy.com/media/r1HGFou3mUwMw/giphy.gif",  # savage
        "https://media.giphy.com/media/l0Iy69RBORbFfl1S0/giphy.gif",  # destruction
        "https://media.giphy.com/media/ZUwjT4TrkElu8/giphy.gif",  # apply cold water
        "https://media.giphy.com/media/Ke4eKC7hYSU1O/giphy.gif",  # ooh snap
        "https://media.giphy.com/media/xT1XGU1AHz9Fe8tmp2/giphy.gif",  # flames
    ],
    "goodbye": [
        "https://media.giphy.com/media/42D3CxaINsAFemFuId/giphy.gif",  # peace
        "https://media.giphy.com/media/m9eG1qVjvN56H0MXt8/giphy.gif",  # wave
        "https://media.giphy.com/media/UQaRUOLveyjNC/giphy.gif",  # peace out
        "https://media.giphy.com/media/IL7hXX77O5OIU/giphy.gif",  # fade away
        "https://media.giphy.com/media/mBdbauuNxUpnqr1B1u/giphy.gif",  # see ya
    ],
    "crypto": [
        "https://media.giphy.com/media/trN9ht5RlE3Dcwavg2/giphy.gif",  # stonks
        "https://media.giphy.com/media/n0AYAELt5C8P6rUVDk/giphy.gif",  # to the moon
        "https://media.giphy.com/media/67ThRZlYBvibtdF9JH/giphy.gif",  # money printer
        "https://media.giphy.com/media/JpG2A9P3dPHXaTYrwu/giphy.gif",  # diamond hands
        "https://media.giphy.com/media/MFsqcBSoOKPbjtmvWz/giphy.gif",  # charts go up
        "https://media.giphy.com/media/YnkMcHgNIMW4Yfmjxr/giphy.gif",  # money rain
        "https://media.giphy.com/media/bMycGOQLESDCEnLNUz/giphy.gif",  # pump it
    ],
    "wtf": [
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",  # excuse me
        "https://media.giphy.com/media/3o6Zt4HU9uwXmXSAuI/giphy.gif",  # wtf
        "https://media.giphy.com/media/ukGm72ZLZvYfS/giphy.gif",  # side eye
        "https://media.giphy.com/media/CDJo4EgHwbaPS/giphy.gif",  # concerned
        "https://media.giphy.com/media/l0IypeKl9NJanC3Ek/giphy.gif",  # shocked face
        "https://media.giphy.com/media/4cQSQYz0a9x9S/giphy.gif",  # blinking guy
        "https://media.giphy.com/media/pPhyAv5t9V8djyRFJH/giphy.gif",  # excuse me what
        "https://media.giphy.com/media/H5C8CevNMbpBqNqFjl/giphy.gif",  # come again
        "https://media.giphy.com/media/Wwn5NKv4At2CIc8XQa/giphy.gif",  # hold up
    ],
    "eye_roll": [
        "https://media.giphy.com/media/Rhhr8D5mKSX7O/giphy.gif",  # eye roll
        "https://media.giphy.com/media/sbwjM9VRh0mLm/giphy.gif",  # sure jan
        "https://media.giphy.com/media/1zSz5MVw4zKg0/giphy.gif",  # ok sure
        "https://media.giphy.com/media/eUrE2DuMKOE0g/giphy.gif",  # massive eye roll
        "https://media.giphy.com/media/B4ORVnBvJCVvq/giphy.gif",  # oh please
        "https://media.giphy.com/media/qmfpjpAT2fJRK/giphy.gif",  # k bye
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
