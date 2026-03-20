"""
gifs -- GIF search for Aura Telegram bot.

Uses the Tenor API (Google) to find contextually relevant GIFs.
Falls back gracefully if no API key or service is unavailable.
"""

from __future__ import annotations

import logging
import random
import re
import requests
from typing import Optional

from config import TENOR_API_KEY

log = logging.getLogger(__name__)

TENOR_SEARCH_URL = "https://tenor.googleapis.com/v2/search"

# GIF probability: how often Aura sends a GIF after her response
GIF_PROBABILITY = 0.12  # ~12% of responses

# Keywords to extract mood/vibe from text for GIF search
_MOOD_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\b(agree|exactly|right|true|correct)\b", re.I), ["nod yes", "exactly right", "agree"]),
    (re.compile(r"\b(disagree|wrong|nah|nope)\b", re.I), ["nope", "disagree", "shaking head no"]),
    (re.compile(r"\b(funny|hilarious|lmao|lol|joke)\b", re.I), ["laughing", "funny reaction", "lol"]),
    (re.compile(r"\b(wow|whoa|damn|impressive|incredible)\b", re.I), ["wow impressed", "mind blown", "shocked"]),
    (re.compile(r"\b(boring|bored|meh|whatever)\b", re.I), ["bored", "yawn", "whatever"]),
    (re.compile(r"\b(thanks|thank|appreciate)\b", re.I), ["you're welcome", "no problem", "thumbs up"]),
    (re.compile(r"\b(sorry|oops|my bad)\b", re.I), ["oops", "sorry", "my bad"]),
    (re.compile(r"\b(love|amazing|awesome|great)\b", re.I), ["love it", "amazing", "awesome reaction"]),
    (re.compile(r"\b(hate|terrible|awful|worst)\b", re.I), ["disgusted", "ugh", "face palm"]),
    (re.compile(r"\b(think|hmm|consider|maybe)\b", re.I), ["thinking", "hmm", "pondering"]),
    (re.compile(r"\b(wait|hold on|pause)\b", re.I), ["wait what", "hold up", "pause"]),
    (re.compile(r"\b(bye|later|see you|goodnight)\b", re.I), ["wave goodbye", "see ya", "peace out"]),
    (re.compile(r"\b(hello|hi|hey|welcome)\b", re.I), ["hello wave", "hey there", "hi"]),
    (re.compile(r"\b(confident|obviously|clearly)\b", re.I), ["confident", "obviously", "hair flip"]),
    (re.compile(r"\b(confused|what|huh)\b", re.I), ["confused", "what", "huh"]),
    (re.compile(r"\b(cheers|celebrate|congrat)\b", re.I), ["celebrate", "cheers", "party"]),
    (re.compile(r"\b(roast|burn|savage)\b", re.I), ["savage", "burn", "roasted"]),
    (re.compile(r"\b(chill|relax|calm)\b", re.I), ["chill", "relax", "calm down"]),
    (re.compile(r"\b(crypto|bitcoin|token|blockchain)\b", re.I), ["crypto", "to the moon", "stonks"]),
]

# Fallback search terms when no mood matches
_FALLBACK_SEARCHES = [
    "reaction gif", "mood", "vibe", "sassy reaction",
    "cool reaction", "mic drop", "deal with it",
]


def _extract_search_term(text: str) -> str:
    """Extract a GIF search term from Aura's response text."""
    for pattern, terms in _MOOD_PATTERNS:
        if pattern.search(text):
            return random.choice(terms)
    return random.choice(_FALLBACK_SEARCHES)


def search_gif(query: str) -> Optional[str]:
    """Search Tenor for a GIF. Returns a URL or None."""
    if not TENOR_API_KEY:
        return None

    try:
        resp = requests.get(
            TENOR_SEARCH_URL,
            params={
                "q": query,
                "key": TENOR_API_KEY,
                "client_key": "aura_telegram",
                "limit": 20,
                "media_filter": "gif",
                "contentfilter": "medium",
            },
            timeout=5,
        )
        if resp.status_code != 200:
            log.warning("Tenor search failed: HTTP %d", resp.status_code)
            return None

        results = resp.json().get("results", [])
        if not results:
            return None

        # Pick a random result from top results
        pick = random.choice(results[:10])
        # Get the gif URL from media_formats
        media = pick.get("media_formats", {})
        gif_url = (
            media.get("gif", {}).get("url")
            or media.get("mediumgif", {}).get("url")
            or media.get("tinygif", {}).get("url")
        )
        return gif_url

    except Exception as e:
        log.debug("Tenor search error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Forced GIF triggers — specific phrases that always produce a GIF
# ---------------------------------------------------------------------------

_FORCE_GIF_TRIGGER = re.compile(r"\baura[,]?\s+what\s+the\s+hell\b", re.IGNORECASE)

# Search terms for forced GIF triggers (pick randomly)
_FORCE_GIF_SEARCHES = [
    "what the hell reaction", "excuse me what", "shook",
    "side eye", "dramatic reaction", "say what",
    "confused screaming", "wtf reaction", "eye roll savage",
    "sassy what", "oh no you didn't",
]

# Pithy one-liners Aura sends with the forced GIF
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
    if not TENOR_API_KEY:
        return None
    if not _FORCE_GIF_TRIGGER.search(text):
        return None

    query = random.choice(_FORCE_GIF_SEARCHES)
    gif_url = search_gif(query)
    if not gif_url:
        return None

    response = random.choice(_FORCE_GIF_RESPONSES)
    log.info("Force GIF triggered: query='%s'", query)
    return (response, gif_url)


def maybe_get_gif(response_text: str) -> Optional[str]:
    """Roll the dice — maybe return a GIF URL based on Aura's response.

    Returns a GIF URL ~12% of the time, or None.
    """
    if not TENOR_API_KEY:
        return None

    if random.random() > GIF_PROBABILITY:
        return None

    query = _extract_search_term(response_text)
    url = search_gif(query)
    if url:
        log.info("GIF selected: query='%s' url=%s", query, url[:80])
    return url
