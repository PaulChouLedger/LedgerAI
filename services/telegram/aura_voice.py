"""Aura's Area31 cadence, measured off the owner's own posts. On the RTX.

Owner, 2026-08-31: "make aura tg on the main channel more active,
mimicked by my past posts on it. model it from the RTX." Then, on being
shown the audit and the three options: "you choose between the 3 options.
send it."

WHY THIS IS THE ONE THAT SHIPPED. The audit said the channel is not
quiet — 182 posts in 14 days, median gap 1.6 h, never a gap over 3.6 h,
straight through 00:00-05:00, and 180 of the 182 from one generator that
circles 13 themes and reuses 17 opening phrases verbatim. "More active"
cannot mean "more often" at that rate. What it can mean, and what the
owner named himself, is that none of it sounds like anybody.

THE CORPUS. data/tg_feed.jsonl, which handle_message writes for the
website. Filtered to the owner's ids BEFORE anything else is read, so no
other member's text is ever loaded — the corpus is his own voice, which
is what was asked for, and it keeps a public channel's traffic out of
everything downstream. data/telegram/ was the obvious place to look and
has nothing: profiles.json records "message_count": 653 for him and not
one word of it.

WHAT IS COPIED IS CADENCE, NOT IDENTITY. She is Aura. He is the owner who
teases her and threatens her with the scrap heap. The first offline run
of this model got that wrong and the result is the reason the filters
below exist: given only style rules, the 70B produced, in his voice, for
a live public channel —

    "we're working on a top secret project with NASA"
    "our team just landed a huge grant for climate research"
    "our engineers just broke the world record for fastest algorithm"

Every one passed a style check perfectly, because style and truth are
orthogonal and only one of them was being enforced. A first-person-plural
claim is the exact shape a fabricated announcement takes, so `we`, `our`
and `us` are refused outright, along with any mention of partners,
funding, records, customers or hiring.

This is the same lesson as the _NO_FABRICATION block in
content_engine.py, one level up: that one stops her inventing a ramen
place she visited, this one stops her inventing a company milestone.

DEGRADES LOUDLY. If the feed is missing or too thin, voice_block()
returns "" and off_voice() passes everything — she reverts to the old
prompt exactly as before — and it logs a warning every time. A voice
model that silently stopped applying would be indistinguishable from one
that was working, which is the failure this project keeps paying for.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger(__name__)

FEED = os.environ.get(
    "AURA_TG_FEED",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "data", "tg_feed.jsonl"))
AREA31 = -1003025733750
OWNER_IDS = {110875514, 5460850697}
MIN_CORPUS = 25
RELOAD_S = 3600           # the corpus grows; re-read hourly

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")

#: worn out by the CURRENT generator in this very channel — measured, not
#: guessed: 17 opening phrases were reused verbatim, up to 5 times each
BANNED_OPENERS = (
    "does anyone actually", "unpopular opinion", "is it just me",
    "anyone else think", "hot take", "am i the only", "what if",
    "the real tragedy", "crypto's", "gaming's",
)

#: see the header. None of these can be known to be true by a generator.
CLAIM_WORDS = re.compile(
    r"\b(nasa|partner(ship|ing|ed)?|grant|funding|raised|round|acquir|"
    r"world record|patent|fda|contract|customer|client|deal|"
    r"announc|hiring|investor|valuation|revenue|series [abc]|ipo|"
    r"university|government|military|defen[cs]e)\b", re.I)
FIRST_PERSON_PLURAL = re.compile(r"\b(we|we're|we've|our|ours|us)\b", re.I)

_CACHE: dict = {"ts": 0.0, "posts": [], "prof": None}


def _load() -> tuple[list[str], dict | None]:
    if time.time() - _CACHE["ts"] < RELOAD_S and _CACHE["prof"]:
        return _CACHE["posts"], _CACHE["prof"]
    posts: list[str] = []
    try:
        with open(FEED) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                #: owner only, this channel only, before anything else
                if (r.get("user_id") in OWNER_IDS
                        and r.get("chat_id") == AREA31 and r.get("text")):
                    posts.append(r["text"].strip())
    except OSError as e:
        log.warning("aura_voice: cannot read %s (%s) — VOICE MODEL OFF, "
                    "falling back to the generic starter prompt", FEED, e)
        _CACHE.update(ts=time.time(), posts=[], prof=None)
        return [], None
    if len(posts) < MIN_CORPUS:
        log.warning("aura_voice: only %d owner posts in the feed (need %d) "
                    "— VOICE MODEL OFF; a cadence fitted to that would "
                    "sound like %d posts", len(posts), MIN_CORPUS, len(posts))
        _CACHE.update(ts=time.time(), posts=[], prof=None)
        return [], None

    n = len(posts)
    w = sorted(len(p.split()) for p in posts)
    prof = dict(
        n=n,
        words_median=w[n // 2],
        words_p90=w[int(n * 0.9)],
        lower_start=sum(1 for p in posts if p[:1].islower()) / n,
        question=sum(1 for p in posts if p.rstrip().endswith("?")) / n,
        one_liner=sum(1 for p in posts if len(p.split()) <= 5) / n,
    )
    _CACHE.update(ts=time.time(), posts=posts, prof=prof)
    log.info("aura_voice: %d owner posts; lowercase %.0f%%, median %d words, "
             "questions %.0f%%", n, 100 * prof["lower_start"],
             prof["words_median"], 100 * prof["question"])
    return posts, prof


def voice_block() -> str:
    """Appended to the starter prompt. Empty (and logged) if no corpus."""
    posts, prof = _load()
    if not prof:
        return ""
    #: longest first shows the model his range; the tail shows his recent
    #: register. Twelve of each is enough to set cadence without the model
    #: starting to paraphrase specific messages.
    shots = sorted(posts, key=lambda t: -len(t))[:12] + posts[-12:]
    return (
        "\n\nVOICE — copy the CADENCE of the messages below. They are the "
        "channel owner's. You are NOT him: he is the one who teases you "
        "and threatens to turn you into scrap, and you answer back as "
        "yourself.\n"
        f"- start lowercase (he does {prof['lower_start']:.0%} of the time)\n"
        f"- median {prof['words_median']} words, almost never over "
        f"{prof['words_p90']}\n"
        f"- statements, not questions — he ends on '?' only "
        f"{prof['question']:.0%} of the time\n"
        f"- {prof['one_liner']:.0%} of his messages are five words or fewer; "
        "short is normal\n"
        "- dry, unbothered, a little sharp; warm to the room\n"
        "- never say 'we', 'our' or 'us' — you speak only for yourself\n"
        "- never state a fact about the company, its partners, customers, "
        "funding, hiring or records; you do not know any of those\n\n"
        + "\n".join(f"- {t}" for t in shots)
        + "\n\nCopy the cadence. Do not reuse their subjects.\n")


def off_voice(text: str) -> str | None:
    """Reason to refuse, or None. Runs at the LAST GATE before sending.

    A lull breaker is an OPTIONAL message: silence costs nothing and a bad
    one costs a lot, because it enters the group history and comes back as
    conversation context on the next lull. Refusing is always cheap.
    """
    posts, prof = _load()
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    #: these two do not depend on the corpus and are the dangerous class,
    #: so they apply even when the voice model is off
    if CLAIM_WORDS.search(t):
        return "unverifiable claim about the business"
    if FIRST_PERSON_PLURAL.search(t):
        return "speaks as the company ('we'/'our')"
    if not prof:
        return None
    for b in BANNED_OPENERS:
        if low.startswith(b):
            return f"opener worn out in this channel: {b!r}"
    if len(t.split()) > prof["words_p90"] + 6:
        return "longer than he ever writes"
    if t[:1].isupper() and prof["lower_start"] > 0.5:
        return "starts uppercase"
    return None
