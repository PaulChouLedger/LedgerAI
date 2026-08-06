"""THE THINGS EVERY CHANNEL SHOULD ALREADY KNOW.

Owner, 2026-08-06:

    "we share context between our conversation here in the CLI and the TG
     Bot, right? so for example, when i said the informal phrase our
     community uses about 'let's pucking go' the TG bot will know that is
     something part of the culture..."

The answer was no. Nothing read anything. She has `learned_directives.json`
(behavioural corrections she taught herself) and `profiles.json` (who people
are), and neither carries the shared vocabulary of a company — the phrases,
the in-jokes, the names for things. So a saying coined in one room was
unknown in every other, and Aura sounded like a stranger in her own company
depending on which channel you reached her through.

    "we want to unify the understanding across all communication channels"

This is that: ONE FILE, read by every channel, injected into every prompt.

WHY IT LIVES OUTSIDE BOTH REPOSITORIES
--------------------------------------
The Telegram bot is in LedgerAI; the room, the puck and the concierge are in
Aura. Culture belongs to neither and is needed by both, and the first copy
of anything in this project is fine while the SECOND copy is what rots — the
wake list, the mute matcher and the fleet addresses have each cost days that
way. So it sits at one absolute path that both read and neither owns.

    AURA_CULTURE_FILE  (default /home/paul/aura-shared/culture.json)

RELOADED ON EVERY READ, WHICH IS THE POINT
------------------------------------------
`learned_directives.json` is loaded ONCE, in `FeedbackEngine.__init__`. A
rule added at noon does nothing until the process restarts, and nobody
remembers that. The same trap has appeared four times in one day: the
Telegram mute held in memory while the JSON said otherwise, the request
queue's dedupe, the classifier guard that was committed but not running.

So this stats the file and re-reads it when the mtime moves. Add a phrase,
and the next message she sends knows it. No restart, nothing to remember.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CULTURE_FILE = Path(os.environ.get(
    "AURA_CULTURE_FILE", "/home/paul/aura-shared/culture.json"))

#: Don't stat more than this often. The prompt is built per message and a
#: busy group would otherwise stat the disk on every one.
_TTL_S = 5.0

_cache: dict = {"mtime": None, "at": 0.0, "data": {}}


def _load() -> dict:
    import time
    now = time.time()
    if now - _cache["at"] < _TTL_S:
        return _cache["data"]
    _cache["at"] = now
    try:
        m = CULTURE_FILE.stat().st_mtime
    except FileNotFoundError:
        return _cache["data"]
    if m == _cache["mtime"]:
        return _cache["data"]
    try:
        _cache["data"] = json.loads(CULTURE_FILE.read_text())
        _cache["mtime"] = m
        log.info("[culture] reloaded %s", CULTURE_FILE)
    except Exception as exc:                       # noqa: BLE001
        # A malformed culture file must not silence her — she carries on
        # with the last good copy and says so once.
        log.warning("[culture] could not read %s: %s", CULTURE_FILE, exc)
    return _cache["data"]


def block() -> str:
    """The prompt fragment. Empty string when there is nothing to say."""
    d = _load()
    lines: list[str] = []

    for p in d.get("phrases", []):
        said = p.get("phrase")
        if not said:
            continue
        mean = p.get("means", "")
        lines.append(f'- "{said}"' + (f" — {mean}" if mean else ""))

    for t in d.get("terms", []):
        term = t.get("term")
        if term:
            lines.append(f"- {term}: {t.get('means','')}")

    if not lines:
        return ""

    who = d.get("company") or "this company"
    return (
        f"\n\n[HOUSE LANGUAGE — how people actually talk at {who}]\n"
        "These are shared across every channel: the room, Telegram, the puck.\n"
        "Use them the way a colleague would — naturally, when they fit, and\n"
        "never by forcing one into a sentence to prove you know it.\n"
        + "\n".join(lines) + "\n"
    )


def add_phrase(phrase: str, means: str = "", source: str = "") -> bool:
    """Teach it something, from any channel. Returns False if already known."""
    import time
    d = _load()
    phrases = d.setdefault("phrases", [])
    if any((p.get("phrase") or "").lower() == phrase.lower() for p in phrases):
        return False
    phrases.append({"phrase": phrase, "means": means, "source": source,
                    "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
    CULTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CULTURE_FILE.write_text(json.dumps(d, indent=2))
    _cache["mtime"] = None          # force the next read to pick it up
    return True
