"""WHAT THE ROOMS ARE ASKING FOR, WRITTEN WHERE A BUILDER WILL SEE IT.

Owner, 2026-08-06:

    "it goes both ways -- people are asking about the ability for our product
     to use twilio to contact loved ones, doctors, customers, etc. and that
     feedback you should know and suggest to me to queue up that integration"

`culture.py` carries language DOWN into every channel. This carries demand
back UP out of them.

The gap it closes is not technical, it is organisational. Aura hears every
conversation in a Telegram group and every word said near a puck, and until
now the only things she did with a product request were answer it politely
and forget it. Nobody building the thing ever learned that four different
people had asked for the same capability — the evidence existed, in her own
transcripts, and no path carried it to anyone who could act.

A signal is NOT a complaint (that is `feedback.py`, and it changes how she
behaves). A signal is somebody wanting the product to DO something it
cannot, which changes what gets built.

    /home/paul/aura-shared/signals.jsonl        one JSON object per line

Append-only, shared with the Aura repo, and deliberately dumb: no schema
migration, no service, nothing to be running for it to work. A file a person
can read with `tail` is a file that still works at 3am.

DEDUPED BY THEME, COUNTED BY VOICE
----------------------------------
The useful unit is not "somebody mentioned Twilio", it is "FOUR people have
asked to reach a human". So a signal carries a `theme`, and repeated asks
raise its count instead of adding noise. One person asking twice is one
voice; that distinction is the whole value of the count.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not decide. It does not open tickets, file concierge requests, or
prioritise anything. It writes down what was asked and by how many people,
and a human reads it. A system that turned overheard sentences into queued
work would be building a product from whoever talked loudest in a group
chat — and one of those groups is a pilot, not a customer base.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

SIGNALS_FILE = Path(os.environ.get(
    "AURA_SIGNALS_FILE", "/home/paul/aura-shared/signals.jsonl"))


def record(theme: str, said: str, who: str = "", where: str = "",
           channel: str = "telegram") -> None:
    """Note that somebody asked for something the product cannot do yet.

    Never raises. A signal that cannot be written must not cost a reply —
    the person is mid-conversation and this is bookkeeping.
    """
    try:
        SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SIGNALS_FILE.open("a") as f:
            f.write(json.dumps({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "theme": theme.strip().lower(),
                "said": said[:300],
                "who": who,
                "where": str(where),
                "channel": channel,
            }) + "\n")
    except Exception as exc:                       # noqa: BLE001
        log.warning("[signal] not recorded (%s): %s", theme, exc)


def summary(min_voices: int = 1) -> list[dict]:
    """Themes, most-wanted first, counted by DISTINCT PERSON.

    Sorted by voices and not by mentions, because one enthusiast repeating
    themselves is not demand and reads exactly like it is if you count rows.
    """
    themes: dict[str, dict] = {}
    try:
        for line in SIGNALS_FILE.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = themes.setdefault(r.get("theme", "?"), {
                "theme": r.get("theme", "?"), "mentions": 0,
                "voices": set(), "channels": set(), "first": r.get("at"),
                "last": r.get("at"), "example": r.get("said", "")})
            t["mentions"] += 1
            if r.get("who"):
                t["voices"].add(r["who"])
            t["channels"].add(r.get("channel", "?"))
            t["last"] = r.get("at")
    except FileNotFoundError:
        return []
    out = []
    for t in themes.values():
        t["voices"] = len(t["voices"])
        t["channels"] = sorted(t["channels"])
        if t["voices"] >= min_voices or t["mentions"] >= min_voices:
            out.append(t)
    out.sort(key=lambda t: (t["voices"], t["mentions"]), reverse=True)
    return out


def report(min_voices: int = 1) -> str:
    rows = summary(min_voices)
    if not rows:
        return "no signals recorded"
    lines = ["THEME                     VOICES  MENTIONS  CHANNELS  LAST"]
    for t in rows:
        lines.append("%-24s %6d  %8d  %-8s  %s" % (
            t["theme"][:24], t["voices"], t["mentions"],
            ",".join(t["channels"])[:8], (t["last"] or "")[:16]))
        lines.append(f'      e.g. "{t["example"][:90]}"')
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
