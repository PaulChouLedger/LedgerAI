"""The project-question detector: fires on the questions the owner is
tired of fielding, stays quiet on casual chat.

(owner, 2026-09-03: "make her defend these questions in area31 with her
RAG context... i just don't have time to deal with these questions
anymore")
"""
import py_compile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain  # noqa: E402

SHOULD_FIRE = [
    "what is ledgerai actually building?",
    "how is this any different from every other ai coin",
    "why not just use chatgpt",
    "what's the point of the token",
    "is my data safe with this thing",
    "where does my data go?",
    "how does the puck work",
    "what makes this different from alexa",
    "whats the roadmap looking like",
    "what does the token even do",
    "why do we need a token for this",
    "what are you guys building here",
    "does $ledger have a max supply?",
    "so the puck runs the model locally?",
]

SHOULD_NOT_FIRE = [
    "gm everyone",
    "lol that game last night was crazy",
    "anyone here from miami?",
    "what time is the game today?",
    "i just got a new car",
    "what's for lunch",
    "did you watch the fight?",
]


def fires(text: str) -> bool:
    t = text.translate(brain._UNICODE_NORMALIZE).lower()
    return any(p.search(t) for p in brain._PROJECT_Q_RE)


def main() -> int:
    bad = 0
    for t in SHOULD_FIRE:
        ok = fires(t)
        print(f"{'PASS' if ok else 'FAIL'}  fires: {t!r}")
        bad += 0 if ok else 1
    for t in SHOULD_NOT_FIRE:
        ok = not fires(t)
        print(f"{'PASS' if ok else 'FAIL'}  quiet: {t!r}")
        bad += 0 if ok else 1

    # bot.py must still compile with the new branches
    try:
        py_compile.compile(
            str(Path(__file__).resolve().parent.parent / "bot.py"),
            doraise=True)
        print("PASS  bot.py compiles")
    except py_compile.PyCompileError as e:
        print(f"FAIL  bot.py compile: {e}")
        bad += 1

    print("ALL PASS" if bad == 0 else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
