"""AAA event mode: any question answered while on, normal rules while off."""
import json
import py_compile
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain  # noqa: E402

QUESTIONS = [
    "what's your favorite blockchain?",
    "how old are you",
    "wen moon",
    "can you write a haiku about miami?",
    "is pizza better than tacos?",
]
NOT_QUESTIONS = [
    "gm everyone",
    "that print came out great",
    "lol",
]

CHAT = -1003025733750


def main() -> int:
    bad = 0

    # redirect the AAA file into a temp dir so the test never touches prod
    tmp = Path(tempfile.mkdtemp())
    brain.AAA_FILE = tmp / "aaa_mode.json"
    brain._aaa_cache = {"mtime": -1.0, "chats": {}}

    def check(label, got, want):
        nonlocal bad
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        bad += 0 if ok else 1

    check("off: aaa_active false (no file)", brain.aaa_active(CHAT), False)

    brain.AAA_FILE.write_text(json.dumps({str(CHAT): {"on": True}}))
    check("on: aaa_active true", brain.aaa_active(CHAT), True)
    check("on: other chat unaffected", brain.aaa_active(12345), False)

    for q in QUESTIONS:
        check(f"question detected: {q!r}",
              bool(brain._QUESTION_RE.search(q.lower().strip())), True)
    for t in NOT_QUESTIONS:
        check(f"not a question: {t!r}",
              bool(brain._QUESTION_RE.search(t.lower().strip())), False)

    # expiry honoured
    brain._aaa_cache = {"mtime": -1.0, "chats": {}}
    brain.AAA_FILE.write_text(json.dumps(
        {str(CHAT): {"on": True, "until": time.time() - 5}}))
    check("expired window: inactive", brain.aaa_active(CHAT), False)

    # off again
    brain._aaa_cache = {"mtime": -1.0, "chats": {}}
    brain.AAA_FILE.write_text(json.dumps({}))
    check("off: inactive after clear", brain.aaa_active(CHAT), False)

    try:
        py_compile.compile(str(Path(__file__).resolve().parent.parent
                               / "bot.py"), doraise=True)
        print("PASS  bot.py compiles")
    except py_compile.PyCompileError as e:
        print(f"FAIL  bot.py compile: {e}")
        bad += 1

    print("ALL PASS" if bad == 0 else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
