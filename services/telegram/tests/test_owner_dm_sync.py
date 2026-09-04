"""Owner-DM briefing export: only the owner's own words, only his thread."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
from config import OWNER_USER_IDS  # noqa: E402

OWNER = sorted(OWNER_USER_IDS)[0]

BY_DM = {
    OWNER: [
        {"ts": 1757000000, "text": "we signed the EDMG contract today, $1M",
         "is_bot": False},
        {"ts": 1757000100, "text": "That's wonderful news!", "is_bot": True},
        {"ts": 1757000200, "text": "/start", "is_bot": False},
        {"ts": 1757000300, "text": "also the white Pearl puck ships to the "
                                   "first investor this month", "is_bot": False},
    ],
    999999999: [
        {"ts": 1757000400, "text": "random stranger DM — must never appear",
         "is_bot": False},
    ],
}


def main() -> int:
    bad = 0
    base = Path(tempfile.mkdtemp())
    (base / "input").mkdir()

    def check(label, ok):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        bad += 0 if ok else 1

    n = rag._sync_owner_dm(base, BY_DM)
    out = base / "input" / "owner_briefing.txt"
    txt = out.read_text() if out.exists() else ""

    check("file written, counted as update", n == 1 and out.exists())
    check("owner's news present", "EDMG contract" in txt
          and "Pearl puck" in txt)
    check("bot replies excluded", "wonderful news" not in txt)
    check("commands excluded", "/start" not in txt)
    check("stranger DMs excluded", "random stranger" not in txt)
    check("no tg_ prefix (so project-question path can use it)",
          not out.name.startswith("tg_"))
    check("idempotent (unchanged content -> 0)",
          rag._sync_owner_dm(base, BY_DM) == 0)

    print("ALL PASS" if bad == 0 else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
