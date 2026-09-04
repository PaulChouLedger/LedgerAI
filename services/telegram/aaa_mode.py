"""aaa_mode -- switch for the "Ask Aura Anything" event.

Usage:
    python3 aaa_mode.py on [hours]     # default chat: Area31, default: no expiry
    python3 aaa_mode.py off
    python3 aaa_mode.py status
    python3 aaa_mode.py on 2 --chat -100xxxx   # another chat, 2-hour window

Writes data/telegram/aaa_mode.json; the running bot re-reads it on mtime
change, so toggling takes effect within a message — no restart. While ON,
Aura answers ANY question in the chat: no cooldowns, no hourly cap.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime

from config import DATA_DIR

AAA_FILE = DATA_DIR / "telegram" / "aaa_mode.json"
AREA31 = -1003025733750


def load() -> dict:
    try:
        return json.loads(AAA_FILE.read_text())
    except (OSError, ValueError):
        return {}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chat = AREA31
    if "--chat" in sys.argv:
        chat = int(sys.argv[sys.argv.index("--chat") + 1])
    cmd = args[0] if args else "status"

    data = load()
    if cmd == "on":
        entry: dict = {"on": True, "since": time.time()}
        if len(args) > 1:
            entry["until"] = time.time() + float(args[1]) * 3600
        data[str(chat)] = entry
        AAA_FILE.parent.mkdir(parents=True, exist_ok=True)
        AAA_FILE.write_text(json.dumps(data, indent=1))
        span = (f"until {datetime.fromtimestamp(entry['until']):%H:%M}"
                if "until" in entry else "until turned off")
        print(f"AAA ON for chat {chat} ({span})")
    elif cmd == "off":
        data.pop(str(chat), None)
        AAA_FILE.parent.mkdir(parents=True, exist_ok=True)
        AAA_FILE.write_text(json.dumps(data, indent=1))
        print(f"AAA OFF for chat {chat}")
    else:
        if not data:
            print("AAA: no chats active")
        for cid, e in data.items():
            until = e.get("until")
            tail = (f" until {datetime.fromtimestamp(until):%H:%M}"
                    if until else "")
            print(f"AAA {'ON' if e.get('on') else 'off'}: chat {cid}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
