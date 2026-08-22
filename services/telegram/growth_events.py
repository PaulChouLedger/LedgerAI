"""
growth_events -- Append-only event pipeline for the TG growth/experimentation
engine (2026-08-22).

METRICS FIRST (Aura docs/PRINCIPLES.md §3: no tuning before measurement).
Every fact the adaptive strategy layer (strategy.py) or the daily report
(growth_report.py) will ever reason about lands here first, as one JSON
object per line in GROWTH_EVENTS_FILE. The file is append-only and never
rotates itself -- it is the evidence.

PRIVACY BY DESIGN: no message text is ever written here. Only ids, types,
lengths, latencies and variant tags. Raw conversation lives where it always
lived (context buffers, dm_history) and never leaves this machine. The puck
feedback loop consumes AGGREGATES from growth_report.py, never this file.

Event schema (each record also carries "ts", unix seconds):

  {"event":"chat_first_seen",  "chat_id","chat_type"}
  {"event":"user_first_seen",  "user_id","chat_id"}
  {"event":"msg_in",           "chat_id","user_id","chat_type","n_chars"}
  {"event":"msg_out",          "chat_id","user_id","kind","latency_s",
                                "n_chars","n_sentences","variant","gif"}
      kind: dm | group | welcome | start | brief | command_reply
      latency_s: user's message timestamp -> first chunk delivered
      variant: the strategy assignment dict for this chat (see strategy.py)
  {"event":"group_add",        "chat_id","by_user"}
  {"event":"group_remove",     "chat_id"}
  {"event":"member_join",      "chat_id","user_id"}
  {"event":"command",          "chat_id","user_id","command"}
  {"event":"reaction",         "chat_id","user_id","emoji","on_message_id"}
  {"event":"referral_click",   "user_id","referrer_id"}
  {"event":"referral_link_issued","user_id"}
  {"event":"share_hook_offered","chat_id","user_id"}
  {"event":"variant_assigned", "chat_id","variant"}
  {"event":"negative",         "chat_id","user_id","kind"}
      kind: complaint | stop | mute | removed

Failure policy (§1 of PRINCIPLES.md): a write that fails logs an ERROR.
This module must never raise into a handler -- losing one metric is bad,
losing the reply it was measuring is worse.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths -- stated HERE and imported by strategy.py / growth_report.py
# (PRINCIPLES.md §15: one place states a shared policy). Deliberately NOT in
# config.py: config.py carries another session's uncommitted work and a
# commit of these paths must not sweep that in (see bot.py's 78421ee6 note).
# AURA_TG_GROWTH_DIR exists so tests can point the whole engine at a
# scratch directory without touching production data.
# ---------------------------------------------------------------------------
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "telegram"
GROWTH_DIR = Path(os.environ.get("AURA_TG_GROWTH_DIR", str(_DEFAULT_DIR)))
GROWTH_EVENTS_FILE = GROWTH_DIR / "growth_events.jsonl"
GROWTH_STATE_FILE = GROWTH_DIR / "growth_state.json"          # first-seen sets
STRATEGY_ASSIGNMENTS_FILE = GROWTH_DIR / "strategy_assignments.json"
BANDIT_STATE_FILE = GROWTH_DIR / "bandit_state.json"
GROWTH_REPORTS_DIR = GROWTH_DIR / "growth_reports"

GROWTH_DIR.mkdir(parents=True, exist_ok=True)

# Master flag. Mirrors the interactivity master switch convention: one env
# var pulls the whole pipeline without touching code.
GROWTH_EVENTS_ON = os.environ.get("AURA_TG_GROWTH_EVENTS", "1") == "1"


class _FirstSeen:
    """Tiny persistent index so chat_first_seen / user_first_seen fire once."""

    def __init__(self) -> None:
        self.chats: set[int] = set()
        self.users: set[int] = set()
        try:
            if GROWTH_STATE_FILE.exists():
                d = json.loads(GROWTH_STATE_FILE.read_text())
                self.chats = set(d.get("chats", []))
                self.users = set(d.get("users", []))
        except (json.JSONDecodeError, OSError) as e:
            log.error("growth_state unreadable (%s) -- first-seen counts will "
                      "over-fire once per chat, not silently vanish", e)

    def save(self) -> None:
        try:
            GROWTH_STATE_FILE.write_text(json.dumps(
                {"chats": sorted(self.chats), "users": sorted(self.users)}))
        except OSError as e:
            log.error("growth_state save FAILED: %s", e)


_seen = _FirstSeen()


def log_event(event: str, **fields) -> None:
    """Append one event. Never raises; failure is loud in the log."""
    if not GROWTH_EVENTS_ON:
        return
    rec = {"ts": round(time.time(), 2), "event": event}
    rec.update({k: v for k, v in fields.items() if v is not None})
    try:
        with open(GROWTH_EVENTS_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        log.error("growth event write FAILED (%s): %s", e, event)


# -- convenience wrappers used from bot.py ----------------------------------

def msg_in(chat_id: int, user_id: int, chat_type: str, n_chars: int) -> None:
    if chat_id not in _seen.chats:
        _seen.chats.add(chat_id)
        _seen.save()
        log_event("chat_first_seen", chat_id=chat_id, chat_type=chat_type)
    if user_id and user_id not in _seen.users:
        _seen.users.add(user_id)
        _seen.save()
        log_event("user_first_seen", user_id=user_id, chat_id=chat_id)
    log_event("msg_in", chat_id=chat_id, user_id=user_id,
              chat_type=chat_type, n_chars=n_chars)


def msg_out(chat_id: int, user_id: int | None, kind: str, latency_s: float | None,
            text: str, variant: dict | None, gif: bool = False) -> None:
    n_sent = len([s for s in _SENT_SPLIT.split(text.strip()) if s])
    log_event("msg_out", chat_id=chat_id, user_id=user_id, kind=kind,
              latency_s=round(latency_s, 2) if latency_s is not None else None,
              n_chars=len(text), n_sentences=n_sent, variant=variant, gif=gif)


def negative(chat_id: int, user_id: int | None, kind: str) -> None:
    log_event("negative", chat_id=chat_id, user_id=user_id, kind=kind)


def command(chat_id: int, user_id: int, name: str) -> None:
    log_event("command", chat_id=chat_id, user_id=user_id, command=name)


import re as _re  # noqa: E402
_SENT_SPLIT = _re.compile(r"(?<=[.!?])\s+")
