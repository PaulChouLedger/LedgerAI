"""
growth -- Growth event logger and opportunity detector for the Aura Telegram bot.

Tracks group joins/kicks, detects organic growth signals in conversations,
and maintains stats.  Never triggers self-promotion — logging only.

Persists to data/telegram/growth_log.json.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from config import (
    GROWTH_LOG_FILE,
    JOIN_COOLDOWN_S,
    MAX_ACTIVE_GROUPS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON persistence (same pattern as memory.py)
# ---------------------------------------------------------------------------

_EMPTY_STATE: dict = {
    "events": [],
    "stats": {
        "total_groups_joined": 0,
        "total_groups_kicked": 0,
        "total_invites_received": 0,
        "active_groups": 0,
        "growth_rate_30d": 0.0,
    },
    "banned_groups": [],
    "join_cooldown_until": None,
}


def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default.copy()


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Opportunity detection patterns
# ---------------------------------------------------------------------------

_OPPORTUNITY_PATTERNS: list[re.Pattern] = [
    re.compile(r"anyone\s+know\s+a?\s*good\s+bot", re.IGNORECASE),
    re.compile(r"we\s+need\s+an?\s+ai", re.IGNORECASE),
    re.compile(r"need\s+a\s+bot\s+for", re.IGNORECASE),
    re.compile(r"wish\s+we\s+had\s+an?\s+ai\s+in", re.IGNORECASE),
    re.compile(r"looking\s+for\s+a\s+(chat\s*)?bot", re.IGNORECASE),
    re.compile(r"recommend\s+a\s+(good\s+)?bot", re.IGNORECASE),
    re.compile(r"any\s+bot\s+that\s+can", re.IGNORECASE),
    re.compile(r"need\s+an?\s+ai\s+(assistant|bot|helper)", re.IGNORECASE),
]

_REFERRAL_PATTERN = re.compile(r"@TheRealAura_bot", re.IGNORECASE)

_MAX_EVENTS = 500


# ---------------------------------------------------------------------------
# GrowthEngine
# ---------------------------------------------------------------------------

class GrowthEngine:
    """Tracks group lifecycle events and detects organic growth signals."""

    def __init__(self) -> None:
        self._data: dict = _load_json(GROWTH_LOG_FILE, _EMPTY_STATE)
        # Ensure all top-level keys exist (forward compat)
        for key, val in _EMPTY_STATE.items():
            self._data.setdefault(key, val if not isinstance(val, (dict, list)) else type(val)(val))

    # -- persistence --------------------------------------------------------

    def _save(self) -> None:
        _save_json(GROWTH_LOG_FILE, self._data)

    # -- event logging ------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        details: str = "",
    ) -> None:
        event = {
            "ts": time.time(),
            "type": event_type,
            "chat_id": chat_id,
            "user_id": user_id,
            "details": details,
        }
        self._data["events"].append(event)
        self._trim_events()
        self._save()

    # -- group lifecycle ----------------------------------------------------

    def on_group_join(
        self,
        chat_id: int,
        group_name: str,
        invited_by: Optional[str] = None,
    ) -> None:
        details = f"Joined {group_name}"
        if invited_by:
            details = f"Added by {invited_by} to {group_name}"
        self.log_event("joined", chat_id=chat_id, details=details)
        self._data["join_cooldown_until"] = time.time() + JOIN_COOLDOWN_S
        self._recompute_stats()
        self._save()

    def on_group_kick(self, chat_id: int) -> None:
        self.log_event("kicked", chat_id=chat_id)
        if chat_id not in self._data["banned_groups"]:
            self._data["banned_groups"].append(chat_id)
        self._recompute_stats()
        self._save()

    # -- queries ------------------------------------------------------------

    def is_banned(self, chat_id: int) -> bool:
        return chat_id in self._data["banned_groups"]

    def can_join_new_group(self) -> bool:
        cooldown = self._data.get("join_cooldown_until")
        if cooldown is not None and time.time() < cooldown:
            return False
        stats = self._recompute_stats()
        return stats["active_groups"] < MAX_ACTIVE_GROUPS

    # -- opportunity detection (log only, never self-promote) ---------------

    def detect_opportunity(
        self,
        text: str,
        chat_id: int,
        user_id: int,
    ) -> Optional[str]:
        # Check for direct referral first
        if _REFERRAL_PATTERN.search(text):
            desc = "Referral: @TheRealAura_bot mentioned"
            self.log_event("referral", chat_id=chat_id, user_id=user_id, details=desc)
            return desc

        # Check organic opportunity patterns
        for pattern in _OPPORTUNITY_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = match.group(0)
                desc = f"Opportunity detected: \"{snippet}\""
                self.log_event(
                    "opportunity_detected",
                    chat_id=chat_id,
                    user_id=user_id,
                    details=desc,
                )
                return desc

        return None

    # -- stats --------------------------------------------------------------

    def get_stats(self) -> dict:
        return self._recompute_stats()

    def _recompute_stats(self) -> dict:
        events = self._data["events"]
        banned = set(self._data["banned_groups"])

        total_joined = sum(1 for e in events if e["type"] == "joined")
        total_kicked = sum(1 for e in events if e["type"] == "kicked")
        total_invites = sum(
            1 for e in events
            if e["type"] == "joined" and e.get("details", "").startswith("Added by")
        )

        # Active groups = joined groups minus kicked groups
        joined_chats: set[int] = set()
        kicked_chats: set[int] = set()
        for e in events:
            cid = e.get("chat_id")
            if cid is None:
                continue
            if e["type"] == "joined":
                joined_chats.add(cid)
            elif e["type"] == "kicked":
                kicked_chats.add(cid)
        active = joined_chats - kicked_chats - banned
        active_groups = len(active)

        # Growth rate: net joins in last 30 days
        cutoff_30d = time.time() - 30 * 86400
        joins_30d = sum(
            1 for e in events if e["type"] == "joined" and e["ts"] >= cutoff_30d
        )
        kicks_30d = sum(
            1 for e in events if e["type"] == "kicked" and e["ts"] >= cutoff_30d
        )
        growth_rate_30d = float(joins_30d - kicks_30d)

        stats = {
            "total_groups_joined": total_joined,
            "total_groups_kicked": total_kicked,
            "total_invites_received": total_invites,
            "active_groups": active_groups,
            "growth_rate_30d": growth_rate_30d,
        }
        self._data["stats"] = stats
        return stats

    # -- housekeeping -------------------------------------------------------

    def _trim_events(self) -> None:
        if len(self._data["events"]) > _MAX_EVENTS:
            self._data["events"] = self._data["events"][-_MAX_EVENTS:]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

growth_engine = GrowthEngine()
