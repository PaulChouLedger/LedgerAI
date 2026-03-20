"""
social_graph -- Cross-group relationship tracker for Aura Telegram bot.

Tracks which users appear in which groups, computes influence scores,
identifies connectors (users in 2+ groups), and monitors relationship depth.

Persisted to data/telegram/social_graph.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import SOCIAL_GRAPH_FILE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON persistence helpers (same pattern as memory.py)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Default user record
# ---------------------------------------------------------------------------

def _default_user() -> dict:
    return {
        "groups_seen_in": [],
        "influence_score": 0.0,
        "is_connector": False,
        "has_invited_aura": False,
        "invite_count": 0,
        "relationship_depth": "stranger",
        "dm_count": 0,
        "group_interactions": 0,
        "last_interaction": None,
        "advocacy_signals": [],
    }


# ---------------------------------------------------------------------------
# Relationship depth thresholds
# ---------------------------------------------------------------------------

def _compute_depth(user: dict) -> str:
    total = user.get("dm_count", 0) + user.get("group_interactions", 0)
    if user.get("has_invited_aura") or total >= 50:
        return "advocate"
    if total >= 15 or user.get("dm_count", 0) >= 5:
        return "familiar"
    if total >= 3:
        return "acquaintance"
    return "stranger"


# ---------------------------------------------------------------------------
# SocialGraph
# ---------------------------------------------------------------------------

class SocialGraph:
    """Cross-group relationship tracker."""

    def __init__(self) -> None:
        raw = _load_json(SOCIAL_GRAPH_FILE, {"users": {}})
        self._data: dict = raw if "users" in raw else {"users": {}}

    # -- internal helpers ---------------------------------------------------

    def _ensure_user(self, user_id: int | str) -> str:
        key = str(user_id)
        if key not in self._data["users"]:
            self._data["users"][key] = _default_user()
        return key

    def _save(self) -> None:
        _save_json(SOCIAL_GRAPH_FILE, self._data)

    # -- recording methods --------------------------------------------------

    def record_user_in_group(self, user_id: int, chat_id: int) -> None:
        """Add chat_id to groups_seen_in (deduplicated), recompute is_connector."""
        key = self._ensure_user(user_id)
        user = self._data["users"][key]
        cid = int(chat_id)
        if cid not in user["groups_seen_in"]:
            user["groups_seen_in"].append(cid)
        user["is_connector"] = len(user["groups_seen_in"]) >= 2
        self._save()

    def record_interaction(self, user_id: int, interaction_type: str) -> None:
        """Record a DM or group interaction. Recompute relationship depth."""
        key = self._ensure_user(user_id)
        user = self._data["users"][key]
        if interaction_type == "dm":
            user["dm_count"] = user.get("dm_count", 0) + 1
        elif interaction_type == "group":
            user["group_interactions"] = user.get("group_interactions", 0) + 1
        user["last_interaction"] = time.time()
        user["relationship_depth"] = _compute_depth(user)
        self._save()

    def record_invite(self, user_id: int, chat_id: int) -> None:
        """Mark user as having invited Aura to a group."""
        key = self._ensure_user(user_id)
        user = self._data["users"][key]
        user["has_invited_aura"] = True
        user["invite_count"] = user.get("invite_count", 0) + 1
        signal = f"Invited Aura to chat {chat_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        user["advocacy_signals"].append(signal)
        user["advocacy_signals"] = user["advocacy_signals"][-10:]
        user["relationship_depth"] = _compute_depth(user)
        self._save()

    def record_advocacy(self, user_id: int, signal_text: str) -> None:
        """Add an advocacy signal (keep last 10)."""
        key = self._ensure_user(user_id)
        user = self._data["users"][key]
        user["advocacy_signals"].append(signal_text)
        user["advocacy_signals"] = user["advocacy_signals"][-10:]
        self._save()

    # -- query methods ------------------------------------------------------

    def get_connectors(self) -> list[dict]:
        """Users in 2+ groups, sorted by influence_score descending."""
        result = []
        for uid, user in self._data["users"].items():
            if len(user.get("groups_seen_in", [])) >= 2:
                result.append({"user_id": uid, **user})
        result.sort(key=lambda u: u.get("influence_score", 0.0), reverse=True)
        return result

    def get_advocates(self) -> list[dict]:
        """Users at 'advocate' depth or who have invited Aura."""
        result = []
        for uid, user in self._data["users"].items():
            if user.get("relationship_depth") == "advocate" or user.get("has_invited_aura"):
                result.append({"user_id": uid, **user})
        return result

    def get_relationship_depth(self, user_id: int) -> str:
        key = str(user_id)
        user = self._data["users"].get(key)
        if not user:
            return "stranger"
        return user.get("relationship_depth", "stranger")

    def get_influence(self, user_id: int) -> float:
        key = str(user_id)
        user = self._data["users"].get(key)
        if not user:
            return 0.0
        return user.get("influence_score", 0.0)

    def is_connector(self, user_id: int) -> bool:
        key = str(user_id)
        user = self._data["users"].get(key)
        if not user:
            return False
        return len(user.get("groups_seen_in", [])) >= 2

    # -- influence computation ----------------------------------------------

    def rebuild_influence_scores(self) -> None:
        """Recompute influence for all users.

        Weights:
          - number of groups: weight 3
          - total interactions (dm + group): weight 1
          - connector status: bonus 0.2
          - invite history: bonus 0.1
        """
        for user in self._data["users"].values():
            groups = len(user.get("groups_seen_in", []))
            total_interactions = user.get("dm_count", 0) + user.get("group_interactions", 0)

            raw = (groups * 3.0) + (total_interactions * 1.0)
            if len(user.get("groups_seen_in", [])) >= 2:
                raw += 0.2
            if user.get("has_invited_aura"):
                raw += 0.1

            # Normalize to 0-1 using a sigmoid-like curve: score / (score + k)
            # k=20 gives a reasonable ramp (20 raw points -> 0.5)
            score = raw / (raw + 20.0) if raw > 0 else 0.0
            user["influence_score"] = round(score, 4)

        self._save()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
social_graph = SocialGraph()
