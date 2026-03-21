"""
callbacks -- Inside joke & callback engine for Aura Telegram bot.

Makes Aura feel like she *remembers* by finding semantically similar past
exchanges and injecting callback context into the LLM prompt.

When callbacks get engagement, they're promoted to "inside jokes" with
trigger patterns for future priority matching.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import (
    CALLBACK_MIN_AGE_S,
    CALLBACK_SIMILARITY_THRESHOLD,
    CALLBACKS_FILE,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_json(path: Path, default) -> dict | list:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CallbackEngine
# ---------------------------------------------------------------------------

class CallbackEngine:
    """Tracks and promotes inside jokes from past exchanges."""

    def __init__(self) -> None:
        raw = _load_json(CALLBACKS_FILE, {"inside_jokes": [], "callback_log": []})
        self._data: dict = raw
        # inside_jokes: [{user_id, trigger_pattern, reference_text, created_at, hit_count}]
        # callback_log: [{user_id, chat_id, query, match_text, similarity, ts}]

    def _save(self) -> None:
        _save_json(CALLBACKS_FILE, self._data)

    def find_callback(
        self,
        user_id: int,
        text: str,
        memory_results: list[dict],
        chat_type: str = "group",
    ) -> Optional[dict]:
        """Check memory results for callback-worthy past exchanges.

        Returns callback context dict or None:
            {
                "reference_text": str,   # the past exchange to reference
                "similarity": float,     # how similar the match was
                "age_hours": float,      # how long ago
                "is_inside_joke": bool,  # promoted inside joke?
            }
        """
        now = time.time()

        # First check inside jokes for this user
        for joke in self._data.get("inside_jokes", []):
            if str(joke.get("user_id")) != str(user_id):
                continue
            trigger = joke.get("trigger_pattern", "").lower()
            if trigger and trigger in text.lower():
                joke["hit_count"] = joke.get("hit_count", 0) + 1
                self._save()
                return {
                    "reference_text": joke["reference_text"],
                    "similarity": 1.0,
                    "age_hours": (now - joke.get("created_at", now)) / 3600,
                    "is_inside_joke": True,
                }

        # Check memory search results for semantic callbacks
        if not memory_results:
            return None

        for result in memory_results:
            # Never surface DM content in group chats
            meta = result.get("metadata", {})
            result_chat_type = meta.get("chat_type", "")
            if chat_type != "private" and result_chat_type == "private":
                continue

            similarity = result.get("similarity", result.get("score", 0.0))
            if similarity < CALLBACK_SIMILARITY_THRESHOLD:
                continue

            # Check age requirement
            result_ts = result.get("timestamp", result.get("ts", 0))
            if isinstance(result_ts, str):
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(result_ts.replace("Z", "+00:00"))
                    result_ts = dt.timestamp()
                except (ValueError, AttributeError):
                    result_ts = 0

            age = now - result_ts
            if age < CALLBACK_MIN_AGE_S:
                continue

            ref_text = result.get("text", "")[:300]
            if not ref_text:
                continue

            # Log this callback
            self._data.setdefault("callback_log", []).append({
                "user_id": str(user_id),
                "query": text[:100],
                "match_text": ref_text[:100],
                "similarity": round(similarity, 3),
                "ts": now,
            })
            # Keep last 100 logs
            self._data["callback_log"] = self._data["callback_log"][-100:]
            self._save()

            return {
                "reference_text": ref_text,
                "similarity": similarity,
                "age_hours": age / 3600,
                "is_inside_joke": False,
            }

        return None

    def promote_to_inside_joke(
        self,
        user_id: int,
        trigger_pattern: str,
        reference_text: str,
    ) -> None:
        """Promote a callback that got engagement to an inside joke."""
        jokes = self._data.setdefault("inside_jokes", [])

        # Don't duplicate
        for joke in jokes:
            if (str(joke.get("user_id")) == str(user_id)
                    and joke.get("trigger_pattern") == trigger_pattern):
                return

        jokes.append({
            "user_id": str(user_id),
            "trigger_pattern": trigger_pattern.lower(),
            "reference_text": reference_text[:300],
            "created_at": time.time(),
            "hit_count": 0,
        })

        # Keep max 50 inside jokes
        self._data["inside_jokes"] = jokes[-50:]
        self._save()
        log.info("Promoted inside joke for user %s: '%s'", user_id, trigger_pattern[:50])

    def format_callback_prompt(self, callback: dict) -> str:
        """Format callback context for injection into the LLM prompt."""
        ref = callback["reference_text"]
        hours = callback["age_hours"]

        if hours < 48:
            time_desc = f"{int(hours)} hours ago"
        else:
            time_desc = f"{int(hours / 24)} days ago"

        if callback.get("is_inside_joke"):
            return (
                f"\n[CALLBACK — Inside joke: You and this person have a running thing about this topic. "
                f"About {time_desc}, you had this exchange: \"{ref}\" "
                f"Reference it naturally like a friend who just remembers things. "
                f"Never say 'I remember when...' — just weave it in.]"
            )
        else:
            return (
                f"\n[CALLBACK — About {time_desc}, you had a similar exchange: \"{ref}\" "
                f"If it fits naturally, reference it like a friend who just remembers things. "
                f"Never say 'I remember when...' — just weave it in. "
                f"If it doesn't fit, ignore this.]"
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
callback_engine = CallbackEngine()
