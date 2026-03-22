"""
reputation -- Per-group reputation tracker for Aura Telegram bot.

Tracks engagement signals per chat_id and computes a rolling reputation
score that controls how aggressively Aura participates in each group.

Data persisted to data/telegram/reputation.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import (
    REPUTATION_FILE,
    WARMTH_MULTIPLIERS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence helpers (same pattern as memory.py)
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
# Defaults for a new group entry
# ---------------------------------------------------------------------------

def _new_entry(group_name: str, invited_by: Optional[int] = None) -> dict:
    now = time.time()
    return {
        "group_name": group_name,
        "joined_at": now,
        "invited_by": invited_by,
        "kicked": False,
        "kicked_at": None,
        # Engagement signals
        "replies_to_aura": 0,
        "mentions_of_aura": 0,
        "reactions_to_aura": 0,
        "questions_answered": 0,
        "ignored_responses": 0,
        # Computed
        "reputation_score": 0.0,
        "warmth_level": "new",
        "activity_multiplier": WARMTH_MULTIPLIERS["new"],
        # Tracking
        "total_responses": 0,
        "last_response_at": None,
        "consecutive_ignores": 0,
        "topic_hits": {},
        "score_history": [],
    }


# ---------------------------------------------------------------------------
# ReputationTracker
# ---------------------------------------------------------------------------

_EVENT_FIELDS = {
    "reply": "replies_to_aura",
    "mention": "mentions_of_aura",
    "reaction": "reactions_to_aura",
    "question_answered": "questions_answered",
}

_EMA_ALPHA = 0.3
_DECAY_FACTOR = 0.7  # 30 % decay


class ReputationTracker:
    """Per-group reputation and warmth tracker."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = _load_json(REPUTATION_FILE, {})

    # -- lifecycle ----------------------------------------------------------

    def mark_joined(
        self,
        chat_id: int,
        group_name: str,
        invited_by: Optional[int] = None,
    ) -> None:
        """Initialize a new group entry (no-op if already present)."""
        key = str(chat_id)
        if key not in self._data:
            self._data[key] = _new_entry(group_name, invited_by)
            self._save()
            log.info("Reputation: joined group %s (%s)", group_name, chat_id)

    def mark_kicked(self, chat_id: int) -> None:
        """Record a permanent ban from a group."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            entry = _new_entry("unknown")
            self._data[key] = entry
        entry["kicked"] = True
        entry["kicked_at"] = time.time()
        self._save()
        log.info("Reputation: kicked from chat %s", chat_id)

    def is_kicked(self, chat_id: int) -> bool:
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return False
        return bool(entry.get("kicked"))

    # -- engagement ---------------------------------------------------------

    def record_engagement(self, chat_id: int, event_type: str) -> None:
        """Increment an engagement counter and recompute score.

        event_type: "reply" | "mention" | "reaction" | "question_answered"
        """
        field = _EVENT_FIELDS.get(event_type)
        if field is None:
            log.warning("Unknown engagement event type: %s", event_type)
            return
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return
        entry[field] = entry.get(field, 0) + 1
        self._recompute_score(key)
        self._save()

    def record_response(self, chat_id: int) -> None:
        """Aura sent a message in this group."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return
        entry["total_responses"] = entry.get("total_responses", 0) + 1
        entry["last_response_at"] = time.time()
        entry["consecutive_ignores"] = 0
        self._save()

    def record_ignore(self, chat_id: int) -> None:
        """Aura's last message was ignored."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return
        entry["consecutive_ignores"] = entry.get("consecutive_ignores", 0) + 1
        entry["ignored_responses"] = entry.get("ignored_responses", 0) + 1
        self._recompute_score(key)
        self._save()

    def record_topic_hit(self, chat_id: int, topic: str) -> None:
        """Increment topic_hits for a topic where Aura got engagement."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return
        hits: dict = entry.setdefault("topic_hits", {})
        hits[topic] = hits.get(topic, 0) + 1
        self._save()

    # -- queries ------------------------------------------------------------

    def get_activity_multiplier(self, chat_id: int) -> float:
        """Return warmth-based activity multiplier."""
        warmth = self.get_warmth_level(chat_id)
        return WARMTH_MULTIPLIERS.get(warmth, WARMTH_MULTIPLIERS["new"])

    def get_warmth_level(self, chat_id: int) -> str:
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return "new"
        return entry.get("warmth_level", "new")

    def get_reputation(self, chat_id: int) -> float:
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return 0.0
        return float(entry.get("reputation_score", 0.0))

    def get_top_topics(self, chat_id: int, n: int = 5) -> list[str]:
        """Return the top-n topics where Aura gets the most engagement."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return []
        hits: dict = entry.get("topic_hits", {})
        sorted_topics = sorted(hits, key=hits.get, reverse=True)  # type: ignore[arg-type]
        return sorted_topics[:n]

    # -- maintenance --------------------------------------------------------

    def weekly_decay(self) -> None:
        """Decay all signal counts by 30 % and snapshot score history."""
        decay_fields = [
            "replies_to_aura",
            "mentions_of_aura",
            "reactions_to_aura",
            "questions_answered",
            "ignored_responses",
        ]
        now = time.time()
        for key, entry in self._data.items():
            for field in decay_fields:
                val = entry.get(field, 0)
                entry[field] = round(val * _DECAY_FACTOR, 2)
            # Reset consecutive ignores on decay (stale signal)
            entry["consecutive_ignores"] = 0
            self._recompute_score(key)
            # Snapshot
            history: list = entry.setdefault("score_history", [])
            history.append({"ts": now, "score": entry["reputation_score"]})
            # Keep last 52 weeks
            if len(history) > 52:
                entry["score_history"] = history[-52:]
        self._save()
        log.info("Reputation: weekly decay applied to %d groups", len(self._data))

    # -- internal -----------------------------------------------------------

    def _recompute_score(self, key: str) -> None:
        """Recompute reputation_score, warmth_level, and activity_multiplier."""
        entry = self._data.get(key)
        if entry is None:
            return

        replies = entry.get("replies_to_aura", 0)
        mentions = entry.get("mentions_of_aura", 0)
        reactions = entry.get("reactions_to_aura", 0)
        questions = entry.get("questions_answered", 0)
        ignored = entry.get("ignored_responses", 0)
        consec = entry.get("consecutive_ignores", 0)
        total = max(entry.get("total_responses", 0), 1)

        raw = (
            (replies * 3 + mentions * 2 + reactions * 1 + questions * 2) / total
            - (ignored * 1.5 / total)
            - (consec * 0.1)
        )

        # EMA blend with previous score
        prev = entry.get("reputation_score", 0.0)
        score = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * prev
        score = max(0.0, min(1.0, score))
        entry["reputation_score"] = round(score, 4)

        # Warmth level
        entry["warmth_level"] = self._compute_warmth(entry)
        entry["activity_multiplier"] = WARMTH_MULTIPLIERS.get(
            entry["warmth_level"], WARMTH_MULTIPLIERS["new"]
        )

    @staticmethod
    def _compute_warmth(entry: dict) -> str:
        score = entry.get("reputation_score", 0.0)
        total = entry.get("total_responses", 0)
        joined_at = entry.get("joined_at", time.time())
        days_since = (time.time() - joined_at) / 86400
        hours_since = days_since * 24

        if score > 0.7 and total > 50 and days_since > 14:
            return "trusted"
        if score > 0.5 and total > 20 and days_since > 7:
            return "established"
        if score > 0.3 and total > 5:
            return "warming"
        # "new" if first 48h or < 5 total responses
        if hours_since < 48 or total < 5:
            return "new"
        # Fallback: past 48h with low score stays warming if enough responses
        if total > 5:
            return "warming"
        return "new"

    # -- cold group activation -----------------------------------------------

    def record_test_post(self, chat_id: int) -> None:
        """Record that Aura made her first test post in a cold group."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return
        entry["test_post_at"] = time.time()
        entry["test_post_msgs_after"] = 0
        entry["test_post_engagement"] = 0
        entry["test_post_negative"] = 0
        self._save()
        log.info("Test post recorded for group %s (%s)", entry.get("group_name"), chat_id)

    def record_test_post_feedback(self, chat_id: int, is_engagement: bool, is_negative: bool) -> None:
        """Track feedback on a test post. Called on each subsequent message."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None or "test_post_at" not in entry:
            return

        entry["test_post_msgs_after"] = entry.get("test_post_msgs_after", 0) + 1

        if is_engagement:
            entry["test_post_engagement"] = entry.get("test_post_engagement", 0) + 1
        if is_negative:
            entry["test_post_negative"] = entry.get("test_post_negative", 0) + 1

        self._save()

    def evaluate_test_post(self, chat_id: int) -> str | None:
        """Evaluate test post outcome after enough messages have passed.

        Returns:
            "positive"  — got engagement, promote warmth
            "negative"  — got negative feedback, back off
            "neutral"   — ignored, stay cautious
            None        — not enough data yet (< 5 messages after)
        """
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None or "test_post_at" not in entry:
            return None

        msgs_after = entry.get("test_post_msgs_after", 0)
        if msgs_after < 5:
            return None  # Need more data

        engagement = entry.get("test_post_engagement", 0)
        negative = entry.get("test_post_negative", 0)

        # Clean up tracking fields
        result = "neutral"
        if negative > 0:
            result = "negative"
        elif engagement > 0:
            result = "positive"

        # Clear test post tracking
        for field in ("test_post_at", "test_post_msgs_after",
                       "test_post_engagement", "test_post_negative"):
            entry.pop(field, None)

        # Apply result
        if result == "positive":
            entry["warmth_level"] = "warming"
            entry["activity_multiplier"] = WARMTH_MULTIPLIERS["warming"]
            log.info("Test post POSITIVE in %s — promoting to warming", entry.get("group_name"))
        elif result == "negative":
            entry["cold_rejected"] = True
            entry["cold_rejected_at"] = time.time()
            log.info("Test post NEGATIVE in %s — backing off", entry.get("group_name"))

        self._save()
        return result

    def is_cold_group_eligible(self, chat_id: int) -> bool:
        """Check if a group is eligible for cold activation (first test post)."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return False
        if entry.get("kicked"):
            return False
        if entry.get("total_responses", 0) > 0:
            return False  # Already posted
        if entry.get("test_post_at"):
            return False  # Test post pending evaluation
        if entry.get("cold_rejected"):
            return False  # Already rejected
        # Must have been in the group long enough (12+ hours)
        joined = entry.get("joined_at", time.time())
        if time.time() - joined < 43200:
            return False
        return True

    def auto_tag_topics(self, chat_id: int, text: str) -> None:
        """Extract and record topic hits from message text.

        Simple keyword-based extraction for the content engine to use
        when picking conversation starters.
        """
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return

        text_lower = text.lower()
        topic_keywords = {
            "crypto": ["crypto", "bitcoin", "ethereum", "token", "blockchain", "defi", "nft"],
            "ai": ["ai", "artificial intelligence", "machine learning", "llm", "gpt", "neural"],
            "tech": ["code", "programming", "software", "developer", "api", "github"],
            "finance": ["stock", "market", "invest", "trading", "portfolio", "fund"],
            "gaming": ["game", "gaming", "play", "steam", "console", "esports"],
            "politics": ["politic", "election", "government", "vote", "congress", "policy"],
            "science": ["science", "research", "study", "physics", "biology", "chemistry"],
            "health": ["health", "fitness", "workout", "diet", "medical", "mental health"],
            "business": ["startup", "entrepreneur", "business", "company", "product", "launch"],
            "culture": ["movie", "music", "art", "book", "show", "series", "film"],
        }

        hits: dict = entry.setdefault("topic_hits", {})
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                hits[topic] = hits.get(topic, 0) + 1

        # Only save if we found hits (avoid excessive writes)
        if any(any(kw in text_lower for kw in keywords) for keywords in topic_keywords.values()):
            self._save()

    def get_joined_at(self, chat_id: int) -> Optional[float]:
        """Return the joined_at timestamp for a group, or None."""
        key = str(chat_id)
        entry = self._data.get(key)
        if entry is None:
            return None
        return entry.get("joined_at")

    def _save(self) -> None:
        _save_json(REPUTATION_FILE, self._data)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
reputation_tracker = ReputationTracker()
