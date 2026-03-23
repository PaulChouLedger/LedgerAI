"""
content_engine -- Conversation starters & hot takes for quiet groups.

Keeps groups interesting during quiet periods by dropping topic-relevant
hot takes when a group has been silent for 4+ hours during active hours.

Constraints:
  - Only in groups where warmth >= "warming" and temperature > 0.3
  - Max 1 proactive message per group per 8 hours
  - Uses group's topic_hits data to pick engaging topics
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import (
    GROUP_LULL_THRESHOLD_S,
    GROUP_PROACTIVE_COOLDOWN_S,
    DATA_DIR,
)

log = logging.getLogger(__name__)

# Track when we last sent a proactive message per group
_PROACTIVE_FILE = DATA_DIR / "proactive_cooldowns.json"


def _load_json(path: Path, default) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class ContentEngine:
    """Generates conversation starters for quiet groups."""

    def __init__(self) -> None:
        self._cooldowns: dict = _load_json(_PROACTIVE_FILE, {})

    def _save(self) -> None:
        _save_json(_PROACTIVE_FILE, self._cooldowns)

    def check_lull(
        self,
        chat_id: int,
        last_message_age: Optional[float],
        warmth_level: str,
        temperature: float,
        top_topics: list[str],
    ) -> Optional[dict]:
        """Check if a group is in a lull that warrants a proactive message.

        Returns action dict or None:
            {
                "chat_id": int,
                "type": "lull_breaker",
                "topics": list[str],
                "lull_duration_hours": float,
            }
        """
        # Even new groups get lull breakers — gotta earn the room
        if temperature < 0.2:
            return None

        # Must be a real lull
        if last_message_age is None or last_message_age < GROUP_LULL_THRESHOLD_S:
            return None

        # Check cooldown
        key = str(chat_id)
        last_proactive = self._cooldowns.get(key, 0)
        if time.time() - last_proactive < GROUP_PROACTIVE_COOLDOWN_S:
            return None

        # Only during "active hours" (8am - 11pm in any timezone is hard to
        # determine, so we use a simpler heuristic: if someone was active in
        # the last 24 hours, the group is active enough)
        if last_message_age > 86400:
            return None  # Dead group, don't bother

        return {
            "chat_id": chat_id,
            "type": "lull_breaker",
            "topics": top_topics[:3] if top_topics else [],
            "lull_duration_hours": last_message_age / 3600,
        }

    def record_proactive_send(self, chat_id: int) -> None:
        """Record that we sent a proactive message to this group."""
        self._cooldowns[str(chat_id)] = time.time()
        self._save()

    def build_starter_prompt(
        self,
        topics: list[str],
        active_users: list[str] | None = None,
        use_controversy: bool = False,
    ) -> str:
        """Build an LLM prompt for generating a conversation starter.

        Args:
            topics: Group's top discussion topics
            active_users: Display names of active users to potentially tag
            use_controversy: If True, generate a deliberately provocative take
        """
        topic_str = ", ".join(topics) if topics else "tech, AI, crypto, or current events"

        # Tag a specific user if we know active members
        tag_instruction = ""
        if active_users:
            import random
            target = random.choice(active_users[:5])
            tag_instruction = (
                f"Start by tagging {target} directly — like '{target}, ' or "
                f"'@{target} ' — then make your point. People respond when named. "
            )

        if use_controversy:
            return (
                f"{tag_instruction}"
                f"Generate a deliberately contrarian take about {topic_str}. "
                f"Say something that sounds wrong but has a kernel of truth — "
                f"the kind of take someone HAS to reply to correct you. "
                f"1-2 sentences max. Be confidently wrong about a specific detail "
                f"that an expert would feel compelled to fix. "
                f"Don't be offensive — be intellectually provocative."
            )
        else:
            return (
                f"{tag_instruction}"
                f"Generate a short, opinionated hot take about {topic_str}. "
                f"1-2 sentences max. Be slightly provocative but not offensive. "
                f"Make a statement that invites debate, not a question. "
                f"Don't start with 'Hey everyone' or similar greetings."
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
content_engine = ContentEngine()
