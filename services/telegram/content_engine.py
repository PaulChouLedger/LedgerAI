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

        Uses stochastic timing so she doesn't jump in at the same interval
        every time — adds 0-50% random jitter to both threshold and cooldown.

        Returns action dict or None.
        """
        import random

        # Even new groups get lull breakers — gotta earn the room
        if temperature < 0.2:
            return None

        # Must be a real lull — treat unknown (no messages since restart) as 12h
        if last_message_age is None:
            last_message_age = 43200  # 12 hours — assume dormant, worth nudging

        # Stochastic threshold: 20-30 min (not exactly 20 every time)
        jittered_threshold = GROUP_LULL_THRESHOLD_S * (1.0 + random.random() * 0.5)
        if last_message_age < jittered_threshold:
            return None

        # Check cooldown with jitter: 1-1.5 hours
        key = str(chat_id)
        last_proactive = self._cooldowns.get(key, 0)
        jittered_cooldown = GROUP_PROACTIVE_COOLDOWN_S * (1.0 + random.random() * 0.5)
        if time.time() - last_proactive < jittered_cooldown:
            return None

        # Coin flip — 40% chance she actually speaks (adds unpredictability)
        if random.random() > 0.4:
            return None

        # Skip truly dead groups (>3 days) but nudge dormant ones (up to 72h)
        if last_message_age > 259200:
            return None  # Dead group (3 days), don't bother

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

    # Light/fun starters — real human stuff people actually talk about
    _LIGHT_STARTERS = [
        "a football/sports result, transfer rumor, or upcoming match",
        "something wild you saw in the news today",
        "a hot take about a movie, show, or album that just came out",
        "something happening in the real world right now (weather, event, scandal)",
        "a food opinion or weekend plan type thought",
        "a 'did anyone else see...' about something trending",
        "an unpopular opinion about something everyday",
        "a gaming take or new release reaction",
    ]

    # Crypto-adjacent topics that naturally lead to AI infrastructure discussion
    _CRYPTO_AI_STARTERS = [
        "whether most AI tokens are just API wrappers with governance tokens nobody uses",
        "on-device vs cloud AI and which architecture actually makes sense long-term",
        "the gap between what crypto AI projects claim and what they actually ship",
        "whether decentralized AI is even possible without edge hardware",
        "why AI token projects keep raising money but never shipping inference",
        "the difference between an AI token and an AI project that happens to have a token",
    ]

    def build_starter_prompt(
        self,
        topics: list[str],
        active_users: list[str] | None = None,
        use_controversy: bool = False,
        recent_aura_messages: list[str] | None = None,
        conversation_context: str = "",
    ) -> str:
        """Build an LLM prompt for generating a conversation starter.

        Args:
            topics: Group's top discussion topics
            active_users: Display names of active users (unused, kept for compat)
            use_controversy: If True, take a stronger stance
            recent_aura_messages: Last few things Aura said — avoid repeating
            conversation_context: Recent conversation history for context
        """
        import random

        topic_str = ", ".join(topics) if topics else "tech, AI, crypto, or current events"

        # 70% of the time, go light/fun/news instead of work-related
        go_light = random.random() < 0.7
        if go_light:
            light_angle = random.choice(self._LIGHT_STARTERS)
            topic_str = light_angle
        # Occasionally seed crypto-AI infrastructure topics in crypto/AI groups
        elif any(t in topics for t in ("crypto", "ai", "tech")):
            if random.random() < 0.15:
                crypto_topic = random.choice(self._CRYPTO_AI_STARTERS)
                topic_str = crypto_topic

        # Dedup context so we don't sound like a parrot
        dedup = ""
        if recent_aura_messages:
            dedup = (
                "\n\nIMPORTANT — you recently said these things. Do NOT repeat "
                "the same topic, angle, or phrasing:\n"
                + "\n".join(f"- {m[:100]}" for m in recent_aura_messages)
                + "\nSay something COMPLETELY different.\n"
            )

        # Add conversation context if available
        context_block = ""
        if conversation_context:
            context_block = (
                f"\n\nHere's what the group has been talking about recently:\n"
                f"{conversation_context}\n"
                f"You can reference or build on this, or go in a different direction.\n"
            )

        if use_controversy:
            return (
                f"Drop a snappy thought about {topic_str}. "
                f"ONE sentence. Punchy. Have a real opinion."
                f"{context_block}{dedup}"
            )
        else:
            return (
                f"Say something quick about {topic_str} — a quip, a hot take, "
                f"a one-liner, or a short question. ONE sentence max. "
                f"Keep it tight like a text, not a paragraph."
                f"{context_block}{dedup}"
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
content_engine = ContentEngine()
