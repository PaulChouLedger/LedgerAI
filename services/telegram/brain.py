"""
brain -- Decision engine for group chats: "should Aura respond?"

Lightweight scoring (<5ms, no LLM call). Produces 0.0-1.0 confidence.
Responds only if score > RESPOND_THRESHOLD.

Philosophy: false negatives (staying silent) are far less costly than
false positives (being annoying). Users can always @mention Aura directly.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from config import (
    BOT_USERNAME,
    RESPOND_THRESHOLD,
    W_DIRECT_MENTION,
    W_UNANSWERED_QUESTION,
    W_CONVERSATION_LULL,
    W_EMOTIONAL_CONTENT,
    W_RECENT_MENTION,
    W_COOLDOWN_PENALTY,
    W_RAPID_FIRE_PENALTY,
    COOLDOWN_MESSAGES,
    COOLDOWN_SECONDS,
    CONVERSATION_LULL_S,
    GROUP_MAX_PER_HOUR,
)
from context import context_buffer, Message

log = logging.getLogger(__name__)

# Patterns that indicate a direct mention of Aura
MENTION_PATTERNS = [
    r"\baura\b",
    r"\b@aura\b",
    r"\bora\b",
]

# Strong emotion markers
EMOTION_MARKERS = [
    "!", "?!", "...", "omg", "wtf", "lol", "lmao", "damn", "holy",
    "amazing", "terrible", "incredible", "hate", "love", "furious",
    "excited", "worried", "scared", "thrilled",
]

# Per-group hourly counters: {chat_id: [(timestamp, ...), ...]}
_hourly_responses: dict[int, list[float]] = {}


@dataclass
class Decision:
    should_respond: bool
    score: float
    reason: str


def should_respond(
    chat_id: int,
    message: Message,
    is_reply_to_bot: bool = False,
) -> Decision:
    """Score whether Aura should respond to this group message.

    Returns a Decision with the score and reason.
    """
    score = 0.0
    reasons: list[str] = []

    text_lower = message.text.lower()

    # --- Positive signals ---

    # Direct mention (@aura, "aura", or reply to bot's message)
    bot_user = BOT_USERNAME.lower() if BOT_USERNAME else ""
    mentioned = any(re.search(p, text_lower) for p in MENTION_PATTERNS)
    if bot_user and f"@{bot_user}" in text_lower:
        mentioned = True
    if mentioned or is_reply_to_bot:
        score += W_DIRECT_MENTION
        reasons.append("direct mention" if mentioned else "reply to Aura")

    # Unanswered question (ends with ?)
    if text_lower.rstrip().endswith("?"):
        score += W_UNANSWERED_QUESTION
        reasons.append("question")

    # Conversation lull (5+ min silence then someone speaks)
    last_age = context_buffer.last_message_age(chat_id)
    if last_age is not None and last_age > CONVERSATION_LULL_S:
        score += W_CONVERSATION_LULL
        reasons.append(f"lull ({last_age:.0f}s)")

    # Emotional content
    emotion_count = sum(1 for m in EMOTION_MARKERS if m in text_lower)
    if emotion_count >= 2:
        score += W_EMOTIONAL_CONTENT
        reasons.append("emotional")

    # Recent mention of Aura in buffer (last 5 msgs), Aura hasn't responded
    recent = context_buffer.get_recent(chat_id, 5)
    aura_mentioned_recently = any(
        any(re.search(p, m.text.lower()) for p in MENTION_PATTERNS)
        for m in recent if not m.is_bot
    )
    aura_responded_recently = any(m.is_bot for m in recent)
    if aura_mentioned_recently and not aura_responded_recently:
        score += W_RECENT_MENTION
        reasons.append("recent mention unanswered")

    # --- Negative signals (penalties) ---

    # Cooldown: Aura responded within last N messages
    msgs_since = context_buffer.messages_since_last_bot(chat_id)
    if msgs_since < COOLDOWN_MESSAGES:
        score += W_COOLDOWN_PENALTY
        reasons.append(f"cooldown ({msgs_since}/{COOLDOWN_MESSAGES} msgs)")

    # Time cooldown
    last_bot_age = context_buffer.last_bot_message_age(chat_id)
    if last_bot_age is not None and last_bot_age < COOLDOWN_SECONDS:
        score += W_COOLDOWN_PENALTY
        reasons.append(f"time cooldown ({last_bot_age:.0f}s)")

    # Rapid-fire: 2+ Aura responses in last 10 messages
    bot_in_10 = context_buffer.bot_messages_in_last_n(chat_id, 10)
    if bot_in_10 >= 2:
        score += W_RAPID_FIRE_PENALTY
        reasons.append(f"rapid-fire ({bot_in_10} in last 10)")

    # Hourly rate limit
    now = time.time()
    hour_ago = now - 3600
    if chat_id not in _hourly_responses:
        _hourly_responses[chat_id] = []
    _hourly_responses[chat_id] = [t for t in _hourly_responses[chat_id] if t > hour_ago]
    if len(_hourly_responses[chat_id]) >= GROUP_MAX_PER_HOUR:
        score = min(score, 0.0)
        reasons.append(f"hourly limit ({GROUP_MAX_PER_HOUR}/hr)")

    # Clamp
    score = max(0.0, min(1.0, score))
    should = score >= RESPOND_THRESHOLD

    reason_str = ", ".join(reasons) if reasons else "no signals"
    log.debug(
        "Decision for chat %d: %.2f (%s) -> %s",
        chat_id, score, reason_str, "RESPOND" if should else "SILENT",
    )

    return Decision(should_respond=should, score=score, reason=reason_str)


def record_response(chat_id: int) -> None:
    """Record that Aura responded in this group (for rate limiting)."""
    if chat_id not in _hourly_responses:
        _hourly_responses[chat_id] = []
    _hourly_responses[chat_id].append(time.time())
