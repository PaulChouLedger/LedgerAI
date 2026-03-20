"""
context -- Rolling message buffer per chat for Aura Telegram bot.

Keeps the last N messages per chat_id in memory. Used to build
conversational context for LLM prompts and decision engine scoring.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import CONTEXT_BUFFER_SIZE, CONTEXT_WINDOW_FOR_PROMPT


@dataclass
class Message:
    user_id: int
    display_name: str
    text: str
    timestamp: float = field(default_factory=time.time)
    is_bot: bool = False


class ContextBuffer:
    """In-memory ring buffer of recent messages, keyed by chat_id."""

    def __init__(self) -> None:
        self._buffers: dict[int, deque[Message]] = {}

    def add(
        self,
        chat_id: int,
        user_id: int,
        display_name: str,
        text: str,
        is_bot: bool = False,
    ) -> None:
        if chat_id not in self._buffers:
            self._buffers[chat_id] = deque(maxlen=CONTEXT_BUFFER_SIZE)
        self._buffers[chat_id].append(
            Message(
                user_id=user_id,
                display_name=display_name,
                text=text,
                is_bot=is_bot,
            )
        )

    def get_recent(
        self, chat_id: int, n: int = CONTEXT_WINDOW_FOR_PROMPT
    ) -> list[Message]:
        """Return the last *n* messages for a chat."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return []
        items = list(buf)
        return items[-n:]

    def get_all(self, chat_id: int) -> list[Message]:
        buf = self._buffers.get(chat_id)
        return list(buf) if buf else []

    def last_bot_message_age(self, chat_id: int) -> Optional[float]:
        """Seconds since Aura's last message in this chat, or None."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return None
        for msg in reversed(buf):
            if msg.is_bot:
                return time.time() - msg.timestamp
        return None

    def bot_messages_in_last_n(self, chat_id: int, n: int = 10) -> int:
        """Count how many of the last *n* messages are from the bot."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return 0
        recent = list(buf)[-n:]
        return sum(1 for m in recent if m.is_bot)

    def messages_since_last_bot(self, chat_id: int) -> int:
        """Count messages from other users since Aura's last message."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return 999  # large number = no cooldown needed
        count = 0
        for msg in reversed(buf):
            if msg.is_bot:
                break
            count += 1
        return count

    def last_message_age(self, chat_id: int) -> Optional[float]:
        """Seconds since any message in this chat, or None."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return None
        return time.time() - buf[-1].timestamp

    def format_for_prompt(self, chat_id: int, n: int = CONTEXT_WINDOW_FOR_PROMPT) -> str:
        """Format recent messages as a conversation transcript for the LLM."""
        messages = self.get_recent(chat_id, n)
        if not messages:
            return "(no recent messages)"
        lines = []
        for m in messages:
            name = "Aura" if m.is_bot else m.display_name
            lines.append(f"{name}: {m.text}")
        return "\n".join(lines)


# Singleton
context_buffer = ContextBuffer()
