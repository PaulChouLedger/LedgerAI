"""
metrics -- Per-message engagement metrics (interactivity upgrade, 2026-08-02).

The data half of the owner's directive: "tweak it to be more interactive and
so we can get data from it." Every message Aura sends gets a type/topic tag;
every reply, reaction, and poll answer it earns is written next to those
tags, so "which kinds of message actually engage people" becomes a one-liner
over a JSONL file instead of a guess.

Output: config.ENGAGEMENT_METRICS_FILE, one JSON object per line, append-only
(this file never rotates itself — it is the evidence):

  {"event":"sent",       "ts","chat_id","message_id","type","topic","text"}
  {"event":"reply",      "ts","chat_id","message_id","user_id","latency_s",
                          "ref_type","ref_topic","text"}
  {"event":"reaction",   "ts","chat_id","message_id","user_id","emoji",
                          "ref_type","ref_topic"}
  {"event":"poll_answer","ts","poll_id","chat_id","user_id","option_ids",
                          "options"}

Attribution: sent messages are also indexed in config.ENGAGEMENT_INDEX_FILE
(message_id -> type/topic/ts) so a reply or reaction arriving hours later --
or after a restart -- still knows what kind of message earned it. Replies
are attributed only within ENGAGEMENT_REPLY_WINDOW_S (24 h) of the send;
later replies are still logged, with "within_window": false.

Known API limitations (measured against Bot API docs, PTB 22.7):
  - message_reaction updates only arrive in chats where the bot is an
    ADMINISTRATOR, must be requested via allowed_updates, and reactions set
    by other bots are never delivered. bot.py logs a visible warning at
    startup for any pilot chat where reaction data will be blind.
  - Anonymous reaction totals (message_reaction_count) arrive with several
    minutes of delay and carry no user id.

Feature flag: config.ENGAGEMENT_METRICS (AURA_TG_METRICS, master
AURA_TG_INTERACTIVE). When off, every call here is a no-op.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import config

log = logging.getLogger(__name__)

# Index entries older than the reply window plus grace are pruned.
_INDEX_MAX_AGE_S = config.ENGAGEMENT_REPLY_WINDOW_S + 6 * 86400  # 7 days


def _key(chat_id: int, message_id: int) -> str:
    return f"{chat_id}:{message_id}"


class EngagementMetrics:
    """JSONL engagement logger with a persistent sent-message index."""

    def __init__(self) -> None:
        self._index: dict[str, dict] = {}
        if config.ENGAGEMENT_INDEX_FILE.exists():
            try:
                self._index = json.loads(
                    config.ENGAGEMENT_INDEX_FILE.read_text())
            except (json.JSONDecodeError, OSError) as e:
                # Visible failure: a corrupt index means replies lose their
                # attribution — say so once instead of quietly starting empty.
                log.error("Metrics index unreadable (%s) — starting fresh; "
                          "replies to pre-existing messages will be "
                          "untyped", e)

    # -- internals ----------------------------------------------------------

    def _append(self, record: dict) -> None:
        record["ts"] = round(time.time(), 2)
        try:
            with open(config.ENGAGEMENT_METRICS_FILE, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            log.error("Metrics write FAILED (%s): %s", e, record.get("event"))

    def _save_index(self) -> None:
        now = time.time()
        self._index = {
            k: v for k, v in self._index.items()
            if now - v.get("ts", 0) < _INDEX_MAX_AGE_S
        }
        try:
            config.ENGAGEMENT_INDEX_FILE.write_text(json.dumps(self._index))
        except OSError as e:
            log.error("Metrics index save FAILED: %s", e)

    # -- recording ----------------------------------------------------------

    def record_sent(
        self,
        chat_id: int,
        message_id: Optional[int],
        msg_type: str,
        topic: str = "",
        text: str = "",
        extra_message_ids: Optional[list[int]] = None,
    ) -> None:
        """Log an outbound message with its type/topic tags and index it.

        extra_message_ids: further Telegram message ids belonging to the
        same logical message (chunked sends) — indexed for attribution but
        not logged as separate "sent" events, so counts stay honest.
        """
        if not config.ENGAGEMENT_METRICS:
            return
        self._append({
            "event": "sent",
            "chat_id": chat_id,
            "message_id": message_id,
            "type": msg_type,
            "topic": topic[:120],
            "text": text[:200],
        })
        now = time.time()
        indexed = False
        for mid in [message_id] + list(extra_message_ids or []):
            if mid is None:
                continue
            self._index[_key(chat_id, mid)] = {
                "type": msg_type,
                "topic": topic[:120],
                "ts": now,
            }
            indexed = True
        if indexed:
            self._save_index()

    def record_reply(
        self,
        chat_id: int,
        replied_to_message_id: int,
        user_id: int,
        text: str = "",
    ) -> None:
        """Log a human reply to one of Aura's messages."""
        if not config.ENGAGEMENT_METRICS:
            return
        ref = self._index.get(_key(chat_id, replied_to_message_id), {})
        latency = round(time.time() - ref["ts"], 1) if "ts" in ref else None
        self._append({
            "event": "reply",
            "chat_id": chat_id,
            "message_id": replied_to_message_id,
            "user_id": user_id,
            "latency_s": latency,
            "within_window": (
                latency is not None
                and latency <= config.ENGAGEMENT_REPLY_WINDOW_S),
            "ref_type": ref.get("type", "unknown"),
            "ref_topic": ref.get("topic", ""),
            "text": text[:200],
        })

    def record_reaction(
        self,
        chat_id: int,
        message_id: int,
        user_id: Optional[int],
        emojis: list[str],
    ) -> None:
        """Log a reaction change on one of Aura's messages.

        Only messages found in the index are logged — reaction updates
        arrive for every message in the chat, not just hers.
        """
        if not config.ENGAGEMENT_METRICS:
            return
        ref = self._index.get(_key(chat_id, message_id))
        if ref is None:
            return  # a reaction to somebody else's message
        self._append({
            "event": "reaction",
            "chat_id": chat_id,
            "message_id": message_id,
            "user_id": user_id,
            "emoji": emojis,
            "ref_type": ref.get("type", "unknown"),
            "ref_topic": ref.get("topic", ""),
        })

    def record_poll_answer(
        self,
        poll_id: str,
        chat_id: Optional[int],
        user_id: int,
        option_ids: list[int],
        options: list[str],
    ) -> None:
        """Log a vote in one of Aura's (non-anonymous) polls."""
        if not config.ENGAGEMENT_METRICS:
            return
        self._append({
            "event": "poll_answer",
            "poll_id": poll_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "option_ids": option_ids,
            "options": [options[i] for i in option_ids
                        if 0 <= i < len(options)],
        })


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
metrics = EngagementMetrics()
