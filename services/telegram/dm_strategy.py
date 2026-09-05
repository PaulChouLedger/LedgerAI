"""
dm_strategy -- Proactive DM outreach logic for Aura Telegram bot.

Moves users from acquaintance -> familiar -> advocate through genuine,
well-timed direct messages. Never spammy, always with a real reason.

DM trigger types (each with independent cooldowns):
  - Post-group followup: 2-8 hours after engaging group exchange
  - Connector cultivation: periodic relationship building for 2+ group users
  - Milestone acknowledgment: user hits 25/50/100 messages
  - Topic-triggered: something relevant to user's interests

Hard constraints:
  - NEVER DM someone who hasn't /start-ed the bot
  - Max 1 proactive DM per user per 48 hours
  - Max 3 proactive DMs per day total
  - Every DM must have a genuine reason
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from config import (
    PROACTIVE_DM_COOLDOWN_PER_USER_S,
    PROACTIVE_DM_MAX_PER_DAY,
    PROACTIVE_DM_FOLLOWUP_DELAY_MIN_S,
    PROACTIVE_DM_FOLLOWUP_DELAY_MAX_S,
    DM_ELIGIBLE_FILE,
    SOCIALITE_STATE_FILE,
    DM_NUDGE_COOLDOWN_PER_GROUP_S,
    DM_NUDGE_PROBABILITY,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DM eligibility tracking (users who have /start-ed the bot)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default) -> dict | list:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default if not isinstance(default, (dict, list)) else type(default)(default)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class DMStrategy:
    """Manages proactive DM outreach."""

    def __init__(self) -> None:
        self._eligible: dict = _load_json(DM_ELIGIBLE_FILE, {})
        # {user_id_str: {"started_at": ts, "display_name": str}}
        self._state: dict = _load_json(SOCIALITE_STATE_FILE, {
            "dm_cooldowns": {},     # {user_id: last_proactive_dm_ts}
            "daily_dm_count": 0,
            "daily_dm_reset": 0,
            "pending_followups": [],  # [{user_id, chat_id, trigger, earliest_send, reason}]
            "milestone_sent": {},   # {user_id: [milestones_sent]}
        })

    def _save_eligible(self) -> None:
        _save_json(DM_ELIGIBLE_FILE, self._eligible)

    def _save_state(self) -> None:
        _save_json(SOCIALITE_STATE_FILE, self._state)

    # -- eligibility --------------------------------------------------------

    def mark_dm_eligible(self, user_id: int, display_name: str = "") -> None:
        """Record that a user has /start-ed the bot and is DM-eligible."""
        key = str(user_id)
        if key not in self._eligible:
            self._eligible[key] = {
                "started_at": time.time(),
                "display_name": display_name,
            }
            self._save_eligible()
            log.info("DM eligible: user %s (%s)", user_id, display_name)

    def is_dm_eligible(self, user_id: int) -> bool:
        return str(user_id) in self._eligible

    def get_eligible_users(self) -> dict:
        return dict(self._eligible)

    # -- prior-DM census (win-back audience, 2026-09-05) --------------------

    _dm_hist_cache: dict = {"mtime": 0.0, "users": {}}

    def prior_dm_users(self) -> dict[int, dict]:
        """{user_id: {last_in, n_in, name, recent_in[]}} for everyone who
        has ever SENT her a DM — the only population proactive DMs may
        target during the pilot. Source is dm_history.jsonl (their own
        thread with her); cached by mtime."""
        from config import DM_HISTORY_FILE
        try:
            mt = DM_HISTORY_FILE.stat().st_mtime
        except OSError:
            return {}
        c = self._dm_hist_cache
        if c["mtime"] == mt:
            return c["users"]
        users: dict[int, dict] = {}
        try:
            with open(DM_HISTORY_FILE) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("direction") != "in":
                        continue
                    u = users.setdefault(int(r["user_id"]), {
                        "last_in": 0.0, "n_in": 0, "name": "",
                        "recent_in": []})
                    u["n_in"] += 1
                    u["last_in"] = max(u["last_in"], float(r.get("ts", 0)))
                    u["name"] = r.get("display_name") or u["name"]
                    txt = (r.get("text") or "").strip()
                    if txt:
                        u["recent_in"] = (u["recent_in"] + [txt[:150]])[-6:]
        except OSError:
            return c["users"]
        c["mtime"], c["users"] = mt, users
        return users

    # -- cooldown checks ----------------------------------------------------

    def _reset_daily_if_needed(self) -> None:
        now = time.time()
        reset_ts = self._state.get("daily_dm_reset", 0)
        if now - reset_ts > 86400:
            self._state["daily_dm_count"] = 0
            self._state["daily_dm_reset"] = now
            self._save_state()

    def can_dm_user(self, user_id: int) -> bool:
        """Check all constraints before DMing a user."""
        if not self.is_dm_eligible(user_id):
            return False

        # Blocked users — never retry
        if str(user_id) in self._state.get("blocked_users", []):
            return False

        self._reset_daily_if_needed()

        # Daily limit
        if self._state.get("daily_dm_count", 0) >= PROACTIVE_DM_MAX_PER_DAY:
            return False

        # Per-user cooldown
        cooldowns = self._state.get("dm_cooldowns", {})
        last_dm = cooldowns.get(str(user_id), 0)
        if time.time() - last_dm < PROACTIVE_DM_COOLDOWN_PER_USER_S:
            return False

        return True

    def mark_blocked(self, user_id: int) -> None:
        """Permanently block DM attempts to a user (blocked bot / can't initiate)."""
        blocked = self._state.setdefault("blocked_users", [])
        uid_str = str(user_id)
        if uid_str not in blocked:
            blocked.append(uid_str)
            self._save_state()
            log.info("Permanently blocked DMs to user %d", user_id)

    def record_proactive_dm(self, user_id: int) -> None:
        """Record that we sent a proactive DM."""
        self._reset_daily_if_needed()
        self._state.setdefault("dm_cooldowns", {})[str(user_id)] = time.time()
        self._state["daily_dm_count"] = self._state.get("daily_dm_count", 0) + 1
        self._save_state()

    # -- trigger: post-group followup ---------------------------------------

    def queue_followup(
        self,
        user_id: int,
        chat_id: int,
        exchange_summary: str,
        topic_tags: list[str] | None = None,
    ) -> None:
        """Queue a DM followup after an engaging group exchange."""
        if not self.is_dm_eligible(user_id):
            return

        # Random delay between 2-8 hours
        delay = random.randint(
            PROACTIVE_DM_FOLLOWUP_DELAY_MIN_S,
            PROACTIVE_DM_FOLLOWUP_DELAY_MAX_S,
        )
        earliest = time.time() + delay

        pending = self._state.setdefault("pending_followups", [])

        # Don't duplicate for same user
        for p in pending:
            if str(p.get("user_id")) == str(user_id):
                return

        pending.append({
            "user_id": str(user_id),
            "chat_id": str(chat_id),
            "trigger": "post_group_followup",
            "earliest_send": earliest,
            "reason": exchange_summary[:200],
            "queued_at": time.time(),
            "topic_tags": topic_tags or [],
        })

        # Keep max 20 pending
        self._state["pending_followups"] = pending[-20:]
        self._save_state()
        log.info(
            "Queued DM followup for user %s (delay: %d min)",
            user_id, delay // 60,
        )

    # -- trigger: milestone -------------------------------------------------

    def check_milestone(self, user_id: int, message_count: int) -> Optional[dict]:
        """Check if user hit a milestone worth acknowledging."""
        milestones = [25, 50, 100, 250, 500]
        sent = self._state.get("milestone_sent", {}).get(str(user_id), [])

        for m in milestones:
            if message_count >= m and m not in sent:
                return {
                    "trigger": "milestone",
                    "milestone": m,
                    "reason": f"Hit {m} messages with Aura",
                }
        return None

    def mark_milestone_sent(self, user_id: int, milestone: int) -> None:
        sent = self._state.setdefault("milestone_sent", {})
        user_sent = sent.setdefault(str(user_id), [])
        if milestone not in user_sent:
            user_sent.append(milestone)
        self._save_state()

    # -- DM nudge (group-to-DM encouragement) --------------------------------

    def should_nudge(self, user_id: int, chat_id: int) -> bool:
        """Check if we should inject a DM nudge for this user in this group.

        Returns True only when:
          1. User is NOT DM-eligible (hasn't /start-ed the bot)
          2. Per-group cooldown has elapsed (5 hours)
          3. Random probability check passes (35%)
        """
        if self.is_dm_eligible(user_id):
            return False

        cooldowns = self._state.get("nudge_cooldowns", {})
        last = cooldowns.get(str(chat_id), 0)
        if time.time() - last < DM_NUDGE_COOLDOWN_PER_GROUP_S:
            return False

        return random.random() < DM_NUDGE_PROBABILITY

    def record_nudge(self, chat_id: int) -> None:
        """Record that a DM nudge was injected in this group."""
        self._state.setdefault("nudge_cooldowns", {})[str(chat_id)] = time.time()
        self._save_state()

    # -- get ready actions --------------------------------------------------

    def get_ready_followups(self) -> list[dict]:
        """Return followups whose delay has elapsed and user is still eligible."""
        now = time.time()
        ready = []
        remaining = []

        for p in self._state.get("pending_followups", []):
            # Expire after 24 hours
            if now - p.get("queued_at", now) > 86400:
                continue
            if now >= p.get("earliest_send", 0) and self.can_dm_user(int(p["user_id"])):
                ready.append(p)
            else:
                remaining.append(p)

        self._state["pending_followups"] = remaining
        self._save_state()
        return ready


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
dm_strategy = DMStrategy()
