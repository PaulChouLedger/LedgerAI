"""
moderation -- Autonomous moderation for Aura Telegram bot.

Provides spam/scam/abuse detection with graduated enforcement:
  - Fast regex layer for obvious violations (scam links, invite spam, slurs)
  - LLM-assisted judgment for borderline cases
  - Graduated escalation: warn → mute → ban

Integrates with social_graph (admin immunity) and reputation (mod action tracking).
Audit trail persisted to data/telegram/moderation.json.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import DATA_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODERATION_FILE = DATA_DIR / "moderation.json"
MODERATION_ENABLED = True

# Paul's Telegram user ID — always immune
OWNER_USER_ID = 110875514

# Escalation thresholds
WARN_BEFORE_MUTE = 3
WARN_BEFORE_BAN = 5
MUTE_DURATIONS = [3600, 86400, 0]  # 1h, 24h, permanent (0 = ban)

# Severity thresholds
SEVERITY_AUTO_ACTION = 0.8   # instant delete + warn/ban
SEVERITY_WARN = 0.5          # warn
SEVERITY_LLM_JUDGE = 0.3    # ask LLM

# Warn expiry — old warns decay after 7 days
WARN_EXPIRY_S = 604800

# ---------------------------------------------------------------------------
# Spam / scam patterns
# ---------------------------------------------------------------------------

# (compiled_regex, severity, category)
_SPAM_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # Telegram invite spam
    (re.compile(r"https?://t\.me/\+\w{10,}", re.I), 0.9, "invite_spam"),
    (re.compile(r"t\.me/\+\w{10,}", re.I), 0.9, "invite_spam"),
    # Crypto scam patterns
    (re.compile(r"(?:earn|free|claim|airdrop).{0,30}(?:\$\d|usd|btc|eth|usdt)", re.I), 0.85, "scam"),
    (re.compile(r"(?:send|transfer)\s+\d+\s*(?:btc|eth|usdt|sol)", re.I), 0.9, "scam"),
    (re.compile(r"(?:click|tap|join).{0,20}(?:link|below|here|now).{0,20}https?://", re.I), 0.7, "scam_link"),
    # WhatsApp / external messenger spam
    (re.compile(r"(?:whatsapp|wa)\.me/", re.I), 0.8, "messenger_spam"),
    # Suspicious TLDs
    (re.compile(r"https?://[^\s]+\.(?:click|top|buzz|icu|surf|monster|rest)/", re.I), 0.6, "suspicious_domain"),
    # Wallet address solicitation
    (re.compile(r"(?:send|dm|message)\s+(?:me|us)\s+(?:your|ur)\s+(?:wallet|address)", re.I), 0.85, "wallet_scam"),
    # "I made $X" type scams
    (re.compile(r"(?:i|I)\s+(?:made|earned|got)\s+\$[\d,]+\s+(?:in|from|with)", re.I), 0.7, "earnings_scam"),
]

_SLUR_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # Keep severity at 0.7 — LLM judges context for borderline cases
    (re.compile(r"\b(?:n[i1]gg[ea3]r?s?|f[a4]gg?[o0]ts?|k[i1]k[e3]s?|sp[i1]cs?|ch[i1]nks?)\b", re.I), 0.85, "slur"),
    (re.compile(r"\b(?:retard(?:ed)?s?)\b", re.I), 0.4, "mild_slur"),
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModerationResult:
    action: str = "none"       # none | warn | delete | mute | ban
    reason: str = ""
    category: str = ""
    severity: float = 0.0
    reply_text: str = ""
    mute_until: float = 0.0    # unix timestamp for mute


# ---------------------------------------------------------------------------
# Warning templates (sound like Aura, not a generic bot)
# ---------------------------------------------------------------------------

_WARN_TEMPLATES = [
    "hey, let's keep it clean yeah?",
    "gonna need you to dial that back.",
    "not the vibe. chill.",
    "easy. keep it respectful.",
    "nah, we're not doing that here.",
]

_MUTE_TEMPLATES = [
    "you've had a few warnings now. taking a break — back in {duration}.",
    "timeout. come back in {duration} and keep it chill.",
    "that's enough for now. see you in {duration}.",
]

# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a group chat moderator assistant. Evaluate whether a message violates community rules.

Violations: spam, scam links, slurs/hate speech, harassment, flood/repetition, wallet scams.
NOT violations: disagreements, mild profanity, off-topic chat, sarcasm, heated debate, dark humor.

Be LENIENT. When in doubt, say CLEAN. Only flag clear, unambiguous violations.

Respond with EXACTLY one word: CLEAN, WARN, or DELETE."""


# ---------------------------------------------------------------------------
# Persistence helpers
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
# Moderator
# ---------------------------------------------------------------------------

class Moderator:
    """Autonomous moderation engine with graduated enforcement."""

    def __init__(self) -> None:
        self._data: dict = _load_json(MODERATION_FILE, {})

    def _save(self) -> None:
        _save_json(MODERATION_FILE, self._data)

    def _ensure_user(self, user_id: int) -> dict:
        key = str(user_id)
        if key not in self._data:
            self._data[key] = {
                "warns": 0,
                "last_warn_at": None,
                "muted_until": None,
                "banned": False,
                "history": [],
            }
        return self._data[key]

    # -- immunity -----------------------------------------------------------

    def _is_immune(self, user_id: int, chat_id: int) -> bool:
        """Owner and per-group admins are immune from moderation."""
        if user_id == OWNER_USER_ID:
            return True
        try:
            from social_graph import social_graph
            # Check per-group admin status
            key = str(user_id)
            user = social_graph._data["users"].get(key)
            if user and int(chat_id) in user.get("admin_of", []):
                return True
        except Exception:
            pass
        return False

    # -- fast scan ----------------------------------------------------------

    def _fast_scan(self, text: str) -> tuple[float, str, str]:
        """Regex-based fast scan. Returns (severity, category, reason)."""
        best_severity = 0.0
        best_category = ""
        best_reason = ""

        for pattern, severity, category in _SPAM_PATTERNS:
            if pattern.search(text):
                if severity > best_severity:
                    best_severity = severity
                    best_category = category
                    best_reason = f"matched spam pattern: {category}"

        for pattern, severity, category in _SLUR_PATTERNS:
            if pattern.search(text):
                if severity > best_severity:
                    best_severity = severity
                    best_category = category
                    best_reason = f"matched slur pattern: {category}"

        # Flood check — same exact message repeated (checked externally)
        return best_severity, best_category, best_reason

    def _check_flood(self, user_id: int, chat_id: int, text: str) -> bool:
        """Check if user is flooding (same message 3+ times in recent buffer)."""
        try:
            from context import context_buffer
            recent = context_buffer.get_recent(chat_id, 15)
            same_count = sum(
                1 for m in recent
                if m.user_id == user_id and m.text.strip().lower() == text.strip().lower()
            )
            return same_count >= 3
        except Exception:
            return False

    # -- LLM judge ----------------------------------------------------------

    def _llm_judge(self, text: str, context_snippet: str, suspected: str) -> str:
        """Ask LLM whether message is a violation. Returns CLEAN, WARN, or DELETE."""
        try:
            from llm import llm_call
            prompt = (
                f"Suspected violation: {suspected}\n\n"
                f"Recent context:\n{context_snippet}\n\n"
                f"Message to evaluate:\n{text}\n\n"
                f"Verdict (one word):"
            )
            result = llm_call(prompt, JUDGE_SYSTEM, max_tokens=10)
            if result:
                word = result.strip().upper().split()[0]
                if word in ("CLEAN", "WARN", "DELETE"):
                    log.info("LLM judge: %s for '%s' (suspected: %s)", word, text[:60], suspected)
                    return word
            return "CLEAN"
        except Exception as e:
            log.error("LLM judge failed: %s", e)
            return "CLEAN"  # fail open

    # -- escalation logic ---------------------------------------------------

    def _get_active_warns(self, user_id: int) -> int:
        """Count non-expired warns."""
        record = self._ensure_user(user_id)
        now = time.time()
        active = sum(
            1 for h in record.get("history", [])
            if h.get("action") in ("warn", "warn_delete")
            and now - h.get("ts", 0) < WARN_EXPIRY_S
        )
        return active

    def _next_action(self, user_id: int, severity: float) -> str:
        """Determine action based on severity and warn history."""
        record = self._ensure_user(user_id)

        # Already banned
        if record.get("banned"):
            return "ban"

        # Currently muted — just delete
        muted_until = record.get("muted_until") or 0
        if muted_until > time.time():
            return "delete"

        active_warns = self._get_active_warns(user_id)

        # High severity = instant delete + action
        if severity >= SEVERITY_AUTO_ACTION:
            if active_warns >= WARN_BEFORE_BAN:
                return "ban"
            if active_warns >= WARN_BEFORE_MUTE:
                return "mute"
            return "warn_delete"  # delete msg + warn

        # Medium severity = warn (keep message)
        if severity >= SEVERITY_WARN:
            if active_warns >= WARN_BEFORE_BAN:
                return "ban"
            if active_warns >= WARN_BEFORE_MUTE:
                return "mute"
            return "warn"

        return "none"

    def _mute_duration(self, user_id: int) -> int:
        """Get mute duration based on how many times user has been muted."""
        record = self._ensure_user(user_id)
        mute_count = sum(
            1 for h in record.get("history", [])
            if h.get("action") == "mute"
        )
        idx = min(mute_count, len(MUTE_DURATIONS) - 1)
        return MUTE_DURATIONS[idx]

    # -- main evaluate ------------------------------------------------------

    def evaluate(
        self, user_id: int, chat_id: int, display_name: str, text: str,
    ) -> ModerationResult:
        """Evaluate a message for moderation action.

        Returns ModerationResult with action to take.
        Called from bot.py _handle_group() before the decision engine.
        """
        if not MODERATION_ENABLED:
            return ModerationResult()

        if self._is_immune(user_id, chat_id):
            return ModerationResult()

        # Fast regex scan
        severity, category, reason = self._fast_scan(text)

        # Flood check
        if not severity and self._check_flood(user_id, chat_id, text):
            severity = 0.85
            category = "flood"
            reason = "repeated identical messages"

        # If nothing caught, pass
        if severity < SEVERITY_LLM_JUDGE:
            return ModerationResult()

        # Borderline — ask LLM
        if severity < SEVERITY_WARN:
            try:
                from context import context_buffer
                recent = context_buffer.get_recent(chat_id, 5)
                context_snippet = "\n".join(
                    f"{m.display_name}: {m.text}" for m in recent
                )
            except Exception:
                context_snippet = ""

            verdict = self._llm_judge(text, context_snippet, reason)
            if verdict == "CLEAN":
                return ModerationResult()
            elif verdict == "DELETE":
                severity = max(severity, SEVERITY_AUTO_ACTION)
            else:  # WARN
                severity = max(severity, SEVERITY_WARN)

        # Determine action
        action = self._next_action(user_id, severity)
        if action == "none":
            return ModerationResult()

        # Build result
        result = ModerationResult(
            action=action,
            reason=reason,
            category=category,
            severity=severity,
        )

        # Generate reply text
        import random as _rand
        if action == "warn" or action == "warn_delete":
            result.reply_text = _rand.choice(_WARN_TEMPLATES)
        elif action == "mute":
            duration_s = self._mute_duration(user_id)
            if duration_s == 0:
                # Escalate to ban
                result.action = "ban"
            else:
                result.mute_until = time.time() + duration_s
                dur_str = f"{duration_s // 3600}h" if duration_s >= 3600 else f"{duration_s // 60}m"
                result.reply_text = _rand.choice(_MUTE_TEMPLATES).format(duration=dur_str)

        return result

    # -- record action ------------------------------------------------------

    def record_action(
        self, user_id: int, chat_id: int, action: str,
        reason: str, text_snippet: str, display_name: str = "",
    ) -> None:
        """Record a moderation action in the user's history."""
        record = self._ensure_user(user_id)

        entry = {
            "ts": time.time(),
            "chat_id": int(chat_id),
            "action": action,
            "reason": reason,
            "text_snippet": text_snippet[:200],
            "display_name": display_name,
        }
        record["history"].append(entry)
        # Keep last 50 actions per user
        record["history"] = record["history"][-50:]

        if action in ("warn", "warn_delete"):
            record["warns"] = record.get("warns", 0) + 1
            record["last_warn_at"] = time.time()
        elif action == "mute":
            record["warns"] = record.get("warns", 0) + 1
            record["last_warn_at"] = time.time()
        elif action == "ban":
            record["banned"] = True

        self._save()
        log.info(
            "MOD recorded: %s on %s (%d) in %d — %s",
            action, display_name, user_id, chat_id, reason,
        )

    # -- queries ------------------------------------------------------------

    def get_user_record(self, user_id: int) -> dict:
        return self._data.get(str(user_id), {})

    def is_banned(self, user_id: int) -> bool:
        record = self._data.get(str(user_id))
        return bool(record and record.get("banned"))

    def get_recent_actions(self, n: int = 20) -> list[dict]:
        """Get the most recent moderation actions across all users."""
        all_actions = []
        for uid, record in self._data.items():
            for h in record.get("history", []):
                h_copy = dict(h)
                h_copy["user_id"] = uid
                all_actions.append(h_copy)
        all_actions.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return all_actions[:n]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
moderator = Moderator()
