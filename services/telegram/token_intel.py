"""
token_intel -- Organic $LEDGER token awareness for Aura Telegram bot.

Controls when and how Aura references LedgerAI / $LEDGER in conversations.
Never salesy, never forced. Personality-driven, warmth-gated, cooldown-enforced.

Hard rules:
  - Never mention $LEDGER in new/warming groups
  - Max 1 token reference per group per 24 hours
  - Never inject if conversation doesn't naturally support it
  - Strip all shill patterns from any response
  - Token DM deepening only for familiar+ relationships
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

from config import (
    TOKEN_INJECTION_PROBABILITY,
    TOKEN_OPINION_PROBABILITY,
    TOKEN_MIN_WARMTH,
    TOKEN_MENTION_COOLDOWN_S,
    TOKEN_DM_MIN_DEPTH,
    TOKEN_MENTION_COOLDOWNS_FILE,
    SHILL_PATTERNS,
    DATA_DIR,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic detection patterns
# ---------------------------------------------------------------------------

_CRYPTO_AI_PATTERNS = [
    re.compile(r'\b(?:decentral\w+|on-?chain|web3|blockchain)\b', re.I),
    re.compile(r'\b(?:ai\s+token|ai\s+crypto|crypto\s+ai)\b', re.I),
    re.compile(r'\b(?:on-?device|edge\s+ai|local\s+inference|edge\s+compute)\b', re.I),
    re.compile(r'\b(?:jetson|nvidia|gpu\s+inference|llm\s+hardware)\b', re.I),
    re.compile(r'\b(?:ai\s+infrastructure|compute\s+network)\b', re.I),
]

_CRYPTO_OPINION_PATTERNS = [
    re.compile(r'\b(?:ai\s+token|ai\s+coin|ai\s+project)s?\b', re.I),
    re.compile(r'\b(?:defi|tokenomics|governance\s+token)\b', re.I),
    re.compile(r'\b(?:vaporware|rug\s*pull|scam)\b', re.I),
    re.compile(r'\b(?:cloud\s+ai|api\s+wrapper)\b', re.I),
]

_TOKEN_CURIOUS_PATTERNS = [
    re.compile(r'\b(?:ledger\s*ai|\$ledger|ledger\s+token|ledger\s+network)\b', re.I),
    re.compile(r'\b(?:what\s+(?:do\s+you|are\s+you)\s+run\s+on)\b', re.I),
    re.compile(r'\b(?:your\s+hardware|your\s+infrastructure|how\s+do\s+you\s+work)\b', re.I),
    re.compile(r'\b(?:are\s+you\s+(?:on-?device|local|decentralized))\b', re.I),
]

_SHILL_ACCUSATION_PATTERNS = [
    re.compile(r'\b(?:shill|shilling|shill\w+)\b', re.I),
    re.compile(r'\b(?:pump\w*|promot\w+)\b.*\b(?:token|coin|ledger)\b', re.I),
    re.compile(r'\bstop\s+(?:promoting|pushing|shilling)\b', re.I),
]

# Compiled shill patterns from config
_COMPILED_SHILL = [re.compile(p, re.I) for p in SHILL_PATTERNS]

# Warmth ordering
_WARMTH_ORDER = ["new", "warming", "established", "trusted"]
_DEPTH_ORDER = ["stranger", "acquaintance", "familiar", "advocate"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_json(path: Path, default) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default.copy() if isinstance(default, dict) else default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# TokenIntel
# ---------------------------------------------------------------------------

class TokenIntel:
    """Controls organic $LEDGER awareness in conversations."""

    def __init__(self) -> None:
        self._cooldowns: dict = _load_json(TOKEN_MENTION_COOLDOWNS_FILE, {})
        self._curious_users: dict = _load_json(
            DATA_DIR / "token_curious.json", {}
        )

    def _save_cooldowns(self) -> None:
        _save_json(TOKEN_MENTION_COOLDOWNS_FILE, self._cooldowns)

    def _save_curious(self) -> None:
        _save_json(DATA_DIR / "token_curious.json", self._curious_users)

    # -- cooldown checks ----------------------------------------------------

    def _cooldown_ok(self, chat_id: int) -> bool:
        """Check if enough time has passed since last token mention in this chat."""
        key = str(chat_id)
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) >= TOKEN_MENTION_COOLDOWN_S

    def _record_mention(self, chat_id: int) -> None:
        self._cooldowns[str(chat_id)] = time.time()
        self._save_cooldowns()

    # -- warmth / depth gates -----------------------------------------------

    def _warmth_sufficient(self, warmth_level: str, required: str) -> bool:
        try:
            return _WARMTH_ORDER.index(warmth_level) >= _WARMTH_ORDER.index(required)
        except ValueError:
            return False

    def _depth_sufficient(self, depth: str, required: str) -> bool:
        try:
            return _DEPTH_ORDER.index(depth) >= _DEPTH_ORDER.index(required)
        except ValueError:
            return False

    # -- topic detection ----------------------------------------------------

    def _is_crypto_ai_topic(self, text: str) -> bool:
        return any(p.search(text) for p in _CRYPTO_AI_PATTERNS)

    def _is_crypto_opinion_topic(self, text: str) -> bool:
        return any(p.search(text) for p in _CRYPTO_OPINION_PATTERNS)

    def _is_token_curious(self, text: str) -> bool:
        return any(p.search(text) for p in _TOKEN_CURIOUS_PATTERNS)

    def _is_shill_accusation(self, text: str) -> bool:
        return any(p.search(text) for p in _SHILL_ACCUSATION_PATTERNS)

    # -- curious user tracking ----------------------------------------------

    def mark_curious(self, user_id: int) -> None:
        """Mark a user as token-curious (asked about LedgerAI/hardware/etc)."""
        key = str(user_id)
        entry = self._curious_users.get(key, {"count": 0, "first_at": time.time()})
        entry["count"] = entry.get("count", 0) + 1
        entry["last_at"] = time.time()
        self._curious_users[key] = entry
        self._save_curious()

    def is_curious(self, user_id: int) -> bool:
        return str(user_id) in self._curious_users

    def get_curiosity_count(self, user_id: int) -> int:
        entry = self._curious_users.get(str(user_id))
        return entry.get("count", 0) if entry else 0

    # -- group injection ----------------------------------------------------

    def maybe_inject_group(
        self,
        chat_id: int,
        user_id: int,
        text: str,
        warmth_level: str,
    ) -> Optional[str]:
        """Decide whether to inject token context into a group response.

        Returns injection prompt string or None.
        """
        # Track curiosity regardless of injection
        if self._is_token_curious(text):
            self.mark_curious(user_id)

        # Handle shill accusations immediately
        if self._is_shill_accusation(text):
            from persona import SHILL_DEFLECT_RESPONSE
            return SHILL_DEFLECT_RESPONSE

        # Gate: warmth level
        if not self._warmth_sufficient(warmth_level, TOKEN_MIN_WARMTH):
            return None

        # Gate: cooldown
        if not self._cooldown_ok(chat_id):
            return None

        # If user is directly asking about LedgerAI/hardware, always answer
        if self._is_token_curious(text):
            from persona import TOKEN_CONTEXT_INJECTION
            self._record_mention(chat_id)
            log.info("Token injection (curious): chat=%d user=%d", chat_id, user_id)
            return TOKEN_CONTEXT_INJECTION

        # Crypto/AI opinion topic — probability gate
        if self._is_crypto_opinion_topic(text):
            if random.random() < TOKEN_OPINION_PROBABILITY:
                from persona import TOKEN_OPINION_INJECTION
                self._record_mention(chat_id)
                log.info("Token injection (opinion): chat=%d user=%d", chat_id, user_id)
                return TOKEN_OPINION_INJECTION

        # General crypto/AI infrastructure topic — lower probability
        if self._is_crypto_ai_topic(text):
            if random.random() < TOKEN_INJECTION_PROBABILITY:
                from persona import TOKEN_CONTEXT_INJECTION
                self._record_mention(chat_id)
                log.info("Token injection (context): chat=%d user=%d", chat_id, user_id)
                return TOKEN_CONTEXT_INJECTION

        return None

    # -- DM injection -------------------------------------------------------

    def maybe_inject_dm(
        self,
        user_id: int,
        text: str,
        relationship_depth: str,
    ) -> Optional[str]:
        """Decide whether to inject token context into a DM response.

        DMs allow deeper engagement than groups. Only for familiar+ users
        who have shown interest.
        """
        # Track curiosity
        if self._is_token_curious(text):
            self.mark_curious(user_id)

        # Handle shill accusations
        if self._is_shill_accusation(text):
            from persona import SHILL_DEFLECT_RESPONSE
            return SHILL_DEFLECT_RESPONSE

        # If user is directly asking about LedgerAI — always answer in DMs
        if self._is_token_curious(text):
            from persona import TOKEN_DM_DEEPENING_INJECTION
            log.info("Token DM injection (curious): user=%d", user_id)
            return TOKEN_DM_DEEPENING_INJECTION

        # For unprompted injection, require relationship depth
        if not self._depth_sufficient(relationship_depth, TOKEN_DM_MIN_DEPTH):
            return None

        # Only inject if they've shown prior curiosity
        if not self.is_curious(user_id):
            return None

        # Crypto/AI topics in DMs with curious familiar+ users
        if self._is_crypto_ai_topic(text) or self._is_crypto_opinion_topic(text):
            if random.random() < TOKEN_OPINION_PROBABILITY:
                from persona import TOKEN_DM_DEEPENING_INJECTION
                log.info("Token DM injection (deepening): user=%d", user_id)
                return TOKEN_DM_DEEPENING_INJECTION

        return None

    # -- shill pattern stripping --------------------------------------------

    def strip_shill_patterns(self, text: str) -> str:
        """Remove any shill-like patterns from a response.

        Applied as a safety net after LLM generation — even if the prompt
        injection slipped, the output gets cleaned.
        """
        original = text
        for pattern in _COMPILED_SHILL:
            text = pattern.sub("", text)
        # Clean up double spaces / orphaned punctuation from removals
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = text.strip()
        if text != original:
            log.info("Stripped shill patterns from response")
        return text

    # -- milestone injections -----------------------------------------------

    def get_milestone_injection(self, user_id: int, message_count: int) -> Optional[str]:
        """Return a milestone injection prompt if the user hit a token-relevant milestone."""
        if message_count >= 100:
            from persona import MILESTONE_100_INJECTION
            return MILESTONE_100_INJECTION
        if message_count >= 50:
            link = "https://t.me/TheRealAura_bot"
            from persona import MILESTONE_50_INJECTION
            return MILESTONE_50_INJECTION.format(link=link)
        return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
token_intel = TokenIntel()
