"""
referral_rewards -- Referral tracking and reward tier system for Aura Telegram bot.

Tracks who referred whom, computes referral tiers (connector/ambassador/founder),
and generates deep links for the /referral command.

Tiers:
  - connector:  3+ referrals
  - ambassador: 10+ referrals
  - founder:    25+ referrals

Referral links use Telegram deep links: t.me/TheRealAura_bot?start=ref_<user_id>
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import REFERRAL_TIERS, DATA_DIR

log = logging.getLogger(__name__)

_REFERRAL_FILE = DATA_DIR / "referral_data.json"


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


class ReferralTracker:
    """Tracks referrals and computes reward tiers."""

    def __init__(self) -> None:
        self._data: dict = _load_json(_REFERRAL_FILE, {
            "referrals": {},   # {referrer_id: [referred_user_ids]}
            "referred_by": {}, # {user_id: referrer_id}
        })

    def _save(self) -> None:
        _save_json(_REFERRAL_FILE, self._data)

    def record_referral(self, referrer_id: int, referred_id: int) -> None:
        """Record that referrer_id referred referred_id."""
        key = str(referrer_id)
        referrals = self._data.setdefault("referrals", {})
        user_list = referrals.setdefault(key, [])
        rid = int(referred_id)
        if rid not in user_list:
            user_list.append(rid)
            self._data.setdefault("referred_by", {})[str(referred_id)] = referrer_id
            self._save()
            log.info("Referral recorded: %d → %d (total: %d)", referrer_id, referred_id, len(user_list))

    def get_referral_count(self, user_id: int) -> int:
        referrals = self._data.get("referrals", {}).get(str(user_id), [])
        return len(referrals)

    def get_tier(self, user_id: int) -> Optional[str]:
        """Compute referral tier for a user."""
        count = self.get_referral_count(user_id)
        tier = None
        for tier_name, threshold in sorted(REFERRAL_TIERS.items(), key=lambda x: x[1]):
            if count >= threshold:
                tier = tier_name
        return tier

    def get_tier_progress(self, user_id: int) -> dict:
        """Return current tier, count, and next tier info."""
        count = self.get_referral_count(user_id)
        current_tier = self.get_tier(user_id)
        sorted_tiers = sorted(REFERRAL_TIERS.items(), key=lambda x: x[1])

        next_tier = None
        next_threshold = None
        for tier_name, threshold in sorted_tiers:
            if count < threshold:
                next_tier = tier_name
                next_threshold = threshold
                break

        return {
            "count": count,
            "tier": current_tier,
            "next_tier": next_tier,
            "next_threshold": next_threshold,
            "remaining": (next_threshold - count) if next_threshold else 0,
        }

    def generate_link(self, user_id: int, bot_username: str = "TheRealAura_bot") -> str:
        """Generate a referral deep link for this user."""
        return f"https://t.me/{bot_username}?start=ref_{user_id}"

    def get_referred_by(self, user_id: int) -> Optional[int]:
        return self._data.get("referred_by", {}).get(str(user_id))

    def get_top_referrers(self, n: int = 10) -> list[dict]:
        """Return top referrers sorted by count."""
        result = []
        for uid, refs in self._data.get("referrals", {}).items():
            result.append({
                "user_id": int(uid),
                "count": len(refs),
                "tier": self.get_tier(int(uid)),
            })
        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:n]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
referral_tracker = ReferralTracker()
