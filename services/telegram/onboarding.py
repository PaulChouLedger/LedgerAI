"""
onboarding -- Graduated engagement ramp for new groups.

First impressions matter. When Aura joins a new group, she follows a
graduated ramp:

  Phase 1 (0-2 hours):  Silent — only respond to direct mentions/replies
  Phase 2 (2-24 hours): Minimal — only hard rules + score > 0.8
  Phase 3 (24-72 hours): Gradual — standard engine with dampening
  Phase 4 (72+ hours):  Full participation

This prevents Aura from dominating a new group before she's read the room.
"""

from __future__ import annotations

import logging
import time

from config import (
    ONBOARDING_SILENT_PHASE_S,
    ONBOARDING_MINIMAL_PHASE_S,
    ONBOARDING_GRADUAL_PHASE_S,
)
from reputation import reputation_tracker

log = logging.getLogger(__name__)


def get_onboarding_phase(chat_id: int) -> str:
    """Return the current onboarding phase for a group.

    Returns: "silent", "minimal", "gradual", or "full"

    Groups that already have engagement history (total_responses > 0 or
    warmth_level beyond "new") are treated as established regardless of
    joined_at timestamp.  This prevents a service restart from resetting
    the onboarding clock on groups Aura has been active in.
    """
    key = str(chat_id)
    entry = reputation_tracker._data.get(key)
    if entry is None:
        return "full"  # Unknown group, treat as established

    # Skip onboarding for groups with existing engagement history
    if entry.get("total_responses", 0) > 0:
        return "full"
    if entry.get("warmth_level", "new") != "new":
        return "full"

    joined_at = entry.get("joined_at")
    if joined_at is None:
        return "full"

    elapsed = time.time() - joined_at

    if elapsed < ONBOARDING_SILENT_PHASE_S:
        return "silent"
    elif elapsed < ONBOARDING_MINIMAL_PHASE_S:
        return "minimal"
    elif elapsed < ONBOARDING_GRADUAL_PHASE_S:
        return "gradual"
    else:
        return "full"


def should_override_decision(chat_id: int, score: float, is_hard_rule: bool) -> bool | None:
    """Check if onboarding phase should override the decision engine.

    Returns:
        True  — force respond (hard rule in any phase)
        False — force silent (onboarding override)
        None  — let the decision engine decide normally
    """
    phase = get_onboarding_phase(chat_id)

    if phase == "full":
        return None  # No override

    if phase == "silent":
        # Only respond to hard rules (direct mention, reply to bot)
        if is_hard_rule:
            log.info("Onboarding [silent]: allowing hard rule in %d", chat_id)
            return True
        return False

    if phase == "minimal":
        # Hard rules always pass
        if is_hard_rule:
            return True
        # Only very high scores (one amazing comment)
        if score > 0.80:
            log.info("Onboarding [minimal]: high score %.2f in %d", score, chat_id)
            return True
        return False

    if phase == "gradual":
        # Hard rules always pass
        if is_hard_rule:
            return True
        # Standard engine but with dampening — raise threshold by 0.15
        # Return None to let engine decide, but the brain will apply dampening
        return None

    return None


def get_warmth_dampening(chat_id: int) -> float:
    """Return a threshold modifier for the gradual onboarding phase.

    Returns 0.0 for full participation, 0.15 for gradual phase.
    """
    phase = get_onboarding_phase(chat_id)
    if phase == "gradual":
        return 0.15
    return 0.0
