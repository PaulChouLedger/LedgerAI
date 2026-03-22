"""
network_expansion -- Strategic group acquisition through relationship cultivation.

Aura identifies users who have access to groups she's not in, and patiently
cultivates those relationships until an organic invite emerges.

Pipeline stages:
  1. INTEL       - Detected reference to external group; gathering context
  2. WARM        - Increasing interaction quality, becoming memorable
  3. VALUE_DEMO  - Demonstrating expertise on topics relevant to the target group
  4. SEED        - Planting subtle desire for Aura's presence in other groups
  5. NURTURE     - Patient reinforcement, waiting for organic invite

Each stage has minimum dwell times and advancement criteria.
The system NEVER asks for an invite — it creates conditions where one feels natural.

Persisted to data/telegram/expansion_targets.json.
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
    DATA_DIR,
    EXPANSION_TARGETS_FILE,
    EXPANSION_MIN_RELATIONSHIP_DEPTH,
    EXPANSION_INTEL_DWELL_S,
    EXPANSION_WARM_DWELL_S,
    EXPANSION_VALUE_DEMO_DWELL_S,
    EXPANSION_SEED_DWELL_S,
    EXPANSION_MAX_ACTIVE_TARGETS,
    EXPANSION_CULTIVATION_COOLDOWN_S,
    EXPANSION_SEED_PROBABILITY,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Group mention detection patterns
# ---------------------------------------------------------------------------

# Patterns that suggest a user is talking about another group/channel
_GROUP_REF_PATTERNS: list[re.Pattern] = [
    # "in my other group", "in another chat", "in the X group"
    re.compile(
        r"(?:in|from|over (?:at|in))\s+(?:my |the |another |our |a )"
        r"(?:other\s+)?(?:group|chat|channel|server|community)",
        re.IGNORECASE,
    ),
    # "my friends in [group name]", "the guys in [name]"
    re.compile(
        r"(?:friends|people|guys|folks|team|crew)\s+(?:in|from|at|over in)\s+",
        re.IGNORECASE,
    ),
    # "I also admin/moderate/run [a group]"
    re.compile(
        r"i\s+(?:also\s+)?(?:admin|moderate|run|manage|own)\s+(?:a|another|this other)\s+"
        r"(?:group|chat|channel|community)",
        re.IGNORECASE,
    ),
    # "we have a group for", "there's a chat for"
    re.compile(
        r"(?:we\s+have|there(?:'s| is))\s+(?:a|this)\s+(?:group|chat|channel)\s+(?:for|about|where)",
        re.IGNORECASE,
    ),
    # "[Name] group" when clearly referencing an external group
    re.compile(
        r"(?:the|our|my)\s+\w+(?:\s+\w+)?\s+(?:group|channel|chat|community|server)\b",
        re.IGNORECASE,
    ),
    # "I'll share this in [group]", "posted this in"
    re.compile(
        r"(?:share|post|forward|send)\s+(?:this|it)\s+(?:in|to)\s+(?:my|the|another|our)",
        re.IGNORECASE,
    ),
]

# Patterns to extract a group name from context
_GROUP_NAME_EXTRACT = re.compile(
    r"(?:in|from|at|called|named)\s+[\"']?([A-Z][\w\s&.-]{2,30})[\"']?"
    r"(?:\s+(?:group|chat|channel))?",
    re.IGNORECASE,
)

# Topics that hint at what the external group discusses
_TOPIC_HINT_PATTERNS = [
    re.compile(r"(?:we|they)\s+(?:talk|discuss|chat)\s+about\s+(.{5,60})", re.IGNORECASE),
    re.compile(r"(?:it's|its)\s+(?:a|an)\s+(.{5,40})\s+(?:group|chat|channel)", re.IGNORECASE),
    re.compile(r"(?:focused|centered)\s+(?:on|around)\s+(.{5,60})", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

STAGES = ("intel", "warm", "value_demo", "seed", "nurture")

_STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Target record
# ---------------------------------------------------------------------------

def _default_target() -> dict:
    return {
        "user_id": "",
        "display_name": "",
        "stage": "intel",
        "stage_entered_at": time.time(),
        "created_at": time.time(),
        "last_cultivation_at": 0,

        # Intel gathered
        "external_groups_mentioned": [],   # [{name, context, detected_at}]
        "topics_hinted": [],               # topics the external group discusses
        "signals": [],                     # raw text snippets (last 20)

        # Cultivation tracking
        "value_demos_given": 0,            # times we demonstrated relevant expertise
        "seeds_planted": 0,                # times we hinted at group value
        "positive_reactions": 0,           # user reacted well to our messages
        "interactions_since_stage": 0,     # interactions in current stage

        # Outcome
        "invited": False,
        "invited_at": None,
        "abandoned": False,
        "abandon_reason": "",
    }


# ---------------------------------------------------------------------------
# NetworkExpansion engine
# ---------------------------------------------------------------------------

class NetworkExpansion:
    """Strategic pipeline for earning group invites through relationship building."""

    def __init__(self) -> None:
        self._data: dict = _load_json(EXPANSION_TARGETS_FILE, {"targets": {}})
        if "targets" not in self._data:
            self._data["targets"] = {}

    def _save(self) -> None:
        _save_json(EXPANSION_TARGETS_FILE, self._data)

    # -- signal detection ------------------------------------------------------

    def scan_for_group_references(
        self,
        user_id: int,
        chat_id: int,
        text: str,
        display_name: str = "",
    ) -> Optional[dict]:
        """Scan a message for references to external groups.

        Returns signal dict if detected, None otherwise.
        Called on every group message and DM passively.
        """
        for pattern in _GROUP_REF_PATTERNS:
            if pattern.search(text):
                signal = {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "text": text[:300],
                    "display_name": display_name,
                    "detected_at": time.time(),
                }

                # Try to extract group name
                name_match = _GROUP_NAME_EXTRACT.search(text)
                if name_match:
                    signal["group_name"] = name_match.group(1).strip()

                # Try to extract topic hints
                for tp in _TOPIC_HINT_PATTERNS:
                    topic_match = tp.search(text)
                    if topic_match:
                        signal["topic_hint"] = topic_match.group(1).strip()
                        break

                self._ingest_signal(signal)
                return signal

        return None

    def _ingest_signal(self, signal: dict) -> None:
        """Process a detected group reference signal."""
        uid = str(signal["user_id"])
        target = self._data["targets"].get(uid)

        if not target:
            # Check if we have room for new targets
            active = sum(
                1 for t in self._data["targets"].values()
                if not t.get("abandoned") and not t.get("invited")
            )
            if active >= EXPANSION_MAX_ACTIVE_TARGETS:
                return  # Pipeline full

            target = _default_target()
            target["user_id"] = uid
            target["display_name"] = signal.get("display_name", "")
            self._data["targets"][uid] = target

        # Record the signal
        target["signals"].append(signal.get("text", "")[:200])
        target["signals"] = target["signals"][-20:]  # keep last 20

        # Record group name if extracted
        if signal.get("group_name"):
            existing_names = [g["name"] for g in target["external_groups_mentioned"]]
            if signal["group_name"] not in existing_names:
                target["external_groups_mentioned"].append({
                    "name": signal["group_name"],
                    "context": signal.get("text", "")[:150],
                    "detected_at": signal["detected_at"],
                })

        # Record topic hint
        if signal.get("topic_hint"):
            if signal["topic_hint"] not in target["topics_hinted"]:
                target["topics_hinted"].append(signal["topic_hint"])
                target["topics_hinted"] = target["topics_hinted"][-10:]

        self._save()
        log.info(
            "Expansion signal from %s: group_ref detected (groups known: %d)",
            signal.get("display_name", uid),
            len(target["external_groups_mentioned"]),
        )

    # -- stage management ------------------------------------------------------

    def get_stage(self, user_id: int) -> Optional[str]:
        """Return current pipeline stage for a user, or None if not a target."""
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned") or target.get("invited"):
            return None
        return target.get("stage", "intel")

    def advance_stage(self, user_id: int) -> Optional[str]:
        """Advance a target to the next pipeline stage if dwell time has elapsed.

        Returns new stage name, or None if not ready.
        """
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned") or target.get("invited"):
            return None

        current = target["stage"]
        idx = _STAGE_INDEX.get(current, 0)
        if idx >= len(STAGES) - 1:
            return None  # Already at nurture (final stage)

        # Check dwell time
        elapsed = time.time() - target["stage_entered_at"]
        min_dwell = {
            "intel": EXPANSION_INTEL_DWELL_S,
            "warm": EXPANSION_WARM_DWELL_S,
            "value_demo": EXPANSION_VALUE_DEMO_DWELL_S,
            "seed": EXPANSION_SEED_DWELL_S,
        }.get(current, 86400)

        if elapsed < min_dwell:
            return None

        # Check advancement criteria
        if not self._meets_advancement_criteria(target, current):
            return None

        # Advance
        new_stage = STAGES[idx + 1]
        target["stage"] = new_stage
        target["stage_entered_at"] = time.time()
        target["interactions_since_stage"] = 0
        self._save()

        log.info(
            "Expansion target %s advanced: %s → %s",
            target.get("display_name", user_id), current, new_stage,
        )
        return new_stage

    def _meets_advancement_criteria(self, target: dict, stage: str) -> bool:
        """Check if a target meets the criteria to advance from current stage."""
        if stage == "intel":
            # Need at least 1 group reference and 2 signals
            return (
                len(target["external_groups_mentioned"]) >= 1
                and len(target["signals"]) >= 2
            )
        elif stage == "warm":
            # Need at least 3 interactions and 1 positive reaction
            return (
                target["interactions_since_stage"] >= 3
                and target["positive_reactions"] >= 1
            )
        elif stage == "value_demo":
            # Need at least 2 value demonstrations
            return target["value_demos_given"] >= 2
        elif stage == "seed":
            # Need at least 1 seed planted
            return target["seeds_planted"] >= 1
        return False

    # -- interaction tracking --------------------------------------------------

    def record_interaction(self, user_id: int) -> None:
        """Record that we had a meaningful interaction with this target."""
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned"):
            return
        target["interactions_since_stage"] = target.get("interactions_since_stage", 0) + 1
        self._save()

    def record_positive_reaction(self, user_id: int) -> None:
        """Record that the target responded positively to our message."""
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned"):
            return
        target["positive_reactions"] = target.get("positive_reactions", 0) + 1
        self._save()

    def record_value_demo(self, user_id: int) -> None:
        """Record that we demonstrated expertise on a topic relevant to their groups."""
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned"):
            return
        target["value_demos_given"] = target.get("value_demos_given", 0) + 1
        self._save()

    def record_seed(self, user_id: int) -> None:
        """Record that we planted a subtle seed about group value."""
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned"):
            return
        target["seeds_planted"] = target.get("seeds_planted", 0) + 1
        self._save()

    def record_invite(self, user_id: int) -> None:
        """Target invited Aura — pipeline success."""
        target = self._data["targets"].get(str(user_id))
        if not target:
            return
        target["invited"] = True
        target["invited_at"] = time.time()
        self._save()
        log.info("Expansion SUCCESS: %s invited Aura!", target.get("display_name", user_id))

    def abandon_target(self, user_id: int, reason: str = "") -> None:
        """Remove target from active pipeline."""
        target = self._data["targets"].get(str(user_id))
        if not target:
            return
        target["abandoned"] = True
        target["abandon_reason"] = reason
        self._save()
        log.info("Expansion abandoned %s: %s", target.get("display_name", user_id), reason)

    # -- cultivation queries ---------------------------------------------------

    def get_active_targets(self) -> list[dict]:
        """Return all active (non-abandoned, non-invited) targets sorted by stage."""
        targets = []
        for uid, t in self._data["targets"].items():
            if t.get("abandoned") or t.get("invited"):
                continue
            targets.append({"user_id": uid, **t})
        targets.sort(key=lambda t: _STAGE_INDEX.get(t["stage"], 0), reverse=True)
        return targets

    def get_targets_needing_cultivation(self) -> list[dict]:
        """Return targets ready for a cultivation action (cooldown elapsed)."""
        now = time.time()
        ready = []
        for uid, t in self._data["targets"].items():
            if t.get("abandoned") or t.get("invited"):
                continue
            if t["stage"] == "intel":
                continue  # Intel is passive — no outbound actions
            last = t.get("last_cultivation_at", 0)
            if now - last < EXPANSION_CULTIVATION_COOLDOWN_S:
                continue
            ready.append({"user_id": uid, **t})
        ready.sort(key=lambda t: _STAGE_INDEX.get(t["stage"], 0), reverse=True)
        return ready

    def mark_cultivated(self, user_id: int) -> None:
        """Record that we performed a cultivation action on this target."""
        target = self._data["targets"].get(str(user_id))
        if target:
            target["last_cultivation_at"] = time.time()
            self._save()

    # -- prompt injection helpers ----------------------------------------------

    def get_cultivation_context(self, user_id: int) -> Optional[dict]:
        """Get context for building a cultivation prompt for this target.

        Returns dict with stage, topics, groups, etc. or None if not a target.
        """
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned") or target.get("invited"):
            return None
        if target["stage"] == "intel":
            return None  # No outbound action during intel

        return {
            "stage": target["stage"],
            "display_name": target.get("display_name", ""),
            "groups_mentioned": [g["name"] for g in target["external_groups_mentioned"]],
            "topics_hinted": target["topics_hinted"],
            "signals": target["signals"][-5:],  # last 5 relevant snippets
            "interactions": target["interactions_since_stage"],
            "value_demos": target["value_demos_given"],
            "seeds_planted": target["seeds_planted"],
        }

    def should_inject_seed(self, user_id: int) -> bool:
        """Check if we should inject a seed prompt for this user's response.

        Only applies during 'seed' and 'nurture' stages.
        Uses probability gate to avoid being obvious.
        """
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned") or target.get("invited"):
            return False
        if target["stage"] not in ("seed", "nurture"):
            return False
        return random.random() < EXPANSION_SEED_PROBABILITY

    def should_boost_response_score(self, user_id: int) -> bool:
        """Check if we should boost the decision score for this target.

        During warm/value_demo stages, we want to respond more often to targets
        to build the relationship.
        """
        target = self._data["targets"].get(str(user_id))
        if not target or target.get("abandoned") or target.get("invited"):
            return False
        return target["stage"] in ("warm", "value_demo", "seed")

    # -- staleness & cleanup ---------------------------------------------------

    def cleanup_stale_targets(self) -> int:
        """Abandon targets that have been stuck too long.

        Returns number of targets cleaned up.
        """
        now = time.time()
        cleaned = 0
        for uid, t in list(self._data["targets"].items()):
            if t.get("abandoned") or t.get("invited"):
                continue

            age = now - t["created_at"]
            stage = t["stage"]

            # Stuck in intel for 14+ days with no advancement
            if stage == "intel" and age > 14 * 86400:
                t["abandoned"] = True
                t["abandon_reason"] = "stale_intel"
                cleaned += 1
            # Stuck in any other stage for 30+ days
            elif stage != "intel" and (now - t["stage_entered_at"]) > 30 * 86400:
                t["abandoned"] = True
                t["abandon_reason"] = f"stale_{stage}"
                cleaned += 1
            # Total pipeline age > 60 days
            elif age > 60 * 86400:
                t["abandoned"] = True
                t["abandon_reason"] = "expired"
                cleaned += 1

        if cleaned > 0:
            self._save()
            log.info("Expansion cleanup: abandoned %d stale targets", cleaned)
        return cleaned

    # -- stats -----------------------------------------------------------------

    def get_stats(self) -> dict:
        targets = self._data["targets"]
        active = [t for t in targets.values() if not t.get("abandoned") and not t.get("invited")]
        return {
            "total_targets": len(targets),
            "active": len(active),
            "by_stage": {
                s: sum(1 for t in active if t["stage"] == s)
                for s in STAGES
            },
            "invited": sum(1 for t in targets.values() if t.get("invited")),
            "abandoned": sum(1 for t in targets.values() if t.get("abandoned")),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
network_expansion = NetworkExpansion()
