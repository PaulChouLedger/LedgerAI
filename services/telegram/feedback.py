"""
feedback -- Self-correcting feedback engine.

Collects explicit and implicit feedback about Aura's behavior, processes it
through the LLM to extract actionable patterns, and writes amendments to:

1. learned_directives.json -- global behavior rules Aura taught herself
2. Per-user behavior_notes in profiles.json -- individual adjustments

Feedback sources:
- Explicit: dedicated feedback channel, /feedback command in DMs
- Implicit: detected complaints in groups ("shut up", "repetitive", etc.)
- Observational: temperature system outcomes (ignored, conversation died)

All changes are logged to feedback_audit.json for human review.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DATA_DIR, LLM_ENDPOINT, LLM_TIMEOUT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
LEARNED_DIRECTIVES_FILE = DATA_DIR / "learned_directives.json"
FEEDBACK_AUDIT_FILE = DATA_DIR / "feedback_audit.json"
FEEDBACK_QUEUE_FILE = DATA_DIR / "feedback_queue.json"

# ---------------------------------------------------------------------------
# Implicit complaint patterns (broader than brain.py NEGATIVE_PHRASES)
# ---------------------------------------------------------------------------
COMPLAINT_PATTERNS = [
    # Repetitiveness
    (re.compile(r"\b(you\s+)?(already|just)\s+said\s+that\b", re.I), "repetitive"),
    (re.compile(r"\brepeti", re.I), "repetitive"),
    (re.compile(r"\bsame\s+thing\s+(again|over)", re.I), "repetitive"),
    (re.compile(r"\bbroken\s+record\b", re.I), "repetitive"),
    (re.compile(r"\bwe\s+know\b", re.I), "repetitive"),
    (re.compile(r"\byou\s+keep\s+saying\b", re.I), "repetitive"),
    # Annoying / too much
    (re.compile(r"\bannoy", re.I), "annoying"),
    (re.compile(r"\btoo\s+much\b", re.I), "too_much"),
    (re.compile(r"\bchill\s+out\b", re.I), "too_much"),
    (re.compile(r"\brelax\b", re.I), "too_much"),
    (re.compile(r"\bcalm\s+down\b", re.I), "too_much"),
    (re.compile(r"\btone\s+it\s+down\b", re.I), "too_much"),
    # Too long
    (re.compile(r"\btoo\s+long\b", re.I), "too_long"),
    (re.compile(r"\btl;?dr\b", re.I), "too_long"),
    (re.compile(r"\bwall\s+of\s+text\b", re.I), "too_long"),
    (re.compile(r"\bshort(er)?\s+(please|pls|plz)\b", re.I), "too_long"),
    # Tone / personality issues
    (re.compile(r"\btry[\s-]?hard", re.I), "forced_personality"),
    (re.compile(r"\bcringe\b", re.I), "forced_personality"),
    (re.compile(r"\bforced\b", re.I), "forced_personality"),
    (re.compile(r"\bweird\s+vibe\b", re.I), "forced_personality"),
    (re.compile(r"\bact(ing)?\s+(like|as)\b", re.I), "forced_personality"),
    # Wrong / unhelpful
    (re.compile(r"\bthat'?s?\s+(not\s+)?(wrong|incorrect|false)\b", re.I), "inaccurate"),
    (re.compile(r"\bmade\s+(that|it)\s+up\b", re.I), "inaccurate"),
    (re.compile(r"\bhallucin", re.I), "inaccurate"),
    # Generic negative
    (re.compile(r"\bshut\s*up\b", re.I), "unwanted"),
    (re.compile(r"\bstfu\b", re.I), "unwanted"),
    (re.compile(r"\bno\s*one\s*asked\b", re.I), "unwanted"),
    (re.compile(r"\bnobody\s*asked\b", re.I), "unwanted"),
    (re.compile(r"\bstop\b", re.I), "unwanted"),
]

# Minimum messages to accumulate before processing a batch
FEEDBACK_BATCH_MIN = 3
# Process feedback no more often than this
FEEDBACK_PROCESS_COOLDOWN_S = 3600  # 1 hour
# Max learned directives to prevent prompt bloat
MAX_LEARNED_DIRECTIVES = 25

# ---------------------------------------------------------------------------
# LLM prompts for feedback processing
# ---------------------------------------------------------------------------

FEEDBACK_ANALYZER_SYSTEM = """\
You are a behavioral analyst for Aura, a conversational AI on Telegram.

You are given a batch of user complaints/feedback about Aura's behavior.
Analyze the patterns and produce actionable amendments.

For each distinct issue, produce a JSON object:

{
  "amendments": [
    {
      "rule": "A clear, specific behavioral rule Aura should follow (1 sentence)",
      "category": "repetitive|too_much|too_long|forced_personality|inaccurate|unwanted|tone|other",
      "scope": "global" or "user:<user_id>",
      "confidence": 0.0-1.0,
      "evidence_count": N
    }
  ],
  "summary": "1-2 sentence summary of the feedback patterns"
}

Rules for generating amendments:
- Only create amendments backed by 2+ pieces of evidence (or 1 very strong explicit complaint)
- Be specific: "Don't repeat the same point twice in a conversation" not "Be less repetitive"
- Don't duplicate rules already in the existing directives
- User-scoped amendments should mention specific behavioral adjustments for that person
- Confidence should reflect how clear and actionable the feedback is
- Respond with ONLY the JSON object\
"""

DEDUP_SYSTEM = """\
You are checking if a new behavioral rule duplicates any existing rules.

Existing rules:
{existing_rules}

New rule: "{new_rule}"

Does this new rule duplicate or contradict any existing rule?
Respond with ONLY a JSON object:
{"is_duplicate": true/false, "reason": "brief explanation", "replace_index": null or index_to_replace}

If the new rule refines or improves an existing one, set replace_index to that rule's index.\
"""


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """Collects, processes, and applies behavioral feedback."""

    def __init__(self) -> None:
        self._queue: list[dict] = self._load_queue()
        self._learned: list[dict] = self._load_learned()
        self._audit: list[dict] = self._load_audit()
        self._last_process_ts: float = 0.0
        self._feedback_channel_id: Optional[int] = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_queue(self) -> list[dict]:
        try:
            return json.loads(FEEDBACK_QUEUE_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_queue(self) -> None:
        FEEDBACK_QUEUE_FILE.write_text(json.dumps(self._queue, indent=2))

    def _load_learned(self) -> list[dict]:
        try:
            data = json.loads(LEARNED_DIRECTIVES_FILE.read_text())
            return data.get("rules", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_learned(self) -> None:
        LEARNED_DIRECTIVES_FILE.write_text(json.dumps({
            "rules": self._learned,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "count": len(self._learned),
        }, indent=2))

    def _load_audit(self) -> list[dict]:
        try:
            return json.loads(FEEDBACK_AUDIT_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_audit(self) -> None:
        # Keep last 200 audit entries
        self._audit = self._audit[-200:]
        FEEDBACK_AUDIT_FILE.write_text(json.dumps(self._audit, indent=2))

    # ------------------------------------------------------------------
    # Public: get learned directives for prompt injection
    # ------------------------------------------------------------------

    def get_learned_directives(self) -> str:
        """Return learned rules as a formatted string for system prompt injection.

        Called before every LLM call to include self-learned behavioral rules.
        """
        if not self._learned:
            return ""
        rules = [r["rule"] for r in self._learned if r.get("scope") == "global"]
        if not rules:
            return ""
        block = "\n".join(f"- {r}" for r in rules)
        return (
            "\n\n[LEARNED BEHAVIORAL RULES — self-corrected from user feedback]\n"
            f"{block}\n"
        )

    def get_user_behavior_notes(self, user_id: int) -> str:
        """Return user-specific behavioral adjustments for prompt injection."""
        notes = [
            r["rule"] for r in self._learned
            if r.get("scope") == f"user:{user_id}"
        ]
        if not notes:
            return ""
        block = "\n".join(f"- {n}" for n in notes)
        return f"\n[BEHAVIOR NOTES for this user]\n{block}\n"

    # ------------------------------------------------------------------
    # Public: set feedback channel
    # ------------------------------------------------------------------

    def set_feedback_channel(self, chat_id: int) -> None:
        """Register a chat as the dedicated feedback channel."""
        self._feedback_channel_id = chat_id
        log.info("Feedback channel set: %d", chat_id)

    def is_feedback_channel(self, chat_id: int) -> bool:
        return self._feedback_channel_id is not None and chat_id == self._feedback_channel_id

    # ------------------------------------------------------------------
    # Public: collect feedback
    # ------------------------------------------------------------------

    def record_explicit(self, user_id: int, chat_id: int,
                        display_name: str, text: str) -> None:
        """Record explicit feedback (from feedback channel or /feedback command)."""
        entry = {
            "type": "explicit",
            "user_id": user_id,
            "chat_id": chat_id,
            "display_name": display_name,
            "text": text,
            "ts": time.time(),
        }
        self._queue.append(entry)
        self._save_queue()
        log.info("Explicit feedback from %s: %s", display_name, text[:80])

    def record_implicit(self, user_id: int, chat_id: int,
                        display_name: str, text: str,
                        aura_last_msg: str = "") -> Optional[str]:
        """Detect and record implicit complaints from regular messages.

        Returns the complaint category if detected, None otherwise.
        """
        for pattern, category in COMPLAINT_PATTERNS:
            if pattern.search(text):
                entry = {
                    "type": "implicit",
                    "category": category,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "display_name": display_name,
                    "text": text,
                    "aura_context": aura_last_msg[:200] if aura_last_msg else "",
                    "ts": time.time(),
                }
                self._queue.append(entry)
                self._save_queue()
                log.info("Implicit feedback (%s) from %s: %s",
                         category, display_name, text[:80])
                return category
        return None

    def record_outcome(self, chat_id: int, outcome: str,
                       aura_msg: str = "", context: str = "") -> None:
        """Record an observational outcome (ignored, conversation_died, etc.)."""
        entry = {
            "type": "outcome",
            "outcome": outcome,
            "chat_id": chat_id,
            "aura_msg": aura_msg[:200] if aura_msg else "",
            "context": context[:200] if context else "",
            "ts": time.time(),
        }
        self._queue.append(entry)
        self._save_queue()

    # ------------------------------------------------------------------
    # Public: process feedback batch
    # ------------------------------------------------------------------

    def should_process(self) -> bool:
        """Check if we have enough feedback and cooldown has elapsed."""
        if len(self._queue) < FEEDBACK_BATCH_MIN:
            return False
        if (time.time() - self._last_process_ts) < FEEDBACK_PROCESS_COOLDOWN_S:
            return False
        return True

    def process_feedback(self, llm_call_fn) -> dict:
        """Process queued feedback through LLM and apply amendments.

        Args:
            llm_call_fn: callable(prompt, system) -> str (the LLM call function)

        Returns:
            dict with processing results
        """
        if not self._queue:
            return {"processed": 0, "amendments": 0}

        self._last_process_ts = time.time()
        batch = list(self._queue)
        self._queue = []
        self._save_queue()

        # Format feedback for LLM
        feedback_text = self._format_batch(batch)

        # Include existing rules for dedup context
        existing = "\n".join(
            f"  {i}. {r['rule']}" for i, r in enumerate(self._learned)
        ) if self._learned else "(none yet)"

        prompt = (
            f"Existing learned rules:\n{existing}\n\n"
            f"New feedback batch ({len(batch)} items):\n{feedback_text}"
        )

        try:
            result = llm_call_fn(prompt, FEEDBACK_ANALYZER_SYSTEM)
            if not result:
                log.warning("Empty LLM response for feedback processing")
                return {"processed": len(batch), "amendments": 0}

            amendments = self._parse_amendments(result)
            applied = self._apply_amendments(amendments, llm_call_fn)

            # Audit log
            audit_entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "batch_size": len(batch),
                "amendments_proposed": len(amendments),
                "amendments_applied": applied,
                "feedback_types": self._summarize_types(batch),
            }
            self._audit.append(audit_entry)
            self._save_audit()

            log.info("Feedback processed: %d items → %d amendments applied",
                     len(batch), applied)
            return {"processed": len(batch), "amendments": applied}

        except Exception as e:
            log.error("Feedback processing failed: %s", e)
            # Put feedback back in queue so it's not lost
            self._queue = batch + self._queue
            self._save_queue()
            return {"processed": 0, "amendments": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal: formatting & parsing
    # ------------------------------------------------------------------

    def _format_batch(self, batch: list[dict]) -> str:
        lines = []
        for i, entry in enumerate(batch, 1):
            kind = entry.get("type", "unknown")
            if kind == "explicit":
                lines.append(
                    f"{i}. [EXPLICIT] {entry.get('display_name', '?')}: "
                    f"\"{entry.get('text', '')}\""
                )
            elif kind == "implicit":
                cat = entry.get("category", "?")
                lines.append(
                    f"{i}. [IMPLICIT/{cat}] {entry.get('display_name', '?')}: "
                    f"\"{entry.get('text', '')}\" "
                    f"(Aura had said: \"{entry.get('aura_context', '')}\")"
                )
            elif kind == "outcome":
                lines.append(
                    f"{i}. [OUTCOME/{entry.get('outcome', '?')}] "
                    f"Aura said: \"{entry.get('aura_msg', '')}\" → {entry.get('outcome', '?')}"
                )
        return "\n".join(lines)

    def _parse_amendments(self, llm_response: str) -> list[dict]:
        """Parse LLM JSON response into amendment list."""
        # Strip markdown code fences if present
        text = llm_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
            return data.get("amendments", [])
        except json.JSONDecodeError:
            # Try to find JSON in the response
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("amendments", [])
                except json.JSONDecodeError:
                    pass
            log.warning("Could not parse amendments JSON: %s", text[:200])
            return []

    def _apply_amendments(self, amendments: list[dict],
                          llm_call_fn) -> int:
        """Apply validated amendments. Returns count applied."""
        applied = 0
        for amend in amendments:
            rule = amend.get("rule", "").strip()
            if not rule:
                continue

            confidence = amend.get("confidence", 0.5)
            if confidence < 0.5:
                log.debug("Skipping low-confidence amendment: %s (%.2f)", rule, confidence)
                continue

            scope = amend.get("scope", "global")

            # Check for duplicates among existing rules
            if self._is_duplicate(rule):
                log.debug("Skipping duplicate: %s", rule)
                continue

            # Enforce max rules
            if scope == "global" and self._count_global() >= MAX_LEARNED_DIRECTIVES:
                log.warning("Max learned directives reached (%d), skipping: %s",
                            MAX_LEARNED_DIRECTIVES, rule)
                continue

            # Add the rule
            entry = {
                "rule": rule,
                "category": amend.get("category", "other"),
                "scope": scope,
                "confidence": confidence,
                "evidence_count": amend.get("evidence_count", 1),
                "added": datetime.now(timezone.utc).isoformat(),
            }
            self._learned.append(entry)
            applied += 1
            log.info("New learned directive [%s]: %s", scope, rule)

        if applied > 0:
            self._save_learned()

        return applied

    def _is_duplicate(self, new_rule: str) -> bool:
        """Simple substring/similarity check for duplicates."""
        new_lower = new_rule.lower()
        for existing in self._learned:
            existing_lower = existing["rule"].lower()
            # Exact or near-exact match
            if new_lower == existing_lower:
                return True
            # High overlap: if 80% of words match
            new_words = set(new_lower.split())
            existing_words = set(existing_lower.split())
            if not new_words or not existing_words:
                continue
            overlap = len(new_words & existing_words) / max(len(new_words), len(existing_words))
            if overlap > 0.75:
                return True
        return False

    def _count_global(self) -> int:
        return sum(1 for r in self._learned if r.get("scope") == "global")

    def _summarize_types(self, batch: list[dict]) -> dict:
        types: dict[str, int] = {}
        for entry in batch:
            key = entry.get("type", "unknown")
            if key == "implicit":
                key = f"implicit/{entry.get('category', '?')}"
            types[key] = types.get(key, 0) + 1
        return types

    # ------------------------------------------------------------------
    # Public: stats for monitoring
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "queue_size": len(self._queue),
            "learned_rules": len(self._learned),
            "global_rules": self._count_global(),
            "user_rules": len(self._learned) - self._count_global(),
            "audit_entries": len(self._audit),
            "last_processed": self._last_process_ts,
        }


# Module-level singleton
feedback_engine = FeedbackEngine()
