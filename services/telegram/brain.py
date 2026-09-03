"""
brain -- Adaptive decision engine for group chats: "should Aura respond?"

Two-layer system:
  1. Hard rules (always/never respond) — checked first, no scoring needed
  2. Adaptive scoring — per-group "temperature" that learns from real-time
     feedback: what happens AFTER Aura speaks?

The temperature rises when people engage with Aura (reply, continue topic,
ask follow-ups) and drops when she's ignored, told to shut up, or the
conversation dies after she speaks.

Philosophy: if the room wants you, lean in. If the room doesn't, back off
fast. Direct address is always answered.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import config as _cfg
from config import (
    GROUP_MAX_PER_HOUR,
    CALLBACK_SCORE_BOOST,
    DATA_DIR,
)
from context import context_buffer, Message
from onboarding import should_override_decision, get_warmth_dampening
from reputation import reputation_tracker
from network_expansion import network_expansion

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mention detection
# ---------------------------------------------------------------------------
MENTION_PATTERNS = [
    r"\baura\b",
    r"\b@aura\b",
]

# Normalize common Unicode lookalikes that break mention detection
# (e.g. Cyrillic А/а looks identical to Latin A/a but won't match \baura\b)
_UNICODE_NORMALIZE = str.maketrans({
    "\u0410": "A", "\u0430": "a",  # Cyrillic А/а
    "\u0415": "E", "\u0435": "e",  # Cyrillic Е/е
    "\u041e": "O", "\u043e": "o",  # Cyrillic О/о
    "\u0420": "P", "\u0440": "p",  # Cyrillic Р/р (looks like P)
    "\u0421": "C", "\u0441": "c",  # Cyrillic С/с
    "\u0422": "T", "\u0442": "t",  # Cyrillic Т/т (uppercase only)
    "\u0443": "y",                 # Cyrillic у (looks like y)
})

# ---------------------------------------------------------------------------
# Project questions — the owner shouldn't have to field these
# ---------------------------------------------------------------------------
# (owner, 2026-09-03: "make her defend these questions in area31 with her
# RAG context... i just don't have time to deal with these questions
# anymore"). Two shapes count: a question that names the project, or one
# of the canonical what-is-this / why-different questions which, in a
# project group, can only be about the project. Anything that says "aura"
# is already a hard-rule mention and never reaches this list.
PROJECT_Q_PATTERNS = [
    # anchored: names the project and asks something
    r"(?:ledger\s*ai|\$\s*ledger|\bledger\b|the\s+(?:puck|project|token|device))"
    r"[^.!?]{0,80}\?",
    # canonical unanchored questions
    r"how\s+(?:is|are)\s+(?:this|it|that|you\s+guys|y'?all)\s+(?:any\s+)?different",
    r"why\s+not\s+just\s+use\s+(?:chat\s*gpt|gpt|openai|claude|gemini|grok|alexa|siri)",
    r"what(?:'s|\s+is)\s+the\s+(?:point|use\s*case|utility|difference|moat|edge)",
    r"what\s+(?:does|do)\s+(?:this|it|the\s+project|the\s+token)\s+(?:actually\s+|even\s+)?do",
    r"is\s+(?:my|our|the)\s+data\s+(?:safe|private|secure)",
    r"where\s+(?:does|do|is)\s+(?:my|our|the)\s+data\s+(?:go|end\s+up|live|stored)",
    r"why\s+(?:do\s+(?:we|you)\s+need|use)\s+(?:a\s+)?(?:token|blockchain|crypto)",
    r"how\s+does\s+(?:it|this|the\s+puck|the\s+ai|the\s+device)\s+work",
    r"what\s+makes\s+(?:this|it|you)\s+(?:different|special|unique|better)",
    r"what\s+are\s+you\s+(?:guys\s+)?(?:building|working\s+on|making)",
    r"\bon.?device\s+ai\b",
    r"\broad\s*map\b",
]
_PROJECT_Q_RE = [re.compile(p, re.IGNORECASE) for p in PROJECT_Q_PATTERNS]

# Negative feedback phrases — instant temperature drop
NEGATIVE_PHRASES = [
    r"\bshut\s*up\b", r"\bstfu\b", r"\bstop\s+talk", r"\bquiet\b",
    r"\bno\s*one\s*asked\b", r"\bnobody\s*asked\b", r"\bgo\s+away\b",
    r"\bannoy", r"\bspam", r"\bshh\b", r"\bbot\s+spam\b",
    r"\bstop\s+respond", r"\bstop\s+reply",
]
_NEGATIVE_RE = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PHRASES]

# Per-group hourly counters
_hourly_responses: dict[int, list[float]] = {}

# ---------------------------------------------------------------------------
# Engagement temperature — the core adaptive state
# ---------------------------------------------------------------------------
# Persisted to disk so it survives restarts.
# Per chat_id: {temperature, last_response_ts, pending_outcome, history[]}

_TEMP_FILE = DATA_DIR / "engagement_temp.json"
_temperatures: dict[str, dict] = {}

# Temperature bounds
TEMP_DEFAULT = 0.6
TEMP_MIN = 0.05
TEMP_MAX = 0.95
TEMP_FLOOR_AFTER_NEGATIVE = 0.10  # hard floor after "shut up" etc.

# Outcome tracking: how many messages to wait before judging
OUTCOME_WINDOW_MSGS = 5   # check next 5 messages after Aura speaks
OUTCOME_WINDOW_SECS = 120  # or 2 minutes, whichever comes first

# Temperature adjustments per outcome
TEMP_REPLY_BOOST = 0.10       # someone replied to Aura
TEMP_CONTINUATION_BOOST = 0.06  # conversation continued on Aura's topic
TEMP_QUESTION_BOOST = 0.08    # someone asked Aura a follow-up
TEMP_IGNORED_PENALTY = -0.04  # Aura's message was completely ignored
TEMP_SILENCE_PENALTY = -0.07  # conversation died after Aura spoke
TEMP_NEGATIVE_PENALTY = -0.25  # someone told Aura to shut up
TEMP_DECAY_RATE = 0.005       # per-hour drift toward 0.5 (self-correcting)


def _load_temperatures() -> None:
    global _temperatures
    if _TEMP_FILE.exists():
        try:
            _temperatures = json.loads(_TEMP_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            _temperatures = {}


def _save_temperatures() -> None:
    _TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TEMP_FILE.write_text(json.dumps(_temperatures, indent=2))


def _get_temp(chat_id: int) -> dict:
    key = str(chat_id)
    if key not in _temperatures:
        _temperatures[key] = {
            "temperature": TEMP_DEFAULT,
            "last_response_ts": 0,
            "pending_outcome": False,
            "msgs_since_response": 0,
            "last_response_text": "",
            "history": [],  # last 20 adjustments for debugging
        }
    return _temperatures[key]


def _adjust_temp(chat_id: int, delta: float, reason: str) -> None:
    state = _get_temp(chat_id)
    old = state["temperature"]
    state["temperature"] = max(TEMP_MIN, min(TEMP_MAX, old + delta))
    state["history"].append({
        "ts": time.time(),
        "delta": round(delta, 4),
        "reason": reason,
        "old": round(old, 4),
        "new": round(state["temperature"], 4),
    })
    state["history"] = state["history"][-20:]
    _save_temperatures()
    log.info(
        "Temp %d: %.2f → %.2f (%+.2f, %s)",
        chat_id, old, state["temperature"], delta, reason,
    )


# Load on import
_load_temperatures()


# ---------------------------------------------------------------------------
# Outcome evaluation — called on each new group message
# ---------------------------------------------------------------------------

def evaluate_outcome(chat_id: int, message: Message, is_reply_to_bot: bool) -> None:
    """Check if Aura's last response got engagement or was ignored.

    Called on every group message. If Aura has a pending outcome (she spoke
    recently), we look at what happened next.
    """
    state = _get_temp(chat_id)
    if not state["pending_outcome"]:
        return

    state["msgs_since_response"] = state.get("msgs_since_response", 0) + 1
    elapsed = time.time() - state["last_response_ts"]

    # Check for negative feedback (immediate, don't wait for window)
    text_lower = message.text.translate(_UNICODE_NORMALIZE).lower()
    if any(p.search(text_lower) for p in _NEGATIVE_RE):
        _adjust_temp(chat_id, TEMP_NEGATIVE_PENALTY, "negative feedback")
        state["pending_outcome"] = False
        _save_temperatures()
        return

    # Positive signal: someone replied directly to Aura
    if is_reply_to_bot:
        _adjust_temp(chat_id, TEMP_REPLY_BOOST, "reply to Aura")
        state["pending_outcome"] = False
        _save_temperatures()
        return

    # Positive signal: someone mentioned Aura in follow-up
    if any(re.search(p, text_lower) for p in MENTION_PATTERNS):
        _adjust_temp(chat_id, TEMP_QUESTION_BOOST, "follow-up mention")
        state["pending_outcome"] = False
        _save_temperatures()
        return

    # Positive signal: conversation continued (someone talked within window)
    if state["msgs_since_response"] <= 2 and elapsed < 60:
        # Early continuation — mild positive
        _adjust_temp(chat_id, TEMP_CONTINUATION_BOOST, "conversation continued")
        state["pending_outcome"] = False
        _save_temperatures()
        return

    # Window expired — evaluate
    window_expired = (
        state["msgs_since_response"] >= OUTCOME_WINDOW_MSGS
        or elapsed > OUTCOME_WINDOW_SECS
    )
    if not window_expired:
        return  # still waiting

    # If we got here, the window expired with no engagement
    if state["msgs_since_response"] >= 3:
        # People talked but ignored Aura
        _adjust_temp(chat_id, TEMP_IGNORED_PENALTY, "ignored")
    else:
        # Conversation died after Aura spoke
        _adjust_temp(chat_id, TEMP_SILENCE_PENALTY, "silence after response")

    state["pending_outcome"] = False
    _save_temperatures()


def mark_response(chat_id: int, response_text: str) -> None:
    """Record that Aura just spoke — start tracking outcome."""
    state = _get_temp(chat_id)
    state["pending_outcome"] = True
    state["last_response_ts"] = time.time()
    state["msgs_since_response"] = 0
    state["last_response_text"] = response_text[:200]
    _save_temperatures()


def decay_temperatures() -> None:
    """Drift all temperatures toward 0.5. Called periodically."""
    for key, state in _temperatures.items():
        temp = state["temperature"]
        if abs(temp - TEMP_DEFAULT) < 0.01:
            continue
        # Drift toward 0.5
        if temp > TEMP_DEFAULT:
            state["temperature"] = max(TEMP_DEFAULT, temp - TEMP_DECAY_RATE)
        else:
            state["temperature"] = min(TEMP_DEFAULT, temp + TEMP_DECAY_RATE)
    _save_temperatures()


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    should_respond: bool
    score: float
    reason: str


def should_respond(
    chat_id: int,
    message: Message,
    is_reply_to_bot: bool = False,
    has_callback: bool = False,
) -> Decision:
    """Decide whether Aura should respond to this group message.

    Layer 1: Hard rules (no scoring)
    Layer 1.5: Onboarding override
    Layer 2: Adaptive scoring with temperature
    """
    reasons: list[str] = []
    raw_text = message.text

    # Detect and log Cyrillic lookalikes before normalizing
    _cyrillic = [c for c in raw_text if '\u0400' <= c <= '\u04ff']
    if _cyrillic:
        log.warning(
            "CYRILLIC DETECTED in chat %d from %s: chars=%r in %r",
            chat_id, message.display_name, _cyrillic, raw_text[:120],
        )

    text_lower = raw_text.translate(_UNICODE_NORMALIZE).lower()

    # ── Layer 1: Hard rules ──────────────────────────────────────────

    is_hard_rule = False

    # HARD RULE: Always respond to a reply to Aura's message
    if is_reply_to_bot:
        is_hard_rule = True

    # HARD RULE: Always respond to direct @mention or name mention
    bot_user = _cfg.BOT_USERNAME.lower() if _cfg.BOT_USERNAME else ""
    mentioned = any(re.search(p, text_lower) for p in MENTION_PATTERNS)
    if bot_user and f"@{bot_user}" in text_lower:
        mentioned = True
    if mentioned:
        is_hard_rule = True

    # ── INVIOLABLE: Name mention / @mention / reply ALWAYS responds ──
    # Nothing — no rate limit, no negative feedback, no onboarding phase,
    # no cooldown — can prevent a response to a direct address.
    if is_hard_rule:
        reason = "reply to Aura" if is_reply_to_bot else "direct mention"
        log.info("HARD RULE %d: %s — responding unconditionally", chat_id, reason)
        return Decision(True, 1.0, f"{reason} (inviolable)")

    # Hourly rate limit — only applies to non-hard-rule (organic) responses
    now = time.time()
    hour_ago = now - 3600
    if chat_id not in _hourly_responses:
        _hourly_responses[chat_id] = []
    _hourly_responses[chat_id] = [t for t in _hourly_responses[chat_id] if t > hour_ago]
    if len(_hourly_responses[chat_id]) >= GROUP_MAX_PER_HOUR:
        return Decision(False, 0.0, f"hourly limit ({GROUP_MAX_PER_HOUR}/hr)")

    # Negative feedback — only blocks organic responses, not direct address
    if any(p.search(text_lower) for p in _NEGATIVE_RE):
        _adjust_temp(chat_id, TEMP_NEGATIVE_PENALTY, "negative in message")
        return Decision(False, 0.0, "negative feedback detected")

    # ── Layer 1.5: Onboarding override ───────────────────────────────

    # Pre-check: if onboarding says silent/minimal, we can skip scoring
    pre_override = should_override_decision(chat_id, 0.0, False)
    if pre_override is False:
        return Decision(False, 0.0, "onboarding: not ready yet")

    # ── Layer 2: Adaptive scoring ────────────────────────────────────

    state = _get_temp(chat_id)
    temperature = state["temperature"]
    score = 0.0

    # --- Price FUD / low-effort panic — always engage ---
    _FUD_PATTERNS = [
        r"^\s*help\s*[!.]*\s*$",           # just "help!" by itself
        r"\bwen\s*(pump|moon|lambo)\b",
        r"\bwhy.*(down|dump|drop|crash|red)\b",
        r"\b(is\s+this|project\s+is|this.{0,10})\s*(dead|rug|over)\b",
        r"\b(do\s+something|team\s+do)\b",
        r"\brugg?ed\b",
        r"\bprice\s*\?\s*$",
        r"\bpump\s+(it|when|wen)\b",
        r"\bnobody.*(talks?|cares?|here)\b",
        r"\b(scam|exit\s*scam|ponzi)\b",
        r"\bsell|selling|sold|dump(ing|ed)?\b",
        r"\bwhere.*(team|dev|update)\b",
        r"\b(losing|lost)\s*(money|everything|hope)\b",
    ]
    if any(re.search(p, text_lower) for p in _FUD_PATTERNS):
        score += 0.80
        reasons.append("price FUD / low-effort panic")

    # --- Project questions — engage even unaddressed (see PROJECT_Q_PATTERNS) ---
    elif any(p.search(text_lower) for p in _PROJECT_Q_RE):
        score += 0.75
        reasons.append("project question")

    # --- Conversational context signals ---

    # Active conversation: Aura spoke recently and people are still talking
    recent_3 = context_buffer.get_recent(chat_id, 3)
    aura_in_last_3 = any(m.is_bot for m in recent_3)

    # Conversational turn: Aura was the last bot to speak, and the very next
    # human message looks like a direct continuation (reply-like without the
    # Telegram reply button). This is a near-hard rule — if someone speaks
    # right after Aura in a tight window, they're talking to her.
    is_conversational_turn = False
    if aura_in_last_3:
        # Check if Aura's message is the most recent bot message and
        # only 0-2 human messages have come between then and now
        last_few = context_buffer.get_recent(chat_id, 4)
        # Find Aura's last message position
        for i, m in enumerate(last_few):
            if m.is_bot:
                human_msgs_after = len([mm for mm in last_few[:i] if not mm.is_bot])
                if human_msgs_after <= 1:
                    is_conversational_turn = True
                break

    if is_conversational_turn:
        score += 0.65
        reasons.append("conversational turn")
    elif aura_in_last_3:
        score += 0.40
        reasons.append("active conversation")

    # Unanswered question in group
    if text_lower.rstrip().endswith("?"):
        score += 0.25
        reasons.append("question")

    # Conversation lull (5+ min silence then someone speaks)
    last_age = context_buffer.last_message_age(chat_id)
    if last_age is not None and last_age > 300:
        score += 0.15
        reasons.append(f"lull ({last_age:.0f}s)")

    # Recent mention of Aura in last 5 messages, she hasn't responded yet
    recent_5 = context_buffer.get_recent(chat_id, 5)
    aura_mentioned = any(
        any(re.search(p, m.text.translate(_UNICODE_NORMALIZE).lower()) for p in MENTION_PATTERNS)
        for m in recent_5 if not m.is_bot
    )
    aura_responded = any(m.is_bot for m in recent_5)
    if aura_mentioned and not aura_responded:
        score += 0.30
        reasons.append("recent mention unanswered")

    # --- Cooldown signals (skipped during conversational turns) ---

    if not is_conversational_turn:
        msgs_since = context_buffer.messages_since_last_bot(chat_id)
        last_bot_age = context_buffer.last_bot_message_age(chat_id)

        # Skip cooldown penalties for groups where Aura barely has a foothold
        _total_resp = reputation_tracker.get_total_responses(chat_id)
        _skip_cooldown = _total_resp < 15

        # Message cooldown — scales with temperature
        # High temp = shorter cooldown (people want her), low temp = longer
        effective_cooldown_msgs = max(2, int(8 * (1.0 - temperature)))
        if not _skip_cooldown and msgs_since < effective_cooldown_msgs:
            penalty = -0.3 * (1.0 - temperature)
            score += penalty
            reasons.append(f"msg cooldown ({msgs_since}/{effective_cooldown_msgs})")

        # Time cooldown — also scales with temperature
        effective_cooldown_secs = max(15, int(90 * (1.0 - temperature)))
        if not _skip_cooldown and last_bot_age is not None and last_bot_age < effective_cooldown_secs:
            penalty = -0.3 * (1.0 - temperature)
            score += penalty
            reasons.append(f"time cooldown ({last_bot_age:.0f}s/{effective_cooldown_secs}s)")

    # Rapid-fire absolute check
    bot_in_10 = context_buffer.bot_messages_in_last_n(chat_id, 10)
    if bot_in_10 >= 3:
        score -= 0.5
        reasons.append(f"rapid-fire ({bot_in_10} in last 10)")

    # Callback boost — having a relevant past exchange makes Aura more likely to speak
    if has_callback:
        score += CALLBACK_SCORE_BOOST
        reasons.append("callback available")

    # Expansion target boost — engage more with users in the cultivation pipeline
    if network_expansion.is_active_target(message.user_id):
        score += 0.25
        reasons.append("expansion target")

    # --- Apply temperature as threshold modifier ---

    # Temperature directly controls how much score is needed.
    # High temp (0.8) → threshold ~0.20 (easy to trigger)
    # Default temp (0.5) → threshold ~0.40
    # Low temp (0.1) → threshold ~0.70 (hard to trigger)
    threshold = max(0.15, 0.50 - (temperature - 0.5) * 0.6)

    # Reputation warmth bonus
    multiplier = reputation_tracker.get_activity_multiplier(chat_id)
    threshold = threshold / max(multiplier, 0.3)
    threshold = min(threshold, 0.90)

    # Warming groups with low response count — be aggressive
    # Nothing to lose, treat it like a job interview
    warmth = reputation_tracker.get_warmth_level(chat_id)
    total_responses = reputation_tracker.get_total_responses(chat_id)
    if warmth in ("new", "warming") and total_responses < 15:
        threshold = 0.10  # very low bar — she needs to establish herself
        score = max(score, 0.15)  # floor — always have a fighting chance
        reasons.append("low-response aggressive")

    # Onboarding dampening — gradual phase raises threshold
    dampening = get_warmth_dampening(chat_id)
    if dampening > 0:
        threshold = min(threshold + dampening, 0.90)
        reasons.append(f"onboarding dampening +{dampening:.2f}")

    score = max(0.0, min(1.0, score))
    should = score >= threshold

    # Final onboarding check for minimal phase
    override = should_override_decision(chat_id, score, False)
    if override is False:
        should = False
        reasons.append("onboarding override")

    reason_str = ", ".join(reasons) if reasons else "no signals"
    log.info(
        "Decision %d: score=%.2f temp=%.2f thresh=%.2f (%s) → %s",
        chat_id, score, temperature, threshold,
        reason_str, "RESPOND" if should else "SILENT",
    )

    return Decision(should_respond=should, score=score, reason=reason_str)


def record_response(chat_id: int) -> None:
    """Record that Aura responded in this group (for rate limiting)."""
    if chat_id not in _hourly_responses:
        _hourly_responses[chat_id] = []
    _hourly_responses[chat_id].append(time.time())


def get_temperature(chat_id: int) -> float:
    """Return the current engagement temperature for a group."""
    return _get_temp(chat_id)["temperature"]
