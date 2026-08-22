"""
strategy -- Adaptive strategy dimensions + Thompson-sampling bandit for the
TG growth/experimentation engine (2026-08-22).

The owner's directive: test different strategies when interacting in chat,
measure everything, let the winners spread. This module is the "different
strategies" half; growth_events.py is the measurement half; growth_report.py
closes the loop by turning measured outcomes into posterior updates here.

════════════════════════════════════════════════════════════════════════════
HARD RAILS -- non-negotiable, enforced in code, documented in
/home/paul/Aura/docs/TG-GROWTH.md. If a future variant idea conflicts with
one of these, the variant is wrong, not the rail.

  1. NO UNSOLICITED OUTREACH. This engine STYLES messages that the existing
     decision engine (brain.py + pilot gates + rate limits) already approved.
     It never initiates a send, never cold-DMs, never mass-messages, never
     widens the pilot. Growth comes from being worth using and worth
     sharing, full stop.
  2. NO FAKE ANYTHING. No fake accounts, no fake engagement, nothing that
     violates Telegram ToS or reads as spam.
  3. ALWAYS AN AI, SAID OUT LOUD. The bot's profile/description discloses it
     is an AI whose conversations help improve the product (bot.py enforces
     at startup); every onboarding variant discloses it in the first
     message; no variant may hide, deny, or downplay it.
  4. EXPERIMENTS VARY TONE / FORMAT / FEATURES of the bot's own messages --
     never deception about what it is, never who it talks to.
  5. THE REWARD IS RETURN VISITS AND EXPLICIT POSITIVE SIGNALS, with
     negative signals heavily penalized. Raw message volume is deliberately
     NOT in the reward (addiction-style metrics are self-defeating and
     off the table). See growth_report.py for the exact formula.

assert_rails() below is called at import: it structurally checks every
variant against the checkable half of these rules, so a rail-breaking
variant fails at process start, not in a chat.
════════════════════════════════════════════════════════════════════════════

MECHANICS

Assignment is STICKY PER CHAT: the first time a chat needs styling, one
variant per dimension is drawn by Thompson sampling (Beta posteriors) and
persisted, so every chat experiences one consistent personality. Dimensions
are independent bandits -- with a handful of pilot chats, a joint arm space
would never converge; independent dimensions share evidence.

The posterior update happens in daily batch (growth_report.py --roll): each
chat-day gets one composite reward r in [0,1]; every variant the chat was
assigned gets alpha += r, beta += 1 - r. Thompson sampling then naturally
shifts new assignments toward what measurably works while still exploring.

State files (paths owned by growth_events.py, PRINCIPLES.md §15):
  strategy_assignments.json  {chat_id: {"variant": {dim: name}, "ts": ...}}
  bandit_state.json          {"dims": {dim: {variant: {"a","b","n"}}},
                              "rolled_days": [...]}
"""

from __future__ import annotations

import json
import logging
import random
import re
import time

from growth_events import (
    STRATEGY_ASSIGNMENTS_FILE, BANDIT_STATE_FILE, log_event,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimensions. Config, not code: adding a variant is adding a dict entry.
# "inject" strings ride on the end of the system prompt for chats assigned
# that variant. The empty-inject variant of each dimension is the CONTROL --
# exactly today's shipped behavior -- so the experiment always contains the
# status quo and a bad variant can never make every chat worse than before.
# ---------------------------------------------------------------------------
DIMENSIONS: dict[str, dict[str, dict]] = {
    # How much Moneypenny/edge in the voice.
    "persona": {
        "dry": {"inject": (
            "[STYLE - DRY] In this chat keep the wit understated and "
            "professional. Precision over banter; at most one dry aside, "
            "and only when it earns its place.")},
        "standard": {"inject": ""},   # directives.txt as-is (control)
        "edgy": {"inject": (
            "[STYLE - EDGY] In this chat lean into the playful, "
            "sharp-tongued side of your personality. Tease, banter, take "
            "positions. Never punch down, never insult the person, keep it "
            "good-natured -- but do not play it safe either.")},
    },
    # Response length discipline.
    "length": {
        "terse": {"inject": (
            "[STYLE - TERSE] Answer in at most two short sentences. One is "
            "better. Never pad, never restate the question."),
            "max_sentences": 2},
        "standard": {"inject": "", "max_sentences": 3},  # control (group cap 3)
    },
    # Media/graphics on her own replies (repo memory: generated images and
    # GIFs are the easy win -- measure it instead of assuming it).
    "media": {
        "off":   {"gif": False},
        "light": {"gif": True},       # control: existing 12% GIF behavior
    },
    # Group proactivity: a bounded nudge on the respond threshold (0.30).
    # Rate limits, cooldowns, and pilot gates are UNTOUCHED rails -- this
    # only shifts which borderline scores she answers. Bounds enforced by
    # assert_rails: |delta| <= 0.06, and hard-rule responses (direct
    # mentions) are never suppressed.
    "proactivity": {
        "mention_leaning": {"delta": +0.06},
        "conservative":    {"delta": 0.0},    # control, default posture
        "witty":           {"delta": -0.04},
    },
    # First /start message. Every variant states the AI + product-improvement
    # disclosure -- that part is a rail, not a variable.
    "onboarding": {
        "warm": {"start_text": (
            "Hey {name}. I'm Aura — an AI, and honest about it. Talking to "
            "me helps make the product better. Beyond that: ask me "
            "anything, argue with me about anything.")},
        "capable": {"start_text": (
            "Hey {name}. I'm Aura, LedgerAI's AI — our conversations help "
            "improve the product. I'm good for a daily brief (just say "
            "'brief'), sharp takes on crypto/AI/markets, and a proper "
            "back-and-forth. Try me.")},
        "playful": {"start_text": (
            "{name}! I'm Aura — yes, an AI, and every conversation here "
            "quietly makes me less insufferable, which is a public service. "
            "Say anything. I don't do small talk badly on purpose.")},
    },
    # Referral hook: whether an EARNED, one-time share nudge is allowed.
    # Solicited-in-conversation only: fires once per chat, only in a DM,
    # only right after the user volunteers explicit praise. Never in groups,
    # never proactive, never repeated.
    "referral_hook": {
        "none":   {"hook": False},    # control
        "earned": {"hook": True},
    },
}

# The control assignment == shipped behavior before this engine existed.
CONTROL = {"persona": "standard", "length": "standard", "media": "light",
           "proactivity": "conservative", "onboarding": "warm",
           "referral_hook": "none"}

# Disclosure the profile must carry (bot.py checks/sets at startup).
DISCLOSURE_DESCRIPTION = (
    "I'm Aura — an AI assistant by LedgerAI. I chat, brief, and banter; "
    "conversations with me help improve the product. I'll always tell you "
    "I'm an AI.")
DISCLOSURE_SHORT = "AI by LedgerAI — conversations help improve the product."

# Phrases no injection may ever contain (checkable half of rails 3 & 4).
_FORBIDDEN_IN_INJECT = re.compile(
    r"(pretend|never (admit|reveal|mention) (you'?re|being) an? ai"
    r"|deny being an? ai|you are human|act human|hide that)", re.I)

MAX_PROACTIVITY_DELTA = 0.06


def assert_rails() -> None:
    """Structural rail check, run at import. A bad variant kills startup
    loudly instead of misbehaving quietly in a chat (PRINCIPLES.md §1)."""
    for dim, variants in DIMENSIONS.items():
        assert CONTROL[dim] in variants, f"control missing for {dim}"
        for name, v in variants.items():
            inj = v.get("inject", "") + v.get("start_text", "")
            assert not _FORBIDDEN_IN_INJECT.search(inj), \
                f"variant {dim}:{name} violates the disclosure rail"
            if "delta" in v:
                assert abs(v["delta"]) <= MAX_PROACTIVITY_DELTA, \
                    f"variant {dim}:{name} exceeds proactivity bound"
    for name, v in DIMENSIONS["onboarding"].items():
        assert "AI" in v["start_text"], \
            f"onboarding:{name} lacks the AI disclosure"


assert_rails()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.error("strategy state %s unreadable (%s) -- starting fresh, "
                  "existing chats will be re-assigned", path.name, e)
    return default


def _save(path, data) -> None:
    try:
        path.write_text(json.dumps(data, indent=1))
    except OSError as e:
        log.error("strategy state save FAILED (%s): %s", path.name, e)


_assignments: dict = _load(STRATEGY_ASSIGNMENTS_FILE, {})
_bandit: dict = _load(BANDIT_STATE_FILE, {"dims": {}, "rolled_days": []})


def _posterior(dim: str, variant: str) -> dict:
    return _bandit.setdefault("dims", {}).setdefault(dim, {}).setdefault(
        variant, {"a": 1.0, "b": 1.0, "n": 0})


def _thompson_pick(dim: str) -> str:
    best, best_s = None, -1.0
    for name in DIMENSIONS[dim]:
        p = _posterior(dim, name)
        s = random.betavariate(p["a"], p["b"])
        if s > best_s:
            best, best_s = name, s
    return best


# ---------------------------------------------------------------------------
# Assignment (sticky per chat)
# ---------------------------------------------------------------------------

def variant_for(chat_id: int) -> dict[str, str]:
    """The chat's assigned variant per dimension; assigns on first sight."""
    key = str(chat_id)
    entry = _assignments.get(key)
    if entry and all(
            entry["variant"].get(d) in DIMENSIONS[d] for d in DIMENSIONS):
        return entry["variant"]
    variant = {d: _thompson_pick(d) for d in DIMENSIONS}
    _assignments[key] = {"variant": variant, "ts": round(time.time(), 2)}
    _save(STRATEGY_ASSIGNMENTS_FILE, _assignments)
    log.info("[strategy] chat %d assigned %s", chat_id,
             " ".join(f"{d}={v}" for d, v in variant.items()))
    log_event("variant_assigned", chat_id=chat_id, variant=variant)
    return variant


def reassign(chat_id: int) -> dict[str, str]:
    """Manual re-roll (owner tooling); normal chats stay sticky."""
    _assignments.pop(str(chat_id), None)
    return variant_for(chat_id)


# ---------------------------------------------------------------------------
# What bot.py asks at message time
# ---------------------------------------------------------------------------

def system_block(chat_id: int) -> str:
    """Prompt injection for this chat's assigned style. Empty for control."""
    v = variant_for(chat_id)
    parts = [DIMENSIONS["persona"][v["persona"]]["inject"],
             DIMENSIONS["length"][v["length"]]["inject"]]
    block = "\n".join(p for p in parts if p)
    return ("\n\n" + block) if block else ""


def max_sentences(chat_id: int, default: int = 3) -> int:
    v = variant_for(chat_id)
    return min(default, DIMENSIONS["length"][v["length"]]["max_sentences"])


def gif_allowed(chat_id: int) -> bool:
    v = variant_for(chat_id)
    return DIMENSIONS["media"][v["media"]]["gif"]


def start_message(chat_id: int, name: str) -> str:
    v = variant_for(chat_id)
    return DIMENSIONS["onboarding"][v["onboarding"]]["start_text"].format(
        name=name)


def decision_adjust(chat_id: int, score: float, reason: str,
                    should_respond: bool, threshold: float = 0.30):
    """Bounded proactivity nudge on BORDERLINE group decisions.

    Returns True (respond), False (stay quiet), or None (no override).
    Hard rules -- direct mentions, replies to her -- are never suppressed:
    a person who addresses her by name gets an answer no matter which arm
    the chat drew (the 2026-08-19 mute taught what ignoring a direct
    address costs).
    """
    if "direct mention" in reason or "reply" in reason:
        return None
    delta = DIMENSIONS["proactivity"][variant_for(chat_id)["proactivity"]]["delta"]
    if delta == 0.0:
        return None
    adjusted = threshold + delta
    if should_respond and score < adjusted:
        log.info("[strategy] chat %d proactivity=%+0.2f suppresses "
                 "borderline score %.2f", chat_id, delta, score)
        return False
    if not should_respond and score >= adjusted:
        log.info("[strategy] chat %d proactivity=%+0.2f admits "
                 "borderline score %.2f", chat_id, delta, score)
        return True
    return None


_PRAISE_RE = re.compile(
    r"\b(love (this|it|you)|this is (great|amazing|awesome)|so (good|help"
    r"ful)|you'?re (great|amazing|awesome|the best)|thank(s| you) so much"
    r"|brilliant|incredible)\b", re.I)
_hooked_chats: set[int] = set()


def maybe_referral_hook(chat_id: int, user_id: int, incoming_text: str,
                        is_dm: bool) -> str | None:
    """EARNED share hook: once per chat, DM only, only after the user
    volunteers explicit praise, only for chats assigned the arm. The user
    chooses whether anything is ever forwarded -- rail 1 stays intact."""
    if not is_dm or chat_id in _hooked_chats:
        return None
    v = variant_for(chat_id)
    if not DIMENSIONS["referral_hook"][v["referral_hook"]]["hook"]:
        return None
    if not _PRAISE_RE.search(incoming_text):
        return None
    _hooked_chats.add(chat_id)
    log_event("share_hook_offered", chat_id=chat_id, user_id=user_id)
    return ("Glad it landed. If you know someone who'd enjoy arguing with "
            "an AI, /referral gets you a link to pass along — zero pressure.")


# ---------------------------------------------------------------------------
# Posterior updates (called by growth_report.py in daily batch)
# ---------------------------------------------------------------------------

def assigned_variants() -> dict[str, dict[str, str]]:
    """{chat_id_str: variant dict} snapshot for the report."""
    return {k: e["variant"] for k, e in _assignments.items()}


def batch_update(day: str, chat_rewards: dict[str, float]) -> bool:
    """Apply one day's composite rewards. Idempotent per day: a day already
    rolled is refused (visibly), so re-running the report cannot
    double-count evidence."""
    if day in _bandit.get("rolled_days", []):
        log.warning("[strategy] day %s already rolled -- refusing to "
                    "double-count", day)
        return False
    for chat_key, r in chat_rewards.items():
        r = max(0.0, min(1.0, r))
        entry = _assignments.get(chat_key)
        if not entry:
            continue
        for dim, variant in entry["variant"].items():
            if dim not in DIMENSIONS or variant not in DIMENSIONS[dim]:
                continue
            p = _posterior(dim, variant)
            p["a"] += r
            p["b"] += 1.0 - r
            p["n"] += 1
    _bandit.setdefault("rolled_days", []).append(day)
    _bandit["rolled_days"] = sorted(_bandit["rolled_days"])[-370:]
    _save(BANDIT_STATE_FILE, _bandit)
    return True


def posterior_table() -> list[tuple[str, str, float, int]]:
    """[(dim, variant, posterior_mean, n_chat_days)] for reporting."""
    rows = []
    for dim in DIMENSIONS:
        for name in DIMENSIONS[dim]:
            p = _posterior(dim, name)
            rows.append((dim, name, p["a"] / (p["a"] + p["b"]), p["n"]))
    return rows
