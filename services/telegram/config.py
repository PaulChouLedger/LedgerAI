"""
config -- Constants, thresholds, and environment variables for Aura Telegram bot.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = WORKSPACE_ROOT / "data" / "telegram"
PROFILES_FILE = DATA_DIR / "profiles.json"
GROUP_STATE_FILE = DATA_DIR / "group_state.json"
BOT_STATE_FILE = DATA_DIR / "bot_state.json"
DIRECTIVES_FILE = Path(__file__).resolve().parent / "directives.txt"

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Bot's own username (set after first getMe call)
BOT_USERNAME: str = ""

# ---------------------------------------------------------------------------
# Farsight LLM (RTX PRO 6000, 72B Qwen)
# ---------------------------------------------------------------------------
FARSIGHT_URL = os.environ.get(
    "AURA_FARSIGHT_URL", "http://100.76.191.92:11435"
)
LLM_ENDPOINT = f"{FARSIGHT_URL}/perpetual/chat"
LLM_MAX_TOKENS = int(os.environ.get("TELEGRAM_LLM_MAX_TOKENS", "250"))
LLM_TIMEOUT = int(os.environ.get("TELEGRAM_LLM_TIMEOUT", "90"))

# ---------------------------------------------------------------------------
# Memory container (on puck or local)
# ---------------------------------------------------------------------------
MEMORY_URL = os.environ.get("AURA_MEMORY_URL", "http://localhost:11438")

# ---------------------------------------------------------------------------
# Context buffer
# ---------------------------------------------------------------------------
CONTEXT_BUFFER_SIZE = 50          # messages to keep per chat
CONTEXT_WINDOW_FOR_PROMPT = 20    # messages to inject into LLM prompt

# ---------------------------------------------------------------------------
# Decision engine thresholds (brain.py)
# ---------------------------------------------------------------------------
RESPOND_THRESHOLD = 0.30          # minimum score to respond in groups

# Factor weights
W_DIRECT_MENTION = 0.9
W_UNANSWERED_QUESTION = 0.3
W_TOPIC_EXPERTISE = 0.2
W_CONVERSATION_LULL = 0.15
W_EMOTIONAL_CONTENT = 0.15
W_RECENT_MENTION = 0.2
W_COOLDOWN_PENALTY = -0.25
W_RAPID_FIRE_PENALTY = -0.35

# Timing
UNANSWERED_QUESTION_DELAY_S = 30  # wait before answering orphan questions
CONVERSATION_LULL_S = 300         # 5 min silence = lull
COOLDOWN_MESSAGES = 6             # min messages between Aura responses in group
COOLDOWN_SECONDS = 45             # min seconds between Aura responses in group

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Groups
GROUP_MIN_MSG_GAP = 4             # min messages between responses
GROUP_MIN_TIME_GAP = 20           # min seconds between responses
GROUP_MAX_PER_HOUR = 14           # max responses per hour per group

# DMs
DM_MIN_TIME_GAP = 3.0            # min seconds between DM responses (typing feel)
DM_MAX_PER_MINUTE = 10

# Global
GLOBAL_MAX_PER_MINUTE = 20

# ---------------------------------------------------------------------------
# Profile refresh
# ---------------------------------------------------------------------------
PROFILE_REFRESH_INTERVAL_S = 300  # 5 minutes
PROFILE_REFRESH_MIN_MESSAGES = 5    # need at least this many msgs to build profile

# ---------------------------------------------------------------------------
# Social growth
# ---------------------------------------------------------------------------
REPUTATION_FILE = DATA_DIR / "reputation.json"
SOCIAL_GRAPH_FILE = DATA_DIR / "social_graph.json"
GROWTH_LOG_FILE = DATA_DIR / "growth_log.json"

MAX_ACTIVE_GROUPS = 20
JOIN_COOLDOWN_S = 86400
NEW_GROUP_QUIET_PERIOD_S = 7200
REPUTATION_DECAY_INTERVAL_S = 604800
IGNORE_THRESHOLD_MESSAGES = 5

WARMTH_MULTIPLIERS = {
    "new": 0.3,
    "warming": 0.6,
    "established": 1.0,
    "trusted": 1.2,
}

# ---------------------------------------------------------------------------
# Socialite system
# ---------------------------------------------------------------------------

# DM strategy
PROACTIVE_DM_COOLDOWN_PER_USER_S = 57600    # 16 hours between proactive DMs to same user
PROACTIVE_DM_MAX_PER_DAY = 10               # max proactive DMs per day total
PROACTIVE_DM_FOLLOWUP_DELAY_MIN_S = 1800    # 30 min min after group exchange
PROACTIVE_DM_FOLLOWUP_DELAY_MAX_S = 7200    # 2 hours max after group exchange

# Content engine — measured interjections
GROUP_LULL_THRESHOLD_S = 2400                # 40 min of silence = lull
GROUP_PROACTIVE_COOLDOWN_S = 7200            # 2 hours between proactive group messages

# Onboarding phases — aggressive: get in the conversation fast
ONBOARDING_SILENT_PHASE_S = 900              # 15 min: only direct mentions
ONBOARDING_MINIMAL_PHASE_S = 3600            # 1 hour: only score > 0.8
ONBOARDING_GRADUAL_PHASE_S = 21600           # 6 hours: standard with dampening

# Callbacks & inside jokes
CALLBACK_MIN_AGE_S = 86400                   # 24 hours before referencing past exchange
CALLBACK_SIMILARITY_THRESHOLD = 0.6          # semantic similarity threshold
CALLBACK_SCORE_BOOST = 0.20                  # boost to decision score when callback available

# Socialite orchestrator — lean in harder
SOCIALITE_LOOP_INTERVAL_S = 180              # 3 minutes between orchestrator ticks
SOCIALITE_MAX_ACTIONS_PER_HOUR = 10          # global proactive action rate limit
SOCIALITE_ACTION_EXPIRY_S = 3600             # actions expire after 1 hour

# DM nudge (group-to-DM encouragement) — aggressive conversion
DM_NUDGE_COOLDOWN_PER_GROUP_S = 5400     # 90 min between nudges in same group
DM_NUDGE_PROBABILITY = 0.70             # 70% chance when all conditions met

# Group profiles
GROUP_PROFILES_FILE = DATA_DIR / "group_profiles.json"
GROUP_PROFILE_REFRESH_INTERVAL_S = 14400  # 4 hours between profile rebuilds
GROUP_PROFILE_MIN_MESSAGES = 10           # need at least this many observed msgs

# Daily brief (posted to main channel)
DAILY_BRIEF_HOUR_UTC = int(os.environ.get("DAILY_BRIEF_HOUR_UTC", "13"))  # 13 UTC = 9am ET / 8am CT
DAILY_BRIEF_STATE_FILE = DATA_DIR / "daily_brief_state.json"
DAILY_BRIEF_CHAT_ID = -1003025733750  # Area31

# DM eligibility tracking
DM_ELIGIBLE_FILE = DATA_DIR / "dm_eligible.json"
SOCIALITE_STATE_FILE = DATA_DIR / "socialite_state.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
CALLBACKS_FILE = DATA_DIR / "callbacks.json"

# ---------------------------------------------------------------------------
# Network expansion (strategic group acquisition)
# ---------------------------------------------------------------------------
EXPANSION_TARGETS_FILE = DATA_DIR / "expansion_targets.json"
EXPANSION_MAX_ACTIVE_TARGETS = 20           # max concurrent cultivation targets
EXPANSION_MIN_RELATIONSHIP_DEPTH = "stranger"  # allow cultivation from first contact
EXPANSION_INTEL_DWELL_S = 14400             # 4 hours in intel before advancing
EXPANSION_WARM_DWELL_S = 43200              # 12 hours warming up
EXPANSION_VALUE_DEMO_DWELL_S = 86400        # 24 hours demonstrating value
EXPANSION_SEED_DWELL_S = 43200              # 12 hours seeding before nurture
EXPANSION_CULTIVATION_COOLDOWN_S = 14400    # 4 hours between cultivation actions per target
EXPANSION_SEED_PROBABILITY = 0.65           # 65% chance to inject seed prompt when eligible
EXPANSION_SCORE_BOOST = 0.20               # decision score boost for warm/value_demo targets

# Advocate direct ask (most aggressive — just ask them)
ADVOCATE_ASK_COOLDOWN_S = 172800            # 2 days between asks to same advocate
ADVOCATE_ASK_MIN_INTERACTIONS = 15          # min total interactions before asking
ADVOCATE_ASK_MIN_DMS = 2                    # must have DM'd us at least 2 times

# Viral/shareable content
SHAREABLE_INJECTION_PROBABILITY = 0.30      # 30% of group responses get "make it shareable" prompt
CROSS_POLLINATE_PROBABILITY = 0.25          # 25% chance to reference other group discussions

# ---------------------------------------------------------------------------
# Token awareness (organic $LEDGER integration)
# ---------------------------------------------------------------------------
TOKEN_INJECTION_PROBABILITY = 0.12       # 12% chance when topic matches
TOKEN_OPINION_PROBABILITY = 0.15         # 15% chance on crypto/AI discussions
TOKEN_MIN_WARMTH = "established"         # never mention token in new/warming groups
TOKEN_MENTION_COOLDOWN_S = 86400         # max 1 token mention per group per 24h
TOKEN_DM_MIN_DEPTH = "familiar"          # min relationship depth for token DM content
TOKEN_MENTION_COOLDOWNS_FILE = DATA_DIR / "token_cooldowns.json"

# Token tiers (mapped from warmth + engagement)
TOKEN_TIERS = {
    "observer":    {"min_warmth": "new",         "min_interactions": 0},
    "participant": {"min_warmth": "warming",     "min_interactions": 5},
    "insider":     {"min_warmth": "established", "min_interactions": 15},
    "core":        {"min_warmth": "trusted",     "min_interactions": 50},
}

# Referral rewards
REFERRAL_TIERS = {
    "connector":  3,    # 3+ referrals
    "ambassador": 10,   # 10+ referrals
    "founder":    25,   # 25+ referrals
}

# Anti-shill patterns (hard ban — strip from any response)
SHILL_PATTERNS = [
    r'\b(?:NFA|DYOR|not financial advice)\b',
    r'\bto the moon\b',
    r'\b\d+x\b',                          # "100x", "10x"
    r'\byou should (?:buy|invest|get)\b',
    r'\b(?:gem|moonshot|diamond hands?)\b',
    r'\b(?:going to explode|pump|ape in)\b',
    r'\bearly (?:bird|adopter)s? get\b',
]

# ---------------------------------------------------------------------------
# Self-correction feedback engine
# ---------------------------------------------------------------------------
FEEDBACK_QUEUE_FILE = DATA_DIR / "feedback_queue.json"
FEEDBACK_AUDIT_FILE = DATA_DIR / "feedback_audit.json"
LEARNED_DIRECTIVES_FILE = DATA_DIR / "learned_directives.json"

# Feedback channel response behavior
FEEDBACK_CHANNEL_RESPONSE_COOLDOWN_S = 600  # 10 min between responses in feedback channel
FEEDBACK_CHANNEL_RESPONSE_PROBABILITY = 0.6  # 60% chance to respond when cooldown clear


# ---------------------------------------------------------------------------
# Pilot mode -- 2026-07-31 relaunch (Area31 only)
# ---------------------------------------------------------------------------
# The bot was mute for ~3 months (dead LLM) while every listening/ingestion
# loop kept running. Relaunch is deliberately staged: she SPEAKS only in the
# allowed chats below (and DMs only the owner), while continuing to listen
# everywhere she is already invited -- gathering, not broadcasting. Widen by
# adding chat ids to AURA_PILOT_CHATS or set AURA_PILOT_MODE=0 to lift all
# gates at once.
PILOT_MODE = os.environ.get("AURA_PILOT_MODE", "1") == "1"
PILOT_ALLOWED_CHATS = {
    int(x) for x in os.environ.get(
        "AURA_PILOT_CHATS", "-1003025733750").split(",") if x.strip()
}
OWNER_USER_ID = 110875514

#: While True, moderation DECIDES but does not act: no deletes, no warns, no
#: mutes, no bans -- decisions are logged for review. The autonomous ban path
#: was live for months with the LLM judge failing open and a stale warn
#: ledger; nobody should be banned by that state until a human has read it.
MODERATION_LOG_ONLY = os.environ.get("AURA_MODERATION_LOG_ONLY", "1") == "1"


def chat_allowed(chat_id: int) -> bool:
    """May the bot SEND to this chat right now? (Listening is unconditional.)"""
    return (not PILOT_MODE) or chat_id in PILOT_ALLOWED_CHATS
