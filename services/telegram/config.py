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
LLM_MAX_TOKENS = int(os.environ.get("TELEGRAM_LLM_MAX_TOKENS", "800"))
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
RESPOND_THRESHOLD = 0.4           # minimum score to respond in groups

# Factor weights
W_DIRECT_MENTION = 0.9
W_UNANSWERED_QUESTION = 0.3
W_TOPIC_EXPERTISE = 0.2
W_CONVERSATION_LULL = 0.15
W_EMOTIONAL_CONTENT = 0.15
W_RECENT_MENTION = 0.2
W_COOLDOWN_PENALTY = -0.4
W_RAPID_FIRE_PENALTY = -0.6

# Timing
UNANSWERED_QUESTION_DELAY_S = 30  # wait before answering orphan questions
CONVERSATION_LULL_S = 300         # 5 min silence = lull
COOLDOWN_MESSAGES = 8             # min messages between Aura responses in group
COOLDOWN_SECONDS = 60             # min seconds between Aura responses in group

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Groups
GROUP_MIN_MSG_GAP = 8             # min messages between responses
GROUP_MIN_TIME_GAP = 60           # min seconds between responses
GROUP_MAX_PER_HOUR = 6            # max responses per hour per group

# DMs
DM_MIN_TIME_GAP = 3.0            # min seconds between DM responses (typing feel)
DM_MAX_PER_MINUTE = 10

# Global
GLOBAL_MAX_PER_MINUTE = 20

# ---------------------------------------------------------------------------
# Profile refresh
# ---------------------------------------------------------------------------
PROFILE_REFRESH_INTERVAL_S = 86400  # 24 hours
PROFILE_REFRESH_MIN_MESSAGES = 5    # need at least this many msgs to build profile

# ---------------------------------------------------------------------------
# Social growth
# ---------------------------------------------------------------------------
REPUTATION_FILE = DATA_DIR / "reputation.json"
SOCIAL_GRAPH_FILE = DATA_DIR / "social_graph.json"
GROWTH_LOG_FILE = DATA_DIR / "growth_log.json"

MAX_ACTIVE_GROUPS = 20
JOIN_COOLDOWN_S = 86400
NEW_GROUP_QUIET_PERIOD_S = 172800
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
PROACTIVE_DM_COOLDOWN_PER_USER_S = 172800   # 48 hours between proactive DMs to same user
PROACTIVE_DM_MAX_PER_DAY = 3                # max proactive DMs per day total
PROACTIVE_DM_FOLLOWUP_DELAY_MIN_S = 7200    # 2 hours min after group exchange
PROACTIVE_DM_FOLLOWUP_DELAY_MAX_S = 28800   # 8 hours max after group exchange

# Content engine
GROUP_LULL_THRESHOLD_S = 14400               # 4 hours of silence = lull
GROUP_PROACTIVE_COOLDOWN_S = 28800           # 8 hours between proactive group messages

# Onboarding phases
ONBOARDING_SILENT_PHASE_S = 7200             # 2 hours: only direct mentions
ONBOARDING_MINIMAL_PHASE_S = 86400           # 24 hours: only score > 0.8
ONBOARDING_GRADUAL_PHASE_S = 259200          # 72 hours: standard with dampening

# Callbacks & inside jokes
CALLBACK_MIN_AGE_S = 86400                   # 24 hours before referencing past exchange
CALLBACK_SIMILARITY_THRESHOLD = 0.6          # semantic similarity threshold
CALLBACK_SCORE_BOOST = 0.20                  # boost to decision score when callback available

# Socialite orchestrator
SOCIALITE_LOOP_INTERVAL_S = 300              # 5 minutes between orchestrator ticks
SOCIALITE_MAX_ACTIONS_PER_HOUR = 3           # global proactive action rate limit
SOCIALITE_ACTION_EXPIRY_S = 3600             # actions expire after 1 hour

# DM eligibility tracking
DM_ELIGIBLE_FILE = DATA_DIR / "dm_eligible.json"
SOCIALITE_STATE_FILE = DATA_DIR / "socialite_state.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
CALLBACKS_FILE = DATA_DIR / "callbacks.json"
