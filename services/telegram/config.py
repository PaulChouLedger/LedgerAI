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
# Tenor GIF API
# ---------------------------------------------------------------------------
TENOR_API_KEY = os.environ.get("TENOR_API_KEY", "")

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
