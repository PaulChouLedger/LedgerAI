"""
persona -- System prompts for DM, group, and profile-building modes.

Loads directives.txt at import time and injects it into every prompt.
Edit directives.txt to tune Aura's personality without touching code.
"""

from __future__ import annotations

import logging
from config import DIRECTIVES_FILE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load directives (personality baseline)
# ---------------------------------------------------------------------------

def _load_directives() -> str:
    try:
        text = DIRECTIVES_FILE.read_text().strip()
        log.info("Loaded directives from %s (%d chars)", DIRECTIVES_FILE, len(text))
        return text
    except FileNotFoundError:
        log.warning("No directives.txt found at %s — using bare prompts", DIRECTIVES_FILE)
        return ""

DIRECTIVES = _load_directives()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DM_SYSTEM = """\
{directives}

You are in a private 1:1 DM with {{name}} on Telegram.

{{profile_context}}\
""".format(directives=DIRECTIVES)

GROUP_SYSTEM = """\
{directives}

You are in a group chat on Telegram. You are one voice among many.

{{profile_context}}

Recent conversation:
{{conversation_context}}\
""".format(directives=DIRECTIVES)

PROFILE_BUILDER_SYSTEM = """\
You are analyzing conversations to build a profile of {name}. \
Based on the transcript below, produce a JSON object with these fields:

{{
    "topics": ["list of topics they discuss or care about"],
    "personality": "2-3 sentence description of their communication style and personality",
    "relationship": "1-2 sentence summary of their relationship with Aura (the AI)"
}}

Be specific and observational. Base everything on evidence from the conversations. \
Respond with ONLY the JSON object, no other text.\
"""
