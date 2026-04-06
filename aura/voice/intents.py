"""
voice.intents -- Local intent detection for voice commands.

Intercepts transcript text BEFORE it reaches the LLM.  Returns an
intent tag if matched, or None to let the text pass through normally.

Currently supported intents:
    "shutdown"  — full system power-off (10s countdown, tap to abort)
    "sleep"     — low-power sleep (screen off, mic listens for wake)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Shutdown patterns — full power off
# ---------------------------------------------------------------------------

_SHUTDOWN_PATTERNS = [
    # Require "aura" nearby OR a complete directive phrase to avoid false positives
    # from ambient speech (e.g. "shut the door", "switch off the light").
    r"\baura\b.*\bshut\s*(?:down|off)\b",
    r"\bshut\s*(?:down|off)\b.*\baura\b",
    r"\baura\b.*\bpower\s*(?:down|off)\b",
    r"\bpower\s*(?:down|off)\b.*\baura\b",
    r"\baura\b.*\bturn\s*(?:yourself\s+)?off\b",
    r"\bturn\s*(?:yourself\s+)?off\b.*\baura\b",
    # "turn yourself off" is unambiguous even without "aura"
    r"\bturn\s+yourself\s+off\b",
    # Polite forms already include context
    r"\bplease\s+shut\s*(?:down|off)\b",
]

_SHUTDOWN_RE = re.compile(
    "|".join(f"(?:{p})" for p in _SHUTDOWN_PATTERNS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Sleep patterns — screen off, mic stays alive
# ---------------------------------------------------------------------------

_SLEEP_PATTERNS = [
    r"\bgo\s+to\s+sleep\b",
    r"\bgood\s*night\s+aura\b",
    r"\bgoodnight\s+aura\b",
    r"\bgood\s*night\b",
    r"\btime\s+to\s+(?:sleep|rest)\b",
    r"\bthat'?s\s+(?:all|enough)\s+for\s+(?:now|today)\b",
    r"\btake\s+a\s+(?:nap|rest|break)\b",
    r"\bsleep\s+(?:mode|now)\b",
    r"\bgo\s+(?:quiet|dark)\b",
]

_SLEEP_RE = re.compile(
    "|".join(f"(?:{p})" for p in _SLEEP_PATTERNS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Quiet patterns — suppress proactive speech for 4 hours
# ---------------------------------------------------------------------------

_QUIET_PATTERNS = [
    r"\bshut\s+up\b",
    r"\bbe\s+quiet\b",
    r"\bstop\s+talking\b",
    r"\bleave\s+me\s+alone\b",
    r"\benough\s+aura\b",
    r"\baura\s+stop\b",
    r"\bstop\s+aura\b",
]

_QUIET_RE = re.compile(
    "|".join(f"(?:{p})" for p in _QUIET_PATTERNS),
    re.IGNORECASE,
)


def detect_intent(text: str) -> str | None:
    """Return an intent tag or None.

    Called on every transcript before LLM routing.
    Must be fast (regex only, no ML).
    Shutdown checked first (more destructive = higher priority).
    """
    if _SHUTDOWN_RE.search(text):
        return "shutdown"
    if _SLEEP_RE.search(text):
        return "sleep"
    if _QUIET_RE.search(text):
        return "quiet"
    return None
