"""
content_engine -- Conversation starters & hot takes for quiet groups.

Keeps groups interesting during quiet periods by dropping topic-relevant
hot takes when a group has been silent for 4+ hours during active hours.

Constraints:
  - Only in groups where warmth >= "warming" and temperature > 0.3
  - Max 1 proactive message per group per 8 hours
  - Uses group's topic_hits data to pick engaging topics
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

import config
from config import (
    GROUP_LULL_THRESHOLD_S,
    GROUP_PROACTIVE_COOLDOWN_S,
    DATA_DIR,
)

log = logging.getLogger(__name__)

# Track when we last sent a proactive message per group
_PROACTIVE_FILE = DATA_DIR / "proactive_cooldowns.json"


def _load_json(path: Path, default) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


#: First-person claims to lived experience, and references to venues or
#: outings as though they were shared knowledge. A prompt rule alone is not
#: enough here: the failure is silent, it self-reinforces through
#: conversation history, and by the time anyone notices, the group has been
#: asked about a place that does not exist three times.
#:
#: Deliberately NOT matching "I think", "I reckon", "I'd argue" and the like
#: — opinions are exactly what we still want. What is banned is having BEEN
#: somewhere or DONE something.
_FABRICATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:i|we)(?:'m|'ve| am| have)?\s+(?:been\s+)?"
     r"(?:tried|trying|visited|went|going)\s+to\b", "claims to have gone somewhere"),
    (r"\b(?:planning|plan|thinking)\s+(?:on\s+|of\s+|to\s+)?"
     r"(?:try|trying|visit|visiting|go|going|check|checking)\b", "claims a plan"),
    # "that new" AND "the new" — the leaked case was "the new ramen
    # place", and the definite article is if anything MORE assertive:
    # it presumes the group already knows which one.
    (r"\b(?:that|the)\s+new\s+\w+\s*(?:place|spot|joint|bar|restaurant|cafe|café|shop)\b",
     "invents a specific venue as shared knowledge"),
    # "thoughts on trying...", "up for checking out..." — a plan
    # proposed without the word "plan" in it.
    (r"\b(?:trying|visiting|checking\s+out|hitting|grabbing)\s+(?:the|that|a)\s+new\b", "proposes an outing"),
    (r"\b(?:i|we)\s+(?:just\s+)?(?:ate|drank|bought|watched|played|attended|"
     r"cooked|ordered|grabbed)\b", "claims a first-person experience"),
    (r"\banyone\s+(?:been|tried|going)\b", "asks about an outing it invented"),
    (r"\b(?:this|next)\s+weekend\b.*\b(?:i|we)\b", "claims a weekend plan"),
    (r"\b(?:i|we)\b.*\b(?:this|next)\s+weekend\b", "claims a weekend plan"),
)


def looks_fabricated(text: str) -> str | None:
    """Why this message invents lived experience, or None if it is clean.

    Returns the REASON rather than a bool so the log says what tripped and
    the next person does not have to re-derive it from a regex.
    """
    low = (text or "").lower()
    for pattern, reason in _FABRICATION_PATTERNS:
        if re.search(pattern, low):
            return reason
    return None


class ContentEngine:
    """Generates conversation starters for quiet groups."""

    def __init__(self) -> None:
        self._cooldowns: dict = _load_json(_PROACTIVE_FILE, {})
        # 2026-08-02 no-repeat ledger (config.LULL_NO_REPEAT):
        # list of {"ts", "chat_id", "theme", "text"}. Persists across
        # restarts — the old dedup used only the in-memory buffer, which is
        # why the same Elon-money-2036 joke landed twice in 12 hours.
        self._ledger: list = _load_json(config.LULL_LEDGER_FILE, [])
        if not isinstance(self._ledger, list):
            self._ledger = []
        # Theme picked in build_starter_prompt, committed on actual send.
        self._pending_theme: dict[int, str] = {}

    def _save(self) -> None:
        _save_json(_PROACTIVE_FILE, self._cooldowns)

    # -- no-repeat ledger -----------------------------------------------

    def _save_ledger(self) -> None:
        # Keep a generous horizon: 4x the no-repeat window, max 400 entries
        cutoff = time.time() - 4 * config.LULL_NO_REPEAT_DAYS * 86400
        self._ledger = [e for e in self._ledger if e.get("ts", 0) > cutoff]
        self._ledger = self._ledger[-400:]
        _save_json(config.LULL_LEDGER_FILE, self._ledger)

    def _recent_ledger(self, chat_id: int, days: float) -> list[dict]:
        cutoff = time.time() - days * 86400
        return [e for e in self._ledger
                if e.get("chat_id") == chat_id and e.get("ts", 0) > cutoff]

    def _pick_theme(self, chat_id: int, pool: list[str]) -> str:
        """Pick a theme not used in this chat within the no-repeat window.

        Never back-to-back similar: the most recently used theme is excluded
        even when the whole pool has been used inside the window — in that
        case the least-recently-used theme wins.
        """
        recent = self._recent_ledger(chat_id, config.LULL_NO_REPEAT_DAYS)
        last_used: dict[str, float] = {}
        for e in recent:
            t = e.get("theme", "")
            last_used[t] = max(last_used.get(t, 0), e.get("ts", 0))

        fresh = [t for t in pool if t not in last_used]
        if fresh:
            return random.choice(fresh)

        # Whole pool used within the window — take the stalest, which by
        # construction is not the one used last (pool has >= 2 entries).
        return min(pool, key=lambda t: last_used.get(t, 0))

    def check_lull(
        self,
        chat_id: int,
        last_message_age: Optional[float],
        warmth_level: str,
        temperature: float,
        top_topics: list[str],
    ) -> Optional[dict]:
        """Check if a group is in a lull that warrants a proactive message.

        Uses stochastic timing so she doesn't jump in at the same interval
        every time — adds 0-50% random jitter to both threshold and cooldown.

        Returns action dict or None.
        """
        import random

        # Even new groups get lull breakers — gotta earn the room
        if temperature < 0.2:
            return None

        # Must be a real lull — treat unknown (no messages since restart) as 12h
        if last_message_age is None:
            last_message_age = 43200  # 12 hours — assume dormant, worth nudging

        # Stochastic threshold: 20-30 min (not exactly 20 every time)
        jittered_threshold = GROUP_LULL_THRESHOLD_S * (1.0 + random.random() * 0.5)
        if last_message_age < jittered_threshold:
            return None

        # Check cooldown with jitter: 1-1.5 hours
        key = str(chat_id)
        last_proactive = self._cooldowns.get(key, 0)
        jittered_cooldown = GROUP_PROACTIVE_COOLDOWN_S * (1.0 + random.random() * 0.5)
        if time.time() - last_proactive < jittered_cooldown:
            return None

        # Coin flip — 40% chance she actually speaks (adds unpredictability)
        if random.random() > 0.4:
            return None

        # Skip truly dead groups (>3 days) but nudge dormant ones (up to 72h)
        if last_message_age > 259200:
            return None  # Dead group (3 days), don't bother

        return {
            "chat_id": chat_id,
            "type": "lull_breaker",
            "topics": top_topics[:3] if top_topics else [],
            "lull_duration_hours": last_message_age / 3600,
        }

    def record_proactive_send(self, chat_id: int, sent_text: str = "") -> None:
        """Record that we sent a proactive message to this group.

        With LULL_NO_REPEAT on, also commits the theme chosen in
        build_starter_prompt (plus the text actually sent) to the
        persistent ledger, so neither survives into the next pick.
        """
        self._cooldowns[str(chat_id)] = time.time()
        self._save()

        if config.LULL_NO_REPEAT:
            theme = self._pending_theme.pop(chat_id, "")
            self._ledger.append({
                "ts": time.time(),
                "chat_id": chat_id,
                "theme": theme,
                "text": sent_text[:300],
            })
            self._save_ledger()
            log.info("Lull ledger: recorded theme %r for %d (%d entries)",
                     theme[:60], chat_id, len(self._ledger))

    def get_last_theme(self, chat_id: int) -> str:
        """Theme picked by the most recent build_starter_prompt (for tagging)."""
        return self._pending_theme.get(chat_id, "")

    # Light/fun starters — real human stuff people actually talk about
    # 2026-08-07: REWRITTEN because these themes were ASKING FOR THE
    # HALLUCINATION. The owner: "the TG aura keeps asking about this
    # fictional ramen place, investigate and stop that type of bad fillers."
    #
    # The culprit was "a food opinion or weekend plan type thought". Aura has
    # no body, no weekend and no way to visit anywhere, so a theme that asks
    # for a weekend plan can only be answered by inventing one. It produced
    # "Planning to try that new ramen spot this weekend. Anyone been there?"
    # — and then the invention became SELF-SUSTAINING: the message entered
    # the group history, came back as conversation_context on the next lull,
    # and got followed up on as though the plan were real. Twice in the
    # ledger, plus once more in engagement_temp as a follow-up question.
    #
    # Half the list had the same defect in weaker form — "something you saw
    # in the news today", "a hot take about a movie that just came out", "a
    # gaming take or new release reaction" all presuppose first-person
    # experience Aura cannot have, and one of them WILL eventually be
    # answered by making something up.
    #
    # So every theme here is now one of two shapes: an OPINION Aura can
    # honestly hold, or a QUESTION put to the group. Neither requires
    # claiming to have been anywhere or done anything.
    _LIGHT_STARTERS = [
        "an opinion about football or a rivalry — an argument, not a result you claim to have watched",
        "a question to the group about something in the news they might have seen",
        "an opinion about a film, show or album — one you can argue about without claiming to have just watched it",
        "a question about how the group feels about something everyday (commuting, group chats, notifications)",
        "a food opinion — a stance about a dish or a way of eating, never a plan to go somewhere",
        "a question starting 'does anyone actually...' about a common habit",
        "an unpopular opinion about something everyday",
        "a question about gaming put to the group rather than a reaction you claim to have had",
    ]

    #: Appended to every starter prompt. Aura is a program in a group chat:
    #: it has no body, no calendar and no venue it can walk into. Inventing
    #: one is not a harmless flourish — it becomes shared context the group
    #: is then expected to play along with, which is how a ramen place that
    #: never existed got asked about three times.
    _NO_FABRICATION = (
        "\n\nHARD RULE — you are a program in a group chat. You have no body, "
        "no weekend, no meals and no plans. NEVER claim to have visited, "
        "eaten at, watched, played, attended, bought or tried anything, and "
        "NEVER propose going somewhere. Do not refer to 'that new place' or "
        "any specific venue, event or outing as though it were shared "
        "knowledge — if it is not in the conversation above, it does not "
        "exist. Have opinions and ask questions instead; those are real.\n"
    )

    # Crypto-adjacent topics that naturally lead to AI infrastructure discussion
    _CRYPTO_AI_STARTERS = [
        "whether most AI tokens are just API wrappers with governance tokens nobody uses",
        "on-device vs cloud AI and which architecture actually makes sense long-term",
        "the gap between what crypto AI projects claim and what they actually ship",
        "whether decentralized AI is even possible without edge hardware",
        "why AI token projects keep raising money but never shipping inference",
        "the difference between an AI token and an AI project that happens to have a token",
    ]

    def build_starter_prompt(
        self,
        topics: list[str],
        active_users: list[str] | None = None,
        use_controversy: bool = False,
        recent_aura_messages: list[str] | None = None,
        conversation_context: str = "",
        chat_id: int | None = None,
    ) -> str:
        """Build an LLM prompt for generating a conversation starter.

        Args:
            topics: Group's top discussion topics
            active_users: Display names of active users (unused, kept for compat)
            use_controversy: If True, take a stronger stance
            recent_aura_messages: Last few things Aura said — avoid repeating
            conversation_context: Recent conversation history for context
            chat_id: Group id — enables the persistent no-repeat ledger
        """
        topic_str = ", ".join(topics) if topics else "tech, AI, crypto, or current events"
        no_repeat = config.LULL_NO_REPEAT and chat_id is not None

        # 70% of the time, go light/fun/news instead of work-related
        go_light = random.random() < 0.7
        if go_light:
            if no_repeat:
                topic_str = self._pick_theme(chat_id, self._LIGHT_STARTERS)
            else:
                topic_str = random.choice(self._LIGHT_STARTERS)
        # Occasionally seed crypto-AI infrastructure topics in crypto/AI groups
        elif any(t in topics for t in ("crypto", "ai", "tech")):
            if random.random() < 0.15:
                if no_repeat:
                    topic_str = self._pick_theme(chat_id, self._CRYPTO_AI_STARTERS)
                else:
                    topic_str = random.choice(self._CRYPTO_AI_STARTERS)

        if no_repeat:
            # Committed to the ledger by record_proactive_send on actual send
            self._pending_theme[chat_id] = topic_str

        # Dedup context so we don't sound like a parrot. With the ledger on,
        # everything sent in the no-repeat window joins the list — the
        # in-memory buffer alone forgets on every restart.
        dedup_lines = [m[:100] for m in (recent_aura_messages or [])]
        if no_repeat:
            for e in self._recent_ledger(chat_id, config.LULL_NO_REPEAT_DAYS):
                t = e.get("text", "")
                if t and t[:100] not in dedup_lines:
                    dedup_lines.append(t[:100])
        dedup = ""
        if dedup_lines:
            dedup = (
                "\n\nIMPORTANT — you recently said these things. Do NOT repeat "
                "the same topic, angle, or phrasing:\n"
                + "\n".join(f"- {m}" for m in dedup_lines[-15:])
                + "\nSay something COMPLETELY different.\n"
            )

        # Add conversation context if available
        context_block = ""
        if conversation_context:
            context_block = (
                f"\n\nHere's what the group has been talking about recently:\n"
                f"{conversation_context}\n"
                f"You can reference or build on this, or go in a different direction.\n"
            )

        if use_controversy:
            return (
                f"Drop a snappy thought about {topic_str}. "
                f"ONE sentence. Punchy. Have a real opinion."
                f"{context_block}{dedup}{self._NO_FABRICATION}"
            )
        else:
            return (
                f"Say something quick about {topic_str} — a quip, a hot take, "
                f"a one-liner, or a short question. ONE sentence max. "
                f"Keep it tight like a text, not a paragraph."
                f"{context_block}{dedup}{self._NO_FABRICATION}"
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
content_engine = ContentEngine()
