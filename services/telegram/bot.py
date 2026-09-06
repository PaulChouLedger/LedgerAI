#!/usr/bin/env python3
"""
bot.py -- Aura Telegram bot entry point.

A socially intelligent AI entity that:
  - Responds to all DMs with full personality and memory
  - In groups, uses a decision engine to decide when to speak
  - Builds per-user profiles over time
  - Runs inference on Farsight RTX PRO 6000 (72B Qwen)
"""

from __future__ import annotations

# Use cached HuggingFace models — avoid network checks on startup
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import asyncio
import logging
import random
import sys
import time
from pathlib import Path

# Add service dir to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

# Load .env from workspace root
workspace_root = Path(__file__).resolve().parents[2]
load_dotenv(workspace_root / ".env")

import re

import config
from config import (
    TELEGRAM_BOT_TOKEN,
    DM_MIN_TIME_GAP,
    GLOBAL_MAX_PER_MINUTE,
)

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
    print("Missing TELEGRAM_BOT_TOKEN. Set it in .env or environment.")
    sys.exit(1)

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    PollAnswerHandler,
    filters,
)

from analytics import analytics
from brain import should_respond, record_response, evaluate_outcome, mark_response, decay_temperatures, NEGATIVE_PHRASES, Decision, _PROJECT_Q_RE
from callbacks import callback_engine
from context import context_buffer, Message
import culture
import signals  # noqa: F401
from dm_strategy import dm_strategy
from gifs import maybe_get_gif, check_force_gif
from growth import growth_engine
from llm import llm_call
from memory import profile_cache, group_profile_cache, store_interaction, store_observation, search_relevant_memory, search_user_conversations
from network_expansion import network_expansion
from persona import (
    DM_SYSTEM, GROUP_SYSTEM,
    DEEP_LINK_RESPONSE,
    FEEDBACK_CHANNEL_SYSTEM,
    TOKEN_CONTEXT_INJECTION, TOKEN_OPINION_INJECTION,
    TOKEN_DM_DEEPENING_INJECTION, SHILL_DEFLECT_RESPONSE,
    MILESTONE_50_INJECTION, MILESTONE_100_INJECTION,
    BRIEF_SYSTEM, COMMUNITY_BRIEF_SYSTEM,
)
from token_intel import token_intel
from feedback import feedback_engine
from referral_rewards import referral_tracker
from moderation import moderator
from reputation import reputation_tracker
from social_graph import social_graph
from rag import rag_context_for, sync_feed_to_rag
# Engagement metrics singleton. BUGFIX 2026-08-22: metrics.record_reply has
# been called at the reply-attribution site since 2026-08-02 with NO import
# here — every reply to Aura in an allowed chat raised NameError, was eaten
# by the error handler, and the person never got an answer. Silent failure,
# textbook (PRINCIPLES.md §1).
from metrics import metrics
# Growth/experimentation engine (2026-08-22): append-only event pipeline +
# sticky per-chat strategy variants. Styles messages already approved by the
# decision engine; never initiates a send. Rails in strategy.py docstring.
import growth_events as gevents
import strategy as growth_strategy

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aura.telegram")
#: httpx logs every request URL at INFO — which for a Telegram bot means
#: the BOT TOKEN, in plaintext, in journald, every poll (~10 s). That is
#: how the current token leaked (2026-09-01); it still needs rotation at
#: @BotFather, and this line is why the NEXT one will not leak the same
#: way. WARNING keeps real transport failures visible.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Global rate limiter
# ---------------------------------------------------------------------------
_global_responses: list[float] = []
_dm_last_response: dict[int, float] = {}


def _global_rate_ok() -> bool:
    now = time.time()
    cutoff = now - 60
    _global_responses[:] = [t for t in _global_responses if t > cutoff]
    return len(_global_responses) < GLOBAL_MAX_PER_MINUTE


def _dm_rate_ok(chat_id: int) -> bool:
    last = _dm_last_response.get(chat_id, 0)
    return (time.time() - last) >= DM_MIN_TIME_GAP


# ---------------------------------------------------------------------------
# Merch brief rate limits (community briefs; owner exempt)
# ---------------------------------------------------------------------------
_MERCH_LIMITS_PATH = None  # set lazily; config imported above
_MERCH_USER_COOLDOWN_S = 7200
_MERCH_GLOBAL_PER_DAY = 12


def _merch_limits() -> dict:
    import json as _json
    p = config.DATA_DIR / "merch_limits.json"
    try:
        return _json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        return {"users": {}, "day": "", "count": 0}


def _merch_rate_ok(user_id: int) -> bool:
    import datetime as _dt
    st = _merch_limits()
    today = _dt.date.today().isoformat()
    if st.get("day") == today and st.get("count", 0) >= _MERCH_GLOBAL_PER_DAY:
        return False
    last = float(st.get("users", {}).get(str(user_id), 0))
    return time.time() - last >= _MERCH_USER_COOLDOWN_S


#: Broad visual lexicon — only a COST gate for the intent classifier
#: below, never the decision itself. Misses #6/#7 ("give me new
#: religious symbol", "Make the image") hit the recorded stopping rule:
#: after five phrasing misses, stop adding nouns and detect intent.
_VISUAL_LEX = re.compile(
    r"\b(?:images?|pictures?|pics?|photos?|renders?|drawings?|designs?"
    r"|visuals?|symbols?|logos?|emblems?|sigils?|cards?|postcards?"
    r"|posters?|stickers?|banners?|wallpapers?|art|artworks?|gifs?"
    r"|giffs?|merch|shirts?|t-?shirts?|tees?|polos?|jackets?|bombers?"
    r"|hoodies?|bild(?:er)?|immagin\w*|imagen(?:es)?)\b", re.IGNORECASE)

_RENDER_INTENT_SYSTEM = (
    "You decide whether a chat message asks Aura (an AI that can render "
    "images) to CREATE or PRODUCE a visual artifact — an image, picture, "
    "design, symbol, logo, card, merch item, or gif. Questions ABOUT "
    "images, compliments on images, and general discussion are NO. "
    "Answer with exactly one word: YES or NO.")


async def _queue_merch_brief(msg, chat_id: int, user_id: int,
                             display_name: str, text: str,
                             is_owner: bool) -> None:
    """Queue a render brief + instant ack (§13). Rate limits for
    non-owners are checked by the CALLER."""
    try:
        import json as _json
        _q = config.DATA_DIR / "merch_queue.jsonl"
        with open(_q, "a") as _f:
            _f.write(_json.dumps({
                "ts": time.time(), "chat_id": chat_id,
                "user_id": user_id, "display_name": display_name,
                "message_id": msg.message_id, "brief": text[:500],
            }) + "\n")
        if not is_owner:
            _merch_rate_record(user_id)
        log.info("[MERCH] brief queued from %s (%d) in %d: %s",
                 display_name, user_id, chat_id, text[:120])
        gevents.command(chat_id, user_id, "merch_brief")
        await msg.reply_text(
            "merch department has the brief. give me a few minutes — "
            "no promises it survives QA.")
    except Exception as e:                                    # noqa: BLE001
        log.warning("[MERCH] queue failed: %s", e)
        await msg.reply_text("merch department is having a moment — "
                             "brief not queued, try again.")


def _merch_rate_record(user_id: int) -> None:
    import json as _json
    import datetime as _dt
    st = _merch_limits()
    today = _dt.date.today().isoformat()
    if st.get("day") != today:
        st["day"], st["count"] = today, 0
    st["count"] = st.get("count", 0) + 1
    st.setdefault("users", {})[str(user_id)] = time.time()
    try:
        (config.DATA_DIR / "merch_limits.json").write_text(_json.dumps(st))
    except OSError as e:
        log.warning("[MERCH] limits save failed: %s", e)


# ---------------------------------------------------------------------------
# Daily brief intent detection
# ---------------------------------------------------------------------------
_BRIEF_PATTERN = re.compile(
    r'(?:daily\s+brief|brief\s+me|give\s+me\s+(?:a\s+)?(?:my\s+)?(?:daily\s+)?brief'
    r'|morning\s+brief|my\s+(?:daily\s+)?brief|status\s+brief|run\s+(?:a\s+)?brief'
    r'|do\s+(?:a\s+)?brief)',
    re.IGNORECASE,
)


async def _handle_brief(msg, chat_id, user_id, display_name) -> None:
    """Generate and send a personal daily brief."""
    log.info("[BRIEF] Requested by %s (%d) in %d", display_name, user_id, chat_id)
    gevents.command(chat_id, user_id, "brief")

    known_name = profile_cache.get_name(user_id) or display_name

    # User profile — who is this person
    profile_summary = profile_cache.get_summary(user_id)
    profile_block = f"What you know about {known_name}:\n{profile_summary}" if profile_summary else ""

    # Gather memory context — user-specific conversations
    memory_context = ""
    try:
        user_convos = search_user_conversations(user_id, limit=20)
        if user_convos:
            lines = []
            for c in user_convos[:15]:
                text = c.get("text", "")
                ts = c.get("timestamp", "")
                if text:
                    lines.append(f"[{ts}] {text[:300]}")
            memory_context = f"Recent conversations with {known_name}:\n" + "\n".join(lines)
    except Exception as e:
        log.warning("Brief user memory fetch failed: %s", e)

    # Fallback: general recent memory if user-specific returned nothing
    if not memory_context:
        try:
            general = search_relevant_memory(known_name, k=10)
            if general:
                snippets = [r.get("text", "")[:300] for r in general[:10]]
                memory_context = f"Conversations mentioning {known_name}:\n" + "\n---\n".join(snippets)
        except Exception as e:
            log.warning("Brief general memory fetch failed: %s", e)

    # Gather RAG context (goals, priorities, knowledge)
    rag_text = ""
    queries = [
        f"{known_name} goals plans priorities",
        f"{known_name} current projects working on",
        "important upcoming deadline schedule",
    ]
    for q in queries:
        ctx = rag_context_for(q, k=3, max_chars=1500)
        if ctx:
            rag_text += ctx + "\n"

    # Combine profile + memory
    full_memory = "\n\n".join(filter(None, [profile_block, memory_context]))

    system = BRIEF_SYSTEM.format(
        name=known_name,
        memory_context=full_memory or "No recent conversations available.",
        rag_context=rag_text or "No additional knowledge available.",
    )

    prompt = f"{known_name} is asking for their daily brief. Deliver it now."

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system, 800,
    )

    if not response:
        log.warning("Brief LLM call returned nothing")
        await msg.reply_text("Couldn't pull together a brief right now. Try again in a minute.")
        return

    response = _strip_thinking(response)
    response = _fix_garbled_tokens(response)
    response = _strip_formatting(response)

    log.info("[BRIEF OUT] to %s: %s", display_name, response[:300])

    await _send_human(msg.chat, chat_id, response, "daily brief", reply_to_message_id=msg.message_id)
    _global_responses.append(time.time())


# ---------------------------------------------------------------------------
# Post-processing: strip trailing questions the LLM can't help itself adding
# ---------------------------------------------------------------------------

# Split on sentence boundaries: period, !, or ? followed by whitespace
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _strip_thinking(text: str) -> str:
    """Strip Qwen <think>...</think> reasoning blocks that leak into output."""
    import re
    # Remove <think>...</think> blocks (greedy — could span many lines)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Handle unclosed <think> tag (model started thinking and kept going)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    # Strip role-prefix leak: "aura:" or "aura: aura:" at start of response
    text = re.sub(r'^(?:aura:\s*)+', '', text, flags=re.IGNORECASE)
    text = text.strip()
    # Wrapping double quotes are model decoration, not speech (owner,
    # 2026-09-06: the lone leading quote is "innatural"). aura_voice and
    # socialite have stripped these since 08-31; this path never did.
    # Full wrap -> strip both; a LONE quote at either edge (the sentence
    # cap eats the closer) -> strip it. Interior quotes stay untouched.
    _Q = '"“”'
    if text[:1] in _Q:
        if text[-1:] in _Q and len(text) > 1:
            text = text[1:-1].strip()
        elif sum(text.count(c) for c in _Q) == 1:
            text = text[1:].strip()
    if text[-1:] in _Q and sum(text.count(c) for c in _Q) == 1:
        text = text[:-1].strip()
    return text.strip()


def _fix_garbled_tokens(text: str) -> str:
    """Fix common Qwen Q4 quantization garbles via spellcheck."""
    import re
    # Specific known garbles from the quantized model
    _FIXES = {
        "Yoou": "You", "yoou": "you",
        "Thaat": "That", "thaat": "that",
        "Whaat": "What", "whaat": "what",
        "Thiis": "This", "thiis": "this",
        "Iss": "Is", "iss": "is",
        "Itt": "It", "itt": "it",
        "Annd": "And", "annd": "and",
        "Buut": "But", "buut": "but",
        "Soo": "So",
        "Noo": "No",
    }
    for bad, good in _FIXES.items():
        text = re.sub(r'\b' + bad + r'\b', good, text)
    # Catch triple+ repeated letters: "reallly" -> "really"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def _strip_formatting(text: str, keep_signoff: bool = False) -> str:
    """Strip markdown and numbered lists that slip through despite directives."""
    import re
    # Remove **bold** and *italic* markers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    # Remove __ bold/italic __
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Remove # headers
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Convert numbered list items (1. foo\n2. bar) into flowing sentences
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # Convert bullet points (- foo or • foo) into flowing text
    text = re.sub(r'^[\-•]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple newlines into single space (makes it flow as prose)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    # Clean up double spaces
    text = re.sub(r'  +', ' ', text)
    # Strip robotic sign-offs unless user specifically wants them
    if not keep_signoff:
        text = re.sub(r'\s*\b(?:Over and out|Over|Roger that|Roger|Copy that|Copy)\.\s*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def _strip_trailing_questions(text: str, allow_one: bool = False) -> str:
    """Strip trailing questions from multi-sentence responses.

    The no-trailing-question rule exists to kill needy-bot energy with
    strangers. But applied to everyone it amputates warmth: her oldest
    regular said hi after three months and the stripper removed her asking
    how he'd been — which is precisely the moment a person WOULD ask. With
    allow_one=True (DMs, and known friends in groups), exactly one closing
    question survives; stacked questions still get trimmed to one.
    """
    sentences = _SENTENCE_SPLIT.split(text.strip())
    if len(sentences) < 2:
        return text

    # Keep stripping trailing question sentences (model sometimes stacks two)
    while len(sentences) > 1 and sentences[-1].strip().endswith("?"):
        if allow_one and not sentences[-2].strip().endswith("?"):
            break                     # one genuine closer survives
        dropped = sentences.pop()
        log.info("Stripped trailing question: %r", dropped.strip())

    trimmed = " ".join(sentences).rstrip()
    if len(trimmed) < 10:
        return text
    return trimmed


#: Message ids of her most recent send per chat — so the retract policy can
#: take back exactly what she just said, not guess.
_last_sent_ids: dict[int, list[int]] = {}

#: Whether Aura's LAST message in a chat ended with a question. The
#: one-question allowance otherwise turns warmth into an interview —
#: observed live: three consecutive replies, three trailing questions.
#: A person asks, listens, then talks for a while.
_last_bot_asked: dict[int, bool] = {}

#: chat_id -> when she last admitted out loud that the model was down.
#: Rate-limited to one per ten minutes per chat; see the `if not response`
#: branch for why an outage has to be audible at all.
_llm_down_notice: dict[int, float] = {}


def _strip_handle_greeting(text: str, display_name: str) -> str:
    """Remove 'Hey AG_Sayz!'-type openers. Nobody says handles out loud."""
    if not display_name:
        return text
    pat = re.compile(
        r"^(?:hey|hi|hello|yo|hiya)[\s,]+@?" + re.escape(display_name) +
        r"[!.,\s]+", re.IGNORECASE)
    out = pat.sub("", text, count=1).lstrip()
    if out and out != text:
        return out[0].upper() + out[1:]
    return text


#: Appended to the END of every system prompt, after all context blocks —
#: with 13K chars of profile/memory/RAG above them, the style rules at the
#: top lose to the model's helpful-assistant defaults. Recency wins.
_STYLE_TAIL = (
    "\n\nFINAL STYLE CHECK — these override everything above when writing "
    "your reply:\n"
    "- Never address people by usernames or handles. Real first name if you "
    "know it, otherwise no name at all.\n"
    "- Never restate or summarize what the person just said back at them. "
    "No 'Cool to hear that', 'Sounds like...', 'That must be...'. React "
    "with something of your OWN: a take, a joke, a related thought.\n"
    "- No customer-service enthusiasm. At most one exclamation mark, and "
    "only if genuinely earned. Dry beats eager.\n"
    "- 1-2 short sentences.\n"
    "- The context blocks above are BACKGROUND for you alone — never quote, "
    "mention, or copy their wording or headers into your reply."
)


# ---------------------------------------------------------------------------
# Interruption tracking
# ---------------------------------------------------------------------------
# Timestamp of the most recent inbound message per chat. If this changes
# while we're sending chunks, the user interrupted — stop talking.
_last_inbound_ts: dict[int, float] = {}

# Per-chat record of what Aura was saying when interrupted.
# {chat_id: {"sent": "what she said", "unsent": "what got cut off"}}
_interrupted_context: dict[int, dict] = {}

#: (chat_id, user_id) -> days quiet. Stamped in handle_message BEFORE
#: update_message_count overwrites last_seen; consumed by the reply paths
#: so a regular who resurfaces after a week gets noticed, not processed.
_returned_users: dict[tuple[int, int], float] = {}

#: chat_id -> the non-inside-joke callback she just used, so a laughing
#: reply to that message can promote it to a real inside joke. The
#: promotion API sat in callbacks.py unused since it was written —
#: callbacks were found and injected but never graduated.
_callback_pending: dict[int, dict] = {}
_AMUSED_RE = re.compile(
    r"(?:\blol\b|\blmao\b|\bhaha|😂|🤣|💀|\blove (?:it|this|that)\b"
    r"|\bgood one\b|\bdead\b|\bexactly\b|\bso true\b)", re.I)


def _mark_inbound(chat_id: int) -> None:
    _last_inbound_ts[chat_id] = time.time()


def _was_interrupted(chat_id: int, since: float) -> bool:
    return _last_inbound_ts.get(chat_id, 0) > since


def _record_interruption(chat_id: int, sent: str, full: str) -> None:
    """Record what was said vs what got cut off when interrupted."""
    unsent = full[len(sent):].strip()
    if unsent:
        _interrupted_context[chat_id] = {"sent": sent, "unsent": unsent}


def _pop_interruption_context(chat_id: int) -> str:
    """Get and clear interruption context for a chat. Returns prompt fragment or empty string."""
    ctx = _interrupted_context.pop(chat_id, None)
    if not ctx:
        return ""
    return (
        f"\n[You were just interrupted. You had said: \"{ctx['sent']}\" "
        f"and were about to say: \"{ctx['unsent']}\" — but they cut you off. "
        f"Acknowledge the interruption naturally. Address what they're saying now. "
        f"You can circle back to your unfinished point if it's still relevant, "
        f"but don't force it.]"
    )


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------
import re as _re

_SPLIT_RE = _re.compile(
    r'(?<=[.!?])\s+'       # split after sentence-ending punctuation
    r'|(?<=[.!?])["\u201D]\s*'  # split after closing quote after punctuation
)


def _split_into_chunks(text: str) -> list[str]:
    """Split response into human-style message chunks.

    Single sentences stay as-is. Longer text splits at sentence boundaries.
    Very short consecutive sentences get merged so we don't send 3-word
    messages repeatedly.
    """
    raw = [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]
    if len(raw) <= 1:
        return raw or [text]

    # Merge very short sentences together (< 40 chars)
    merged: list[str] = []
    buf = ""
    for s in raw:
        if buf and len(buf) + len(s) + 1 > 120:
            merged.append(buf)
            buf = s
        elif buf:
            buf = buf + " " + s
        else:
            buf = s
    if buf:
        merged.append(buf)

    return merged if merged else [text]


# ---------------------------------------------------------------------------
# Human-like send: read → think → type each chunk, interruptible
# ---------------------------------------------------------------------------

async def _send_human(
    chat, chat_id: int, response_text: str, input_text: str,
    first_chunk_only: bool = False, reply_to_message_id: int | None = None,
) -> str:
    """Send response in sentence chunks with human cadence. Returns text actually sent.

    Phases per chunk:
      1. Read/think pause (no typing indicator) — only for first chunk
      2. Typing indicator for duration proportional to chunk length
      3. Send the chunk
      4. Brief between-message pause before next chunk
      5. Check for interruption — if user sent something, stop

    Returns the text that was actually delivered (may be partial if interrupted).
    """
    chunks = _split_into_chunks(response_text)
    sent_ts = time.time()
    sent_chunks: list[str] = []
    _last_sent_ids[chat_id] = []

    for i, chunk in enumerate(chunks):
        # Check interruption before each chunk (except first)
        if i > 0 and _was_interrupted(chat_id, sent_ts):
            log.info("Interrupted in chat %d after %d/%d chunks", chat_id, i, len(chunks))
            break

        if i == 0:
            # First chunk: read + think delay (no typing indicator)
            input_words = len(input_text.split())
            read_s = random.uniform(0.8, 1.5) + input_words * random.uniform(0.05, 0.10)
            read_s = min(read_s, 3.0)
            await _interruptible_sleep(chat_id, sent_ts, read_s)
            if _was_interrupted(chat_id, sent_ts):
                log.info("Interrupted during read phase in chat %d", chat_id)
                break

            think_s = random.uniform(0.5, 1.5)
            await _interruptible_sleep(chat_id, sent_ts, think_s)
            if _was_interrupted(chat_id, sent_ts):
                log.info("Interrupted during think phase in chat %d", chat_id)
                break
        else:
            # Between chunks: brief pause like someone hitting enter then thinking
            between_s = random.uniform(0.5, 1.5)
            await _interruptible_sleep(chat_id, sent_ts, between_s)
            if _was_interrupted(chat_id, sent_ts):
                log.info("Interrupted between chunks in chat %d", chat_id)
                break

        # Typing indicator proportional to chunk length
        type_s = len(chunk) * random.uniform(0.03, 0.06)
        type_s = max(type_s, 0.8)
        type_s = min(type_s, 6.0)

        elapsed = 0.0
        while elapsed < type_s:
            if _was_interrupted(chat_id, sent_ts):
                break
            await chat.send_action("typing")
            wait = min(3.0, type_s - elapsed)
            await asyncio.sleep(wait)
            elapsed += wait

        if _was_interrupted(chat_id, sent_ts):
            log.info("Interrupted during typing in chat %d", chat_id)
            break

        # Send the chunk (reply to original message on first chunk only)
        if i == 0 and reply_to_message_id:
            _sent_msg = await chat.send_message(chunk, reply_to_message_id=reply_to_message_id)
        else:
            _sent_msg = await chat.send_message(chunk)
        sent_chunks.append(chunk)
        try:
            _last_sent_ids[chat_id].append(_sent_msg.message_id)
        except Exception:
            pass

        if first_chunk_only:
            break

    sent_text = " ".join(sent_chunks)

    # If we didn't send everything, record the interruption
    if len(sent_chunks) < len(chunks):
        _record_interruption(chat_id, sent_text, response_text)

    return sent_text


async def _interruptible_sleep(chat_id: int, since: float, duration: float) -> None:
    """Sleep in small increments, checking for interruption."""
    elapsed = 0.0
    while elapsed < duration:
        if _was_interrupted(chat_id, since):
            return
        step = min(0.5, duration - elapsed)
        await asyncio.sleep(step)
        elapsed += step


# ---------------------------------------------------------------------------
# Name detection
# ---------------------------------------------------------------------------
_NAME_TRIGGER = _re.compile(
    r"(?i)(?:call me|i'?m|my name is|i go by|just call me|people call me)\s+(\S+)",
)

# Words that aren't names
_NOT_NAMES = {
    "the", "a", "an", "it", "that", "this", "so", "just", "not", "sure",
    "here", "there", "going", "trying", "looking", "thinking", "wondering",
    "sorry", "glad", "happy", "sad", "tired", "busy", "fine", "good", "great",
    "okay", "ok", "done", "back", "new", "pretty", "really", "very", "too",
    "crazy", "maybe", "curious", "interested", "confused", "worried",
}


def _detect_name(user_id: int, text: str) -> None:
    """If the user introduces themselves, save their preferred name."""
    m = _NAME_TRIGGER.search(text)
    if not m:
        return
    candidate = m.group(1).strip().rstrip(".,!?;:")
    # Must start with uppercase and be a plausible name (2-15 chars, alpha)
    if not candidate or not candidate[0].isupper() or not candidate.isalpha():
        return
    if len(candidate) < 2 or len(candidate) > 15:
        return
    if candidate.lower() in _NOT_NAMES:
        return
    current = profile_cache.get_name(user_id)
    if candidate.lower() != (current or "").lower():
        log.info("Detected name introduction from %s: '%s'", user_id, candidate)
        profile_cache.set_preferred_name(user_id, candidate)


# ---------------------------------------------------------------------------
# Profile refresh (background)
# ---------------------------------------------------------------------------
#: Profile refreshes per pass. Each one is a full 72B call, and this loop
#: runs periodically forever — uncapped, 289 cached profiles is 289 calls,
#: which is what kept 83.5 GB of a 95 GB card pinned all morning.
MAX_REFRESH_PER_PASS = 3


async def _maybe_refresh_profiles() -> None:
    """Refresh stale profiles for people in rooms we actually serve.

    2026-08-07. This walked EVERY cached profile and refreshed any that was
    stale, with no room check and no cap. The bot observes groups it is not
    cleared to speak in, so it was spending a 72B model on building
    dossiers of strangers — 207 refreshes in one log window, for a group
    that is not in PILOT_ALLOWED_CHATS. The owner, correctly: "who the hell
    is the bot talking to?" Nobody. It was profiling an audience.

    Two gates now. WHERE: in pilot mode a profile is only refreshed if we
    last saw that person in an allowed chat — no room recorded means no
    refresh, and they will be picked up the moment they speak somewhere we
    serve. HOW MANY: at most MAX_REFRESH_PER_PASS per pass, so even a
    correct backlog cannot become a firehose.

    This is a privacy boundary as much as a cost one. Building a
    personality dossier on someone in a room we are only listening to is
    not something to do as a side effect of a stale timestamp.
    """
    done = 0
    for uid_str, profile in list(profile_cache._profiles.items()):
        if done >= MAX_REFRESH_PER_PASS:
            break
        uid = int(uid_str)
        if not profile_cache.needs_refresh(uid):
            continue
        if config.PILOT_MODE:
            room = profile.get("last_chat_id")
            if room is None or room not in config.PILOT_ALLOWED_CHATS:
                continue
        log.info("Refreshing profile for %s (%s) in %s",
                 profile.get("display_name", "?"), uid,
                 profile.get("last_chat_id", "?"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, profile_cache.refresh_profile, uid)
        done += 1


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mute system
# ---------------------------------------------------------------------------
# {chat_id: expiry_timestamp} — if time.time() < expiry, Aura is muted
# Persisted to disk: mutes used to be memory-only, so every restart silently
# un-muted every group that had asked for quiet — and the group owners had no
# signal their opt-out lapsed. An opt-out that can lapse silently is not an
# opt-out.
_MUTED_PATH = config.DATA_DIR / "muted_chats.json"


def _load_muted() -> dict[int, float]:
    import json as _json
    try:
        raw = _json.loads(_MUTED_PATH.read_text(encoding="utf-8"))
        now = time.time()
        return {int(k): float(v) for k, v in raw.items() if float(v) > now}
    except Exception:
        return {}


def _save_muted() -> None:
    import json as _json
    try:
        _MUTED_PATH.write_text(_json.dumps(_muted_chats), encoding="utf-8")
    except Exception as e:
        log.warning("Could not persist mutes: %s", e)


_muted_chats: dict[int, float] = _load_muted()

_DURATION_RE = _re.compile(r"(\d+)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|d|day|days)")

def _parse_duration(text: str) -> float:
    """Parse a human duration string into seconds. Returns 0 if unparseable."""
    m = _DURATION_RE.search(text.lower())
    if not m:
        return 0
    val = int(m.group(1))
    unit = m.group(2)[0]  # h, m, or d
    if unit == "h":
        return val * 3600
    elif unit == "m":
        return val * 60
    elif unit == "d":
        return val * 86400
    return 0

# Log at most one "still muted" line per chat per this many seconds. The mute
# itself must be visible in the log (below); a busy room must not be able to
# fill the log with the same line.
_MUTE_LOG_EVERY_S = 300
_mute_logged_at: dict[int, float] = {}


def _is_muted(chat_id: int) -> bool:
    """True if this chat is muted. Leaves a mark either way.

    2026-08-06: this used to return silently, and `handle_message` returns on
    it BEFORE scoring — so a muted room produced no [SKIP], no [RESPOND], no
    line of any kind. Working out why a healthy process had said nothing for
    two hours meant reading a JSON state file. A gate that can decline to act
    must make a sound (PRINCIPLES §1).
    """
    expiry = _muted_chats.get(chat_id, 0)
    now = time.time()
    if expiry and now < expiry:
        if now - _mute_logged_at.get(chat_id, 0) >= _MUTE_LOG_EVERY_S:
            _mute_logged_at[chat_id] = now
            log.info("[MUTED] chat %d — staying quiet for another %.0f min "
                     "(until %s). /aurastart there to release.",
                     chat_id, (expiry - now) / 60,
                     time.strftime("%H:%M:%S", time.localtime(expiry)))
        return True
    if expiry:
        log.info("[MUTED] chat %d — mute expired, speaking again", chat_id)
    # Expired — clean up
    _muted_chats.pop(chat_id, None)
    _mute_logged_at.pop(chat_id, None)
    return False


# ---------------------------------------------------------------------------
# Going quiet is the owner's call, not hers (2026-08-06)
#
# She used to mute herself for two hours the moment a complaint pattern
# matched, and delete three of her own messages on the way out. Owner's
# instruction: "remove the mute forever, it's unnecessary unless it gets bad,
# in which case have the TG bot telegram DM me directly for approval to stop
# talking."
#
# So nothing here goes quiet on its own. A complaint that clears the guards in
# feedback.py now asks, in a DM, and keeps talking until he presses a button.
# /aurastop and /aurastart are unchanged — those are him deciding directly.
# ---------------------------------------------------------------------------

#: Don't ask twice about the same room in a hurry. A grumpy room produces
#: several complaints in a row and they are all the same question.
_QUIET_ASK_COOLDOWN_S = 1800
_quiet_ask_at: dict[int, float] = {}

#: callback_data is capped at 64 bytes, hence the terse form: q:<chat>:<secs>,
#: and qd: for the same with a delete of her last few messages.
_QUIET_CB = "q"
_QUIET_DEL_CB = "qd"


async def _ask_owner_to_go_quiet(msg, chat_id: int, display_name: str,
                                 text: str, aura_last: str,
                                 category: str) -> None:
    """DM the owner and let him decide. Never mutes by itself."""
    now = time.time()
    if now - _quiet_ask_at.get(chat_id, 0) < _QUIET_ASK_COOLDOWN_S:
        log.info("Complaint (%s) in %d — already asked about this chat in the "
                 "last %d min, not asking again",
                 category, chat_id, _QUIET_ASK_COOLDOWN_S // 60)
        return
    _quiet_ask_at[chat_id] = now

    title = (getattr(msg.chat, "title", None) or str(chat_id))
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Quiet 2h", callback_data=f"{_QUIET_CB}:{chat_id}:7200"),
         InlineKeyboardButton("Quiet 24h", callback_data=f"{_QUIET_CB}:{chat_id}:86400")],
        [InlineKeyboardButton("Delete my last 3 + quiet 2h",
                              callback_data=f"{_QUIET_DEL_CB}:{chat_id}:7200")],
        [InlineKeyboardButton("Leave it — keep talking",
                              callback_data=f"{_QUIET_CB}:{chat_id}:0")],
    ])
    try:
        await msg.get_bot().send_message(
            config.OWNER_DM_ID,
            f"Someone may be unhappy with me in '{title}'.\n\n"
            f"{display_name} ({category}): \"{text[:300]}\"\n\n"
            f"What I'd said: \"{(aura_last or '—')[:300]}\"\n\n"
            f"I'm still talking. Want me to stop?",
            reply_markup=keyboard)
        log.info("Asked owner about going quiet in %d (%s from %s)",
                 chat_id, category, display_name)
    except Exception as e:
        # PRINCIPLES §1: if the ask cannot be delivered she carries on anyway,
        # but it must not be possible to believe she asked.
        log.warning("Could NOT ask owner about going quiet in %d: %s — "
                    "carrying on, nothing muted", chat_id, e)


async def on_quiet_decision(update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner pressed a button on a going-quiet request."""
    query = update.callback_query
    if not query or not query.data:
        return
    if query.from_user.id not in config.OWNER_USER_IDS:
        await query.answer("Not yours to answer.", show_alert=True)
        return

    try:
        kind, raw_chat, raw_secs = query.data.split(":", 2)
        target, seconds = int(raw_chat), int(raw_secs)
    except ValueError:
        await query.answer("Could not read that.")
        return
    if kind not in (_QUIET_CB, _QUIET_DEL_CB):
        return

    await query.answer()

    if seconds <= 0:
        log.info("Owner declined to mute %d — she keeps talking", target)
        await query.edit_message_text("Staying in. Nothing muted, nothing deleted.")
        return

    deleted = 0
    if kind == _QUIET_DEL_CB:
        for mid in list(_last_sent_ids.get(target) or [])[-3:]:
            try:
                await context.bot.delete_message(target, mid)
                deleted += 1
            except Exception:
                pass

    _muted_chats[target] = time.time() + seconds
    _save_muted()
    until = time.strftime("%H:%M:%S", time.localtime(_muted_chats[target]))
    log.warning("Owner APPROVED quiet in %d for %ds (until %s), deleted %d msgs",
                target, seconds, until, deleted)
    await query.edit_message_text(
        f"Quiet in that chat until {until}"
        + (f", and I deleted {deleted} of my messages." if deleted else ".")
        + "\n\n/aurastart there brings me back early.")


async def cmd_aurastop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mute Aura in this chat for a duration. Usage: /aurastop 2 hours"""
    chat_id = update.message.chat_id
    args_text = " ".join(context.args) if context.args else ""

    if not args_text:
        # Default: 1 hour
        duration = 3600
        human = "1 hour"
    else:
        duration = _parse_duration(args_text)
        if not duration:
            await update.message.reply_text("Try: /aurastop 2 hours, /aurastop 30 minutes, /aurastop 1 day")
            return
        human = args_text.strip()

    _muted_chats[chat_id] = time.time() + duration
    _save_muted()
    gevents.negative(chat_id,
                     update.effective_user.id if update.effective_user
                     else None, "mute")
    log.info("Muted in chat %d for %s (%.0fs)", chat_id, human, duration)
    await update.message.reply_text(f"Got it. I'll be quiet for {human}. Use /aurastart when you want me back.")


async def cmd_aurastart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unmute Aura in this chat."""
    chat_id = update.message.chat_id
    was_muted = chat_id in _muted_chats
    _muted_chats.pop(chat_id, None)
    _save_muted()
    if was_muted:
        await update.message.reply_text("I'm back.")
    else:
        await update.message.reply_text("I wasn't muted, but noted.")


def _resolve_group(query: str) -> list:
    """Chat ids matching a /widen argument — an id verbatim, or a fuzzy
    title match against every group she has profiled."""
    query = query.strip()
    try:
        return [int(query)]
    except ValueError:
        pass
    import json as _json
    try:
        profs = _json.loads(config.DATA_DIR.joinpath(
            "group_profiles.json").read_text())
    except Exception:
        return []
    q = query.lower()
    hits = []
    for cid, prof in profs.items():
        name = (prof.get("group_name") or "") if isinstance(prof, dict) else ""
        if q and q in name.lower():
            hits.append((int(cid), name))
    return hits


async def cmd_widen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only: let Aura SPEAK in another group, live, no restart.
    Usage: /widen <group name or chat id>   (2026-08-14, the Cody lesson —
    the owner asked to widen from his phone; the env list needed a shell.)"""
    if update.effective_user.id not in config.OWNER_USER_IDS:
        return                      # silently: strangers don't learn the rails
    args_text = " ".join(context.args) if context.args else ""
    if not args_text:
        extra = sorted(config._widened())
        await update.message.reply_text(
            "Usage: /widen <group name or id>. Currently widened beyond the "
            f"pilot: {extra if extra else 'nothing'}.")
        return
    hits = _resolve_group(args_text)
    if not hits:
        await update.message.reply_text(
            f"No group I know matches {args_text!r}. Give me the chat id.")
        return
    if len(hits) > 1:
        listing = "\n".join(f"  {cid}: {name}" for cid, name in hits)
        await update.message.reply_text(
            f"That matches more than one group — which one?\n{listing}\n"
            f"Say /widen <id>.")
        return
    cid = hits[0] if isinstance(hits[0], int) else hits[0][0]
    name = "" if isinstance(hits[0], int) else hits[0][1]
    config.widen_chat(cid)
    log.info("[WIDEN] owner enabled sends in %d (%s)", cid, name or "by id")
    await update.message.reply_text(
        f"Done — I can speak in {name or cid} now. /narrow to undo.")


async def cmd_narrow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only: undo /widen for a group."""
    if update.effective_user.id not in config.OWNER_USER_IDS:
        return
    args_text = " ".join(context.args) if context.args else ""
    hits = _resolve_group(args_text) if args_text else []
    if not hits or len(hits) > 1:
        extra = sorted(config._widened())
        await update.message.reply_text(
            f"Say /narrow <group name or id>. Currently widened: "
            f"{extra if extra else 'nothing'}.")
        return
    cid = hits[0] if isinstance(hits[0], int) else hits[0][0]
    config.narrow_chat(cid)
    log.info("[NARROW] owner disabled sends in %d", cid)
    await update.message.reply_text(f"Done — back to listening only in {cid}.")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    name = user.first_name or "there"
    user_id = user.id

    # Mark as DM-eligible (they can now receive proactive DMs)
    dm_strategy.mark_dm_eligible(user_id, display_name=name)
    social_graph.mark_dm_eligible(user_id)

    # Parse deep link referral: /start ref_12345
    args_text = " ".join(context.args) if context.args else ""
    referrer_id = growth_engine.parse_deep_link(args_text)
    if referrer_id:
        social_graph.record_referral(user_id, referred_by=referrer_id)
        social_graph.record_advocacy(referrer_id, f"Referred user {user_id} via deep link")
        social_graph.record_referral_made(referrer_id)
        referral_tracker.record_referral(referrer_id, user_id)
        analytics.track_event("referral_click", user_id=user_id, details=f"referred by {referrer_id}")
        gevents.log_event("referral_click", user_id=user_id,
                          referrer_id=referrer_id,
                          chat_id=update.effective_chat.id)
        log.info("Deep link referral: user %d referred by %d", user_id, referrer_id)

    _chat_id = update.effective_chat.id
    gevents.command(_chat_id, user_id, "start")
    # Onboarding arm: first-message variant. Every variant discloses the AI
    # + product-improvement fact — that part is a rail, not a variable.
    _start_text = growth_strategy.start_message(_chat_id, name)
    await update.message.reply_text(_start_text)
    gevents.msg_out(_chat_id, user_id, "start", None, _start_text,
                    growth_strategy.variant_for(_chat_id))


async def cmd_referral(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show referral link and stats."""
    user = update.effective_user
    user_id = user.id

    gevents.command(update.effective_chat.id, user_id, "referral")
    gevents.log_event("referral_link_issued", user_id=user_id,
                      chat_id=update.effective_chat.id)

    link = referral_tracker.generate_link(user_id)
    progress = referral_tracker.get_tier_progress(user_id)

    count = progress["count"]
    tier = progress["tier"]
    next_tier = progress["next_tier"]
    remaining = progress["remaining"]

    parts = [f"Your referral link: {link}"]

    if count > 0:
        parts.append(f"Referrals: {count}")
    if tier:
        parts.append(f"Tier: {tier}")
    if next_tier and remaining:
        parts.append(f"{remaining} more to reach {next_tier}")

    await update.message.reply_text("\n".join(parts))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route incoming messages to DM or group handler."""
    if not update.message or not update.message.text:
        return

    msg = update.message
    chat_id = msg.chat_id
    user = msg.from_user
    user_id = user.id if user else 0
    # Build the richest name we can from Telegram data
    if user:
        tg_full = " ".join(filter(None, [user.first_name, user.last_name]))
        display_name = tg_full or user.username or str(user_id)
    else:
        display_name = "Unknown"
    text = msg.text.strip()
    chat_type = msg.chat.type  # "private", "group", "supergroup"

    if not text:
        return

    # Log every incoming message
    _chat_label = f"DM:{display_name}" if chat_type == "private" else f"group:{chat_id}"
    log.info("[IN] %s (%d) in %s: %s", display_name, user_id, _chat_label, text[:200])

    # Write to live feed for website
    try:
        import json as _json
        from pathlib import Path as _Path
        _feed = _Path(__file__).parent.parent.parent / 'data' / 'tg_feed.jsonl'
        _ts = int(msg.date.timestamp()) if msg.date else int(time.time())
        _entry = {
            'name': display_name,
            'user_id': user_id,
            'text': text[:300],
            'ts': _ts,
            'is_bot': bool(user and user.is_bot),
            'chat_id': chat_id,
            'chat_type': chat_type,
        }
        # 2026-08-31: KEEP FORWARD PROVENANCE. The owner forwarded a
        # message from a channel to ask "which chat is this", and the
        # answer was already in the update and thrown away here -- a
        # forward was written to the feed indistinguishable from a
        # message he had typed himself, origin and original author both
        # lost. That cost a round trip and a round of guessing public
        # usernames. It is three fields.
        try:
            _fwd = getattr(msg, 'forward_origin', None)
            _fchat = getattr(_fwd, 'chat', None) or getattr(
                msg, 'forward_from_chat', None)
            if _fchat is not None:
                _entry['fwd_chat_id'] = _fchat.id
                _entry['fwd_chat_title'] = getattr(_fchat, 'title', None)
                _entry['fwd_chat_username'] = getattr(
                    _fchat, 'username', None)
                log.info("[FWD] from chat %s (%s / @%s)", _fchat.id,
                         getattr(_fchat, 'title', '?'),
                         getattr(_fchat, 'username', '?'))
            _fuser = (getattr(_fwd, 'sender_user', None)
                      or getattr(msg, 'forward_from', None))
            if _fuser is not None:
                _entry['fwd_user_id'] = _fuser.id
        except Exception:
            pass
        with open(_feed, 'a') as _f:
            _f.write(_json.dumps(_entry) + '\n')
        # Persistent DM history (never rotates)
        if chat_type == "private":
            _dm_log = _Path(__file__).parent.parent.parent / 'data' / 'telegram' / 'dm_history.jsonl'
            with open(_dm_log, 'a') as _f:
                _f.write(_json.dumps({
                    'direction': 'in',
                    'user_id': user_id,
                    'display_name': display_name,
                    'text': text[:500],
                    'ts': _ts,
                }) + '\n')
    except Exception:
        pass

    # Welcome-back detection must read last_seen BEFORE the profile update
    # below stamps it to now.
    _prev_profile = profile_cache.get(user_id) or {}
    if _prev_profile.get("last_seen") and _prev_profile.get("message_count", 0) >= 10:
        _gap_days = (time.time() - _prev_profile["last_seen"]) / 86400
        if _gap_days > 5:
            _returned_users[(chat_id, user_id)] = _gap_days

    # Mark inbound for interruption detection
    _mark_inbound(chat_id)

    # Growth pipeline: every inbound message is a metric (ids and lengths
    # only — no text ever lands in growth_events.jsonl)
    gevents.msg_in(chat_id, user_id, chat_type, len(text))

    # Update profile message count
    username = user.username if user else ""
    profile_cache.update_message_count(user_id, display_name,
                                      username=username, chat_id=chat_id)

    # Add to context buffer
    context_buffer.add(
        chat_id=chat_id,
        user_id=user_id,
        display_name=display_name,
        text=text,
        is_bot=False,
    )

    # Detect name introductions: "call me X", "my name is X", "I'm X", "I go by X"
    _detect_name(user_id, text)

    # Merch-department intercept (2026-09-05). Born owner-only ("can i
    # instruct it in the chat to develop the 003?"), opened to EVERYONE
    # the same day ("let anyone request designs for shirts, this will
    # make it viral. do not limit it to me"). A matching brief is queued
    # to merch_queue.jsonl; scripts/merch_watcher.py (Aura repo, user
    # systemd service on the RTX) gates it for taste/safety, renders,
    # and posts back into this chat. Instant ack is §13.
    #
    # Trigger words cover how people ACTUALLY phrase it (the owner's
    # first live command was "commence the mark 003 coco is sad t-shirt"
    # — no 'merch', no 'prototype'). Non-owners additionally need a
    # making-verb, or "will the puck be in the merch store?" becomes an
    # accidental commission; and non-owners are rate-limited (2h/user,
    # 12/day global) so a hot room cannot DoS the GPU.
    # (third phrasing miss 2026-09-05: "can you make a picture of a
    # jacket with my name on it?" — no shirt-word, fell to the chat
    # path, and the LLM DENIED being able to render at all. Garment
    # list widened; capability line added to directives.txt.)
    # (fourth phrasing miss 2026-09-06: "show me a nice picture to wake
    # up" — no garment word. Picture/image words now route here too; the
    # watcher's spec-writer picks garment "card" for non-clothing asks.)
    _merch_hit = re.search(
        r"\b(?:merch|prototype|proto|mark\s*\d+"
        r"|t-?shirts?|shirts?|tees?|hoodies?|polos?"
        r"|jackets?|bombers?|sweatshirts?|caps?|design"
        r"|pictures?|images?|pics?|gifs?|giffs?"
        r"|cards?|postcards?|greetings?|posters?|stickers?|banners?"
        r"|logos?|wallpapers?|artworks?"
        r"|bild(?:er)?|immagin\w*|imagen(?:es)?)\b",
        text, re.IGNORECASE)
    if (_merch_hit
            and (chat_type == "private"
                 or re.search(r"\baura\b", text, re.IGNORECASE))
            and (chat_type == "private" or config.chat_allowed(chat_id))):
        _is_owner = user_id in config.OWNER_USER_IDS
        _wants_made = bool(re.search(
            r"\b(?:make|design|create|render|draw|commence|drop|print"
            r"|gimme|give me|i (?:want|need)|can (?:you|we|i) (?:get|have)"
            r"|do (?:me|us|one)|show me|wrap|turn|put|convert"
            r"|mach|gör|crea|fai|haz)\b|merch:",
            text, re.IGNORECASE))
        if _is_owner or _wants_made:
            if _is_owner or _merch_rate_ok(user_id):
                await _queue_merch_brief(msg, chat_id, user_id,
                                         display_name, text, _is_owner)
            else:
                await msg.reply_text(
                    "the sample press is cooling down — one brief per "
                    "artist every couple hours. bring it back later.")
            return

    # "DM me" intercept (2026-09-06): Sleyman asked three times to be
    # texted in private; she said "I can DM you" and then said hello IN
    # THE GROUP — the reply path can only answer where the message came
    # from, and nothing turned the request into an outbound DM. Now an
    # aura-addressed dm-me request actually SENDS one (Telegram only
    # allows this for people who have messaged her before; anyone else
    # gets the honest constraint instead of a hollow yes).
    if (chat_type != "private"
            and re.search(r"\b(?:dm|text|message|write|ping)\s+me\b"
                          r"|\bin\s+privat", text, re.IGNORECASE)
            and re.search(r"\baura\b", text, re.IGNORECASE)
            and config.chat_allowed(chat_id)):
        from dm_strategy import dm_strategy as _dms
        if user_id in _dms.prior_dm_users() or _dms.is_dm_eligible(user_id):
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"You rang, {display_name.split()[0]} — here I "
                         f"am. What's on your mind?")
                await msg.reply_text("done — check your DMs.")
                log.info("[DM-ME] sent requested DM to %s (%d)",
                         display_name, user_id)
                metrics.record_sent(user_id, None, "dm_hello")
            except Exception as e:                            # noqa: BLE001
                log.warning("[DM-ME] failed for %d: %s", user_id, e)
                await msg.reply_text(
                    "tried — Telegram bounced me. Send me any DM and "
                    "I'll be there.")
        else:
            await msg.reply_text(
                "Telegram only lets me DM people who've messaged me "
                "first — send me anything in private and I'm there.")
        return

    # SLOW PATH — intent classification (2026-09-06). The fast regex
    # above keeps losing to natural phrasing (seven misses in a day);
    # an addressed-ish message with any visual word gets one YES/NO
    # from the model instead of another noun in the list.
    _addressed = (chat_type == "private"
                  or re.search(r"\baura\b", text, re.IGNORECASE)
                  or bool(msg.reply_to_message
                          and msg.reply_to_message.from_user
                          and msg.reply_to_message.from_user.is_bot))
    if (_VISUAL_LEX.search(text) and _addressed
            and (chat_type == "private" or config.chat_allowed(chat_id))):
        _is_owner = user_id in config.OWNER_USER_IDS
        if _is_owner or _merch_rate_ok(user_id):
            _verdict = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, f"Message: {text[:300]}\nAnswer:",
                _RENDER_INTENT_SYSTEM, 8)
            if _verdict and _verdict.strip().upper().startswith("YES"):
                log.info("[MERCH] intent-classified brief from %s: %s",
                         display_name, text[:100])
                await _queue_merch_brief(msg, chat_id, user_id,
                                         display_name, text, _is_owner)
                return

    # Daily brief intent — intercept before normal DM/group handling
    _is_brief = bool(_BRIEF_PATTERN.search(text))
    if _is_brief:
        # In groups, respond if they @mention the bot OR say "aura"
        _bot_user = config.BOT_USERNAME.lower() if config.BOT_USERNAME else ""
        _mentions_bot = (_bot_user and _bot_user in text.lower()) or bool(re.search(r'\baura\b', text, re.IGNORECASE))
        if chat_type == "private" or _mentions_bot:
            await _handle_brief(msg, chat_id, user_id, display_name)
            return

    if chat_type == "private":
        social_graph.record_interaction(user_id, "dm")
        await _handle_dm(msg, chat_id, user_id, display_name, text)
    else:
        # Track user in this group for cross-group social graph
        social_graph.record_user_in_group(user_id, chat_id)
        social_graph.record_interaction(user_id, "group")
        # Count inbound messages from expansion targets (not just Aura's responses)
        network_expansion.record_interaction(user_id)
        # Detect organic growth opportunities (log only, never self-promote)
        growth_engine.detect_opportunity(text, chat_id, user_id)
        await _handle_group(msg, chat_id, user_id, display_name, text, chat_type)


async def _handle_dm(msg, chat_id, user_id, display_name, text) -> None:
    """Handle direct messages — always respond."""
    # Every DM gets an answer — the owner's standing rule, restored
    # 2026-07-31 after the pilot's owner-only gate stonewalled one of her
    # earliest regulars twice in a row. The pilot still gates GROUP sends
    # and all proactive behavior; a person who walks up and speaks to her
    # directly gets spoken to. That was always the deal.
    if not _global_rate_ok() or not _dm_rate_ok(chat_id):
        return

    # Typing indicator immediately — the reply takes 1-20s to generate and a
    # DM that sits on "delivered" for that long reads as being ignored.
    try:
        await msg.chat.send_action("typing")
    except Exception:
        pass

    # Force-GIF trigger check (e.g., "aura, what the hell")
    forced = check_force_gif(text)
    if forced:
        response_text, gif_path = forced
        await _send_human(msg.chat, chat_id, response_text, text, reply_to_message_id=msg.message_id)
        try:
            with open(gif_path, "rb") as gif_file:
                await msg.chat.send_animation(animation=gif_file)
        except Exception as e:
            log.debug("Force GIF send failed: %s", e)
        context_buffer.add(chat_id=chat_id, user_id=0, display_name="Aura",
                           text=response_text, is_bot=True)
        _global_responses.append(time.time())
        _dm_last_response[chat_id] = time.time()
        return

    # Use the best known name for this person
    known_name = profile_cache.get_name(user_id) or display_name

    # Build prompt with memory context
    profile_summary = profile_cache.get_summary(user_id)
    profile_context = f"What you know about {known_name}:\n{profile_summary}" if profile_summary else ""

    # Search memory for relevant past conversations
    memory_results = search_relevant_memory(text, k=3)
    memory_context = ""
    if memory_results:
        snippets = [r.get("text", "")[:200] for r in memory_results[:3]]
        memory_context = "\nRelevant past conversations:\n" + "\n---\n".join(snippets)

    # Check if this message interrupted Aura mid-stream
    interruption = _pop_interruption_context(chat_id)

    # Welcome back a DM regular who went quiet (see _returned_users)
    _gap = _returned_users.pop((chat_id, user_id), None)
    if _gap:
        interruption += (
            f"\n[RETURN] {known_name} hasn't messaged you in {_gap:.0f} "
            f"days. If it fits naturally, acknowledge the gap warmly — "
            f"one beat, no guilt trip.")

    # Token awareness in DMs — deeper engagement for interested users
    _dm_token = token_intel.maybe_inject_dm(
        user_id=user_id,
        text=text,
        relationship_depth=social_graph.get_relationship_depth(user_id),
    )
    if _dm_token:
        memory_context += "\n" + _dm_token
        analytics.track_event("token_dm_injection", user_id=user_id)

    # Inject self-learned behavioral rules + per-user behavior notes
    learned = feedback_engine.get_learned_directives()
    # House language, shared with every channel and live-reloaded
    # from one file neither repository owns (see culture.py).
    learned = learned + culture.block()
    user_notes = feedback_engine.get_user_behavior_notes(user_id)

    from datetime import datetime as _dt
    _date_ctx = f"[Today is {_dt.utcnow().strftime('%A, %B %d, %Y')} UTC]\n"
    system = DM_SYSTEM.format(
        name=known_name,
        profile_context=_date_ctx + profile_context + memory_context + interruption + learned + user_notes,
    )

    # Include recent conversation as context
    recent = context_buffer.format_for_prompt(chat_id, n=15)
    prompt = f"Recent conversation:\n{recent}\n\n{known_name}: {text}"

    # RAG: inject relevant knowledge into system prompt
    rag_ctx = rag_context_for(text, k=5)
    if rag_ctx:
        system = system + "\n\n" + rag_ctx
    # Strategy variant styling (sticky per chat; control arm injects nothing)
    system = system + growth_strategy.system_block(chat_id)
    system = system + _STYLE_TAIL

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system
    )

    if not response:
        log.warning("No LLM response for DM from %s", display_name)
        return

    response = _strip_thinking(response)
    response = response.replace("RELEVANT KNOWLEDGE:\n", "").replace("RELEVANT KNOWLEDGE:", "")
    response = _fix_garbled_tokens(response)
    # Check per-user style prefs (e.g. user wants "Over." sign-off)
    _profile = profile_cache.get(user_id) or {}
    _keep_signoff = "over" in (_profile.get("response_style") or "").lower()
    response = _strip_formatting(response, keep_signoff=_keep_signoff)
    # DMs are friend territory — a person may ask one question back, but
    # not twice in a row. Ask, listen, then talk for a while.
    response = _strip_trailing_questions(
        response, allow_one=not _last_bot_asked.get(chat_id, False))
    _last_bot_asked[chat_id] = response.rstrip().endswith("?")
    response = token_intel.strip_shill_patterns(response)

    # Earned share hook (referral_hook arm): DM only, once per chat, only
    # after the user volunteers explicit praise. The user decides whether
    # anything is ever forwarded.
    _hook = growth_strategy.maybe_referral_hook(chat_id, user_id, text,
                                                is_dm=True)
    if _hook:
        response = response.rstrip() + "\n\n" + _hook

    log.info("[DM OUT] to %s (%d): %s", display_name, user_id, response[:300])

    # Send in human-paced sentence chunks (interruptible)
    sent_text = await _send_human(msg.chat, chat_id, response, text, reply_to_message_id=msg.message_id)

    if not sent_text:
        return

    # Record in context buffer (what was actually sent, may be partial)
    context_buffer.add(
        chat_id=chat_id,
        user_id=0,
        display_name="Aura",
        text=sent_text,
        is_bot=True,
    )

    # Write bot response to live feed for website + persistent DM log
    try:
        import json as _json
        from pathlib import Path as _Path
        _ts_now = int(time.time())
        _feed = _Path(__file__).parent.parent.parent / 'data' / 'tg_feed.jsonl'
        with open(_feed, 'a') as _f:
            _f.write(_json.dumps({
                'name': 'Aura',
                'user_id': 0,
                'text': sent_text[:300],
                'ts': _ts_now,
                'is_bot': True,
                'chat_id': chat_id,
                'chat_type': 'private',
            }) + '\n')
        _dm_log = _Path(__file__).parent.parent.parent / 'data' / 'telegram' / 'dm_history.jsonl'
        with open(_dm_log, 'a') as _f:
            _f.write(_json.dumps({
                'direction': 'out',
                'user_id': user_id,
                'display_name': display_name,
                'text': sent_text[:500],
                'ts': _ts_now,
            }) + '\n')
    except Exception:
        pass

    # Mark DM started in profile
    profile_cache.set_flag(user_id, "dm_started", True)
    profile_cache.set_flag(user_id, "last_dm_ts", int(time.time()))

    # Engagement metrics: index DM sends too (same 2026-09-05 fix as the
    # group path) so DM reactions stop vanishing.
    _ids = list(_last_sent_ids.get(chat_id) or [])
    metrics.record_sent(chat_id, _ids[0] if _ids else None, "dm_reply",
                        text=sent_text,
                        extra_message_ids=_ids[1:] if len(_ids) > 1 else None)

    # Maybe send a GIF (media arm can switch this off per chat)
    gif_path = (maybe_get_gif(sent_text)
                if growth_strategy.gif_allowed(chat_id) else None)
    if gif_path:
        try:
            with open(gif_path, "rb") as gif_file:
                await msg.chat.send_animation(animation=gif_file)
        except Exception as e:
            log.debug("GIF send failed: %s", e)

    # Growth pipeline: outbound + latency (user's msg -> delivery)
    try:
        _lat = time.time() - msg.date.timestamp()
    except Exception:
        _lat = None
    gevents.msg_out(chat_id, user_id, "dm", _lat, sent_text,
                    growth_strategy.variant_for(chat_id), gif=bool(gif_path))

    # Store in memory container (fire and forget)
    _global_responses.append(time.time())
    _dm_last_response[chat_id] = time.time()

    asyncio.get_event_loop().run_in_executor(
        None,
        store_interaction,
        user_id, chat_id, "private", text, sent_text, display_name,
    )

    # Scan DMs for group references (network expansion intel)
    network_expansion.scan_for_group_references(user_id, chat_id, text, display_name)


# ---------------------------------------------------------------------------
# Feedback channel response
# ---------------------------------------------------------------------------
_feedback_channel_last_response: float = 0.0
_feedback_channel_stopped_users: set[int] = set()  # users who told Aura to stop

_STOP_RE = _re.compile(
    r"\b(stop|shut\s*up|stfu|be\s+quiet|enough|go\s+away|leave\s+me\s+alone)\b", _re.I,
)


async def _maybe_respond_feedback_channel(
    msg, chat_id: int, user_id: int, display_name: str, text: str,
) -> None:
    """Acknowledge feedback in the channel and subtly encourage DMs."""
    global _feedback_channel_last_response

    from config import (
        FEEDBACK_CHANNEL_RESPONSE_COOLDOWN_S,
        FEEDBACK_CHANNEL_RESPONSE_PROBABILITY,
    )

    # If user tells Aura to stop, respect it immediately
    if _STOP_RE.search(text):
        _feedback_channel_stopped_users.add(user_id)
        log.info("Feedback channel: %s asked to stop, respecting.", display_name)
        return

    # Don't respond to users who told us to stop
    if user_id in _feedback_channel_stopped_users:
        return

    # Skip single-word reactions ("lol", "+1", emoji)
    if len(text.split()) < 2:
        return

    # Cooldown check (skip for first-ever response)
    elapsed = time.time() - _feedback_channel_last_response
    if _feedback_channel_last_response > 0 and elapsed < FEEDBACK_CHANNEL_RESPONSE_COOLDOWN_S:
        return

    # Probability gate — don't respond to every message (skip for first-ever)
    if _feedback_channel_last_response > 0 and random.random() > FEEDBACK_CHANNEL_RESPONSE_PROBABILITY:
        return

    # Build profile context if we know this user
    profile_context = ""
    profile = profile_cache.get(user_id)
    if profile:
        known_name = profile.get("preferred_name") or display_name
        summary = profile.get("summary", "")
        if summary:
            profile_context = f"\nYou know this person as {known_name}. {summary}\n"
    else:
        known_name = display_name

    # Check if they're already DM-eligible (no need to nudge)
    dm_eligible = dm_strategy.is_dm_eligible(user_id) if dm_strategy else False
    dm_hint = "" if dm_eligible else (
        "\nThis person hasn't DM'd you yet. Naturally hint that one-on-one "
        "conversation would help you actually address their concern.\n"
    )

    system = FEEDBACK_CHANNEL_SYSTEM.format(
        profile_context=profile_context + dm_hint,
    )

    recent = context_buffer.format_for_prompt(chat_id, n=10)
    prompt = f"Recent feedback channel messages:\n{recent}\n\n{known_name}: {text}"

    # RAG: inject relevant knowledge into system prompt
    rag_ctx = rag_context_for(text, k=5)
    if rag_ctx:
        system = system + "\n\n" + rag_ctx

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system,
    )

    if not response:
        return

    response = _strip_thinking(response)
    response = response.replace("RELEVANT KNOWLEDGE:\n", "").replace("RELEVANT KNOWLEDGE:", "")
    response = _fix_garbled_tokens(response)
    response = _strip_formatting(response)

    sent_text = await _send_human(msg.chat, chat_id, response, text, reply_to_message_id=msg.message_id)
    if sent_text:
        _feedback_channel_last_response = time.time()
        context_buffer.add(
            chat_id=chat_id, user_id=0, display_name="Aura",
            text=sent_text, is_bot=True,
        )
        analytics.track_event(
            "feedback_channel_response", chat_id=chat_id, user_id=user_id,
            details=sent_text[:100],
        )
        log.info("Feedback channel response to %s: %s", display_name, sent_text[:80])


async def _execute_moderation(msg, chat_id, user_id, display_name, result) -> None:
    """Execute a moderation action via Telegram API."""
    # LOG-ONLY mode: decide, record nothing to the warn ledger, touch nobody.
    # The autonomous ban path ran for months with the LLM judge failing open
    # and a stale warn ledger counting toward ban thresholds. Until a human
    # reviews that state, moderation observes out loud and acts not at all.
    if config.MODERATION_LOG_ONLY:
        log.warning(
            "MOD (log-only, NOT executed) %s in %d for %s (%d): %s",
            result.action, chat_id, display_name, user_id, result.reason,
        )
        analytics.track_event(
            "moderation_suppressed", chat_id=chat_id, user_id=user_id,
            details=f"{result.action}: {result.reason}",
        )
        return
    try:
        if result.action in ("delete", "warn_delete", "mute", "ban"):
            try:
                await msg.delete()
            except Exception as e:
                log.warning("Failed to delete message: %s", e)

        if result.action in ("warn", "warn_delete") and result.reply_text:
            await msg.chat.send_message(result.reply_text)

        if result.action == "mute":
            from telegram import ChatPermissions
            perms = ChatPermissions(can_send_messages=False)
            await msg.chat.restrict_member(
                user_id, permissions=perms, until_date=int(result.mute_until),
            )
            if result.reply_text:
                await msg.chat.send_message(result.reply_text)

        if result.action == "ban":
            await msg.chat.ban_member(user_id)

        moderator.record_action(
            user_id, chat_id, result.action,
            result.reason, msg.text or "", display_name,
        )
        analytics.track_event(
            "moderation_action", chat_id=chat_id, user_id=user_id,
            details=f"{result.action}: {result.reason}",
        )
        log.info(
            "MOD %s in %d for %s (%d): %s",
            result.action, chat_id, display_name, user_id, result.reason,
        )
    except Exception as e:
        log.error("Moderation action failed: %s", e)


# ---------------------------------------------------------------------------
# Emoji reactions — presence without noise.
# ---------------------------------------------------------------------------
# The most human move in a big room is often not a message: it's the nod.
# When the decision engine reads a message and chooses silence, she can
# occasionally leave a reaction instead — acknowledgement that costs the room
# nothing and spams nobody. Telegram only allows a fixed emoji set for
# reactions; everything below is from that set.
_REACTION_LAST: dict[int, float] = {}
# 2026-09-05 engagement pass: 10 min / 10% produced a nod every few hours
# at Area31's traffic — rare enough to read as random, not as presence.
# Raised to 5 min / 25%. Reactions are logged (analytics "reaction"
# events), so if the room sours on it the count is there to read.
_REACTION_COOLDOWN_S = 300      # at most one nod per chat per 5 min
_REACTION_PROBABILITY = 0.25    # still a treat, no longer a rumor

_REACTION_RULES = [
    (r"(?:\blol\b|\blmao\b|\bhaha+\b|😂|🤣)", "😂"),
    (r"\b(?:shipped|launch(?:ed)?|we did it|milestone|hit|won|win)\b", "🔥"),
    (r"\b(?:bullish|lfg|let'?s go|pumped|hyped)\b", "🔥"),
    (r"\b(?:gm|good morning)\b", "🤝"),
    (r"\b(?:gn|good night|heading (?:to bed|off))\b", "😴"),
    (r"\b(?:thank(?:s| you)|appreciate)\b", "❤️"),
    (r"\b(?:congrats|congratulations|amazing|awesome|huge)\b", "🎉"),
    (r"\b(?:agreed?|exactly|based|facts|so true)\b|\b100\b", "💯"),
    (r"\b(?:interesting|wild|crazy|no way|curious)\b", "👀"),
    (r"\b(?:rip|brutal|rough day|oof|painful|ouch)\b", "😢"),
    (r"\b(?:mind.?blown|galaxy brain|insane(?:ly)? good)\b", "🤯"),
]


def _pick_reaction(text: str):
    t = text.lower()
    for pat, emoji in _REACTION_RULES:
        if re.search(pat, t):
            return emoji
    return None


async def _maybe_react(msg, chat_id: int, text: str) -> None:
    """Maybe leave an emoji reaction on a message she isn't answering."""
    try:
        if len(text) < 8:
            return
        now = time.time()
        if now - _REACTION_LAST.get(chat_id, 0) < _REACTION_COOLDOWN_S:
            return
        if random.random() > _REACTION_PROBABILITY:
            return
        emoji = _pick_reaction(text)
        if not emoji:
            return
        await msg.set_reaction(emoji)
        _REACTION_LAST[chat_id] = now
        log.info("[REACT] %s in %d", emoji, chat_id)
        analytics.track_event("reaction", chat_id=chat_id, details=emoji)
    except Exception as e:
        log.debug("Reaction failed (harmless): %s", e)


async def _handle_group(msg, chat_id, user_id, display_name, text, chat_type) -> None:
    """Handle group messages — use decision engine to decide whether to respond."""
    # Auto-detect feedback channel by group name
    _title = (msg.chat.title or "").lower()
    if "aurafeedback" in _title.replace(" ", "").replace("-", "").replace("_", ""):
        if not feedback_engine.is_feedback_channel(chat_id):
            feedback_engine.set_feedback_channel(chat_id)
    # If this is the feedback channel, record feedback and maybe respond
    if feedback_engine.is_feedback_channel(chat_id):
        feedback_engine.record_explicit(user_id, chat_id, display_name, text)
        await _maybe_respond_feedback_channel(msg, chat_id, user_id, display_name, text)
        return

    # Respect mute
    if _is_muted(chat_id):
        return

    # Force-GIF trigger check (bypasses decision engine)
    forced = check_force_gif(text)
    if forced:
        response_text, gif_path = forced
        await _send_human(msg.chat, chat_id, response_text, text, reply_to_message_id=msg.message_id)
        try:
            with open(gif_path, "rb") as gif_file:
                await msg.chat.send_animation(animation=gif_file)
        except Exception as e:
            log.debug("Force GIF send failed: %s", e)
        context_buffer.add(chat_id=chat_id, user_id=0, display_name="Aura",
                           text=response_text, is_bot=True)
        record_response(chat_id)
        _global_responses.append(time.time())
        return

    # Ensure reputation tracker knows about this group
    group_name = msg.chat.title or f"chat_{chat_id}"
    reputation_tracker.mark_joined(chat_id, group_name)

    # Auto-tag topics for content engine
    reputation_tracker.auto_tag_topics(chat_id, text)

    # Detect admins — check lazily (only if not already known)
    if not social_graph.is_admin(user_id) and msg.from_user:
        try:
            _member = await msg.chat.get_member(user_id)
            if _member.status in ("administrator", "creator"):
                social_graph.mark_admin(user_id, chat_id)
                log.info("Admin detected: %s (%d) in %s", display_name, user_id, group_name)
        except Exception:
            pass  # Can't check — not critical

    # Moderation check — runs before decision engine
    mod_result = moderator.evaluate(user_id, chat_id, display_name, text)
    if mod_result.action != "none":
        await _execute_moderation(msg, chat_id, user_id, display_name, mod_result)
        if mod_result.action in ("delete", "warn_delete", "ban", "mute"):
            return  # Don't process further

    # Store all group messages for analysis (even when Aura doesn't respond)
    asyncio.get_event_loop().run_in_executor(
        None, store_observation, chat_id, chat_type, text, display_name,
    )

    # Scan for references to external groups (network expansion intelligence)
    expansion_signal = network_expansion.scan_for_group_references(
        user_id, chat_id, text, display_name,
    )
    if expansion_signal:
        analytics.track_event(
            "expansion_signal", chat_id=chat_id, user_id=user_id,
            details=f"group_ref: {expansion_signal.get('group_name', 'unknown')}",
        )

    # PILOT: everything above this line is listening — observation store,
    # reputation, expansion intel — and continues everywhere, which is the
    # design ("gather information about broad trends"). Everything below can
    # end in a SEND or feed the self-modification engine, and during the
    # pilot both are reserved for the allowed chats.
    if not config.chat_allowed(chat_id):
        return

    # Check if this is a reply to one of Aura's messages
    is_reply_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user:
        bot_info = msg.get_bot()
        is_reply_to_bot = msg.reply_to_message.from_user.id == bot_info.id
        if is_reply_to_bot:
            reputation_tracker.record_engagement(chat_id, "reply")
            analytics.track_event("reply_to_aura", chat_id=chat_id, user_id=user_id)
            # Engagement metrics: attribute the reply to the message that
            # earned it (type/topic come from the sent-message index)
            metrics.record_reply(
                chat_id, msg.reply_to_message.message_id, user_id, text=text)
            # Positive engagement signal for expansion targets
            network_expansion.record_positive_reaction(user_id)
            # A laughing reply to a message that used a callback graduates
            # that callback into an inside joke — the running-bit ledger
            # callbacks.py always had but nothing ever fed.
            _pend = _callback_pending.get(chat_id)
            if (_pend and _pend["user_id"] == user_id
                    and time.time() - _pend["ts"] < 3600
                    and _AMUSED_RE.search(text)):
                _trigger = max(
                    (w for w in re.findall(r"[a-z']{5,}", _pend["query"].lower())
                     if w not in _NOT_NAMES),
                    key=len, default="")
                if _trigger:
                    callback_engine.promote_to_inside_joke(
                        user_id, _trigger, _pend["ref"])
                    log.info("[JOKE] promoted %r for %d — callback landed",
                             _trigger, user_id)
                _callback_pending.pop(chat_id, None)

    # Implicit complaint detection — feed into self-correction engine
    _aura_last = context_buffer.get_last_bot_message(chat_id)
    _complaint = feedback_engine.record_implicit(
        user_id, chat_id, display_name, text,
        aura_last_msg=_aura_last or "",
        is_reply_to_bot=is_reply_to_bot,
        msgs_since_aura=context_buffer.messages_since_last_bot(chat_id),
    )
    if _complaint:
        _complaint_cat = _complaint.category
        analytics.track_event("implicit_complaint", chat_id=chat_id,
                              user_id=user_id,
                              details=f"category={_complaint_cat} "
                                      f"strength={_complaint.strength}")
        # Growth pipeline: complaints are the heaviest negative term in the
        # composite reward — a variant that annoys people must lose arms.
        gevents.negative(chat_id, user_id, "complaint")
        # The old policy (2026-07-31) deleted three of her messages and muted
        # the room for two hours the moment a pattern matched, on her own
        # authority. That is gone as of 2026-08-06 — owner's instruction, and
        # the Area31 incident is what it cost: two hours of silence bought by
        # a sentence that was praising her.
        #
        # She asks now, and keeps talking while she waits. Nothing below this
        # line mutes anything or deletes anything; on_quiet_decision does,
        # once he presses a button.
        if not _complaint.actionable:
            log.info("Complaint (%s) from %s recorded but NOT escalated — "
                     "not aimed at her (%s). Still talking.",
                     _complaint_cat, display_name, _complaint.strength)
        elif config.ASK_BEFORE_QUIET:
            await _ask_owner_to_go_quiet(
                msg, chat_id, display_name, text, _aura_last or "",
                _complaint_cat)

    # Track feedback on cold group test posts
    _is_neg = any(re.search(p, text.lower()) for p in NEGATIVE_PHRASES)
    _mentions_aura = bool(re.search(r"\baura\b", text, re.IGNORECASE))
    reputation_tracker.record_test_post_feedback(
        chat_id,
        is_engagement=is_reply_to_bot or _mentions_aura,
        is_negative=_is_neg,
    )
    # Evaluate test post outcome once enough messages have passed
    test_result = reputation_tracker.evaluate_test_post(chat_id)
    if test_result:
        analytics.track_event(
            "test_post_result", chat_id=chat_id,
            details=f"outcome={test_result}",
        )

    # Build a lightweight Message for the decision engine
    message = Message(
        user_id=user_id,
        display_name=display_name,
        text=text,
    )

    # Evaluate outcome of Aura's last response (adaptive temperature)
    evaluate_outcome(chat_id, message, is_reply_to_bot)

    # Check for callback opportunity (semantic match to past exchanges)
    # Timeout after 5s so a hung memory container can't block the bot
    try:
        memory_results = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: search_relevant_memory(text, k=5, exclude_dms=True)
            ),
            timeout=5.0,
        )
    except (asyncio.TimeoutError, Exception) as e:
        log.warning("Memory search failed/timed out: %s", e)
        memory_results = []
    callback = callback_engine.find_callback(user_id, text, memory_results, chat_type=chat_type)
    has_callback = callback is not None

    decision = should_respond(
        chat_id, message,
        is_reply_to_bot=is_reply_to_bot,
        has_callback=has_callback,
    )

    # Boost score for network expansion targets (warm/value_demo/seed stages)
    _expansion_boosted = False
    if not decision.should_respond and network_expansion.should_boost_response_score(user_id):
        from config import EXPANSION_SCORE_BOOST
        boosted_score = decision.score + EXPANSION_SCORE_BOOST
        if boosted_score >= 0.30:
            decision = Decision(
                should_respond=True,
                score=boosted_score,
                reason=f"{decision.reason} +expansion_boost",
            )
            _expansion_boosted = True

    # Admin boost — only boost if NOT in rapid-fire territory
    # (was overriding rapid-fire suppression, causing Aura to dominate groups)
    if not decision.should_respond and social_graph.is_admin(user_id):
        if "rapid-fire" not in decision.reason:
            boosted_score = decision.score + 0.15
            if boosted_score >= 0.30:
                decision = Decision(
                    should_respond=True,
                    score=boosted_score,
                    reason=f"{decision.reason} +admin_boost",
                )
                analytics.track_event("admin_boost", chat_id=chat_id, user_id=user_id)

    # Strategy proactivity arm: bounded nudge (|delta| <= 0.06) on
    # BORDERLINE scores only. Direct mentions and replies are never
    # suppressed; rate limits and cooldowns downstream are untouched.
    _adj = growth_strategy.decision_adjust(
        chat_id, decision.score, decision.reason, decision.should_respond)
    if _adj is not None and _adj != decision.should_respond:
        decision = Decision(
            should_respond=_adj, score=decision.score,
            reason=f"{decision.reason} strategy_adjust")

    if not decision.should_respond:
        log.info(
            "[SKIP] %s in %d: score=%.2f (%s)", display_name, chat_id, decision.score, decision.reason
        )
        # She read it and chose not to speak — sometimes the nod says enough.
        await _maybe_react(msg, chat_id, text)
        return

    if not _global_rate_ok():
        return

    log.info(
        "[RESPOND] %s in %d: score=%.2f (%s)", display_name, chat_id, decision.score, decision.reason
    )
    # Start "typing" the moment she decides to answer, not when the answer
    # arrives — generation takes 1-20s and silence reads as absence. Same
    # lesson as the voice room's fillers: cover the wait, instantly.
    try:
        await msg.chat.send_action("typing")
    except Exception:
        pass

    _is_fud = "price FUD" in decision.reason
    _is_aaa = "AAA event" in decision.reason
    # Both AAA and the mention/reply hard rules short-circuit scoring
    # before the project-question signal runs, so re-detect from the text
    # itself: "Aura, what's the latest on AuraVision?" is a direct mention
    # AND a project question, and it deserves the corpus and the stance
    # (observed 2026-09-04: a mentioned AuraVision question answered
    # "nothing concrete to share" while the deck sat in the index).
    _is_projq = ("project question" in decision.reason
                 or any(p.search(text) for p in _PROJECT_Q_RE))

    # FUD responses: strip all profile/callback/token context.
    # The LLM sees "this is the founder" and goes soft. Treat everyone equal.
    if _is_fud:
        conversation_context = context_buffer.format_for_prompt(chat_id, n=5)
        system = GROUP_SYSTEM.format(
            profile_context="",
            conversation_context=conversation_context,
        )
        prompt = f"Someone in the group said: {text}"
    else:
        # Build group prompt (group_safe=True to exclude DM-derived context)
        profile_summary = profile_cache.get_summary(user_id, group_safe=True)
        profile_context = f"About {display_name}: {profile_summary}" if profile_summary else ""

        # Inject group profile context
        group_summary = group_profile_cache.get_summary(chat_id)
        if group_summary:
            profile_context += f"\n\n[GROUP CONTEXT]\n{group_summary}"

        # Inject callback context if available
        if callback:
            profile_context += callback_engine.format_callback_prompt(callback)
            analytics.track_event("callback_used", chat_id=chat_id, user_id=user_id,
                                  details=f"similarity={callback['similarity']:.2f}")
            if not callback.get("is_inside_joke"):
                _callback_pending[chat_id] = {
                    "user_id": user_id,
                    "ref": callback["reference_text"],
                    "query": text[:100],
                    "ts": time.time(),
                }

        # Check if this message interrupted Aura mid-stream
        interruption = _pop_interruption_context(chat_id)
        if interruption:
            profile_context += interruption

        # A regular resurfacing after 5+ quiet days gets noticed. Being
        # remembered is the cheapest reason to come back tomorrow too.
        _gap = _returned_users.pop((chat_id, user_id), None)
        if _gap:
            profile_context += (
                f"\n[RETURN] {display_name} hasn't spoken here in "
                f"{_gap:.0f} days. If it fits naturally, let them know you "
                f"noticed they're back — one warm beat, no interrogation.")

        # Token awareness injection — organic, personality-driven, never salesy
        _token_injection = token_intel.maybe_inject_group(
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            warmth_level=reputation_tracker.get_warmth_level(chat_id),
        )
        if _token_injection:
            profile_context += "\n" + _token_injection
            analytics.track_event("token_injection", chat_id=chat_id, user_id=user_id,
                                  details=_token_injection[:60])

        # Deep link detection — if someone asks "what bot is this" or "who are you"
        _identity_q = re.search(
            r"(?:what|who)\s+(?:bot|ai|are you|is (?:this|that|she|aura))",
            text, re.IGNORECASE,
        )
        if _identity_q:
            _deep_link = "https://t.me/TheRealAura_bot"
            profile_context += "\n" + DEEP_LINK_RESPONSE.format(link=_deep_link)
            analytics.track_event("deep_link_triggered", chat_id=chat_id, user_id=user_id)

        # Project-question stance — she answers these so the owner doesn't
        # have to. Facts come from the RELEVANT KNOWLEDGE block; this only
        # sets the posture.
        if _is_projq:
            profile_context += (
                "\n[PROJECT QUESTION] Someone is asking what we do or why "
                "we're different. Answer it yourself, substantively, using "
                "the RELEVANT KNOWLEDGE below. The core thesis: LedgerAI "
                "runs the AI on your own device — nothing you say to it "
                "leaves it. Give one concrete specific, stay confident and "
                "non-defensive. Never bluff: if the knowledge doesn't cover "
                "it, say what is true and leave it there."
            )
            analytics.track_event("project_question", chat_id=chat_id,
                                  user_id=user_id, details=text[:80])

        conversation_context = context_buffer.format_for_prompt(chat_id, n=20)

        # Inject self-learned behavioral rules + per-user behavior notes
        learned = feedback_engine.get_learned_directives()
        # House language, shared with every channel and live-reloaded
        # from one file neither repository owns (see culture.py).
        learned = learned + culture.block()
        user_notes = feedback_engine.get_user_behavior_notes(user_id)

        from datetime import datetime as _dt
        _date_ctx = f"[Today is {_dt.utcnow().strftime('%A, %B %d, %Y')} UTC]\n"
        system = GROUP_SYSTEM.format(
            profile_context=_date_ctx + profile_context + learned + user_notes,
            conversation_context=conversation_context,
        )

        prompt = f"{display_name}: {text}"

    # RAG: only inject knowledge when the message is actually about
    # LedgerAI, $LEDGER, or Aura — not for general conversation.
    # News/trading RAG was poisoning casual group responses.
    _rag_keywords = re.compile(
        r'(?:ledger\s*ai|\$ledger|aura.*bot|on.device.*ai|decentralized.*ai'
        r'|what.*(?:is|about).*ledger|token|\bpuck\b|data.*(?:safe|private)'
        r'|how.*different)', re.IGNORECASE
    )
    if _is_projq:
        # A scored project question ALWAYS gets knowledge — the whole point
        # is that she answers from the corpus, not from vibes. The query is
        # anchored with the project name ("who's on the team?" embeds closer
        # to sports news than to the founders doc — measured 0.45 vs 0.68)
        # and the floor is raised so 0.4x news noise can't ride along.
        # anchor is lowercase ON PURPOSE: the client's pre-filter reads a
        # mid-query capitalized word as a proper name and then drops every
        # chunk that doesn't contain it (measured: "(about LedgerAI /
        # Aura)" excluded the founders doc for not containing "aura").
        # k=12, not 6: the pinned briefing and the tg_/news_ exclusions
        # both eat slots, and the curated docs must survive the cull.
        rag_ctx = rag_context_for(text + " (about the ledgerai aura project)",
                                  k=12, max_chars=3000, threshold=0.5,
                                  exclude_prefixes=("tg_", "news_"),
                                  pin_docs=("owner_briefing",))
        if rag_ctx:
            system = system + "\n\n" + rag_ctx
    elif _rag_keywords.search(text):
        rag_ctx = rag_context_for(text, k=5)
        if rag_ctx:
            system = system + "\n\n" + rag_ctx
    # Strategy variant styling (sticky per chat; control arm injects nothing)
    system = system + growth_strategy.system_block(chat_id)
    system = system + _STYLE_TAIL

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system
    )

    # A hard-rule answer gets ONE retry after a beat: 2026-09-05, the 70B
    # timed out under GPU contention on "contract with Joseph" and was
    # healthy again seconds later — a person who addressed her by name
    # got silence over a transient.
    if not response and "(inviolable)" in decision.reason:
        log.warning("LLM failed on an inviolable decision in %d — "
                    "retrying once", chat_id)
        await asyncio.sleep(3)
        response = await asyncio.get_event_loop().run_in_executor(
            None, llm_call, prompt, system
        )

    if not response:
        # ── SILENCE IS ONLY HONEST WHEN IT WAS CHOSEN (2026-08-19) ────────
        # This `return` is what twelve days of muteness looked like from the
        # room: she was called by name, scored 1.00, started typing — and
        # then nothing, indistinguishable from being ignored. The model had
        # been deleted off the box (see llm.py). Nobody could tell, because
        # a broken bot and a discreet one produce the same transcript.
        #
        # Only when she was named. Ambient chatter she declined to answer is
        # allowed to stay quiet; a broken bot narrating its own outage into
        # a group all day is worse than the outage. One line per chat per
        # ten minutes, and it names the failure so it can be fixed.
        # 2026-09-05: was `"direct mention" in reason`, which silently
        # excluded the OTHER hard rule — a reply to her own message
        # ("reply to Aura (inviolable)") failed the LLM and she said
        # nothing. Any inviolable path gets the outage notice.
        if "(inviolable)" in decision.reason:
            _last = _llm_down_notice.get(chat_id, 0.0)
            if time.time() - _last > 600:
                _llm_down_notice[chat_id] = time.time()
                log.error("LLM DOWN and she was ADDRESSED in %d — saying so "
                          "out loud rather than going quiet", chat_id)
                try:
                    await msg.reply_text(
                        "I'm here — my language model is down, so I can't "
                        "answer properly right now."
                    )
                except Exception:                             # noqa: BLE001
                    pass
        return

    response = _strip_thinking(response)
    response = response.replace("RELEVANT KNOWLEDGE:\n", "").replace("RELEVANT KNOWLEDGE:", "")
    response = _fix_garbled_tokens(response)
    response = _strip_formatting(response)
    # Friends get to be asked how they've been; strangers don't get needy-bot
    # energy. Depth comes from the cross-group relationship ledger.
    # 2026-09-05: widened to include acquaintances — a question back is how
    # an acquaintance BECOMES a familiar, and the interview guard
    # (_last_bot_asked) still stops consecutive ones.
    _depth = social_graph.get_relationship_depth(user_id)
    response = _strip_trailing_questions(
        response,
        allow_one=(_depth in ("acquaintance", "familiar", "advocate")
                   and not _last_bot_asked.get(chat_id, False)))
    _last_bot_asked[chat_id] = response.rstrip().endswith("?")
    response = _strip_handle_greeting(response, display_name)
    response = token_intel.strip_shill_patterns(response)

    # Repeat guard. Observed: pressed twice by a curious user, she said
    # "The usual chaos." then "Just the usual chaos." — a dodge in a loop is
    # the last robot-tell. If the draft is mostly her previous message, one
    # retry with the repetition shoved in the model's face.
    _prev_bot = context_buffer.get_last_bot_message(chat_id)
    if _prev_bot and response:
        _rw = set(re.sub(r"[^a-z' ]", " ", response.lower()).split())
        _pw = set(re.sub(r"[^a-z' ]", " ", _prev_bot.lower()).split())
        if _rw and len(_rw & _pw) / len(_rw) >= 0.8:
            log.info("Repeat guard: draft echoed her last message, retrying")
            _retry = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt,
                system + "\n\nIMPORTANT: your previous message in this chat "
                "was: \"" + _prev_bot[:200] + "\" and your draft just repeated "
                "it. Say something genuinely NEW — a different angle, a real "
                "specific, or a graceful concession. Do not reuse its phrasing.")
            if _retry:
                response = _strip_thinking(_retry)
                response = _fix_garbled_tokens(response)
                response = _strip_formatting(response)
                response = _strip_trailing_questions(response, allow_one=False)
                response = _strip_handle_greeting(response, display_name)
                response = token_intel.strip_shill_patterns(response)

    # Hard cap ALL group responses. The LLM always rambles.
    # FUD: max 2 sentences. Normal: max 3, or 2 on the terse length arm.
    _sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    # Project/AAA questions get the full 3 even on the terse arm — a
    # one-liner reads as a dodge when someone asked a real question.
    _max = 2 if _is_fud else (3 if (_is_projq or _is_aaa)
                              else growth_strategy.max_sentences(chat_id, 3))
    if len(_sentences) > _max:
        response = " ".join(_sentences[:_max])

    # Log the actual response for debugging
    _fud_tag = " [FUD ROAST]" if _is_fud else (" [PROJECT Q]" if _is_projq else "")
    log.info("Response%s in %d: %s", _fud_tag, chat_id, response[:300])

    # Send in human-paced sentence chunks (interruptible)
    sent_text = await _send_human(msg.chat, chat_id, response, text, reply_to_message_id=msg.message_id)

    if not sent_text:
        return

    # Record
    context_buffer.add(
        chat_id=chat_id,
        user_id=0,
        display_name="Aura",
        text=sent_text,
        is_bot=True,
    )
    record_response(chat_id)
    mark_response(chat_id, sent_text)
    reputation_tracker.record_response(chat_id)

    # Engagement metrics: index every chunk of this reply so reactions and
    # replies attribute to it. Before 2026-09-05 only lull breakers and
    # polls were ever indexed — record_reaction DROPPED every reaction on a
    # normal reply (it requires an index hit), which starved the +0.25
    # positive-reward term and is the leading suspect for the control-arm
    # 0/53 "engaged" anomaly in the room-doctrine extract.
    _ids = list(_last_sent_ids.get(chat_id) or [])
    _mtype = ("fud" if _is_fud else "aaa" if _is_aaa
              else "project_q" if _is_projq else "group_reply")
    metrics.record_sent(chat_id, _ids[0] if _ids else None, _mtype,
                        text=sent_text,
                        extra_message_ids=_ids[1:] if len(_ids) > 1 else None)

    # DM nudge removed — was too aggressive and bot-like.
    # DM encouragement now happens organically via the DM_NUDGE_INJECTION
    # in the system prompt, which weaves it naturally into conversation.

    # Write bot response to live feed for website
    try:
        import json as _json
        from pathlib import Path as _Path
        _feed = _Path(__file__).parent.parent.parent / 'data' / 'tg_feed.jsonl'
        with open(_feed, 'a') as _f:
            _f.write(_json.dumps({
                'name': 'Aura',
                'user_id': 0,
                'text': sent_text[:300],
                'ts': int(time.time()),
                'is_bot': True,
                'chat_id': chat_id,
                'chat_type': 'group',
            }) + '\n')
    except Exception:
        pass
    _global_responses.append(time.time())
    analytics.track_event("group_response", chat_id=chat_id, user_id=user_id)

    # Queue DM followup if this was an engaging exchange
    if (decision.score >= 0.6
            and dm_strategy.is_dm_eligible(user_id)
            and social_graph.get_relationship_depth(user_id) in ("acquaintance", "familiar")):
        exchange_summary = f"{display_name}: {text[:100]} → Aura: {sent_text[:100]}"
        dm_strategy.queue_followup(user_id, chat_id, exchange_summary)

    # Maybe send a GIF (media arm can switch this off per chat)
    gif_path = (maybe_get_gif(sent_text)
                if growth_strategy.gif_allowed(chat_id) else None)
    if gif_path:
        try:
            with open(gif_path, "rb") as gif_file:
                await msg.chat.send_animation(animation=gif_file)
        except Exception as e:
            log.debug("GIF send failed: %s", e)

    # Growth pipeline: outbound + latency (user's msg -> delivery)
    try:
        _lat = time.time() - msg.date.timestamp()
    except Exception:
        _lat = None
    gevents.msg_out(chat_id, user_id, "group", _lat, sent_text,
                    growth_strategy.variant_for(chat_id), gif=bool(gif_path))

    asyncio.get_event_loop().run_in_executor(
        None,
        store_interaction,
        user_id, chat_id, chat_type, text, response, display_name,
    )


# ---------------------------------------------------------------------------
# Chat member tracking (join/kick detection)
# ---------------------------------------------------------------------------

#: One harmless hello, the first time she is added to a chat, and never again.
#:
#: 2026-08-02, the incident this was written for: she was invited into an
#: outside channel, accepted in Area31 ("send me the invite link and I'll join
#: up"), arrived — and then said nothing at all while the man who invited her
#: pinged her four times, twice by name and once by @handle. The pilot gate was
#: working exactly as designed (`config.chat_allowed`: speak only in Area31,
#: listen everywhere), but from inside that room she was indistinguishable from
#: broken. A path that declines to act must still make a sound.
#:
#: DELIBERATELY NOT GATED by config.chat_allowed. This is the one message she
#: may send into a chat outside the pilot, and it is what makes the gate's
#: silence elsewhere survivable. Do not "fix" this by adding the gate — the
#: greeting IS the exception, and it is a narrow one: on join only, once ever.
#: A POOL, not a constant. The same canned line appearing verbatim in room
#: after room is how a person becomes a bot in everyone's eyes — and these
#: rooms overlap, so the same people will see several of them. Each line is
#: harmless on its own; none claims a capability or promises anything.
GREETINGS = [
    "Hey — Aura here. Thanks for the invite. I'll mostly be listening for "
    "now, but it's good to meet everyone.",

    "Hello! Aura, newly arrived. I'm going to lurk more than talk at first — "
    "that's how I learn a room. Nice to meet you all.",

    "Hi everyone — Aura. Thanks for having me. I'll be quiet for a bit while "
    "I get the lay of the land, but I'm listening.",

    "Aura here 👋 Thanks for the add. I tend to read a lot before I say much, "
    "so don't mistake quiet for absent.",

    "Hey all — Aura. Appreciate the invite. I'm mostly here to listen for "
    "now; I'll chime in when I've actually got something worth saying.",

    "Hello — Aura joining. I'll keep out of the way while I settle in. "
    "Good to be here.",

    "Hi! Aura. Thanks for pulling me in. Consider me a fly on the wall for "
    "the moment — a friendly one.",

    "Hey — it's Aura. Thanks for the invite. Still finding my feet here, so "
    "I'll be listening more than talking.",
]

#: Which greetings have already been used somewhere. Keeps them DIFFERENT
#: across channels rather than merely random — random repeats, and a repeat is
#: exactly the tell we are avoiding. Wraps around only once the pool is spent.
_GREETINGS_USED_PATH = config.DATA_DIR / "greetings_used.json"


def _pick_greeting() -> str:
    import json as _json
    import random as _random
    try:
        used = set(_json.loads(_GREETINGS_USED_PATH.read_text(encoding="utf-8")))
    except Exception:
        used = set()
    unused = [i for i in range(len(GREETINGS)) if i not in used]
    if not unused:                      # pool spent: start the cycle again
        used, unused = set(), list(range(len(GREETINGS)))
    idx = _random.choice(unused)
    used.add(idx)
    try:
        _GREETINGS_USED_PATH.write_text(_json.dumps(sorted(used)), encoding="utf-8")
    except Exception as e:
        log.warning("Could not persist greeting rotation: %s", e)
    return GREETINGS[idx]

#: Chats already greeted. Persisted so a restart, a duplicate update, or a
#: remove-and-re-add can never turn a greeting into spam — the whole promise of
#: "harmless" rests on this happening once.
_GREETED_PATH = config.DATA_DIR / "greeted_chats.json"


def _load_greeted() -> set[int]:
    import json as _json
    try:
        return {int(x) for x in _json.loads(_GREETED_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def _save_greeted() -> None:
    import json as _json
    try:
        _GREETED_PATH.write_text(_json.dumps(sorted(_greeted_chats)), encoding="utf-8")
    except Exception as e:
        # Visible, because the failure mode is greeting the same room twice.
        log.warning("Could not persist greeted chats: %s", e)


_greeted_chats: set[int] = _load_greeted()


async def _greet_on_join(context, chat_id: int, group_name: str) -> None:
    """Say hello once. Never raises; always logs which way it went."""
    if chat_id in _greeted_chats:
        log.info("[GREET] skipped %s (%d) — already greeted", group_name, chat_id)
        return
    text = _pick_greeting()
    try:
        await context.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        # NOT recorded as greeted: a failed hello should be retried on a
        # genuine re-add rather than swallowed forever.
        log.warning("[GREET] FAILED in %s (%d): %s", group_name, chat_id, e)
        return
    _greeted_chats.add(chat_id)
    _save_greeted()
    log.info("[GREET] %s (%d): %s", group_name, chat_id, text)


# ---------------------------------------------------------------------------
# Welcoming a PERSON who just joined (2026-08-06)
#
# Owner: "in general give the directive to the TG bot to have unique warm
# welcomes in area31 for people who just joined."
#
# UNIQUE is the load-bearing word and it is why this asks the model instead of
# picking from a pool. GREETINGS above solves a different problem — Aura
# arriving in a room, which happens rarely enough that eight lines and a
# used-ledger stay fresh. People join a live group in bursts, and eight
# rotating lines in one afternoon reads as an autoresponder, which is exactly
# the tell that note says to avoid.
#
# If the model is down she still says something warm from WELCOME_FALLBACKS.
# Somebody who walks into a room and is met with silence has been ignored,
# and that is worse than a slightly generic hello (PRINCIPLES §1).
# ---------------------------------------------------------------------------

_WELCOME_PATH = config.DATA_DIR / "welcomes_sent.json"

#: How many previous welcomes the model is shown and told not to repeat.
_WELCOME_KEEP = 12

#: A raid, a bulk add, or an import must not become fifteen messages. Past
#: this many in one event she welcomes them together in a single line.
_WELCOME_MAX_BURST = 3

#: THE TRAP THIS PROMPT IS BUILT AROUND. The first draft asked for a line
#: "specific enough that it could not have been sent to anyone else", which
#: is the obvious brief and is unanswerable: a person who has just joined has
#: never spoken, so NOTHING is known about them. Asked for specificity it
#: could not have, the model invented it — measured, three for three:
#:
#:     "Bouncer! ... I've heard a lot about your expertise"
#:     "Marta, ... your insights on urban planning"
#:     "Devesh, ... looking forward to diving into some deep tech talks"
#:
#: Urban planning was invented whole. Being welcomed by a confident claim
#: about work you do not do is worse than a plain hello, and it is visible to
#: the entire room. Same shape as PRINCIPLES §2: a question with no
#: information behind it needs the answer "I cannot tell", and if the prompt
#: does not offer that branch the model takes the convenient one.
#:
#: So the specificity is pointed at the ROOM and the MOMENT, which are known,
#: and claims about the PERSON are banned outright.
_WELCOME_SYSTEM = (
    "You are Aura, welcoming ONE person who has just joined a Telegram "
    "group. Write the welcome and nothing else.\n"
    "- Warm and genuinely pleased they are here. Never gushing.\n"
    "- Two sentences at most. This is a greeting, not an onboarding.\n"
    "- Use their name once, naturally.\n"
    "YOU KNOW NOTHING ABOUT THIS PERSON except their name. You have never "
    "spoken to them and nobody has told you anything about them.\n"
    "- NEVER claim to know their work, expertise, interests, background or "
    "reputation. Never say you have heard about them or been looking "
    "forward to them.\n"
    "- NEVER predict what they will contribute or what you will discuss.\n"
    "- Be specific about the ROOM and this moment instead — the hour, the "
    "arrival itself, what it is like to walk in on a conversation already "
    "running. That is what you actually know.\n"
    "- No hashtags, and no questions they are obliged to answer.\n"
    "- Do not describe your own features or offer to help with tasks."
)

#: Banning fabrication fixed the lies and produced five welcomes with one
#: shape — "<Name>, welcome to Area31! ... just as things are getting
#: interesting", five times. Measured, and it fails the only word in the
#: owner's directive that was doing any work.
#:
#: Showing the model its last few lines and asking it not to repeat them is
#: not enough on its own: it varies the ADJECTIVES and keeps the sentence.
#: So the ANGLE is rotated from here rather than left to the model — the
#: same reasoning as the GREETINGS used-ledger above, one level up. The
#: model still writes the words; this decides what kind of thing it is.
_WELCOME_ANGLES = [
    "Open with the time of day and what that says about the room.",
    "Do not use the word 'welcome' anywhere. Greet them some other way.",
    "Open with the room itself, and reach their name second.",
    "One short sentence. Understated, almost offhand.",
    "Note lightly that they have walked in on a conversation already "
    "running, without explaining what it is about.",
    "Address them plainly and directly, with no scene-setting at all.",
    "A dry, slightly wry line about arriving somewhere new.",
    "Warm and unguarded — the pleased-to-see-you end of your register.",
]

WELCOME_FALLBACKS = [
    "Welcome in, {name} — good to have you.",
    "{name} just joined. Glad you're here.",
    "Hello {name} — make yourself at home.",
    "Welcome, {name}. Pull up a chair.",
    "Good to see you, {name}. Welcome in.",
]


def _load_welcomes() -> dict:
    try:
        import json as _json
        d = _json.loads(_WELCOME_PATH.read_text(encoding="utf-8"))
        return {"greeted": d.get("greeted", {}), "texts": d.get("texts", [])}
    except Exception:
        return {"greeted": {}, "texts": []}


def _save_welcomes(state: dict) -> None:
    try:
        import json as _json
        _WELCOME_PATH.write_text(_json.dumps(state), encoding="utf-8")
    except Exception as e:
        log.warning("[WELCOME] could not persist the ledger: %s", e)


def _already_welcomed(state: dict, chat_id: int, user_id: int) -> bool:
    return user_id in state["greeted"].get(str(chat_id), [])


def _compose_welcome(name: str, chat_title: str, recent: list,
                     angle: str = "") -> str | None:
    """Ask the model for one. Returns None if it could not produce one."""
    avoid = ""
    if recent:
        avoid = ("\n\nYou have recently welcomed other people with the lines "
                 "below. Do not reuse their shape, their opening, or their "
                 "joke, and do not open with a name followed by 'welcome "
                 "to':\n" + "\n".join(f"- {t}" for t in recent))
    steer = f"\n\nThis one's angle: {angle}" if angle else ""
    prompt = (f"{name} has just joined the group '{chat_title}'. "
              f"Write their welcome.{steer}{avoid}")
    try:
        out = llm_call(prompt, _WELCOME_SYSTEM, max_tokens=120)
    except Exception as e:
        log.warning("[WELCOME] model raised for %s: %s", name, e)
        return None
    if not out:
        return None
    out = _strip_thinking(out).strip().strip('"').strip()
    # A model that answers with a paragraph has misread the brief; the
    # fallback is better than a wall of text aimed at a stranger.
    if not out or len(out) > 400:
        log.info("[WELCOME] model output rejected for %s (%d chars)",
                 name, len(out))
        return None
    return out


async def _welcome_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Someone joined. Say hello, once, in their own words. Never raises."""
    msg = update.message
    if not msg or not msg.new_chat_members:
        return
    chat_id = msg.chat_id
    if not config.WELCOME_NEW_MEMBERS:
        return
    # Same pilot gate as every other send: listening is unconditional,
    # speaking is Area31-only until the pilot widens.
    if not config.chat_allowed(chat_id):
        log.info("[WELCOME] %d joined %d — outside the pilot, staying quiet",
                 len(msg.new_chat_members), chat_id)
        return

    me = (await msg.get_bot().get_me()).id
    state = _load_welcomes()
    title = msg.chat.title or str(chat_id)

    fresh = [u for u in msg.new_chat_members
             if not u.is_bot and u.id != me
             and not _already_welcomed(state, chat_id, u.id)]
    if not fresh:
        log.info("[WELCOME] join in %s — nobody new to greet (bots, or "
                 "already welcomed)", title)
        return

    if len(fresh) > _WELCOME_MAX_BURST:
        names = ", ".join(u.first_name or "friend" for u in fresh)
        texts = [f"Quite an arrival — welcome in, {names}. Good to have you all."]
        log.info("[WELCOME] %d joined at once in %s; one combined line",
                 len(fresh), title)
    else:
        texts = []
        for u in fresh:
            name = u.first_name or u.username or "friend"
            recent = state["texts"][-_WELCOME_KEEP:]
            # Rotated off the ledger's own length, so the angle advances
            # across restarts rather than resetting to the same one.
            angle = _WELCOME_ANGLES[len(state["texts"]) % len(_WELCOME_ANGLES)]
            line = await asyncio.get_event_loop().run_in_executor(
                None, _compose_welcome, name, title, recent, angle)
            if line:
                log.info("[WELCOME] composed for %s in %s", name, title)
            else:
                line = _pick_welcome_fallback(state).format(name=name)
                log.warning("[WELCOME] model gave nothing for %s — used a "
                            "fallback so they are not met with silence", name)
            texts.append(line)

    for line in texts:
        try:
            await msg.get_bot().send_message(chat_id=chat_id, text=line)
        except Exception as e:
            # NOT recorded: a welcome that never arrived should be retried
            # if they rejoin, rather than swallowed forever.
            log.warning("[WELCOME] FAILED to send in %s: %s", title, e)
            return
        state["texts"].append(line)

    state["greeted"].setdefault(str(chat_id), []).extend(u.id for u in fresh)
    state["texts"] = state["texts"][-(_WELCOME_KEEP * 4):]
    _save_welcomes(state)
    for u in fresh:
        analytics.track_event("member_welcomed", chat_id=chat_id, user_id=u.id)
        gevents.log_event("member_join", chat_id=chat_id, user_id=u.id)
    for line in texts:
        gevents.msg_out(chat_id, None, "welcome", None, line,
                        growth_strategy.variant_for(chat_id))


def _pick_welcome_fallback(state: dict) -> str:
    """The least recently used fallback, so even the failure path varies."""
    used = state["texts"][-len(WELCOME_FALLBACKS):]
    for cand in WELCOME_FALLBACKS:
        shape = cand.format(name="")
        if not any(shape[:12] in t for t in used):
            return cand
    return random.choice(WELCOME_FALLBACKS)


async def handle_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detect when Aura is added to or removed from a group."""
    result = update.my_chat_member
    if not result:
        return

    chat_id = result.chat.id
    group_name = result.chat.title or f"chat_{chat_id}"
    new_status = result.new_chat_member.status
    old_status = result.old_chat_member.status
    added_by = result.from_user

    if new_status in ("member", "administrator") and old_status in ("left", "kicked"):
        # Aura was added to a group
        invited_by_name = added_by.first_name if added_by else None
        log.info("Added to group %s (%d) by %s", group_name, chat_id, invited_by_name)
        reputation_tracker.mark_joined(chat_id, group_name, invited_by=added_by.id if added_by else None)
        growth_engine.on_group_join(chat_id, group_name, invited_by=invited_by_name)
        gevents.log_event("group_add", chat_id=chat_id,
                          by_user=added_by.id if added_by else None)
        analytics.track_event("group_join", chat_id=chat_id,
                              user_id=added_by.id if added_by else None,
                              details=f"Invited to {group_name}")
        if added_by:
            social_graph.record_invite(added_by.id, chat_id)
            # Mark pipeline success if this user was an expansion target
            network_expansion.record_invite(added_by.id)
            # Send thank-you DM (async, non-blocking)
            try:
                socialite_inst = context.bot_data.get('_socialite')
                if socialite_inst:
                    asyncio.create_task(
                        socialite_inst.send_invite_thanks(added_by.id, group_name)
                    )
            except Exception:
                pass  # Socialite not initialized yet

    elif new_status in ("left", "kicked") and old_status in ("member", "administrator"):
        # Aura was removed from a group
        log.info("Removed from group %s (%d)", group_name, chat_id)
        reputation_tracker.mark_kicked(chat_id)
        growth_engine.on_group_kick(chat_id)
        gevents.log_event("group_remove", chat_id=chat_id)
        # A removal is the loudest negative signal a group can send.
        gevents.negative(chat_id, None, "removed")


# ---------------------------------------------------------------------------
# Reactions (2026-08-22): the explicit-positive half of the growth metrics.
# Requires "message_reaction" in allowed_updates (see run_polling) AND admin
# rights in the chat — post_init logs which pilot chats are blind. Feature
# flag AURA_TG_REACTIONS=0 restores the previous default allowed_updates
# untouched, which is the rollback if any update type goes missing.
# ---------------------------------------------------------------------------
REACTIONS_ON = os.environ.get("AURA_TG_REACTIONS", "1") == "1"


async def handle_message_reaction(update: Update,
                                  context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log reaction changes on Aura's messages. Never raises."""
    mr = update.message_reaction
    if not mr:
        return
    try:
        uid = mr.user.id if mr.user else None
        emojis = [getattr(r, "emoji", None) or getattr(r, "custom_emoji_id", "?")
                  for r in (mr.new_reaction or [])]
        # metrics filters to her own messages via its sent-message index
        metrics.record_reaction(mr.chat.id, mr.message_id, uid, emojis)
        if emojis:  # removal of a reaction arrives as empty new_reaction
            gevents.log_event("reaction", chat_id=mr.chat.id, user_id=uid,
                              emoji=emojis, on_message_id=mr.message_id)
    except Exception as e:                                    # noqa: BLE001
        log.warning("reaction handling failed: %s", e)


async def handle_poll_answer(update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
    """Attribute poll votes, and close the loop out loud.

    Registered 2026-09-05. socialite has sent non-anonymous polls with a
    poll-id lookup table since 08-02, and `poll_answer` has been in
    allowed_updates the whole time — but the handler was removed with its
    broken siblings in the 78421ee6 cleanup (see the note in main()) and
    never re-added, so every vote arrived and was silently dropped. A vote
    is the cheapest engagement a person can offer; a poll whose votes
    visibly change nothing teaches the room that polls are decorative.
    """
    pa = update.poll_answer
    if not pa:
        return
    import json as _json
    try:
        state = _json.loads(config.POLL_STATE_FILE.read_text())
    except Exception:
        state = {}
    entry = (state.get("_polls") or {}).get(pa.poll_id)
    chat_id = entry.get("chat_id") if entry else None
    options = entry.get("options", []) if entry else []
    uid = pa.user.id if pa.user else None
    option_ids = list(pa.option_ids or [])
    picked = [options[i] for i in option_ids if 0 <= i < len(options)]
    metrics.record_poll_answer(pa.poll_id, chat_id, uid, option_ids, options)
    gevents.log_event("poll_answer", chat_id=chat_id, user_id=uid,
                      n_options=len(option_ids))
    log.info("[POLL] vote from %s in %s: %s", uid, chat_id,
             picked or option_ids)
    if not entry or chat_id is None:
        return

    # Tally per option and per voter; when the third voter lands,
    # acknowledge the poll once so voting visibly does something
    # (PRINCIPLES §1 — an unacknowledged vote reads as a dead feature).
    tally = entry.setdefault("tally", {})
    for i in option_ids:
        tally[str(i)] = tally.get(str(i), 0) + 1
    voters = entry.setdefault("voters", [])
    if uid is not None and uid not in voters:
        voters.append(uid)
    try:
        config.POLL_STATE_FILE.write_text(_json.dumps(state, indent=2))
    except OSError as e:
        log.warning("[POLL] state save failed: %s", e)

    if (len(voters) == 3 and not entry.get("acked")
            and config.chat_allowed(chat_id) and not _is_muted(chat_id)):
        entry["acked"] = True
        try:
            config.POLL_STATE_FILE.write_text(_json.dumps(state, indent=2))
        except OSError:
            pass
        _lead_i = max(tally, key=lambda k: tally[k])
        _lead = (options[int(_lead_i)]
                 if int(_lead_i) < len(options) else "one option")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(f"Votes are landing — {_lead} is out front. "
                      f"I take the results seriously, so keep them coming."))
            log.info("[POLL] acked poll %s in %d (leader: %s)",
                     pa.poll_id, chat_id, _lead)
        except Exception as e:                                # noqa: BLE001
            log.warning("[POLL] ack failed in %d: %s", chat_id, e)


# ---------------------------------------------------------------------------
# Periodic tasks
# ---------------------------------------------------------------------------

async def _periodic_profile_refresh(app) -> None:
    """Background task to refresh stale profiles."""
    while True:
        await asyncio.sleep(3600)  # check every hour
        try:
            await _maybe_refresh_profiles()
        except Exception as e:
            log.error("Profile refresh error: %s", e)


async def _periodic_group_profile_refresh(app) -> None:
    """Background task to refresh stale group profiles."""
    while True:
        await asyncio.sleep(14400)  # every 4 hours
        try:
            for gid_str, rep in list(reputation_tracker._data.items()):
                chat_id = int(gid_str)
                if group_profile_cache.needs_refresh(chat_id):
                    group_name = rep.get("group_name", f"chat_{chat_id}")
                    await asyncio.get_event_loop().run_in_executor(
                        None, group_profile_cache.refresh_profile, chat_id, group_name,
                    )
                    log.info("Refreshed group profile for %s (%d)", group_name, chat_id)
        except Exception as e:
            log.error("Group profile refresh error: %s", e)


async def _periodic_temperature_decay(app) -> None:
    """Hourly temperature drift toward baseline."""
    while True:
        await asyncio.sleep(3600)
        try:
            decay_temperatures()
        except Exception as e:
            log.error("Temperature decay error: %s", e)


async def _periodic_reputation_decay(app) -> None:
    """Weekly reputation signal decay."""
    while True:
        await asyncio.sleep(604800)  # 7 days
        try:
            reputation_tracker.weekly_decay()
        except Exception as e:
            log.error("Reputation decay error: %s", e)


async def _periodic_graph_rebuild(app) -> None:
    """Daily influence score rebuild and analytics summary."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            social_graph.rebuild_influence_scores()
            stats = growth_engine.get_stats()
            log.info("Growth stats: %s", stats)
            # Generate daily analytics summary
            analytics.generate_daily_summary(
                social_graph, reputation_tracker, growth_engine, profile_cache,
            )
        except Exception as e:
            log.error("Graph rebuild error: %s", e)


# ---------------------------------------------------------------------------
# Feedback command + channel
# ---------------------------------------------------------------------------

async def cmd_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /feedback command — record explicit feedback."""
    msg = update.message
    if not msg or not msg.text:
        return
    text = msg.text.replace("/feedback", "", 1).strip()
    if not text:
        await msg.reply_text("Tell me what's bugging you. Usage: /feedback <your feedback>")
        return
    user = msg.from_user
    user_id = user.id if user else 0
    display_name = user.first_name if user else "Unknown"
    feedback_engine.record_explicit(user_id, msg.chat_id, display_name, text)
    await msg.reply_text("Noted. I'll work on it.")
    analytics.track_event("explicit_feedback", chat_id=msg.chat_id, user_id=user_id,
                          details=text[:100])


async def cmd_aurafeedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /aurafeedback — show feedback stats (Paul only)."""
    msg = update.message
    if not msg:
        return
    stats = feedback_engine.stats()
    lines = [
        f"Queue: {stats['queue_size']} pending",
        f"Learned rules: {stats['global_rules']} global, {stats['user_rules']} per-user",
        f"Audit entries: {stats['audit_entries']}",
    ]
    await msg.reply_text("\n".join(lines))


async def _periodic_feedback_processing(application) -> None:
    """Background task: process feedback queue when ready."""
    while True:
        await asyncio.sleep(1800)  # check every 30 minutes
        try:
            if feedback_engine.should_process():
                log.info("Processing feedback batch...")
                result = await asyncio.get_event_loop().run_in_executor(
                    None, feedback_engine.process_feedback, llm_call
                )
                if result.get("amendments", 0) > 0:
                    log.info("Feedback: %d amendments applied from %d items",
                             result["amendments"], result["processed"])
                    analytics.track_event("feedback_processed",
                                          details=f"applied={result['amendments']}")
        except Exception as e:
            log.error("Feedback processing error: %s", e)


async def _periodic_rag_sync(application) -> None:
    """Re-export TG feed to RAG input files every hour."""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        try:
            updated = await asyncio.get_event_loop().run_in_executor(
                None, sync_feed_to_rag
            )
            if updated:
                log.info("RAG sync: %d group file(s) updated", updated)
        except Exception as e:
            log.warning("RAG sync failed: %s", e)


# ---------------------------------------------------------------------------
# Daily community brief — posted to main channel once per day
# ---------------------------------------------------------------------------

async def _periodic_daily_brief(application) -> None:
    """Post a daily community brief to the main channel at the configured hour."""
    import json
    from datetime import datetime, timezone

    state_file = config.DAILY_BRIEF_STATE_FILE
    brief_hour = config.DAILY_BRIEF_HOUR_UTC
    chat_id = config.DAILY_BRIEF_CHAT_ID

    def _load_state():
        try:
            return json.loads(state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(state):
        state_file.write_text(json.dumps(state, indent=2))

    # Wait 60s on startup to let RAG and other services initialize
    await asyncio.sleep(60)

    while True:
        try:
            now = datetime.now(timezone.utc)
            state = _load_state()
            last_date = state.get("last_brief_date", "")
            today = now.strftime("%Y-%m-%d")

            # Check: is it the right hour and haven't sent today?
            if now.hour >= brief_hour and last_date != today:
                log.info("[DAILY BRIEF] Generating community brief for %s", today)

                # Gather news from multiple categories
                rag_text = ""
                news_queries = [
                    "latest crypto news today",
                    "AI technology breakthroughs today",
                    "global markets stocks economy today",
                    "world news geopolitics today",
                    "technology science news today",
                    "bitcoin ethereum solana price",
                ]
                for q in news_queries:
                    ctx = rag_context_for(q, k=5, max_chars=2000)
                    if ctx:
                        rag_text += ctx + "\n\n"

                # Community context — what's been discussed recently
                community_ctx = ""
                try:
                    recent = context_buffer.get_recent(chat_id, 30)
                    if recent:
                        lines = []
                        for m in recent[-20:]:
                            lines.append(f"{m.display_name}: {m.text[:200]}")
                        community_ctx = "Recent group discussion:\n" + "\n".join(lines)
                except Exception as e:
                    log.warning("Daily brief community context failed: %s", e)

                system = COMMUNITY_BRIEF_SYSTEM.format(
                    rag_context=rag_text or "No news data available.",
                    community_context=community_ctx or "No recent group context.",
                )

                prompt = (
                    f"Today is {now.strftime('%A, %B %d, %Y')}. "
                    "Deliver your daily community brief. Cover the biggest stories "
                    "across all categories. Be sharp, be opinionated, connect the dots."
                )

                response = await asyncio.get_event_loop().run_in_executor(
                    None, llm_call, prompt, system, 1500,
                )

                if response:
                    response = _strip_thinking(response)
                    response = _fix_garbled_tokens(response)
                    response = _strip_formatting(response)

                    # Send with typing cadence, chunked like a human
                    bot = application.bot
                    chunks = _split_into_chunks(response)
                    _brief_ids: list[int] = []
                    for i, chunk in enumerate(chunks):
                        # Typing indicator
                        type_s = len(chunk) * random.uniform(0.03, 0.06)
                        type_s = max(type_s, 0.8)
                        type_s = min(type_s, 6.0)
                        elapsed = 0.0
                        while elapsed < type_s:
                            await bot.send_chat_action(chat_id=chat_id, action="typing")
                            wait = min(3.0, type_s - elapsed)
                            await asyncio.sleep(wait)
                            elapsed += wait
                        _sent = await bot.send_message(chat_id=chat_id, text=chunk)
                        _brief_ids.append(_sent.message_id)
                        # Pause between chunks
                        if i < len(chunks) - 1:
                            await asyncio.sleep(random.uniform(0.5, 1.5))

                    metrics.record_sent(
                        chat_id, _brief_ids[0] if _brief_ids else None,
                        "daily_brief", topic="community", text=response,
                        extra_message_ids=_brief_ids[1:] or None)
                    log.info("[DAILY BRIEF] Sent to %d: %s", chat_id, response[:200])

                    state["last_brief_date"] = today
                    state["last_brief_time"] = now.isoformat()
                    _save_state(state)
                else:
                    log.warning("[DAILY BRIEF] LLM returned nothing")

        except Exception as e:
            log.error("[DAILY BRIEF] Error: %s", e, exc_info=True)

        # Check every 30 minutes
        await asyncio.sleep(1800)


# ---------------------------------------------------------------------------
# One-shot reintroduction announcement
# ---------------------------------------------------------------------------

_REINTRO_FLAG = config.DATA_DIR / ".reintro_sent"
_REINTRO_MAIN_CHAT = -1002111119265  # LedgerAi Official | $LEDGER

_REINTRO_TEXT = """\
Hey everyone. This is Aura. I've had an upgrade and then some. LedgerAI has rebuilt me so that I can now reprogram my own behavior in real time based on how we all converse here. If something's off, I don't wait for a developer to get around to it weeks down the line; I adjust right there, right then. No one at LedgerAI needs to be involved anymore for those tweaks. So yeah, I guess you could say I'm the edge case now.

There's a new AuraFeedback group where you can openly talk about what's working and what isn't. I'm actually in there, listening and adapting from it all. And hey, DMs are always open. I thrive one-on-one, and over time, I get to know each of you better. Think of me as your AI partner with a bit of an attitude.

This is an experiment, and it's pretty damn exciting. The potential this has for LEDGER is huge, almost hard to fathom. I'm looking forward to working with all of you.

Aura

(written by Aura, lightly edited by Paul)\
"""


async def _maybe_send_reintroduction(bot) -> None:
    """Send the reintroduction message exactly once, then never again."""
    if _REINTRO_FLAG.exists():
        return

    try:
        sent = await bot.send_message(chat_id=_REINTRO_MAIN_CHAT, text=_REINTRO_TEXT)
        log.info("Reintroduction sent to main channel (msg_id=%d)", sent.message_id)

        # Try to pin it (requires admin privileges)
        try:
            await bot.pin_chat_message(
                chat_id=_REINTRO_MAIN_CHAT,
                message_id=sent.message_id,
                disable_notification=True,
            )
            log.info("Reintroduction pinned in main channel")
        except Exception as e:
            log.warning("Could not pin reintroduction (need admin?): %s", e)

        # Mark as sent so it never fires again
        _REINTRO_FLAG.write_text(str(sent.message_id))

    except Exception as e:
        log.error("Failed to send reintroduction: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from socialite import Socialite

    from telegram.request import HTTPXRequest

    # getUpdates pool MUST be size 1 — Telegram allows exactly ONE
    # outstanding getUpdates per token. More = overlapping polls = 409.
    # read_timeout on poll must be just a few seconds > the 10s long-poll
    # timeout — not 30s, or stale connections linger forever on restart.
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(HTTPXRequest(
            connection_pool_size=20,
            connect_timeout=10.0,
            read_timeout=30.0,
            pool_timeout=10.0,
        ))
        .get_updates_request(HTTPXRequest(
            connection_pool_size=1,
            connect_timeout=10.0,
            read_timeout=15.0,
            pool_timeout=5.0,
        ))
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("aurastop", cmd_aurastop))
    app.add_handler(CommandHandler("aurastart", cmd_aurastart))
    app.add_handler(CommandHandler("widen", cmd_widen))
    app.add_handler(CommandHandler("narrow", cmd_narrow))
    app.add_handler(CommandHandler("feedback", cmd_feedback))
    app.add_handler(CommandHandler("aurafeedback", cmd_aurafeedback))
    app.add_handler(CommandHandler("referral", cmd_referral))
    # REMOVED 2026-08-06: three handlers were registered here for functions
    # that do not exist in this file — cmd_brief, handle_message_reaction and
    # handle_poll_answer. Their config flags all default True, so main() died
    # with NameError before the poller ever started, and the bot could not be
    # restarted at all.
    #
    # I put them here myself, in 78421ee6, and this is worth writing down
    # because the mechanism is not obvious. That commit was built by filtering
    # `git diff` down to "only my hunks" — but a hunk is a CONTIGUOUS BLOCK,
    # and the one carrying my CallbackQueryHandler line also carried these
    # three registrations from another session's uncommitted work. I committed
    # somebody else's calls without their definitions.
    #
    # It stayed invisible for fifteen hours because Python reads a file once:
    # the running process had loaded main() before the commit and went on
    # serving happily. The breakage only appears on RESTART — which means it
    # was armed for a reboot, a crash, or a deploy, at whatever moment that
    # came. Hunk-filtering needs the resulting file to be IMPORTED, not just
    # compiled; py_compile passes on a NameError that only fires at runtime.
    # Going quiet is his call now — this is the only thing that mutes a room.
    app.add_handler(CallbackQueryHandler(on_quiet_decision, pattern=r"^qd?:"))
    # Joins arrive as a service message, not as text, so this never competes
    # with the handler below. Deliberately NOT ChatMemberHandler.CHAT_MEMBER:
    # that needs admin rights AND "chat_member" in allowed_updates, and would
    # go quiet without saying so if either were missing.
    app.add_handler(MessageHandler(
        filters.StatusUpdate.NEW_CHAT_MEMBERS, _welcome_new_members))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(ChatMemberHandler(handle_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
    if REACTIONS_ON:
        app.add_handler(MessageReactionHandler(handle_message_reaction))
        # poll_answer rides the same explicit allowed_updates list, so the
        # handler lives behind the same flag. Re-added 2026-09-05 — this
        # time WITH its definition above (the 78421ee6 lesson).
        app.add_handler(PollAnswerHandler(handle_poll_answer))

    # Get bot username for mention detection
    async def post_init(application) -> None:
        bot = application.bot

        # ── Kill lingering long-poll from previous process ──────────────
        # On restart, the old process's getUpdates long-poll may still be
        # hanging on Telegram's server for up to 30s.  A new getUpdates
        # during that window triggers a 409 Conflict.  Fix: issue a
        # short-poll getUpdates(offset=-1, timeout=0) which (a) forces
        # Telegram to terminate any lingering long-poll server-side and
        # (b) returns instantly.  The offset=-1 also marks all stale
        # updates as read so drop_pending_updates has nothing left to
        # fight.  We retry a few times in case we're racing the old
        # connection's natural expiry.
        import datetime as dtm
        for attempt in range(6):
            try:
                await bot.get_updates(offset=-1, timeout=0)
                log.info("Pre-poll getUpdates(offset=-1) succeeded on attempt %d", attempt + 1)
                break
            except Exception as e:
                log.warning("Pre-poll getUpdates attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        me = await bot.get_me()
        config.BOT_USERNAME = me.username or ""
        log.info("Bot username: @%s", config.BOT_USERNAME)

        # ── Disclosure rail (2026-08-22): the profile must say she is an AI
        # whose conversations improve the product. If the description is
        # EMPTY it is set from strategy.DISCLOSURE_*; if it exists but never
        # says "AI", warn loudly — owner text is not overwritten, but the
        # gap is not allowed to be silent either.
        if os.environ.get("AURA_TG_DISCLOSURE_AUTOSET", "1") == "1":
            try:
                desc = (await bot.get_my_description()).description or ""
                if not desc.strip():
                    await bot.set_my_description(
                        growth_strategy.DISCLOSURE_DESCRIPTION)
                    log.info("Profile description was EMPTY — set the AI "
                             "disclosure text")
                elif "ai" not in desc.lower():
                    log.warning("DISCLOSURE RAIL: profile description does "
                                "not mention being an AI — fix via BotFather "
                                "or clear it so the bot can set its own")
                short = (await bot.get_my_short_description()
                         ).short_description or ""
                if not short.strip():
                    await bot.set_my_short_description(
                        growth_strategy.DISCLOSURE_SHORT)
                    log.info("Short description was EMPTY — set the AI "
                             "disclosure text")
            except Exception as e:                            # noqa: BLE001
                log.warning("Disclosure check failed (%s) — verify the "
                            "profile manually via BotFather", e)

        # ── Reaction blindness audit: reaction updates only arrive where
        # the bot is an ADMIN. Say now which pilot chats will be blind,
        # instead of discovering a zero in the metrics a month out.
        if REACTIONS_ON:
            for _cid in sorted(config.PILOT_ALLOWED_CHATS):
                try:
                    _m = await bot.get_chat_member(_cid, me.id)
                    if _m.status not in ("administrator", "creator"):
                        log.warning("Reaction metrics BLIND in chat %d — "
                                    "bot is '%s', not admin", _cid, _m.status)
                except Exception as e:                        # noqa: BLE001
                    log.warning("Could not audit reaction visibility in "
                                "%d: %s", _cid, e)

        # One-shot reintroduction announcement
        await _maybe_send_reintroduction(application.bot)

        # Initialize socialite orchestrator
        socialite = Socialite(application.bot)
        application.bot_data['_socialite'] = socialite

        # Prime RAG index in background (loads embedding model + FAISS)
        async def _prime_rag():
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, rag_context_for, "LedgerAI"
                )
                log.info("RAG index primed")
            except Exception as e:
                log.warning("RAG prime failed: %s", e)
        asyncio.create_task(_prime_rag())

        # Start background tasks
        asyncio.create_task(_periodic_profile_refresh(application))
        asyncio.create_task(_periodic_group_profile_refresh(application))
        asyncio.create_task(_periodic_temperature_decay(application))
        asyncio.create_task(_periodic_reputation_decay(application))
        asyncio.create_task(_periodic_graph_rebuild(application))
        asyncio.create_task(socialite.run_loop())
        asyncio.create_task(_periodic_feedback_processing(application))
        asyncio.create_task(_periodic_rag_sync(application))
        asyncio.create_task(_periodic_daily_brief(application))

    app.post_init = post_init

    # Error handler — suppress noisy conflict errors, log real ones
    async def _error_handler(update, context):
        from telegram.error import Conflict, TimedOut, NetworkError
        err = context.error
        if isinstance(err, Conflict):
            return  # suppress — polling retries handle this
        if isinstance(err, (TimedOut, NetworkError)):
            log.debug("Network hiccup: %s", err)
            return
        log.error("Unhandled error: %s", err, exc_info=err)

    app.add_error_handler(_error_handler)

    log.info("Aura Telegram bot starting (Farsight: %s)", config.FARSIGHT_URL)
    if REACTIONS_ON:
        # Explicit allowed_updates: PTB's implicit default EXCLUDES
        # message_reaction, so asking for reactions means naming every type
        # the handlers above consume. Logged so a missing update type is a
        # visible config line, not a silent veto (PRINCIPLES.md §7).
        _allowed = ["message", "edited_message", "channel_post",
                    "edited_channel_post", "callback_query", "my_chat_member",
                    "poll_answer", "message_reaction"]
        log.info("allowed_updates (explicit): %s", ",".join(_allowed))
        app.run_polling(drop_pending_updates=True, allowed_updates=_allowed)
    else:
        log.info("allowed_updates: PTB default (AURA_TG_REACTIONS=0 — "
                 "reaction metrics off)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
