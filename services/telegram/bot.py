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

import asyncio
import logging
import os
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

import config
from config import (
    TELEGRAM_BOT_TOKEN,
    DM_MIN_TIME_GAP,
    GLOBAL_MAX_PER_MINUTE,
)

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
    print("Missing TELEGRAM_BOT_TOKEN. Set it in .env or environment.")
    sys.exit(1)

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from brain import should_respond, record_response, evaluate_outcome, mark_response, decay_temperatures
from context import context_buffer, Message
from growth import growth_engine
from llm import llm_call
from memory import profile_cache, store_interaction, search_relevant_memory
from persona import DM_SYSTEM, GROUP_SYSTEM
from reputation import reputation_tracker
from social_graph import social_graph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aura.telegram")

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
# Interruption tracking
# ---------------------------------------------------------------------------
# Timestamp of the most recent inbound message per chat. If this changes
# while we're sending chunks, the user interrupted — stop talking.
_last_inbound_ts: dict[int, float] = {}

# Per-chat record of what Aura was saying when interrupted.
# {chat_id: {"sent": "what she said", "unsent": "what got cut off"}}
_interrupted_context: dict[int, dict] = {}


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
    chat, chat_id: int, response_text: str, input_text: str, first_chunk_only: bool = False,
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

        # Send the chunk
        await chat.send_message(chunk)
        sent_chunks.append(chunk)

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
async def _maybe_refresh_profiles() -> None:
    """Check and refresh stale profiles. Called periodically."""
    for uid_str, profile in list(profile_cache._profiles.items()):
        uid = int(uid_str)
        if profile_cache.needs_refresh(uid):
            log.info("Refreshing profile for %s (%s)", profile.get("display_name", "?"), uid)
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, profile_cache.refresh_profile, uid)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mute system
# ---------------------------------------------------------------------------
# {chat_id: expiry_timestamp} — if time.time() < expiry, Aura is muted
_muted_chats: dict[int, float] = {}

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

def _is_muted(chat_id: int) -> bool:
    expiry = _muted_chats.get(chat_id, 0)
    if expiry and time.time() < expiry:
        return True
    # Expired — clean up
    _muted_chats.pop(chat_id, None)
    return False


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
    log.info("Muted in chat %d for %s (%.0fs)", chat_id, human, duration)
    await update.message.reply_text(f"Got it. I'll be quiet for {human}. Use /aurastart when you want me back.")


async def cmd_aurastart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unmute Aura in this chat."""
    chat_id = update.message.chat_id
    was_muted = chat_id in _muted_chats
    _muted_chats.pop(chat_id, None)
    if was_muted:
        await update.message.reply_text("I'm back.")
    else:
        await update.message.reply_text("I wasn't muted, but noted.")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"Hey {name}. I'm Aura. Send me a message anytime."
    )


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

    # Mark inbound for interruption detection
    _mark_inbound(chat_id)

    # Update profile message count
    username = user.username if user else ""
    profile_cache.update_message_count(user_id, display_name, username=username)

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

    if chat_type == "private":
        social_graph.record_interaction(user_id, "dm")
        await _handle_dm(msg, chat_id, user_id, display_name, text)
    else:
        # Track user in this group for cross-group social graph
        social_graph.record_user_in_group(user_id, chat_id)
        social_graph.record_interaction(user_id, "group")
        # Detect organic growth opportunities (log only, never self-promote)
        growth_engine.detect_opportunity(text, chat_id, user_id)
        await _handle_group(msg, chat_id, user_id, display_name, text, chat_type)


async def _handle_dm(msg, chat_id, user_id, display_name, text) -> None:
    """Handle direct messages — always respond."""
    if not _global_rate_ok() or not _dm_rate_ok(chat_id):
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

    system = DM_SYSTEM.format(
        name=known_name,
        profile_context=profile_context + memory_context + interruption,
    )

    # Include recent conversation as context
    recent = context_buffer.format_for_prompt(chat_id, n=15)
    prompt = f"Recent conversation:\n{recent}\n\n{known_name}: {text}"

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system
    )

    if not response:
        log.warning("No LLM response for DM from %s", display_name)
        return

    # Send in human-paced sentence chunks (interruptible)
    sent_text = await _send_human(msg.chat, chat_id, response, text)

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

    # Store in memory container (fire and forget)
    _global_responses.append(time.time())
    _dm_last_response[chat_id] = time.time()

    asyncio.get_event_loop().run_in_executor(
        None,
        store_interaction,
        user_id, chat_id, "private", text, sent_text, display_name,
    )


async def _handle_group(msg, chat_id, user_id, display_name, text, chat_type) -> None:
    """Handle group messages — use decision engine to decide whether to respond."""
    # Respect mute
    if _is_muted(chat_id):
        return

    # Ensure reputation tracker knows about this group
    group_name = msg.chat.title or f"chat_{chat_id}"
    reputation_tracker.mark_joined(chat_id, group_name)

    # Check if this is a reply to one of Aura's messages
    is_reply_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user:
        bot_info = await msg.get_bot()
        is_reply_to_bot = msg.reply_to_message.from_user.id == bot_info.id
        if is_reply_to_bot:
            reputation_tracker.record_engagement(chat_id, "reply")

    # Build a lightweight Message for the decision engine
    message = Message(
        user_id=user_id,
        display_name=display_name,
        text=text,
    )

    # Evaluate outcome of Aura's last response (adaptive temperature)
    evaluate_outcome(chat_id, message, is_reply_to_bot)

    decision = should_respond(chat_id, message, is_reply_to_bot=is_reply_to_bot)

    if not decision.should_respond:
        log.debug(
            "Silent in %d: %.2f (%s)", chat_id, decision.score, decision.reason
        )
        return

    if not _global_rate_ok():
        return

    log.info(
        "Responding in group %d: %.2f (%s)", chat_id, decision.score, decision.reason
    )

    # Build group prompt
    profile_summary = profile_cache.get_summary(user_id)
    profile_context = f"About {display_name}: {profile_summary}" if profile_summary else ""

    # Check if this message interrupted Aura mid-stream
    interruption = _pop_interruption_context(chat_id)
    if interruption:
        profile_context += interruption

    conversation_context = context_buffer.format_for_prompt(chat_id, n=20)

    system = GROUP_SYSTEM.format(
        profile_context=profile_context,
        conversation_context=conversation_context,
    )

    prompt = f"{display_name}: {text}"

    response = await asyncio.get_event_loop().run_in_executor(
        None, llm_call, prompt, system
    )

    if not response:
        return

    # Send in human-paced sentence chunks (interruptible)
    sent_text = await _send_human(msg.chat, chat_id, response, text)

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
    _global_responses.append(time.time())

    asyncio.get_event_loop().run_in_executor(
        None,
        store_interaction,
        user_id, chat_id, chat_type, text, response, display_name,
    )


# ---------------------------------------------------------------------------
# Chat member tracking (join/kick detection)
# ---------------------------------------------------------------------------

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
        if added_by:
            social_graph.record_invite(added_by.id, chat_id)

    elif new_status in ("left", "kicked") and old_status in ("member", "administrator"):
        # Aura was removed from a group
        log.info("Removed from group %s (%d)", group_name, chat_id)
        reputation_tracker.mark_kicked(chat_id)
        growth_engine.on_group_kick(chat_id)


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
    """Daily influence score rebuild."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            social_graph.rebuild_influence_scores()
            stats = growth_engine.get_stats()
            log.info("Growth stats: %s", stats)
        except Exception as e:
            log.error("Graph rebuild error: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("aurastop", cmd_aurastop))
    app.add_handler(CommandHandler("aurastart", cmd_aurastart))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(ChatMemberHandler(handle_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))

    # Get bot username for mention detection
    async def post_init(application) -> None:
        bot = await application.bot.get_me()
        config.BOT_USERNAME = bot.username or ""
        log.info("Bot username: @%s", config.BOT_USERNAME)
        # Start background tasks
        asyncio.create_task(_periodic_profile_refresh(application))
        asyncio.create_task(_periodic_temperature_decay(application))
        asyncio.create_task(_periodic_reputation_decay(application))
        asyncio.create_task(_periodic_graph_rebuild(application))

    app.post_init = post_init

    log.info("Aura Telegram bot starting (Farsight: %s)", config.FARSIGHT_URL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
