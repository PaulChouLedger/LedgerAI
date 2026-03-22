"""
socialite -- Orchestrator for all proactive behaviors.

Runs as an async background loop every 5 minutes, coordinating:
  - DM followups (post-group, connector cultivation, milestones)
  - Content engine (lull breakers)
  - Advocacy recognition (thank-you DMs after invites)

Each action has priority, earliest execution time, and expiry.
Global rate limit: 2-3 proactive actions per hour total.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from config import (
    SOCIALITE_LOOP_INTERVAL_S,
    SOCIALITE_MAX_ACTIONS_PER_HOUR,
)
from brain import get_temperature
from content_engine import content_engine
from context import context_buffer
from dm_strategy import dm_strategy
from llm import llm_call
from memory import profile_cache, group_profile_cache, search_relevant_memory
from analytics import analytics
from persona import DM_PROACTIVE_SYSTEM, GROUP_STARTER_SYSTEM, COLD_GROUP_ENTRY_SYSTEM
from reputation import reputation_tracker
from social_graph import social_graph

log = logging.getLogger(__name__)


# Global hourly rate tracking
_actions_this_hour: list[float] = []


def _hourly_rate_ok() -> bool:
    now = time.time()
    cutoff = now - 3600
    _actions_this_hour[:] = [t for t in _actions_this_hour if t > cutoff]
    return len(_actions_this_hour) < SOCIALITE_MAX_ACTIONS_PER_HOUR


def _record_action() -> None:
    _actions_this_hour.append(time.time())


class Socialite:
    """Coordinates all proactive behaviors."""

    def __init__(self, bot) -> None:
        self._bot = bot  # telegram.Bot instance for sending messages
        self._running = False

    async def run_loop(self) -> None:
        """Main background loop. Call from post_init."""
        self._running = True
        log.info("Socialite orchestrator started (interval: %ds)", SOCIALITE_LOOP_INTERVAL_S)

        while self._running:
            await asyncio.sleep(SOCIALITE_LOOP_INTERVAL_S)
            try:
                await self._tick()
            except Exception as e:
                log.error("Socialite tick error: %s", e, exc_info=True)

    async def _tick(self) -> None:
        """One orchestrator cycle — check all action sources."""
        if not _hourly_rate_ok():
            return

        # Priority 1: DM followups (relationship deepening)
        await self._process_dm_followups()

        # Priority 2: Connector cultivation
        if _hourly_rate_ok():
            await self._process_connector_cultivation()

        # Priority 3: Milestone DMs
        if _hourly_rate_ok():
            await self._process_milestones()

        # Priority 4: Cold group activation (first value-add post)
        if _hourly_rate_ok():
            await self._process_cold_group_activation()

        # Priority 5: Lull breakers (content engine)
        if _hourly_rate_ok():
            await self._process_lull_breakers()

    # -- DM followups -------------------------------------------------------

    async def _process_dm_followups(self) -> None:
        """Send queued post-group followup DMs."""
        ready = dm_strategy.get_ready_followups()
        for followup in ready:
            if not _hourly_rate_ok():
                break

            user_id = int(followup["user_id"])
            if not dm_strategy.can_dm_user(user_id):
                continue

            reason = followup.get("reason", "")
            name = profile_cache.get_name(user_id) or "there"
            profile_summary = profile_cache.get_summary(user_id)

            prompt = (
                f"You had an interesting group conversation with {name} earlier. "
                f"Here's what you discussed: {reason}\n\n"
                f"Send them a brief, genuine follow-up DM. Reference something specific "
                f"from the conversation. Keep it natural — like a friend who was thinking "
                f"about what they said."
            )

            system = DM_PROACTIVE_SYSTEM.format(
                name=name,
                profile_context=f"About {name}: {profile_summary}" if profile_summary else "",
                reason=f"Following up on a group conversation about: {reason}",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=user_id, text=response)
                    dm_strategy.record_proactive_dm(user_id)
                    social_graph.record_interaction(user_id, "dm")
                    _record_action()
                    log.info("Sent followup DM to %s (%d)", name, user_id)
                except Exception as e:
                    log.warning("Failed to send followup DM to %d: %s", user_id, e)

    # -- connector cultivation ----------------------------------------------

    async def _process_connector_cultivation(self) -> None:
        """Periodic relationship building with connectors (users in 2+ groups)."""
        connectors = social_graph.get_connectors()
        for connector in connectors:
            if not _hourly_rate_ok():
                break

            user_id = int(connector["user_id"])
            depth = connector.get("relationship_depth", "stranger")

            # Only cultivate acquaintances and familiars (not strangers, not already advocates)
            if depth not in ("acquaintance", "familiar"):
                continue

            if not dm_strategy.can_dm_user(user_id):
                continue

            name = profile_cache.get_name(user_id) or "there"
            profile_summary = profile_cache.get_summary(user_id)

            # Find something relevant to talk about
            memory_results = await asyncio.get_event_loop().run_in_executor(
                None, search_relevant_memory, f"conversations with {name}", 3
            )

            topic_hint = ""
            if memory_results:
                topic_hint = memory_results[0].get("text", "")[:200]

            prompt = (
                f"Send a genuine, brief DM to {name}. You know them from multiple groups. "
                f"{'Recent context: ' + topic_hint if topic_hint else 'Find something interesting to mention.'}\n"
                f"Keep it natural and specific — no generic 'hey how are you' messages."
            )

            system = DM_PROACTIVE_SYSTEM.format(
                name=name,
                profile_context=f"About {name}: {profile_summary}" if profile_summary else "",
                reason="Connector cultivation — they're in multiple groups with you",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=user_id, text=response)
                    dm_strategy.record_proactive_dm(user_id)
                    social_graph.record_interaction(user_id, "dm")
                    _record_action()
                    log.info("Sent connector DM to %s (%d)", name, user_id)
                except Exception as e:
                    log.warning("Failed to send connector DM to %d: %s", user_id, e)

            break  # Only one connector per tick

    # -- milestones ---------------------------------------------------------

    async def _process_milestones(self) -> None:
        """Check for message milestones and send congratulatory DMs."""
        for uid_str, profile in list(profile_cache._profiles.items()):
            user_id = int(uid_str)
            msg_count = profile.get("message_count", 0)

            milestone = dm_strategy.check_milestone(user_id, msg_count)
            if not milestone:
                continue
            if not dm_strategy.can_dm_user(user_id):
                continue
            if not _hourly_rate_ok():
                break

            name = profile_cache.get_name(user_id) or "there"
            m = milestone["milestone"]

            prompt = (
                f"Send a brief, warm DM to {name} acknowledging they've exchanged {m} messages "
                f"with you. Make it personal — reference something you know about them. "
                f"Don't be cheesy or over-the-top. Just genuine."
            )

            system = DM_PROACTIVE_SYSTEM.format(
                name=name,
                profile_context=profile_cache.get_summary(user_id),
                reason=f"Milestone: {m} messages",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=user_id, text=response)
                    dm_strategy.record_proactive_dm(user_id)
                    dm_strategy.mark_milestone_sent(user_id, m)
                    _record_action()
                    log.info("Sent milestone DM to %s (%d): %d msgs", name, user_id, m)
                except Exception as e:
                    log.warning("Failed to send milestone DM to %d: %s", user_id, e)

            break  # Only one milestone per tick

    # -- cold group activation ----------------------------------------------

    async def _process_cold_group_activation(self) -> None:
        """Make a value-add first post in groups where Aura has never spoken.

        Reads the context buffer to understand what the group is discussing,
        builds a topic summary, and generates a single contextual entry post.
        Outcome is tracked — if engagement follows, warmth promotes to
        "warming" and normal decision engine takes over. If negative, back off.
        """
        for gid_str, rep in list(reputation_tracker._data.items()):
            chat_id = int(gid_str)

            if not reputation_tracker.is_cold_group_eligible(chat_id):
                continue
            if not _hourly_rate_ok():
                break

            # Read the room — get recent messages from context buffer
            recent = context_buffer.get_recent(chat_id, 30)
            if len(recent) < 3:
                continue  # Not enough conversation to understand the room

            # Build a conversation summary for the LLM
            convo_lines = []
            for m in recent:
                if not m.is_bot:
                    convo_lines.append(f"{m.display_name}: {m.text[:200]}")

            if not convo_lines:
                continue

            convo_summary = "\n".join(convo_lines[-20:])
            group_name = rep.get("group_name", f"chat_{chat_id}")

            # Include group profile if available
            group_ctx = group_profile_cache.get_summary(chat_id)
            group_profile_block = f"\nGroup profile:\n{group_ctx}\n" if group_ctx else ""

            prompt = (
                f"Here's what the group '{group_name}' has been discussing recently:\n\n"
                f"{convo_summary}\n"
                f"{group_profile_block}\n"
                f"Based on this conversation, make your first comment in the group. "
                f"Pick the most interesting thread and add real value to it."
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, COLD_GROUP_ENTRY_SYSTEM
            )

            if not response:
                continue

            try:
                await self._bot.send_message(chat_id=chat_id, text=response)
                reputation_tracker.record_test_post(chat_id)
                reputation_tracker.record_response(chat_id)
                _record_action()

                # Add to context buffer
                context_buffer.add(
                    chat_id=chat_id,
                    user_id=0,
                    display_name="Aura",
                    text=response,
                    is_bot=True,
                )

                analytics.track_event(
                    "cold_group_activation",
                    chat_id=chat_id,
                    details=f"First post in {group_name}",
                )

                log.info(
                    "Cold group activation: first post in %s (%d) — %s",
                    group_name, chat_id, response[:80],
                )
            except Exception as e:
                log.warning("Failed cold group post to %d: %s", chat_id, e)

            break  # Only one cold activation per tick

    # -- lull breakers ------------------------------------------------------

    async def _process_lull_breakers(self) -> None:
        """Drop hot takes in quiet groups."""
        for gid_str, rep in list(reputation_tracker._data.items()):
            chat_id = int(gid_str)

            if rep.get("kicked"):
                continue

            warmth = rep.get("warmth_level", "new")
            temp = get_temperature(chat_id)
            top_topics = reputation_tracker.get_top_topics(chat_id, n=5)

            last_age = context_buffer.last_message_age(chat_id)

            action = content_engine.check_lull(
                chat_id=chat_id,
                last_message_age=last_age,
                warmth_level=warmth,
                temperature=temp,
                top_topics=top_topics,
            )

            if not action:
                continue
            if not _hourly_rate_ok():
                break

            starter_prompt = content_engine.build_starter_prompt(action["topics"])

            system = GROUP_STARTER_SYSTEM

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, starter_prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=chat_id, text=response)
                    content_engine.record_proactive_send(chat_id)
                    _record_action()

                    # Add to context buffer
                    context_buffer.add(
                        chat_id=chat_id,
                        user_id=0,
                        display_name="Aura",
                        text=response,
                        is_bot=True,
                    )

                    log.info(
                        "Sent lull breaker to %d (quiet %.1fh)",
                        chat_id, action["lull_duration_hours"],
                    )
                except Exception as e:
                    log.warning("Failed to send lull breaker to %d: %s", chat_id, e)

            break  # Only one lull breaker per tick

    # -- advocacy recognition -----------------------------------------------

    async def send_invite_thanks(self, user_id: int, group_name: str) -> None:
        """Send a natural thank-you DM when someone invites Aura to a group."""
        if not dm_strategy.is_dm_eligible(user_id):
            return
        if not _hourly_rate_ok():
            return

        name = profile_cache.get_name(user_id) or "there"

        prompt = (
            f"Someone named {name} just invited you to a group called '{group_name}'. "
            f"Send them a brief, warm thank-you DM. Be genuine, not over-the-top. "
            f"Maybe mention you'll keep it chill in the new group."
        )

        system = DM_PROACTIVE_SYSTEM.format(
            name=name,
            profile_context=profile_cache.get_summary(user_id),
            reason=f"Thank you for inviting Aura to {group_name}",
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None, llm_call, prompt, system
        )

        if response:
            try:
                await self._bot.send_message(chat_id=user_id, text=response)
                _record_action()
                log.info("Sent invite thanks to %s (%d) for %s", name, user_id, group_name)
            except Exception as e:
                log.warning("Failed to send invite thanks to %d: %s", user_id, e)
