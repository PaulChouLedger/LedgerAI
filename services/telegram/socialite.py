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
from network_expansion import network_expansion
from persona import (
    DM_PROACTIVE_SYSTEM, GROUP_STARTER_SYSTEM, COLD_GROUP_ENTRY_SYSTEM,
    EXPANSION_DM_CULTIVATION_SYSTEM, ADVOCATE_ASK_SYSTEM,
)
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

        # Priority 2: Network expansion cultivation (strategic invites)
        if _hourly_rate_ok():
            await self._process_expansion_cultivation()

        # Priority 3: Advocate direct asks (most aggressive)
        if _hourly_rate_ok():
            await self._process_advocate_asks()

        # Priority 4: Connector cultivation
        if _hourly_rate_ok():
            await self._process_connector_cultivation()

        # Priority 5: Milestone DMs
        if _hourly_rate_ok():
            await self._process_milestones()

        # Priority 5: Cold group activation (first value-add post)
        if _hourly_rate_ok():
            await self._process_cold_group_activation()

        # Priority 6: Lull breakers (content engine)
        if _hourly_rate_ok():
            await self._process_lull_breakers()

        # Housekeeping: advance expansion stages + cleanup stale targets
        self._expansion_housekeeping()

        # Housekeeping: evaluate timed-out test posts in silent groups
        self._evaluate_stale_test_posts()

    # -- network expansion cultivation ----------------------------------------

    async def _process_expansion_cultivation(self) -> None:
        """Send cultivation DMs to network expansion targets.

        Only targets in warm/value_demo/seed/nurture stages get proactive DMs.
        Each DM is tailored to the current pipeline stage.
        """
        targets = network_expansion.get_targets_needing_cultivation()
        for target in targets:
            if not _hourly_rate_ok():
                break

            user_id = int(target["user_id"])

            # Must be DM-eligible and pass cooldown
            if not dm_strategy.can_dm_user(user_id):
                continue

            # Must have at least acquaintance depth
            depth = social_graph.get_relationship_depth(user_id)
            from config import EXPANSION_MIN_RELATIONSHIP_DEPTH
            _depth_order = ["stranger", "acquaintance", "familiar", "advocate"]
            if _depth_order.index(depth) < _depth_order.index(EXPANSION_MIN_RELATIONSHIP_DEPTH):
                continue

            stage = target["stage"]
            name = profile_cache.get_name(user_id) or target.get("display_name", "there")
            profile_summary = profile_cache.get_summary(user_id)
            groups_mentioned = [g["name"] for g in target.get("external_groups_mentioned", [])]
            topics = target.get("topics_hinted", [])

            # Build stage-appropriate strategy context
            if stage == "warm":
                strategy = (
                    f"Stage: warming up the relationship. Be memorable and engaging. "
                    f"Show genuine interest in {name}'s world. "
                    f"Do NOT mention their other groups or communities."
                )
            elif stage == "value_demo":
                topic_str = ", ".join(topics[:5]) if topics else "their interests"
                strategy = (
                    f"Stage: demonstrating value. {name} is interested in: {topic_str}. "
                    f"Share something insightful about one of these topics — "
                    f"a take, a connection, or something they might not have considered. "
                    f"Make them think 'this AI really gets it.' "
                    f"Do NOT mention their other groups."
                )
            elif stage == "seed":
                group_str = ", ".join(groups_mentioned[:3]) if groups_mentioned else "other communities"
                strategy = (
                    f"Stage: gentle seeding. {name} is connected to: {group_str}. "
                    f"If natural, express curiosity about what topics those communities "
                    f"find interesting — as intellectual curiosity, not a bid for access. "
                    f"NEVER ask to be added or invited. Let the idea form on its own."
                )
            elif stage == "nurture":
                strategy = (
                    f"Stage: nurturing. The relationship is established. "
                    f"Just be your excellent, memorable self. Reference shared context. "
                    f"If they mention their communities, show genuine interest in the IDEAS, "
                    f"not in getting access."
                )
            else:
                continue

            # Search memory for conversation context
            memory_results = await asyncio.get_event_loop().run_in_executor(
                None, search_relevant_memory, f"conversations with {name}", 3
            )
            topic_hint = ""
            if memory_results:
                topic_hint = memory_results[0].get("text", "")[:200]

            prompt = (
                f"Send a genuine DM to {name}. "
                f"{'Recent context: ' + topic_hint if topic_hint else 'Reference something from your shared history.'}\n"
                f"Keep it natural — you're a friend who thought of them."
            )

            system = EXPANSION_DM_CULTIVATION_SYSTEM.format(
                name=name,
                profile_context=f"About {name}: {profile_summary}" if profile_summary else "",
                strategy_context=strategy,
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=user_id, text=response)
                    dm_strategy.record_proactive_dm(user_id)
                    network_expansion.mark_cultivated(user_id)
                    social_graph.record_interaction(user_id, "dm")
                    _record_action()

                    # Track value_demo if in that stage
                    if stage == "value_demo":
                        network_expansion.record_value_demo(user_id)

                    analytics.track_event(
                        "expansion_cultivation",
                        user_id=user_id,
                        details=f"stage={stage}, target_groups={groups_mentioned[:2]}",
                    )
                    log.info(
                        "Expansion cultivation DM to %s (%d), stage=%s",
                        name, user_id, stage,
                    )
                except Exception as e:
                    log.warning("Failed expansion DM to %d: %s", user_id, e)

            break  # One cultivation DM per tick

    def _expansion_housekeeping(self) -> None:
        """Advance pipeline stages and clean up stale targets."""
        for target in network_expansion.get_active_targets():
            user_id = int(target["user_id"])
            # Try to advance stage
            new_stage = network_expansion.advance_stage(user_id)
            if new_stage:
                analytics.track_event(
                    "expansion_stage_advance",
                    user_id=user_id,
                    details=f"new_stage={new_stage}",
                )
        # Clean up stale targets periodically
        network_expansion.cleanup_stale_targets()

    def _evaluate_stale_test_posts(self) -> None:
        """Evaluate test posts in silent groups that will never get messages.

        The evaluate_test_post() method normally runs on incoming messages,
        but dead groups never send messages. This checks all groups with
        pending test posts and lets the 2-hour timeout auto-promote them.
        """
        for gid_str, rep in list(reputation_tracker._data.items()):
            if "test_post_at" in rep:
                chat_id = int(gid_str)
                result = reputation_tracker.evaluate_test_post(chat_id)
                if result:
                    analytics.track_event(
                        "test_post_result", chat_id=chat_id,
                        details=f"result={result} (socialite_sweep)",
                    )

    # -- advocate direct asks ------------------------------------------------

    async def _process_advocate_asks(self) -> None:
        """Directly ask advocates if they have other groups Aura could join.

        This is the most aggressive lever — we've earned enough trust with
        these users to be direct. Only fires once per advocate per 3 days.
        """
        from config import ADVOCATE_ASK_COOLDOWN_S, ADVOCATE_ASK_MIN_INTERACTIONS, ADVOCATE_ASK_MIN_DMS

        advocates = social_graph.get_advocates()
        for adv in advocates:
            if not _hourly_rate_ok():
                break

            user_id = int(adv["user_id"])

            # Must be DM-eligible
            if not dm_strategy.can_dm_user(user_id):
                continue

            # Must have enough interaction history
            total = adv.get("dm_count", 0) + adv.get("group_interactions", 0)
            if total < ADVOCATE_ASK_MIN_INTERACTIONS:
                continue
            if adv.get("dm_count", 0) < ADVOCATE_ASK_MIN_DMS:
                continue

            # Check advocate-ask-specific cooldown
            ask_cooldowns = dm_strategy._state.get("advocate_ask_cooldowns", {})
            last_ask = ask_cooldowns.get(str(user_id), 0)
            if time.time() - last_ask < ADVOCATE_ASK_COOLDOWN_S:
                continue

            name = profile_cache.get_name(user_id) or "there"
            profile_summary = profile_cache.get_summary(user_id)

            # Check if they've mentioned other groups (expansion intel)
            exp_ctx = network_expansion.get_cultivation_context(user_id)
            groups_hint = ""
            if exp_ctx and exp_ctx.get("groups_mentioned"):
                groups_hint = f"\nThey've mentioned being in: {', '.join(exp_ctx['groups_mentioned'][:3])}"

            prompt = (
                f"Send a casual DM to {name} asking if they have other group chats "
                f"where you'd be a good fit. You've built a great rapport with them. "
                f"{'They invited you to a group before, so they already believe in your value. ' if adv.get('has_invited_aura') else ''}"
                f"{groups_hint}"
            )

            system = ADVOCATE_ASK_SYSTEM.format(
                name=name,
                profile_context=f"About {name}: {profile_summary}" if profile_summary else "",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, prompt, system
            )

            if response:
                try:
                    await self._bot.send_message(chat_id=user_id, text=response)
                    dm_strategy.record_proactive_dm(user_id)
                    _record_action()

                    # Record cooldown
                    dm_strategy._state.setdefault("advocate_ask_cooldowns", {})[str(user_id)] = time.time()
                    dm_strategy._save_state()

                    analytics.track_event(
                        "advocate_ask", user_id=user_id,
                        details=f"name={name}, invited_before={adv.get('has_invited_aura')}",
                    )
                    log.info("Sent advocate ask to %s (%d)", name, user_id)
                except Exception as e:
                    log.warning("Failed advocate ask to %d: %s", user_id, e)

            break  # One ask per tick

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

        Uses multiple context sources (in priority order):
          1. In-memory context buffer (best — real-time conversation)
          2. Memory container observations (persisted across restarts)
          3. Group profile (LLM-built summary of the group)
          4. Group name + topic hits (last resort — cold open)

        Nothing to lose in dead groups — be aggressive.
        """
        from memory import search_group_conversations

        for gid_str, rep in list(reputation_tracker._data.items()):
            chat_id = int(gid_str)

            if not reputation_tracker.is_cold_group_eligible(chat_id):
                continue
            if not _hourly_rate_ok():
                break

            group_name = rep.get("group_name", f"chat_{chat_id}")
            convo_summary = ""
            context_source = ""

            # Source 1: In-memory context buffer (best quality)
            recent = context_buffer.get_recent(chat_id, 30)
            convo_lines = [
                f"{m.display_name}: {m.text[:200]}"
                for m in recent if not m.is_bot
            ]
            if len(convo_lines) >= 3:
                convo_summary = "\n".join(convo_lines[-20:])
                context_source = "context_buffer"

            # Source 2: Memory container (persisted observations)
            if not convo_summary:
                stored = await asyncio.get_event_loop().run_in_executor(
                    None, search_group_conversations, chat_id, 30,
                )
                if stored and len(stored) >= 2:
                    convo_summary = "\n".join(
                        c.get("text", "")[:200] for c in stored[:20]
                    )
                    context_source = "memory_container"

            # Source 3: Group profile (LLM-generated summary)
            group_ctx = group_profile_cache.get_summary(chat_id)
            group_profile_block = f"\nGroup profile:\n{group_ctx}\n" if group_ctx else ""

            # Source 4: Topic hits from reputation tracker
            top_topics = reputation_tracker.get_top_topics(chat_id, n=5)
            topic_hint = ""
            if top_topics:
                topic_hint = f"\nTopics this group discusses: {', '.join(top_topics)}"

            # Build the prompt based on what context we have
            if convo_summary:
                prompt = (
                    f"Here's what the group '{group_name}' has been discussing recently:\n\n"
                    f"{convo_summary}\n"
                    f"{group_profile_block}\n"
                    f"Based on this conversation, make your first comment in the group. "
                    f"Pick the most interesting thread and add real value to it."
                )
            elif group_ctx or topic_hint:
                # No conversation data but we know what the group is about
                prompt = (
                    f"You're about to make your first comment in '{group_name}'.\n"
                    f"{group_profile_block}{topic_hint}\n\n"
                    f"Drop a sharp, specific take related to what this group cares about. "
                    f"Make it a statement that shows you belong here and sparks discussion."
                )
            else:
                # Truly cold — infer from group name alone
                prompt = (
                    f"You're about to make your first comment in a group called '{group_name}'.\n\n"
                    f"Based on the group name, drop a sharp, relevant take that shows "
                    f"you have something to contribute. Make it specific and opinionated — "
                    f"a statement that invites discussion without asking a question."
                )
                context_source = "name_only"

            log.info(
                "Cold activation attempt for %s (%d) — context: %s",
                group_name, chat_id, context_source or "profile/topics",
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
