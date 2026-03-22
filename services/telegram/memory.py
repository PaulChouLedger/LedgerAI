"""
memory -- Per-user profiles and memory container integration for Aura Telegram bot.

Two layers:
  1. Memory container (port 11438): long-term storage via /store, /search, /recent
  2. Local profile cache (data/telegram/profiles.json): LLM-generated user summaries
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from config import (
    MEMORY_URL,
    PROFILES_FILE,
    BOT_STATE_FILE,
    PROFILE_REFRESH_INTERVAL_S,
    PROFILE_REFRESH_MIN_MESSAGES,
    GROUP_PROFILES_FILE,
    GROUP_PROFILE_REFRESH_INTERVAL_S,
    GROUP_PROFILE_MIN_MESSAGES,
)
from llm import llm_call
from persona import PROFILE_BUILDER_SYSTEM, GROUP_PROFILE_BUILDER_SYSTEM

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile cache
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class ProfileCache:
    """LLM-generated user summaries, refreshed periodically."""

    def __init__(self) -> None:
        self._profiles: dict[str, dict] = _load_json(PROFILES_FILE, {})
        self._bot_state: dict = _load_json(BOT_STATE_FILE, {})

    def get(self, user_id: int) -> Optional[dict]:
        return self._profiles.get(str(user_id))

    def get_name(self, user_id: int) -> str:
        """Return the best known name for a user.

        Priority: preferred_name (user told us) > display_name (Telegram) > 'friend'
        """
        profile = self.get(user_id)
        if not profile:
            return ""
        return (
            profile.get("preferred_name")
            or profile.get("display_name")
            or ""
        )

    def set_preferred_name(self, user_id: int, name: str) -> None:
        """Set a user's preferred name (they told us what to call them)."""
        key = str(user_id)
        if key in self._profiles:
            self._profiles[key]["preferred_name"] = name
            self._save()
            log.info("Set preferred name for %s: %s", user_id, name)

    def get_summary(self, user_id: int, group_safe: bool = False) -> str:
        """Return a human-readable summary for the LLM prompt, or empty string.

        When group_safe=True, excludes relationship_summary and personality_notes
        which may contain private DM context that shouldn't leak into groups.
        """
        profile = self.get(user_id)
        if not profile:
            return ""
        parts = []
        # Name — use preferred if they told us, otherwise Telegram name
        name = profile.get("preferred_name") or profile.get("display_name")
        if name:
            parts.append(f"Name: {name}")
        tg_name = profile.get("display_name", "")
        preferred = profile.get("preferred_name", "")
        if preferred and tg_name and preferred != tg_name:
            parts.append(f"(Telegram name: {tg_name}, but they prefer: {preferred})")
        if profile.get("username"):
            parts.append(f"Telegram username: @{profile['username']}")
        if not group_safe and profile.get("personality_notes"):
            parts.append(f"Personality: {profile['personality_notes']}")
        if profile.get("topics_discussed"):
            parts.append(f"Topics they care about: {', '.join(profile['topics_discussed'][:8])}")
        if not group_safe and profile.get("relationship_summary"):
            parts.append(f"Your relationship: {profile['relationship_summary']}")
        msg_count = profile.get("message_count", 0)
        if msg_count > 0:
            parts.append(f"Messages exchanged: {msg_count}")
        return "\n".join(parts)

    def update_message_count(self, user_id: int, display_name: str, username: str = "") -> None:
        """Increment message count for a user and update Telegram metadata."""
        key = str(user_id)
        if key not in self._profiles:
            self._profiles[key] = {
                "display_name": display_name,
                "preferred_name": "",
                "username": username,
                "topics_discussed": [],
                "personality_notes": "",
                "relationship_summary": "",
                "message_count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
        self._profiles[key]["message_count"] = self._profiles[key].get("message_count", 0) + 1
        self._profiles[key]["last_seen"] = time.time()
        if display_name:
            self._profiles[key]["display_name"] = display_name
        if username:
            self._profiles[key]["username"] = username
        self._save()

    def needs_refresh(self, user_id: int) -> bool:
        """Check if this user's profile is stale enough to rebuild."""
        key = str(user_id)
        profile = self._profiles.get(key)
        if not profile:
            return False
        if profile.get("message_count", 0) < PROFILE_REFRESH_MIN_MESSAGES:
            return False
        last_refresh = self._bot_state.get("last_profile_refresh", {}).get(key, 0)
        return (time.time() - last_refresh) > PROFILE_REFRESH_INTERVAL_S

    def refresh_profile(self, user_id: int) -> None:
        """Rebuild a user's profile from recent memory container data."""
        key = str(user_id)
        profile = self._profiles.get(key)
        if not profile:
            return

        # Pull recent conversations for this user from memory
        conversations = search_user_conversations(user_id, limit=30)
        if not conversations:
            log.info("No conversations found for user %s, skipping refresh", user_id)
            return

        # Build transcript
        transcript = "\n".join(
            f"[{c.get('timestamp', '?')}] {c.get('text', '')[:300]}"
            for c in conversations
        )

        display_name = profile.get("display_name", f"User {user_id}")
        prompt = f"Here are recent conversations involving {display_name}:\n\n{transcript}"

        result = llm_call(
            prompt=prompt,
            system_prompt=PROFILE_BUILDER_SYSTEM.format(name=display_name),
            max_tokens=400,
        )
        if not result:
            return

        # Parse LLM output (expect JSON)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(result[start:end])
                profile["topics_discussed"] = parsed.get("topics", profile.get("topics_discussed", []))
                profile["personality_notes"] = parsed.get("personality", profile.get("personality_notes", ""))
                profile["relationship_summary"] = parsed.get("relationship", profile.get("relationship_summary", ""))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse profile LLM output: %s", e)
            # Store raw as personality notes
            profile["personality_notes"] = result[:500]

        profile["last_refreshed"] = time.time()
        self._profiles[key] = profile
        self._save()

        # Record refresh time
        if "last_profile_refresh" not in self._bot_state:
            self._bot_state["last_profile_refresh"] = {}
        self._bot_state["last_profile_refresh"][key] = time.time()
        _save_json(BOT_STATE_FILE, self._bot_state)

        log.info("Refreshed profile for %s (%s)", display_name, user_id)

    def _save(self) -> None:
        _save_json(PROFILES_FILE, self._profiles)


# ---------------------------------------------------------------------------
# Group profile cache
# ---------------------------------------------------------------------------

class GroupProfileCache:
    """LLM-generated profiles for each group chat Aura is in."""

    def __init__(self) -> None:
        self._profiles: dict = _load_json(GROUP_PROFILES_FILE, {})

    def get(self, chat_id: int) -> dict | None:
        return self._profiles.get(str(chat_id))

    def get_summary(self, chat_id: int) -> str:
        """Return a formatted group profile for injection into LLM prompts."""
        profile = self.get(chat_id)
        if not profile:
            return ""
        parts = []
        if profile.get("purpose"):
            parts.append(f"Group purpose: {profile['purpose']}")
        if profile.get("culture"):
            parts.append(f"Group culture: {profile['culture']}")
        if profile.get("topics"):
            parts.append(f"Common topics: {', '.join(profile['topics'][:7])}")
        if profile.get("key_players"):
            parts.append(f"Key voices: {', '.join(profile['key_players'][:5])}")
        if profile.get("value_add"):
            parts.append(f"Where you add value: {profile['value_add']}")
        if profile.get("avoid"):
            parts.append(f"Avoid: {profile['avoid']}")
        return "\n".join(parts)

    def needs_refresh(self, chat_id: int) -> bool:
        profile = self.get(chat_id)
        if not profile:
            return True
        last = profile.get("last_refreshed", 0)
        return (time.time() - last) > GROUP_PROFILE_REFRESH_INTERVAL_S

    def refresh_profile(self, chat_id: int, group_name: str) -> None:
        """Rebuild a group's profile from observed conversations."""
        conversations = search_group_conversations(chat_id, limit=50)
        if len(conversations) < GROUP_PROFILE_MIN_MESSAGES:
            log.debug("Not enough conversations for group %s (%d), skipping", group_name, chat_id)
            return

        transcript = "\n".join(
            f"{c.get('text', '')[:300]}"
            for c in conversations
        )

        prompt = f"Here are recent conversations from the group '{group_name}':\n\n{transcript}"
        result = llm_call(
            prompt,
            system_prompt=GROUP_PROFILE_BUILDER_SYSTEM,
            max_tokens=500,
        )
        if not result:
            return

        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(result[start:end])
                key = str(chat_id)
                profile = self._profiles.get(key, {})
                profile["group_name"] = group_name
                profile["purpose"] = parsed.get("purpose", profile.get("purpose", ""))
                profile["topics"] = parsed.get("topics", profile.get("topics", []))
                profile["culture"] = parsed.get("culture", profile.get("culture", ""))
                profile["key_players"] = parsed.get("key_players", profile.get("key_players", []))
                profile["value_add"] = parsed.get("value_add", profile.get("value_add", ""))
                profile["avoid"] = parsed.get("avoid", profile.get("avoid", ""))
                profile["last_refreshed"] = time.time()
                self._profiles[key] = profile
                self._save()
                log.info("Refreshed group profile for %s (%d)", group_name, chat_id)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse group profile LLM output: %s", e)

    def _save(self) -> None:
        _save_json(GROUP_PROFILES_FILE, self._profiles)


def search_group_conversations(chat_id: int, limit: int = 50) -> list[dict]:
    """Get recent observed conversations for a specific group."""
    try:
        resp = requests.get(
            f"{MEMORY_URL}/recent",
            params={"hours": 168, "limit": 500},  # 7 days, oversample
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        all_convos = resp.json().get("conversations", [])
        chat_str = str(chat_id)
        filtered = [
            c for c in all_convos
            if c.get("metadata", {}).get("chat_id") == chat_str
        ]
        return filtered[:limit]
    except Exception as e:
        log.debug("Group conversation search error: %s", e)
    return []


# ---------------------------------------------------------------------------
# Memory container integration
# ---------------------------------------------------------------------------

def store_observation(
    chat_id: int,
    chat_type: str,
    user_text: str,
    display_name: str = "",
) -> None:
    """Store an observed message (no Aura response) for passive analysis."""
    text = f"{display_name or 'User'}: {user_text}"
    try:
        resp = requests.post(
            f"{MEMORY_URL}/store",
            json={
                "text": text,
                "source": "telegram",
                "metadata": {
                    "platform": "telegram",
                    "chat_id": str(chat_id),
                    "chat_type": chat_type,
                    "display_name": display_name,
                    "observed_only": True,
                },
            },
            timeout=5,
        )
        if resp.status_code != 200:
            log.debug("Memory observe store failed: HTTP %d", resp.status_code)
    except Exception:
        pass  # Silent — don't log noise for passive observations


def store_interaction(
    user_id: int,
    chat_id: int,
    chat_type: str,
    user_text: str,
    bot_response: str,
    display_name: str = "",
) -> None:
    """Store a user+bot exchange in the memory container."""
    text = f"{display_name or 'User'}: {user_text}\nAura: {bot_response}"
    try:
        resp = requests.post(
            f"{MEMORY_URL}/store",
            json={
                "text": text,
                "source": "telegram",
                "metadata": {
                    "platform": "telegram",
                    "user_id": str(user_id),
                    "chat_id": str(chat_id),
                    "chat_type": chat_type,
                    "display_name": display_name,
                },
            },
            timeout=5,
        )
        if resp.status_code != 200:
            log.warning("Memory store failed: HTTP %d", resp.status_code)
    except Exception as e:
        log.warning("Memory store error: %s", e)


def search_relevant_memory(query: str, k: int = 5) -> list[dict]:
    """Semantic search in the memory container."""
    try:
        resp = requests.post(
            f"{MEMORY_URL}/search",
            json={"query": query, "k": k, "threshold": 0.3},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        log.debug("Memory search error: %s", e)
    return []


def search_user_conversations(user_id: int, limit: int = 20) -> list[dict]:
    """Get recent conversations for a specific user.

    The memory container doesn't support server-side metadata filtering,
    so we fetch more results and filter client-side.
    """
    try:
        resp = requests.get(
            f"{MEMORY_URL}/recent",
            params={"hours": 168, "limit": 200},  # 7 days, oversample
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        all_convos = resp.json().get("conversations", [])
        # Client-side filter by user_id in metadata or text
        user_str = str(user_id)
        filtered = []
        for c in all_convos:
            meta = c.get("metadata", {})
            if meta.get("user_id") == user_str or meta.get("platform") == "telegram":
                filtered.append(c)
            if len(filtered) >= limit:
                break
        return filtered
    except Exception as e:
        log.debug("Memory recent fetch error: %s", e)
        return []


# Singletons
profile_cache = ProfileCache()
group_profile_cache = GroupProfileCache()
