"""
services.household -- Household engagement daemon.

Identifies speakers by voice, enrolls unknowns mid-conversation, builds
personality profiles over time, and personalizes every interaction.  Aura
becomes a family member who knows everyone in the house.

Subscribes to:
    audio.captured   — per-utterance speaker identification
    transcript.ready — name capture during enrollment, profile updates

Integrates with Presence for proactive-speech budget management.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from core.bus import bus
from core.config import (
    HOUSEHOLD_PROFILES_FILE,
    HOUSEHOLD_IDENTIFY_COOLDOWN,
    HOUSEHOLD_UNKNOWN_GREET_COOLDOWN,
    HOUSEHOLD_CONVERSATION_MODE_S,
    HOUSEHOLD_GREETING_COOLDOWN,
)
from core.state import state
from voice.llm_engine import llm_engine


class HouseholdEngagement:
    """Per-utterance speaker ID, discovery enrollment, personalized greetings."""

    def __init__(self, speaker, llm_client, enrollment) -> None:
        self._speaker = speaker
        self._llm_client = llm_client
        self._enrollment = enrollment
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Profiles: {user_id: {name, preferred_topics, communication_style,
        #            last_seen, interaction_count}}
        self._profiles: dict = {}

        # Cooldown tracking
        self._last_identify_ts: dict[str, float] = {}   # user_id → timestamp
        self._last_greeting_ts: dict[str, float] = {}   # user_id → timestamp
        self._last_unknown_greet_ts: float = 0.0

        # Enrollment flow state
        self._enrolling = False
        self._enrollment_audio: Optional[np.ndarray] = None

        # Budget callback (set by Presence)
        self._spend_budget_fn = None

        self._load_profiles()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        bus.on("audio.captured", self._on_audio_captured)
        bus.on("transcript.ready", self._on_transcript)
        print("[household] Started")

    def stop(self) -> None:
        self._stop.set()
        bus.off("audio.captured", self._on_audio_captured)
        bus.off("transcript.ready", self._on_transcript)
        print("[household] Stopped")

    # ------------------------------------------------------------------
    # Bus handlers
    # ------------------------------------------------------------------

    def _on_audio_captured(self, audio=None, sr: int = 16000, **_kw) -> None:
        """Identify speaker from captured audio. Runs on listener thread."""
        if audio is None or self._stop.is_set():
            return
        # Don't identify while Aura is speaking (echo)
        if state.playing or self._speaker.is_playing():
            return
        # Run identification on a daemon thread to avoid blocking listener
        threading.Thread(
            target=self._identify_speaker,
            args=(audio, sr),
            daemon=True,
            name="household-id",
        ).start()

    def _on_transcript(self, text: str = "", **_kw) -> None:
        """Handle transcript: enrollment name capture + profile updates."""
        if not text or self._stop.is_set():
            return

        # Reset conversation mode timer on each transcript
        if time.time() < state.conversation_mode_until:
            state.conversation_mode_until = time.time() + HOUSEHOLD_CONVERSATION_MODE_S

        # Enrollment flow: capture the name response
        if self._enrolling and self._enrollment_audio is not None:
            self._enrolling = False
            threading.Thread(
                target=self._complete_enrollment,
                args=(text, self._enrollment_audio),
                daemon=True,
                name="household-enroll",
            ).start()
            self._enrollment_audio = None
            return

        # Profile update for active user
        user_id = state.active_household_user
        if user_id and user_id in self._profiles:
            self._update_profile(user_id, text)

    # ------------------------------------------------------------------
    # Speaker identification
    # ------------------------------------------------------------------

    def _identify_speaker(self, audio: np.ndarray, sr: int) -> None:
        """Run enrollment.identify() and trigger greeting or enrollment."""
        try:
            user_id, score = self._enrollment.identify(audio, sr)
        except Exception as e:
            print(f"[household] identify error: {e}")
            return

        now = time.time()

        if user_id:
            # Known user — check cooldown before re-identifying
            last = self._last_identify_ts.get(user_id, 0.0)
            if now - last < HOUSEHOLD_IDENTIFY_COOLDOWN:
                return
            self._last_identify_ts[user_id] = now
            state.active_household_user = user_id

            # Ensure profile exists
            if user_id not in self._profiles:
                # Known to enrollment but no household profile yet
                name = self._enrollment.get_name(user_id)
                self._profiles[user_id] = {
                    "name": name or "Friend",
                    "preferred_topics": [],
                    "communication_style": "",
                    "last_seen": datetime.now().isoformat(),
                    "interaction_count": 0,
                }

            self._profiles[user_id]["last_seen"] = datetime.now().isoformat()
            self._save_profiles()

            # Greet if it's been a while
            last_greet = self._last_greeting_ts.get(user_id, 0.0)
            if now - last_greet >= HOUSEHOLD_GREETING_COOLDOWN:
                self._greet_known_user(user_id)
        else:
            # Unknown voice — trigger discovery enrollment
            if now - self._last_unknown_greet_ts < HOUSEHOLD_UNKNOWN_GREET_COOLDOWN:
                return
            # Require minimum score to avoid enrolling pure noise
            if score < 0.15:
                return
            self._greet_unknown_visitor(audio)

    # ------------------------------------------------------------------
    # Known user greeting
    # ------------------------------------------------------------------

    def _greet_known_user(self, user_id: str) -> None:
        """LLM-personalized greeting for a recognized household member."""
        if state.playing or self._speaker.is_playing():
            return

        profile = self._profiles.get(user_id, {})
        name = profile.get("name", "friend")
        topics = profile.get("preferred_topics", [])
        style = profile.get("communication_style", "")
        count = profile.get("interaction_count", 0)

        # Build context for LLM
        context_parts = [f"Name: {name}"]
        if topics:
            context_parts.append(f"Recent interests: {', '.join(topics[:5])}")
        if style:
            context_parts.append(f"Communication style: {style}")
        context_parts.append(f"Total interactions: {count}")
        context = "\n".join(context_parts)

        system = (
            "You are Aura, a personal AI assistant in someone's home. Generate a single "
            "warm, personalized greeting sentence for a household member who just spoke. "
            "Reference their interests if you know them. Keep it brief and natural — "
            "one sentence max. No markdown. No emojis."
        )
        prompt = f"Household member profile:\n{context}\n\nGenerate a short personalized greeting."

        try:
            greeting = llm_engine.chat_direct(
                system=system, user=prompt,
                max_tokens=100, temperature=0.8,
            )
            if greeting:
                print(f"[household] Greeting {name}: \"{greeting}\"")
                self._speaker.enqueue(greeting)
                self._last_greeting_ts[user_id] = time.time()
                state.conversation_mode_until = time.time() + HOUSEHOLD_CONVERSATION_MODE_S
                if self._spend_budget_fn:
                    self._spend_budget_fn()
                return
        except Exception as e:
            print(f"[household] LLM greeting error: {e}")

        # Fallback: simple name greeting
        fallback = f"Hey {name}."
        print(f"[household] Greeting {name} (fallback): \"{fallback}\"")
        self._speaker.enqueue(fallback)
        self._last_greeting_ts[user_id] = time.time()
        state.conversation_mode_until = time.time() + HOUSEHOLD_CONVERSATION_MODE_S
        if self._spend_budget_fn:
            self._spend_budget_fn()

    # ------------------------------------------------------------------
    # Unknown visitor enrollment
    # ------------------------------------------------------------------

    def _greet_unknown_visitor(self, audio: np.ndarray) -> None:
        """Greet an unknown voice and begin discovery enrollment."""
        if state.playing or self._speaker.is_playing():
            return
        if self._enrolling:
            return

        now = time.time()
        self._last_unknown_greet_ts = now
        self._enrolling = True
        self._enrollment_audio = audio.copy()

        print("[household] Unknown voice detected — starting discovery enrollment")
        self._speaker.enqueue(
            "Hey there, I don't think we've met. I'm Aura. What's your name?"
        )
        state.conversation_mode_until = time.time() + HOUSEHOLD_CONVERSATION_MODE_S
        if self._spend_budget_fn:
            self._spend_budget_fn()

    def _complete_enrollment(self, transcript: str, audio: np.ndarray) -> None:
        """Extract name from transcript, enroll voice, create profile."""
        name = self._extract_name(transcript)
        if not name:
            print(f"[household] Could not extract name from: \"{transcript}\"")
            self._speaker.enqueue("Sorry, I didn't catch that. We'll try again later.")
            return

        try:
            user_id = self._enrollment.enroll(name, audio)
        except Exception as e:
            print(f"[household] Enrollment failed: {e}")
            self._speaker.enqueue("Something went wrong with enrollment. We'll try again later.")
            return

        # Create household profile
        self._profiles[user_id] = {
            "name": name,
            "preferred_topics": [],
            "communication_style": "",
            "last_seen": datetime.now().isoformat(),
            "interaction_count": 1,
        }
        self._save_profiles()
        state.active_household_user = user_id

        print(f"[household] Enrolled new user: {name} ({user_id})")
        self._speaker.enqueue(f"Nice to meet you, {name}. I'll remember your voice.")
        state.conversation_mode_until = time.time() + HOUSEHOLD_CONVERSATION_MODE_S

    def _extract_name(self, transcript: str) -> Optional[str]:
        """Use LLM to extract a person's name from their response."""
        system = (
            "Extract the person's name from their response. Return ONLY the name, "
            "nothing else. If you can't determine a name, return 'UNKNOWN'. "
            "Examples:\n"
            "  'My name is Sarah' → 'Sarah'\n"
            "  'I'm Bob Carella' → 'Bob Carella'\n"
            "  'Call me Mike' → 'Mike'\n"
            "  'Sarah' → 'Sarah'\n"
            "  'um what' → 'UNKNOWN'"
        )
        try:
            name = llm_engine.chat_direct(
                system=system, user=transcript,
                max_tokens=30, temperature=0.1,
            ).strip("'\".")
            if name and name.upper() != "UNKNOWN" and len(name) < 50:
                return name
        except Exception as e:
            print(f"[household] Name extraction error: {e}")

        # Simple fallback: if transcript is 1-3 words, use it as the name
        words = transcript.strip().split()
        if 1 <= len(words) <= 3:
            name = " ".join(w.capitalize() for w in words)
            if name.isalpha() or all(w.isalpha() for w in words):
                return name

        return None

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    def _update_profile(self, user_id: str, transcript: str) -> None:
        """Update interaction count; extract topics every 5 interactions."""
        profile = self._profiles.get(user_id)
        if not profile:
            return

        profile["last_seen"] = datetime.now().isoformat()
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1
        count = profile["interaction_count"]

        # Every 5 interactions, ask LLM to extract topics and style
        if count > 0 and count % 5 == 0:
            threading.Thread(
                target=self._extract_profile_insights,
                args=(user_id,),
                daemon=True,
                name="household-profile",
            ).start()

        self._save_profiles()

    def _extract_profile_insights(self, user_id: str) -> None:
        """LLM extracts topics/interests/style from recent conversation turns."""
        profile = self._profiles.get(user_id)
        if not profile:
            return

        # Get recent conversation context
        turns = self._llm_client._turn_history[-5:] if self._llm_client._turn_history else []
        if not turns:
            return

        context = "\n".join(
            f"User: {u}\nAura: {a[:150]}" for u, a in turns
        )

        system = (
            "Analyze this conversation and extract:\n"
            "1. topics: comma-separated list of topics the user is interested in\n"
            "2. style: one brief phrase describing their communication style\n\n"
            "Format your response exactly as:\n"
            "topics: topic1, topic2, topic3\n"
            "style: description"
        )

        try:
            result = llm_engine.chat_direct(
                system=system,
                user=f"Recent conversation:\n{context}",
                max_tokens=100, temperature=0.3,
            )
            if not result:
                return

            topics = []
            style = ""
            for line in result.strip().split("\n"):
                line = line.strip()
                if line.lower().startswith("topics:"):
                    raw = line.split(":", 1)[1].strip()
                    topics = [t.strip() for t in raw.split(",") if t.strip()]
                elif line.lower().startswith("style:"):
                    style = line.split(":", 1)[1].strip()

            if topics:
                # Merge with existing, keep most recent
                existing = profile.get("preferred_topics", [])
                merged = list(dict.fromkeys(topics + existing))[:10]
                profile["preferred_topics"] = merged
            if style:
                profile["communication_style"] = style

            self._save_profiles()
            name = profile.get("name", user_id)
            print(f"[household] Profile updated for {name}: "
                  f"topics={profile.get('preferred_topics')}, style={profile.get('communication_style')}")

        except Exception as e:
            print(f"[household] Profile extraction error: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        path = Path(HOUSEHOLD_PROFILES_FILE)
        if path.exists():
            try:
                self._profiles = json.loads(path.read_text())
                print(f"[household] Loaded {len(self._profiles)} household profiles")
            except Exception as e:
                print(f"[household] Profile load error: {e}")
                self._profiles = {}
        else:
            self._profiles = {}

    def _save_profiles(self) -> None:
        path = Path(HOUSEHOLD_PROFILES_FILE)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._profiles, indent=2))
            except Exception as e:
                print(f"[household] Profile save error: {e}")
