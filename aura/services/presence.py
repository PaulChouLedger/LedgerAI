"""
services.presence -- Proactive voice initiation.

Daemon thread that monitors ambient RMS for room presence, evaluates
four trigger types, and makes Aura speak first without waiting for user
input.  Coordinates with perpetual.py through state — never speaks
simultaneously, consumes pending briefings/questions when appropriate.

Triggers (in priority order):
    1. Morning briefing   — overnight Telegram summary (6am-11am)
    2. Room presence greeting — short greeting after 30+ min silence
    3. Idle commentary    — follow-up on earlier conversation
    4. Telegram alert     — high-signal event notification
"""

from __future__ import annotations

import collections
import json
import random
import threading
import time
from datetime import datetime
from typing import Optional

import requests

from core.bus import bus
from core.config import (
    DATA_DIR,
    LLM_URL,
    MEMORY_URL,
    PRESENCE_RMS_QUIET,
    PRESENCE_RMS_ACTIVE,
    PRESENCE_WINDOW_SIZE,
    PRESENCE_MIN_SILENCE_S,
    PRESENCE_GREETING_COOLDOWN,
    MORNING_BRIEFING_HOUR_MIN,
    MORNING_BRIEFING_HOUR_MAX,
    IDLE_COMMENT_COOLDOWN_S,
    IDLE_COMMENT_MIN_SILENCE_S,
    IDLE_COMMENT_MAX_SILENCE_S,
    IDLE_SESSION_WINDOW_S,
    TELEGRAM_ALERT_COOLDOWN_S,
    TELEGRAM_ALERT_MAX_HOUR,
    PROACTIVE_DAILY_BUDGET,
)
from core.state import state

# ---------------------------------------------------------------------------
# Pre-written greeting pool (no LLM needed)
# ---------------------------------------------------------------------------

_GREETINGS = [
    "Hey.",
    "Oh hey, didn't hear you come in.",
    "Morning.",
    "Hey there.",
    "Oh, hey.",
    "There you are.",
    "Welcome back.",
]

# High-signal Telegram event types worth alerting on
_ALERT_EVENT_TYPES = {
    "advocate_intro",     # new advocate discovered
    "dm_received",        # someone DMed asking about LedgerAI
    "warmth_milestone",   # group warmth crossed a threshold
}


class Presence:
    """Proactive voice initiation daemon."""

    def __init__(self, speaker, llm_client, household=None) -> None:
        self._speaker = speaker
        self._llm_client = llm_client
        self._household = household
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # RMS sliding window (deque of (timestamp, rms) tuples)
        self._rms_window: collections.deque = collections.deque(
            maxlen=PRESENCE_WINDOW_SIZE
        )

        # Cooldown / budget tracking
        self._last_greeting_ts = 0.0
        self._last_morning_briefing_date: Optional[str] = None
        self._last_idle_comment_ts = 0.0
        self._last_telegram_alert_ts = 0.0
        self._telegram_alert_hour_ts: list[float] = []  # timestamps this hour
        self._daily_budget_remaining = PROACTIVE_DAILY_BUDGET
        self._budget_date: Optional[str] = None

        # Last silence start (when RMS dropped below quiet threshold)
        self._silence_start: Optional[float] = None
        self._last_active_ts = 0.0  # last time RMS was in "active" range

        # Last transcript timestamp (set by bus subscriber)
        self._last_transcript_ts = 0.0

        # Seen alert event timestamps (avoid re-alerting)
        self._seen_alert_ts: set[float] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Subscribe to ambient RMS from listener
        bus.on("ambient.level", self._on_ambient)
        bus.on("transcript.ready", self._on_transcript)
        # Wire household budget spending to our budget
        if self._household:
            self._household._spend_budget_fn = self._spend_budget
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="presence"
        )
        self._thread.start()
        print("[presence] Started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[presence] Stopped")

    # ------------------------------------------------------------------
    # Bus subscribers
    # ------------------------------------------------------------------

    def _on_ambient(self, rms: float = 0.0, **_kw) -> None:
        now = time.time()
        self._rms_window.append((now, rms))

        # Track silence/active transitions
        if rms < PRESENCE_RMS_QUIET:
            if self._silence_start is None:
                self._silence_start = now
        else:
            self._silence_start = None
            if rms > PRESENCE_RMS_ACTIVE:
                self._last_active_ts = now

    def _on_transcript(self, text: str = "", **_kw) -> None:
        if text:
            self._last_transcript_ts = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Wait a bit after boot before evaluating triggers
        self._stop.wait(30)

        while not self._stop.is_set():
            try:
                self._reset_daily_budget()
                # Evaluate triggers in priority order
                if self._check_morning_briefing():
                    pass
                elif self._check_presence_greeting():
                    pass
                elif self._check_idle_commentary():
                    pass
                elif self._check_telegram_alert():
                    pass
            except Exception as e:
                print(f"[presence] Error in main loop: {e}")

            self._stop.wait(10)  # tick every 10s

    # ------------------------------------------------------------------
    # Universal gate — can we speak right now?
    # ------------------------------------------------------------------

    def _can_speak(self) -> bool:
        """Check all conditions that must be true before proactive speech."""
        import os
        if os.environ.get("AURA_DEMO_MODE", "").lower() in ("1", "true", "yes"):
            return False

        if self._daily_budget_remaining <= 0:
            return False

        # Suppressed by user ("shut up" / "be quiet")
        if time.time() < state.presence_suppressed_until:
            return False

        # System busy
        if state.playing:
            return False
        if state.shutdown_requested:
            return False
        if state.perpetual_active:
            return False

        # Speaker currently playing
        if self._speaker.is_playing():
            return False

        # Check if VAD is active (someone is speaking right now)
        # We can't directly check VAD, but if a transcript arrived
        # very recently, someone is mid-conversation
        if time.time() - self._last_transcript_ts < 5:
            return False

        return True

    def _spend_budget(self) -> None:
        self._daily_budget_remaining -= 1
        # Update last_conversation_ts so perpetual's idle timer resets
        state.last_conversation_ts = time.time()
        print(f"[presence] Budget spent — {self._daily_budget_remaining} remaining today")

    def _reset_daily_budget(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self._budget_date != today:
            self._budget_date = today
            self._daily_budget_remaining = PROACTIVE_DAILY_BUDGET
            self._telegram_alert_hour_ts.clear()

    # ------------------------------------------------------------------
    # Presence detection (dual-threshold hysteresis)
    # ------------------------------------------------------------------

    def _detect_presence_transition(self) -> bool:
        """Return True if someone just entered the room.

        Uses dual-threshold hysteresis on the RMS sliding window:
        older half was quiet (<0.003), recent samples are active (>0.008).
        """
        if len(self._rms_window) < 6:
            return False

        samples = list(self._rms_window)
        n = len(samples)

        # Older half
        older = [rms for _, rms in samples[: n // 2]]
        old_avg = sum(older) / len(older) if older else 0

        # Recent 3 samples
        recent = [rms for _, rms in samples[-3:]]
        new_avg = sum(recent) / len(recent) if recent else 0

        return old_avg < PRESENCE_RMS_QUIET and new_avg > PRESENCE_RMS_ACTIVE

    def _silence_duration(self) -> float:
        """How long has it been quiet (seconds)?"""
        if self._silence_start is None:
            return 0.0
        return time.time() - self._silence_start

    # ------------------------------------------------------------------
    # Trigger 1: Room presence greeting
    # ------------------------------------------------------------------

    def _check_presence_greeting(self) -> bool:
        now = time.time()

        # Cooldown
        if now - self._last_greeting_ts < PRESENCE_GREETING_COOLDOWN:
            return False

        # Must have been silent for 30+ min before the transition
        # We check if last_active_ts was a long time ago (room was empty)
        if self._last_active_ts > 0 and (now - self._last_active_ts) < PRESENCE_MIN_SILENCE_S:
            # Room hasn't been silent long enough — unless this is the first
            # active sample after a long gap
            pass

        # Need a fresh presence transition
        if not self._detect_presence_transition():
            return False

        # Verify the silence was long enough by checking when the last
        # active RMS sample was before the current burst
        samples = list(self._rms_window)
        if len(samples) < 6:
            return False

        # Find the last active sample in the older half
        older = samples[: len(samples) // 2]
        any_old_active = any(rms > PRESENCE_RMS_ACTIVE for _, rms in older)
        if any_old_active:
            return False  # room wasn't actually quiet in the older window

        # Also require no transcript for 30+ min (user wasn't here)
        if self._last_transcript_ts > 0 and (now - self._last_transcript_ts) < PRESENCE_MIN_SILENCE_S:
            return False

        if not self._can_speak():
            return False

        # Speak! Prefer a random pre-baked idle-prompt WAV (variety from
        # the 10k+ pool) over the tiny built-in _GREETINGS list.
        try:
            from voice.voicelines import random_voiceline
            picked = random_voiceline("idle_prompt")
        except Exception:
            picked = None
        if picked:
            wav, text, _style = picked
            print(f"[presence] Presence greeting (pre-baked): \"{text}\"")
            self._speaker.enqueue_wav(wav)
        else:
            greeting = random.choice(_GREETINGS)
            print(f"[presence] Presence greeting: \"{greeting}\"")
            self._speaker.enqueue(greeting)
        self._last_greeting_ts = now
        self._spend_budget()
        return True

    # ------------------------------------------------------------------
    # Trigger 2: Morning briefing
    # ------------------------------------------------------------------

    def _check_morning_briefing(self) -> bool:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Only between 6am-11am
        if not (MORNING_BRIEFING_HOUR_MIN <= now.hour < MORNING_BRIEFING_HOUR_MAX):
            return False

        # Once per day
        if self._last_morning_briefing_date == today:
            return False

        # Need presence transition (someone just arrived)
        if not self._detect_presence_transition():
            return False

        if not self._can_speak():
            return False

        # Get overnight Telegram summary
        summary = self._get_telegram_summary(hours=12)
        if not summary:
            # Nothing interesting overnight — skip
            self._last_morning_briefing_date = today
            return False

        # Generate briefing via LLM
        system = (
            "You are Aura, a personal AI assistant. Generate a brief 2-3 sentence "
            "morning summary of overnight Telegram activity. Be specific about "
            "groups, people, and topics. Warm but concise. No greetings — the user "
            "has already been greeted. No markdown."
        )
        prompt = f"Overnight Telegram activity (last 12 hours):\n{summary}"

        response = self._llm_direct(system, prompt)
        if not response:
            return False

        # Prepend a short greeting
        greeting = "Morning."
        full_text = f"{greeting} {response}"

        print(f"[presence] Morning briefing: \"{full_text[:80]}...\"")
        self._speaker.enqueue(full_text)
        self._last_morning_briefing_date = today
        self._spend_budget()
        return True

    # ------------------------------------------------------------------
    # Trigger 3: Idle commentary
    # ------------------------------------------------------------------

    def _check_idle_commentary(self) -> bool:
        now = time.time()

        # Cooldown
        if now - self._last_idle_comment_ts < IDLE_COMMENT_COOLDOWN_S:
            return False

        # Must have had a conversation within the session window (2h)
        if self._last_transcript_ts <= 0:
            return False
        silence = now - self._last_transcript_ts
        if silence < IDLE_COMMENT_MIN_SILENCE_S:
            return False  # not idle long enough
        if silence > IDLE_COMMENT_MAX_SILENCE_S:
            return False  # too long — user may have left

        # User must still be within session window
        if silence > IDLE_SESSION_WINDOW_S:
            return False

        # Presence validation: check recent RMS shows ambient activity
        # (user didn't leave the room)
        if not self._rms_window:
            return False
        recent_rms = [rms for _, rms in list(self._rms_window)[-3:]]
        avg_recent = sum(recent_rms) / len(recent_rms) if recent_rms else 0
        if avg_recent < PRESENCE_RMS_QUIET:
            return False  # room seems empty

        if not self._can_speak():
            return False

        # Get recent conversation context
        context = self._get_recent_context()
        if not context:
            return False

        system = (
            "You are Aura, a personal AI assistant. You had a conversation with the "
            "user a while ago and want to add a follow-up thought. Generate a single "
            "natural sentence — as if you just thought of something. Start with "
            "'Actually,' or 'You know,' or 'Going back to' or similar. No markdown. "
            "Be specific to the conversation topic. 1-2 sentences max."
        )
        prompt = f"Recent conversation:\n{context}\n\nGenerate a brief follow-up thought."

        response = self._llm_direct(system, prompt)
        if not response:
            return False

        print(f"[presence] Idle commentary: \"{response[:80]}...\"")
        self._speaker.enqueue(response)
        self._last_idle_comment_ts = now
        self._spend_budget()
        return True

    # ------------------------------------------------------------------
    # Trigger 4: Telegram alert
    # ------------------------------------------------------------------

    def _check_telegram_alert(self) -> bool:
        now = time.time()

        # Cooldown between alerts
        if now - self._last_telegram_alert_ts < TELEGRAM_ALERT_COOLDOWN_S:
            return False

        # Max alerts per hour
        cutoff = now - 3600
        self._telegram_alert_hour_ts = [
            t for t in self._telegram_alert_hour_ts if t > cutoff
        ]
        if len(self._telegram_alert_hour_ts) >= TELEGRAM_ALERT_MAX_HOUR:
            return False

        if not self._can_speak():
            return False

        event = self._get_high_signal_event()
        if not event:
            return False

        # Format alert (no LLM — pre-formatted)
        alert_text = self._format_alert(event)
        if not alert_text:
            return False

        print(f"[presence] Telegram alert: \"{alert_text[:80]}\"")
        self._speaker.enqueue(alert_text)
        self._last_telegram_alert_ts = now
        self._telegram_alert_hour_ts.append(now)
        self._spend_budget()
        return True

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _llm_direct(self, system: str, prompt: str) -> Optional[str]:
        """Call /chat-direct for short LLM-generated responses."""
        try:
            resp = requests.post(
                f"{LLM_URL}/chat-direct",
                json={
                    "prompt": prompt,
                    "system": system,
                    "max_tokens": 200,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"[presence] LLM /chat-direct HTTP {resp.status_code}")
                return None
            result = resp.json().get("response", "").strip()
            return result if result else None
        except Exception as e:
            print(f"[presence] LLM /chat-direct error: {e}")
            return None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_telegram_summary(self, hours: int = 12) -> Optional[str]:
        """Read recent engagement events from analytics.json + memory."""
        lines = []

        # 1. Read analytics.json engagement events
        analytics_path = DATA_DIR / "telegram" / "analytics.json"
        try:
            if analytics_path.exists():
                data = json.loads(analytics_path.read_text())
                events = data.get("engagement_events", [])
                cutoff = time.time() - (hours * 3600)
                recent = [e for e in events if e.get("ts", 0) > cutoff]
                if recent:
                    # Summarize event types
                    type_counts: dict[str, int] = {}
                    for e in recent:
                        t = e.get("type", "unknown")
                        type_counts[t] = type_counts.get(t, 0) + 1
                    summary_parts = [
                        f"{count} {etype.replace('_', ' ')} events"
                        for etype, count in type_counts.items()
                    ]
                    lines.append(f"Engagement: {', '.join(summary_parts)}")

                    # Include details from notable events
                    for e in recent[-5:]:
                        details = e.get("details", "")
                        if details:
                            lines.append(f"  - {e['type']}: {details}")
        except Exception as e:
            print(f"[presence] analytics read error: {e}")

        # 2. Check memory service for recent conversations
        try:
            resp = requests.get(
                f"{MEMORY_URL}/recent", params={"hours": hours}, timeout=5
            )
            if resp.status_code == 200:
                memories = resp.json()
                if isinstance(memories, list) and memories:
                    lines.append(f"Memory: {len(memories)} recent entries")
        except Exception:
            pass  # memory service may not be running

        return "\n".join(lines) if lines else None

    def _get_recent_context(self) -> Optional[str]:
        """Pull recent turn history from llm_client for idle commentary."""
        if not self._llm_client._turn_history:
            return None

        # Format last 2-3 turns
        turns = self._llm_client._turn_history[-3:]
        lines = []
        for user_msg, asst_msg in turns:
            lines.append(f"User: {user_msg}")
            lines.append(f"Aura: {asst_msg[:200]}")
        return "\n".join(lines) if lines else None

    def _get_high_signal_event(self) -> Optional[dict]:
        """Scan analytics.json for a new high-signal event to alert on."""
        analytics_path = DATA_DIR / "telegram" / "analytics.json"
        try:
            if not analytics_path.exists():
                return None
            data = json.loads(analytics_path.read_text())
            events = data.get("engagement_events", [])

            # Only look at events from the last 30 minutes
            cutoff = time.time() - 1800
            for event in reversed(events):
                ts = event.get("ts", 0)
                if ts < cutoff:
                    break
                if ts in self._seen_alert_ts:
                    continue
                if event.get("type") in _ALERT_EVENT_TYPES:
                    self._seen_alert_ts.add(ts)
                    # Cap the seen set to prevent unbounded growth
                    if len(self._seen_alert_ts) > 200:
                        oldest = sorted(self._seen_alert_ts)[:100]
                        self._seen_alert_ts -= set(oldest)
                    return event
        except Exception as e:
            print(f"[presence] analytics scan error: {e}")
        return None

    def _format_alert(self, event: dict) -> Optional[str]:
        """Format a high-signal event into a short spoken alert."""
        etype = event.get("type", "")
        details = event.get("details", "")

        if etype == "advocate_intro":
            # Extract name from details like "name=John, topics=ai, crypto"
            name = "someone new"
            if "name=" in details:
                name = details.split("name=")[1].split(",")[0].strip()
            return f"Heads up — {name} just showed up as a new advocate in Telegram."

        if etype == "dm_received":
            return "Heads up — someone just sent a direct message on Telegram."

        if etype == "warmth_milestone":
            return f"Heads up — a Telegram group just hit a warmth milestone. {details}"

        return None
