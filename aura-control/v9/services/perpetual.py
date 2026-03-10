"""
services.perpetual -- Aura Perpetual: background rumination engine.

When the user is idle (no conversation for IDLE_THRESHOLD seconds), Aura
begins a self-directed thinking loop:

    1. Gather seeds from conversation memory + RAG data
    2. Advisor generates insight about the user's goals
    3. Challenger critiques and identifies weaknesses
    4. Advisor refines based on critique
    5. Check convergence (semantic similarity between iterations)
    6. If insight is significant, generate a Presidential Daily Brief
    7. On next user interaction, deliver the briefing proactively

The engine yields immediately when the user speaks, and can optionally
hot-swap from a 1.5B to 7B model during idle for deeper thinking.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from core.bus import bus
from core.config import (
    BRIEFINGS_DIR,
    FARSIGHT_URL,
    FARSIGHT_TTS_STEPS,
    LLM_URL,
    MEMORY_URL,
    PERPETUAL_IDLE_THRESHOLD_S,
    PERPETUAL_MAX_ITERATIONS,
    PERPETUAL_CONVERGENCE,
    PERPETUAL_BRIEFING_COOLDOWN_S,
    PERPETUAL_QUESTION_COOLDOWN_S,
    PERPETUAL_7B_MODEL_PATH,
    VOICE_PROFILES_DIR,
)
from core.state import state


# ---------------------------------------------------------------------------
# System prompts for advisor / challenger roles
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM = """You are Aura, a thoughtful personal AI advisor. You are reflecting \
deeply on what you know about {name}. Based on their recent conversations, goals, \
and the documents they've shared with you, provide your most valuable insight or \
actionable advice.

Be specific. Reference concrete details from their life. Don't give generic \
self-help advice — give advice that could ONLY apply to this person.

Recent context:
{context}

RAG knowledge:
{rag_context}"""

CHALLENGER_SYSTEM = """You are a rigorous critical thinker reviewing advice given to \
{name}. Your job is to:

1. Identify weaknesses, blind spots, or assumptions in the advice
2. Suggest alternative perspectives the advisor hasn't considered
3. Point out what information is MISSING that would improve the advice
4. Be constructive but unsparing — mediocre advice helps no one

The advice to critique:
{advice}"""

REFINER_SYSTEM = """You are Aura, refining your advice to {name} based on valid \
criticism. Incorporate the strongest critiques while keeping your advice \
concrete and actionable. If the critique identified knowledge gaps, note \
them explicitly — you will ask {name} about these later.

Your original advice:
{advice}

Critique received:
{critique}

Produce your refined insight. Be concise (3-5 sentences max)."""

QUESTION_SYSTEM = """You are Aura, {name}'s AI advisor. You've been reflecting on recent \
conversations and have identified gaps in your understanding that would help you \
give better advice.

Generate ONE natural, conversational question to ask {name}. The question should:
- Reference something specific from their recent conversations or goals
- Help you understand their situation better so you can advise more effectively
- Sound like a thoughtful friend checking in, not an interview
- Be concise (1-2 sentences max)

Recent context:
{context}

RAG knowledge:
{rag_context}

Knowledge gaps you've already identified:
{gaps}

Respond with ONLY the question text — no preamble, no explanation, just the question \
as you would say it to {name}."""

BRIEFING_SYSTEM = """You are Aura, preparing a Presidential Daily Brief for {name}. \
Summarize your refined insight into a structured briefing.

Your refined insight:
{insight}

Respond in this exact JSON format (no markdown, no code blocks):
{{
    "insight": "Your key insight in 2-3 sentences, written as if speaking directly to {name}",
    "supporting_evidence": ["Evidence point 1 from their conversations/documents", "Evidence point 2"],
    "confidence": 0.0 to 1.0,
    "knowledge_gaps": ["What you'd need to know to give better advice"]
}}"""


# ---------------------------------------------------------------------------
# Perpetual engine
# ---------------------------------------------------------------------------

class Perpetual:
    """Background rumination daemon thread."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()  # set = paused
        self._7b_active = False
        self._use_farsight = bool(FARSIGHT_URL)
        self._last_question_ts = 0.0
        self._accumulated_gaps: list[str] = []  # knowledge gaps across cycles

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="perpetual"
        )
        self._thread.start()
        print("[perpetual] Engine started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        print("[perpetual] Engine stopped")

    # ------------------------------------------------------------------
    # Bus event handlers
    # ------------------------------------------------------------------

    def _wire_bus(self) -> None:
        """Subscribe to conversation events to pause/resume rumination."""
        bus.on("transcript.ready", self._on_conversation)
        bus.on("llm.started", self._on_conversation)
        bus.on("tts.started", self._on_conversation)
        bus.on("tts.finished", self._on_tts_done)
        bus.on("shutdown", self._on_shutdown)

    def _on_conversation(self, **_kw) -> None:
        """User is talking — pause rumination immediately."""
        state.last_conversation_ts = time.time()
        if state.perpetual_active:
            self._paused.set()
            state.perpetual_paused = True

    def _on_tts_done(self, **_kw) -> None:
        """TTS finished — update conversation timestamp."""
        state.last_conversation_ts = time.time()

    def _on_shutdown(self, **_kw) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Yield point — call between every LLM request
    # ------------------------------------------------------------------

    def _check_yield(self) -> bool:
        """Returns True if we should abort the current rumination cycle."""
        if self._stop.is_set():
            return True
        if self._paused.is_set():
            # Restore 1.5B if we swapped to 7B
            if self._7b_active:
                self._restore_model()
            # Wait until conversation is done + idle threshold passes
            print("[perpetual] Paused for conversation")
            while not self._stop.is_set():
                time.sleep(5.0)
                idle = time.time() - state.last_conversation_ts
                if idle >= PERPETUAL_IDLE_THRESHOLD_S and not state.playing:
                    break
            self._paused.clear()
            state.perpetual_paused = False
            if self._stop.is_set():
                return True
            print("[perpetual] Resumed after conversation")
            # Abort this cycle — start fresh
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._wire_bus()
        # Wait for initial boot to complete
        time.sleep(30.0)

        while not self._stop.is_set():
            try:
                self._idle_wait()
                if self._stop.is_set():
                    break

                # Decide: full briefing or proactive question?
                briefing_ready = (time.time() - state.last_briefing_ts) >= PERPETUAL_BRIEFING_COOLDOWN_S
                question_ready = (time.time() - self._last_question_ts) >= PERPETUAL_QUESTION_COOLDOWN_S

                if briefing_ready:
                    self._ruminate()
                elif question_ready and not state.pending_briefing:
                    self._generate_proactive_question()
                else:
                    # Nothing to do — sleep and check again
                    time.sleep(60.0)
            except Exception as e:
                print(f"[perpetual] Error in rumination cycle: {e}")
                time.sleep(60.0)

    def _idle_wait(self) -> None:
        """Block until the user has been idle for IDLE_THRESHOLD seconds."""
        while not self._stop.is_set():
            idle = time.time() - state.last_conversation_ts
            if idle >= PERPETUAL_IDLE_THRESHOLD_S and not state.playing:
                return
            time.sleep(10.0)

    # ------------------------------------------------------------------
    # Rumination cycle
    # ------------------------------------------------------------------

    def _ruminate(self) -> None:
        """Run one full advisor → challenger → refine cycle."""
        # Rate limit: max 1 briefing per 24h
        if (time.time() - state.last_briefing_ts) < PERPETUAL_BRIEFING_COOLDOWN_S:
            remaining = PERPETUAL_BRIEFING_COOLDOWN_S - (time.time() - state.last_briefing_ts)
            print(f"[perpetual] Briefing cooldown: {remaining/3600:.1f}h remaining")
            time.sleep(min(remaining, 300.0))  # check again in 5min or when cooldown ends
            return

        name = state.active_user_name or "the user"
        # Re-check Farsight availability each cycle
        self._use_farsight = bool(FARSIGHT_URL)
        mode = "Farsight" if self._use_farsight else "local Puck"
        print(f"[perpetual] Starting rumination cycle for {name} ({mode})")
        state.perpetual_active = True
        bus.emit("perpetual.started")

        try:
            # 1. Gather seeds
            context = self._gather_memory_context()
            rag_context = self._gather_rag_context()

            if not context and not rag_context:
                print("[perpetual] No context available — skipping cycle")
                state.perpetual_active = False
                time.sleep(300.0)  # retry in 5 min
                return

            # 2. Advisor generates initial insight
            if self._check_yield():
                return

            advisor_prompt = ADVISOR_SYSTEM.format(
                name=name, context=context, rag_context=rag_context
            )
            advice = self._llm_call(advisor_prompt, "What is your most valuable insight for this person right now?")
            if not advice or self._check_yield():
                return
            print(f"[perpetual] Advisor (iter 0): {advice[:100]}...")

            # 3. Iterate: challenge → refine → converge
            prev_advice = advice
            for iteration in range(1, PERPETUAL_MAX_ITERATIONS):
                if self._check_yield():
                    return

                # Challenger critiques
                challenger_prompt = CHALLENGER_SYSTEM.format(
                    name=name, advice=prev_advice
                )
                critique = self._llm_call(challenger_prompt, "What are the weaknesses in this advice?")
                if not critique or self._check_yield():
                    return
                print(f"[perpetual] Challenger (iter {iteration}): {critique[:100]}...")

                # Advisor refines
                if self._check_yield():
                    return
                refiner_prompt = REFINER_SYSTEM.format(
                    name=name, advice=prev_advice, critique=critique
                )
                refined = self._llm_call(refiner_prompt, "Refine your advice incorporating the valid critiques.")
                if not refined or self._check_yield():
                    return
                print(f"[perpetual] Refined (iter {iteration}): {refined[:100]}...")

                # Check convergence
                similarity = self._compute_similarity(prev_advice, refined)
                print(f"[perpetual] Convergence: {similarity:.3f} (threshold: {PERPETUAL_CONVERGENCE})")

                if similarity >= PERPETUAL_CONVERGENCE:
                    print(f"[perpetual] Converged after {iteration} iterations")
                    break

                prev_advice = refined

            # 4. Generate briefing
            if self._check_yield():
                return
            final_insight = refined if 'refined' in dir() else advice
            briefing = self._generate_briefing(name, final_insight)
            if briefing:
                self._save_briefing(briefing)

                # 5. Pre-synthesize audio on Farsight GPU (if available)
                if self._check_yield():
                    return
                wav_path = self._pre_synthesize_briefing(briefing)
                if wav_path:
                    briefing["audio_path"] = wav_path

                # Accumulate knowledge gaps for proactive questions
                gaps = briefing.get("knowledge_gaps", [])
                for g in gaps:
                    if g and g not in self._accumulated_gaps:
                        self._accumulated_gaps.append(g)
                # Keep only the most recent gaps
                self._accumulated_gaps = self._accumulated_gaps[-10:]

                state.pending_briefing = briefing
                state.last_briefing_ts = time.time()
                print(f"[perpetual] Briefing generated: {briefing.get('insight', '')[:80]}...")
                bus.emit("perpetual.briefing_ready", briefing=briefing)

        except Exception as e:
            print(f"[perpetual] Rumination error: {e}")
        finally:
            state.perpetual_active = False
            bus.emit("perpetual.finished")

    # ------------------------------------------------------------------
    # Proactive questioning
    # ------------------------------------------------------------------

    def _generate_proactive_question(self) -> None:
        """Generate a contextual question to ask the user proactively.

        Runs between briefing cooldowns. Uses accumulated knowledge gaps
        and recent context to ask something specific and useful.
        """
        name = state.active_user_name or "the user"
        self._use_farsight = bool(FARSIGHT_URL)
        mode = "Farsight" if self._use_farsight else "local"
        print(f"[perpetual] Generating proactive question for {name} ({mode})")

        try:
            # Gather context
            context = self._gather_memory_context()
            rag_context = self._gather_rag_context()

            if not context and not rag_context:
                print("[perpetual] No context for question — skipping")
                return

            if self._check_yield():
                return

            gaps_text = "\n".join(f"- {g}" for g in self._accumulated_gaps) if self._accumulated_gaps else "None identified yet."

            prompt = QUESTION_SYSTEM.format(
                name=name,
                context=context,
                rag_context=rag_context,
                gaps=gaps_text,
            )
            question = self._llm_call(prompt, f"What is one thing you want to ask {name} right now?")
            if not question or self._check_yield():
                return

            # Clean up — remove quotes, prefixes like "Hey Paul,"
            question = question.strip().strip('"').strip("'")

            print(f"[perpetual] Proactive question: \"{question[:100]}\"")

            # Store for delivery on next interaction
            state.pending_question = {
                "text": question,
                "generated_at": time.time(),
                "user_name": name,
                "gaps_used": list(self._accumulated_gaps[-3:]),
            }
            self._last_question_ts = time.time()
            bus.emit("perpetual.question_ready", question=question)

        except Exception as e:
            print(f"[perpetual] Question generation error: {e}")

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _gather_memory_context(self) -> str:
        """Pull recent conversations from the memory container."""
        try:
            resp = requests.get(
                f"{MEMORY_URL}/recent",
                params={"hours": 72, "limit": 20},
                timeout=5,
            )
            if resp.status_code != 200:
                return ""
            convos = resp.json().get("conversations", [])
            if not convos:
                return ""
            # Build context string from recent conversations
            lines = []
            for c in convos[:15]:
                text = c.get("text", "")
                ts = c.get("timestamp", "")
                if text:
                    lines.append(f"[{ts}] {text[:300]}")
            return "\n".join(lines)
        except Exception as e:
            print(f"[perpetual] Memory fetch error: {e}")
            return ""

    def _gather_rag_context(self) -> str:
        """Search RAG for user goals, plans, priorities."""
        name = state.active_user_name or "user"
        queries = [
            f"{name} goals plans priorities",
            f"{name} wants needs help with",
            "important upcoming deadline schedule",
        ]
        results = []
        for q in queries:
            try:
                resp = requests.post(
                    f"{LLM_URL}/cpu-faiss/search" if self._faiss_available() else f"{MEMORY_URL}/search",
                    json={"query": q, "k": 3, "threshold": 0.3},
                    timeout=5,
                )
                if resp.status_code == 200:
                    for r in resp.json().get("results", []):
                        text = r.get("text", r.get("content", ""))
                        if text and text not in results:
                            results.append(text[:400])
            except Exception:
                continue
        return "\n---\n".join(results[:8]) if results else ""

    def _faiss_available(self) -> bool:
        """Check if CPU FAISS is available on the LLM container."""
        try:
            resp = requests.get(f"{LLM_URL}/cpu-faiss/status", timeout=2)
            return resp.status_code == 200 and resp.json().get("total_chunks", 0) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _llm_call(self, system_prompt: str, user_message: str) -> str:
        """Direct LLM call — routes to Farsight (remote GPU) or local Puck."""
        url = f"{FARSIGHT_URL}/perpetual/chat" if self._use_farsight else f"{LLM_URL}/perpetual/chat"
        timeout = 60 if self._use_farsight else 300  # Farsight is fast
        try:
            resp = requests.post(
                url,
                json={
                    "prompt": user_message,
                    "system_prompt": system_prompt,
                    "max_tokens": 500,
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                print(f"[perpetual] LLM HTTP {resp.status_code} from {url}")
                # Fallback to local if Farsight fails
                if self._use_farsight:
                    print("[perpetual] Farsight unavailable, falling back to local")
                    self._use_farsight = False
                    return self._llm_call(system_prompt, user_message)
                return ""
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[perpetual] LLM call error ({url}): {e}")
            # Fallback to local if Farsight fails
            if self._use_farsight:
                print("[perpetual] Farsight unavailable, falling back to local")
                self._use_farsight = False
                return self._llm_call(system_prompt, user_message)
            return ""

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Estimate semantic similarity between two texts.

        Uses a simple word-overlap Jaccard similarity. For a small model on
        Jetson, this is faster and more reliable than loading another
        embedding model. The convergence threshold is tuned for this metric.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    # ------------------------------------------------------------------
    # Briefing generation
    # ------------------------------------------------------------------

    def _generate_briefing(self, name: str, insight: str) -> Optional[dict]:
        """Ask the LLM to structure the refined insight as a briefing."""
        prompt = BRIEFING_SYSTEM.format(name=name, insight=insight)
        raw = self._llm_call(prompt, "Generate the structured briefing now.")
        if not raw:
            return None

        # Parse JSON from LLM response
        try:
            # Try to find JSON in the response (LLM might wrap it in text)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                briefing = json.loads(raw[start:end])
            else:
                # Fallback: use the raw text as insight
                briefing = {
                    "insight": raw,
                    "supporting_evidence": [],
                    "confidence": 0.5,
                    "knowledge_gaps": [],
                }
        except json.JSONDecodeError:
            briefing = {
                "insight": raw,
                "supporting_evidence": [],
                "confidence": 0.5,
                "knowledge_gaps": [],
            }

        # Enrich with metadata
        briefing["date"] = datetime.now().strftime("%Y-%m-%d")
        briefing["generated_at"] = time.time()
        briefing["delivered"] = False
        briefing["user_name"] = name
        return briefing

    def _save_briefing(self, briefing: dict) -> None:
        """Persist briefing to disk."""
        BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = BRIEFINGS_DIR / f"{briefing['date']}.json"
        # Don't overwrite — append a sequence number if needed
        if path.exists():
            for i in range(1, 10):
                alt = BRIEFINGS_DIR / f"{briefing['date']}_{i}.json"
                if not alt.exists():
                    path = alt
                    break
        with open(path, "w") as f:
            json.dump(briefing, f, indent=2)
        print(f"[perpetual] Briefing saved: {path}")

    # ------------------------------------------------------------------
    # TTS pre-synthesis (Farsight GPU)
    # ------------------------------------------------------------------

    def _pre_synthesize_briefing(self, briefing: dict) -> Optional[str]:
        """Pre-render briefing audio on Farsight GPU for high-quality delivery.

        Sends the briefing text + user's voice sample to Farsight's TTS endpoint.
        Returns the path to the cached WAV file, or None if synthesis fails
        (in which case the Puck will fall back to local Kokoro TTS on delivery).
        """
        if not self._use_farsight:
            print("[perpetual] No Farsight — briefing will use local TTS on delivery")
            return None

        # Build the full briefing speech text (same format as delivery in aura.py)
        import datetime as _dt
        name = briefing.get("user_name", "friend")
        insight = briefing.get("insight", "")
        gaps = briefing.get("knowledge_gaps", [])
        hour = _dt.datetime.now().hour

        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        speech_text = (
            f"{greeting}, {name}. I have a brief prepared for you "
            f"about something that may impact your interests. {insight}"
        )
        if gaps:
            speech_text += f" I could refine this further if you could tell me about {gaps[0]}."

        # Load voice reference audio from the active user's enrollment sample
        voice_b64 = self._get_voice_sample_b64()

        # Send to Farsight for high-quality synthesis
        try:
            print(f"[perpetual] Pre-synthesizing briefing on Farsight ({FARSIGHT_TTS_STEPS} steps)...")
            payload = {
                "text": speech_text,
                "steps": FARSIGHT_TTS_STEPS,
            }
            if voice_b64:
                payload["voice_sample"] = voice_b64
            resp = requests.post(
                f"{FARSIGHT_URL}/perpetual/synthesize",
                json=payload,
                timeout=120,  # high-quality synthesis can take a while
            )
            if resp.status_code != 200:
                print(f"[perpetual] Farsight TTS failed: HTTP {resp.status_code}")
                return None

            # Save the returned WAV
            wav_path = str(BRIEFINGS_DIR / f"{briefing['date']}.wav")
            with open(wav_path, "wb") as f:
                f.write(resp.content)

            # Verify it's a valid WAV
            import wave
            with wave.open(wav_path, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
            print(f"[perpetual] Briefing audio pre-synthesized: {duration:.1f}s → {wav_path}")
            return wav_path

        except Exception as e:
            print(f"[perpetual] Farsight TTS error: {e}")
            return None

    def _get_voice_sample_b64(self) -> Optional[str]:
        """Load the active user's voice enrollment sample as base64-encoded WAV.

        The enrollment .npy files contain raw float32 audio at 16kHz.
        We convert to a proper WAV for the Farsight Chatterbox endpoint.
        """
        import base64
        import io
        import wave

        try:
            # Find the active user's profile
            profiles_file = VOICE_PROFILES_DIR / "profiles.json"
            if not profiles_file.exists():
                return None

            profiles = json.loads(profiles_file.read_text())
            active_name = (state.active_user_name or "").lower()

            # Find most recent profile matching the active user
            best_id = None
            best_ts = ""
            for pid, info in profiles.items():
                if info.get("name", "").lower().startswith(active_name[:4]) if active_name else False:
                    created = info.get("created", "")
                    if created > best_ts:
                        best_ts = created
                        best_id = pid

            if not best_id:
                print("[perpetual] No voice profile found for TTS cloning")
                return None

            # Load the raw audio enrollment sample
            import numpy as np
            sample_dir = VOICE_PROFILES_DIR / f"{best_id}_samples"
            if not sample_dir.exists():
                return None
            sample_files = sorted(sample_dir.glob("enroll_*.npy"))
            if not sample_files:
                return None

            audio = np.load(str(sample_files[-1]))  # most recent
            # Convert float32 audio to int16 WAV
            peak = float(np.max(np.abs(audio))) if audio.size else 1.0
            if peak > 1e-8:
                audio = audio / peak * 0.95
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            # Write to in-memory WAV
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())

            return base64.b64encode(buf.getvalue()).decode("ascii")

        except Exception as e:
            print(f"[perpetual] Voice sample load error: {e}")
            return None

    # ------------------------------------------------------------------
    # Model hot-swap (Phase 2 — called when 7B model is available)
    # ------------------------------------------------------------------

    def _7b_model_available(self) -> bool:
        """Check if the 7B model file exists inside the container."""
        try:
            resp = requests.get(f"{LLM_URL}/model/status", timeout=5)
            return resp.status_code == 200  # endpoint exists = swap supported
        except Exception:
            return False

    def _swap_to_7b(self) -> bool:
        """Request the LLM container to swap to the 7B model."""
        if self._7b_active:
            return True
        if not self._7b_model_available():
            return False
        try:
            state.perpetual_model_swapping = True
            bus.emit("perpetual.model_swapping", model="7B")
            print("[perpetual] Swapping to 7B model...")
            resp = requests.post(
                f"{LLM_URL}/model/swap",
                json={"model_path": PERPETUAL_7B_MODEL_PATH},
                timeout=60,
            )
            if resp.status_code == 200:
                self._7b_active = True
                print("[perpetual] 7B model loaded")
                return True
            else:
                print(f"[perpetual] 7B swap failed: HTTP {resp.status_code}")
                return False
        except Exception as e:
            print(f"[perpetual] 7B swap error: {e}")
            return False
        finally:
            state.perpetual_model_swapping = False

    def _restore_model(self) -> None:
        """Restore the original 1.5B model by restarting the LLM container.

        Hot-restore doesn't work because llama.cpp caches CUDA state at
        import time, and the 7B was loaded with CUDA disabled.  A container
        restart (10-20s) cleanly restores GPU-accelerated 1.5B.
        """
        if not self._7b_active:
            return
        try:
            state.perpetual_model_swapping = True
            bus.emit("perpetual.model_swapping", model="1.5B")
            print("[perpetual] Restoring 1.5B model (container restart)...")
            import subprocess
            subprocess.run(
                ["docker", "restart", "setup-llm-generic-1"],
                timeout=30, capture_output=True,
            )
            # Wait for container to be ready
            import time as _time
            for _ in range(30):
                try:
                    r = requests.get(f"{LLM_URL}/model/status", timeout=2)
                    if r.status_code == 200 and r.json().get("loaded"):
                        print("[perpetual] 1.5B model restored")
                        break
                except Exception:
                    pass
                _time.sleep(5)
        except Exception as e:
            print(f"[perpetual] Restore error: {e}")
        finally:
            self._7b_active = False
            state.perpetual_model_swapping = False
            bus.emit("perpetual.model_ready")
