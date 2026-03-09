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
    LLM_URL,
    MEMORY_URL,
    PERPETUAL_IDLE_THRESHOLD_S,
    PERPETUAL_MAX_ITERATIONS,
    PERPETUAL_CONVERGENCE,
    PERPETUAL_BRIEFING_COOLDOWN_S,
    PERPETUAL_7B_MODEL_PATH,
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
                self._ruminate()
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
        print(f"[perpetual] Starting rumination cycle for {name}")
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
        """Non-streaming LLM call via /chat-tg."""
        try:
            resp = requests.post(
                f"{LLM_URL}/chat-tg",
                json={
                    "prompt": user_message,
                    "system_prompt": system_prompt,
                    "session_id": "perpetual",
                    "chat_id": "perpetual",
                    "stream": False,
                },
                timeout=120,
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                print(f"[perpetual] LLM HTTP {resp.status_code}")
                return ""
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[perpetual] LLM call error: {e}")
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
    # Model hot-swap (Phase 2 — called when 7B model is available)
    # ------------------------------------------------------------------

    def _swap_to_7b(self) -> bool:
        """Request the LLM container to swap to the 7B model."""
        if self._7b_active:
            return True
        if not os.path.exists(PERPETUAL_7B_MODEL_PATH):
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
        """Restore the original 1.5B model."""
        if not self._7b_active:
            return
        try:
            state.perpetual_model_swapping = True
            bus.emit("perpetual.model_swapping", model="1.5B")
            print("[perpetual] Restoring 1.5B model...")
            resp = requests.post(
                f"{LLM_URL}/model/restore",
                timeout=60,
            )
            if resp.status_code == 200:
                print("[perpetual] 1.5B model restored")
            else:
                print(f"[perpetual] Restore failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[perpetual] Restore error: {e}")
        finally:
            self._7b_active = False
            state.perpetual_model_swapping = False
            bus.emit("perpetual.model_ready")
