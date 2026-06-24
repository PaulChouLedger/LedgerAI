"""
services.demo_pipeline -- Orchestrator for the healthcare executive demo.

State machine that runs after boot.complete, driving 6 stages:
  1. FILE_DROP   — wait for files via BLE/scp, animate arrivals
  2. ANALYZING   — trigger RAG ingest, generate multi-part LLM brief
  3. BRIEFING    — play pre-synthesized 3-minute narrated brief
  4. FOLLOWUP    — identify high-risk patients, offer to schedule with Dr.
  5. QA          — normal voice pipeline with RAG context
  6. IDLE        — demo complete, sitting in normal mode

Emits bus events consumed by gui/demo_renderer.py:
  demo.stage     — stage name + progress
  demo.file      — new file ingested (name, index, total)
  demo.chunk     — RAG chunk indexed (count, total)
  demo.brief_segment — narration segment started (index, total, text)
  demo.kpi       — KPI card to show (label, value, unit)
  demo.tokens    — $LEDGER tokens consumed (count, delta, operation)
  demo.followup  — follow-up patient identified (name, risk, action)
  demo.qa_ready  — Q&A mode active
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from core.bus import bus
from core.state import state


class DemoStage(Enum):
    WAITING = auto()
    FILE_DROP = auto()
    ANALYZING = auto()
    BRIEFING = auto()
    FOLLOWUP = auto()
    QA = auto()
    IDLE = auto()


FOLLOWUP_PATIENTS = [
    {
        "name": "Margaret Whitfield",
        "age": 72,
        "risk": "Critical",
        "conditions": "Uncontrolled T2DM (HbA1c 74), CKD stage 3b, polypharmacy (11 meds)",
        "action": "Urgent medication review — renal dosing adjustment needed",
        "doctor": "Dr. Priya Sharma",
        "specialty": "Diabetes & Renal",
    },
    {
        "name": "James Okonkwo",
        "age": 58,
        "risk": "High",
        "conditions": "COPD Gold 3, 4 exacerbations in 12 months, current smoker",
        "action": "Pulmonary rehab referral + smoking cessation escalation",
        "doctor": "Dr. Rachel Chen",
        "specialty": "Respiratory",
    },
    {
        "name": "Sarah Pemberton",
        "age": 45,
        "risk": "High",
        "conditions": "Severe anxiety + depression, 3 missed Talking Therapy appointments",
        "action": "Welfare check + crisis plan review",
        "doctor": "Dr. Anil Gupta",
        "specialty": "Mental Health",
    },
    {
        "name": "Robert Kavanagh",
        "age": 67,
        "risk": "Moderate-High",
        "conditions": "AF on warfarin, 2 INR readings out of range, falls risk",
        "action": "Switch to DOAC assessment + falls prevention referral",
        "doctor": "Dr. Emily Torres",
        "specialty": "Cardiovascular",
    },
    {
        "name": "Fatima Al-Rashid",
        "age": 34,
        "risk": "Moderate",
        "conditions": "Gestational diabetes (previous pregnancy), BMI 31, pre-diabetic HbA1c",
        "action": "Diabetes prevention programme enrolment + 3-month HbA1c recheck",
        "doctor": "Dr. Priya Sharma",
        "specialty": "Diabetes & Women's Health",
    },
]


# KPIs extracted from the narration, shown as overlay cards during the brief.
# Each tuple: (time_offset_fraction, label, value, unit, duration_fraction)
DEMO_KPIS = [
    (0.05, "Practitioners", "12,847", "", 0.08),
    (0.10, "GP Practices", "482", "", 0.08),
    (0.15, "Revenue", "£1.94B", "FY25-26", 0.08),
    (0.22, "EBITDA Margin", "11.8%", "target 12%", 0.07),
    (0.30, "GP Retirement Risk", "19.6%", "within 5yr", 0.08),
    (0.38, "Agency Spend", "£94M", "7.9% of staff", 0.08),
    (0.48, "CDC Margin", "18%", "highest line", 0.07),
    (0.55, "Nurse Vacancy", "10.8%", "Band 5", 0.07),
    (0.62, "Carr-Hill Impact", "-£2.8M", "EBITDA", 0.07),
    (0.70, "Digital Spend", "£14M", "FY25-26", 0.07),
    (0.78, "AI Incidents", "4", "under review", 0.07),
    (0.85, "Market Share Risk", "15%", "CMA threshold", 0.07),
    (0.92, "CDC Expansion", "£42M", "capex 2yr", 0.07),
]


class DemoPipeline:
    """Healthcare demo orchestrator. Call start() after boot.complete."""

    def __init__(self, speaker, llm_client):
        self._speaker = speaker
        self._llm = llm_client
        self._stage = DemoStage.WAITING
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._files_received: list[str] = []
        self._last_file_time = 0.0
        self._brief_segments: list[str] = []
        self._tokens_used = 0

        self._input_dir = Path(__file__).resolve().parents[2] / "data" / "input"

        bus.on("ble.file_received", self._on_file)
        bus.on("ingest.file_processed", self._on_file)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="demo-pipeline"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stage(self) -> DemoStage:
        return self._stage

    def _set_stage(self, stage: DemoStage, progress: float = 0.0,
                   text: str = "") -> None:
        self._stage = stage
        bus.emit("demo.stage", stage=stage.name, progress=progress, text=text)
        print(f"[demo] Stage: {stage.name} — {text}")

    def _emit_tokens(self, delta: int, operation: str) -> None:
        self._tokens_used += delta
        bus.emit("demo.tokens", count=self._tokens_used,
                 delta=delta, operation=operation)

    def _on_file(self, path: str = "", name: str = "", **_kw) -> None:
        fname = name or os.path.basename(path or "")
        if fname and fname not in self._files_received:
            self._files_received.append(fname)
            self._last_file_time = time.time()
            bus.emit("demo.file", name=fname,
                     index=len(self._files_received),
                     total=len(self._files_received))
            print(f"[demo] File received: {fname} "
                  f"({len(self._files_received)} total)")

    def _run(self) -> None:
        print("[demo] Pipeline started")
        time.sleep(1.0)

        self._run_file_drop()
        if self._stop.is_set():
            return

        self._run_analysis()
        if self._stop.is_set():
            return

        self._run_followup()
        if self._stop.is_set():
            return

        self._run_qa()

    # ------------------------------------------------------------------
    # Stage 1: File drop
    # ------------------------------------------------------------------

    def _run_file_drop(self) -> None:
        self._set_stage(DemoStage.FILE_DROP, 0.0, "Waiting for files")

        self._speaker.enqueue(
            "I'm ready for your data. Drop your files and I'll take it "
            "from here. Just say analyze when you're done."
        )

        self._scan_existing_files()

        analyze_requested = threading.Event()

        def _on_transcript(text: str = "", **_kw):
            lower = text.lower().strip().rstrip(".,!?")
            triggers = (
                "analyze", "analyse", "start", "go", "begin",
                "that's it", "that's all", "done", "ready",
                "process", "run it", "let's go",
            )
            if any(t in lower for t in triggers):
                analyze_requested.set()

        bus.on("transcript.unfiltered", _on_transcript)

        idle_since = time.time()
        announced = False

        try:
            while not self._stop.is_set() and not analyze_requested.is_set():
                prev_n = len(self._files_received)
                self._scan_existing_files()
                n = len(self._files_received)

                if n > prev_n:
                    idle_since = time.time()

                self._set_stage(
                    DemoStage.FILE_DROP,
                    min(0.9, n * 0.05),
                    f"{n} file{'s' if n != 1 else ''} received"
                )

                if n > 0 and not announced:
                    self._speaker.enqueue(
                        f"I've received {n} file{'s' if n != 1 else ''}. "
                        "Say analyze when you're ready, or keep dropping files."
                    )
                    announced = True

                # Auto-advance after 30s idle with files present
                if n > 0 and time.time() - idle_since > 30.0:
                    print(f"[demo] Auto-advancing (30s idle, {n} files)")
                    break

                analyze_requested.wait(timeout=2.0)
        finally:
            bus.off("transcript.unfiltered", _on_transcript)

        n = len(self._files_received)
        self._set_stage(DemoStage.FILE_DROP, 1.0,
                        f"{n} files ready for analysis")
        print(f"[demo] File drop complete: {n} files")

    def _scan_existing_files(self) -> None:
        """Pick up files already in data/input/."""
        if not self._input_dir.exists():
            return
        for f in sorted(self._input_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".csv", ".json", ".pdf", ".md"):
                name = f.name
                if name not in self._files_received:
                    self._files_received.append(name)
                    self._last_file_time = time.time()
                    bus.emit("demo.file", name=name,
                             index=len(self._files_received),
                             total=len(self._files_received))
                    self._emit_tokens(24, f"index:{name[:20]}")

    # ------------------------------------------------------------------
    # Stage 2: RAG analysis + brief generation
    # ------------------------------------------------------------------

    def _run_analysis(self) -> None:
        n = len(self._files_received)
        self._set_stage(DemoStage.ANALYZING, 0.0, "Indexing documents")

        self._speaker.enqueue(
            f"Processing {n} documents now. I'm cross-referencing your "
            "financial data, market reports, and operational metrics. "
            "I'll begin presenting each section as soon as it's ready."
        )

        self._wait_for_indexing()
        self._set_stage(DemoStage.ANALYZING, 0.2, "Generating strategic brief")

        kpi_thread = threading.Thread(
            target=self._emit_kpis, daemon=True, name="demo-kpis"
        )
        kpi_thread.start()

        self._generate_and_present_brief()

    def _wait_for_indexing(self) -> None:
        """Wait for FAISS ingest to finish processing any pending files."""
        time.sleep(3.0)

        try:
            import sys
            repo_root = Path(__file__).resolve().parents[2]
            llm_dir = str(repo_root / "containers" / "llm")
            if llm_dir not in sys.path:
                sys.path.insert(0, llm_dir)
            from rag import get_rag_client
            client = get_rag_client()
            if client:
                test = client.search("healthcare", k=1)
                n = getattr(client, 'total_chunks', None)
                if n:
                    bus.emit("demo.chunk", count=n, total=n)
                    self._emit_tokens(n * 48, "embedding")
                    print(f"[demo] RAG index: {n} chunks")
                else:
                    self._emit_tokens(len(test) * 48, "embedding")
                    print(f"[demo] RAG search returned {len(test)} results")
        except Exception as e:
            print(f"[demo] RAG check: {e}")

    def _generate_and_present_brief(self) -> None:
        """Generate each brief section and speak it immediately.

        LLM runs on GPU, Piper TTS on CPU — they overlap, so the user
        hears section N while section N+1 is being generated.
        """

        parts = [
            {
                "label": "Executive Summary",
                "prompt": (
                    "You are Aura, an executive intelligence advisor. Based on "
                    "the uploaded documents about Meridian Health Group (a UK "
                    "healthcare company with 12,847 practitioners and £1.94B "
                    "revenue), write the opening of a strategic briefing for "
                    "the CEO. Cover: company overview, current position, and "
                    "the one-line thesis of what they need to focus on. "
                    "Write 3-4 paragraphs, conversational tone, as if speaking. "
                    "No bullet points, no JSON."
                ),
                "filler": (
                    "I'm pulling together your executive summary now. "
                    "Cross-referencing Meridian's revenue figures, practitioner "
                    "headcount, and EBITDA margins against your operational "
                    "data."
                ),
            },
            {
                "label": "Market Landscape",
                "prompt": (
                    "Continue the strategic briefing. Now cover the external "
                    "landscape: NHS funding pressures, Carr-Hill formula changes, "
                    "workforce crisis (GP retirement cliff, nursing vacancies), "
                    "and the CDC expansion opportunity. Reference specific "
                    "numbers from the documents. 3-4 paragraphs, spoken tone."
                ),
                "filler": (
                    "Now mapping the external landscape. Analysing NHS "
                    "funding allocation data and workforce pipeline numbers."
                ),
            },
            {
                "label": "Strategic Risks",
                "prompt": (
                    "Continue the briefing. Cover the key risks: AI governance "
                    "concerns (Hera Health incidents), digital transformation "
                    "build-vs-partner decision, agency spend at £94M, and "
                    "political/reputational risk from CMA market share "
                    "thresholds. Be specific about the numbers and decisions "
                    "required. 3-4 paragraphs, spoken tone."
                ),
                "filler": (
                    "Assessing your risk exposure now. Correlating incident "
                    "reports with governance frameworks."
                ),
            },
            {
                "label": "Recommendations",
                "prompt": (
                    "Conclude the briefing with actionable recommendations. "
                    "What should the CEO prioritize this quarter? Cover: the "
                    "GP retention package decision, CDC expansion timeline, "
                    "Hera Health AI triage suspension question, and the "
                    "Meridian Connect build-vs-pivot decision. End with an "
                    "offer to dive deeper into any topic. 3-4 paragraphs, "
                    "spoken tone."
                ),
                "filler": (
                    "Final section — building your recommendation framework."
                ),
            },
        ]

        self._brief_segments = []
        for i, part in enumerate(parts):
            if self._stop.is_set():
                return

            if i == 0:
                self._speaker.enqueue(part["filler"])
                self._set_stage(DemoStage.ANALYZING, 0.3,
                                f"Generating: {part['label']}")
            else:
                self._set_stage(DemoStage.BRIEFING,
                                (i - 1) / len(parts),
                                f"Generating: {part['label']}")

            print(f"[demo] Generating brief part {i+1}/{len(parts)}: "
                  f"{part['label']}")
            text = self._llm_with_rag(part["prompt"])

            if text:
                text = re.sub(r"[\x00-\x1f\x7f]", " ", text).strip()
                self._brief_segments.append(text)
                token_est = len(text.split()) * 2
                self._emit_tokens(token_est, f"llm:{part['label']}")
                print(f"[demo] Brief part {i+1}/{len(parts)}: "
                      f"{len(text)} chars, ~{len(text.split())} words")
            else:
                text = (f"I wasn't able to generate the "
                        f"{part['label'].lower()} section. Let me move on.")
                self._brief_segments.append(text)

            bus.emit("demo.brief_segment",
                     index=i, total=len(parts),
                     text=text[:100])

            if i == 0:
                self._set_stage(DemoStage.BRIEFING, 0.0,
                                "Strategic Briefing")

            self._speaker.enqueue(text)

            if i < len(parts) - 1:
                self._speaker.enqueue(parts[i + 1]["filler"])

        self._set_stage(DemoStage.BRIEFING, 0.9, "Finishing briefing")

        while self._speaker.is_playing() and not self._stop.is_set():
            time.sleep(0.3)

        self._set_stage(DemoStage.BRIEFING, 1.0, "Briefing complete")
        time.sleep(2.0)

        total_words = sum(len(s.split()) for s in self._brief_segments)
        print(f"[demo] Full brief: {total_words} words, "
              f"~{total_words / 150:.1f} min at 150 wpm")

    def _llm_with_rag(self, prompt: str) -> Optional[str]:
        """Query LLM with embedded context (skip slow RAG pre-filter)."""
        try:
            from voice.llm_engine import llm_engine

            context = (
                "Key facts about Meridian Health Group PLC:\n"
                "- UK healthcare company, 12,847 practitioners across 482 GP practices\n"
                "- Revenue £1.94B (FY25-26), EBITDA margin 11.8% (target 12%)\n"
                "- 4.2M registered patients across England\n"
                "- GP retirement risk: 19.6% within 5 years (retirement cliff)\n"
                "- Agency spend £94M (7.9% of staff costs)\n"
                "- Nurse vacancy rate 10.8% (Band 5)\n"
                "- CDC (Community Diagnostic Centre) margin 18% — highest service line\n"
                "- CDC expansion capex £42M over 2 years\n"
                "- Carr-Hill formula impact: -£2.8M EBITDA\n"
                "- Digital transformation spend £14M FY25-26\n"
                "- AI incidents: 4 under review (Hera Health triage system)\n"
                "- CMA market share threshold risk at 15%\n"
                "- 48 practices in Group division + NHS elective surgery + prison healthcare\n"
                "- Specialties: primary care, diagnostics, urgent care, mental health, pharma trials\n"
                "- Key strategic challenges: workforce crisis, digital transformation\n"
                "  build-vs-partner, CDC expansion timing, AI governance"
            )

            system_msg = (
                "You are Aura, a sophisticated AI executive advisor. "
                "Speak naturally as if delivering a live briefing to a "
                "CEO. Use specific numbers and facts from the context. "
                "Never use markdown, bullet points, or formatting."
            )

            user_msg = f"{context}\n\n{prompt}"

            response = llm_engine.chat_direct(
                system=system_msg,
                user=user_msg,
                max_tokens=600,
                temperature=0.7,
            )
            return response
        except Exception as e:
            print(f"[demo] LLM generation failed: {e}")
            return None

    def _emit_kpis(self) -> None:
        """Emit KPI events spread over ~8 minutes (4 sections x ~2 min)."""
        brief_est = 480.0

        start = time.time()
        emitted = set()

        while not self._stop.is_set():
            elapsed = time.time() - start
            frac = elapsed / brief_est

            if frac > 1.2:
                break

            for j, (t_frac, label, value, unit, dur_frac) in enumerate(DEMO_KPIS):
                if j not in emitted and frac >= t_frac:
                    emitted.add(j)
                    bus.emit("demo.kpi",
                             label=label, value=value, unit=unit,
                             duration=dur_frac * brief_est,
                             index=j, total=len(DEMO_KPIS))

            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Stage 4: Patient follow-up
    # ------------------------------------------------------------------

    def _run_followup(self) -> None:
        self._set_stage(DemoStage.FOLLOWUP, 0.0, "Patient Risk Analysis")

        self._speaker.enqueue(
            "Now let me flag some patients that need immediate attention. "
            "I've cross-referenced your practice data against clinical "
            "guidelines and identified several high-risk individuals."
        )

        time.sleep(3.0)

        for i, patient in enumerate(FOLLOWUP_PATIENTS):
            if self._stop.is_set():
                return

            progress = (i + 1) / len(FOLLOWUP_PATIENTS)
            self._set_stage(DemoStage.FOLLOWUP, progress,
                            f"Patient {i+1} of {len(FOLLOWUP_PATIENTS)}")

            bus.emit("demo.followup_patient",
                     name=patient["name"],
                     age=patient["age"],
                     risk=patient["risk"],
                     conditions=patient["conditions"],
                     action=patient["action"],
                     doctor=patient["doctor"],
                     specialty=patient["specialty"],
                     index=i, total=len(FOLLOWUP_PATIENTS))

            self._emit_tokens(340, f"risk:{patient['name'].split()[0]}")

            narration = (
                f"{patient['name']}, age {patient['age']}. "
                f"Risk level: {patient['risk']}. "
                f"{patient['conditions']}. "
                f"Recommended action: {patient['action']}. "
                f"I can schedule this with {patient['doctor']} in "
                f"{patient['specialty']}."
            )
            self._speaker.enqueue(narration)

            time.sleep(0.5)
            while self._speaker.is_playing() and not self._stop.is_set():
                time.sleep(0.3)
            time.sleep(5.0)

        self._speaker.enqueue(
            "Those are the priority patients. Would you like me to "
            "schedule follow-up appointments with the recommended "
            "specialists, or shall we move to questions?"
        )

        schedule_requested = threading.Event()
        qa_requested = threading.Event()

        def _on_transcript(text: str = "", **_kw):
            lower = text.lower().strip().rstrip(".,!?")
            if any(w in lower for w in ("schedule", "book", "yes", "go ahead", "do it")):
                schedule_requested.set()
            elif any(w in lower for w in ("question", "skip", "no", "move on", "next")):
                qa_requested.set()

        bus.on("transcript.unfiltered", _on_transcript)
        try:
            while (not self._stop.is_set()
                   and not schedule_requested.is_set()
                   and not qa_requested.is_set()):
                time.sleep(1.0)
                if not self._speaker.is_playing():
                    break
            time.sleep(8.0)

            if schedule_requested.is_set():
                self._speaker.enqueue(
                    "Done. I've queued appointment requests for all five "
                    "patients with their respective specialists. "
                    "Confirmation notifications will be sent to the practice "
                    "managers within the hour. Now, what questions do you have?"
                )
                self._emit_tokens(120, "scheduling")
                time.sleep(1.0)
                while self._speaker.is_playing() and not self._stop.is_set():
                    time.sleep(0.3)
        finally:
            bus.off("transcript.unfiltered", _on_transcript)

        self._set_stage(DemoStage.FOLLOWUP, 1.0, "Follow-up complete")
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # Stage 5: Q&A
    # ------------------------------------------------------------------

    def _run_qa(self) -> None:
        self._set_stage(DemoStage.QA, 0.0, "Ready for questions")

        self._speaker.enqueue(
            "That concludes the strategic briefing. I'm ready for your "
            "questions. You can ask me about any aspect of Meridian's "
            "operations, the UK healthcare landscape, or specific "
            "recommendations."
        )

        bus.emit("demo.qa_ready")
        self._set_stage(DemoStage.QA, 1.0, "Q&A active")
        print("[demo] Q&A mode active — pipeline complete")
