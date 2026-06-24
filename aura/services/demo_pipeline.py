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

    def _wait_for_speech(self, est_seconds: float) -> None:
        """Wait for enqueued speech to finish playing.

        Uses word-count estimate as a floor, then polls is_playing()
        with a grace window to handle inter-clause gaps.
        """
        deadline = time.time() + est_seconds
        while time.time() < deadline and not self._stop.is_set():
            time.sleep(1.0)
        grace_end = time.time() + 5.0
        while time.time() < grace_end and not self._stop.is_set():
            if self._speaker.is_playing():
                grace_end = time.time() + 3.0
            time.sleep(0.5)

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
        """Present the 4-part strategic brief using pre-written content.

        Pre-baked text eliminates LLM latency — each section starts
        playing immediately with no dead air between sections.
        """

        sections = [
            {
                "label": "Executive Summary",
                "text": (
                    "Let me start with the big picture on Meridian Health Group. "
                    "You're looking at a company with twelve thousand eight hundred "
                    "and forty seven practitioners operating across four hundred "
                    "and eighty two GP practices, serving four point two million "
                    "registered patients across England. Revenue came in at one "
                    "point nine four billion pounds for the current fiscal year, "
                    "with an EBITDA margin of eleven point eight percent, just "
                    "shy of the twelve percent target.\n\n"
                    "Now, the headline here is this. Meridian is at an inflection "
                    "point. The core primary care business is solid, but there "
                    "are three forces converging that demand your attention this "
                    "quarter: a workforce crisis that's accelerating faster than "
                    "your retention programs, a digital transformation decision "
                    "that will define your competitive position for the next five "
                    "years, and an AI governance question that could become a "
                    "reputational liability if not addressed proactively.\n\n"
                    "The thesis is straightforward. Meridian needs to shift from "
                    "a volume-driven growth model to a margin-optimised portfolio "
                    "strategy, and the CDC expansion is your best lever to do that."
                ),
            },
            {
                "label": "Market Landscape",
                "text": (
                    "Turning to the external landscape. NHS funding remains under "
                    "severe pressure, and the recent Carr-Hill formula revisions "
                    "are going to hit your bottom line directly. We're modelling "
                    "a negative two point eight million pound EBITDA impact from "
                    "those changes alone.\n\n"
                    "The workforce picture is where it gets critical. Nearly "
                    "one in five of your GPs, nineteen point six percent, are "
                    "within five years of retirement. That's the retirement cliff "
                    "everyone's been warning about, and it's now hitting your "
                    "planning horizon. On top of that, Band five nurse vacancy "
                    "rates are running at ten point eight percent, and your "
                    "agency spend has ballooned to ninety four million pounds, "
                    "that's seven point nine percent of total staff costs.\n\n"
                    "But here's the opportunity. Your Community Diagnostic Centres "
                    "are delivering an eighteen percent margin, the highest of any "
                    "service line. The forty two million pound expansion programme "
                    "over two years could meaningfully shift your margin profile, "
                    "especially as NHS England continues to push diagnostic capacity "
                    "into community settings."
                ),
            },
            {
                "label": "Strategic Risks",
                "text": (
                    "Now let me walk you through the risk landscape. First, AI "
                    "governance. You have four incidents currently under review "
                    "related to the Hera Health triage system. Given the current "
                    "regulatory climate around AI in healthcare, this needs "
                    "executive-level attention immediately. One high-profile "
                    "adverse event linked to automated triage could set back your "
                    "entire digital strategy.\n\n"
                    "Second, the build versus partner decision on your digital "
                    "transformation. You're spending fourteen million pounds this "
                    "fiscal year, and the question of whether to build Meridian "
                    "Connect in-house or pivot to a partner solution is overdue. "
                    "Every quarter you delay increases switching costs.\n\n"
                    "Third, the competition and markets authority. You're "
                    "approaching the fifteen percent market share threshold in "
                    "three regions. Cross that line and you trigger enhanced "
                    "scrutiny, which creates both operational drag and political "
                    "risk. This is not just a legal issue, it's a reputational "
                    "one given the current discourse around NHS privatisation."
                ),
            },
            {
                "label": "Recommendations",
                "text": (
                    "Here's what I recommend you prioritise this quarter. First, "
                    "fast-track the GP retention package. With nearly a fifth of "
                    "your workforce approaching retirement, every month of delay "
                    "costs you practitioners you can't replace. The data suggests "
                    "a targeted retention programme could reduce attrition by "
                    "thirty to forty percent in the highest-risk cohort.\n\n"
                    "Second, greenlight the CDC expansion but phase it. Start with "
                    "the eight highest-margin locations, validate the operating "
                    "model, then scale. This protects your capital while proving "
                    "the thesis.\n\n"
                    "Third, suspend the Hera Health AI triage system pending a full "
                    "clinical safety review. The downside risk here far outweighs "
                    "the operational efficiency gains. You can restart it with "
                    "proper guardrails in place.\n\n"
                    "And fourth, on Meridian Connect, my recommendation is to pivot "
                    "to a partner. The build costs are escalating and you need to "
                    "be live within eighteen months to remain competitive.\n\n"
                    "I'm ready to dive deeper into any of these areas. Just tell "
                    "me which topic you'd like to explore further."
                ),
            },
        ]

        self._brief_segments = []
        for i, section in enumerate(sections):
            if self._stop.is_set():
                return

            progress = i / len(sections)
            if i == 0:
                self._set_stage(DemoStage.ANALYZING, 0.5,
                                f"Composing: {section['label']}")
                time.sleep(5.0)
                self._set_stage(DemoStage.BRIEFING, 0.0, "Strategic Briefing")
            else:
                self._set_stage(DemoStage.BRIEFING, progress,
                                f"Part {i+1}: {section['label']}")

            text = section["text"]
            self._brief_segments.append(text)
            self._emit_tokens(len(text.split()) * 2,
                              f"brief:{section['label']}")

            bus.emit("demo.brief_segment",
                     index=i, total=len(sections),
                     text=text[:100])

            print(f"[demo] Brief part {i+1}/{len(sections)}: "
                  f"{section['label']} ({len(text.split())} words)")

            self._speaker.enqueue(text)

            est_seconds = len(text.split()) / 2.3
            print(f"[demo] Waiting ~{est_seconds:.0f}s for section audio")
            self._wait_for_speech(est_seconds)

        self._set_stage(DemoStage.BRIEFING, 1.0, "Briefing complete")
        time.sleep(2.0)

        total_words = sum(len(s.split()) for s in self._brief_segments)
        print(f"[demo] Full brief: {total_words} words, "
              f"~{total_words / 150:.1f} min at 150 wpm")


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

            est_seconds = len(narration.split()) / 2.3
            self._wait_for_speech(est_seconds)
            time.sleep(3.0)

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
