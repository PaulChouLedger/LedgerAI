#!/usr/bin/env python3
"""
Aura 2.0 -- Master entry point.

Boot-first startup: shows Falcon animation while containers load,
optionally enrolls the user's voice, then crossfades to the normal GUI.

    python3 aura-control/v2/aura.py
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import warnings

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

# Ensure package imports resolve from v8/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("DISPLAY", ":0")

from core.bus import bus          # noqa: E402
from core.state import state      # noqa: E402
from core import config           # noqa: E402
from services.memlog import memlog  # noqa: E402


def main() -> int:
    memlog("process start", include_docker=True)

    # 1. Load persisted settings
    state.load()
    print(f"[aura] settings loaded  (llm={state.llm_mode}, tts={state.tts_engine})")
    print(f"[aura] dock: {state.dock}")

    # 2. Qt application (create early so window can render during boot)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    QApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    QApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    app = QApplication(sys.argv)
    memlog.delta("PyQt5 app created")

    # 3. GUI window in boot mode (falcon animation)
    from gui.window import AuraWindow
    window = AuraWindow(boot_mode=True)
    window.show()
    memlog.delta("GUI window shown")

    # 4. Start boot orchestrator (runs on daemon thread)
    from boot.orchestrator import BootOrchestrator
    orchestrator = BootOrchestrator()
    orchestrator.start()

    # 5. When boot completes: discover complications, start voice pipeline,
    #    transition GUI to normal mode.
    #    Closures capture `window`, `app` etc.
    _voice_started = threading.Event()

    def _on_boot_complete(**_kw):
        """Fires on the orchestrator thread — kick work back to main."""
        print("[aura] boot.complete received — starting normal mode")
        memlog.delta("boot.complete received")

        # Discover & register all complications
        from gui.complications import registry
        registry.load()
        print(f"[aura] complications: {[c.name for c in registry.get_all()]}")

        # Voice pipeline
        from voice.listener import Listener
        from voice.speaker import Speaker
        from voice.llm_client import LLMClient

        llm_client = LLMClient()
        listener = Listener()
        speaker = Speaker()

        # Wire: transcript → thinking filler + LLM → speaker
        def _on_transcript(text: str = "", **_kw2):
            if text:
                state.last_conversation_ts = time.time()

                # Deliver pending briefing on first interaction
                if state.pending_briefing and not state.perpetual_paused:
                    briefing = state.pending_briefing
                    state.pending_briefing = None
                    insight = briefing.get("insight", "")
                    gaps = briefing.get("knowledge_gaps", [])
                    if insight:
                        # Time-aware greeting
                        import datetime as _dt
                        hour = _dt.datetime.now().hour
                        if hour < 12:
                            greeting = "Good morning"
                        elif hour < 17:
                            greeting = "Good afternoon"
                        else:
                            greeting = "Good evening"
                        preamble = (
                            f"{greeting}, {name}. I have a brief prepared for you "
                            f"about something that may impact your interests. {insight}"
                        )
                        if gaps:
                            preamble += f" I could refine this further if you could tell me about {gaps[0]}."
                        speaker.play_thinking_filler()
                        threading.Thread(
                            target=llm_client.stream_chat,
                            args=(f"{preamble}\n\nNow, regarding what you just said: {text}",),
                            daemon=True,
                            name="llm-stream",
                        ).start()
                        # Mark briefing as delivered on disk
                        try:
                            from core.config import BRIEFINGS_DIR
                            bp = BRIEFINGS_DIR / f"{briefing.get('date', '')}.json"
                            if bp.exists():
                                import json as _json
                                bd = _json.loads(bp.read_text())
                                bd["delivered"] = True
                                bp.write_text(_json.dumps(bd, indent=2))
                        except Exception:
                            pass
                        return

                # Play a thinking filler immediately so user hears instant response
                speaker.play_thinking_filler()
                threading.Thread(
                    target=llm_client.stream_chat,
                    args=(text,),
                    daemon=True,
                    name="llm-stream",
                ).start()

        bus.on("transcript.ready", _on_transcript)

        # Wire: speaking state → GUI
        bus.on("tts.started", lambda **_kw2: setattr(window, "speaking", True))
        bus.on("tts.finished", lambda **_kw2: setattr(window, "speaking", False))

        # Start voice threads
        speaker.start()
        listener.start()
        print("[aura] voice pipeline started")
        memlog.delta("voice pipeline started")

        # Start Aura Perpetual (background rumination engine)
        from services.perpetual import Perpetual
        perpetual = Perpetual()
        perpetual.start()
        state.last_conversation_ts = time.time()  # start idle timer from now

        # Store references for shutdown
        _on_boot_complete._listener = listener
        _on_boot_complete._speaker = speaker
        _on_boot_complete._perpetual = perpetual
        _voice_started.set()

        # Begin GUI crossfade from boot → normal
        window.transition_to_normal()

        # Welcome greeting + optional tour for first-time users
        from voice.speaker import _synth_to_file
        name = state.active_user_name or "friend"
        is_first = orchestrator._enrolled_this_boot

        # 1. Personalized greeting — synthesize to file, then enqueue as WAV
        if is_first:
            welcome_text = f"Welcome to AuraVision, {name}. I'm so glad you're here."
        else:
            welcome_text = f"Welcome back, {name}. Good to see you again."
        welcome_wav = "/tmp/aura_welcome.wav"
        print(f"[aura] Synthesizing welcome: \"{welcome_text}\"")
        ms = _synth_to_file(welcome_text, "warm", welcome_wav)
        print(f"[aura] Welcome synthesized: {ms:.0f}ms")
        speaker.enqueue_wav(welcome_wav)

        # 2. Tour of complications — only on first boot
        #    Each tour line can highlight a specific complication on the dial
        if is_first:
            def _play_tour():
                """Play tour WAVs with complication highlights (runs on speaker thread)."""
                import time as _time
                for i, line in enumerate(BootOrchestrator.TOUR_LINES):
                    highlight = line[2] if len(line) > 2 else None
                    # Highlight the complication being described
                    bus.emit("tour.highlight", comp_name=highlight)
                    if BootOrchestrator.tour_wavs_ready():
                        speaker.enqueue_wav(BootOrchestrator.tour_wav_path(i))
                    else:
                        speaker.enqueue(line[0], style=line[1])
                    # Wait for the WAV to start playing
                    _time.sleep(0.5)
                    # Wait for playback to finish (state.playing is True while aplay runs)
                    while state.playing or not speaker._wav_q.empty() or not speaker._sentence_q.empty():
                        _time.sleep(0.2)
                    _time.sleep(0.6)  # brief pause between tour steps
                # Clear highlight after tour
                bus.emit("tour.highlight", comp_name=None)

            print("[aura] Starting complication tour (first boot)")
            threading.Thread(target=_play_tour, daemon=True, name="tour").start()
        else:
            print("[aura] Returning user — skipping tour")

    bus.once("boot.complete", _on_boot_complete)

    # 6. Graceful shutdown
    def _quit(*_):
        if _voice_started.is_set():
            if hasattr(_on_boot_complete, "_perpetual"):
                _on_boot_complete._perpetual.stop()
            if hasattr(_on_boot_complete, "_listener"):
                _on_boot_complete._listener.stop()
            if hasattr(_on_boot_complete, "_speaker"):
                _on_boot_complete._speaker.stop()
        state.request_shutdown()
        app.quit()

    signal.signal(signal.SIGINT,  _quit)
    signal.signal(signal.SIGTERM, _quit)

    print("[aura] boot mode active — waiting for services")
    rc = app.exec_()
    memlog("shutdown", include_docker=True)
    memlog.summary()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
