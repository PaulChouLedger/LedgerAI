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

        # Wire: transcript → LLM → speaker
        def _on_transcript(text: str = "", **_kw2):
            if text:
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

        # Store references for shutdown
        _on_boot_complete._listener = listener
        _on_boot_complete._speaker = speaker
        _voice_started.set()

        # Begin GUI crossfade from boot → normal
        window.transition_to_normal()

        # Welcome tour: only the personalized greeting needs live synthesis.
        # The 5 generic tour sentences are pre-recorded WAVs (generated once
        # on first boot, instant playback on every subsequent boot).
        from boot.orchestrator import BootOrchestrator
        name = state.active_user_name or "friend"

        # 1. Personalized greeting — must synthesize live (has user's name)
        speaker.enqueue(f"Welcome to AuraVision, {name}.", style="warm")

        # 2. Generic tour — play pre-recorded WAVs if available
        if BootOrchestrator.tour_wavs_ready():
            print("[aura] Playing pre-recorded tour WAVs (no synthesis needed)")
            for i in range(len(BootOrchestrator.TOUR_LINES)):
                speaker.enqueue_wav(BootOrchestrator.tour_wav_path(i))
        else:
            # Fallback: synthesize live (first boot or standard TTS)
            print("[aura] Tour WAVs not ready — synthesizing live")
            for text, style in BootOrchestrator.TOUR_LINES:
                speaker.enqueue(text, style=style)

    bus.once("boot.complete", _on_boot_complete)

    # 6. Graceful shutdown
    def _quit(*_):
        if _voice_started.is_set():
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
