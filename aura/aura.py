#!/usr/bin/env python3
"""
Aura 2.0 -- Master entry point.

Boot-first startup: shows Falcon animation while containers load,
optionally enrolls the user's voice, then crossfades to the normal GUI.

    cd aura && python3 aura.py
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

    # 1b. Load undelivered briefing from disk (survives restarts)
    import json as _json
    _today = time.strftime("%Y-%m-%d")
    _bp = config.BRIEFINGS_DIR / f"{_today}.json"
    if _bp.exists() and not state.pending_briefing:
        try:
            _bd = _json.loads(_bp.read_text())
            if not _bd.get("delivered", False) and _bd.get("insight", "").strip():
                state.pending_briefing = _bd
                print(f"[aura] Loaded undelivered briefing from {_bp.name}")
        except Exception:
            pass

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

        # Wire: transcript → intent check → thinking filler + LLM → speaker
        from voice.intents import detect_intent

        # Track mute state so we can gate transcript processing.
        # The listener stops the mic stream when muted, but a frame captured
        # just before the toggle can still produce a transcript.  This flag
        # ensures we discard it.
        _muted = [False]  # mutable container for closure

        def _on_mute_for_transcript(muted: bool = False, **_kw):
            _muted[0] = muted

        bus.on("mute.toggled", _on_mute_for_transcript)

        def _on_transcript(text: str = "", **_kw2):
            if text:
                # ── Inviolable mute gate ──────────────────────────────
                # If muted, discard everything — no LLM, no TTS, nothing.
                if _muted[0]:
                    print(f"[aura] Ignoring transcript while muted: \"{text[:60]}\"")
                    return

                # If sleeping, any speech wakes up
                if window._sleeping:
                    print(f"[aura] Voice wake from sleep: \"{text}\"")
                    bus.emit("sleep.wake")
                    return

                # Check for local intents (shutdown, etc.) before LLM
                intent = detect_intent(text)
                if intent == "shutdown":
                    print(f"[aura] Shutdown intent detected: \"{text}\"")
                    speaker.enqueue("Initiating shutdown.")
                    bus.emit("shutdown.begin")
                    return
                if intent == "sleep":
                    print(f"[aura] Sleep intent detected: \"{text}\"")
                    speaker.enqueue("Going to sleep. Say my name when you need me.")
                    bus.emit("sleep.begin")
                    return
                state.last_conversation_ts = time.time()

                # Offer pending briefing on first interaction (don't auto-play)
                # Respects defer window — if user said "later", wait 2h before re-offering
                _defer_until = getattr(state, '_briefing_deferred_until', 0)
                if (state.pending_briefing
                        and not getattr(state, '_briefing_offered', False)
                        and time.time() >= _defer_until):
                    import datetime as _dt
                    _now = _dt.datetime.now()
                    hour = _now.hour
                    if hour < 12:
                        greeting = "Good morning"
                    elif hour < 17:
                        greeting = "Good afternoon"
                    else:
                        greeting = "Good evening"
                    _date_str = _now.strftime("%A, %B %-d, %Y")
                    _uname = state.active_user_name or "sir"
                    speaker.enqueue(
                        f"{greeting}, {_uname}. Today is {_date_str}. "
                        "Your daily briefing has been prepared. "
                        "When you're ready, just say: play my daily brief. "
                        "Or you can say skip, or offer it to me later."
                    )
                    state._briefing_offered = True
                    return

                # Handle briefing acceptance/decline/defer
                if getattr(state, '_briefing_offered', False) and state.pending_briefing:
                    state._briefing_offered = False
                    lower = text.lower().strip().rstrip('.!?,')

                    # Check for "later" / defer
                    deferred = any(p in lower for p in (
                        'later', 'offer it to me later', 'not now', 'not right now',
                        'bring it up later', 'maybe later', 'in a bit',
                    ))
                    if deferred:
                        state._briefing_deferred_until = time.time() + 7200  # 2 hours
                        speaker.enqueue("No problem. I'll bring it up again in a couple of hours.")
                        return

                    # Check for "skip" / dismiss for today
                    skipped = any(p in lower for p in (
                        'skip', 'skip the brief', 'skip it', 'skip for today',
                        'no thanks', 'not today', 'pass',
                    ))
                    if skipped:
                        state.pending_briefing = None
                        speaker.enqueue("Got it, skipping today's brief.")
                        return

                    # Check for acceptance — be generous since we just offered.
                    # Whisper often mangles "play my daily brief" into things
                    # like "might be like brief", so match loosely.
                    accepted = any(p in lower for p in (
                        'play my daily brief', 'play my brief', 'play the brief',
                        'daily brief', 'daily breathe', 'daily breed', 'dailybre',
                        'brief', 'grief', 'breathe', 'breed', 'breeze',  # Whisper mishears
                        'deliver', 'let me hear', 'go ahead', 'offer me',
                        'yes', 'yeah', 'sure', 'ok', 'okay', 'please', 'go for it',
                        'play it', 'read it', 'tell me', 'let\'s hear it',
                    ))
                    if accepted:
                        briefing = state.pending_briefing
                        state.pending_briefing = None
                        insight = briefing.get("insight", "")
                        audio_path = briefing.get("audio_path")
                        # Prefer pre-synthesized WAV (smooth, no gaps between clauses)
                        if audio_path and os.path.exists(audio_path):
                            print(f"[aura] Playing pre-synthesized briefing: {audio_path}")
                            speaker.enqueue_wav(audio_path)
                        elif insight:
                            # Fall back to live TTS synthesis
                            print("[aura] No pre-synth audio — synthesizing briefing live")
                            speaker.enqueue(insight)
                        # Sign-off
                        speaker.enqueue(
                            "As always, I am happy to follow up on any items "
                            "you have questions on."
                        )
                        # Mark delivered
                        try:
                            from core.config import BRIEFINGS_DIR
                            import json as _json
                            bp = BRIEFINGS_DIR / f"{briefing.get('date', '')}.json"
                            if bp.exists():
                                bd = _json.loads(bp.read_text())
                                bd["delivered"] = True
                                bp.write_text(_json.dumps(bd, indent=2))
                        except Exception:
                            pass
                        return
                    else:
                        # Unrecognized response — don't silently defer, just
                        # proceed with normal LLM processing of the transcript.
                        # The briefing will be re-offered on the next transcript.
                        print(f"[aura] Briefing response not recognized: \"{text}\" — proceeding normally")
                        pass  # fall through to normal LLM handling below

                # Deliver proactive question (lighter than briefing)
                if state.pending_question:
                    question_data = state.pending_question
                    state.pending_question = None
                    q_text = question_data.get("text", "")
                    if q_text:
                        print(f"[aura] Delivering proactive question: \"{q_text[:80]}\"")
                        # Weave the question into the LLM response naturally
                        speaker.play_thinking_filler(text)
                        threading.Thread(
                            target=llm_client.stream_chat,
                            args=(
                                f"You had a question you wanted to ask the user: \"{q_text}\"\n\n"
                                f"The user just said: \"{text}\"\n\n"
                                f"Respond to what they said, but also naturally work in your question. "
                                f"If their message already answers your question, skip it.",
                            ),
                            daemon=True,
                            name="llm-stream",
                        ).start()
                        return

                # Play a thinking filler immediately so user hears instant response
                speaker.play_thinking_filler(text)
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
        bus.on("tts.amplitude", lambda level=0.0, **_kw2: setattr(window, "audio_amplitude", level))

        # Start voice threads
        speaker.start()
        # Farsight broadcast receiver (accepts TTS from hub)
        from services.broadcast_receiver import start as _start_broadcast
        _start_broadcast()
        listener.start()
        print("[aura] voice pipeline started")
        memlog.delta("voice pipeline started")

        # Start OTA update checker
        try:
            from core.updater import updater
            updater.start()
        except Exception as exc:
            print(f"[aura] updater failed to start: {exc}", flush=True)

        # Start Aura Perpetual (background rumination engine)
        from services.perpetual import Perpetual
        perpetual = Perpetual()
        perpetual.start()
        state.last_conversation_ts = time.time()  # start idle timer from now

        # Start system monitor (hardware metrics → bus events for GUI)
        from services.system_monitor import SystemMonitor
        sysmon = SystemMonitor()
        sysmon.start()

        # Store references for shutdown
        _on_boot_complete._listener = listener
        _on_boot_complete._speaker = speaker
        _on_boot_complete._perpetual = perpetual
        _on_boot_complete._sysmon = sysmon
        _voice_started.set()

        # Begin GUI crossfade from boot → normal
        window.transition_to_normal()

        # Welcome greeting + optional tour for first-time users
        is_first = orchestrator._enrolled_this_boot

        # 1. Play pre-synthesized welcome (rendered during boot, zero delay)
        welcome_wav = orchestrator._welcome_wav
        if welcome_wav and os.path.isfile(welcome_wav):
            print(f"[aura] Playing pre-synthesized welcome: {welcome_wav}")
            speaker.enqueue_wav(welcome_wav)
        else:
            # Fallback: synthesize now (adds ~3s delay)
            from voice.speaker import _synth_to_file
            name = state.active_user_name or "friend"
            if is_first:
                welcome_text = f"Welcome to AuraVision, {name}. I'm so glad you're here."
            else:
                welcome_text = "Hey there. I'm Aura. Say something, I dare you."
            welcome_wav = "/tmp/aura_welcome.wav"
            print(f"[aura] Synthesizing welcome (fallback): \"{welcome_text}\"")
            _synth_to_file(welcome_text, "warm", welcome_wav)
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

    # 7. Voice-triggered shutdown (countdown completed without abort)
    def _on_shutdown_execute(**_kw):
        print("[aura] Shutdown countdown complete — powering off")
        _quit()
        # Give Qt a moment to tear down, then hard poweroff
        import subprocess
        subprocess.Popen(["sudo", "systemctl", "poweroff"], close_fds=True)

    def _on_shutdown_abort(**_kw):
        print("[aura] Shutdown aborted by user")
        if _voice_started.is_set() and hasattr(_on_boot_complete, "_speaker"):
            _on_boot_complete._speaker.interrupt()
            _on_boot_complete._speaker.enqueue("Shutdown cancelled.")

    _countdown_words = {
        10: "ten", 9: "nine", 8: "eight", 7: "seven", 6: "six",
        5: "five", 4: "four", 3: "three", 2: "two", 1: "one",
        0: "Goodbye.",
    }

    def _on_shutdown_tick(secs_left: int = 0, **_kw):
        if not _voice_started.is_set():
            return
        if not hasattr(_on_boot_complete, "_speaker"):
            return
        spk = _on_boot_complete._speaker
        word = _countdown_words.get(secs_left)
        if word:
            spk.enqueue(word)

    bus.on("shutdown.execute", _on_shutdown_execute)
    bus.on("shutdown.abort", _on_shutdown_abort)
    bus.on("shutdown.tick", _on_shutdown_tick)

    # 8. Sleep mode (screen off, mic stays alive for wake word)
    def _on_sleep_begin(**_kw):
        print("[aura] Entering sleep mode")
        window.enter_sleep()

    def _on_sleep_wake(**_kw):
        print("[aura] Waking from sleep")
        window.exit_sleep()
        if _voice_started.is_set() and hasattr(_on_boot_complete, "_speaker"):
            _on_boot_complete._speaker.enqueue("Good morning. I'm here.")

    bus.on("sleep.begin", _on_sleep_begin)
    bus.on("sleep.wake", _on_sleep_wake)

    print("[aura] boot mode active — waiting for services")
    rc = app.exec_()
    memlog("shutdown", include_docker=True)
    memlog.summary()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
