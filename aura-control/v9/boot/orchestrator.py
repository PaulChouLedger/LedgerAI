"""
boot.orchestrator -- State machine for the boot phase.

Runs on a daemon thread. Drives the scripted conversation (play prompt →
capture response → next), extracts speaker embeddings, polls container
health, and retroactively transcribes the user's name once Whisper comes
online.

State machine:
    WAITING_MIC → GREETING → ASK_NAME → ASK_VOICE_SAMPLE → ENROLLMENT
    → WAITING_SERVICES → TRANSCRIBING → COMPLETE

Returning users: detects via is_first_boot(), captures brief audio for
identification, greets by name, uses shorter script.

Emits bus events: boot.phase, boot.service_up, boot.complete,
boot.user_enrolled, boot.name_resolved.

Subscribes: boot.skip (user tapped screen → skip enrollment).
"""

from __future__ import annotations

import glob as _glob
import os
import random
import shutil
import subprocess
import threading
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from core.bus import bus
from core.config import (
    BOOT_MUSIC_PATH,
    BOOT_TOTAL_S,
    BOOT_SERVICE_TIMEOUT_S,
    BOOT_MIC_TIMEOUT_S,
    SAMPLE_RATE,
    TTS_VOLUME,
)
from core.state import state
from boot.prompts import (
    FIRST_BOOT_SCRIPT,
    RETURNING_USER_SCRIPT,
    ResponseType,
    BootPrompt,
)
from boot.voice_capture import BootMic
from services.health import check_service, ensure_containers
from services.memlog import memlog


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class Phase(Enum):
    INIT = auto()
    WAITING_MIC = auto()
    GREETING = auto()
    ASK_NAME = auto()
    ASK_VOICE_SAMPLE = auto()
    ENROLLMENT = auto()
    WAITING_SERVICES = auto()
    TRANSCRIBING = auto()
    COMPLETE = auto()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BootOrchestrator:
    """Boot-phase state machine. Call start() to run on a daemon thread."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._skip = threading.Event()
        self._mic = BootMic()
        self._enrollment = None  # lazy — may not be available
        self._music_proc: Optional[subprocess.Popen] = None
        self._music_ff_proc: Optional[subprocess.Popen] = None
        self._music_sink_input: Optional[int] = None

        # Pre-generated filler WAVs (shuffled, played round-robin)
        filler_dir = BOOT_MUSIC_PATH.parent / "boot_prompts" / "fillers"
        self._filler_wavs = sorted(_glob.glob(str(filler_dir / "filler_*.wav")))
        random.shuffle(self._filler_wavs)
        self._filler_idx = 0

        # Captured audio (for retroactive transcription)
        self._name_audio: Optional[np.ndarray] = None
        self._voice_audio: Optional[np.ndarray] = None
        self._user_id: Optional[str] = None

        # Pre-synthesized welcome WAV (set by _warmup_tts)
        self._welcome_wav: Optional[str] = None

        # Progress tracking
        self._phase = Phase.INIT
        self._start_time = 0.0
        # LLM not required for boot — both llm-medical and llm-generic bind
        # :11434 so only the active one will respond, and it takes longer than
        # the boot timeout to load models on Jetson anyway.  It will be ready
        # by the time the user actually speaks post-boot.
        self._services_up: dict[str, bool] = {
            "whisper": False, "memory": False,
        }

        bus.on("boot.skip", self._on_skip)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the orchestrator on a daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="boot-orch")
        self._thread.start()

    # ------------------------------------------------------------------
    # Bus callbacks
    # ------------------------------------------------------------------

    def _on_skip(self, **_kw) -> None:
        """User tapped screen → skip enrollment, wait only for services."""
        print("[boot] Skip requested")
        self._skip.set()

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _set_phase(self, phase: Phase, progress: float, text: str) -> None:
        self._phase = phase
        bus.emit("boot.phase", phase=phase.name, progress=progress, text=text)

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def _progress_from_time(self) -> float:
        """Map elapsed time to 0..0.7 (services fill the remaining 0.3)."""
        return min(0.7, self._elapsed() / max(1.0, BOOT_TOTAL_S) * 0.7)

    def _services_progress(self) -> float:
        """Map service readiness to 0.7..1.0."""
        n_up = sum(1 for v in self._services_up.values() if v)
        n_total = len(self._services_up)
        return 0.7 + 0.3 * (n_up / max(1, n_total))

    # ------------------------------------------------------------------
    # System volume (set early, before any audio plays)
    # ------------------------------------------------------------------

    def _set_system_volume(self) -> None:
        """Set PulseAudio/ALSA volume from TTS_VOLUME before any audio plays."""
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            sink = result.stdout.strip()
            if sink:
                subprocess.run(
                    ["pactl", "set-sink-volume", sink, f"{vol_pct}%"],
                    check=True, timeout=2,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"[boot] System volume set to {vol_pct}%")
                return
        except Exception:
            pass
        # ALSA fallback
        for ctrl in ("PCM", "Speaker", "Master"):
            try:
                subprocess.run(
                    ["amixer", "sset", ctrl, f"{vol_pct}%"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True, timeout=2,
                )
                print(f"[boot] ALSA volume set to {vol_pct}% ({ctrl})")
                return
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Music
    # ------------------------------------------------------------------

    # Music tempo (1.0 = normal speed; set via env var if needed)
    MUSIC_TEMPO = float(os.environ.get("AURA_BOOT_MUSIC_TEMPO", "1.0"))

    def _start_music(self) -> None:
        from core.config import ALSA_PLAYBACK_DEVICE
        path = str(BOOT_MUSIC_PATH)
        if not os.path.isfile(path):
            print(f"[boot] Music file not found: {path}")
            return
        try:
            if shutil.which("ffmpeg") and shutil.which("aplay"):
                # Decode MP3 with ffmpeg, pipe raw PCM to aplay on the ALSA device
                tempo = self.MUSIC_TEMPO
                ff_cmd = ["ffmpeg", "-i", path, "-loglevel", "quiet"]
                if tempo != 1.0:
                    ff_cmd += ["-af", f"atempo={tempo}"]
                ff_cmd += ["-f", "s16le", "-acodec", "pcm_s16le",
                           "-ac", "2", "-ar", "48000", "-"]
                aplay_cmd = ["aplay", "-D", ALSA_PLAYBACK_DEVICE,
                             "-f", "S16_LE", "-c", "2", "-r", "48000", "-q"]
                ff_proc = subprocess.Popen(
                    ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                cmd = aplay_cmd
                self._music_proc = subprocess.Popen(
                    aplay_cmd, stdin=ff_proc.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ff_proc.stdout.close()  # allow SIGPIPE
                self._music_ff_proc = ff_proc
                tempo_info = f" (tempo={self.MUSIC_TEMPO}x)" if tempo != 1.0 else ""
                print(f"[boot] Music started: ffmpeg|aplay{tempo_info} → {ALSA_PLAYBACK_DEVICE}")
                return
            elif shutil.which("aplay"):
                cmd = ["aplay", "-D", ALSA_PLAYBACK_DEVICE, path]
            else:
                print("[boot] No audio player found for music")
                return
            self._music_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            tempo_info = f" (tempo={self.MUSIC_TEMPO}x)" if shutil.which("ffplay") else ""
            print(f"[boot] Music started: {cmd[0]}{tempo_info} {path}")
            # Give PulseAudio a moment to register the stream, then cache sink-input
            time.sleep(0.5)
            self._music_sink_input = self._find_music_sink_input()
            if self._music_sink_input is not None:
                print(f"[boot] Music sink-input: #{self._music_sink_input}")
            else:
                print("[boot] Warning: could not find music sink-input (fades will be skipped)")
        except Exception as e:
            print(f"[boot] Music playback error: {e}")
            self._music_proc = None

    def _find_music_sink_input(self) -> Optional[int]:
        """Find the PulseAudio sink-input index for the music process."""
        if self._music_proc is None or self._music_proc.poll() is not None:
            return None
        pid = self._music_proc.pid
        try:
            result = subprocess.run(
                ["pactl", "list", "sink-inputs"],
                capture_output=True, text=True, timeout=3,
            )
            if not result.stdout.strip():
                print("[boot] pactl: no sink-inputs found")
                return None

            # Pass 1: match by exact PID
            current_idx = None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Sink Input #"):
                    current_idx = int(stripped.split("#")[1])
                elif "application.process.id" in stripped and current_idx is not None:
                    try:
                        found_pid = int(stripped.split('"')[-2])
                    except (ValueError, IndexError):
                        continue
                    if found_pid == pid:
                        return current_idx

            # Pass 2: fallback — match by binary name "ffplay"
            current_idx = None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Sink Input #"):
                    current_idx = int(stripped.split("#")[1])
                elif "application.process.binary" in stripped and current_idx is not None:
                    if "ffplay" in stripped:
                        print(f"[boot] Matched sink-input #{current_idx} by binary name")
                        return current_idx

            print(f"[boot] No sink-input found for PID {pid} or ffplay")
        except Exception as e:
            print(f"[boot] Error querying sink-inputs: {e}")
        return None

    def _fade_sink_input(self, sink_input: int, from_pct: int, to_pct: int,
                         steps: int = 8, duration: float = 0.5) -> bool:
        """Gradually change a sink-input's volume. Returns True on success."""
        step_delay = duration / max(steps, 1)
        try:
            for i in range(1, steps + 1):
                pct = from_pct + (to_pct - from_pct) * i / steps
                subprocess.run(
                    ["pactl", "set-sink-input-volume", str(sink_input), f"{int(pct)}%"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=1,
                )
                time.sleep(step_delay)
            return True
        except Exception:
            return False

    def _alsa_set_vol(self, pct: int) -> None:
        """Set USB DAC PCM volume via amixer (hardware mixer, no device lock)."""
        try:
            subprocess.run(
                ["amixer", "-D", "hw:CARD=UACDemoV10", "sset", "PCM", f"{pct}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1,
            )
        except Exception:
            pass

    def _alsa_fade(self, from_pct: int, to_pct: int,
                   steps: int = 12, duration: float = 1.0) -> None:
        """Smoothly ramp USB DAC PCM volume between two levels."""
        step_delay = duration / max(steps, 1)
        for i in range(1, steps + 1):
            pct = from_pct + (to_pct - from_pct) * i / steps
            self._alsa_set_vol(int(pct))
            time.sleep(step_delay)

    def _duck_music(self) -> None:
        """Fade music out then kill aplay to release ALSA device.

        ALSA direct hardware (plughw:) doesn't support concurrent access.
        We fade the hardware mixer to 0, kill the process, then the device
        is free for prompt playback.
        """
        if not (self._music_proc and self._music_proc.poll() is None):
            return
        # Smooth fade-out via hardware mixer
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
        self._alsa_fade(from_pct=vol_pct, to_pct=0, steps=15, duration=1.2)
        # Now kill the silent process to release the device
        try:
            self._music_proc.terminate()
            self._music_proc.wait(timeout=2)
        except Exception:
            try:
                self._music_proc.kill()
            except Exception:
                pass
        if self._music_ff_proc is not None:
            try:
                self._music_ff_proc.kill()
            except Exception:
                pass
            self._music_ff_proc = None
        self._music_proc = None
        # Brief delay for ALSA kernel driver to fully release the device
        time.sleep(0.3)
        print("[boot] Music ducked (faded out + killed)")

    def _unduck_music(self) -> None:
        """Restart music with a smooth fade-in after prompt finishes."""
        # Brief delay so the prompt's aplay fully releases the ALSA device
        time.sleep(0.3)
        # Start music at 0 volume, then fade in
        self._alsa_set_vol(0)
        self._start_music()
        if self._music_proc and self._music_proc.poll() is None:
            vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
            self._alsa_fade(from_pct=0, to_pct=vol_pct, steps=15, duration=1.5)
            print("[boot] Music unducked (faded in)")
        else:
            print("[boot] Music unduck: restart failed")

    def _play_filler_during_pause(self, pause_s: float) -> None:
        """Play a pre-generated filler WAV during a pause, with music.

        Ducks music, plays the filler, unducks music, then sleeps any
        remaining time. If no fillers are available, just sleeps.
        """
        if not self._filler_wavs or pause_s < 3.0:
            time.sleep(pause_s)
            return

        # Pick next filler (round-robin through shuffled list)
        wav = self._filler_wavs[self._filler_idx % len(self._filler_wavs)]
        self._filler_idx += 1

        # Let music play for a couple seconds first, then duck + play filler
        music_lead = min(2.0, pause_s * 0.3)
        time.sleep(music_lead)
        remaining = pause_s - music_lead

        if self._skip.is_set():
            return

        # Duck music, play filler, unduck
        self._duck_music()
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
        self._alsa_set_vol(vol_pct)

        fname = os.path.basename(wav)
        print(f"[boot] Playing filler: {fname}")
        self._mic.play_prompt(wav)
        self._mic.wait_for_prompt(timeout=12.0)

        # Unduck music (restarts it with fade-in)
        self._unduck_music()

        # Sleep any remaining time
        elapsed_approx = music_lead + 5.0  # rough estimate of filler + fade time
        leftover = pause_s - elapsed_approx
        if leftover > 0:
            time.sleep(leftover)

    def _stop_music(self) -> None:
        """Stop music playback (kills both ffmpeg feeder and aplay)."""
        if self._music_proc is None:
            return
        try:
            if self._music_proc.poll() is None:
                import signal as sig
                # Ensure it's running (not SIGSTOP'd)
                try:
                    self._music_proc.send_signal(sig.SIGCONT)
                except Exception:
                    pass
                self._music_proc.terminate()
                self._music_proc.wait(timeout=3)
                print("[boot] Music stopped")
        except Exception:
            try:
                self._music_proc.kill()
            except Exception:
                pass
        # Kill ffmpeg feeder if piped
        if self._music_ff_proc is not None:
            try:
                self._music_ff_proc.kill()
            except Exception:
                pass
            self._music_ff_proc = None
        self._music_proc = None
        self._music_sink_input = None

    # ------------------------------------------------------------------
    # Enrollment (lazy init)
    # ------------------------------------------------------------------

    def _get_enrollment(self):
        if self._enrollment is None:
            try:
                from boot.enrollment import VoiceEnrollment
                self._enrollment = VoiceEnrollment()
            except ImportError as e:
                print(f"[boot] resemblyzer not available, skipping enrollment: {e}")
        return self._enrollment

    # ------------------------------------------------------------------
    # Script execution
    # ------------------------------------------------------------------

    def _run_prompt(self, prompt: BootPrompt) -> Optional[np.ndarray]:
        """Play a prompt and optionally capture a response.

        Music handling: ALL prompts fade music to 0% before playing, then
        fade back up afterwards. Capture prompts keep music ducked during
        the mic capture phase.

        Returns captured audio (float32 mono) or None.
        """
        if self._skip.is_set():
            return None

        needs_capture = prompt.response_type != ResponseType.NONE

        # Pause before prompt — play a filler WAV over music to fill the gap
        if prompt.pause_before > 0:
            self._play_filler_during_pause(prompt.pause_before)
            if self._skip.is_set():
                return None

        # Always fade music down before the main prompt
        self._duck_music()

        # Restore volume for prompt playback (duck faded to 0)
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
        self._alsa_set_vol(vol_pct)

        # Play the prompt audio
        wav = prompt.wav_path
        if os.path.isfile(wav):
            print(f"[boot] Playing prompt: {prompt.phase_name} ({os.path.basename(wav)})"
                  f" [{'capture' if needs_capture else 'announce'}]")
            self._mic.play_prompt(wav)
            self._mic.wait_for_prompt(timeout=15.0)
        else:
            print(f"[boot] Prompt not found: {wav}")
            time.sleep(1.0)

        if self._skip.is_set():
            self._unduck_music()
            return None

        # No capture needed — fade music back up and return
        if not needs_capture:
            if prompt.pause_after > 0:
                time.sleep(prompt.pause_after)
            self._unduck_music()
            return None

        # Brief pause before listening
        if prompt.pause_after > 0:
            time.sleep(prompt.pause_after)

        # Capture user's response (music stays ducked for clear recording)
        audio = self._mic.capture_utterance(
            max_duration=prompt.capture_max_s,
            wait_timeout=prompt.timeout_s,
        )

        if audio is None and prompt.fallback_path and os.path.isfile(prompt.fallback_path):
            print(f"[boot] Playing fallback: {os.path.basename(prompt.fallback_path)}")
            self._mic.play_prompt(prompt.fallback_path)
            self._mic.wait_for_prompt(timeout=10.0)

        # Resume music after capture
        self._unduck_music()

        return audio

    # ------------------------------------------------------------------
    # Service polling
    # ------------------------------------------------------------------

    def _poll_services_once(self) -> bool:
        """Check all services once. Returns True if all are up."""
        for name in self._services_up:
            if not self._services_up[name]:
                if check_service(name):
                    self._services_up[name] = True
                    print(f"[boot] Service UP: {name}")
                    bus.emit("boot.service_up", name=name)
        return all(self._services_up.values())

    def _wait_for_services(self, timeout: float = BOOT_SERVICE_TIMEOUT_S) -> bool:
        """Poll until all services are up or timeout."""
        deadline = time.time() + timeout
        start = time.time()
        while time.time() < deadline:
            if self._skip.is_set():
                print("[boot] Service wait skipped")
                return False
            if self._poll_services_once():
                self._set_phase(Phase.WAITING_SERVICES, 1.0, "All systems ready")
                return True
            # Progress: blend service-based + time-based so dial always moves
            svc_prog = self._services_progress()
            elapsed_frac = min(1.0, (time.time() - start) / max(1.0, timeout))
            time_prog = 0.7 + 0.29 * elapsed_frac   # 0.7 → 0.99 over timeout
            prog = max(svc_prog, time_prog)
            self._set_phase(Phase.WAITING_SERVICES, prog, "Loading AI models")
            time.sleep(2.0)
        print("[boot] Service timeout — proceeding anyway")
        return False

    # ------------------------------------------------------------------
    # Retroactive transcription
    # ------------------------------------------------------------------

    def _retroactive_transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio via Whisper (only call when Whisper is up)."""
        if not self._services_up.get("whisper"):
            return None
        try:
            from voice.listener import transcribe
            text = transcribe(audio, SAMPLE_RATE)
            if text:
                # Clean up — the user probably just said their name
                text = text.strip().strip(".,!?").strip()
                # Remove common filler (user was prompted: "Aura, my name is X")
                for prefix in ("aura ", "aura, ", "aura. ", "hey aura ", "hey aura, "):
                    lower = text.lower()
                    if lower.startswith(prefix):
                        text = text[len(prefix):].strip()
                        break
                for prefix in ("my name is ", "i'm ", "i am ", "it's ", "call me "):
                    lower = text.lower()
                    if lower.startswith(prefix):
                        text = text[len(prefix):].strip()
                        break
                # Capitalize
                if text:
                    text = text[0].upper() + text[1:]
            return text if text else None
        except Exception as e:
            print(f"[boot] Retroactive transcription failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TTS warmup (parallel with service polling)
    # ------------------------------------------------------------------

    # Pre-recorded tour WAVs (generated once, reused every boot)
    # Each tuple: (text, style, complication_name_to_highlight_or_None)
    TOUR_LINES = [
        ("Let me give you a quick tour of your Aura.", "energy", None),
        ("On the top left you'll find Topics Center. That's where you can browse and pin different tools to your dial.", "neutral", "Topics Center"),
        ("Next to it is Settings, where you can adjust your voice, language model, and wake word preferences.", "technical", "Settings"),
        ("The Mute button lets you toggle the microphone on and off. When it's green, I'm listening.", "soft", "Mute"),
        ("And over here is Aura Concierge, your personal assistant for tasks, reminders, and recommendations.", "warm", "Aura Concierge"),
        ("You can talk to me anytime. Just say what's on your mind, and I'll do my best to help.", "playful", None),
    ]
    TOUR_DIR = BOOT_MUSIC_PATH.parent / "boot_prompts" / "tour"

    @classmethod
    def tour_wav_path(cls, idx: int) -> str:
        return str(cls.TOUR_DIR / f"tour_{idx}.wav")

    @classmethod
    def tour_wavs_ready(cls) -> bool:
        """True if all 5 pre-recorded tour WAVs exist."""
        return all(
            os.path.isfile(cls.tour_wav_path(i))
            for i in range(len(cls.TOUR_LINES))
        )

    def _warmup_tts(self) -> None:
        """Load Kokoro TTS and generate tour WAVs if missing.

        Runs on a background thread. Tour WAVs are permanent static assets —
        generated exactly once on first-ever boot, then reused forever.
        """
        try:
            import voice.speaker as _spk

            memlog.delta("TTS warmup: before model load")
            print("[boot] TTS warmup: loading Kokoro TTS...")
            _spk._get_kokoro()
            print(f"[boot] TTS model loaded: Kokoro-82M (voice={_spk.KOKORO_VOICE})")
            memlog.delta("TTS warmup: model loaded")

            # Tour WAVs are permanent static assets — generate ONCE, reuse forever
            if self.tour_wavs_ready():
                print("[boot] Tour WAVs already exist — skipping synthesis")
            else:
                self.TOUR_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[boot] Generating {len(self.TOUR_LINES)} tour WAVs...")
                for i, line in enumerate(self.TOUR_LINES):
                    text, style = line[0], line[1]
                    out = self.tour_wav_path(i)
                    ms = _spk._synth_to_file(text, style, out)
                    print(f"[boot] Tour WAV {i+1}/{len(self.TOUR_LINES)}: {ms:.0f}ms → \"{text[:40]}\"")
                print("[boot] Tour WAVs generated")

            print(f"[boot] TTS warmup complete")
        except Exception as e:
            import traceback
            print(f"[boot] TTS warmup failed (non-fatal): {e}")
            traceback.print_exc()

    def _run(self) -> None:
        """Main orchestrator loop (runs on daemon thread)."""
        self._start_time = time.time()
        print("[boot] Orchestrator started")
        memlog.delta("boot orchestrator started")

        # Ensure Docker containers are running
        ensure_containers()
        memlog.delta("containers ensured")

        # Set system volume before any audio plays
        self._set_system_volume()

        # Start background music
        self._start_music()

        enrollment = self._get_enrollment()
        is_first = enrollment.is_first_boot() if enrollment else True

        if is_first:
            self._run_first_boot(enrollment)
        else:
            self._run_returning_user(enrollment)

        # ---- Kick off TTS warmup AFTER enrollment conversation ----
        # (loading ChatterboxTTS into VRAM is heavy — would kill audio playback)
        tts_ready = threading.Event()
        def _tts_thread():
            self._warmup_tts()
            tts_ready.set()
        threading.Thread(target=_tts_thread, daemon=True, name="tts-warmup").start()

        # ---- Wait for services (shared for both paths) ----
        self._set_phase(Phase.WAITING_SERVICES, self._progress_from_time(),
                        "Loading AI models")

        # Poll services in parallel with TTS warmup
        all_up = self._wait_for_services()

        # ---- Retroactive transcription ----
        if self._name_audio is not None and self._services_up.get("whisper"):
            self._set_phase(Phase.TRANSCRIBING, 0.95, "Recognizing your name")
            dur = len(self._name_audio) / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(self._name_audio ** 2)))
            print(f"[boot] Transcribing name audio: {dur:.2f}s, rms={rms:.4f}")
            raw_name = self._retroactive_transcribe(self._name_audio)
            print(f"[boot] Whisper heard: \"{raw_name}\"" if raw_name else
                  "[boot] Whisper returned empty transcription")
            if raw_name and self._user_id and enrollment:
                enrollment.update_name(self._user_id, raw_name)
                state.active_user_name = raw_name
                bus.emit("boot.name_resolved", name=raw_name)
                print(f"[boot] Name resolved: {raw_name}")
        elif self._name_audio is not None:
            print("[boot] Whisper not available — skipping name transcription")
        else:
            print("[boot] No name audio was captured")

        # ---- Wait for TTS warmup before transitioning ----
        # No timeout — tour WAVs MUST be ready before boot.complete fires,
        # otherwise aura.py falls back to live synthesis (double work + GPU
        # contention).  On first-ever boot this can take ~2–3 min for 5 WAVs;
        # on all subsequent boots the WAVs already exist and this returns
        # instantly.
        if not tts_ready.is_set():
            self._set_phase(Phase.WAITING_SERVICES, 0.97, "Warming up voice")
            print("[boot] Waiting for TTS warmup to finish...")
            tts_ready.wait()

        # ---- Stop music ----
        if self._music_proc and self._music_proc.poll() is None:
            self._set_phase(Phase.WAITING_SERVICES, 0.99, "Almost ready")
            self._stop_music()

        # ---- Complete ----
        self._mic.close()
        self._set_phase(Phase.COMPLETE, 1.0, "Ready")
        memlog("boot complete (pre-emit)", include_docker=True)
        bus.emit("boot.complete")
        print(f"[boot] Complete ({self._elapsed():.1f}s)")

    def _run_first_boot(self, enrollment) -> None:
        """Execute the first-boot conversation script."""
        # Phase: Wait for mic
        self._set_phase(Phase.WAITING_MIC, 0.02, "Waiting for microphone")
        mic_ready = self._mic.wait_for_mic()

        if not mic_ready or self._skip.is_set():
            self._set_phase(Phase.WAITING_SERVICES, 0.15, "Loading AI models")
            return

        script = FIRST_BOOT_SCRIPT
        for prompt in script:
            if self._skip.is_set():
                break

            phase_map = {
                "greeting": Phase.GREETING,
                "ask_name": Phase.ASK_NAME,
                "confirm_name": Phase.ASK_NAME,
                "ask_voice_sample": Phase.ASK_VOICE_SAMPLE,
                "enrollment_done": Phase.ENROLLMENT,
                "waiting": Phase.WAITING_SERVICES,
            }
            phase = phase_map.get(prompt.phase_name, Phase.GREETING)
            prog = self._progress_from_time()
            self._set_phase(phase, prog, prompt.progress_text)

            audio = self._run_prompt(prompt)

            # Store captures
            if prompt.response_type == ResponseType.NAME:
                if audio is not None:
                    print(f"[boot] NAME audio captured: {len(audio)/SAMPLE_RATE:.2f}s")
                else:
                    print("[boot] NAME prompt: no audio captured (timeout or silence)")
                self._name_audio = audio
            elif prompt.response_type == ResponseType.VOICE_SAMPLE and audio is not None:
                self._voice_audio = audio

        # Attempt enrollment
        if not self._skip.is_set() and enrollment:
            voice = self._voice_audio if self._voice_audio is not None else self._name_audio
            if voice is not None:
                try:
                    self._set_phase(Phase.ENROLLMENT, self._progress_from_time(),
                                    "Creating voice profile")
                    self._user_id = enrollment.enroll(
                        name="User",  # placeholder until retroactive transcription
                        audio=voice,
                    )
                    state.active_user_id = self._user_id
                    state.active_user_name = "User"
                    state.boot_enrollment_done = True
                    bus.emit("boot.user_enrolled", user_id=self._user_id, name="User")
                    print(f"[boot] Enrolled as {self._user_id}")
                except Exception as e:
                    print(f"[boot] Enrollment failed: {e}")

    def _run_returning_user(self, enrollment) -> None:
        """Execute the returning-user identification flow."""
        self._set_phase(Phase.WAITING_MIC, 0.02, "Waiting for microphone")
        mic_ready = self._mic.wait_for_mic()

        if not mic_ready or self._skip.is_set():
            # Fall back to last known user
            if state.active_user_id and enrollment:
                name = enrollment.get_name(state.active_user_id)
                if name:
                    state.active_user_name = name
            return

        script = RETURNING_USER_SCRIPT
        captured_audio = None

        for prompt in script:
            if self._skip.is_set():
                break

            phase = Phase.GREETING if prompt.phase_name == "identify" else Phase.WAITING_SERVICES
            prog = self._progress_from_time()
            self._set_phase(phase, prog, prompt.progress_text)

            audio = self._run_prompt(prompt)
            if audio is not None and prompt.response_type == ResponseType.VOICE_SAMPLE:
                captured_audio = audio

        # Identify returning user
        if captured_audio is not None and enrollment and not self._skip.is_set():
            try:
                uid, score = enrollment.identify(captured_audio)
                if uid:
                    name = enrollment.get_name(uid)
                    state.active_user_id = uid
                    state.active_user_name = name
                    print(f"[boot] Identified: {name} (score={score:.3f})")
                    bus.emit("boot.user_enrolled", user_id=uid, name=name or "User")
                else:
                    print(f"[boot] No match (best score={score:.3f})")
                    # Fall back to last known
                    if state.active_user_id:
                        name = enrollment.get_name(state.active_user_id)
                        if name:
                            state.active_user_name = name
            except Exception as e:
                print(f"[boot] Identification failed: {e}")
        elif state.active_user_id and enrollment:
            name = enrollment.get_name(state.active_user_id)
            if name:
                state.active_user_name = name
