"""
boot.orchestrator -- State machine for the boot phase.

Runs on a daemon thread. Drives the scripted conversation (play prompt →
capture response → next), extracts speaker embeddings, polls container
health, and retroactively transcribes the user's name once Whisper comes
online.

State machine:
    WAITING_MIC → GREETING → (identify) → ASK_NAME → ASK_VOICE_SAMPLE
    → ENROLLMENT → WAITING_SERVICES → TRANSCRIBING → COMPLETE

Unified flow: always plays a natural greeting to elicit voice, tries to
match against stored profiles. If identified → skips enrollment. If not
→ runs full enrollment (ask name, voice sample, enroll).

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

        # Pre-generated short responses ("Oh I love that", "Nice answer", etc.)
        resp_dir = BOOT_MUSIC_PATH.parent / "boot_prompts" / "filler_responses"
        self._response_wavs = sorted(_glob.glob(str(resp_dir / "response_*.wav")))
        random.shuffle(self._response_wavs)
        self._response_idx = 0

        # Captured audio (for retroactive transcription)
        self._name_audio: Optional[np.ndarray] = None
        self._voice_audio: Optional[np.ndarray] = None
        self._user_id: Optional[str] = None
        self._enrolled_this_boot: bool = False  # True if enrollment happened this boot
        self._extra_voice_samples: list[np.ndarray] = []  # filler responses for profile deepening

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
        had_music = self._music_proc is not None
        if self._music_proc and self._music_proc.poll() is None:
            # Quick fade-out via hardware mixer
            vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
            self._alsa_fade(from_pct=vol_pct, to_pct=0, steps=8, duration=0.5)
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
        time.sleep(0.2)
        if had_music:
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
        """Play a conversational question, capture response for voice profiling.

        Ducks music, plays question, captures user's response (deepens voice
        profile), unducks music. Longer music lead for natural pacing.
        """
        if not self._filler_wavs or pause_s < 10.0:
            time.sleep(pause_s)
            return

        # Pick next filler (round-robin through shuffled list)
        wav = self._filler_wavs[self._filler_idx % len(self._filler_wavs)]
        self._filler_idx += 1

        # Longer music lead — let it breathe before the next question
        music_lead = min(4.0, pause_s * 0.4)
        time.sleep(music_lead)

        if self._skip.is_set():
            return

        # Duck music, play question, capture response
        self._duck_music()
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)
        self._alsa_set_vol(vol_pct)

        fname = os.path.basename(wav)
        print(f"[boot] Asking: {fname}")
        self._mic.play_prompt(wav)
        self._mic.wait_for_prompt(timeout=12.0)
        self._mic.drain_echo(1.0)

        # Capture response — doubles as voice profile deepening
        audio = self._mic.capture_utterance(
            max_duration=6.0,
            wait_timeout=6.0,
        )
        if audio is not None:
            dur = len(audio) / SAMPLE_RATE
            print(f"[boot] Filler response captured: {dur:.1f}s (for voice profile)")
            self._extra_voice_samples.append(audio)
            # Play a short acknowledgment ("Oh I love that", "Nice answer", etc.)
            if self._response_wavs:
                resp = self._response_wavs[self._response_idx % len(self._response_wavs)]
                self._response_idx += 1
                print(f"[boot] Response: {os.path.basename(resp)}")
                self._mic.play_prompt(resp)
                self._mic.wait_for_prompt(timeout=5.0)
        else:
            print("[boot] No response to filler (that's okay)")

        # Unduck music (restarts it with fade-in)
        self._unduck_music()

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

        # Drain mic buffer to flush speaker echo before listening
        self._mic.drain_echo(max(prompt.pause_after, 1.0))

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

    def _wait_for_services_with_chat(self) -> bool:
        """Wait for services while keeping the user company with conversation."""
        deadline = time.time() + BOOT_SERVICE_TIMEOUT_S
        start = time.time()
        last_filler = 0.0  # timestamp of last filler

        while time.time() < deadline:
            if self._skip.is_set():
                return False
            if self._poll_services_once():
                self._set_phase(Phase.WAITING_SERVICES, 1.0, "All systems ready")
                return True

            svc_prog = self._services_progress()
            elapsed_frac = min(1.0, (time.time() - start) / max(1.0, BOOT_SERVICE_TIMEOUT_S))
            time_prog = 0.7 + 0.29 * elapsed_frac
            prog = max(svc_prog, time_prog)
            self._set_phase(Phase.WAITING_SERVICES, prog, "Loading AI models")

            # Play a conversational filler every ~30s during the wait
            if (time.time() - last_filler > 28.0 and self._filler_wavs
                    and self._music_proc and self._music_proc.poll() is None):
                self._play_filler_during_pause(12.0)
                last_filler = time.time()
            else:
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
            result = transcribe(audio, SAMPLE_RATE)
            text = result[0] if isinstance(result, tuple) else result
            if text:
                # Clean up — the user probably just said their name
                text = text.strip().strip(".,!?").strip()
                # Remove "Aura" / "Laura" / "Ora" prefix (Whisper often mishears)
                # Apply repeatedly in case of "Hey Laura, my name is X"
                _aura_prefixes = (
                    "hey laura, ", "hey laura ", "hey aura, ", "hey aura ",
                    "laura, ", "laura ", "aura, ", "aura. ", "aura ",
                    "ora, ", "ora ", "hey ora, ", "hey ora ",
                )
                for _ in range(2):  # two passes to catch nested prefixes
                    lower = text.lower()
                    for prefix in _aura_prefixes:
                        if lower.startswith(prefix):
                            text = text[len(prefix):].strip()
                            break
                # Remove "my name is" / "I'm" / etc.
                _name_prefixes = (
                    "my name is ", "i'm ", "i am ", "it's ", "call me ",
                    "this is ", "they call me ",
                )
                lower = text.lower()
                for prefix in _name_prefixes:
                    if lower.startswith(prefix):
                        text = text[len(prefix):].strip()
                        break
                # Strip trailing filler
                text = text.strip().strip(".,!?").strip()
                # Capitalize first letter
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
    # Each tuple: (text, style, name_to_highlight_or_None)
    # Order follows the dial clockwise matching actual layout:
    #   Glyphs load alphabetically: AuraNet, Education, Financial, Medical
    #   Positions: TC(top) → AuraNet(TR) → Settings(R) → Education(BR)
    #             → Mute(B) → Financial(BL) → Concierge(L) → Medical(TL)
    TOUR_LINES = [
        ("Let me give you a quick tour of your Aura.", "energy", None),
        ("This is Topics Center. Browse and pin different tools to your dial.", "neutral", "Topics Center"),
        ("This is AuraNet, your connection to the wider Aura community and network.", "energy", "AuraNet"),
        ("This is Settings, where you adjust your voice, language model, and preferences.", "technical", "Settings"),
        ("This is Education. Tap it for interactive lessons and learning tools.", "neutral", "Education"),
        ("This is the Mute button. It toggles the microphone, and also stops me mid-sentence.", "soft", "Mute"),
        ("This is your Financial domain. Market data, portfolio tracking, and financial insights.", "neutral", "Financial"),
        ("You also have Aura Concierge and Medical. You can pin them to your dial from Topics Center.", "warm", None),
        ("You can talk to me anytime. Just say what is on your mind, and I will do my best to help.", "playful", None),
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
            print("[boot] TTS warmup: loading Piper...")
            _spk._get_piper()
            print("[boot] TTS model loaded: Piper")
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
        """Main orchestrator loop (runs on daemon thread).

        Architecture (v2):
        ──────────────────
        Three mutually-exclusive boot paths, chosen ONCE:

          A) RETURNING USER (profiles exist)
             → Play ONE greeting prompt, capture voice, match against profiles.
             → If identified: set user, done.
             → If mic busy or no match: fall back to persisted user from
               app_settings.json (always has the last identified user).
             → NEVER fall through to enrollment.

          B) FIRST BOOT (no profiles)
             → Full enrollment: greeting → ask name → voice sample → enroll.

          C) SKIP (user tapped screen)
             → Jump straight to service wait.

        After the voice phase: wait for services, wait for TTS warmup,
        pre-synthesize welcome, emit boot.complete.  The mic is released
        in a finally block so it can never leak.
        """
        self._start_time = time.time()
        print("[boot] Orchestrator started")
        memlog.delta("boot orchestrator started")

        # Services started by start_aura.sh (memory only — Whisper+LLM now in-process)
        ensure_containers()
        memlog.delta("services checked")

        # Set system volume before any audio plays
        self._set_system_volume()

        # ---- Load in-process inference engines (sequential: Whisper first, then LLM) ----
        # Must be sequential: Whisper (~1GB) must claim GPU memory before LLM (~5GB)
        # to avoid CUDA OOM from fragmentation on Jetson unified memory.
        def _load_engines():
            try:
                from voice.whisper_engine import whisper_engine
                whisper_engine.load()
                self._services_up["whisper"] = True
                bus.emit("boot.service_up", name="whisper")
                memlog.delta("whisper loaded (in-process)")
            except Exception as e:
                print(f"[boot] Whisper in-process load failed: {e}")

            try:
                from voice.llm_engine import llm_engine
                llm_engine.load()
                memlog.delta("LLM loaded (in-process)")
            except Exception as e:
                print(f"[boot] LLM in-process load failed: {e}")

        threading.Thread(target=_load_engines, daemon=True, name="engine-load").start()

        # ---- TTS warmup (parallel, non-blocking) ----
        from voice.speaker import warm_local_tts_background
        warm_local_tts_background()

        tts_ready = threading.Event()
        def _tts_thread():
            self._warmup_tts()
            tts_ready.set()
        threading.Thread(target=_tts_thread, daemon=True, name="tts-warmup").start()

        # Start background music
        self._start_music()

        # ── Voice phase (mic acquired + released here) ──────────────
        try:
            enrollment = self._get_enrollment()
            has_profiles = enrollment and not enrollment.is_first_boot()

            if has_profiles:
                self._boot_returning_user(enrollment)
            elif not self._skip.is_set():
                self._boot_first_time(enrollment)
        finally:
            # Guarantee the boot mic is released no matter what happens
            self._mic.close()

        # ── Wait for services ──────────────────────────────────────
        self._set_phase(Phase.WAITING_SERVICES, self._progress_from_time(),
                        "Loading AI models")
        self._wait_for_services_with_chat()
        # Close boot mic again — fillers may have reopened it
        self._mic.close()

        # ── Retroactive name transcription (only if name wasn't resolved during enrollment) ──
        already_named = (state.active_user_name and
                         state.active_user_name not in ("User", "friend"))
        if (self._name_audio is not None and self._services_up.get("whisper")
                and not already_named):
            self._set_phase(Phase.TRANSCRIBING, 0.95, "Recognizing your name")
            raw_name = self._retroactive_transcribe(self._name_audio)
            if raw_name and self._user_id and enrollment:
                enrollment.update_name(self._user_id, raw_name)
                state.active_user_name = raw_name
                bus.emit("boot.name_resolved", name=raw_name)
                print(f"[boot] Name resolved: {raw_name}")

        # ── Wait for TTS warmup ───────────────────────────────────
        if not tts_ready.is_set():
            self._set_phase(Phase.WAITING_SERVICES, 0.97, "Warming up voice")
            print("[boot] Waiting for TTS warmup to finish...")
            while not tts_ready.is_set():
                if (self._filler_wavs and self._music_proc
                        and self._music_proc.poll() is None):
                    self._play_filler_during_pause(12.0)
                else:
                    tts_ready.wait(timeout=5.0)
            # Fillers during TTS warmup may have reopened mic
            self._mic.close()

        # ── Pre-synthesize welcome greeting ───────────────────────
        # Fixed intro line — the user picked this exact phrasing and
        # wants it consistent every boot. (The conversation_engine's
        # variety system still drives mid-conversation utterances.)
        welcome_text = (
            "Hey — good to see you. I'm Aura. As you know, just give me "
            "a minute or two to boot up and then we can get to work."
        )
        welcome_style = "warm"

        welcome_wav = "/tmp/aura_welcome.wav"
        try:
            from voice.speaker import _synth_to_file
            print(f"[boot] Pre-synthesizing welcome [{welcome_style}]: \"{welcome_text}\"")
            ms = _synth_to_file(welcome_text, welcome_style, welcome_wav)
            self._welcome_wav = welcome_wav
            print(f"[boot] Welcome pre-synthesized: {ms:.0f}ms")
        except Exception as e:
            print(f"[boot] Welcome pre-synth failed: {e}")
            self._welcome_wav = None

        # ── Stop music → complete ─────────────────────────────────
        if self._music_proc and self._music_proc.poll() is None:
            self._set_phase(Phase.WAITING_SERVICES, 0.99, "Almost ready")
            self._stop_music()

        self._set_phase(Phase.COMPLETE, 1.0, "Ready")
        memlog("boot complete (pre-emit)")
        bus.emit("boot.complete")
        print(f"[boot] Complete ({self._elapsed():.1f}s)")

    # ------------------------------------------------------------------
    # Boot path A: returning user (profiles exist)
    # ------------------------------------------------------------------

    def _boot_returning_user(self, enrollment) -> None:
        """Identify a returning user by voice.

        Plays ONE greeting, captures voice, matches against profiles.
        If mic is unavailable or voice doesn't match, falls back to the
        last known user persisted in app_settings.json.  NEVER runs
        enrollment — the user already has a profile.
        """
        self._set_phase(Phase.WAITING_MIC, 0.02, "Waiting for microphone")
        mic_ready = self._mic.wait_for_mic()

        if not mic_ready or self._skip.is_set():
            self._use_persisted_user("mic_unavailable")
            return

        # Play the single identification prompt
        script = RETURNING_USER_SCRIPT
        captured_audio = None
        for prompt in script:
            if self._skip.is_set():
                break
            self._set_phase(Phase.GREETING, self._progress_from_time(),
                            prompt.progress_text)
            audio = self._run_prompt(prompt)
            if audio is not None and prompt.response_type == ResponseType.VOICE_SAMPLE:
                captured_audio = audio

        if captured_audio is None or self._skip.is_set():
            self._use_persisted_user("no_audio_captured")
            return

        dur = len(captured_audio) / SAMPLE_RATE
        print(f"[boot] [VOICE] Identification capture: {dur:.1f}s")

        # Match voice against stored profiles
        try:
            uid, score = enrollment.identify(captured_audio)
            if uid:
                name = enrollment.get_name(uid)
                state.active_user_id = uid
                state.active_user_name = name
                state.boot_enrollment_done = True
                print(f"[boot] ═══════════════════════════════════════════")
                print(f"[boot] Identified: {name} (score={score:.3f})")
                print(f"[boot] ═══════════════════════════════════════════")
                bus.emit("boot.user_enrolled", user_id=uid, name=name or "User")
                enrollment.deepen_profile(uid, [captured_audio])
                return
            else:
                print(f"[boot] Voice not recognized (best score={score:.3f})")
                # Offer enrollment to the unknown speaker
                self._enroll_unknown_user(enrollment, captured_audio)
        except Exception as e:
            print(f"[boot] Identification error: {e}")
            self._use_persisted_user("identify_exception")

    # ------------------------------------------------------------------
    # Mid-life enrollment: unknown voice on a puck that already has profiles
    # ------------------------------------------------------------------

    def _speak(self, text: str, style: str = "warm") -> None:
        """Synthesize text with Piper and play it through the boot mic.

        `style` controls Piper voice expressiveness — varies per
        utterance via the conversation engine so Aura doesn't sound
        monotone.
        """
        from voice.speaker import _synth_to_file
        tmp_wav = "/tmp/aura_boot_speak.wav"
        _synth_to_file(text, style, tmp_wav)
        if os.path.isfile(tmp_wav):
            self._mic.play_prompt(tmp_wav)
            self._mic.wait_for_prompt(timeout=15.0)

    def _try_transcribe_name(self, audio: np.ndarray) -> Optional[str]:
        """Try to transcribe name audio immediately if Whisper is up."""
        if not self._services_up.get("whisper"):
            return None
        return self._retroactive_transcribe(audio)

    def _enroll_unknown_user(self, enrollment, initial_audio: np.ndarray) -> None:
        """Enroll a new user who doesn't match any existing voice profile.

        Uses boot.conversation_engine.EnrollmentConversation so every
        enrollment uses different opening lines, questions, reactions,
        and closings — LLM-driven when the engine is loaded, deeply
        varied curated banks otherwise. Replaces the old script that
        always asked the same four questions in the same order.
        """
        self._stop_music()
        print("[boot] ═══════════════════════════════════════════")
        print("[boot] Unknown voice — starting new-user enrollment")
        self._set_phase(Phase.GREETING, self._progress_from_time(),
                        "New user detected")

        # Hook up the conversation engine. llm_engine may not be loaded
        # yet; the engine's own check (`loaded` attribute) handles that
        # and falls back to curated banks transparently.
        try:
            from boot.conversation_engine import EnrollmentConversation
            from voice.llm_engine import llm_engine as _llm
            convo = EnrollmentConversation(llm=_llm)
        except Exception as e:
            print(f"[boot] conversation engine init failed (using fallback): {e}")
            from boot.conversation_engine import EnrollmentConversation
            convo = EnrollmentConversation(llm=None)

        name = "User"
        all_voice_samples = [initial_audio]
        print(f"[boot] [VOICE] Initial capture: {len(initial_audio)/SAMPLE_RATE:.1f}s")

        # ── Step 1: Ask full name ─────────────────────────────────
        opener_text, opener_style = convo.opener()
        self._speak(opener_text, opener_style)
        print(f"[boot] [AURA/{opener_style}] {opener_text}")
        self._mic.drain_echo(1.5)

        name_audio = self._mic.capture_utterance(
            max_duration=8.0, wait_timeout=10.0)
        if name_audio is not None:
            dur = len(name_audio) / SAMPLE_RATE
            print(f"[boot] [VOICE] Name response: {dur:.1f}s")
            self._name_audio = name_audio
            all_voice_samples.append(name_audio)

            # Try to transcribe name right now
            transcribed = self._try_transcribe_name(name_audio)
            if transcribed:
                name = transcribed
                print(f"[boot] [TRANSCRIPT] Name heard: '{name}'")

                # ── Step 2: Confirm spelling ──────────────────────
                confirm_text, confirm_style = convo.name_confirm(name)
                self._speak(confirm_text, confirm_style)
                print(f"[boot] [AURA/{confirm_style}] {confirm_text}")
                self._mic.drain_echo(1.5)

                confirm_audio = self._mic.capture_utterance(
                    max_duration=8.0, wait_timeout=10.0)
                if confirm_audio is not None:
                    dur = len(confirm_audio) / SAMPLE_RATE
                    print(f"[boot] [VOICE] Spelling confirmation: {dur:.1f}s")
                    all_voice_samples.append(confirm_audio)

                    # Transcribe spelling confirmation
                    spelling = self._try_transcribe_name(confirm_audio)
                    if spelling:
                        print(f"[boot] [TRANSCRIPT] Spelling response: '{spelling}'")
                        lower = spelling.lower().strip()
                        # Check for corrections like "No, it's Bob" or "B-O-B"
                        for prefix in ("no it's ", "no, it's ", "it's ", "actually it's ",
                                       "no ", "actually "):
                            if lower.startswith(prefix):
                                corrected = spelling[len(prefix):].strip().strip(".,!?")
                                if corrected:
                                    name = corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
                                    print(f"[boot] [TRANSCRIPT] Name corrected to: '{name}'")
                                    break
                        # If they said "yes" / "yeah" / "correct" / "that's right", keep the name
                        if lower.startswith(("yes", "yeah", "yep", "correct", "that's right", "right")):
                            print(f"[boot] [TRANSCRIPT] Name confirmed: '{name}'")
            else:
                print("[boot] Whisper not ready yet — name will be resolved after services load")
        else:
            print("[boot] [VOICE] No name audio captured")

        # ── Step 3: Conversational voice gathering ────────────────
        # 4-5 questions through the conversation engine — LLM-driven
        # when loaded, theme-rotating curated bank otherwise. Each
        # capture also doubles as a voice-print sample.
        n_voice_questions = 4
        for qi in range(n_voice_questions):
            q_text, q_style = convo.next_question(qi, name)
            max_dur = 12.0 if qi == 0 else 10.0
            self._speak(q_text, q_style)
            print(f"[boot] [AURA/{q_style}] {q_text}")
            self._mic.drain_echo(1.5)

            audio = self._mic.capture_utterance(
                max_duration=max_dur, wait_timeout=12.0)
            if audio is not None:
                dur = len(audio) / SAMPLE_RATE
                print(f"[boot] [VOICE] Sample {qi+1}: {dur:.1f}s")
                all_voice_samples.append(audio)

                # Transcribe so the engine can react contextually.
                transcript = None
                if self._services_up.get("whisper"):
                    try:
                        from voice.listener import transcribe
                        result = transcribe(audio, SAMPLE_RATE)
                        transcript = result[0] if isinstance(result, tuple) else result
                    except Exception:
                        pass
                if transcript:
                    print(f"[boot] [TRANSCRIPT] Response {qi+1}: '{transcript}'")

                if qi < n_voice_questions - 1:
                    ack_text, ack_style = convo.react(transcript=transcript)
                    self._speak(ack_text, ack_style)
                    print(f"[boot] [AURA/{ack_style}] {ack_text}")
                    self._mic.drain_echo(1.0)
                else:
                    convo.react(transcript=transcript)  # update history only
            else:
                print(f"[boot] [VOICE] No response to question {qi+1}")

        # ── Step 3b: Retry name transcription if still "User" ─────
        # Whisper may have hit CUDA OOM during name capture (Piper was
        # still on GPU).  By now GPU memory has settled — retry.
        if name == "User" and self._name_audio is not None:
            retried = self._try_transcribe_name(self._name_audio)
            if retried:
                name = retried
                print(f"[boot] [TRANSCRIPT] Retroactive name transcription: '{name}'")

        # ── Step 4: Enroll ────────────────────────────────────────
        valid_samples = [s for s in all_voice_samples if s is not None and len(s) > SAMPLE_RATE * 0.5]
        if not valid_samples:
            print("[boot] No usable voice samples — cannot enroll")
            self._use_persisted_user("no_voice_samples")
            return

        primary = max(valid_samples, key=len)
        try:
            self._set_phase(Phase.ENROLLMENT, self._progress_from_time(),
                            "Creating voice profile")
            user_id = enrollment.enroll(name=name, audio=primary)

            # Deepen with all remaining samples
            extras = [s for s in valid_samples if s is not primary]
            if extras:
                enrollment.deepen_profile(user_id, extras)

            state.active_user_id = user_id
            state.active_user_name = name
            state.boot_enrollment_done = True
            self._enrolled_this_boot = True
            self._user_id = user_id
            print(f"[boot] ═══════════════════════════════════════════")
            print(f"[boot] New user enrolled: {name} [{user_id}] "
                  f"({len(valid_samples)} voice samples)")
            print(f"[boot] ═══════════════════════════════════════════")
            bus.emit("boot.user_enrolled", user_id=user_id, name=name)

            # Closing — varied "I've got your voice" line from the
            # conversation engine's CLOSINGS pool.
            close_text, close_style = convo.closing(name)
            self._speak(close_text, close_style)
            print(f"[boot] [AURA/{close_style}] {close_text}")
        except Exception as e:
            print(f"[boot] Enrollment failed: {e}")
            import traceback
            traceback.print_exc()
            self._use_persisted_user("enrollment_failed")

    def _use_persisted_user(self, reason: str) -> None:
        """Fall back to the last known user from app_settings.json.

        This covers: mic busy, no audio captured, voice didn't match.
        The user already has profiles — we just couldn't verify this time.
        """
        uid = state.active_user_id
        name = state.active_user_name

        if uid and name:
            print(f"[boot] Using persisted user ({reason}): {name} [{uid[:8]}]")
            state.boot_enrollment_done = True
            bus.emit("boot.user_enrolled", user_id=uid, name=name)
        else:
            # Profiles exist on disk but no persisted state — grab first profile
            try:
                from boot.enrollment import VoiceEnrollment
                enr = self._get_enrollment()
                if enr and enr._profiles:
                    first_uid = next(iter(enr._profiles))
                    first_name = enr.get_name(first_uid) or "User"
                    state.active_user_id = first_uid
                    state.active_user_name = first_name
                    state.boot_enrollment_done = True
                    print(f"[boot] Recovered user from profiles ({reason}): "
                          f"{first_name} [{first_uid[:8]}]")
                    bus.emit("boot.user_enrolled", user_id=first_uid, name=first_name)
                else:
                    print(f"[boot] No persisted user and no profiles ({reason}) — "
                          "proceeding as guest")
            except Exception as e:
                print(f"[boot] Could not recover user ({reason}): {e}")

    # ------------------------------------------------------------------
    # Boot path B: first-time user (no profiles)
    # ------------------------------------------------------------------

    def _boot_first_time(self, enrollment) -> None:
        """First boot: continuously listen and enroll anyone who speaks.

        No tight capture windows. Sits silently until it hears a voice,
        then runs the conversational enrollment flow. After enrolling one
        person, keeps listening for more voices (other family members).
        Continues until services are ready or skip is requested.
        """
        self._set_phase(Phase.WAITING_MIC, 0.02, "Waiting for microphone")
        mic_ready = self._mic.wait_for_mic()

        if not mic_ready or self._skip.is_set():
            self._set_phase(Phase.WAITING_SERVICES, 0.15, "Loading AI models")
            print("[boot] Mic unavailable for enrollment — skipping")
            return

        # Stop music so the mic doesn't hear the speaker
        self._stop_music()
        print("[boot] ═══════════════════════════════════════════")
        print("[boot] First boot — listening for voices to enroll")
        print("[boot] Will enroll anyone who speaks (no time limit)")
        print("[boot] ═══════════════════════════════════════════")
        self._set_phase(Phase.WAITING_MIC, 0.05, "Listening for a voice")

        _enrolled_count = 0
        _max_wait = 3600  # listen up to 1 hour total

        _start = time.time()
        while time.time() - _start < _max_wait and not self._skip.is_set():
            # Listen for any voice
            if _enrolled_count == 0:
                print("[boot] Waiting for someone to speak...")
            else:
                print(f"[boot] Listening for another voice ({_enrolled_count} enrolled so far)...")

            audio = self._mic.capture_utterance(
                max_duration=5.0,
                wait_timeout=60.0,
                silence_timeout=1.5,
            )

            if audio is None or len(audio) / SAMPLE_RATE < 0.5:
                continue

            dur = len(audio) / SAMPLE_RATE
            print(f"[boot] Voice detected! ({dur:.1f}s)")

            # If we already have profiles, check if this voice matches someone enrolled
            if _enrolled_count > 0:
                try:
                    uid, score = enrollment.identify(audio)
                    if uid:
                        name = enrollment.get_name(uid)
                        print(f"[boot] Already enrolled: {name} (score={score:.3f}) — skipping")
                        if _enrolled_count == 1:
                            # First person spoke again — we're done, they're alone
                            self._speak("I already have your voice. Is anyone else here who'd like to introduce themselves?")
                            print("[boot] [AURA] I already have your voice. Is anyone else here who'd like to introduce themselves?")
                            self._mic.drain_echo(1.5)
                            resp = self._mic.capture_utterance(max_duration=8.0, wait_timeout=15.0)
                            if resp is None:
                                print("[boot] No response — done enrolling")
                                break
                            # Check if the response is a new voice
                            uid2, score2 = enrollment.identify(resp)
                            if uid2:
                                print(f"[boot] Same person again (score={score2:.3f}) — done enrolling")
                                break
                            else:
                                print(f"[boot] New voice detected (score={score2:.3f}) — enrolling")
                                self._enroll_unknown_user(enrollment, resp)
                                _enrolled_count += 1
                                continue
                        continue
                except Exception as e:
                    print(f"[boot] Identification check failed: {e}")

            # New voice — run conversational enrollment
            self._enroll_unknown_user(enrollment, audio)
            _enrolled_count += 1

            # After first enrollment, ask if anyone else is around
            if _enrolled_count >= 1:
                self._speak("Is there anyone else here who'd like me to learn their voice?")
                print("[boot] [AURA] Is there anyone else here who'd like me to learn their voice?")
                self._mic.drain_echo(1.5)
                next_audio = self._mic.capture_utterance(
                    max_duration=5.0, wait_timeout=15.0)
                if next_audio is None:
                    print("[boot] No response — done enrolling")
                    break
                # Check if it's a new voice or same person
                if enrollment._profiles:
                    try:
                        uid, score = enrollment.identify(next_audio)
                        if uid:
                            print(f"[boot] Same person responded (score={score:.3f}) — checking response")
                            # Transcribe to see if they said yes/no
                            transcript = self._try_transcribe_name(next_audio)
                            if transcript:
                                lower = transcript.lower().strip()
                                print(f"[boot] [TRANSCRIPT] '{transcript}'")
                                if any(w in lower for w in ('no', 'nope', 'just me', 'only me', 'that\'s it', 'nobody')):
                                    print("[boot] No more people — done enrolling")
                                    break
                            # Keep listening for another person
                            continue
                        else:
                            # New voice! Enroll them
                            print(f"[boot] New voice! (score={score:.3f})")
                            self._enroll_unknown_user(enrollment, next_audio)
                            _enrolled_count += 1
                            continue
                    except Exception:
                        pass

        print(f"[boot] ═══════════════════════════════════════════")
        print(f"[boot] First boot enrollment complete: {_enrolled_count} people enrolled")
        print(f"[boot] ═══════════════════════════════════════════")
