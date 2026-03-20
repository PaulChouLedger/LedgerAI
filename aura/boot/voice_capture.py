"""
boot.voice_capture -- Lightweight mic capture for the boot phase.

Opens sounddevice InputStream on the XVF3800 (hw:1,0, 2ch, 16kHz, int16).
Uses Silero VAD to detect speech boundaries. Returns raw numpy arrays --
does NOT call Whisper.

Reuses voice.listener.find_device_index() and voice.listener._get_vad()
(same lazy-init singleton).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import threading
from typing import Optional

import numpy as np

from core.config import (
    SAMPLE_RATE,
    BOOT_CAPTURE_MAX_S,
    BOOT_CAPTURE_SILENCE_S,
    BOOT_CAPTURE_TIMEOUT_S,
    BOOT_MIC_TIMEOUT_S,
)


class BootMic:
    """Boot-time mic interface: wait for hardware, capture utterances, play prompts."""

    def __init__(self) -> None:
        self._stream = None
        self._alsa_device: Optional[str] = None  # set by wait_for_mic()
        self._play_proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Mic availability
    # ------------------------------------------------------------------

    def wait_for_mic(self, timeout: float = BOOT_MIC_TIMEOUT_S) -> bool:
        """Block until the XVF3800 USB device appears in ALSA.

        Also pre-loads the Silero VAD model so the first capture_utterance()
        call doesn't stall while the model initialises.

        Returns True if found, False on timeout.
        """
        from voice.listener import _find_alsa_card
        deadline = time.time() + timeout
        while time.time() < deadline:
            alsa_dev = _find_alsa_card("Array", max_retries=1)
            if alsa_dev is not None:
                self._alsa_device = alsa_dev
                print(f"[boot_mic] XVF3800 found: {alsa_dev}")
                # Pre-load VAD so first capture doesn't stall
                from voice.listener import _get_vad
                _get_vad()
                return True
            time.sleep(2.0)
        print("[boot_mic] XVF3800 not found within timeout")
        return False

    def _open_stream(self):
        """Open ALSA capture on the ReSpeaker (lazy, reusable).

        Uses alsaaudio directly — PortAudio/sounddevice cannot see the
        ReSpeaker as an input device on Jetson.
        """
        if self._stream is not None:
            return True

        from voice.alsa_mic import AlsaMic

        device = getattr(self, '_alsa_device', None)
        if device is None:
            from voice.listener import _find_alsa_card
            device = _find_alsa_card("Array", max_retries=3) or "hw:1,0"

        for attempt in range(5):
            try:
                self._stream = AlsaMic(
                    device=device,
                    rate=SAMPLE_RATE,
                    period_size=int(SAMPLE_RATE * 0.032),
                )
                print(f"[boot_mic] Stream opened (device={device})")
                return True
            except Exception as e:
                print(f"[boot_mic] Cannot open stream (attempt {attempt+1}/5): {e}")
                self._stream = None
                if attempt < 4:
                    time.sleep(1.0 * (attempt + 1))

        return False

    def close(self):
        """Release the mic stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_utterance(
        self,
        max_duration: float = BOOT_CAPTURE_MAX_S,
        silence_timeout: float = BOOT_CAPTURE_SILENCE_S,
        wait_timeout: float = BOOT_CAPTURE_TIMEOUT_S,
    ) -> Optional[np.ndarray]:
        """VAD-gated capture. Returns float32 mono array, or None on timeout.

        1. Waits up to *wait_timeout* for speech to start (VAD > threshold).
        2. Records up to *max_duration* seconds.
        3. Ends on *silence_timeout* seconds of silence.
        """
        if not self._open_stream():
            return None

        import torch
        from voice.listener import _get_vad

        vad = _get_vad()
        frame_size = int(SAMPLE_RATE * 0.032)
        vad_start_thresh = 0.06
        vad_silence_thresh = 0.04
        mic_channel = 0

        # Phase 1: wait for speech onset
        print(f"[boot_mic] Listening for speech (timeout={wait_timeout:.1f}s, "
              f"vad_thresh={vad_start_thresh})...")
        deadline = time.time() + wait_timeout
        frame_count = 0
        while time.time() < deadline:
            try:
                data, _ = self._stream.read(frame_size)
            except Exception:
                time.sleep(0.01)
                continue

            if data.ndim > 1:
                mono = data[:, mic_channel].astype(np.float32) / 32768.0
            else:
                mono = data.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(mono * mono)))
            tensor = torch.from_numpy(mono)
            prob = float(vad(tensor, SAMPLE_RATE).detach())
            frame_count += 1
            # Log every ~1s (roughly 31 frames at 32ms each)
            if frame_count % 31 == 0:
                print(f"[boot_mic] waiting: rms={rms:.4f}  vad={prob:.3f}  "
                      f"(need >={vad_start_thresh})")
            if prob >= vad_start_thresh:
                print(f"[boot_mic] Speech detected! vad={prob:.3f} rms={rms:.4f}")
                break
        else:
            # Timed out waiting for speech
            print(f"[boot_mic] Timed out waiting for speech after {wait_timeout:.1f}s")
            vad.reset_states()
            return None

        # Phase 2: record until silence or max_duration
        buffer = []
        rec_deadline = time.time() + max_duration
        silence_start: Optional[float] = None

        # Include the triggering frame
        if data.ndim > 1:
            mono_full = data[:, mic_channel].astype(np.float32) / 32768.0
        else:
            mono_full = data.astype(np.float32) / 32768.0
        buffer.append(mono_full)

        while time.time() < rec_deadline:
            try:
                data2, _ = self._stream.read(frame_size)
            except Exception:
                break

            if data2.ndim > 1:
                mono2 = data2[:, mic_channel].astype(np.float32) / 32768.0
            else:
                mono2 = data2.astype(np.float32) / 32768.0

            buffer.append(mono2)

            tensor2 = torch.from_numpy(mono2)
            prob2 = float(vad(tensor2, SAMPLE_RATE).detach())

            if prob2 < vad_silence_thresh:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) >= silence_timeout:
                    break
            else:
                silence_start = None

        vad.reset_states()

        if not buffer:
            print("[boot_mic] No audio frames captured")
            return None

        audio = np.concatenate(buffer)
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio * audio)))
        peak = float(np.max(np.abs(audio)))

        if len(audio) < 2000:  # ~125ms, too short
            print(f"[boot_mic] Capture too short: {duration:.2f}s ({len(audio)} samples)")
            return None

        print(f"[boot_mic] Captured {duration:.2f}s of audio  "
              f"rms={rms:.4f}  peak={peak:.4f}  samples={len(audio)}")

        # Save to /tmp for diagnostic playback
        try:
            import wave as _wave
            diag_path = f"/tmp/boot_capture_{int(time.time())}.wav"
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with _wave.open(diag_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            print(f"[boot_mic] Diagnostic WAV saved: {diag_path}")
        except Exception as e:
            print(f"[boot_mic] Could not save diagnostic WAV: {e}")

        return audio

    # ------------------------------------------------------------------
    # Prompt playback
    # ------------------------------------------------------------------

    def play_prompt(self, path: str) -> None:
        """Non-blocking audio playback (WAV or MP3) via direct ALSA.

        MP3s are decoded with ffmpeg and piped to aplay.
        WAVs are played directly with aplay.
        """
        from core.config import ALSA_PLAYBACK_DEVICE
        if not os.path.isfile(path):
            print(f"[boot_mic] Prompt not found: {path}")
            return

        # Kill any currently playing prompt
        self.stop_prompt()

        is_mp3 = path.lower().endswith(".mp3")

        try:
            if is_mp3 and shutil.which("ffmpeg"):
                # Decode MP3 → raw PCM, pipe to aplay
                ff = subprocess.Popen(
                    ["ffmpeg", "-i", path, "-loglevel", "quiet",
                     "-f", "s16le", "-acodec", "pcm_s16le",
                     "-ac", "2", "-ar", "48000", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                self._play_proc = subprocess.Popen(
                    ["aplay", "-D", ALSA_PLAYBACK_DEVICE,
                     "-f", "S16_LE", "-c", "2", "-r", "48000", "-q"],
                    stdin=ff.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                ff.stdout.close()
            else:
                self._play_proc = subprocess.Popen(
                    ["aplay", "-D", ALSA_PLAYBACK_DEVICE, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
        except Exception as e:
            print(f"[boot_mic] Playback error: {e}")
            self._play_proc = None

    def wait_for_prompt(self, timeout: float = 15.0) -> None:
        """Block until the current prompt finishes playing."""
        if self._play_proc is None:
            return
        try:
            self._play_proc.wait(timeout=timeout)
            rc = self._play_proc.returncode
            if rc != 0:
                stderr = ""
                try:
                    stderr = self._play_proc.stderr.read().decode(errors="replace").strip()
                except Exception:
                    pass
                print(f"[boot_mic] Prompt player exited with code {rc}"
                      + (f": {stderr[:200]}" if stderr else ""))
        except subprocess.TimeoutExpired:
            self.stop_prompt()

    def stop_prompt(self) -> None:
        """Kill any running playback."""
        if self._play_proc is not None:
            try:
                if self._play_proc.poll() is None:
                    self._play_proc.terminate()
            except Exception:
                pass
            self._play_proc = None
