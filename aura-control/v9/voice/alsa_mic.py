"""
voice.alsa_mic -- Direct ALSA mic capture for the ReSpeaker XVF3800.

PortAudio/sounddevice cannot see the ReSpeaker as an input device on Jetson,
so we use alsaaudio to talk to ALSA directly.  This module provides a thin
wrapper with a .read() interface compatible with what listener.py and
boot/voice_capture.py expect.
"""

from __future__ import annotations

import numpy as np

# ReSpeaker UAC1.0 on Jetson: 6 channels, 16kHz, S16_LE.
# Channel 0 = beamformed output (the one we want).
_CHANNELS = 6
_FORMAT_BYTES = 2  # S16_LE


class AlsaMic:
    """ALSA PCM capture handle for the ReSpeaker."""

    def __init__(self, device: str, rate: int = 16000, period_size: int = 512):
        import alsaaudio

        self._pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=device,
            channels=_CHANNELS,
            rate=rate,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=period_size,
        )
        self._rate = rate
        self._period_size = period_size
        self._channels = _CHANNELS
        print(f"[alsa_mic] Opened {device} ({_CHANNELS}ch, {rate}Hz, period={period_size})")

    def read(self, frame_count: int) -> tuple[np.ndarray, bool]:
        """Read frames, return (data, overflowed).

        Returns a (N, channels) int16 ndarray — same shape as sounddevice.
        """
        # alsaaudio.read() returns (length, bytes)
        n_bytes_needed = frame_count * self._channels * _FORMAT_BYTES
        chunks = []
        collected = 0

        while collected < n_bytes_needed:
            length, data = self._pcm.read()
            if length < 0:
                # Overrun — return what we have
                break
            if length > 0 and data:
                chunks.append(data)
                collected += len(data)

        if not chunks:
            # Return silence frame
            return np.zeros((frame_count, self._channels), dtype=np.int16), True

        raw = b"".join(chunks)
        samples = np.frombuffer(raw, dtype=np.int16)
        # Reshape to (frames, channels)
        n_frames = len(samples) // self._channels
        data_2d = samples[:n_frames * self._channels].reshape(n_frames, self._channels)
        return data_2d, False

    def stop(self):
        """No-op for API compat."""
        pass

    def close(self):
        """Close the PCM handle."""
        try:
            self._pcm.close()
        except Exception:
            pass

    def start(self):
        """No-op for API compat (ALSA capture starts on open)."""
        pass
