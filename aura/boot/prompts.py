"""
boot.prompts -- Scripted conversation definitions for the boot phase.

Each BootPrompt maps a phase to a pre-recorded WAV file, expected response
type, capture duration, and fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from core.config import BOOT_PROMPTS_DIR


def _resolve_audio(filename: str) -> str:
    """Find the actual file, trying .wav/.mp3 alternates if needed.

    Given 'greeting_first.wav', checks for that file first, then tries
    'greeting_first.mp3' (and vice versa). Returns the path that exists,
    or the original path if neither is found (caller handles missing).
    """
    import os
    path = BOOT_PROMPTS_DIR / filename
    if os.path.isfile(path):
        return str(path)
    # Try alternate extension
    stem = path.stem
    alt_ext = ".mp3" if path.suffix == ".wav" else ".wav"
    alt = path.with_suffix(alt_ext)
    if os.path.isfile(alt):
        return str(alt)
    return str(path)  # original (will be caught as missing by orchestrator)


class ResponseType(Enum):
    """What kind of user response a prompt expects."""
    NONE = auto()        # No response expected (filler / announcement)
    NAME = auto()        # Short utterance — user says their name
    VOICE_SAMPLE = auto()  # Longer utterance for voice-print enrollment


@dataclass
class BootPrompt:
    """A single step in the boot conversation."""
    phase_name: str
    wav_file: str                           # filename inside BOOT_PROMPTS_DIR
    response_type: ResponseType = ResponseType.NONE
    capture_max_s: float = 5.0              # max recording duration
    timeout_s: float = 8.0                  # give up waiting for speech
    fallback_wav: Optional[str] = None      # played if user doesn't respond
    progress_text: str = ""                 # shown on the dial
    pause_before: float = 0.0              # seconds to wait before playing this prompt
    pause_after: float = 0.0               # seconds to wait after prompt finishes (before capture)

    @property
    def wav_path(self) -> str:
        return _resolve_audio(self.wav_file)

    @property
    def fallback_path(self) -> Optional[str]:
        if self.fallback_wav:
            return _resolve_audio(self.fallback_wav)
        return None


# ---------------------------------------------------------------------------
# First-boot script (no voice profile exists yet)
# ---------------------------------------------------------------------------

FIRST_BOOT_SCRIPT: list[BootPrompt] = [
    BootPrompt(
        phase_name="greeting",
        wav_file="greeting_first.wav",
        progress_text="Welcome to Aura",
        pause_before=0.0,         # no pause — jump right in after identification
        pause_after=2.5,         # breathing room after greeting
    ),
    BootPrompt(
        phase_name="ask_name",
        wav_file="ask_name.wav",
        response_type=ResponseType.NAME,
        capture_max_s=6.0,
        timeout_s=12.0,
        fallback_wav="no_name_fallback.wav",
        progress_text="What is your name?",
        pause_before=3.0,
        pause_after=0.5,
    ),
    BootPrompt(
        phase_name="confirm_name",
        wav_file="confirm_name.wav",
        progress_text="Confirming identity",
        pause_before=3.0,
        pause_after=2.0,
    ),
    BootPrompt(
        phase_name="ask_voice_sample",
        wav_file="ask_voice_sample.wav",
        response_type=ResponseType.VOICE_SAMPLE,
        capture_max_s=8.0,
        timeout_s=12.0,
        fallback_wav="no_voice_fallback.wav",
        progress_text="Voice enrollment",
        pause_before=3.0,
        pause_after=0.5,
    ),
    BootPrompt(
        phase_name="enrollment_done",
        wav_file="enrollment_done.wav",
        progress_text="Setting up your profile",
        pause_before=3.0,
    ),
]


# ---------------------------------------------------------------------------
# Returning-user script (voice profile already exists)
# ---------------------------------------------------------------------------

RETURNING_USER_SCRIPT: list[BootPrompt] = [
    # No spoken prompt at boot start — the old greeting_natural.wav
    # ("Hey, good to see you again") was a hardcoded recording the
    # user wanted removed. We still capture a voice sample silently
    # for ID; the boot-complete welcome (synthesized fresh in
    # orchestrator._run) is the only thing she says aloud.
    BootPrompt(
        phase_name="identify",
        wav_file="",                                # no audio playback
        response_type=ResponseType.VOICE_SAMPLE,
        capture_max_s=5.0,
        timeout_s=8.0,
        progress_text="Listening",
        pause_before=0.0,
        pause_after=0.3,
    ),
]
