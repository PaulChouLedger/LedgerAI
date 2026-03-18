#!/usr/bin/env python3
"""Regenerate all boot prompt WAVs using Piper + Olga voice model.

This ensures all boot audio (fillers, responses, greetings) use the same
voice as the live Piper TTS, eliminating the ElevenLabs voice mismatch.

Uses the same synthesis parameters as the puck: length_scale=1.15,
noise_scale=0.667, noise_w=0.8.
"""

import subprocess
import sys
from pathlib import Path

PIPER_MODEL = str(Path(__file__).resolve().parents[1] / "voices" / "aura_olga_1250.onnx")
BOOT_DIR = Path(__file__).resolve().parents[1] / "assets" / "boot_prompts"
LENGTH_SCALE = "1.15"
NOISE_SCALE = "0.667"
NOISE_W = "0.8"

# ---------------------------------------------------------------------------
# Filler questions (20) — conversational, warm, asked during boot pauses
# ---------------------------------------------------------------------------
FILLERS = [
    "So, what's been on your mind lately?",
    "Tell me something good that happened to you recently.",
    "If you could travel anywhere right now, where would you go?",
    "What's your favorite way to unwind after a long day?",
    "Have you been watching anything interesting lately?",
    "What's the best meal you've had this week?",
    "If you could learn any new skill overnight, what would it be?",
    "What's something you're looking forward to?",
    "Do you have any weekend plans coming up?",
    "What kind of music have you been listening to?",
    "Is there a book or podcast you'd recommend?",
    "What's the most interesting thing you learned recently?",
    "If you could have dinner with anyone, who would it be?",
    "What's your favorite thing about where you live?",
    "Do you have a morning routine that helps you start the day?",
    "What's a hobby you've always wanted to try?",
    "What's the best advice anyone ever gave you?",
    "If you had a free afternoon with nothing to do, how would you spend it?",
    "What's something small that always makes you smile?",
    "Tell me about someone who inspires you.",
]

# ---------------------------------------------------------------------------
# Filler responses (10) — short acknowledgments after user answers
# ---------------------------------------------------------------------------
RESPONSES = [
    "Oh, I love that.",
    "That's really nice.",
    "What a great answer.",
    "I like the way you think.",
    "That sounds wonderful.",
    "Oh, that's lovely.",
    "I can see why.",
    "That's a good one.",
    "I appreciate you sharing that.",
    "How nice.",
]

# ---------------------------------------------------------------------------
# Boot greetings
# ---------------------------------------------------------------------------
GREETINGS = {
    "greeting_first": "Hello! I'm Aura. It's so nice to meet you.",
    "greeting_natural": "Hey there! Good to see you.",
    "greeting_returning": "Welcome back. It's good to see you again.",
}

# ---------------------------------------------------------------------------
# Boot enrollment prompts — played during first-boot and returning-user flows
# ---------------------------------------------------------------------------
BOOT_PROMPTS = {
    "ask_name": "I'd love to know your name. What should I call you?",
    "confirm_name": "Got it, thank you.",
    "ask_voice_sample": "Now, tell me something about yourself so I can learn your voice.",
    "no_name_fallback": "No worries, we can do that later.",
    "no_voice_fallback": "That's alright. I'll learn your voice as we talk.",
    "enrollment_done": "Perfect, you're all set. I've saved your voice profile.",
    "casual_waiting": "Just a moment while I finish setting things up.",
    "waiting_filler": "Still loading a few things. Won't be much longer.",
}


def synthesize(text: str, out_path: Path):
    """Run piper CLI to synthesize text to WAV."""
    result = subprocess.run(
        [
            "piper",
            "--model", PIPER_MODEL,
            "--length-scale", LENGTH_SCALE,
            "--noise-scale", NOISE_SCALE,
            "--noise-w-scale", NOISE_W,
            "--output-file", str(out_path),
        ],
        input=text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False

    # Get duration
    try:
        dur = subprocess.run(
            ["soxi", "-D", str(out_path)],
            capture_output=True, text=True, timeout=5
        )
        print(f"  {out_path.name}: {float(dur.stdout.strip()):.1f}s")
    except Exception:
        print(f"  {out_path.name}: done")
    return True


def main():
    filler_dir = BOOT_DIR / "fillers"
    response_dir = BOOT_DIR / "filler_responses"
    filler_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {PIPER_MODEL}")
    print(f"Output: {BOOT_DIR}")
    print(f"length_scale={LENGTH_SCALE}, noise_scale={NOISE_SCALE}, noise_w={NOISE_W}")
    print()

    # Fillers
    print(f"--- Generating {len(FILLERS)} fillers ---")
    for i, text in enumerate(FILLERS):
        out = filler_dir / f"filler_{i:03d}.wav"
        print(f"  [{i+1}/{len(FILLERS)}] \"{text}\"")
        synthesize(text, out)

    # Responses
    print(f"\n--- Generating {len(RESPONSES)} responses ---")
    for i, text in enumerate(RESPONSES):
        out = response_dir / f"response_{i:03d}.wav"
        print(f"  [{i+1}/{len(RESPONSES)}] \"{text}\"")
        synthesize(text, out)

    # Greetings
    print(f"\n--- Generating {len(GREETINGS)} greetings ---")
    for name, text in GREETINGS.items():
        out = BOOT_DIR / f"{name}.wav"
        print(f"  \"{text}\"")
        synthesize(text, out)

    # Boot enrollment prompts
    print(f"\n--- Generating {len(BOOT_PROMPTS)} boot prompts ---")
    for name, text in BOOT_PROMPTS.items():
        out = BOOT_DIR / f"{name}.wav"
        print(f"  \"{text}\"")
        synthesize(text, out)

    print("\nDone! All boot WAVs regenerated with Piper Olga voice.")


if __name__ == "__main__":
    main()
