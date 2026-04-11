#!/usr/bin/env python3
"""Regenerate ALL pre-synthesized WAVs using current Piper voice model.

Boot prompts, fillers, responses, greetings, tour lines, and thinking fillers.
Run on a puck: cd ~/Aura4 && python3 scripts/regenerate_all_wavs.py
"""

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIPER_MODEL = str(ROOT / "voices" / "aura_olga_19499.onnx")
BOOT_DIR = ROOT / "assets" / "boot_prompts"
THINK_DIR = ROOT / "assets" / "thinking_fillers"
LENGTH_SCALE = 1.05
NOISE_SCALE = 0.667
NOISE_W = 0.8
SAMPLE_RATE = 22050

# ---------------------------------------------------------------------------
# Boot fillers (20) — conversational questions during boot pauses
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
# Filler responses (10) — short acknowledgments
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
# Greetings
# ---------------------------------------------------------------------------
GREETINGS = {
    "greeting_first": "Hello! I'm Aura. It's so nice to meet you.",
    "greeting_natural": "Hey there! Good to see you.",
    "greeting_returning": "Welcome back. It's good to see you again.",
}

# ---------------------------------------------------------------------------
# Boot enrollment prompts
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

# ---------------------------------------------------------------------------
# Tour lines (from orchestrator.py)
# ---------------------------------------------------------------------------
TOUR_LINES = [
    "Let me give you a quick tour of your Aura.",
    "This is Topics Center. Browse and pin different tools to your dial.",
    "This is AuraNet, your connection to the wider Aura community and network.",
    "This is Settings, where you adjust your voice, language model, and preferences.",
    "This is Education. Tap it for interactive lessons and learning tools.",
    "This is the Mute button. It toggles the microphone, and also stops me mid-sentence.",
    "This is your Financial domain. Market data, portfolio tracking, and financial insights.",
    "This is Aura Concierge, your personal assistant for tasks, reminders, and recommendations.",
    "And this is Medical. Tap it for health insights, vitals, and clinical guidance.",
    "You can talk to me anytime. Just say what is on your mind, and I will do my best to help.",
]

# ---------------------------------------------------------------------------
# Breath fillers (5) — quick backchannel
# ---------------------------------------------------------------------------
BREATH_FILLERS = {
    "breath_givemeasecond": "Give me a second.",
    "breath_happytotalkaboutthat": "Happy to talk about that.",
    "breath_i'lldomybest": "I'll do my best.",
    "breath_okwell": "Okay, well.",
    "breath_yeahcoolso": "Yeah, cool, so.",
}

# ---------------------------------------------------------------------------
# Think fillers (55) — verbal thinking sounds
# ---------------------------------------------------------------------------
THINK_FILLERS = [
    "Let me think about that.",
    "That's a good question.",
    "Hmm, interesting.",
    "Let me consider that for a moment.",
    "Oh, that's a great question.",
    "Well, let me see.",
    "Okay, give me a second.",
    "Hmm, let me think.",
    "That's interesting, actually.",
    "Sure, I get it.",
    "Alright, let me work through that.",
    "Oh, that's a thoughtful one.",
    "Let me pull that together.",
    "Interesting question.",
    "Yeah, let me figure that out.",
    "Hmm, okay.",
    "Good question, let me think.",
    "That's worth thinking about.",
    "Let me consider a few things.",
    "Alright, one moment.",
    "Oh, I see what you mean.",
    "Let me look into that.",
    "Yeah, that's a fair question.",
    "Hmm, let me work on that.",
    "Okay, interesting.",
    "Let me give that some thought.",
    "Sure, let me think on that.",
    "Alright, let me see what I can do.",
    "That's a really good point.",
    "Hmm, give me a moment.",
    "Let me think through that carefully.",
    "Oh, that's an interesting one.",
    "Okay, let me figure this out.",
    "Yeah, I can work with that.",
    "Let me sort through that.",
    "Hmm, that's thought-provoking.",
    "Alright, bear with me.",
    "Let me take a moment on that.",
    "Oh, sure, let me think.",
    "Okay, that's a good one.",
    "Let me gather my thoughts.",
    "Hmm, there's a lot to consider.",
    "Yeah, let me work through this.",
    "Alright, interesting question.",
    "Let me dig into that.",
    "Oh, I like that question.",
    "Give me just a moment.",
    "Hmm, let me ponder that.",
    "Okay, let me think about it.",
    "That's worth exploring.",
    "Let me see what comes to mind.",
    "Alright, let me process that.",
    "Hmm, I want to give that a proper answer.",
    "Let me reflect on that.",
    "Oh, that's a deep one.",
]

# ---------------------------------------------------------------------------
# Verbal fillers (50) — longer thinking responses
# ---------------------------------------------------------------------------
VERBAL_FILLERS = [
    "That's a really interesting question, let me think about it.",
    "Hmm, give me a moment to consider that.",
    "Oh, I want to give you a good answer on that one.",
    "Let me think about the best way to explain this.",
    "That's worth taking a moment to think through.",
    "Okay, there are a few things to consider here.",
    "Hmm, let me put my thoughts together on that.",
    "Sure, that's a great question to dig into.",
    "Let me take a second to think about that properly.",
    "Oh, interesting. Give me a moment.",
    "I want to make sure I give you a thoughtful answer.",
    "Hmm, there's a lot I could say about that.",
    "Let me think about what matters most here.",
    "That's a nuanced question, let me work through it.",
    "Okay, let me consider the different angles.",
    "Hmm, I'm thinking about the best way to put this.",
    "Oh, that's a really good one. Let me think.",
    "Let me organize my thoughts on that.",
    "Sure, give me just a second.",
    "That deserves a careful answer, let me think.",
    "Hmm, I want to be accurate about this.",
    "Let me consider that from a few perspectives.",
    "Oh, there's a lot to unpack there.",
    "Okay, let me think through the key points.",
    "Hmm, that's something worth exploring.",
    "Let me take a moment to reflect on that.",
    "Sure, I want to get this right.",
    "That's a thoughtful question, give me a moment.",
    "Hmm, let me see how to best answer that.",
    "Let me think about what would be most helpful.",
    "Oh, I have some thoughts on that, let me organize them.",
    "Okay, that's an interesting angle to think about.",
    "Hmm, let me work through the details.",
    "Let me consider what's most relevant here.",
    "Sure, there's a few ways to look at this.",
    "That's a great topic, let me think about it.",
    "Hmm, I want to give you something useful here.",
    "Let me take a beat on that.",
    "Oh, that touches on something interesting.",
    "Okay, let me pull together what I know.",
    "Hmm, I'm working through that in my head.",
    "Let me think about the big picture here.",
    "Sure, that's worth a thoughtful response.",
    "That's an important question, let me reflect.",
    "Hmm, let me consider the implications.",
    "Let me think about how to frame this.",
    "Oh, there's a lot to that question.",
    "Okay, give me a moment to think clearly.",
    "Hmm, I'm processing that.",
    "Let me take a second and think this through.",
]


_voice = None
_syn_config = None


def _load_piper():
    global _voice, _syn_config
    if _voice is not None:
        return
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    _voice = PiperVoice.load(PIPER_MODEL)
    _syn_config = SynthesisConfig(
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_w_scale=NOISE_W,
        normalize_audio=False,
    )


def synthesize(text: str, out_path: Path):
    """Synthesize text to WAV using Piper Python API."""
    _load_piper()
    parts = []
    for chunk in _voice.synthesize(text, syn_config=_syn_config):
        parts.append(chunk.audio_float_array)
    if not parts:
        print(f"  WARNING: no audio for '{text}'")
        return False
    audio = np.concatenate(parts)
    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return True


def main():
    filler_dir = BOOT_DIR / "fillers"
    response_dir = BOOT_DIR / "filler_responses"
    tour_dir = BOOT_DIR / "tour"
    filler_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    tour_dir.mkdir(parents=True, exist_ok=True)
    THINK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {PIPER_MODEL}")
    print(f"length_scale={LENGTH_SCALE}, noise_scale={NOISE_SCALE}, noise_w={NOISE_W}")
    print()

    # Load model once
    _load_piper()
    print("Piper loaded.\n")

    total = 0

    # Boot fillers
    print(f"--- {len(FILLERS)} boot fillers ---")
    for i, text in enumerate(FILLERS):
        out = filler_dir / f"filler_{i:03d}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Responses
    print(f"--- {len(RESPONSES)} filler responses ---")
    for i, text in enumerate(RESPONSES):
        out = response_dir / f"response_{i:03d}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Greetings
    print(f"--- {len(GREETINGS)} greetings ---")
    for name, text in GREETINGS.items():
        out = BOOT_DIR / f"{name}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Boot prompts
    print(f"--- {len(BOOT_PROMPTS)} boot prompts ---")
    for name, text in BOOT_PROMPTS.items():
        out = BOOT_DIR / f"{name}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Tour
    print(f"--- {len(TOUR_LINES)} tour lines ---")
    for i, text in enumerate(TOUR_LINES):
        out = tour_dir / f"tour_{i}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Breath fillers
    print(f"--- {len(BREATH_FILLERS)} breath fillers ---")
    for name, text in BREATH_FILLERS.items():
        out = THINK_DIR / f"{name}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Think fillers
    print(f"--- {len(THINK_FILLERS)} think fillers ---")
    for i, text in enumerate(THINK_FILLERS):
        out = THINK_DIR / f"think_{i:03d}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    # Verbal fillers
    print(f"--- {len(VERBAL_FILLERS)} verbal fillers ---")
    for i, text in enumerate(VERBAL_FILLERS):
        out = THINK_DIR / f"verbal_{i:03d}.wav"
        synthesize(text, out)
        total += 1
    print(f"  done ({total})")

    print(f"\nAll done! {total} WAVs regenerated with {Path(PIPER_MODEL).stem}")


if __name__ == "__main__":
    main()
