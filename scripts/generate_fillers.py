#!/usr/bin/env python3
"""
Generate 100 thinking filler WAVs using Piper TTS.
Run on puck: python3 scripts/generate_fillers.py
"""

import subprocess
import os

PIPER_BIN = "/home/ledger/piper/piper"
PIPER_MODEL = "/home/ledger/Aura4/aura-control/voices/aura_olga_1250.onnx"
OUTPUT_DIR = "/home/ledger/Aura4/assets/thinking_fillers"

# Quick breath fillers (short, 1-3 words) — for simple queries
BREATH_FILLERS = [
    "Okay, so",
    "Alright",
    "Sure thing",
    "Let's see",
    "Hmm, okay",
    "Right",
    "Gotcha",
    "Mmhmm",
    "Oh, sure",
    "Ah, yes",
    "Of course",
    "Okay, let me think",
    "So",
    "Well",
    "Alright, so",
    "Yeah, let's see",
    "Okay, one sec",
    "Sure",
    "Right, okay",
    "Let me see",
    "Hmm",
    "Ah",
    "Oh, okay",
    "Got it",
    "Alright, alright",
    "Yep",
    "Ooh, okay",
    "Mmm, let's see",
    "Oh, right",
    "Okay, okay",
    "Yeah, so",
    "Alright, let me check",
    "Hmm, sure",
    "Good question",
    "Ah, gotcha",
    "Okay, here we go",
    "Let me look",
    "One moment",
    "Alright, here we go",
    "Yeah, okay",
    "Oh, interesting",
    "Let me think",
    "Hmm, well",
    "Right, so",
    "Okay, well",
    "Sure, let's see",
    "Yeah, hold on",
    "Oh, let me check",
    "Alright, sure",
    "Gotcha, one sec",
]

# Complex thinking fillers (longer, ~3-5s) — for complex queries
THINK_FILLERS = [
    "That's a great question, let me think about that.",
    "Okay, let me pull that together for you.",
    "Hmm, give me just a moment on that one.",
    "Alright, let me work through this.",
    "Oh, that's interesting. Let me think.",
    "Sure, let me figure that out.",
    "Good question. Let me consider that.",
    "Okay, there's a few things to consider here.",
    "Let me think about the best way to explain this.",
    "Hmm, that's a thoughtful question.",
    "Alright, let me get that information for you.",
    "Oh, I know this one. Just a sec.",
    "Let me break that down for you.",
    "Okay, so there's a couple of ways to look at this.",
    "Hmm, let me recall what I know about that.",
    "Sure, that's a good one to think about.",
    "Alright, give me a moment to think that through.",
    "Oh, right. Let me pull up what I know.",
    "Let me consider that for just a second.",
    "Okay, I want to give you a good answer here.",
    "That's worth thinking about carefully.",
    "Hmm, let me put this together.",
    "Alright, I want to make sure I get this right.",
    "Oh, there's actually a lot to say about that.",
    "Let me think about how to best answer this.",
    "Okay, so this is an interesting topic.",
    "Good one. Let me gather my thoughts.",
    "Hmm, I'm thinking about that right now.",
    "Alright, let me work on that for you.",
    "Sure, give me a quick second.",
    "Let me sort through this.",
    "Okay, I know what you're asking. Let me think.",
    "Oh, that's a fun question.",
    "Hmm, there are a few angles on this.",
    "Alright, I'll have that for you in just a moment.",
    "Let me think about the key points here.",
    "Okay, so I want to be thorough with this.",
    "Good question. Give me just a sec.",
    "Hmm, I'm working on that.",
    "Oh, right. I've got some thoughts on this.",
    "Let me figure out the best way to say this.",
    "Okay, there's definitely an answer here.",
    "Alright, I'm thinking it through.",
    "Sure, let me gather what I know.",
    "Hmm, that takes a bit of thought.",
    "Let me work through that for a moment.",
    "Okay, I want to be precise about this.",
    "Oh, interesting question. Let me think.",
    "Alright, so let me consider the options.",
    "Give me just a second to think about that.",
]


def generate(text, output_path):
    """Generate a WAV using Piper CLI binary."""
    try:
        result = subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", output_path],
            input=text, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  PIPER ERROR: {result.stderr.strip()[:80]}")
            return False
        if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate breath fillers (breath_100 through breath_149)
    print(f"Generating {len(BREATH_FILLERS)} breath fillers...")
    ok = 0
    for i, phrase in enumerate(BREATH_FILLERS):
        fname = f"breath_{100 + i:03d}.wav"
        path = os.path.join(OUTPUT_DIR, fname)
        if generate(phrase, path):
            ok += 1
            print(f"  [{ok}] {fname}: \"{phrase}\"")
        else:
            print(f"  FAIL: {fname}: \"{phrase}\"")
    print(f"  Done: {ok}/{len(BREATH_FILLERS)} breath fillers\n")

    # Generate think fillers (think_100 through think_149)
    print(f"Generating {len(THINK_FILLERS)} think fillers...")
    ok2 = 0
    for i, phrase in enumerate(THINK_FILLERS):
        fname = f"think_{100 + i:03d}.wav"
        path = os.path.join(OUTPUT_DIR, fname)
        if generate(phrase, path):
            ok2 += 1
            print(f"  [{ok2}] {fname}: \"{phrase}\"")
        else:
            print(f"  FAIL: {fname}: \"{phrase}\"")
    print(f"  Done: {ok2}/{len(THINK_FILLERS)} think fillers\n")

    print(f"TOTAL: {ok + ok2} new fillers generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
