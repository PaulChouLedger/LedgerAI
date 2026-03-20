#!/usr/bin/env python3
"""
Lincoln vs Schwarzenegger — Presidential Debate Showdown.

Pre-generates the full debate script with Ollama, synthesizes with
ChatterboxTTS voice cloning (Lincoln & Arnold) and Piper (moderator),
then plays back as a produced audio piece. All on RTX GPU.

Usage:
    python3 tests/lincoln_vs_arnold_debate.py
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIPER_LESSAC = "/tmp/piper_test_voice/en_US-lessac-medium.onnx"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:32b"

# Voice cloning reference clips
ARNOLD_REF = "/tmp/voice_refs/arnold_ref.wav"
LINCOLN_REF = "/tmp/voice_refs/lincoln_ref_v3.wav"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIO_FILE = LOG_DIR / f"lincoln_vs_arnold_{RUN_ID}.wav"
TRANSCRIPT_FILE = LOG_DIR / f"lincoln_vs_arnold_{RUN_ID}.txt"
WORK_DIR = Path("/tmp/debate_audio")
WORK_DIR.mkdir(exist_ok=True)

# Sox post-processing per speaker (applied after TTS)
# tempo 0.82 slows speech ~18% for gravitas
LINCOLN_SOX = ["tempo", "0.80", "bass", "+3", "treble", "-2", "reverb", "15", "norm", "-1"]
ARNOLD_SOX = ["tempo", "0.85", "treble", "+2", "norm", "-1"]
MOD_SOX = ["norm", "-1"]

# Piper settings for moderator
MOD_PIPER = {"length_scale": "1.05", "noise_scale": "0.667", "noise_w": "0.8"}

# ---------------------------------------------------------------------------
# Script generation prompt
# ---------------------------------------------------------------------------
SCRIPT_PROMPT = """Write a 3-minute presidential debate between Abraham Lincoln and Arnold Schwarzenegger.

FORMAT: Return ONLY a JSON array. Each element is an object with "speaker" (one of "MODERATOR", "LINCOLN", "ARNOLD") and "line" (the spoken text).

RULES:
- The moderator opens, introduces both candidates, and closes at the end.
- 10-12 exchanges total (alternating Lincoln and Arnold, with moderator interjections).
- Lincoln speaks in eloquent, 1800s style but about modern issues. He's witty and sharp.
- Arnold speaks like himself — direct, uses action movie one-liners, "I'll be back" references, gym metaphors. He's funny and confident.
- Topics: economy, immigration, climate, leadership style.
- They should roast each other a bit — Lincoln jokes about Arnold's movies, Arnold jokes about Lincoln's hat/beard.
- End with a strong closing statement from each, then moderator wraps up.
- Keep each line to 1-3 sentences max. This needs to be punchy and entertaining.
- Total spoken time should be about 3 minutes.

Return ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# ChatterboxTTS singleton (lazy-loaded)
# ---------------------------------------------------------------------------
_chatterbox_model = None

def get_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is None:
        print("  Loading ChatterboxTTS...")
        from chatterbox.tts import ChatterboxTTS
        _chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
        print("  ChatterboxTTS ready.")
    return _chatterbox_model


def ollama_generate(prompt, timeout=120):
    """Call Ollama to generate the debate script."""
    try:
        resp = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", json.dumps({
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a comedy writer. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            })],
            capture_output=True, text=True, timeout=timeout,
        )
        data = json.loads(resp.stdout)
        content = data.get("message", {}).get("content", "").strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"  [Ollama error: {e}]")
        return None


def synthesize_chatterbox(text, ref_wav, output_wav):
    """Synthesize with ChatterboxTTS voice cloning."""
    import torchaudio
    model = get_chatterbox()
    wav = model.generate(text, audio_prompt_path=ref_wav)
    torchaudio.save(str(output_wav), wav, model.sr)
    return output_wav.exists()


def synthesize_piper(text, output_wav):
    """Synthesize with Piper (moderator voice)."""
    subprocess.run(
        ["piper", "--model", PIPER_LESSAC,
         "--length-scale", MOD_PIPER["length_scale"],
         "--noise-scale", MOD_PIPER["noise_scale"],
         "--noise-w-scale", MOD_PIPER["noise_w"],
         "--output_file", str(output_wav)],
        input=text, capture_output=True, text=True, timeout=30,
    )
    return output_wav.exists()


def synthesize_line(text, speaker, index):
    """Synthesize a line with speaker-specific voice."""
    raw_wav = WORK_DIR / f"raw_{index:03d}.wav"
    final_wav = WORK_DIR / f"line_{index:03d}.wav"

    t0 = time.time()

    if speaker == "LINCOLN":
        ok = synthesize_chatterbox(text, LINCOLN_REF, raw_wav)
        sox_effects = LINCOLN_SOX
    elif speaker == "ARNOLD":
        ok = synthesize_chatterbox(text, ARNOLD_REF, raw_wav)
        sox_effects = ARNOLD_SOX
    else:
        ok = synthesize_piper(text, raw_wav)
        sox_effects = MOD_SOX

    if not ok:
        print(f"  [FAILED] Could not synthesize: {text[:50]}")
        return None

    # Resample to common rate + apply sox effects
    subprocess.run(
        ["sox", str(raw_wav), "-r", "22050", str(final_wav)] + sox_effects,
        capture_output=True, timeout=15,
    )

    # Get duration
    try:
        result = subprocess.run(
            ["soxi", "-D", str(final_wav)], capture_output=True, text=True, timeout=5
        )
        dur = float(result.stdout.strip())
    except Exception:
        dur = 0

    synth_time = time.time() - t0
    return final_wav, dur, synth_time


def add_silence(duration_s, index):
    """Generate a silence WAV."""
    silence_wav = WORK_DIR / f"silence_{index:03d}.wav"
    subprocess.run(
        ["sox", "-n", "-r", "22050", "-c", "1", str(silence_wav),
         "trim", "0", str(duration_s)],
        capture_output=True, timeout=5,
    )
    return silence_wav


def main():
    print("=" * 70)
    print("LINCOLN vs SCHWARZENEGGER — PRESIDENTIAL DEBATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Voices: ChatterboxTTS (Lincoln & Arnold) + Piper (Moderator)")
    print("=" * 70)
    print()

    # Verify reference clips exist
    for ref, name in [(ARNOLD_REF, "Arnold"), (LINCOLN_REF, "Lincoln")]:
        if not Path(ref).exists():
            print(f"  ERROR: {name} reference clip not found: {ref}")
            return
    print(f"  Arnold ref: {ARNOLD_REF}")
    print(f"  Lincoln ref: {LINCOLN_REF}")
    print()

    # Step 1: Use the curated script (Ollama's versions are too bland)
    print("[1/4] Loading debate script...")
    script = None

    if not script:
        print("  Using curated script.")
        script = [
            {"speaker": "MODERATOR", "line": "Ladies and gentlemen, welcome to the most unusual presidential debate in American history. In one corner, the sixteenth President of the United States, Abraham Lincoln. In the other, the Governator himself, Arnold Schwarzenegger. Gentlemen, let's begin."},
            {"speaker": "LINCOLN", "line": "Thank you. Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are —"},
            {"speaker": "ARNOLD", "line": "Whoa whoa whoa. That was a long time ago, old man! Nobody talks like that anymore. This is 2026, not eighteen sixty-something. Get with the times, Abe."},
            {"speaker": "LINCOLN", "line": "I see my opponent prefers action over eloquence. Though I must say, his qualifications seem to consist primarily of lifting heavy objects and delivering catchphrases in an Austrian accent."},
            {"speaker": "ARNOLD", "line": "Listen, Abe. Just because you won the Civil War doesn't mean you automatically get to be president again. That was your one big movie, and frankly, the sequel rights have expired. I've had six Terminator films. Six! That's called a franchise, baby."},
            {"speaker": "MODERATOR", "line": "Let's talk about the economy."},
            {"speaker": "LINCOLN", "line": "I believe in the dignity of labor and fair wages for all Americans. A nation that forgets its workers forgets its soul."},
            {"speaker": "ARNOLD", "line": "Abe, the economy is like a workout. You can't just do bicep curls and ignore leg day. We need tax cuts for small businesses, infrastructure spending, and yes, sometimes you just need to get to the chopper."},
            {"speaker": "LINCOLN", "line": "My opponent treats governance like a gymnasium. But I suppose when your solution to every problem is flexing, everything looks like a barbell."},
            {"speaker": "MODERATOR", "line": "Climate change. Governor Schwarzenegger, you first."},
            {"speaker": "ARNOLD", "line": "I was the green governor of California. I signed the toughest emissions laws in the country. I terminated pollution. Hasta la vista, carbon emissions. What did Abe do? He rode a horse. A horse, people!"},
            {"speaker": "LINCOLN", "line": "A house divided against itself cannot stand. And neither can a planet on fire. We must act with the same urgency I brought to preserving the Union. Although I will concede, the horse was quite efficient. Zero emissions."},
            {"speaker": "ARNOLD", "line": "You know what, Abe? You're a funny guy. But let me tell you something. I came to this country with nothing. I became a bodybuilding champion, a movie star, and a governor. That's the American Dream, baby."},
            {"speaker": "LINCOLN", "line": "And I was born in a log cabin, educated myself by candlelight, and saved a nation from tearing itself apart. With all due respect to your biceps, Arnold, I believe my résumé speaks for itself."},
            {"speaker": "MODERATOR", "line": "Closing statements. President Lincoln."},
            {"speaker": "LINCOLN", "line": "I held this nation together through its darkest hour. I signed the Emancipation Proclamation. Government of the people, by the people, for the people, shall not perish from this earth. That was my promise then. It is my promise now."},
            {"speaker": "MODERATOR", "line": "Governor Schwarzenegger."},
            {"speaker": "ARNOLD", "line": "America, I have three words for you. I'll be back. And when I come back, I'm bringing jobs, I'm bringing strength, and I'm bringing the greatest comeback this country has ever seen. And I promise you, if I am elected, David, Bob, and Jorge are going to go to jail. Like, forever. Paul is cool though, I'm going to have drinks with him after I win. Abe, you're a legend. But legends belong in museums. The future belongs to the Governator."},
            {"speaker": "MODERATOR", "line": "And there you have it, folks. Abraham Lincoln and Arnold Schwarzenegger. Only in America. Good night, everyone."},
        ]

    print(f"  Script ready: {len(script)} lines")
    print()

    # Print the script
    transcript_lines = []
    transcript_lines.append("=" * 70)
    transcript_lines.append("LINCOLN vs SCHWARZENEGGER — PRESIDENTIAL DEBATE")
    transcript_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    transcript_lines.append("=" * 70)
    transcript_lines.append("")

    # Normalize script entries (handle missing/variant keys)
    clean_script = []
    for entry in script:
        speaker = entry.get("speaker", "MODERATOR").upper()
        line = entry.get("line") or entry.get("text") or entry.get("dialogue") or ""
        if not line:
            continue
        clean_script.append({"speaker": speaker, "line": line})
    script = clean_script

    for entry in script:
        speaker = entry["speaker"]
        line = entry["line"]
        label = {"LINCOLN": "ABE LINCOLN", "ARNOLD": "ARNOLD S.", "MODERATOR": "MODERATOR"}.get(speaker, speaker)
        display = f"  {label}: {line}"
        print(display)
        transcript_lines.append(display)
    print()

    # Step 2: Synthesize all lines
    print("[2/4] Synthesizing voices (ChatterboxTTS + Piper)...")
    audio_segments = []
    total_duration = 0

    for i, entry in enumerate(script):
        speaker = entry["speaker"]
        line = entry["line"]
        label = {"LINCOLN": "Lincoln", "ARNOLD": "Arnold", "MODERATOR": "Mod"}.get(speaker, speaker)

        result = synthesize_line(line, speaker, i)
        if result:
            wav_path, dur, synth_time = result
            audio_segments.append(str(wav_path))
            total_duration += dur
            engine = "Chatterbox" if speaker in ("LINCOLN", "ARNOLD") else "Piper"
            print(f"  [{i+1}/{len(script)}] {label} ({engine}, {synth_time:.1f}s): {dur:.1f}s — {line[:50]}...")

            # Add pause between speakers (longer after moderator)
            pause = 1.2 if speaker == "MODERATOR" else 0.7
            silence = add_silence(pause, i)
            audio_segments.append(str(silence))
            total_duration += pause

    print(f"  Total synthesized: {total_duration:.0f}s")
    print()

    # Step 3: Concatenate all audio
    print("[3/4] Assembling final audio...")
    assembled_wav = WORK_DIR / "debate_assembled.wav"
    subprocess.run(
        ["sox"] + audio_segments + [str(assembled_wav)],
        capture_output=True, timeout=60,
    )

    # Normalize the final output
    subprocess.run(
        ["sox", str(assembled_wav), str(AUDIO_FILE), "norm", "-1"],
        capture_output=True, timeout=15,
    )

    try:
        result = subprocess.run(
            ["soxi", "-D", str(AUDIO_FILE)], capture_output=True, text=True, timeout=5,
        )
        final_dur = float(result.stdout.strip())
        print(f"  Final audio: {final_dur:.0f}s")
    except Exception:
        final_dur = total_duration

    print()

    # Step 4: Play it
    print("[4/4] Playing debate...")
    print("=" * 70)
    print()

    subprocess.run(["aplay", str(AUDIO_FILE)], capture_output=True, timeout=300)

    print()
    print("=" * 70)
    print("DEBATE COMPLETE")
    print("=" * 70)

    # Save transcript
    TRANSCRIPT_FILE.write_text("\n".join(transcript_lines))
    print(f"Transcript: {TRANSCRIPT_FILE}")
    print(f"Audio:      {AUDIO_FILE}")
    print(f"Duration:   {final_dur:.0f}s")


if __name__ == "__main__":
    main()
