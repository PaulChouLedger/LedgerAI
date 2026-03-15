"""
Step 2: Synthesize training audio via ElevenLabs API.

Reads sentences.txt, sends each to ElevenLabs TTS with Olga's cloned voice,
saves WAV files to dataset/wavs/. Supports resuming from where it left off.

Prerequisites:
    pip install elevenlabs requests

Usage:
    export ELEVENLABS_API_KEY="your-key-here"
    export ELEVENLABS_VOICE_ID="iy0lEidUIpheWxyur2p8"  # Olga's clone

    python synthesize_elevenlabs.py [--input sentences.txt] [--output-dir dataset]
    python synthesize_elevenlabs.py --resume   # continue from last checkpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "iy0lEidUIpheWxyur2p8")
API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

# Emotional style presets — vary delivery across training samples so
# Piper learns the full range of Olga's expressiveness.
# Each preset tweaks ElevenLabs stability/style knobs.
STYLE_PRESETS = [
    # (name, stability, similarity_boost, style, weight)
    ("neutral",    0.75, 0.85, 0.0,  4),   # Baseline — most samples
    ("warm",       0.60, 0.85, 0.25, 2),   # Friendly, inviting
    ("calm",       0.85, 0.90, 0.0,  2),   # Steady, reassuring
    ("energetic",  0.50, 0.80, 0.40, 1),   # Upbeat, enthusiastic
    ("empathetic", 0.65, 0.85, 0.20, 1),   # Gentle, caring
    ("assertive",  0.70, 0.80, 0.30, 1),   # Confident, direct
]

def _pick_style():
    """Weighted random style selection."""
    import random as _r
    names, stabs, sims, styles, weights = zip(*STYLE_PRESETS)
    idx = _r.choices(range(len(STYLE_PRESETS)), weights=weights, k=1)[0]
    return {
        "stability": stabs[idx],
        "similarity_boost": sims[idx],
        "style": styles[idx],
        "use_speaker_boost": True,
    }, names[idx]

# Rate limiting
REQUESTS_PER_SECOND = 2  # ElevenLabs rate limit (conservative)
RETRY_DELAYS = [5, 15, 60, 300]  # Backoff on rate limit / errors

CHECKPOINT_EVERY = 50  # Save progress every N sentences


def synthesize_one(text: str, out_path: Path) -> tuple[bool, str]:
    """Synthesize a single sentence. Returns (success, style_name)."""
    voice_settings, style_name = _pick_style()
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": voice_settings,
    }

    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=30,
                stream=True,
            )

            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True, style_name

            if resp.status_code == 429:
                print(f"  Rate limited, waiting {delay}s...")
                time.sleep(delay)
                continue

            if resp.status_code == 401:
                print("ERROR: Invalid API key. Set ELEVENLABS_API_KEY.")
                sys.exit(1)

            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < len(RETRY_DELAYS) - 1:
                time.sleep(delay)

        except requests.Timeout:
            print(f"  Timeout, retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"  Error: {e}, retrying in {delay}s...")
            time.sleep(delay)

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Synthesize training audio via ElevenLabs")
    parser.add_argument("--input", type=str, default="sentences.txt")
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--start", type=int, default=0, help="Start from sentence index")
    parser.add_argument("--limit", type=int, default=0, help="Max sentences (0=all)")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set ELEVENLABS_API_KEY environment variable.")
        sys.exit(1)

    # Load sentences
    sentences = Path(args.input).read_text().strip().split("\n")
    print(f"Loaded {len(sentences)} sentences from {args.input}")

    # Setup output
    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint file
    checkpoint_path = out_dir / "synthesis_checkpoint.json"
    manifest_path = out_dir / "manifest.json"

    # Load or create manifest (maps filename -> text)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # Resume from checkpoint
    start_idx = args.start
    if args.resume and checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        start_idx = cp.get("next_index", 0)
        print(f"Resuming from sentence {start_idx}")

    end_idx = len(sentences) if args.limit == 0 else min(start_idx + args.limit, len(sentences))

    # Stats
    success = 0
    failed = 0
    skipped = 0
    total_chars = 0
    t_start = time.time()

    print(f"\nSynthesizing sentences {start_idx} to {end_idx - 1}...")
    print(f"Voice ID: {VOICE_ID}")
    print(f"Output: {wav_dir}/")
    print()

    for i in range(start_idx, end_idx):
        text = sentences[i].strip()
        if not text:
            continue

        fname = f"aura_{i:05d}.wav"
        wav_path = wav_dir / fname

        # Skip if already synthesized
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            skipped += 1
            continue

        # Synthesize with random emotional style
        ok, style = synthesize_one(text, wav_path)
        if ok:
            success += 1
            total_chars += len(text)
            manifest[fname] = text

            if success % 10 == 0:
                elapsed = time.time() - t_start
                rate = success / elapsed if elapsed > 0 else 0
                est_remain = (end_idx - i) / rate / 3600 if rate > 0 else 0
                print(f"  [{i}/{end_idx}] {success} done ({style}), {rate:.1f}/s, "
                      f"~{est_remain:.1f}h remaining")
        else:
            failed += 1
            print(f"  FAILED: sentence {i}: {text[:60]}...")

        # Checkpoint
        if (success + failed) % CHECKPOINT_EVERY == 0:
            checkpoint_path.write_text(json.dumps({"next_index": i + 1}))
            manifest_path.write_text(json.dumps(manifest, indent=2))

        # Rate limiting
        time.sleep(1.0 / REQUESTS_PER_SECOND)

    # Final save
    checkpoint_path.write_text(json.dumps({"next_index": end_idx}))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Synthesis complete!")
    print(f"  Success: {success}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total characters sent: {total_chars:,}")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Manifest: {manifest_path}")
    print(f"  WAVs: {wav_dir}/")

    # Estimate audio duration
    wav_files = list(wav_dir.glob("*.wav"))
    total_bytes = sum(f.stat().st_size for f in wav_files)
    # Rough estimate: 48kHz 16-bit mono = 96KB/s
    est_hours = total_bytes / (96000 * 3600)
    print(f"  Estimated audio: {est_hours:.1f} hours ({len(wav_files)} files)")


if __name__ == "__main__":
    main()
