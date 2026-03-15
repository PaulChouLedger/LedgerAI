"""
Step 3: Build a Piper-compatible training dataset from ElevenLabs WAVs.

Converts the synthesized audio into the format Piper expects:
  - 22050 Hz mono WAV files (Piper's default sample rate)
  - LJSpeech-style metadata CSV
  - Config JSON for single-speaker training

Prerequisites:
    pip install librosa soundfile numpy

Usage:
    python build_piper_dataset.py [--input-dir dataset] [--output-dir piper_dataset]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# Piper training expects 22050 Hz mono
TARGET_SR = 22050


def convert_wav(src: Path, dst: Path):
    """Convert WAV to 22050 Hz mono using librosa."""
    import librosa
    import soundfile as sf

    audio, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # Trim silence from start and end
    audio, _ = librosa.effects.trim(audio, top_db=30)

    sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR  # duration in seconds


def main():
    parser = argparse.ArgumentParser(description="Build Piper training dataset")
    parser.add_argument("--input-dir", type=str, default="dataset",
                        help="Directory with wavs/ and manifest.json")
    parser.add_argument("--output-dir", type=str, default="piper_dataset",
                        help="Output directory for Piper-formatted dataset")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Skip clips shorter than N seconds")
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Skip clips longer than N seconds")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    wav_in = input_dir / "wavs"
    wav_out = output_dir / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run synthesize_elevenlabs.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"Loaded manifest with {len(manifest)} entries")

    # Process each WAV
    metadata = []
    total_duration = 0.0
    skipped_short = 0
    skipped_long = 0
    skipped_missing = 0
    processed = 0

    for fname, text in sorted(manifest.items()):
        src = wav_in / fname
        if not src.exists():
            skipped_missing += 1
            continue

        dst = wav_out / fname
        try:
            duration = convert_wav(src, dst)
        except Exception as e:
            print(f"  Error converting {fname}: {e}")
            continue

        if duration < args.min_duration:
            skipped_short += 1
            dst.unlink(missing_ok=True)
            continue

        if duration > args.max_duration:
            skipped_long += 1
            dst.unlink(missing_ok=True)
            continue

        # Piper metadata format: file_id|text|text (LJSpeech style)
        file_id = fname.replace(".wav", "")
        metadata.append((file_id, text))
        total_duration += duration
        processed += 1

        if processed % 500 == 0:
            print(f"  Processed {processed} files ({total_duration/3600:.1f}h)")

    # Write metadata CSV (LJSpeech format: id|text|text)
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for file_id, text in metadata:
            writer.writerow([file_id, text, text])

    # Write Piper training config
    config = {
        "dataset": str(output_dir.resolve()),
        "audio": {
            "sample_rate": TARGET_SR,
        },
        "espeak": {
            "voice": "en-us",
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
        "training": {
            "seed": 42,
            "epochs": 10000,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "fp16_run": True,
            "num_workers": 4,
        },
        "num_speakers": 1,
        "speaker_id_map": {},
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Split into train/val (95/5)
    import random
    random.seed(42)
    random.shuffle(metadata)
    split = int(len(metadata) * 0.95)
    train = metadata[:split]
    val = metadata[split:]

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    with open(train_path, "w") as f:
        for file_id, text in train:
            f.write(f"{file_id}|{text}|{text}\n")
    with open(val_path, "w") as f:
        for file_id, text in val:
            f.write(f"{file_id}|{text}|{text}\n")

    hours = total_duration / 3600
    print(f"\n{'='*60}")
    print(f"Piper dataset ready!")
    print(f"  Files: {processed}")
    print(f"  Duration: {hours:.1f} hours ({total_duration:.0f}s)")
    print(f"  Skipped (short): {skipped_short}")
    print(f"  Skipped (long): {skipped_long}")
    print(f"  Skipped (missing): {skipped_missing}")
    print(f"  Train: {len(train)} | Val: {len(val)}")
    print(f"  Metadata: {meta_path}")
    print(f"  Config: {config_path}")
    print(f"  WAVs: {wav_out}/")
    print(f"\nNext step: train Piper with train_piper.sh")


if __name__ == "__main__":
    main()
