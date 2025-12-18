#!/usr/bin/env python3
# scripts/generate_cached_prompts.py

import os
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# === Load API Key ===
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise EnvironmentError("ELEVENLABS_API_KEY not set in .env file.")

# === ElevenLabs Client Setup ===
client = ElevenLabs(api_key=api_key)
VOICE_ID = "iy0lEidUIpheWxyur2p8"
VOICE_MODEL = "eleven_monolingual_v1"

# === Voice settings ===
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.0,
    "style": 0.0,
    "use_speaker_boost": False
}

# === Audio format ===
PCM_SAMPLE_RATE = 22050  # Match speaker.py
PCM_FORMAT = "pcm_22050"  # PCM format (raw audio, not MP3)

# === Output directory ===
PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/prompts"))
os.makedirs(PROMPT_DIR, exist_ok=True)

# === Prompts to generate ===
prompts = {
    "audio3": "Hello, and thank you for listening.This is a natural speaking sample recorded for voice training. The goal is clarity, consistency, and a calm, conversational tone. Please speak clearly, at a comfortable pace, with natural pauses and steady volume throughout."
}

# === Generate and save prompts ===
for name, text in prompts.items():
    print(f"Generating '{name}' prompt...")

    try:
        # Request PCM format (raw audio) instead of MP3
        stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=VOICE_MODEL,
            text=text,
            voice_settings=VOICE_SETTINGS,
            output_format=PCM_FORMAT,
            optimize_streaming_latency=1
        )

        # Collect all audio chunks
        audio_bytes = b"".join(stream)
        
        # Convert PCM bytes to numpy array
        # PCM format is 16-bit signed integers (little-endian)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 in range [-1.0, 1.0]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Save as WAV using soundfile (already in requirements)
        wav_path = os.path.join(PROMPT_DIR, f"{name}.wav")
        sf.write(wav_path, audio_float, PCM_SAMPLE_RATE)
        
        # Get file info
        file_size = os.path.getsize(wav_path)
        duration = len(audio_float) / PCM_SAMPLE_RATE
        print(f"  ✅ Saved: {wav_path} ({duration:.2f}s, {file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Failed to generate '{name}': {e}")
        import traceback
        traceback.print_exc()

print()
print("✅ Prompts saved to:", PROMPT_DIR)
