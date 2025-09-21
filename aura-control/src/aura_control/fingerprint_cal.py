import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from fingerprint import compute_fingerprint

SAMPLE_RATE = 22050
PLAY_DURATION = 3.0  # seconds

print("🔊 Playing test audio and capturing mic...")

# === Load your known audio clip (used for TTS playback) ===
TTS_WAV_FILE = "test_clip.wav"  # Replace with real output
tts_audio, _ = sf.read(TTS_WAV_FILE, dtype="float32")

# === Play TTS audio while recording from mic (6 channels) ===
recording = np.zeros((int(SAMPLE_RATE * PLAY_DURATION), 6), dtype="float32")

def callback(indata, frames, time_info, status):
    recording[:frames] = indata

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=6, callback=callback)
with stream:
    sd.play(tts_audio, samplerate=SAMPLE_RATE)
    sd.wait()

# === Extract mic channel (e.g., channel 0) ===
mic_audio = recording[:, 0]  # Change if testing other channels

# === Compute cosine similarity ===
tts_fp = compute_fingerprint(tts_audio[:2048])
mic_fp = compute_fingerprint(mic_audio[:2048])

cos_sim = np.dot(tts_fp, mic_fp) / (np.linalg.norm(tts_fp) * np.linalg.norm(mic_fp) + 1e-8)
print(f"\n🎯 Cosine similarity (playback echo): {cos_sim:.4f}")
