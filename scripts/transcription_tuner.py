import os
import time
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
SILENCE_DURATION = 0.2
FINGERPRINT_MIN_SAMPLES = 2048

# VAD confidence threshold (range 0.0 to 1.0)
VAD_CONFIDENCE_THRESHOLD = 0.4

# Gain control
TARGET_RMS = 0.05
MAX_GAIN = 2.0

# === Load Models ===
print("[Tuner] ⏳ Loading Silero VAD...")
whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils
print("[Tuner] ✅ Silero VAD ready.")
print("[Tuner] ✅ Whisper model ready.")

# === Audio Processing ===
def highpass_filter(audio, cutoff=200):
    b, a = butter(1, cutoff / (0.5 * SAMPLE_RATE), btype='high')
    return lfilter(b, a, audio)

def apply_gain(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    signal = signal * gain
    print(f"[Tuner] 🌺 Applied gain multiplier: {gain:.2f}")
    return np.clip(signal, -1.0, 1.0)

def compute_centroid(frame, sr):
    spectrum = np.abs(rfft(frame))
    freqs = rfftfreq(len(frame), 1 / sr)
    return np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) != 0 else 0

def find_device_index():
    for i, device in enumerate(sd.query_devices()):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] > 0:
            print(f"[Tuner] 🎧 Using input device: {device['name']} (index {i})")
            return i
    raise RuntimeError("Microphone not found. Check DEVICE_NAME.")

# === Main Transcription Loop ===
def transcribe_from_microphone():
    device_index = find_device_index()
    print("[Tuner] 🎙️ Starting real-time transcription tuning...\n")

    while True:
        silence_start = None
        full_audio = []
        recording = False
        start_time = time.time()
        silence_detected_time = None  # Track when silence is first detected

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SIZE, device=device_index) as stream:
            while True:
                chunk, _ = stream.read(FRAME_SIZE)
                chunk = chunk.flatten()
                chunk = apply_gain(chunk)

                tensor = torch.from_numpy(chunk).unsqueeze(0)
                vad_prob = vad_model(tensor, SAMPLE_RATE).item()

                if vad_prob > VAD_CONFIDENCE_THRESHOLD:
                    print(f"[Tuner] 🟢 Speech Detected (confidence={vad_prob:.2f})")
                    if not recording:
                        print("[Tuner] 🎙 Transcribing...")
                        recording = True
                        full_audio = [chunk]
                        silence_start = None
                        silence_detected_time = None  # Reset silence detection time
                        start_time = time.time()
                    else:
                        full_audio.append(chunk)
                        silence_start = None
                        silence_detected_time = None  # Reset silence detection time
                else:
                    print(f"[Tuner] ⚪️ Silence Detected (confidence={vad_prob:.2f})")
                    if recording:
                        full_audio.append(chunk)
                        if silence_start is None:
                            silence_start = time.time()
                            silence_detected_time = time.time()  # Record when silence is first detected
                        elif time.time() - silence_start > SILENCE_DURATION:
                            print("[Tuner] 🔇 Silence threshold reached → Ending transcription.")
                            break

                if recording and time.time() - start_time > 10:
                    print("[Tuner] ⏳ Timeout exceeded.")
                    break

        if not full_audio:
            print("[Tuner] ❌ No speech captured.\n")
            continue

        # Start transcription timing
        transcription_start_time = time.time()
        start_timestamp = time.strftime('%H:%M:%S.%f')
        print(f"[Tuner] 🚀 TRANSCRIPTION PROCESSING START: {start_timestamp}")
        
        audio_data = np.concatenate(full_audio).astype(np.float16)
        t0 = time.time()
        segments, _ = whisper_model.transcribe(audio_data, language="en", beam_size=5)
        whisper_latency = time.time() - t0
        
        # Complete transcription timing
        transcription_end_time = time.time()
        end_timestamp = time.strftime('%H:%M:%S.%f')
        print(f"[Tuner] ✅ TRANSCRIPTION PROCESSING END: {end_timestamp}")
        
        text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        
        # Calculate comprehensive timing metrics
        if silence_detected_time:
            silence_to_transcription_time = transcription_start_time - silence_detected_time
            silence_to_completion_time = transcription_end_time - silence_detected_time
        else:
            silence_to_transcription_time = 0
            silence_to_completion_time = 0
        
        # Print comprehensive timing information
        print(f"[Tuner] ⏱️ TIMING METRICS:")
        print(f"  🔇 Silence detected to transcription start: {silence_to_transcription_time:.3f}s")
        print(f"  🎯 Silence detected to completion: {silence_to_completion_time:.3f}s")
        print(f"  🧠 Whisper processing latency: {whisper_latency:.3f}s")
        print(f"  📊 Total audio duration: {len(audio_data) / SAMPLE_RATE:.3f}s")
        
        if text:
            print(f"[Tuner] 📜 Transcribed Text: \"{text}\"")
        else:
            print("[Tuner] ❌ No text detected.")
        
        print()  # Empty line for readability

if __name__ == "__main__":
    transcribe_from_microphone()
