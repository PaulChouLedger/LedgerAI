import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
from speaker import speak_llm_response, is_playing

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.5
VAD_THRESHOLD = 0.2
MIN_AUDIO_SAMPLES = 8000
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None

# === Detect correct mic index ===
def find_device_index():
    global DEVICE_INDEX
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            DEVICE_INDEX = i
            print(f"[Aura/listener] 🎧 Using input device: {device['name']} (index {i})")
            return
    raise RuntimeError("Microphone not found. Check DEVICE_NAME.")

# === Load Silero VAD ===
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

# === Transcribe with Whisper container ===
def transcribe(audio):
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, SAMPLE_RATE, format="WAV")
    wav_io.seek(0)
    try:
        response = requests.post(
            "http://localhost:5000/transcribe",
            files={"audio": ("speech.wav", wav_io, "audio/wav")},
            timeout=10
        )
        result = response.json()
        text = result["text"].get("text", "").strip() if isinstance(result["text"], dict) else result.get("text", "").strip()
        print(f"📝 Transcription: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        while True:
            if is_playing():
                print("[Listener] ⏸️ Pausing mic during playback")
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                print("[Listener] ▶️ Mic resumed after playback")

            buffer = []
            silence_start = None

            # === Wait for speech ===
            while True:
                if is_playing():
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]  # VAD only
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                print(f"[Debug] VAD prob: {vad_prob:.2f}")
                if vad_prob > VAD_THRESHOLD:
                    print(f"[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    buffer.append(audio_block)
                    break

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]  # VAD only
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                buffer.append(audio_block)

                if vad_prob < VAD_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print("\n⏹️ Speech ended. Processing...")
                        break
                else:
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]  # ✅ Use only channel 0 (beamformed)

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                continue

            text = transcribe(mono_mix)
            if text:
                speak_llm_response(text)

