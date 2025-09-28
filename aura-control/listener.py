import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
import subprocess
from speaker import speak_llm_response, is_playing
from pydub import AudioSegment

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.2
VAD_THRESHOLD = 0.25  # Balanced threshold - filters noise but catches initial words
MIC_GAIN = 3.0  # Moderate amplification - avoid distortion
MIN_SPEECH_DURATION = 0.25  # Minimum speech duration in seconds to prevent noise triggers
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

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

# === Welcome prompt (pause mic before playing) ===
def play_welcome_prompt(stream):
    try:
        print("[Aura] 🔊 Playing welcome prompt...")
        stream.stop()
        subprocess.run(["aplay", WELCOME_AUDIO_PATH])
        time.sleep(0.25)
        stream.start()
        print("[Aura] 🎤 Mic resumed after welcome prompt")
    except Exception as e:
        print(f"[Aura] ❌ Failed to play welcome prompt: {e}")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # ✅ Play welcome.wav before entering listening loop
        play_welcome_prompt(stream)

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
                # Apply gain to increase microphone sensitivity
                audio_block = audio_block * MIC_GAIN
                channel_0 = audio_block[:, 0]
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                # Reduce debug output for performance
                if vad_prob > 0.1:  # Only log significant VAD activity
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
                # Apply gain to increase microphone sensitivity
                audio_block = audio_block * MIC_GAIN
                channel_0 = audio_block[:, 0]
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
            mono_mix = full_audio[:, 0]
            
            # Audio preprocessing for better transcription
            # Normalize volume
            max_val = np.max(np.abs(mono_mix))
            if max_val > 0:
                mono_mix = mono_mix / max_val * 0.95
            
            # Apply high-pass filter to remove low-frequency noise (temporarily disabled for debugging)
            # from scipy import signal
            # nyquist = SAMPLE_RATE / 2
            # high = 300 / nyquist  # Remove frequencies below 300Hz
            # b, a = signal.butter(4, high, btype='high')
            # mono_mix = signal.filtfilt(b, a, mono_mix)

            # Check audio duration
            audio_duration = len(mono_mix) / SAMPLE_RATE
            print(f"[Debug] Audio duration: {audio_duration:.2f}s, samples: {len(mono_mix)}")
            print(f"[Debug] Audio amplitude range: [{np.min(mono_mix):.3f}, {np.max(mono_mix):.3f}]")
            
            if audio_duration < MIN_SPEECH_DURATION:
                print(f"⚠️ Skipped: too short (duration: {audio_duration:.2f}s)")
                # Reset VAD state after failed detection to prevent noise loops
                time.sleep(0.1)  # Brief pause to let VAD reset
                continue

            text = transcribe(mono_mix)
            if not text:
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
