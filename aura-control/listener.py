import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
import subprocess
# Removed scipy imports - using simpler filtering approach
from speaker import speak_llm_response, is_playing
from pydub import AudioSegment
from aura_gui import set_transcribing

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.2
VAD_CONFIDENCE_THRESHOLD = 0.45  # Higher threshold to avoid background noise (45%)
MIN_SPEECH_DURATION = 0.25  # Minimum speech duration in seconds (allows "yes", "no", etc.)
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# === Real-time voice frequency tracking ===
_current_voice_frequency = 0.5  # Default frequency (0.0 to 1.0 range)
_last_audio_rms = 0.0  # Track audio amplitude

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

# === Adaptive Audio Processing (from transcription_tuner) ===
TARGET_RMS = 0.05
MAX_GAIN = 2.0

def apply_adaptive_gain(signal):
    """Apply adaptive gain based on signal RMS - more robust than fixed gain"""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    signal = signal * gain
    return np.clip(signal, -1.0, 1.0)

def analyze_voice_frequency(audio_block):
    """
    Analyze audio block to extract voice characteristics for organic border pulsation
    Returns normalized frequency (0.0 to 1.0) based on audio amplitude and pitch
    """
    global _current_voice_frequency, _last_audio_rms
    
    try:
        # Get mono channel
        if audio_block.ndim > 1:
            mono = audio_block[:, 0]
        else:
            mono = audio_block
        
        # Calculate RMS (amplitude) for intensity
        rms = np.sqrt(np.mean(mono ** 2))
        
        # Smooth RMS changes for organic feel
        smoothing = 0.7  # 70% previous, 30% current
        _last_audio_rms = _last_audio_rms * smoothing + rms * (1 - smoothing)
        
        # Calculate zero-crossing rate for pitch estimation
        zero_crossings = np.sum(np.diff(np.sign(mono)) != 0)
        zcr_rate = zero_crossings / len(mono)
        
        # Combine RMS and ZCR for organic frequency
        # High RMS = louder = faster pulsation
        # High ZCR = higher pitch = faster pulsation
        amplitude_component = min(_last_audio_rms * 50, 1.0)  # Scale RMS to 0-1
        pitch_component = min(zcr_rate * 2.0, 1.0)  # Scale ZCR to 0-1
        
        # Weighted combination (60% amplitude, 40% pitch)
        frequency = amplitude_component * 0.6 + pitch_component * 0.4
        
        # Add organic variation (small random-like fluctuation)
        import random
        organic_noise = random.uniform(-0.05, 0.05)
        frequency = max(0.1, min(frequency + organic_noise, 1.0))
        
        # Smooth frequency changes
        _current_voice_frequency = _current_voice_frequency * 0.8 + frequency * 0.2
        
        return _current_voice_frequency
        
    except Exception as e:
        # Fallback to moderate frequency on error
        return 0.5

def get_transcription_frequency():
    """Get current voice frequency for GUI pulsation (0.0 to 1.0)"""
    return _current_voice_frequency


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
            recording_start = None

            # === Wait for speech (simpler approach like tuner) ===
            while True:
                if is_playing():
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                
                # Apply adaptive gain to improve VAD accuracy
                channel_0 = apply_adaptive_gain(channel_0)
                
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Simple speech detection like tuner - no complex reset logic
                if vad_prob > VAD_CONFIDENCE_THRESHOLD:
                    print(f"[VAD] 🔊 Speech detected (confidence={vad_prob:.2f})")
                    set_transcribing(True)  # Notify GUI: transcription started
                    buffer.append(audio_block)
                    recording_start = time.time()
                    break

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)  # Reset transcribing state if interrupted
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                
                # Apply adaptive gain for consistent VAD performance
                channel_0 = apply_adaptive_gain(channel_0)
                
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                buffer.append(audio_block)
                
                # Analyze voice frequency in real-time for organic border pulsation
                if vad_prob > VAD_CONFIDENCE_THRESHOLD:
                    analyze_voice_frequency(audio_block)
                    silence_start = None  # Reset silence timer
                else:
                    # Silence detected
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print("\n⏹️ Speech ended. Processing...")
                        set_transcribing(False)  # Notify GUI: transcription ended
                        break
                
                # Timeout safety check (like tuner)
                if recording_start and time.time() - recording_start > 10:
                    print("\n⏳ Recording timeout (10s). Processing...")
                    set_transcribing(False)
                    break
                
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)  # Reset transcribing state if interrupted
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Apply adaptive gain for final audio (like tuner)
            mono_mix = apply_adaptive_gain(mono_mix)

            # Check audio duration
            audio_duration = len(mono_mix) / SAMPLE_RATE
            if audio_duration < MIN_SPEECH_DURATION:
                print(f"⚠️ Skipped: too short (duration: {audio_duration:.2f}s)")
                set_transcribing(False)  # Reset transcribing state
                # Reset VAD state after failed detection to prevent noise loops
                time.sleep(0.1)  # Brief pause to let VAD reset
                continue

            text = transcribe(mono_mix)
            if not text:
                set_transcribing(False)  # Reset transcribing state
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
