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
VAD_CONFIDENCE_THRESHOLD = 0.2  # Lower threshold for better speech detection
MIC_GAIN = 1.5  # Minimal gain multiplier
MIN_SPEECH_DURATION = 0.25  # Minimum speech duration in seconds (allows "yes", "no", etc.)
VAD_RESET_THRESHOLD = 0.15  # Lower threshold for reset
# If VAD stays below this for too long, reset
VAD_RESET_COUNT = 20  # Base number of consecutive low VAD readings before reset
VAD_RESET_MULTIPLIER = 3  # Multiply by this to get actual reset threshold (60 readings)
# Removed consecutive reading requirement for immediate speech detection
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

# === Simple Audio Processing ===
def apply_minimal_gain(signal, gain_multiplier=1.5):
    """Apply minimal gain without complex processing"""
    return np.clip(signal * gain_multiplier, -1.0, 1.0)

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
            low_vad_count = 0  # Counter for consecutive low VAD readings

            # === Wait for speech (immediate trigger) ===
            while True:
                if is_playing():
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Track consecutive low VAD readings
                if vad_prob < VAD_RESET_THRESHOLD:
                    low_vad_count += 1
                    # Only reset VAD if we've been in low VAD for a very long time
                    if low_vad_count >= VAD_RESET_COUNT * VAD_RESET_MULTIPLIER:  # 60 consecutive low readings
                        print(f"[VAD] 🔄 Resetting VAD after {low_vad_count} consecutive low readings")
                        low_vad_count = 0
                        time.sleep(0.1)  # Brief pause to reset VAD state
                        continue
                else:
                    low_vad_count = 0  # Reset counter on higher VAD readings
                
                # Immediate speech detection - no consecutive reading requirement
                if vad_prob > VAD_CONFIDENCE_THRESHOLD:
                    print(f"[VAD] 🔊 High VAD detected (prob={vad_prob:.2f}) - starting recording")
                    set_transcribing(True)  # Notify GUI: transcription started
                    buffer.append(audio_block)
                    break
                
                # Reduce debug output for performance
                if vad_prob > 0.3:  # Only log significant VAD activity
                    print(f"[Debug] VAD prob: {vad_prob:.2f}")

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)  # Reset transcribing state if interrupted
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                buffer.append(audio_block)
                
                # Analyze voice frequency in real-time for organic border pulsation
                if vad_prob > VAD_CONFIDENCE_THRESHOLD:
                    analyze_voice_frequency(audio_block)

                if vad_prob < VAD_CONFIDENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print("\n⏹️ Speech ended. Processing...")
                        set_transcribing(False)  # Notify GUI: transcription ended
                        break
                else:
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)  # Reset transcribing state if interrupted
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Minimal audio processing - just basic gain
            mono_mix = apply_minimal_gain(mono_mix, MIC_GAIN)

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
