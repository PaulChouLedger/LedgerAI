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
from aura_gui import set_transcribing
from scipy import signal

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.20  # Time of continuous silence to stop - balance between responsiveness and cutoff prevention
VAD_START_THRESHOLD = 0.35  # Threshold to START detecting speech
VAD_SILENCE_THRESHOLD = 0.10  # Threshold for silence (closer to actual silence ~0.05, minimizes dead air to Whisper)
MIN_AUDIO_SAMPLES = 4000  # Reduced from 8000 to allow shorter utterances

# Audio processing
USE_AUTO_GAIN = True  # Enable automatic gain control
AGC_TARGET_RMS = 0.15  # Target RMS for Whisper (optimal speech recognition)
AGC_MAX_GAIN = 15.0  # Maximum gain to apply (prevents over-amplification)
ENABLE_NOISE_REDUCTION = True  # Enable noise reduction
HIGHPASS_CUTOFF = 20  # Hz - Fan noise < 50Hz, speech > 80Hz (preserves all voice frequencies)


DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Debug: Show audio levels to help diagnose mic issues
DEBUG_AUDIO_LEVELS = True  # Set to False to disable
DEBUG_NOISE_REDUCTION = True  # Show detailed noise reduction stats during transcription

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

# === Audio Processing Functions ===

def auto_gain_control(audio):
    """
    Automatic gain control - adapts gain based on input level
    Prevents clipping while maximizing signal strength
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:  # Avoid division by zero
        return audio, 1.0
    
    # Calculate required gain to reach target RMS
    required_gain = AGC_TARGET_RMS / rms
    
    # Limit gain to prevent over-amplification
    actual_gain = min(required_gain, AGC_MAX_GAIN)
    
    # Apply gain with clipping prevention
    audio = audio * actual_gain
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio, actual_gain

def highpass_filter(audio, cutoff=200, order=5):
    """
    Apply high-pass filter to remove low-frequency fan noise
    - Removes everything below cutoff frequency (typically 200 Hz)
    - Preserves speech frequencies (>200 Hz)
    - Simple and artifact-free solution for fan noise
    """
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure frequency is in valid range (0 < Wn < 1)
    normalized_cutoff = max(0.01, min(normalized_cutoff, 0.99))
    
    try:
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ High-pass filter failed: {e}")
        return audio

def process_audio(audio):
    """
    Simple audio processing pipeline:
    1. High-pass filter (removes fan noise < 200Hz)
    2. Automatic gain control (adapts to speech distance)
    
    Returns:
        processed_audio, applied_gain
    """
    # Step 1: High-pass filter to remove fan noise
    if ENABLE_NOISE_REDUCTION:
        audio = highpass_filter(audio, cutoff=HIGHPASS_CUTOFF)
    
    # Step 2: Automatic gain control
    if USE_AUTO_GAIN:
        audio, applied_gain = auto_gain_control(audio)
    else:
        audio = np.clip(audio, -1.0, 1.0)
        applied_gain = 1.0
    
    return audio, applied_gain


# === Simple frequency function for GUI border (placeholder) ===
def get_transcription_frequency():
    """Return default frequency for GUI border pulsation"""
    return 0.7  # Moderate pulsation speed

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
        
        # Signal GUI that setup is complete, welcome is done, and listener is ready
        try:
            from aura_gui import set_setup_complete, set_welcome_played, set_listening_ready
            set_setup_complete()   # Mark setup as complete
            set_welcome_played()   # Welcome prompt finished
            set_listening_ready()  # Listener is ready
            print("[Aura] ✅ Setup complete, listener ready - aura eye now static")
        except ImportError:
            pass
    except Exception as e:
        print(f"[Aura] ❌ Failed to play welcome prompt: {e}")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")
    
    # Audio processing configuration
    print("\n" + "="*70)
    print("[Audio] ✅ Audio processing pipeline: High-Pass → AGC")
    print(f"[Audio] 🔧 High-pass filter: {HIGHPASS_CUTOFF} Hz (preserves speech >80Hz)")
    print(f"[Audio] 🔧 Auto Gain Control: Target={AGC_TARGET_RMS}, Max={AGC_MAX_GAIN}x")
    print(f"[Audio] 💡 AGC adapts to speech distance (near/far)")
    print(f"[Audio] 💡 VAD handles speech detection")
    print("="*70 + "\n")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # Play welcome.wav before entering listening loop
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
                channel_0 = audio_block[:, 0]
                
                # Run VAD on raw audio (no processing)
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Debug: Show audio levels
                if DEBUG_AUDIO_LEVELS:
                    rms = np.sqrt(np.mean(channel_0 ** 2))
                    print(f"[Debug] VAD: {vad_prob:.2f}, RMS: {rms:.4f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    set_transcribing(True)  # Notify GUI: transcription started
                    buffer.append(audio_block)  # Store original audio
                    break

            # === Continue recording ===
            # Use VAD with LOWER threshold for silence detection
            # This allows continuous recording through quiet speech/distance variations
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)  # Reset transcribing state if interrupted
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                buffer.append(audio_block)  # Store original audio
                
                # Run VAD on raw audio for silence detection
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Check if VAD indicates silence (below LOWER threshold)
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n⏹️ Speech ended (VAD silence: {vad_prob:.2f} < {VAD_SILENCE_THRESHOLD}). Processing...")
                        set_transcribing(False)  # Notify GUI: transcription ended
                        break
                else:
                    # Reset silence timer if VAD detects any speech
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)  # Reset transcribing state if interrupted
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Debug: Show raw audio stats before processing
            if DEBUG_NOISE_REDUCTION:
                raw_rms = np.sqrt(np.mean(mono_mix ** 2))
                raw_peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 RAW: RMS={raw_rms:.6f}, Peak={raw_peak:.4f}, Length={len(mono_mix)} samples")
            
            # Apply full audio processing pipeline
            mono_mix, applied_gain = process_audio(mono_mix)
            
            # Debug: Show processed audio stats
            if DEBUG_NOISE_REDUCTION:
                clean_rms = np.sqrt(np.mean(mono_mix ** 2))
                clean_peak = np.max(np.abs(mono_mix))
                
                print(f"[Audio] ✅ PROCESSED: RMS={clean_rms:.6f}, Peak={clean_peak:.4f}, AGC={applied_gain:.2f}x")
                print(f"[Audio] 📈 AMPLIFICATION: {raw_rms:.6f} → {clean_rms:.6f} (×{clean_rms/raw_rms if raw_rms > 0 else 0:.2f})")

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)  # Reset transcribing state
                continue

            text = transcribe(mono_mix)
            
            # Debug: Show transcription result with audio quality
            if DEBUG_NOISE_REDUCTION and text:
                print(f"[Audio] 🎤 ✅ TRANSCRIBED: '{text}' (Clean RMS: {clean_rms:.6f}, Gain: {applied_gain:.2f}x)")
            elif DEBUG_NOISE_REDUCTION and not text:
                print(f"[Audio] 🎤 ❌ FAILED: No transcription (Clean RMS: {clean_rms:.6f}, Peak: {clean_peak:.4f})")
            
            if not text:
                set_transcribing(False)  # Reset transcribing state
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
