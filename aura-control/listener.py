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
from aura_gui import set_transcribing

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.20
VAD_START_THRESHOLD = 0.35
VAD_SILENCE_THRESHOLD = 0.10
MIN_AUDIO_SAMPLES = 4000

# Spectral noise reduction (using digital signature)
ENABLE_SPECTRAL_FILTERING = True
NOISE_PROFILE_PATH = os.path.expanduser("~/LedgerAI/data/noise_profile.npy")
NOISE_REDUCTION_STRENGTH = 0.7  # How much noise to subtract (0.0-1.0)

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Global noise profile (loaded from disk at startup)
noise_profile = None

# Debug
DEBUG_AUDIO_LEVELS = True
DEBUG_NOISE_REDUCTION = True

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

# === Spectral Filtering Functions ===

def load_noise_profile():
    """Load pre-recorded noise profile from disk"""
    global noise_profile
    
    if not os.path.exists(NOISE_PROFILE_PATH):
        print(f"[Audio] ⚠️  No noise profile found at {NOISE_PROFILE_PATH}")
        print(f"[Audio] ℹ️  Run: python3 scripts/record_noise_profile.py")
        print(f"[Audio] ℹ️  Proceeding without spectral filtering...")
        noise_profile = None
        return
    
    try:
        noise_profile = np.load(NOISE_PROFILE_PATH)
        print(f"[Audio] ✅ Loaded noise profile: {len(noise_profile)} frequency bins")
        
        # Show dominant noise frequencies
        freqs = np.fft.rfftfreq(len(noise_profile) * 2 - 2, 1/SAMPLE_RATE)
        top_indices = np.argsort(noise_profile)[-3:][::-1]
        print(f"[Audio] 🎵 Top noise frequencies: ", end="")
        for idx in top_indices:
            if idx < len(freqs):
                print(f"{freqs[idx]:.0f}Hz ", end="")
        print()
        
    except Exception as e:
        print(f"[Audio] ⚠️  Failed to load noise profile: {e}")
        noise_profile = None

def spectral_subtraction(audio, noise_profile, strength=0.7):
    """
    Spectral subtraction - removes noise using pre-recorded digital signature
    
    Args:
        audio: Audio signal to clean
        noise_profile: Pre-recorded noise spectrum
        strength: How much noise to subtract (0.0-1.0)
    
    Returns:
        Cleaned audio with noise removed
    """
    if noise_profile is None:
        return audio
    
    # FFT of input audio
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    
    # Resize noise profile to match audio length if needed
    if len(noise_profile) != len(magnitude):
        from scipy import interpolate
        x_old = np.linspace(0, 1, len(noise_profile))
        x_new = np.linspace(0, 1, len(magnitude))
        f = interpolate.interp1d(x_old, noise_profile, kind='linear', fill_value='extrapolate')
        noise_profile_resized = f(x_new)
    else:
        noise_profile_resized = noise_profile
    
    # Subtract noise profile (scaled by strength)
    cleaned_magnitude = magnitude - (strength * noise_profile_resized)
    
    # Ensure non-negative (keep at least 10% of original to avoid artifacts)
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
    
    # Reconstruct signal with cleaned magnitude and original phase
    cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = np.fft.irfft(cleaned_fft, n=len(audio))
    
    return cleaned_audio

def process_audio(audio):
    """
    Simple audio processing pipeline:
    1. Spectral filtering (removes noise using digital signature)
    
    That's it! No AGC, no noise gate, just clean noise removal.
    
    Args:
        audio: Raw audio to process
    
    Returns:
        Cleaned audio
    """
    if ENABLE_SPECTRAL_FILTERING and noise_profile is not None:
        audio = spectral_subtraction(audio, noise_profile, strength=NOISE_REDUCTION_STRENGTH)
    
    return audio

# === Simple frequency function for GUI border (placeholder) ===
def get_transcription_frequency():
    """Return default frequency for GUI border pulsation"""
    return 0.7

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
        
        try:
            from aura_gui import set_setup_complete, set_welcome_played, set_listening_ready
            set_setup_complete()
            set_welcome_played()
            set_listening_ready()
            print("[Aura] ✅ Setup complete, listener ready")
        except ImportError:
            pass
    except Exception as e:
        print(f"[Aura] ❌ Failed to play welcome prompt: {e}")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")
    
    # Load noise profile
    load_noise_profile()
    
    # Show configuration
    print("\n" + "="*70)
    print("[Audio] ✅ Simple pipeline: Spectral Filtering → Whisper")
    if noise_profile is not None:
        print(f"[Audio] 🔧 Spectral filtering: strength={NOISE_REDUCTION_STRENGTH}")
        print(f"[Audio] 🎯 Noise signature: {len(noise_profile)} frequency bins")
    else:
        print(f"[Audio] ⚠️  No noise profile - raw audio only")
    print(f"[Audio] 💡 No AGC, no noise gate, just clean filtering")
    print("="*70 + "\n")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # Play welcome.wav
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
                
                # Run VAD on raw audio
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Debug: Show audio levels
                if DEBUG_AUDIO_LEVELS:
                    rms = np.sqrt(np.mean(channel_0 ** 2))
                    print(f"[Debug] VAD: {vad_prob:.2f}, RMS: {rms:.4f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    set_transcribing(True)
                    buffer.append(audio_block)
                    break

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                buffer.append(audio_block)
                
                # Run VAD for silence detection
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n⏹️ Speech ended (VAD silence: {vad_prob:.2f} < {VAD_SILENCE_THRESHOLD}). Processing...")
                        set_transcribing(False)
                        break
                else:
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Debug: Show raw audio stats
            if DEBUG_NOISE_REDUCTION:
                raw_rms = np.sqrt(np.mean(mono_mix ** 2))
                raw_peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 RAW: RMS={raw_rms:.6f}, Peak={raw_peak:.4f}, Length={len(mono_mix)} samples")
            
            # Apply spectral filtering
            mono_mix = process_audio(mono_mix)
            
            # Debug: Show processed audio stats
            if DEBUG_NOISE_REDUCTION:
                clean_rms = np.sqrt(np.mean(mono_mix ** 2))
                clean_peak = np.max(np.abs(mono_mix))
                
                print(f"[Audio] ✅ CLEANED: RMS={clean_rms:.6f}, Peak={clean_peak:.4f}")
                if noise_profile is not None:
                    print(f"[Audio] 📈 CHANGE: RMS {raw_rms:.6f} → {clean_rms:.6f} ({clean_rms/raw_rms*100:.1f}% of original)")

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)
                continue

            text = transcribe(mono_mix)
            
            if not text:
                set_transcribing(False)
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
