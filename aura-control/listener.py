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
AUDIO_GAIN = 4.0  # No gain (testing native microphone level)
ENABLE_NOISE_REDUCTION = True  # Enable noise reduction
NOISE_REDUCTION_METHOD = "highpass"  # "highpass" or "spectral" - highpass removes <200Hz (fan noise)
HIGHPASS_CUTOFF = 200  # Hz - Fan noise < 200Hz, speech > 200Hz
NOISE_REDUCTION_STRENGTH = 0.6  # Spectral subtraction strength (only if method="spectral")

# Adaptive RMS-based noise gate
ENABLE_NOISE_GATE = True  # Enable RMS-based noise gate
NOISE_GATE_MODE = "adaptive"  # "fixed" or "adaptive"
NOISE_GATE_FIXED_THRESHOLD = 0.008  # Used if mode="fixed"
NOISE_GATE_RATIO = 3.0  # Adaptive: speech must be 3x louder than noise floor
NOISE_FLOOR_LEARNING_RATE = 0.1  # How fast to adapt to changing noise (0.1 = slow, 0.5 = fast)

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Adaptive noise floor tracking
noise_floor_rms = 0.003  # Initial estimate (will adapt quickly)
noise_floor_locked = False  # Lock after initial learning to prevent drift

# Debug: Show audio levels to help diagnose mic issues
DEBUG_AUDIO_LEVELS = True  # Set to False to disable

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# Noise profile path (pre-recorded using scripts/record_noise_profile.py)
NOISE_PROFILE_PATH = os.path.expanduser("~/LedgerAI/data/noise_profile.npy")

# Global noise profile (loaded from disk at startup)
noise_profile = None

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

def noise_gate(audio, vad_active=False, learning_phase=False):
    """
    Adaptive RMS-based noise gate - zeros out audio below dynamic threshold
    
    Adaptive Mode:
    - Learns noise floor during initial silence (first 30 frames after startup)
    - Locks threshold to prevent learning from speech
    - Threshold = noise_floor * NOISE_GATE_RATIO
    - Adapts to different environments (quiet room, noisy cafe, etc.)
    
    Fixed Mode:
    - Uses fixed threshold (calibrated for specific environment)
    
    Args:
        audio: Audio frame to process
        vad_active: Whether VAD detected speech
        learning_phase: Whether we're in initial learning phase (only learn then)
    """
    global noise_floor_rms, noise_floor_locked
    
    rms = np.sqrt(np.mean(audio ** 2))
    
    if NOISE_GATE_MODE == "adaptive":
        # Adaptive: Only update noise floor during learning phase AND when no speech
        if learning_phase and not vad_active and not noise_floor_locked:
            # Exponential moving average - slowly adapt to noise floor
            noise_floor_rms = (1 - NOISE_FLOOR_LEARNING_RATE) * noise_floor_rms + NOISE_FLOOR_LEARNING_RATE * rms
        
        # Threshold is a multiple of current noise floor
        threshold = noise_floor_rms * NOISE_GATE_RATIO
    else:
        # Fixed threshold mode
        threshold = NOISE_GATE_FIXED_THRESHOLD
    
    if rms < threshold:
        # Below threshold - it's noise, zero it out
        return np.zeros_like(audio)
    else:
        # Above threshold - it's likely speech, keep it
        return audio

def bandpass_filter(audio, lowcut=80, highcut=7000, order=5):
    """
    Apply bandpass filter to focus on human voice frequencies
    - Removes low-frequency fan rumble (< 80 Hz)
    - Removes high-frequency hiss (> 7000 Hz)
    - Note: highcut must be < Nyquist (8000 Hz for 16kHz sample rate)
    """
    nyquist = SAMPLE_RATE / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are in valid range (0 < Wn < 1)
    low = max(0.01, min(low, 0.99))
    high = max(0.01, min(high, 0.99))
    
    # Ensure low < high
    if low >= high:
        print(f"[Audio] ⚠️ Invalid filter frequencies: low={low}, high={high}")
        return audio
    
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Bandpass filter failed: {e}")
        return audio

def spectral_noise_subtraction(audio, noise_profile, strength=0.5, debug=False):
    """
    Subtract noise spectrum from audio using spectral subtraction
    - noise_profile: FFT of background noise (pre-recorded)
    - strength: how much noise to subtract (0.0-1.0)
    - debug: show detailed noise subtraction stats
    """
    if noise_profile is None:
        return audio
    
    try:
        # Compute FFT of signal
        fft_signal = np.fft.rfft(audio)
        magnitude = np.abs(fft_signal)
        phase = np.angle(fft_signal)
        
        # Match noise profile length to signal length
        if len(noise_profile) != len(magnitude):
            # Interpolate noise profile to match signal length
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(noise_profile))
            x_new = np.linspace(0, 1, len(magnitude))
            f = interpolate.interp1d(x_old, noise_profile, kind='linear', fill_value='extrapolate')
            noise_profile_matched = f(x_new)
        else:
            noise_profile_matched = noise_profile
        
        # Calculate noise energy before subtraction
        noise_energy = np.mean(noise_profile_matched)
        signal_energy = np.mean(magnitude)
        
        # Subtract noise profile from magnitude
        magnitude_clean = np.maximum(magnitude - strength * noise_profile_matched, 0)
        
        # Calculate reduction stats
        clean_energy = np.mean(magnitude_clean)
        noise_removed = signal_energy - clean_energy
        reduction_db = 20 * np.log10(signal_energy / (clean_energy + 1e-10))
        
        # Debug output (periodic)
        if debug and hasattr(spectral_noise_subtraction, '_debug_counter'):
            spectral_noise_subtraction._debug_counter += 1
            if spectral_noise_subtraction._debug_counter % 20 == 0:  # Every ~1 second
                print(f"[Noise] 🔇 Spectral subtraction:")
                print(f"        Signal energy: {signal_energy:.6f}")
                print(f"        Noise profile: {noise_energy:.6f}")
                print(f"        Clean energy: {clean_energy:.6f}")
                print(f"        Noise removed: {noise_removed:.6f} ({reduction_db:.1f} dB)")
                print(f"        Reduction: {(noise_removed/signal_energy)*100:.1f}%")
        elif debug:
            spectral_noise_subtraction._debug_counter = 0
        
        # Reconstruct signal with cleaned magnitude
        fft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = np.fft.irfft(fft_clean, n=len(audio))
        
        return audio_clean
    except Exception as e:
        print(f"[Audio] ⚠️ Spectral subtraction failed: {e}")
        return audio

def load_noise_profile():
    """
    Load pre-recorded noise profile from disk
    Created using scripts/record_noise_profile.py
    """
    global noise_profile
    
    if not ENABLE_NOISE_REDUCTION:
        print("[Audio] ℹ️  Noise reduction disabled")
        return
    
    if NOISE_REDUCTION_METHOD == "highpass":
        print("\n" + "="*70)
        print("[Audio] ✅ Using high-pass filter + adaptive RMS noise gate")
        print(f"[Audio] 🔧 High-pass cutoff: {HIGHPASS_CUTOFF} Hz (removes low-freq fan noise)")
        if ENABLE_NOISE_GATE:
            if NOISE_GATE_MODE == "adaptive":
                print(f"[Audio] 🔧 Noise gate: ADAPTIVE (learns noise floor)")
                print(f"[Audio] 💡 Threshold = noise_floor × {NOISE_GATE_RATIO}")
                print(f"[Audio] 💡 Adapts to any environment (quiet room, noisy cafe, etc.)")
            else:
                print(f"[Audio] 🔧 Noise gate: FIXED threshold = {NOISE_GATE_FIXED_THRESHOLD}")
        print("="*70 + "\n")
        noise_profile = None
        return
    
    # Spectral method requires noise profile
    try:
        if not os.path.exists(NOISE_PROFILE_PATH):
            print(f"[Audio] ⚠️  Noise profile not found: {NOISE_PROFILE_PATH}")
            print(f"[Audio] 💡 Run: python3 scripts/record_noise_profile.py")
            print(f"[Audio] ⚠️  Falling back to no noise reduction")
            noise_profile = None
            return
        
        # Load the noise profile
        noise_profile = np.load(NOISE_PROFILE_PATH)
        
        print("\n" + "="*70)
        print("[Audio] ✅ Noise profile loaded for spectral subtraction!")
        print(f"[Audio] 📁 Source: {NOISE_PROFILE_PATH}")
        print(f"[Audio] 📊 Frequency bins: {len(noise_profile)}")
        print(f"[Audio] 🎯 Spectral subtraction enabled (strength={NOISE_REDUCTION_STRENGTH})")
        print("[Audio] 🎤 Fan noise reduction active!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"[Audio] ❌ Failed to load noise profile: {e}")
        print(f"[Audio] ⚠️  Noise reduction disabled")
        noise_profile = None

def process_audio(audio, vad_active=False, learning_phase=False, debug=False):
    """
    Full audio processing pipeline:
    1. Noise reduction (highpass or spectral)
    2. Adaptive RMS-based noise gate (learns noise floor, removes low-energy artifacts)
    3. Gain normalization
    
    Args:
        audio: Audio frame to process
        vad_active: Whether VAD detected speech
        learning_phase: Whether in initial noise floor learning phase
        debug: Show debug output
    """
    if ENABLE_NOISE_REDUCTION:
        if NOISE_REDUCTION_METHOD == "highpass":
            # Step 1: Remove all frequencies below cutoff
            audio = highpass_filter(audio, cutoff=HIGHPASS_CUTOFF)
        elif NOISE_REDUCTION_METHOD == "spectral" and noise_profile is not None:
            # Step 1: Subtract learned noise pattern
            audio = spectral_noise_subtraction(audio, noise_profile, strength=NOISE_REDUCTION_STRENGTH, debug=debug)
    
    # Step 2: Adaptive RMS-based noise gate (learns environment, removes low-energy noise)
    if ENABLE_NOISE_GATE:
        audio = noise_gate(audio, vad_active=vad_active, learning_phase=learning_phase)
    
    # Step 3: Apply gain
    audio = np.clip(audio * AUDIO_GAIN, -1.0, 1.0)
    
    return audio

# === Simple Audio Gain (kept for backward compatibility) ===
def apply_gain(audio):
    """Apply simple gain to audio signal"""
    return np.clip(audio * AUDIO_GAIN, -1.0, 1.0)

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

# === Learn Noise Floor ===
def learn_noise_floor(stream, num_frames=30):
    """
    Learn the noise floor from initial silence
    Called once after welcome prompt to set adaptive noise gate threshold
    """
    global noise_floor_rms, noise_floor_locked
    
    if not ENABLE_NOISE_GATE or NOISE_GATE_MODE != "adaptive":
        return
    
    print(f"[Audio] 🔇 Learning noise floor from {num_frames} frames of silence...")
    
    rms_samples = []
    for i in range(num_frames):
        audio_block, _ = stream.read(FRAME_SIZE)
        channel_0 = audio_block[:, 0]
        
        # Apply high-pass filter first
        if ENABLE_NOISE_REDUCTION and NOISE_REDUCTION_METHOD == "highpass":
            channel_0 = highpass_filter(channel_0, cutoff=HIGHPASS_CUTOFF)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(channel_0 ** 2))
        rms_samples.append(rms)
    
    # Set noise floor to average RMS during silence
    noise_floor_rms = np.mean(rms_samples)
    threshold = noise_floor_rms * NOISE_GATE_RATIO
    
    # Lock the noise floor to prevent learning from speech
    noise_floor_locked = True
    
    print(f"[Audio] ✅ Noise floor learned: {noise_floor_rms:.6f}")
    print(f"[Audio] 🎯 Noise gate threshold locked: {threshold:.6f} (floor × {NOISE_GATE_RATIO})")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")
    
    # Load pre-recorded noise profile for noise reduction
    load_noise_profile()

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # Play welcome.wav before entering listening loop
        play_welcome_prompt(stream)
        
        # Learn noise floor from initial silence (before any speech)
        learn_noise_floor(stream, num_frames=30)

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
            
            # Apply full audio processing pipeline
            # Note: VAD was active during recording, noise floor already learned and locked
            mono_mix = process_audio(mono_mix, vad_active=True, learning_phase=False, debug=False)

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)  # Reset transcribing state
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
