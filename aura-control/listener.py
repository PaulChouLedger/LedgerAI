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
AGC_TARGET_RMS = 0.15  # Target RMS for Whisper (balanced for near+far field)
AGC_MAX_GAIN = 40.0  # Maximum gain to apply (increased for 16m range)
AGC_SOFT_CLIP_THRESHOLD = 0.95  # Start soft-clipping above this level
ENABLE_NOISE_REDUCTION = True  # Enable noise reduction

# Filter options: "highpass", "bandpass", or "none"
FILTER_TYPE = "bandpass"  # bandpass filters speech range (80-3400 Hz)
HIGHPASS_CUTOFF = 80  # Hz - Low cutoff (balanced - preserves speech, removes rumble)
LOWPASS_CUTOFF = 3400  # Hz - High cutoff (removes hiss/noise above speech)

# Beam forming for far-field (uses all 6 microphones)
ENABLE_BEAM_FORMING = True  # Use all 6 mics for directional enhancement
BEAM_FORMING_MODE = "delay_sum"  # Options: "delay_sum", "average"


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

def beam_forming(audio_channels, mode="delay_sum"):
    """
    Multi-channel beam forming for far-field speech enhancement
    Uses all 6 microphones to create directional beam
    
    Args:
        audio_channels: (N, 6) array where N is samples, 6 is channels
        mode: "delay_sum" or "average"
    
    Returns:
        mono_audio: Enhanced mono signal
    """
    if mode == "average":
        # Simple averaging - sums all channels with equal weight
        # This provides ~3dB SNR improvement (√6 = 2.45x)
        mono = np.mean(audio_channels, axis=1)
        return mono
    
    elif mode == "delay_sum":
        # Delay-and-sum beam forming
        # Aligns signals from all mics toward target direction
        # Provides directional gain and noise rejection
        
        # For ReSpeaker 4 Mic Array, mics are typically in circular pattern
        # Without exact geometry, we use cross-correlation to align
        
        # Use channel 0 as reference
        reference = audio_channels[:, 0]
        aligned_channels = []
        
        for ch in range(6):
            channel = audio_channels[:, ch]
            
            # Find optimal delay using cross-correlation
            correlation = np.correlate(reference, channel, mode='full')
            delay = len(channel) - 1 - np.argmax(correlation)
            
            # Apply delay compensation (shift signal)
            if delay > 0:
                # Delay is positive: pad beginning
                shifted = np.pad(channel, (delay, 0), mode='constant')[:len(channel)]
            elif delay < 0:
                # Delay is negative: pad end
                shifted = np.pad(channel, (0, -delay), mode='constant')[-len(channel):]
            else:
                shifted = channel
            
            aligned_channels.append(shifted)
        
        # Sum aligned channels and normalize
        mono = np.sum(aligned_channels, axis=0) / 6.0
        return mono
    
    else:
        # Fallback: just use channel 0
        return audio_channels[:, 0]

def soft_clip(audio, threshold=0.85, max_peak=0.98):
    """
    Two-stage soft clipping with dynamic range compression
    Stage 1: Gradual compression above threshold (0.85)
    Stage 2: Hard limit at max_peak (0.98) to prevent distortion
    
    This preserves waveform shape while preventing peaks from destroying audio
    """
    # Stage 1: Soft compression for peaks above threshold
    mask = np.abs(audio) > threshold
    if np.any(mask):
        # Use tanh for smooth compression
        excess = audio[mask] - np.sign(audio[mask]) * threshold
        compressed = threshold + np.tanh(excess / (max_peak - threshold)) * (max_peak - threshold)
        audio[mask] = np.sign(audio[mask]) * compressed
    
    # Stage 2: Safety hard limit (should rarely trigger after soft compression)
    audio = np.clip(audio, -max_peak, max_peak)
    
    return audio

def auto_gain_control(audio):
    """
    Automatic gain control with two-stage soft clipping
    - Adapts gain based on input level (far-field support up to 40x)
    - Uses progressive soft clipping to preserve waveform shape
    - Prevents distortion while maximizing signal strength
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:  # Avoid division by zero
        return audio, 1.0
    
    # Calculate required gain to reach target RMS
    required_gain = AGC_TARGET_RMS / rms
    
    # Limit gain to maximum
    actual_gain = min(required_gain, AGC_MAX_GAIN)
    
    # Apply gain
    audio = audio * actual_gain
    
    # Apply two-stage soft clipping to preserve waveform
    # Stage 1: Soft compression starts at 0.85 (gradual)
    # Stage 2: Hard limit at 0.98 (prevents peak distortion)
    audio = soft_clip(audio, threshold=0.85, max_peak=0.98)
    
    return audio, actual_gain

def highpass_filter(audio, cutoff=200, order=5):
    """
    Apply high-pass filter to remove low-frequency fan noise
    - Removes everything below cutoff frequency
    - Preserves speech frequencies above cutoff
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

def lowpass_filter(audio, cutoff=3400, order=5):
    """
    Apply low-pass filter to remove high-frequency noise
    - Removes everything above cutoff frequency
    - Preserves speech frequencies below cutoff
    """
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure frequency is in valid range (0 < Wn < 1)
    normalized_cutoff = max(0.01, min(normalized_cutoff, 0.99))
    
    try:
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Low-pass filter failed: {e}")
        return audio

def bandpass_filter(audio, low_cutoff=80, high_cutoff=3400, order=5):
    """
    Apply band-pass filter to isolate speech frequencies
    - Removes frequencies below low_cutoff (rumble/fan noise)
    - Removes frequencies above high_cutoff (hiss/high-freq noise)
    - Preserves speech band (typically 80-3400 Hz)
    """
    nyquist = SAMPLE_RATE / 2
    low_norm = low_cutoff / nyquist
    high_norm = high_cutoff / nyquist
    
    # Ensure frequencies are in valid range
    low_norm = max(0.01, min(low_norm, 0.98))
    high_norm = max(0.02, min(high_norm, 0.99))
    
    # Ensure low < high
    if low_norm >= high_norm:
        print(f"[Audio] ⚠️ Invalid bandpass range: {low_cutoff}-{high_cutoff} Hz")
        return audio
    
    try:
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Band-pass filter failed: {e}")
        return audio

def process_audio(audio):
    """
    Audio processing pipeline:
    1. Frequency filtering (highpass/bandpass/none) - removes noise bands
    2. Automatic gain control (adapts to speech distance) - amplifies signal
    3. Soft clipping (prevents distortion)
    
    Returns:
        processed_audio, applied_gain
    """
    # Step 1: Frequency filtering
    if ENABLE_NOISE_REDUCTION:
        if FILTER_TYPE == "bandpass":
            audio = bandpass_filter(audio, low_cutoff=HIGHPASS_CUTOFF, high_cutoff=LOWPASS_CUTOFF)
        elif FILTER_TYPE == "highpass":
            audio = highpass_filter(audio, cutoff=HIGHPASS_CUTOFF)
        # "none" or other values = no filtering
    
    # Step 2: Automatic gain control with soft clipping
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
    if ENABLE_BEAM_FORMING:
        print(f"[Audio] ✅ Processing: 6-Mic Beam Forming ({BEAM_FORMING_MODE}) → Band-Pass → AGC → Soft Clip")
        print(f"[Audio] 🔧 Beam forming: {BEAM_FORMING_MODE} (uses all 6 microphones)")
    else:
        print(f"[Audio] ✅ Processing: Single Mic → Band-Pass → AGC → Soft Clip")
    
    if FILTER_TYPE == "bandpass":
        print(f"[Audio] 🔧 Band-pass filter: {HIGHPASS_CUTOFF}-{LOWPASS_CUTOFF} Hz (speech band)")
    elif FILTER_TYPE == "highpass":
        print(f"[Audio] 🔧 High-pass filter: {HIGHPASS_CUTOFF} Hz")
    
    print(f"[Audio] 🔧 Auto Gain Control: Target={AGC_TARGET_RMS}, Max={AGC_MAX_GAIN}x")
    print(f"[Audio] 🔧 Soft Clipping: 0.85→0.98 (preserves waveform, prevents distortion)")
    print(f"[Audio] 💡 Optimized for far-field speech (up to 16 meters)")
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
            
            # Apply beam forming to combine all 6 microphones
            if ENABLE_BEAM_FORMING:
                mono_mix = beam_forming(full_audio, mode=BEAM_FORMING_MODE)
            else:
                mono_mix = full_audio[:, 0]  # Fallback: use channel 0 only
            
            # Debug: Show raw audio stats before processing
            if DEBUG_NOISE_REDUCTION:
                raw_rms = np.sqrt(np.mean(mono_mix ** 2))
                raw_peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 RAW (6-mic beam formed): RMS={raw_rms:.6f}, Peak={raw_peak:.4f}, Length={len(mono_mix)} samples")
            
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
