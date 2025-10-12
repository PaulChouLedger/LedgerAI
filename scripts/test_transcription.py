"""
Pure Transcription Testing Script with Advanced Audio Feature Analysis

This script helps find optimal VAD thresholds by displaying multiple audio features
commonly used in commercial voice assistants (Alexa, Siri, Google Assistant).

AUDIO FEATURES EXPLAINED:
========================

1. RMS Energy (0.0-1.0) ⭐ CRITICAL DISCRIMINATOR
   - Root Mean Square energy level
   - Speech: 0.05-0.20 (varies with distance/volume)
   - Low-level noise: 0.015-0.030
   - Silence: < 0.015
   - **Best discriminator between speech and noise bursts**

1b. Peak Amplitude (0.0-1.0) ⭐ CRITICAL DISCRIMINATOR
   - Maximum amplitude in signal
   - Speech: 0.20-1.0 (varies with volume)
   - Low-level noise: 0.05-0.15
   - **Works together with RMS to reject weak noise**

2. Zero Crossing Rate (ZCR) (0.0-0.3)
   - How often signal crosses zero amplitude
   - Vowels: 0.02-0.05 (low)
   - Fricatives (s, sh, f): 0.08-0.15 (high)
   - White noise/hiss: 0.15+ (very high)
   - Fan noise: 0.03-0.08

3. Spectral Centroid (Hz)
   - "Center of mass" of frequency spectrum
   - Male speech: 1000-1500 Hz
   - Female speech: 1500-2500 Hz
   - Fan/HVAC: 200-800 Hz
   - Hiss: 3000+ Hz

4. Spectral Flatness (0.0-1.0)
   - Measures how "tonal" (speech) vs "noisy" (random)
   - Speech: 0.01-0.1 (very tonal)
   - Music: 0.1-0.3
   - White noise: 0.8-1.0 (very flat)
   - **KEY DISCRIMINATOR for speech vs noise**

5. Speech Band Ratio (0.0-1.0)
   - Energy in 300-3400 Hz (telephone frequency range)
   - Speech: 0.5-0.8 (most energy here)
   - Fan noise: 0.2-0.4
   - **IMPORTANT for detecting speech**

6. High Frequency Ratio (0.0-1.0)
   - Energy above 4000 Hz
   - Speech: 0.05-0.15
   - Hiss/fans: 0.2+

7. Low Frequency Ratio (0.0-1.0)
   - Energy below 100 Hz (rumble)
   - Speech: 0.05-0.15
   - HVAC/fans: 0.3+

8. Duration (seconds)
   - Meaningful speech: 0.8+ seconds
   - Short words: 0.5-0.8 seconds
   - Noise bursts: < 0.6 seconds
   - **Simple but effective filter**

TYPICAL PATTERNS:
=================
✅ SPEECH: RMS >0.035 + Peak >0.15 + Duration >0.4s + Moderate flatness (<0.55)
⚠️  LOW-LEVEL NOISE: RMS 0.015-0.030 + Peak 0.05-0.15 + Triggers VAD but not real speech
❌ NOISE BURST: RMS <0.03 + Peak <0.15 + High flatness (>0.55)

KEY INSIGHT: RMS and Peak are the most reliable discriminators.
Your noise has RMS ~0.02 and Peak ~0.10, while speech has RMS ~0.10 and Peak ~0.80+
"""

import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
from scipy import signal
from scipy.fft import rfft, rfftfreq

# === Config ===
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.3  # 300ms of silence before stopping
VAD_START_THRESHOLD = 0.25  # Higher = less sensitive to fan noise
VAD_SILENCE_THRESHOLD = 0.15  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

# === Advanced Multi-Feature Speech Detection (OPTIONAL) ===
# Set ENABLE_ADVANCED_FILTER = True to enable secondary checks beyond VAD
# RECOMMENDED: Enable when using beamforming for best results!
ENABLE_ADVANCED_FILTER = True  # Toggle this to test

# Thresholds based on your ACTUAL speech patterns:
# Updated after comparing real speech vs noise bursts
SPEECH_ZCR_MAX = 0.40           # Reject if ZCR > this (speech can be 0.15-0.35)
SPEECH_FLATNESS_MAX = 0.55      # Reject if too "flat" (speech can be 0.15-0.45)
SPEECH_CENTROID_MIN = 300       # Hz - reject if too low (rumble/fan)
SPEECH_CENTROID_MAX = 3000      # Hz - reject if too high (hiss) - raised for fricatives
SPEECH_BAND_MIN = 0.30          # Reject if insufficient energy in speech band
SPEECH_DURATION_MIN = 0.4       # Seconds - reject if too short (noise bursts)

# CRITICAL: Energy thresholds (most reliable for your noise pattern)
SPEECH_RMS_MIN = 0.035          # Reject if RMS < this (noise is 0.018-0.026, speech is 0.097)
SPEECH_PEAK_MIN = 0.15          # Reject if peak < this (noise is 0.08-0.12, speech is 0.96)

# === VAD Thresholds (can be lowered with beamforming) ===
# With beamforming enabled, audio is cleaner so you can use lower thresholds for better responsiveness

# BARE-BONES: No software processing - testing hardware optimization only

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None

# Freeze detection removed - VAD returning 0.00 with ambient noise RMS is normal behavior

# === Stats Tracking ===
transcription_count = 0
total_audio_duration = 0.0
start_time = time.time()

# === Find Device ===
def find_device_index():
    global DEVICE_INDEX
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower():
            DEVICE_INDEX = i
            print(f"[Listener] 🎧 Found: {device['name']} (index {i})")
            return 6  # Always use 6 channels
    raise RuntimeError("Microphone not found")

# === Audio Feature Extraction ===
def calculate_audio_features(audio_chunk, sample_rate=SAMPLE_RATE):
    """
    Calculate multiple audio features used in commercial voice assistants
    Returns: dict with all features
    """
    features = {}
    
    # 1. RMS Energy (already used, but included for completeness)
    features['rms'] = np.sqrt(np.mean(audio_chunk ** 2))
    
    # Peak amplitude (critical for distinguishing speech from low-level noise)
    features['peak'] = np.max(np.abs(audio_chunk))
    
    # 2. Zero Crossing Rate - speech has characteristic ZCR
    # High ZCR = fricatives/noise, Low ZCR = vowels, Very high = fan/hiss
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
    features['zcr'] = zero_crossings / len(audio_chunk)
    
    # 3. Spectral features (frequency domain)
    # Compute FFT
    fft_vals = rfft(audio_chunk)
    fft_freq = rfftfreq(len(audio_chunk), 1/sample_rate)
    magnitude = np.abs(fft_vals)
    
    # Spectral Centroid - "center of mass" of spectrum
    # Speech typically 1000-2000 Hz, noise varies
    if np.sum(magnitude) > 0:
        features['spectral_centroid'] = np.sum(fft_freq * magnitude) / np.sum(magnitude)
    else:
        features['spectral_centroid'] = 0
    
    # Spectral Flatness - how "tonal" vs "noisy"
    # Speech is tonal (low ~0.01-0.1), white noise is flat (high ~1.0)
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude)
    if arithmetic_mean > 0:
        features['spectral_flatness'] = geometric_mean / arithmetic_mean
    else:
        features['spectral_flatness'] = 0
    
    # 4. Speech Band Energy (300-3400 Hz - telephone quality range)
    # Human speech fundamental + harmonics are here
    speech_band_mask = (fft_freq >= 300) & (fft_freq <= 3400)
    speech_band_energy = np.sum(magnitude[speech_band_mask] ** 2)
    total_energy = np.sum(magnitude ** 2)
    if total_energy > 0:
        features['speech_band_ratio'] = speech_band_energy / total_energy
    else:
        features['speech_band_ratio'] = 0
    
    # 5. High Frequency Ratio (4000+ Hz)
    # Fan noise / hiss often has more high frequency content
    high_freq_mask = fft_freq >= 4000
    high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
    if total_energy > 0:
        features['high_freq_ratio'] = high_freq_energy / total_energy
    else:
        features['high_freq_ratio'] = 0
    
    # 6. Low Frequency Rumble (0-100 Hz)
    # HVAC, fans, etc. Can indicate noise
    low_freq_mask = fft_freq <= 100
    low_freq_energy = np.sum(magnitude[low_freq_mask] ** 2)
    if total_energy > 0:
        features['low_freq_ratio'] = low_freq_energy / total_energy
    else:
        features['low_freq_ratio'] = 0
    
    return features

def is_likely_speech(features, duration=None):
    """
    Apply advanced multi-feature analysis to distinguish speech from noise.
    
    IMPORTANT: Thresholds tuned for REAL speech patterns observed in testing.
    Initial thresholds were too strict and rejected legitimate speech.
    
    Speech characteristics vary widely:
    - Vowels: Low ZCR (0.05-0.12), Low SpCent (800-1500Hz)
    - Fricatives (s, sh, f): High ZCR (0.20-0.35), High SpCent (2000-3000Hz)
    - Mixed speech: ZCR 0.10-0.35, SpCent 1000-2500Hz
    
    Args:
        features: Dict of audio features
        duration: Audio duration in seconds (optional)
    
    Returns: (is_speech: bool, reason: str)
    """
    reasons = []
    
    # Check Energy Levels FIRST (most reliable for your noise pattern)
    if features['rms'] < SPEECH_RMS_MIN:
        reasons.append(f"RMS too low ({features['rms']:.4f} < {SPEECH_RMS_MIN})")
    
    if features['peak'] < SPEECH_PEAK_MIN:
        reasons.append(f"Peak too low ({features['peak']:.4f} < {SPEECH_PEAK_MIN})")
    
    # Check Duration (if provided) - quick rejection for short noise bursts
    if duration is not None and duration < SPEECH_DURATION_MIN:
        reasons.append(f"Too short ({duration:.2f}s < {SPEECH_DURATION_MIN}s)")
    
    # Check Zero Crossing Rate
    if features['zcr'] > SPEECH_ZCR_MAX:
        reasons.append(f"ZCR too high ({features['zcr']:.3f} > {SPEECH_ZCR_MAX})")
    
    # Check Spectral Flatness (second most reliable)
    if features['spectral_flatness'] > SPEECH_FLATNESS_MAX:
        reasons.append(f"Too flat/noisy ({features['spectral_flatness']:.3f} > {SPEECH_FLATNESS_MAX})")
    
    # Check Spectral Centroid (frequency range)
    if features['spectral_centroid'] < SPEECH_CENTROID_MIN:
        reasons.append(f"SpCent too low ({features['spectral_centroid']:.0f}Hz < {SPEECH_CENTROID_MIN}Hz)")
    elif features['spectral_centroid'] > SPEECH_CENTROID_MAX:
        reasons.append(f"SpCent too high ({features['spectral_centroid']:.0f}Hz > {SPEECH_CENTROID_MAX}Hz)")
    
    # Check Speech Band Energy
    if features['speech_band_ratio'] < SPEECH_BAND_MIN:
        reasons.append(f"Low speech band energy ({features['speech_band_ratio']:.2f} < {SPEECH_BAND_MIN})")
    
    is_speech = len(reasons) == 0
    reason = " | ".join(reasons) if reasons else "All checks passed"
    
    return is_speech, reason

# === Load VAD ===
print("[VAD] 🔄 Loading Silero VAD model...")
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
print("[VAD] ✅ VAD model loaded")

# === Load Hardware Config ===
def load_respeaker_config():
    """Load saved ReSpeaker configuration (no permissions needed)"""
    config_file = os.path.expanduser("~/LedgerAI/data/respeaker_config.json")
    try:
        with open(config_file, 'r') as f:
            import json
            state = json.load(f)
            return state
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[Config] ⚠️ Could not load config: {e}")
        return {}

def display_hardware_config(state):
    """Display current hardware configuration"""
    if not state:
        print("\n[Hardware] ⚠️  No saved configuration found")
        print("[Hardware] 💡 Run: sudo python3 scripts/tune_respeaker.py [profile]\n")
        return
    
    preset = state.get('preset', 'unknown')
    config = state.get('config', {})
    
    print("\n" + "="*70)
    print(f"  📊 HARDWARE CONFIGURATION: {preset.upper()}")
    print("="*70)
    
    # AGC
    if config.get('AGCONOFF', 0) == 1:
        print(f"  AGC:                    ✅ ENABLED")
        print(f"    Target Level:         {config.get('AGCDESIREDLEVEL', 0):.2f} RMS")
        print(f"    Max Gain:             {config.get('AGCMAXGAIN', 0):.0f} dB")
    else:
        print(f"  AGC:                    ❌ DISABLED")
    
    # High-pass Filter
    hpf_labels = ["OFF", "70 Hz", "125 Hz", "180 Hz"]
    hpf_val = config.get('HPFONOFF', 0)
    hpf_label = hpf_labels[hpf_val] if hpf_val < len(hpf_labels) else str(hpf_val)
    print(f"  High-Pass Filter:       {hpf_label}")
    
    # Noise Suppression
    if config.get('STATNOISEONOFF_SR', 0) == 1:
        gamma = config.get('GAMMA_NS_SR', 1.0)
        print(f"  Stationary Noise Supp:  ✅ ENABLED (gamma={gamma:.1f})")
    else:
        print(f"  Stationary Noise Supp:  ❌ DISABLED")
    
    print("="*70 + "\n")

# === Transcribe ===
def transcribe(audio):
    """Send raw audio to Whisper with detailed audio feature analysis"""
    global transcription_count, total_audio_duration
    
    duration = len(audio) / SAMPLE_RATE
    
    # Calculate comprehensive audio features
    features = calculate_audio_features(audio)
    peak = np.max(np.abs(audio))
    
    # Display audio characteristics
    print(f"\n{'='*70}")
    print(f"[Audio] Duration={duration:.2f}s | Peak={peak:.4f}")
    print(f"[Audio] RMS Energy:         {features['rms']:.6f}")
    print(f"[Audio] Zero Crossing Rate: {features['zcr']:.4f}")
    print(f"[Audio] Spectral Centroid:  {features['spectral_centroid']:.0f} Hz")
    print(f"[Audio] Spectral Flatness:  {features['spectral_flatness']:.4f} (speech ~0.01-0.1, noise ~0.5-1.0)")
    print(f"[Audio] Speech Band Ratio:  {features['speech_band_ratio']:.3f} (300-3400Hz)")
    print(f"[Audio] High Freq Ratio:    {features['high_freq_ratio']:.3f} (4000+ Hz, noise if high)")
    print(f"[Audio] Low Freq Ratio:     {features['low_freq_ratio']:.3f} (0-100 Hz, rumble if high)")
    
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, SAMPLE_RATE, format="WAV")
    wav_io.seek(0)
    
    try:
        transcribe_start = time.time()
        response = requests.post(
            "http://localhost:5000/transcribe",
            files={"audio": ("speech.wav", wav_io, "audio/wav")},
            timeout=10
        )
        transcribe_time = time.time() - transcribe_start
        
        result = response.json()
        text = result["text"].get("text", "").strip() if isinstance(result["text"], dict) else result.get("text", "").strip()
        
        transcription_count += 1
        total_audio_duration += duration
        
        print(f"[Whisper] ⏱️  Transcription time: {transcribe_time:.3f}s")
        print(f"[Whisper] 📝 Text: '{text}'")
        
        # Apply advanced filter analysis with duration (even if not actively filtering)
        is_speech_result, reason = is_likely_speech(features, duration)
        classification = "✅ SPEECH" if is_speech_result else "⚠️  POSSIBLE NOISE"
        print(f"[Analysis] Classification: {classification}")
        if not is_speech_result:
            print(f"[Analysis] Rejection reason: {reason}")
        
        # Show filter status
        filter_status = "ENABLED" if ENABLE_ADVANCED_FILTER else "DISABLED"
        print(f"[Analysis] Advanced filter: {filter_status}")
        
        print(f"[Stats] 🔢 Transcriptions: {transcription_count} | Total audio: {total_audio_duration:.1f}s")
        print(f"{'='*70}")
        
        return text
    except Exception as e:
        print(f"[Whisper] ❌ {e}")
        print(f"{'='*70}")
        return ""

def warmup_whisper():
    """Warm up Whisper model with dummy audio to trigger PyTorch JIT compilation"""
    print("[Whisper] 🔥 Warming up model (PyTorch JIT compilation)...")
    try:
        # Generate 1 second of silence
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        
        wav_io = io.BytesIO()
        sf.write(wav_io, dummy_audio, SAMPLE_RATE, format="WAV")
        wav_io.seek(0)
        
        response = requests.post(
            "http://localhost:5000/transcribe",
            files={"audio": ("warmup.wav", wav_io, "audio/wav")},
            timeout=10
        )
        print("[Whisper] ✅ Model warmed up - first transcription will be fast!")
    except Exception as e:
        print(f"[Whisper] ⚠️ Warmup failed: {e}")

def print_session_stats():
    """Print statistics for this testing session"""
    elapsed = time.time() - start_time
    if transcription_count > 0:
        avg_gap = (elapsed - total_audio_duration) / transcription_count
    else:
        avg_gap = 0
    
    print("\n" + "="*70)
    print("  📊 SESSION STATISTICS")
    print("="*70)
    print(f"  Total transcriptions:   {transcription_count}")
    print(f"  Total audio captured:   {total_audio_duration:.1f}s")
    print(f"  Session duration:       {elapsed:.1f}s")
    print(f"  Avg gap between:        {avg_gap:.2f}s")
    print("="*70 + "\n")

# === Main Loop ===
def listen():
    channels = find_device_index()
    
    # Display current hardware configuration
    config = load_respeaker_config()
    display_hardware_config(config)
    
    # Warm up Whisper model (eliminates slow first transcription)
    warmup_whisper()
    
    print("\n" + "="*70)
    print("[Mode] 🧪 PURE TRANSCRIPTION TEST MODE")
    print("[Mode]    No LLM | No TTS | No GUI | Continuous transcription")
    print("[Audio] BARE-BONES PIPELINE")
    print("[Audio]   Hardware DSP → Channel 0 → VAD → Whisper")
    print("[Audio]   (Configure: sudo python3 scripts/tune_respeaker.py [profile])")
    
    if ENABLE_ADVANCED_FILTER:
        print("\n[Filter] ✅ ADVANCED MULTI-FEATURE FILTER: ENABLED")
        print(f"[Filter]    Duration > {SPEECH_DURATION_MIN}s | ZCR < {SPEECH_ZCR_MAX} | Flatness < {SPEECH_FLATNESS_MAX}")
        print(f"[Filter]    SpCent: {SPEECH_CENTROID_MIN}-{SPEECH_CENTROID_MAX}Hz | SpBand > {SPEECH_BAND_MIN}")
    else:
        print("\n[Filter] 💤 Advanced filter: DISABLED (VAD only)")
        print("[Filter]    Set ENABLE_ADVANCED_FILTER = True to enable multi-feature filtering")
    
    print("="*70 + "\n")
    
    # ARM/Jetson-specific audio configuration
    import platform
    is_arm = platform.machine().startswith('aarch') or platform.machine().startswith('arm')
    
    # Create stream with ARM-compatible settings
    stream_params = {
        'device': DEVICE_INDEX,
        'channels': channels,
        'samplerate': SAMPLE_RATE,
        'blocksize': FRAME_SIZE,
        'dtype': 'float32'
    }
    
    # Add latency for ARM devices (helps with ALSA buffer issues)
    if is_arm:
        print("[Audio] 🔧 Detected ARM architecture - using latency='high' for stability")
        stream_params['latency'] = 'high'
    
    try:
        stream = sd.InputStream(**stream_params)
    except Exception as e:
        print(f"[Audio] ⚠️  Failed to open stream with default settings: {e}")
        print("[Audio] 🔄 Trying alternative configuration...")
        # Fallback: use lower latency or different blocksize
        stream_params['latency'] = 0.2  # 200ms latency
        stream_params['blocksize'] = 1024  # Smaller blocksize
        try:
            stream = sd.InputStream(**stream_params)
            print("[Audio] ✅ Stream opened with fallback settings")
        except Exception as e2:
            print(f"[Audio] ❌ Failed to open audio stream: {e2}")
            print("[Audio] 💡 Try: sudo apt-get install --reinstall libportaudio2")
            raise
    
    with stream:
        print("[Listener] 🎤 Ready! Speak to test transcription...\n")
        
        while True:
            buffer = []
            silence_start = None
            last_vad_reset = time.time()  # Track last VAD reset to prevent decay
            
            # === Wait for speech ===
            while True:
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Stream error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Periodic VAD reset to prevent state decay during long silence
                # Reset every 5 seconds to keep VAD responsive
                if time.time() - last_vad_reset > 5.0:
                    model_vad.reset_states()
                    last_vad_reset = time.time()
                    print(f"\n[VAD] 🔄 Periodic state reset (prevents decay)", end="\r")
                
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Calculate audio features
                features = calculate_audio_features(channel_0)
                
                # Display key features in real-time (emphasize energy levels)
                print(f"[VAD] {vad_prob:.2f} | RMS {features['rms']:.4f} | Peak {features['peak']:.3f} | ZCR {features['zcr']:.3f} | SpFlat {features['spectral_flatness']:.2f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech detected (VAD={vad_prob:.2f}, RMS={features['rms']:.4f}, Peak={features['peak']:.3f})")
                    print(f"[Features] ZCR={features['zcr']:.3f} | SpCentroid={features['spectral_centroid']:.0f}Hz | SpFlat={features['spectral_flatness']:.3f}")
                    
                    # Apply advanced filter if enabled
                    if ENABLE_ADVANCED_FILTER:
                        is_speech_result, reason = is_likely_speech(features)
                        if not is_speech_result:
                            print(f"[Filter] ❌ REJECTED: {reason}")
                            print("[Filter] 🔄 Returning to listening (not speech)\n")
                            continue  # Back to waiting for speech
                        else:
                            print(f"[Filter] ✅ PASSED: {reason}")
                    
                    buffer.append(audio_block)
                    break
            
            # === Record speech ===
            while True:
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Error: {e}")
                    break
                
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                buffer.append(audio_block)
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n[VAD] ⏹️  Speech ended")
                        break
                else:
                    silence_start = None
                
                print(".", end="", flush=True)
            
            # === Process audio ===
            full_audio = np.concatenate(buffer)
            mono = full_audio[:, 0]  # Channel 0 only
            
            # RAW audio from hardware - no software processing
            if len(mono) < MIN_AUDIO_SAMPLES:
                print("⚠️  Too short\n")
                # Reset VAD state before next utterance
                model_vad.reset_states()
                continue
            
            text = transcribe(mono)
            
            # Reset VAD state for next utterance (critical for consistent performance)
            model_vad.reset_states()
            
            print("[Listener] 🎤 Ready for next input...\n")

if __name__ == "__main__":
    try:
        listen()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  👋 Testing Session Ended")
        print_session_stats()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print_session_stats()

