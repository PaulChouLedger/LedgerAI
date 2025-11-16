import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
import subprocess
import re
from scipy.fft import rfft, rfftfreq
# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from speaker import speak_llm_response, is_playing
from gui.aura_gui import set_transcribing

# === Config ===
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.2  # 500ms of silence before stopping
VAD_START_THRESHOLD = 0.25  # Lowered - beamforming provides good noise rejection
VAD_SILENCE_THRESHOLD = 0.15  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

# === Device Configuration ===
DEVICE_NAME = "reSpeaker"
MICROPHONE_CHANNEL = 0  # Channel to use for audio processing (0 or 1, device has 2 channels total)

# === Advanced Multi-Feature Speech Detection ===
# Enabled - Filters out low-energy noise bursts that trigger VAD
ENABLE_ADVANCED_FILTER = True

# Thresholds tuned from empirical testing
SPEECH_ZCR_MAX = 0.40           # Reject if ZCR > this
SPEECH_FLATNESS_MAX = 0.60      # More tolerant of flatness for far-field/quiet speech
SPEECH_CENTROID_MIN = 300       # Hz - reject if too low (rumble/fan)
SPEECH_CENTROID_MAX = 3000      # Hz - reject if too high (hiss)
SPEECH_BAND_MIN = 0.30          # Reject if insufficient energy in speech band
SPEECH_DURATION_MIN = 0.4       # Seconds - reject if too short (noise bursts)
SPEECH_HIGH_FREQ_MAX = 0.08     # Allow a bit more high-frequency content

# CRITICAL: Energy thresholds (most reliable discriminators)
# Updated after firmware tweaks - speech now has lower RMS/Peak values
SPEECH_RMS_MIN = 0.0015         # Lower RMS threshold to accept quieter speech
SPEECH_RMS_MAX = 0.40           # Reject if RMS > this (abnormally loud = likely noise/artifact)
SPEECH_PEAK_MIN = 0.003         # Lower peak threshold to accept softer speech

# BARE-BONES: Hardware DSP → Channel 0 → VAD → Advanced Filter → Whisper

# === Soft Clipping Prevention ===
ENABLE_SOFT_LIMITER = False      # Prevent clipping from near-field speech
LIMITER_THRESHOLD = 0.95        # Start limiting above this peak level
LIMITER_KNEE = 0.05             # Soft knee width for smooth limiting

DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# === Transcription Blocking ===
_transcription_blocked = False  # Global flag to block transcription
_block_reason = None  # Reason for blocking (for debugging)

# === Transcription Blocking Functions ===
def block_transcription(reason="Manual block"):
    """Block transcription (e.g., when dialog is open or mic button pressed)"""
    global _transcription_blocked, _block_reason
    _transcription_blocked = True
    _block_reason = reason
    print(f"[Listener] 🚫 Transcription BLOCKED: {reason}")

def unblock_transcription():
    """Unblock transcription"""
    global _transcription_blocked, _block_reason
    _transcription_blocked = False
    reason = _block_reason
    _block_reason = None
    print(f"[Listener] ✅ Transcription UNBLOCKED (was: {reason})")

def is_transcription_blocked():
    """Check if transcription is currently blocked"""
    return _transcription_blocked

def toggle_transcription():
    """Toggle transcription blocking (for microphone button)"""
    if _transcription_blocked:
        unblock_transcription()
        return False  # Now unblocked
    else:
        block_transcription("Microphone button")
        return True  # Now blocked

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# === Find Device ===
def find_device_index():
    global DEVICE_INDEX
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower():
            DEVICE_INDEX = i
            print(f"[Listener] 🎧 Found: {device['name']} (index {i})")
            return 2  # Device has 2 channels total (2 in, 2 out)
    raise RuntimeError("Microphone not found")

# === Audio Feature Extraction ===
def calculate_audio_features(audio_chunk, sample_rate=SAMPLE_RATE):
    """
    Calculate multiple audio features for speech detection
    Returns: dict with all features
    """
    features = {}
    
    # Energy metrics (most important)
    features['rms'] = np.sqrt(np.mean(audio_chunk ** 2))
    features['peak'] = np.max(np.abs(audio_chunk))
    
    # Zero Crossing Rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
    features['zcr'] = zero_crossings / len(audio_chunk)
    
    # Spectral features (frequency domain)
    fft_vals = rfft(audio_chunk)
    fft_freq = rfftfreq(len(audio_chunk), 1/sample_rate)
    magnitude = np.abs(fft_vals)
    
    # Spectral Centroid
    if np.sum(magnitude) > 0:
        features['spectral_centroid'] = np.sum(fft_freq * magnitude) / np.sum(magnitude)
    else:
        features['spectral_centroid'] = 0
    
    # Spectral Flatness
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude)
    if arithmetic_mean > 0:
        features['spectral_flatness'] = geometric_mean / arithmetic_mean
    else:
        features['spectral_flatness'] = 0
    
    # Speech Band Energy (300-3400 Hz)
    speech_band_mask = (fft_freq >= 300) & (fft_freq <= 3400)
    speech_band_energy = np.sum(magnitude[speech_band_mask] ** 2)
    total_energy = np.sum(magnitude ** 2)
    if total_energy > 0:
        features['speech_band_ratio'] = speech_band_energy / total_energy
    else:
        features['speech_band_ratio'] = 0
    
    # High Frequency Ratio (4000+ Hz - indicates noise/hiss)
    high_freq_mask = fft_freq >= 4000
    high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
    if total_energy > 0:
        features['high_freq_ratio'] = high_freq_energy / total_energy
    else:
        features['high_freq_ratio'] = 0
    
    return features

def soft_limit(audio_data):
    """
    Apply soft limiting to prevent clipping from near-field speech
    
    Uses a smooth tanh-based limiter that:
    - Passes audio below threshold unchanged
    - Gradually compresses audio above threshold
    - Prevents hard clipping (peak = 1.0)
    
    This allows AGC to boost far-field speech without clipping near-field.
    """
    if not ENABLE_SOFT_LIMITER:
        return audio_data
    
    peak = np.max(np.abs(audio_data))
    
    # No limiting needed if below threshold
    if peak <= LIMITER_THRESHOLD:
        return audio_data
    
    # Apply soft knee compression using tanh
    # This creates a smooth transition from linear to compressed
    def soft_knee_compress(x, threshold, knee):
        """Soft knee compression with smooth transition"""
        # Linear region (below threshold - knee)
        linear_threshold = threshold - knee
        
        # For samples below linear threshold, pass through unchanged
        if np.abs(x) <= linear_threshold:
            return x
        
        # For samples in knee region, apply smooth compression
        sign = np.sign(x)
        abs_x = np.abs(x)
        
        if abs_x <= threshold:
            # Smooth transition zone
            ratio = (abs_x - linear_threshold) / knee
            compressed = linear_threshold + knee * np.tanh(ratio)
            return sign * compressed
        else:
            # Above threshold - strong compression
            excess = abs_x - threshold
            compressed = threshold + LIMITER_KNEE * np.tanh(excess / LIMITER_KNEE)
            return sign * compressed
    
    # Vectorize the compression function
    vectorized_compress = np.vectorize(soft_knee_compress)
    limited = vectorized_compress(audio_data, LIMITER_THRESHOLD, LIMITER_KNEE)
    
    return limited.astype(np.float32)

def is_likely_speech(features, duration=None):
    """
    Apply multi-feature analysis to distinguish speech from noise
    
    Returns: (is_speech: bool, reason: str)
    """
    reasons = []
    
    # Check Energy Levels FIRST (most reliable)
    if features['rms'] < SPEECH_RMS_MIN:
        reasons.append(f"RMS too low ({features['rms']:.4f} < {SPEECH_RMS_MIN})")
    
    if features['rms'] > SPEECH_RMS_MAX:
        reasons.append(f"RMS too high ({features['rms']:.4f} > {SPEECH_RMS_MAX}) - likely noise/artifact")
    
    if features['peak'] < SPEECH_PEAK_MIN:
        reasons.append(f"Peak too low ({features['peak']:.4f} < {SPEECH_PEAK_MIN})")
    
    # Check Duration
    if duration is not None and duration < SPEECH_DURATION_MIN:
        reasons.append(f"Too short ({duration:.2f}s < {SPEECH_DURATION_MIN}s)")
    
    # Check Zero Crossing Rate
    if features['zcr'] > SPEECH_ZCR_MAX:
        reasons.append(f"ZCR too high ({features['zcr']:.3f} > {SPEECH_ZCR_MAX})")
    
    # Check Spectral Flatness
    if features['spectral_flatness'] > SPEECH_FLATNESS_MAX:
        reasons.append(f"Too flat/noisy ({features['spectral_flatness']:.3f} > {SPEECH_FLATNESS_MAX})")
    
    # Check Spectral Centroid
    if features['spectral_centroid'] < SPEECH_CENTROID_MIN:
        reasons.append(f"SpCent too low ({features['spectral_centroid']:.0f}Hz < {SPEECH_CENTROID_MIN}Hz)")
    elif features['spectral_centroid'] > SPEECH_CENTROID_MAX:
        reasons.append(f"SpCent too high ({features['spectral_centroid']:.0f}Hz > {SPEECH_CENTROID_MAX}Hz)")
    
    # Check Speech Band Energy
    if features['speech_band_ratio'] < SPEECH_BAND_MIN:
        reasons.append(f"Low speech band energy ({features['speech_band_ratio']:.2f} < {SPEECH_BAND_MIN})")
    
    # Check High Frequency Ratio (noise/hiss indicator)
    if features['high_freq_ratio'] > SPEECH_HIGH_FREQ_MAX:
        reasons.append(f"High freq noise ({features['high_freq_ratio']:.3f} > {SPEECH_HIGH_FREQ_MAX})")
    
    is_speech = len(reasons) == 0
    reason = " | ".join(reasons) if reasons else "All checks passed"
    
    return is_speech, reason

# === Load VAD ===
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)

# === Load Hardware Config ===
def load_xvf3800_config():
    """Load saved XVF3800 configuration (no permissions needed)"""
    config_file = os.path.expanduser("~/LedgerAI/data/xvf3800_config.json")
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
        print("[Hardware] 💡 Run: sudo python3 setup/scripts/tune_xvf3800.py [preset]\n")
        return
    
    preset = state.get('preset', 'unknown')
    config = state.get('config', {})
    
    print("\n" + "="*70)
    print(f"  📊 HARDWARE CONFIGURATION: {preset.upper()}")
    print("="*70)
    
    # AGC
    if config.get('PP_AGCONOFF', 0) == 1:
        print(f"  AGC:                    ✅ ENABLED")
        print(f"    Target Level:         {config.get('PP_AGCDESIREDLEVEL', 0):.2f} RMS")
        print(f"    Max Gain:             {config.get('PP_AGCMAXGAIN', 0):.0f} linear")
    else:
        print(f"  AGC:                    ❌ DISABLED")
    
    # High-pass Filter
    hpf_labels = {0: "OFF", 1: "70 Hz", 2: "125 Hz", 3: "150 Hz", 4: "180 Hz"}
    hpf_val = config.get('AEC_HPFONOFF', 0)
    hpf_label = hpf_labels.get(hpf_val, str(hpf_val))
    print(f"  High-Pass Filter:       {hpf_label}")
    
    # Echo Cancellation
    if config.get('PP_ECHOONOFF', 0) == 1:
        print(f"  Echo Cancellation:      ✅ ENABLED")
    else:
        print(f"  Echo Cancellation:      ❌ DISABLED")
    
    print("="*70 + "\n")

# === Transcribe ===
def transcribe(audio):
    """Send raw audio to Whisper"""
    # Apply soft limiting to prevent clipping from near-field speech
    original_peak = np.max(np.abs(audio))
    audio = soft_limit(audio)
    limited_peak = np.max(np.abs(audio))
    
    if original_peak > LIMITER_THRESHOLD:
        print(f"[Limiter] 🎚️  Peak reduced: {original_peak:.4f} → {limited_peak:.4f}")
    
    rms = np.sqrt(np.mean(audio ** 2))
    peak = limited_peak
    print(f"[Audio] RMS={rms:.6f}, Peak={peak:.4f}, Duration={len(audio)/SAMPLE_RATE:.2f}s")
    
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
        print(f"[Whisper] '{text}'")
        return text
    except Exception as e:
        print(f"[Whisper] ❌ {e}")
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

# === Send to LLM ===
def send_to_llm(text):
    global prompt_history
    
    if not text:
        return
    
    prompt_history.append(text)
    if len(prompt_history) > CONTEXT_DEPTH:
        prompt_history.pop(0)
    
    # speaker.speak_llm_response() handles the LLM request itself
    speak_llm_response(text)

# === Welcome Prompt ===
def detect_output_device_for_welcome():
    """Auto-detect audio output device for welcome prompt (same logic as speaker.py)"""
    try:
        output = subprocess.check_output(["aplay", "-l"], text=True)
        # First, try to find UACDemoV1.0
        for line in output.splitlines():
            if "UACDemoV1.0" in line:
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    return f"plughw:{card_num},0"  # Use plug plugin for format conversion
        
        # Fallback: find any USB audio device with output
        for line in output.splitlines():
            if "USB Audio" in line and ("0 in" in line or "out" in line):
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    return f"plughw:{card_num},0"  # Use plug plugin for format conversion
        
        # No USB device found - use default with plug plugin
        return "plug:default"
    except Exception as e:
        print(f"[Listener] ⚠️ Failed to detect output device for welcome: {e}")
        return "default"

def play_welcome_prompt(stream):
    try:
        print("[Aura] 🔊 Playing welcome prompt...")
        stream.stop()
        
        # Detect output device (same as speaker module)
        output_device = detect_output_device_for_welcome()
        print(f"[Aura] 🔊 Using output device: {output_device}")
        
        # Use plughw or plug:default for automatic format conversion (44100 Hz mono -> device format)
        # Welcome audio is 44100 Hz mono, device might need different format
        cmd = ["aplay", "-D", output_device, WELCOME_AUDIO_PATH]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Aura] ⚠️ aplay failed with device {output_device}: {result.stderr}")
            # Fallback to default device
            print(f"[Aura] 🔄 Trying default device...")
            subprocess.run(["aplay", "-D", "default", WELCOME_AUDIO_PATH], check=False)
        
        time.sleep(0.25)
        stream.start()
        
        # Flush buffer
        for _ in range(5):
            try:
                stream.read(FRAME_SIZE)
            except:
                break
        
        print("[Aura] 🎤 Mic resumed after welcome prompt")
        
        try:
            from gui.aura_gui import set_setup_complete, set_welcome_played, set_listening_ready
            set_setup_complete()
            set_welcome_played()
            set_listening_ready()
            print("[Aura] ✅ Setup complete, listener ready")
        except ImportError:
            pass
    except Exception as e:
        print(f"[Aura] ❌ Failed to play welcome prompt: {e}")
        import traceback
        traceback.print_exc()

# === Main Loop ===
def listen():
    global vad_zero_count
    
    channels = find_device_index()
    
    # Display current hardware configuration
    config = load_xvf3800_config()
    display_hardware_config(config)
    
    # Warm up Whisper model (eliminates slow first transcription)
    warmup_whisper()
    
    print("\n" + "="*70)
    print("[Audio] BARE-BONES PIPELINE")
    print("[Audio]   Hardware DSP → Channel 0 → VAD → Whisper")
    print("[Audio]   (Configure: python3 setup/scripts/tune_xvf3800.py [preset])")
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
        
        play_welcome_prompt(stream)
        
        while True:
            # Pause during TTS
            if is_playing():
                print("[Listener] ⏸️ Pausing mic during playback")
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                
                # Flush buffer
                print("[Listener] 🧹 Flushing mic buffer...")
                for _ in range(5):
                    try:
                        stream.read(FRAME_SIZE)
                    except:
                        break
                
                print("[Listener] ▶️ Mic resumed after playback (buffer flushed)")
            
            buffer = []
            silence_start = None
            last_vad_reset = time.time()  # Track last VAD reset to prevent decay
            
            # === Wait for speech ===
            while True:
                # Check if transcription is blocked (dialog open or mic button pressed)
                if is_transcription_blocked():
                    time.sleep(0.1)
                    continue
                
                if is_playing():
                    break
                
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
                
                channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                
                if channel_audio.size < 512:
                    continue
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_audio), SAMPLE_RATE).item()
                
                # Calculate audio features
                features = calculate_audio_features(channel_audio)
                
                print(f"[VAD] {vad_prob:.2f} | RMS {features['rms']:.4f} | Peak {features['peak']:.3f}", end="\r")
                
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
                    
                    set_transcribing(True)
                    buffer.append(audio_block)
                    break
            
            # === Record speech ===
            while True:
                if is_playing():
                    set_transcribing(False)
                    break
                
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Error: {e}")
                    set_transcribing(False)
                    break
                
                channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                
                if channel_audio.size < 512:
                    continue
                
                buffer.append(audio_block)
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_audio), SAMPLE_RATE).item()
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n[VAD] ⏹️  Speech ended")
                        set_transcribing(False)
                        break
                else:
                    silence_start = None
                
                print(".", end="", flush=True)
            
            if is_playing():
                set_transcribing(False)
                continue
            
            # === Process audio ===
            full_audio = np.concatenate(buffer)
            mono = full_audio[:, 0]  # Channel 0 only
            
            # RAW audio from hardware - no software processing
            if len(mono) < MIN_AUDIO_SAMPLES:
                print("⚠️  Too short\n")
                set_transcribing(False)
                # Reset VAD state before next utterance
                model_vad.reset_states()
                continue
            
            # Send to Whisper (initial filter already passed)
            text = transcribe(mono)
            
            # Reset VAD state for next utterance (critical for consistent performance)
            model_vad.reset_states()
            
            if text:
                send_to_llm(text)

if __name__ == "__main__":
    try:
        listen()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
