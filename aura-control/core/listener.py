import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
from sounddevice import PortAudioError
import requests
import subprocess
import re
# Import rfft and rfftfreq - compatible with both old and new scipy versions
try:
    from scipy.fft import rfft, rfftfreq
except ImportError:
    # Fallback for older scipy versions (< 1.4.0)
    from scipy.fftpack import rfft
    from numpy.fft import rfftfreq
# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from speaker import speak_llm_response, is_playing
from gui.aura_gui import set_transcribing
# Wake word detection engines (Precise and OpenWakeWord)
from precise_wake_word import create_precise_wake_word_detector
from openwakeword_wake_word import create_openwakeword_detector

# === Config ===
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.2  # 500ms of silence before stopping
VAD_START_THRESHOLD = 0.25  # Lowered - beamforming provides good noise rejection
VAD_SILENCE_THRESHOLD = 0.15  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

# === Wake Word Configuration ===
# Wake word detection is controlled via Settings dialog (state.py)
# Default: disabled (can be toggled in Settings → AI Model Settings)

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
SPEECH_RMS_MIN = 0.0011         # Lower RMS threshold to accept quieter speech
SPEECH_RMS_MAX = 0.40           # Reject if RMS > this (abnormally loud = likely noise/artifact)
SPEECH_PEAK_MIN = 0.0023        # Lower peak threshold to accept softer speech

# BARE-BONES: Hardware DSP → Channel 0 → VAD → Advanced Filter → Whisper

# (Pre-gain removed)

# === Audio Normalization (for optimal Whisper transcription) ===
ENABLE_AUDIO_NORMALIZATION = True  # Normalize audio to optimal RMS for Whisper
TARGET_RMS_FOR_WHISPER = 0.12      # Optimal RMS level for Whisper (found via find_optimal_rms.py)

# === Wake Word Audio Normalization ===
# Wake word uses same normalization function and target RMS as Whisper for consistency
# Import from precise_wake_word.py to avoid duplication
try:
    from precise_wake_word import WAKE_WORD_TARGET_RMS, WAKE_WORD_MAX_GAIN
except ImportError:
    # Fallback if import fails - use same as Whisper for consistency
    WAKE_WORD_TARGET_RMS = 0.12  # Same as TARGET_RMS_FOR_WHISPER
    WAKE_WORD_MAX_GAIN = 10.0

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
def find_device_index(max_retries=10, initial_delay=1.0, max_delay=5.0):
    """
    Find microphone device with retry logic for boot-time initialization.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
    """
    global DEVICE_INDEX
    
    for attempt in range(max_retries):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if DEVICE_NAME.lower() in device["name"].lower():
                DEVICE_INDEX = i
                print(f"[Listener] 🎧 Found: {device['name']} (index {i})")
                if attempt > 0:
                    print(f"[Listener] ✅ Microphone available after {attempt} retry attempt(s)")
            return 2  # Device has 2 channels total (2 in, 2 out)
        
        # Device not found yet
        if attempt < max_retries - 1:
            # Exponential backoff: delay increases with each attempt, capped at max_delay
            delay = min(initial_delay * (2 ** attempt), max_delay)
            print(f"[Listener] ⏳ Microphone not found (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
            time.sleep(delay)
        else:
            # Last attempt failed
            print(f"[Listener] ❌ Microphone '{DEVICE_NAME}' not found after {max_retries} attempts")
            print(f"[Listener] 💡 Available devices:")
            for i, device in enumerate(devices):
                if device.get('max_input_channels', 0) > 0:
                    print(f"[Listener]    - {i}: {device['name']} ({device.get('max_input_channels', 0)} input channels)")
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

    # Low Frequency Rumble (0-100 Hz) - for diagnostics parity with test script
    low_freq_mask = fft_freq <= 100
    low_freq_energy = np.sum(magnitude[low_freq_mask] ** 2)
    if total_energy > 0:
        features['low_freq_ratio'] = low_freq_energy / total_energy
    else:
        features['low_freq_ratio'] = 0
    
    return features

def normalize_audio_for_whisper(audio_data, target_rms=TARGET_RMS_FOR_WHISPER, max_gain=None):
    """
    Normalize audio to target RMS level for optimal Whisper transcription.
    This matches the approach in find_optimal_rms.py which shows better results.
    Also used for wake word detection with different target RMS.
    
    Args:
        audio_data: Raw audio array
        target_rms: Target RMS level (default 0.12 - optimal for Whisper, 0.05 for wake word)
        max_gain: Maximum gain factor to prevent distortion (default: no limit for Whisper, 10.0 for wake word)
    
    Returns:
        Normalized audio array
    """
    if not ENABLE_AUDIO_NORMALIZATION:
        return audio_data
    
    current_rms = np.sqrt(np.mean(audio_data ** 2))
    
    # Use same threshold as wake word logic for consistency (0.0001 = 1e-4)
    if current_rms < 0.0001:
        return audio_data
    
    gain = target_rms / current_rms
    
    # Apply gain limiting if specified (for wake word detection)
    if max_gain is not None:
        gain = min(gain, max_gain)
    
    normalized = audio_data * gain
    # Soft clip to prevent distortion
    normalized = np.clip(normalized, -0.95, 0.95)
    
    return normalized

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
    # Normalize audio to optimal RMS level for Whisper (like find_optimal_rms.py)
    original_rms = np.sqrt(np.mean(audio ** 2))
    audio = normalize_audio_for_whisper(audio)
    normalized_rms = np.sqrt(np.mean(audio ** 2))
    
    # Apply soft limiting to prevent clipping from near-field speech
    original_peak = np.max(np.abs(audio))
    audio = soft_limit(audio)
    limited_peak = np.max(np.abs(audio))
    
    if ENABLE_AUDIO_NORMALIZATION and abs(original_rms - normalized_rms) > 0.001:
        print(f"[Normalize] 🔊 RMS normalized: {original_rms:.4f} → {normalized_rms:.4f} (target: {TARGET_RMS_FOR_WHISPER})")
    
    if original_peak > LIMITER_THRESHOLD:
        print(f"[Limiter] 🎚️  Peak reduced: {original_peak:.4f} → {limited_peak:.4f}")
    
    rms = normalized_rms
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
        
        # Small delay to let buffer fill after flush (critical for wake word to get audio)
        time.sleep(0.1)
        
        # Prime the stream by reading frames until we get real audio
        # This ensures wake word detection starts with valid audio data
        print("[Aura] 🔄 Priming audio stream...")
        for i in range(20):  # Try up to 20 frames (640ms at 16kHz)
            try:
                audio_block, _ = stream.read(FRAME_SIZE)
                if audio_block is not None and audio_block.size > 0:
                    channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                    audio_sum = np.abs(channel_audio).sum()
                    if audio_sum > 0.0001:  # Got real audio (not all zeros)
                        print(f"[Aura] ✅ Stream primed with real audio (frame {i+1}, RMS={np.sqrt(np.mean(channel_audio**2)):.6f})")
                        break
            except:
                pass
            time.sleep(0.01)  # Small delay between reads
        else:
            print("[Aura] ⚠️  Stream priming completed (may still be zeros)")
        
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
    import sys
    print("[Listener] 🎧 listen() function called", flush=True)
    sys.stdout.flush()
    global vad_zero_count
    
    channels = find_device_index()
    
    # Display current hardware configuration
    config = load_xvf3800_config()
    display_hardware_config(config)
    
    # Initialize wake word detector (if enabled)
    # Check if wake word is enabled in settings (even if detector fails to initialize)
    try:
        from state import get_wake_word_enabled
        wake_word_setting_enabled = get_wake_word_enabled()
    except ImportError:
        wake_word_setting_enabled = False
    
    # Initialize wake word detection only if enabled in settings
    import sys
    wake_word_detector = None
    wake_word_enabled = False
    wake_word_engine_name = "none"
    
    if wake_word_setting_enabled:
        # CRITICAL: Block transcription immediately if wake word is enabled
        # This prevents transcription from starting before the detector is ready
        print("[Wake Word] 🔒 Blocking transcription until wake word detector is ready...")
        block_transcription("Wake word enabled - waiting for detector initialization")
        
        # Get selected engine from settings
        try:
            from state import get_wake_word_engine
            selected_engine = get_wake_word_engine()
        except ImportError:
            selected_engine = "precise"  # Default to Precise
        
        print(f"[Wake Word] 🔄 Initializing wake word engine: {selected_engine}...", flush=True)
        sys.stdout.flush()
        
        # Try to initialize the selected engine
        try:
            if selected_engine == "openwakeword":
                wake_word_detector = create_openwakeword_detector()
                wake_word_engine_name = "OpenWakeWord"
            else:  # Default to Precise
                wake_word_detector = create_precise_wake_word_detector()
                wake_word_engine_name = "Mycroft Precise"
            
            if wake_word_detector:
                print(f"[Wake Word] ✅ {wake_word_engine_name} initialized successfully", flush=True)
                sys.stdout.flush()
                wake_word_enabled = True
                # Unblock transcription now that detector is ready (but still require wake word)
                unblock_transcription()
                print(f"[Wake Word] ✅ Wake word detector ready ({wake_word_engine_name}) - transcription will require wake word")
            else:
                print(f"[Wake Word] ❌ {wake_word_engine_name} initialization returned None", flush=True)
                sys.stdout.flush()
                print("[Wake Word] ⚠️  Wake word enabled in settings but detector failed to initialize")
                print("[Wake Word] 🔒 Transcription will remain BLOCKED until wake word detector is fixed")
                print("[Wake Word] 💡 Check logs above for initialization errors")
                if selected_engine == "openwakeword":
                    print("[Wake Word] 💡 OpenWakeWord setup:")
                    print("[Wake Word]     1. Install: pip install openwakeword")
                    print("[Wake Word]     2. Models are downloaded automatically on first use")
                else:
                    print("[Wake Word] 💡 Mycroft Precise setup:")
                    print("[Wake Word]     1. Install: pip install precise-runner")
                    print("[Wake Word]     2. Download binary: See install_mycroft_precise.sh")
                    print("[Wake Word]     3. Download model: wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb")
                    print("[Wake Word]     4. Place model in: ~/precise-models/ or ~/")
                print("[Wake Word] 💡 Or disable wake word in Settings → AI Model Settings")
        except Exception as e:
            print(f"[Wake Word] ❌ {wake_word_engine_name} failed: {e}", flush=True)
            import traceback
            print(f"[Wake Word] 🔍 Traceback: {traceback.format_exc()}", flush=True)
            sys.stdout.flush()
            wake_word_detector = None
            print("[Wake Word] ⚠️  Wake word enabled in settings but detector failed to initialize")
            print("[Wake Word] 🔒 Transcription will remain BLOCKED until wake word detector is fixed")
            print("[Wake Word] 💡 Check logs above for initialization errors")
            if selected_engine == "openwakeword":
                print("[Wake Word] 💡 OpenWakeWord setup:")
                print("[Wake Word]     1. Install: pip install openwakeword")
                print("[Wake Word]     2. Models are downloaded automatically on first use")
            else:
                print("[Wake Word] 💡 Mycroft Precise setup:")
                print("[Wake Word]     1. Install: pip install precise-runner")
                print("[Wake Word]     2. Download binary: See install_mycroft_precise.sh")
                print("[Wake Word]     3. Download model: wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb")
                print("[Wake Word]     4. Place model in: ~/precise-models/ or ~/")
            print("[Wake Word] 💡 Or disable wake word in Settings → AI Model Settings")
    else:
        print("[Wake Word] ℹ️  Wake word detection disabled (toggle in Settings)")
        wake_word_enabled = False
    
    if wake_word_enabled:
        print("\n" + "="*70)
        print("[Audio] WAKE WORD PIPELINE")
        print("[Audio]   Hardware DSP → Wake Word → VAD → Whisper")
        print("[Audio]   (Configure: python3 setup/scripts/tune_xvf3800.py [preset])")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("[Audio] BARE-BONES PIPELINE")
        print("[Audio]   Hardware DSP → Channel 0 → VAD → Whisper")
        print("[Audio]   (Configure: python3 setup/scripts/tune_xvf3800.py [preset])")
        print("="*70 + "\n")
    
    # Warm up Whisper model (eliminates slow first transcription)
    warmup_whisper()
    
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
    
    try:
        with stream:
            
            play_welcome_prompt(stream)
            
            # Wake word buffer for Mycroft Precise (needs 2048 samples = 128ms at 16kHz)
            wake_word_buffer = []
            listening_active = False  # True after wake word detected
            stream_valid = True  # Track if stream is still valid
            
            while stream_valid:
                # Pause during TTS
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    try:
                        stream.stop()
                        while is_playing():
                            time.sleep(0.1)
                        stream.start()
                        
                        # Flush buffer
                        print("[Listener] 🧹 Flushing mic buffer...")
                        for _ in range(5):
                            try:
                                stream.read(FRAME_SIZE)
                            except (PortAudioError, Exception):
                                break
                        
                        print("[Listener] ▶️ Mic resumed after playback (buffer flushed)")
                    except PortAudioError as pa_error:
                        error_code = getattr(pa_error, 'errno', None)
                        if error_code in [-9999, -9988]:
                            print(f"[Listener] ⚠️  Stream error during TTS pause: {pa_error}")
                            stream_valid = False
                            break
                    listening_active = False  # Reset after TTS
                
                # === STAGE 1: Wake Word Detection (if enabled) ===
                # Only process wake word if detector is actually available
                # Check stream_valid at start of each iteration
                if not stream_valid:
                    break
                
                # Debug: Log wake word detection state
                if wake_word_enabled and wake_word_detector is not None and not hasattr(wake_word_detector, '_state_logged'):
                    print(f"[Wake Word] 🔄 Wake word detection active: enabled={wake_word_enabled}, listening_active={listening_active}, detector=available")
                    wake_word_detector._state_logged = True
                elif wake_word_enabled and wake_word_detector is None:
                    # Only log once if detector is None
                    if not hasattr(listen, '_wake_word_none_logged'):
                        print(f"[Wake Word] ⚠️  Wake word enabled but detector is None - transcription will be blocked")
                        listen._wake_word_none_logged = True
                    
                if wake_word_enabled and not listening_active and wake_word_detector is not None:
                    try:
                        # Read audio using EXACT same method as VAD loop (line 1011)
                        # VAD loop wraps this in try/except, so we do too
                        audio_block, overflowed = stream.read(FRAME_SIZE)
                        
                        # Debug: Check if audio_block is valid (first few reads)
                        if not hasattr(wake_word_detector, '_stream_debug'):
                            wake_word_detector._stream_debug = 0
                        wake_word_detector._stream_debug += 1
                        if wake_word_detector._stream_debug <= 3:
                            print(f"[Wake Word] 🔍 Stream Debug (read {wake_word_detector._stream_debug}): audio_block shape={audio_block.shape if audio_block is not None else None}, dtype={audio_block.dtype if audio_block is not None else None}, min={audio_block.min() if audio_block is not None else None:.6f}, max={audio_block.max() if audio_block is not None else None:.6f}")
                            print(f"[Wake Word] 🔍 Stream state: active={stream.active}, stopped={stream.stopped}")
                        
                        if audio_block is None or audio_block.size == 0:
                            if wake_word_detector._stream_debug <= 10:
                                print(f"[Wake Word] ⚠️  Empty audio_block on read {wake_word_detector._stream_debug}")
                            continue
                        
                        # Extract channel using EXACT same method as VAD loop (line 1036)
                        channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                        
                        # VAD loop checks size < 512 and continues - we do the same
                        if channel_audio.size < 512:
                            continue
                        
                        # Use exact same audio processing as main listener VAD
                        # This ensures wake word sees identical audio levels and features
                        # VAD calculates features AFTER size check, so we do too
                        # Use same audio feature calculation as main listener
                        features = calculate_audio_features(channel_audio)
                        rms = features['rms']
                        peak = features['peak']
                        
                        # Store features for use when wake word is detected
                        if not hasattr(wake_word_detector, '_last_features'):
                            wake_word_detector._last_features = None
                        wake_word_detector._last_features = features
                        
                        # Debug: check audio levels occasionally (every 100 frames or first 5)
                        # Show same format as main listener VAD output
                        if not hasattr(wake_word_detector, '_audio_level_debug'):
                            wake_word_detector._audio_level_debug = 0
                        wake_word_detector._audio_level_debug += 1
                        if wake_word_detector._audio_level_debug <= 5 or wake_word_detector._audio_level_debug % 100 == 0:
                            print(f"[Wake Word] 🔍 DEBUG Audio: RMS={rms:.4f}, Peak={peak:.4f} (Frame {wake_word_detector._audio_level_debug}) - Same as VAD processing")
                        
                        # Handle different wake word detector types
                        # Mycroft Precise requires 2048 samples (128ms at 16kHz), so we buffer frames
                        # OpenWakeWord handles buffering internally, so we can call process() directly
                        
                        # Check if detector uses frame_length (Precise) or handles buffering internally (OpenWakeWord)
                        frame_length = getattr(wake_word_detector, 'frame_length', None)
                        
                        if frame_length:
                            # Precise-style detector: needs external buffering
                            # Ensure channel_audio is 1D before appending
                            if channel_audio.ndim > 1:
                                channel_audio = channel_audio.flatten()
                            wake_word_buffer.append(channel_audio)
                            
                            required_samples = frame_length
                            total_samples = sum(len(chunk) for chunk in wake_word_buffer)
                            
                            if total_samples >= required_samples:
                                # Concatenate enough samples for one Mycroft Precise frame
                                # Ensure all arrays in buffer are 1D before concatenation
                                flattened_buffer = [chunk.flatten() if chunk.ndim > 1 else chunk for chunk in wake_word_buffer]
                                combined_audio = np.concatenate(flattened_buffer)
                                wake_word_frame = combined_audio[:required_samples]
                                
                                # Keep remaining samples for next frame
                                if len(combined_audio) > required_samples:
                                    wake_word_buffer = [combined_audio[required_samples:]]
                                else:
                                    wake_word_buffer = []
                                
                                # Detect wake word
                                wake_detected, confidence = wake_word_detector.process(wake_word_frame)
                                
                                # Debug output (show confidence less frequently to reduce spam)
                                if not hasattr(wake_word_detector, '_debug_counter'):
                                    wake_word_detector._debug_counter = 0
                                wake_word_detector._debug_counter += 1
                                
                                # Show confidence every 100 frames, or if confidence > threshold/10 (getting close), or if confidence > 0
                                threshold = getattr(wake_word_detector, 'threshold', 0.5)
                                show_debug = (wake_word_detector._debug_counter % 100 == 0) or (confidence > threshold / 10) or (confidence > 0.001)
                                if show_debug:
                                    status = "🔴" if confidence < threshold * 0.5 else "🟡" if confidence < threshold else "🟢"
                                    print(f"[Wake Word] {status} Confidence: {confidence:.6f} (threshold: {threshold:.6f}) - Frame {wake_word_detector._debug_counter}")
                                    
                                # Heartbeat every 500 frames to confirm we're still listening
                                if wake_word_detector._debug_counter % 500 == 0:
                                    print(f"[Wake Word] 💓 Still listening for wake word... (Frame {wake_word_detector._debug_counter})")
                                
                                if wake_detected:
                                    # Print RMS and audio features at detection time (using same calculation as VAD)
                                    if hasattr(wake_word_detector, '_last_features') and wake_word_detector._last_features:
                                        det_features = wake_word_detector._last_features
                                        print(f"\n[Wake Word] ✅ Wake word detected! (confidence: {confidence:.6f})")
                                        print(f"[Wake Word] 📊 Audio at detection: RMS={det_features['rms']:.4f}, Peak={det_features['peak']:.4f}")
                                        print(f"[Wake Word] 📊 Features: ZCR={det_features['zcr']:.3f} | SpCentroid={det_features['spectral_centroid']:.0f}Hz | SpFlat={det_features['spectral_flatness']:.3f}")
                                    else:
                                        # Fallback if features not available
                                        detection_rms = np.sqrt(np.mean(wake_word_frame**2))
                                        detection_peak = np.abs(wake_word_frame).max()
                                        print(f"\n[Wake Word] ✅ Wake word detected! (confidence: {confidence:.6f})")
                                        print(f"[Wake Word] 📊 Audio at detection: RMS={detection_rms:.4f}, Peak={detection_peak:.4f}")
                                    listening_active = True
                                    
                                    # Visual feedback (if GUI available) - trigger solid red LED (not pulsating yet)
                                    try:
                                        from gui.aura_gui import set_wake_word_detected
                                        set_wake_word_detected(True)  # Solid red LED, waiting for speech
                                    except (ImportError, NameError):
                                        pass
                                    
                                    # Clear wake word buffer
                                    wake_word_buffer = []
                                    
                                    # Wait a moment before starting VAD (avoid wake word in transcription)
                                    time.sleep(0.3)
                                    
                                    # Reset VAD state for fresh start
                                    model_vad.reset_states()
                                    
                                    print("[Wake Word] 🎤 Listening for speech...")
                            else:
                                # Not enough samples yet, continue buffering - loop back to read more
                                continue
                        else:
                            # OpenWakeWord-style detector: handles buffering internally
                            # Use exact same channel_audio as VAD (no conversion needed - stream already provides float32 normalized to [-1, 1])
                            # Ensure channel_audio is 1D (same as VAD uses it)
                            if channel_audio.ndim > 1:
                                channel_audio = channel_audio.flatten()
                            
                            # Call process directly with same audio VAD uses - OpenWakeWord handles buffering internally
                            wake_detected, confidence = wake_word_detector.process(channel_audio)
                            
                            # Debug output (show confidence less frequently to reduce spam)
                            if not hasattr(wake_word_detector, '_debug_counter'):
                                wake_word_detector._debug_counter = 0
                            wake_word_detector._debug_counter += 1
                            
                            # Show confidence every 100 frames, or if confidence > threshold/10 (getting close), or if confidence > 0
                            threshold = getattr(wake_word_detector, 'threshold', 0.5)
                            show_debug = (wake_word_detector._debug_counter % 100 == 0) or (confidence > threshold / 10) or (confidence > 0.001)
                            if show_debug:
                                status = "🔴" if confidence < threshold * 0.5 else "🟡" if confidence < threshold else "🟢"
                                print(f"[Wake Word] {status} Confidence: {confidence:.6f} (threshold: {threshold:.6f}) - Frame {wake_word_detector._debug_counter}")
                                
                            # Heartbeat every 500 frames to confirm we're still listening
                            if wake_word_detector._debug_counter % 500 == 0:
                                print(f"[Wake Word] 💓 Still listening for wake word... (Frame {wake_word_detector._debug_counter})")
                            
                            if wake_detected:
                                # Print RMS and audio features at detection time (using same calculation as VAD)
                                if hasattr(wake_word_detector, '_last_features') and wake_word_detector._last_features:
                                    det_features = wake_word_detector._last_features
                                    print(f"\n[Wake Word] ✅ Wake word detected! (confidence: {confidence:.6f})")
                                    print(f"[Wake Word] 📊 Audio at detection: RMS={det_features['rms']:.4f}, Peak={det_features['peak']:.4f}")
                                    print(f"[Wake Word] 📊 Features: ZCR={det_features['zcr']:.3f} | SpCentroid={det_features['spectral_centroid']:.0f}Hz | SpFlat={det_features['spectral_flatness']:.3f}")
                                else:
                                    # Fallback if features not available
                                    detection_rms = np.sqrt(np.mean(channel_audio**2))
                                    detection_peak = np.abs(channel_audio).max()
                                    print(f"\n[Wake Word] ✅ Wake word detected! (confidence: {confidence:.6f})")
                                    print(f"[Wake Word] 📊 Audio at detection: RMS={detection_rms:.4f}, Peak={detection_peak:.4f}")
                                listening_active = True
                                
                                # Visual feedback (if GUI available) - trigger solid red LED (not pulsating yet)
                                try:
                                    from gui.aura_gui import set_wake_word_detected
                                    set_wake_word_detected(True)  # Solid red LED, waiting for speech
                                except (ImportError, NameError):
                                    pass
                                
                                # Clear wake word buffer (for consistency, even though OpenWakeWord doesn't use it)
                                wake_word_buffer = []
                                
                                # Wait a moment before starting VAD (avoid wake word in transcription)
                                time.sleep(0.3)
                                
                                # Reset VAD state for fresh start
                                model_vad.reset_states()
                                
                                print("[Wake Word] 🎤 Listening for speech...")
                    except KeyboardInterrupt:
                        # Allow clean exit on Ctrl+C
                        raise
                    except PortAudioError as pa_error:
                        # Stream error - stream may be invalid, exit entire loop
                        error_code = getattr(pa_error, 'errno', None)
                        if error_code in [-9999, -9988]:  # Invalid stream or host error
                            print(f"\n[Wake Word] ⚠️  Audio stream error: {pa_error}")
                            print("[Wake Word] 🔄 Stream invalid, exiting listener")
                            stream_valid = False
                            break  # Exit wake word loop
                        else:
                            # Other PortAudio errors - log and continue listening
                            print(f"[Wake Word] ⚠️  PortAudio error: {pa_error}")
                            wake_word_buffer = []
                            continue  # Continue to next iteration to keep listening
                    except Exception as e:
                        print(f"[Wake Word] ⚠️ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        wake_word_buffer = []
                        continue  # Continue to next iteration to keep listening
                
                # Exit main loop if stream became invalid during wake word detection
                if not stream_valid:
                    break
                
                # === STAGE 2: VAD + Speech Processing (only after wake word or if wake word disabled) ===
                # Only allow transcription if:
                # 1. Wake word is disabled in settings, OR
                # 2. Wake word was detected (listening_active = True)
                # If wake word is enabled but not detected, loop back to wake word detection
                if wake_word_enabled and not listening_active:
                    # Wake word enabled but not detected yet - continue listening for wake word
                    # But check if transcription is blocked first
                    if is_transcription_blocked():
                        time.sleep(0.1)
                    continue
                
                # Transcription is allowed (either wake word disabled, or wake word was detected)
                # But check if transcription is blocked (dialog open, etc.)
                if is_transcription_blocked():
                    time.sleep(0.1)
                    continue
                
                allow_transcription = (not wake_word_setting_enabled) or listening_active
                if allow_transcription:
                    buffer = []
                    silence_start = None
                    last_vad_reset = time.time()  # Track last VAD reset to prevent decay
                    
                    # === Wait for speech ===
                    while True:
                        # Check if transcription is blocked (dialog open or mic button pressed)
                        if is_transcription_blocked():
                            # Check every 100ms, but also check stream_valid to avoid infinite loop
                            if not stream_valid:
                                break
                            time.sleep(0.1)
                            continue
                        
                        if is_playing():
                            break
                        
                        # Ensure stream is still valid
                        if not stream_valid:
                            break
                        
                        try:
                            audio_block, _ = stream.read(FRAME_SIZE)
                        except PortAudioError as pa_error:
                            # Stream error - stream may be invalid
                            error_code = getattr(pa_error, 'errno', None)
                            if error_code in [-9999, -9988]:  # Invalid stream or host error
                                print(f"\n[Listener] ⚠️  Audio stream error: {pa_error}")
                                print("[Listener] 🔄 Stream invalid, exiting listener")
                                stream_valid = False
                                break
                            else:
                                print(f"\n[Listener] ⚠️  PortAudio error: {pa_error}")
                                time.sleep(0.1)
                                continue
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
                        
                        # Calculate audio features (no pre-gain)
                        features = calculate_audio_features(channel_audio)
                        
                        if wake_word_enabled:
                            print(f"[Wake Word Active] VAD {vad_prob:.2f} | RMS {features['rms']:.4f} | Peak {features['peak']:.3f}", end="\r")
                        else:
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
                            
                            # Speech detected - switch from solid red to pulsating red
                            try:
                                from gui.aura_gui import set_wake_word_detected
                                set_wake_word_detected(False)  # Clear wake word state
                            except (ImportError, NameError):
                                pass
                            set_transcribing(True)  # Start pulsating red LED
                            buffer.append(audio_block)
                            break
                    
                    # === Record speech ===
                    while True:
                        if is_playing():
                            set_transcribing(False)
                            break
                        
                        try:
                            audio_block, _ = stream.read(FRAME_SIZE)
                        except PortAudioError as pa_error:
                            # Stream error - stream may be invalid
                            error_code = getattr(pa_error, 'errno', None)
                            if error_code in [-9999, -9988]:  # Invalid stream or host error
                                print(f"\n[Listener] ⚠️  Audio stream error: {pa_error}")
                                print("[Listener] 🔄 Stream invalid, exiting listener")
                                stream_valid = False
                                set_transcribing(False)
                                break
                            else:
                                print(f"\n[Listener] ⚠️  PortAudio error: {pa_error}")
                                time.sleep(0.1)
                                continue
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
                        # Reset listening state for next wake word (if enabled)
                        if wake_word_enabled:
                            listening_active = False
                            print("[Wake Word] 🔄 Waiting for wake word...")
                        continue
                    
                    # Send to Whisper (initial filter already passed)
                    text = transcribe(mono)
                    
                    # Reset VAD state for next utterance (critical for consistent performance)
                    model_vad.reset_states()
                    
                    if text:
                        # Optional: Strip wake word from transcription if present
                        if wake_word_enabled:
                            # Remove common wake word phrases from start of text
                            text_lower = text.lower().strip()
                            wake_phrases = ["hey aura", "hey aura,", "hey aura.", "aura", "aura,", "hey jarvis", "hey jarvis,", "hey jarvis."]
                            for phrase in wake_phrases:
                                if text_lower.startswith(phrase):
                                    text = text[len(phrase):].strip().lstrip(",.")
                                    print(f"[Wake Word] 🧹 Removed wake word from transcription")
                                    break
                        
                        speak_llm_response(text)
                    
                    # Reset listening state for next wake word (if enabled) - ALWAYS reset, even if no text
                    if wake_word_enabled:
                        listening_active = False
                        try:
                            from gui.aura_gui import set_wake_word_activated, set_wake_word_detected
                            set_wake_word_activated(False)
                            set_wake_word_detected(False)  # Clear wake word LED state
                        except ImportError:
                            pass
                        print("[Wake Word] 🔄 Waiting for wake word...")
                    
                    set_transcribing(False)
    finally:
        # Cleanup wake word detector on exit
        if wake_word_detector:
            try:
                wake_word_detector.cleanup()
            except AttributeError:
                try:
                    wake_word_detector.release()
                except AttributeError:
                    pass
            print("[Wake Word] 🧹 Cleaned up wake word detector")

if __name__ == "__main__":
    try:
        listen()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
