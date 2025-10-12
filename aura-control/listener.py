import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
import subprocess
from scipy.fft import rfft, rfftfreq
from speaker import speak_llm_response, is_playing
from aura_gui import set_transcribing

# === Config ===
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.3  # 300ms of silence before stopping
VAD_START_THRESHOLD = 0.25  # Lowered - beamforming provides good noise rejection
VAD_SILENCE_THRESHOLD = 0.15  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

# === Advanced Multi-Feature Speech Detection ===
# Enabled - Filters out low-energy noise bursts that trigger VAD
ENABLE_ADVANCED_FILTER = True

# Thresholds tuned from empirical testing
SPEECH_ZCR_MAX = 0.40           # Reject if ZCR > this
SPEECH_FLATNESS_MAX = 0.55      # Reject if too "flat" (noisy, not tonal)
SPEECH_CENTROID_MIN = 300       # Hz - reject if too low (rumble/fan)
SPEECH_CENTROID_MAX = 3000      # Hz - reject if too high (hiss)
SPEECH_BAND_MIN = 0.30          # Reject if insufficient energy in speech band
SPEECH_DURATION_MIN = 0.4       # Seconds - reject if too short (noise bursts)

# CRITICAL: Energy thresholds (most reliable discriminators)
SPEECH_RMS_MIN = 0.035          # Reject if RMS < this (noise is 0.018-0.026, speech is 0.097)
SPEECH_PEAK_MIN = 0.15          # Reject if peak < this (noise is 0.08-0.12, speech is 0.96)

# BARE-BONES: Hardware DSP → Channel 0 → VAD → Advanced Filter → Whisper

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
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
            return 6  # Always use 6 channels
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
    
    return features

def is_likely_speech(features, duration=None):
    """
    Apply multi-feature analysis to distinguish speech from noise
    
    Returns: (is_speech: bool, reason: str)
    """
    reasons = []
    
    # Check Energy Levels FIRST (most reliable)
    if features['rms'] < SPEECH_RMS_MIN:
        reasons.append(f"RMS too low ({features['rms']:.4f} < {SPEECH_RMS_MIN})")
    
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
    
    is_speech = len(reasons) == 0
    reason = " | ".join(reasons) if reasons else "All checks passed"
    
    return is_speech, reason

# === Load VAD ===
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)

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
    """Send raw audio to Whisper"""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
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
def play_welcome_prompt(stream):
    try:
        print("[Aura] 🔊 Playing welcome prompt...")
        stream.stop()
        subprocess.run(["aplay", WELCOME_AUDIO_PATH], check=False)
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
    global vad_zero_count
    
    channels = find_device_index()
    
    # Display current hardware configuration
    config = load_respeaker_config()
    display_hardware_config(config)
    
    # Warm up Whisper model (eliminates slow first transcription)
    warmup_whisper()
    
    print("\n" + "="*70)
    print("[Audio] BARE-BONES PIPELINE")
    print("[Audio]   Hardware DSP → Channel 0 → VAD → Whisper")
    print("[Audio]   (Configure: sudo python3 scripts/tune_respeaker.py [profile])")
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
                
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Calculate audio features
                features = calculate_audio_features(channel_0)
                
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
            
            # Final filter check on full audio (before sending to Whisper)
            if ENABLE_ADVANCED_FILTER:
                duration = len(mono) / SAMPLE_RATE
                full_features = calculate_audio_features(mono)
                is_speech_final, reason_final = is_likely_speech(full_features, duration)
                
                if not is_speech_final:
                    print(f"\n[Filter] ❌ Final check REJECTED: {reason_final}")
                    print(f"[Filter] RMS={full_features['rms']:.4f}, Peak={full_features['peak']:.3f}, Duration={duration:.2f}s\n")
                    set_transcribing(False)
                    model_vad.reset_states()
                    continue
                else:
                    print(f"[Filter] ✅ Final check passed")
            
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
