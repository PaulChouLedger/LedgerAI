import os
import io
import time
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests

# === Config ===
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.3  # 300ms of silence before stopping
VAD_START_THRESHOLD = 0.35  # Higher = less sensitive to fan noise
VAD_SILENCE_THRESHOLD = 0.15  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

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
    """Send raw audio to Whisper"""
    global transcription_count, total_audio_duration
    
    duration = len(audio) / SAMPLE_RATE
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    print(f"\n[Audio] RMS={rms:.6f}, Peak={peak:.4f}, Duration={duration:.2f}s")
    
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
        print(f"[Stats] 🔢 Transcriptions: {transcription_count} | Total audio: {total_audio_duration:.1f}s")
        
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
                rms = np.sqrt(np.mean(channel_0 ** 2))
                
                print(f"[VAD] {vad_prob:.2f} | RMS {rms:.6f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (VAD={vad_prob:.2f}, RMS={rms:.6f})")
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
            
            # Print a separator and immediately loop back
            print("-" * 70)
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

