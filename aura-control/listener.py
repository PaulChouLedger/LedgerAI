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
FRAME_SIZE = int(SAMPLE_RATE * 0.032)
SILENCE_TIMEOUT = 0.3  # 300ms of silence before stopping
VAD_START_THRESHOLD = 0.35  # Higher = less sensitive to fan noise
VAD_SILENCE_THRESHOLD = 0.05  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 2000

# BARE-BONES: No software processing - testing hardware optimization only

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Smart freeze detection - refresh when VAD appears stuck
# Detects: VAD=0.00 continuously with non-zero RMS (frozen state)
# Action: Immediate refresh when freeze detected (not timer-based)
# Benefits: 
#   - No missed speech during refresh
#   - Only refreshes when actually frozen
#   - ~1 second detection time (30 frames × 32ms = 0.96s)
VAD_FREEZE_THRESHOLD = 30  # Consecutive 0.00 frames with RMS > 0.01
vad_zero_count = 0

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
    
    print("="*70)
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
            vad_zero_count = 0  # Reset freeze counter for each new listening session
            
            # === Wait for speech ===
            while True:
                if is_playing():
                    break
                
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Stream error: {e}")
                    time.sleep(0.1)
                    continue
                
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                # Hardware HPF already applied in ReSpeaker DSP
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                rms = np.sqrt(np.mean(channel_0 ** 2))
                
                # Smart freeze detection: VAD stuck at 0.00 with non-zero RMS
                if vad_prob < 0.01 and rms > 0.01:  # VAD frozen
                    vad_zero_count += 1
                    if vad_zero_count >= VAD_FREEZE_THRESHOLD:
                        print(f"\n[Listener] ⚠️  VAD appears frozen (0.00 for {vad_zero_count} frames with RMS={rms:.4f})")
                        print("[Listener] 🔄 Refreshing stream to unfreeze...")
                        stream.stop()
                        time.sleep(0.1)
                        stream.start()
                        # Flush buffer
                        for _ in range(3):
                            try:
                                stream.read(FRAME_SIZE)
                            except:
                                break
                        vad_zero_count = 0
                        print("[Listener] ✅ Stream refreshed")
                        continue
                else:
                    vad_zero_count = 0  # Reset counter if VAD responds
                
                print(f"[VAD] {vad_prob:.2f} | RMS {rms:.6f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (VAD={vad_prob:.2f}, RMS={rms:.6f})")
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
                continue
            
            text = transcribe(mono)
            
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
