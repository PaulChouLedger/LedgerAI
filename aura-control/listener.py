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
SILENCE_TIMEOUT = 0.25  # 300ms of silence before stopping
VAD_START_THRESHOLD = 0.25  # Higher = less sensitive to fan noise
VAD_SILENCE_THRESHOLD = 0.10  # Lower = more conservative about ending
MIN_AUDIO_SAMPLES = 8000  # Minimum samples to send to Whisper

# === Hardware & Software AGC Configuration ===
# 
# HARDWARE CONFIGURATION (via systemd service on boot):
#   The ReSpeaker hardware is configured by: /etc/systemd/system/respeaker-tuning.service
#   
#   To change hardware preset:
#     1. sudo nano /etc/systemd/system/respeaker-tuning.service
#     2. Edit line 28: Change "clean" to: clean/near_field/far_field/reset
#     3. sudo systemctl daemon-reload
#     4. sudo systemctl restart respeaker-tuning.service
#   
#   Available presets (see scripts/tune_respeaker.py):
#     - clean      : HPF + Optimal AGC (0.08 RMS) + Max NS (gamma=3.0) - DEFAULT
#     - near_field : HPF + moderate AGC (0.08 RMS) - for 1-6 feet
#     - far_field  : High AGC + noise suppression (0.03 RMS) - for 8-16 feet
#     - reset      : Factory defaults - all OFF
#
# SOFTWARE PROCESSING: DISABLED
# Using RAW audio from hardware DSP only
# Optimize hardware first, then add software processing if needed

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# No stream refresh - keeping it simple
# If issues occur, we'll debug the root cause instead of working around it

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

# === Hardware Configuration Display (No Permissions Needed) ===
def load_respeaker_config():
    """Load ReSpeaker configuration from state file (no USB permissions needed)"""
    import json
    from datetime import datetime
    
    config_file = os.path.expanduser("~/LedgerAI/data/respeaker_config.json")
    
    if not os.path.exists(config_file):
        return None
    
    try:
        with open(config_file, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"[Hardware] ⚠️  Error reading config: {e}")
        return None

def display_hardware_config(state):
    """Display hardware configuration from saved state"""
    if state is None:
        print("[Hardware] ℹ️  No configuration found")
        print("[Hardware] 💡 Run: sudo python3 scripts/tune_respeaker.py [preset]")
        return
    
    from datetime import datetime
    timestamp = datetime.fromtimestamp(state['timestamp'])
    config = state['config']
    preset = state['preset']
    
    print("\n" + "="*70)
    print(f"[Hardware] 📋 ReSPEAKER CONFIGURATION (Preset: {preset.upper()})")
    print(f"[Hardware] ⏱️  Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # AGC
    if config['AGCONOFF'] == 1:
        print(f"  AGC:                    ✅ ENABLED")
        print(f"    Target Level:         {config['AGCDESIREDLEVEL']:.2f}")
        print(f"    Max Gain:             {config['AGCMAXGAIN']:.1f} dB")
    else:
        print(f"  AGC:                    ❌ DISABLED")
    
    # High-pass Filter
    hpf_labels = ["OFF", "70 Hz", "125 Hz", "180 Hz"]
    hpf_val = config['HPFONOFF']
    hpf_label = hpf_labels[hpf_val] if hpf_val < len(hpf_labels) else str(hpf_val)
    print(f"  High-Pass Filter:       {hpf_label}")
    
    # Noise Suppression
    if config['STATNOISEONOFF_SR'] == 1:
        gamma = config.get('GAMMA_NS_SR', 1.0)
        print(f"  Stationary Noise Supp:  ✅ ENABLED (gamma={gamma:.1f})")
    else:
        print(f"  Stationary Noise Supp:  ❌ DISABLED")
    
    if config['NONSTATNOISEONOFF_SR'] == 1:
        print(f"  Non-Stat Noise Supp:    ✅ ENABLED")
    else:
        print(f"  Non-Stat Noise Supp:    ❌ DISABLED")
    
    # Echo
    if config['ECHOONOFF'] == 1:
        print(f"  Echo Cancellation:      ✅ ENABLED")
    else:
        print(f"  Echo Cancellation:      ❌ DISABLED")
    
    print("="*70 + "\n")

# === Software AGC === (DISABLED - Hardware optimization first)
def apply_software_agc(audio, target_rms=0.08):
    """
    Apply software AGC by normalizing audio to target RMS level
    Simple gain adjustment - not adaptive like hardware AGC
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 0.001:  # Avoid division by zero for silence
        return audio
    
    # Calculate required gain
    gain = target_rms / current_rms
    
    # Limit gain to prevent excessive amplification
    gain = min(gain, 10.0)  # Max 20dB gain
    
    # Apply gain
    amplified = audio * gain
    
    # Soft clip to prevent harsh clipping
    amplified = np.tanh(amplified)
    
    new_rms = np.sqrt(np.mean(amplified ** 2))
    print(f"[Software AGC] 🎚️  RMS: {current_rms:.4f} → {new_rms:.4f} (gain={gain:.2f}x)")
    
    return amplified

# === Transcribe ===
def transcribe(audio):
    """Send raw audio to Whisper"""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"[Audio] RMS={rms:.6f}, Peak={peak:.4f}, Duration={audio_duration:.2f}s")
    
    # Track token usage for transcription (based on audio duration)
    try:
        from wallet_integration import get_usage_tracker
        tracker = get_usage_tracker()
        tracker.record_usage('transcription', multiplier=audio_duration)
    except Exception as e:
        print(f"[TokenUsage] ⚠️ Failed to track transcription usage: {e}")
    
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
        stream.stop()
        subprocess.run(["aplay", WELCOME_AUDIO_PATH], check=False)
        stream.start()
        
        # Brief buffer flush
        for _ in range(3):
            try:
                stream.read(FRAME_SIZE)
            except:
                break
        
        try:
            from aura_gui import set_setup_complete, set_welcome_played, set_listening_ready
            set_setup_complete()
            set_welcome_played()
            set_listening_ready()
        except ImportError:
            pass
    except Exception as e:
        print(f"[Aura] ❌ Welcome prompt error: {e}")

# === Main Loop ===
def listen():
    
    channels = find_device_index()
    
    # Display hardware configuration from saved state (no permissions needed)
    config_state = load_respeaker_config()
    display_hardware_config(config_state)
    
    print("\n" + "="*70)
    print("[Audio] SIGNAL PROCESSING PIPELINE:")
    print("[Audio]   1. Hardware DSP (HPF + AGC + NS via tune_respeaker.py)")
    print("[Audio]   2. Channel 0 selection (6ch → 1ch)")
    print("[Audio]   3. VAD → Whisper")
    print("[Audio] ℹ️  RAW audio from hardware - no software processing")
    print("[Audio] ℹ️  Optimize hardware settings first!")
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
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                # Brief buffer flush
                for _ in range(5):
                    try:
                        stream.read(FRAME_SIZE)
                    except:
                        break
            
            buffer = []
            silence_start = None
            
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
            
            # RAW audio from hardware DSP (HPF + AGC + NS already applied)
            # NO software processing - testing hardware optimization only
            
            # Check if audio is too short
            if len(mono) < MIN_AUDIO_SAMPLES:
                print("⚠️  Too short\n")
                set_transcribing(False)
                continue
            
            # Send RAW hardware audio to Whisper (no RMS filtering)
            # Let Whisper handle low-level audio - hardware AGC should boost it anyway
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
