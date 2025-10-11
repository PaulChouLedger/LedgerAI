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

# Stream refresh interval (prevents idle staleness AND active buffer accumulation)
STREAM_REFRESH_INTERVAL = 15.0  # Refresh stream after 15s idle (more frequent = more responsive)
MAX_ACTIVE_TIME = 120.0  # Force refresh after 2 minutes even during active use
# This prevents: 1) Stale audio after idle, 2) Buffer accumulation during heavy use,
#                3) ALSA/hardware buffer issues, 4) VAD/PyTorch state accumulation,
#                5) Stationary NS losing learned noise profile
# After refresh: Quick warmup (10 frames ~0.3s) - VAD checks during warmup!
last_activity_time = 0
stream_start_time = 0  # Track when stream was last refreshed

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
        
        # Quick warmup: flush buffer and prime VAD
        print("[Aura] 🔧 Quick warmup...")
        for i in range(10):  # Quick flush (10 frames ~0.3s)
            try:
                audio_block, _ = stream.read(FRAME_SIZE)
                # Run VAD on first few frames to warm up PyTorch model
                if i < 3:
                    channel_0 = audio_block[:, 0]
                    if channel_0.size >= 512:
                        _ = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
            except:
                break
        print("[Aura] ✅ Ready")
        
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
    global last_activity_time, stream_start_time
    last_activity_time = time.time()  # Initialize activity timer
    stream_start_time = time.time()  # Initialize stream refresh timer
    
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
        
        # Initial warmup: Quick buffer flush and VAD prime
        print("\n[Listener] 🔧 Initial warmup: clearing buffers and priming VAD...")
        time.sleep(0.3)  # Let hardware settle after stream open
        for i in range(10):  # Quick flush (10 frames ~0.3s)
            try:
                audio_block, _ = stream.read(FRAME_SIZE)
                # Warm up VAD model on first few frames
                if i < 3:
                    channel_0 = audio_block[:, 0]
                    if channel_0.size >= 512:
                        _ = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
            except Exception as e:
                print(f"[Listener] ⚠️ Warmup frame {i} error: {e}")
                break
        print("[Listener] ✅ Hardware stabilized, VAD primed\n")
        
        play_welcome_prompt(stream)
        
        while True:
            # Pause during TTS
            if is_playing():
                print("[Listener] ⏸️ Pausing mic during playback")
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                
                # Flush buffer aggressively
                print("[Listener] 🧹 Flushing mic buffer...")
                for _ in range(10):  # More aggressive
                    try:
                        stream.read(FRAME_SIZE)
                    except:
                        break
                
                print("[Listener] ▶️ Mic resumed after playback (buffer flushed)")
                last_activity_time = time.time()  # Reset idle timer after playback
                stream_start_time = time.time()  # Reset active timer too
            
            buffer = []
            silence_start = None
            
            # === Wait for speech ===
            while True:
                if is_playing():
                    break
                
                # Check if stream needs refresh after long idle OR extended active use
                idle_time = time.time() - last_activity_time
                active_time = time.time() - stream_start_time
                
                if idle_time > STREAM_REFRESH_INTERVAL:
                    print(f"\n[Listener] 🔄 Refreshing stream after {idle_time:.0f}s idle...")
                    stream.stop()
                    time.sleep(0.5)  # Longer pause for hardware to stabilize
                    stream.start()
                    
                    # Quick warmup: Flush buffer and prime VAD
                    print("[Listener] 🔧 Quick warmup after idle...")
                    warmup_buffer = []
                    speech_detected_during_warmup = False
                    for i in range(10):  # Reduced from 20 to 10 (~0.3s)
                        try:
                            audio_block, _ = stream.read(FRAME_SIZE)
                            channel_0 = audio_block[:, 0]
                            
                            # Always check VAD during warmup to avoid missing speech
                            if channel_0.size >= 512:
                                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                                # If speech detected during warmup, save it!
                                if vad_prob > VAD_START_THRESHOLD and i >= 3:  # After first 3 frames
                                    print(f"[Listener] 🎤 Speech detected during warmup! (VAD={vad_prob:.2f})")
                                    warmup_buffer.append(audio_block)
                                    speech_detected_during_warmup = True
                                elif speech_detected_during_warmup:
                                    warmup_buffer.append(audio_block)
                        except:
                            break
                    
                    print("[Listener] ✅ Stream refreshed (idle)")
                    last_activity_time = time.time()
                    stream_start_time = time.time()
                    
                    # If speech was detected during warmup, add it to main buffer
                    if warmup_buffer:
                        print(f"[Listener] 📦 Adding {len(warmup_buffer)} warmup frames to buffer")
                        buffer.extend(warmup_buffer)
                        # Skip the "wait for speech" phase since we already have speech
                        if speech_detected_during_warmup:
                            print("[Listener] ⏩ Skipping wait - speech already captured")
                            break  # Exit wait loop, go to recording phase
                elif active_time > MAX_ACTIVE_TIME:
                    print(f"\n[Listener] 🔄 Force refreshing stream after {active_time:.0f}s active use...")
                    stream.stop()
                    time.sleep(0.5)  # Longer pause for full reset
                    stream.start()
                    
                    # Quick warmup: Flush buffer and prime VAD
                    print("[Listener] 🔧 Quick warmup after active use...")
                    warmup_buffer = []
                    speech_detected_during_warmup = False
                    for i in range(10):  # Reduced from 20 to 10 (~0.3s)
                        try:
                            audio_block, _ = stream.read(FRAME_SIZE)
                            channel_0 = audio_block[:, 0]
                            
                            # Always check VAD during warmup to avoid missing speech
                            if channel_0.size >= 512:
                                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                                # If speech detected during warmup, save it!
                                if vad_prob > VAD_START_THRESHOLD and i >= 3:  # After first 3 frames
                                    print(f"[Listener] 🎤 Speech detected during warmup! (VAD={vad_prob:.2f})")
                                    warmup_buffer.append(audio_block)
                                    speech_detected_during_warmup = True
                                elif speech_detected_during_warmup:
                                    warmup_buffer.append(audio_block)
                        except:
                            break
                    
                    print("[Listener] ✅ Stream force-refreshed")
                    stream_start_time = time.time()
                    
                    # If speech was detected during warmup, add it to main buffer
                    if warmup_buffer:
                        print(f"[Listener] 📦 Adding {len(warmup_buffer)} warmup frames to buffer")
                        buffer.extend(warmup_buffer)
                        # Skip the "wait for speech" phase since we already have speech
                        if speech_detected_during_warmup:
                            print("[Listener] ⏩ Skipping wait - speech already captured")
                            break  # Exit wait loop, go to recording phase
                
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Error: {e}")
                    # Try to recover by refreshing stream
                    try:
                        stream.stop()
                        time.sleep(0.1)
                        stream.start()
                        # Flush stale buffer
                        for _ in range(10):
                            try:
                                stream.read(FRAME_SIZE)
                            except:
                                break
                        print("[Listener] 🔄 Stream restarted after error")
                        last_activity_time = time.time()
                        stream_start_time = time.time()
                    except:
                        pass
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
                    last_activity_time = time.time()
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
                        last_activity_time = time.time()
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
            
            # Aggressive buffer cleanup after transcription
            del buffer, full_audio, mono
            import gc
            gc.collect()
            
            # Flush hardware buffer to prevent stale audio accumulation
            print("[Listener] 🧹 Flushing buffers after transcription...")
            for _ in range(10):  # More aggressive than before
                try:
                    stream.read(FRAME_SIZE)
                except:
                    break
            
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
