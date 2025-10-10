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
MIN_SPEECH_RMS = 0.008  # Filter out low-level noise (more permissive for AGC)

# === AGC Testing Configuration ===
# Enable/disable hardware AGC (in ReSpeaker DSP chip)
USE_HARDWARE_AGC = True
HARDWARE_AGC_TARGET = 0.08  # Target RMS level (0.01-0.99)
HARDWARE_AGC_MAX_GAIN = 30.0  # Maximum gain in dB

# Enable/disable software AGC (in Python after audio capture)
USE_SOFTWARE_AGC = False
SOFTWARE_AGC_TARGET = 0.1  # Target RMS level for normalization

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Stream refresh interval (prevents idle staleness)
STREAM_REFRESH_INTERVAL = 30.0  # Refresh stream after 30s idle
last_activity_time = 0

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

# === Configure ReSpeaker Hardware ===
def configure_respeaker_hardware():
    """Configure ReSpeaker: Hardware HPF and optional AGC"""
    try:
        import sys
        import usb.core
        tuning_path = os.path.expanduser('~/usb_4_mic_array')
        if tuning_path not in sys.path:
            sys.path.insert(0, tuning_path)
        
        from tuning import Tuning
        
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            print("[Hardware] ⚠️  ReSpeaker not found")
            return
        
        tuning = Tuning(dev)
        
        print("[Hardware] 🔧 Configuring ReSpeaker...")
        
        # Enable hardware high-pass filter (removes noise before amplification)
        tuning.write("HPFONOFF", 1)  # Enable hardware HPF
        
        # Disable aggressive noise suppression (causes artifacts)
        tuning.write("STATNOISEONOFF_SR", 0)
        tuning.write("NONSTATNOISEONOFF_SR", 0)
        tuning.write("ECHOONOFF", 0)
        
        # Configure AGC based on flag
        if USE_HARDWARE_AGC:
            tuning.write("AGCONOFF", 1)
            tuning.write("AGCDESIREDLEVEL", HARDWARE_AGC_TARGET)
            tuning.write("AGCMAXGAIN", HARDWARE_AGC_MAX_GAIN)
            print(f"[Hardware] ✅ HPF + AGC enabled (target={HARDWARE_AGC_TARGET}, max_gain={HARDWARE_AGC_MAX_GAIN}dB)")
        else:
            tuning.write("AGCONOFF", 0)
            print("[Hardware] ✅ HPF enabled, AGC disabled")
        
    except Exception as e:
        print(f"[Hardware] ⚠️  Configuration failed: {e}")
        print(f"[Hardware] 💡 Continuing without hardware configuration...")

# === Software AGC ===
def apply_software_agc(audio, target_rms=SOFTWARE_AGC_TARGET):
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
    global last_activity_time
    last_activity_time = time.time()  # Initialize activity timer
    
    channels = find_device_index()
    
    # Configure hardware HPF → AGC pipeline (filter BEFORE amplify)
    configure_respeaker_hardware()
    
    print("\n" + "="*70)
    print("[Audio] SIGNAL PROCESSING PIPELINE:")
    print("[Audio]   1. Hardware HPF in DSP (removes low-freq noise)")
    
    # Show AGC configuration
    if USE_HARDWARE_AGC and USE_SOFTWARE_AGC:
        print("[Audio]   2. Hardware AGC in DSP ⚠️  BOTH AGC ENABLED - NOT RECOMMENDED")
        print(f"[Audio]      - Hardware: target={HARDWARE_AGC_TARGET}, max={HARDWARE_AGC_MAX_GAIN}dB")
        print(f"[Audio]   3. Software AGC in Python (target={SOFTWARE_AGC_TARGET})")
        print("[Audio]   4. Channel 0 selection")
        print("[Audio]   5. VAD → Whisper")
    elif USE_HARDWARE_AGC:
        print(f"[Audio]   2. Hardware AGC in DSP (target={HARDWARE_AGC_TARGET}, max={HARDWARE_AGC_MAX_GAIN}dB)")
        print("[Audio]   3. Channel 0 selection")
        print("[Audio]   4. VAD → Whisper")
        print("[Audio] ✅ Using HARDWARE AGC")
    elif USE_SOFTWARE_AGC:
        print("[Audio]   2. Channel 0 selection (NO hardware AGC)")
        print(f"[Audio]   3. Software AGC in Python (target={SOFTWARE_AGC_TARGET})")
        print("[Audio]   4. VAD → Whisper")
        print("[Audio] ✅ Using SOFTWARE AGC")
    else:
        print("[Audio]   2. Channel 0 selection")
        print("[Audio]   3. VAD → Whisper")
        print("[Audio] ⚠️  NO AGC ENABLED - audio may be too quiet")
    
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
                last_activity_time = time.time()  # Reset idle timer after playback
            
            buffer = []
            silence_start = None
            
            # === Wait for speech ===
            while True:
                if is_playing():
                    break
                
                # Check if stream needs refresh after long idle
                idle_time = time.time() - last_activity_time
                if idle_time > STREAM_REFRESH_INTERVAL:
                    print(f"\n[Listener] 🔄 Refreshing stream after {idle_time:.0f}s idle...")
                    stream.stop()
                    time.sleep(0.1)
                    stream.start()
                    # Flush stale buffer
                    for _ in range(3):
                        try:
                            stream.read(FRAME_SIZE)
                        except:
                            break
                    print("[Listener] ✅ Stream refreshed")
                    last_activity_time = time.time()
                
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"\n[Listener] ⚠️  Error: {e}")
                    # Try to recover by refreshing stream
                    try:
                        stream.stop()
                        time.sleep(0.1)
                        stream.start()
                        print("[Listener] 🔄 Stream restarted after error")
                        last_activity_time = time.time()
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
            
            # Hardware HPF already applied in ReSpeaker DSP (before AGC)
            mono_filtered = mono
            
            # Apply software AGC if enabled
            if USE_SOFTWARE_AGC:
                mono_filtered = apply_software_agc(mono_filtered, target_rms=SOFTWARE_AGC_TARGET)
            
            # Check if audio RMS is too low (likely fan noise, not speech)
            rms = np.sqrt(np.mean(mono_filtered ** 2))
            if rms < MIN_SPEECH_RMS:
                print(f"⚠️  RMS too low ({rms:.4f} < {MIN_SPEECH_RMS}), likely noise - skipping\n")
                set_transcribing(False)
                continue
            
            if len(mono_filtered) < MIN_AUDIO_SAMPLES:
                print("⚠️  Too short\n")
                continue
            
            text = transcribe(mono_filtered)
            
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
