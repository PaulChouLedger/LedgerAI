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

def wait_for_rag_ready(timeout=30):
    """Wait for RAG system to be fully initialized"""
    print("[Listener] 🔧 Waiting for RAG system to initialize...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:11435/ready", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get('ready'):
                    print(f"[Listener] ✅ RAG system ready: {data.get('rag_components', {}).get('index_size', 0)} vectors, {data.get('rag_components', {}).get('chunks_loaded', 0)} chunks")
                    return True
        except:
            pass
        
        time.sleep(1)
        print(f"[Listener] ⏳ Waiting for RAG... ({int(time.time() - start_time)}s)")
    
    print(f"[Listener] ⚠️  RAG initialization timeout - proceeding anyway")
    return False

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.20
VAD_START_THRESHOLD = 0.3
VAD_SILENCE_THRESHOLD = 0.03  # Lower threshold - don't cut off trailing words (was 0.10)
MIN_AUDIO_SAMPLES = 4000
MIN_SPEECH_RMS = 0.010  # Minimum RMS to consider as speech (filter out noise/drift)

# Hardware AGC monitoring (prevents drift)
AGC_ENABLE_RUNTIME_RESET = False  # Requires sudo - use systemd service instead
AGC_KEEPALIVE_INTERVAL = 30.0  # Reset AGC every 30 seconds of idle
AGC_MIN_RMS_THRESHOLD = 0.015  # If frame RMS below this during speech, reset AGC immediately
AGC_RESET_COOLDOWN = 5.0  # Don't reset AGC more than once per 5 seconds
last_agc_reset_time = 0

# Auto Gain Control (AGC)
# Hardware AGC on ReSpeaker handles initial processing (gentle, no clipping)
# Software AGC boosts to optimal level for Whisper (does most of the work)
USE_SOFTWARE_AGC = True
AGC_TARGET_RMS = 0.20  # Target RMS for Whisper
AGC_MAX_GAIN = 10.0  # Maximum software boost (hardware outputs ~0.03, need ~7x)

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Global state
last_speech_time = 0
user_name = None  # Track user's name for Whisper initial_prompt

# Debug
DEBUG_AUDIO_LEVELS = True
DEBUG_NOISE_REDUCTION = True

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# === Detect correct mic index and channels ===
def find_device_index():
    global DEVICE_INDEX
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower():
            DEVICE_INDEX = i
            channels = device["max_input_channels"]
            print(f"[Aura/listener] 🎧 Using input device: {device['name']} (index {i})")
            print(f"[Aura/listener] 🎙️  Available input channels: {channels}")
            return channels
    raise RuntimeError("Microphone not found. Check DEVICE_NAME.")

# === Load Silero VAD ===
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

# === AGC Reset Helper ===
def reset_hardware_agc(reason=""):
    """Reset hardware AGC to prevent drift/sleep"""
    global last_agc_reset_time
    
    # Check if runtime reset is enabled
    if not AGC_ENABLE_RUNTIME_RESET:
        return False
    
    # Check cooldown
    if time.time() - last_agc_reset_time < AGC_RESET_COOLDOWN:
        return False
    
    try:
        import usb.core
        import sys
        tuning_path = os.path.expanduser('~/usb_4_mic_array')
        if tuning_path not in sys.path:
            sys.path.insert(0, tuning_path)
        from tuning import Tuning
        usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if usb_dev:
            tuning = Tuning(usb_dev)
            # Force reset: OFF → ON with settings
            tuning.write("AGCONOFF", 0)
            time.sleep(0.2)
            tuning.write("AGCONOFF", 1)
            tuning.write("AGCDESIREDLEVEL", 0.03)
            tuning.write("AGCMAXGAIN", 20.0)
            last_agc_reset_time = time.time()
            print(f"\n[Hardware] 🔄 AGC reset: {reason}")
            return True
    except Exception as e:
        if "Access denied" in str(e) or "insufficient permissions" in str(e):
            print(f"\n[Hardware] ⚠️  AGC reset failed: insufficient permissions")
            print(f"[Hardware] 💡 Install boot-time tuning service (run once):")
            print(f"[Hardware]    sudo bash scripts/install_auto_tune.sh")
        else:
            print(f"[Hardware] ❌ AGC reset failed: {e}")
        return False

# === Audio Processing Functions ===

def software_agc_boost(audio):
    """
    Software AGC - ensures consistent RMS for Whisper
    Works on top of hardware AGC processing
    """
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-6:
        return audio, 1.0
    
    # Calculate gain needed
    required_gain = AGC_TARGET_RMS / rms
    actual_gain = min(required_gain, AGC_MAX_GAIN)
    
    # Apply gain with clipping prevention
    audio = audio * actual_gain
    audio = np.clip(audio, -0.95, 0.95)
    
    return audio, actual_gain

# === Extract user's name from text ===
def extract_user_name(text):
    """Extract user's name from phrases like 'My name is Rafael'"""
    import re
    global user_name
    
    patterns = [
        r"my name is ([A-Z][a-z]+)",
        r"i'm ([A-Z][a-z]+)",
        r"i am ([A-Z][a-z]+)",
        r"this is ([A-Z][a-z]+)",
        r"call me ([A-Z][a-z]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            detected_name = match.group(1).capitalize()
            
            # Check RAG database for phonetically similar names to correct spelling
            corrected_name = correct_name_from_rag(detected_name)
            
            if corrected_name:
                name = corrected_name
                print(f"[Listener] 🔍 Name correction: '{detected_name}' → '{corrected_name}' (from RAG database)")
            else:
                name = detected_name
            
            if user_name != name:
                user_name = name
                print(f"[Listener] 👤 User name set: '{user_name}' (will guide Whisper spelling)")
            return name
    
    return None

def correct_name_from_rag(detected_name):
    """Check if detected name has a phonetically similar match in RAG database"""
    try:
        # Query RAG for person names in the database
        response = requests.get("http://localhost:11435/rag/names", timeout=2)
        if response.status_code == 200:
            known_names = response.json().get('names', [])
            
            # Use simple string similarity check (no metaphone needed on host)
            from difflib import SequenceMatcher
            
            best_match = None
            best_score = 0.0
            threshold = 0.65  # Same as RAG fuzzy matching
            
            for known_name in known_names:
                # Extract first name from full name (e.g., "Rafael Cabello" → "Rafael")
                first_name = known_name.split()[0] if ' ' in known_name else known_name
                
                # Calculate similarity
                similarity = SequenceMatcher(None, detected_name.lower(), first_name.lower()).ratio()
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = first_name
            
            return best_match
    except:
        pass
    
    # No correction found - use detected name as-is
    return None

# === Transcribe with Whisper container ===
def transcribe(audio):
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, SAMPLE_RATE, format="WAV")
    wav_io.seek(0)
    
    # Build custom initial_prompt if we know the user's name
    data = {}
    if user_name:
        # Guide Whisper to use the correct spelling
        data["initial_prompt"] = f"{user_name} is speaking. This is a medical conversation with proper names."
        print(f"[Whisper] 🎯 Using name guidance: '{user_name}'")
    
    try:
        response = requests.post(
            "http://localhost:5000/transcribe",
            files={"audio": ("speech.wav", wav_io, "audio/wav")},
            data=data,
            timeout=10
        )
        result = response.json()
        text = result["text"].get("text", "").strip() if isinstance(result["text"], dict) else result.get("text", "").strip()
        print(f"📝 Transcription: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# === Welcome prompt ===
def play_welcome_prompt(stream):
    try:
        print("[Aura] 🔊 Playing welcome prompt...")
        stream.stop()
        subprocess.run(["aplay", WELCOME_AUDIO_PATH])
        time.sleep(0.25)
        stream.start()
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

# === Configure ReSpeaker Hardware DSP ===
def configure_respeaker_hardware():
    """Auto-configure ReSpeaker hardware DSP (single channel firmware)"""
    try:
        import sys
        import usb.core
        tuning_path = os.path.expanduser('~/usb_4_mic_array')
        if tuning_path not in sys.path:
            sys.path.insert(0, tuning_path)
        
        from tuning import Tuning
        
        # Find ReSpeaker USB device
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            print("[Hardware] ⚠️  ReSpeaker USB device not found")
            return
        
        tuning = Tuning(dev)
        
        print("[Hardware] 🔧 Configuring ReSpeaker DSP (single channel)...")
        
        # Disable hardware high-pass (preserves all speech frequencies)
        tuning.write("HPFONOFF", 0)
        
        # Enable hardware AGC with conservative target (prevent clipping & drift)
        tuning.write("AGCONOFF", 1)
        tuning.write("AGCDESIREDLEVEL", 0.03)  # Gentle but stable - prevents clipping & drift
        tuning.write("AGCMAXGAIN", 20.0)  # 26 dB - conservative to prevent clipping
        
        # Disable all noise suppression (preserves speech quality)
        tuning.write("STATNOISEONOFF_SR", 0)
        tuning.write("NONSTATNOISEONOFF_SR", 0)
        tuning.write("ECHOONOFF", 0)
        
        print("[Hardware] ✅ ReSpeaker configured: AGC only, no filtering")
        
    except Exception as e:
        print(f"[Hardware] ⚠️  Could not configure DSP: {e}")
        if "Access denied" in str(e):
            print(f"[Hardware] 💡 Run: sudo bash scripts/setup_usb_permissions.sh")
        print(f"[Hardware] ℹ️  Proceeding with current settings...")

# === Main Loop ===
def listen():
    global last_speech_time
    last_speech_time = time.time()
    
    # Wait for RAG to be ready before starting
    wait_for_rag_ready()
    
    # Detect device and available channels
    available_channels = find_device_index()
    print(f"🎤 Listening ({available_channels}-channel hardware, using channel 0 only)...")
    
    # Note: Hardware configured by systemd service (respeaker-tuning.service)
    # No need to configure here - boot service handles it
    
    # Show configuration
    print("\n" + "="*70)
    print(f"[Audio] ✅ Single-Channel Processing ({available_channels}-ch firmware detected)")
    if available_channels == 1:
        print("[Audio] 🎉 Using single-channel firmware (cleaner signal!)")
    print("[Audio] 🔧 Hardware: Gentle AGC → ~0.03 RMS (prevents clipping & drift)")
    print(f"[Audio] 🔧 Software: Boost to {AGC_TARGET_RMS} RMS (max {AGC_MAX_GAIN}x, does main work)")
    print(f"[Audio] 🔧 Low-RMS Filter: Skips audio below {MIN_SPEECH_RMS} RMS (filters noise)")
    print("[Audio] 💡 Hardware tuned by systemd service (boot-time configuration)")
    print("="*70 + "\n")

    # Use detected channel count
    with sd.InputStream(device=DEVICE_INDEX, channels=available_channels, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        play_welcome_prompt(stream)

        while True:
            if is_playing():
                print("[Listener] ⏸️ Pausing mic during playback")
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                
                # Flush buffer - discard stale audio from before/during TTS playback
                print("[Listener] 🧹 Flushing mic buffer...")
                flush_frames = 5  # Discard ~160ms of audio
                for _ in range(flush_frames):
                    try:
                        stream.read(FRAME_SIZE)
                    except:
                        break
                
                print("[Listener] ▶️ Mic resumed after playback (buffer flushed)")

            buffer = []
            silence_start = None

            # === Wait for speech ===
            while True:
                if is_playing():
                    break
                
                # Check if AGC needs reset after long idle
                idle_time = time.time() - last_speech_time
                if idle_time > AGC_KEEPALIVE_INTERVAL:
                    reset_hardware_agc(f"idle for {idle_time:.0f}s")
                    last_speech_time = time.time()

                audio_block, _ = stream.read(FRAME_SIZE)
                
                # Extract channel 0 (handle both 1-ch and multi-ch firmware)
                if available_channels == 1:
                    channel_0 = audio_block.flatten()  # Ensure 1D array
                else:
                    channel_0 = audio_block[:, 0]  # Extract channel 0
                
                # Check if we have enough samples for VAD (minimum 512 samples)
                # VAD model requires: sr / samples > 31.25 → samples >= sr/31.25 = 512
                if channel_0.size < 512:
                    continue
                
                # Calculate RMS for debugging
                rms = np.sqrt(np.mean(channel_0 ** 2))
                # Note: Real-time AGC reset disabled (requires sudo)
                # Hardware configured by systemd service on boot
                
                # Run VAD
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if DEBUG_AUDIO_LEVELS:
                    print(f"[Debug] VAD: {vad_prob:.2f}, RMS: {rms:.4f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    set_transcribing(True)
                    buffer.append(audio_block)
                    last_speech_time = time.time()
                    break

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                
                # Extract channel 0 (handle both 1-ch and multi-ch firmware)
                if available_channels == 1:
                    channel_0 = audio_block.flatten()  # Ensure 1D array
                else:
                    channel_0 = audio_block[:, 0]  # Extract channel 0
                
                # Check if we have enough samples for VAD (minimum 512 samples)
                if channel_0.size < 512:
                    continue
                    
                buffer.append(audio_block)
                
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n⏹️ Speech ended (VAD silence: {vad_prob:.2f} < {VAD_SILENCE_THRESHOLD}). Processing...")
                        set_transcribing(False)
                        last_speech_time = time.time()
                        break
                else:
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)
                continue

            # Concatenate buffer and extract channel 0
            full_audio = np.concatenate(buffer)
            
            # Handle both 1-ch and multi-ch firmware
            if available_channels == 1:
                mono_mix = full_audio.flatten()  # Ensure 1D array
            else:
                mono_mix = full_audio[:, 0]  # Extract channel 0
            
            # Debug: Show hardware output and check for AGC drift
            if DEBUG_NOISE_REDUCTION:
                hw_rms = np.sqrt(np.mean(mono_mix ** 2))
                hw_peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 FROM HARDWARE: RMS={hw_rms:.6f}, Peak={hw_peak:.4f}, Length={len(mono_mix)} samples")
                
                # Check if audio is too quiet (AGC drift or noise, not speech)
                if hw_rms < MIN_SPEECH_RMS:
                    print(f"[Audio] ⚠️  RMS too low ({hw_rms:.6f} < {MIN_SPEECH_RMS}), skipping (likely noise/drift)")
                    print(f"[Audio] 💡 AGC may have drifted - restart listener or speak louder")
                    set_transcribing(False)
                    continue
            
            # Apply software AGC boost
            if USE_SOFTWARE_AGC:
                mono_mix, sw_gain = software_agc_boost(mono_mix)
                if DEBUG_NOISE_REDUCTION:
                    final_rms = np.sqrt(np.mean(mono_mix ** 2))
                    final_peak = np.max(np.abs(mono_mix))
                    print(f"[Audio] 📢 SOFTWARE BOOST: RMS={final_rms:.6f}, Peak={final_peak:.4f}, Gain={sw_gain:.2f}x")

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)
                continue

            text = transcribe(mono_mix)
            
            if not text:
                set_transcribing(False)
                continue
            
            # Extract user's name to guide future Whisper transcriptions
            extract_user_name(text)

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
