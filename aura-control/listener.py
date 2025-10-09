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
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.20
VAD_START_THRESHOLD = 0.25
VAD_SILENCE_THRESHOLD = 0.10
MIN_AUDIO_SAMPLES = 4000

# Auto Gain Control (AGC)
# Hardware AGC on ReSpeaker handles initial processing
# Software AGC boosts to consistent level (hardware may not reach target at far-field)
USE_SOFTWARE_AGC = True  # Light boost on top of hardware processing
AGC_TARGET_RMS = 0.20  # Target RMS for Whisper
AGC_MAX_GAIN = 5.0  # Light boost only (hardware already did most work)

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Debug
DEBUG_AUDIO_LEVELS = True
DEBUG_NOISE_REDUCTION = True

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# === Detect correct mic index ===
def find_device_index():
    global DEVICE_INDEX
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            DEVICE_INDEX = i
            print(f"[Aura/listener] 🎧 Using input device: {device['name']} (index {i})")
            return
    raise RuntimeError("Microphone not found. Check DEVICE_NAME.")

# === Load Silero VAD ===
model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

# === Audio Processing Functions ===

def software_agc_boost(audio):
    """
    Light software AGC boost on top of hardware processing
    - Hardware DSP does most of the work
    - Software ensures consistent 0.10 RMS for Whisper
    - Max 5x gain (just a light boost)
    """
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-6:
        return audio, 1.0
    
    # Calculate gain needed to reach target
    required_gain = AGC_TARGET_RMS / rms
    
    # Limit to max gain (light boost only)
    actual_gain = min(required_gain, AGC_MAX_GAIN)
    
    # Apply gain with clipping prevention
    audio = audio * actual_gain
    audio = np.clip(audio, -0.95, 0.95)
    
    return audio, actual_gain

# === Simple frequency function for GUI border (placeholder) ===
def get_transcription_frequency():
    """Return default frequency for GUI border pulsation"""
    return 0.7

# === Transcribe with Whisper container ===
def transcribe(audio):
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
        print(f"📝 Transcription: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# === Welcome prompt (pause mic before playing) ===
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
    """
    Auto-configure ReSpeaker hardware DSP for far-field speech recognition
    This runs on every startup to ensure settings are applied
    """
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
        
        # Initialize tuning with device
        tuning = Tuning(dev)
        
        print("[Hardware] 🔧 Configuring ReSpeaker DSP for far-field...")
        
        # Disable hardware high-pass (we found it hurts far-field)
        tuning.write("HPFONOFF", 0)
        
        # Enable hardware AGC
        tuning.write("AGCONOFF", 1)
        tuning.write("AGCDESIREDLEVEL", 0.10)  # 0.10 RMS = -10 dBov
        tuning.write("AGCMAXGAIN", 31.6)  # 30 dB = 31.6x
        
        # DISABLE noise suppression (test if it's hurting far-field recognition)
        tuning.write("STATNOISEONOFF_SR", 0)  # Stationary noise OFF
        tuning.write("NONSTATNOISEONOFF_SR", 0)  # Non-stationary noise OFF
        
        # Disable echo cancellation (not needed)
        tuning.write("ECHOONOFF", 0)
        
        print("[Hardware] ✅ ReSpeaker DSP configured for far-field (8-16 feet)")
        
    except Exception as e:
        print(f"[Hardware] ⚠️  Could not configure ReSpeaker DSP: {e}")
        if "Access denied" in str(e) or "insufficient permissions" in str(e):
            print(f"[Hardware] 💡 USB permissions needed. Run once with sudo:")
            print(f"[Hardware]    sudo python3 scripts/tune_respeaker.py far_field")
            print(f"[Hardware]    (Settings persist until USB unplug/replug)")
        print(f"[Hardware] ℹ️  Proceeding with current hardware settings...")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")
    
    # Auto-configure ReSpeaker hardware on startup
    configure_respeaker_hardware()
    
    # Show configuration
    print("\n" + "="*70)
    print("[Audio] ✅ Hybrid processing: Hardware AGC + Software Boost")
    print("[Audio] 🔧 Hardware: AGC only (NO noise suppression - preserves speech)")
    print(f"[Audio] 🔧 Software: Light boost (max {AGC_MAX_GAIN}x) to ensure {AGC_TARGET_RMS} RMS")
    print("[Audio] 💡 Clean amplification, no aggressive filtering")
    print("="*70 + "\n")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # Play welcome.wav
        play_welcome_prompt(stream)

        while True:
            if is_playing():
                print("[Listener] ⏸️ Pausing mic during playback")
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                print("[Listener] ▶️ Mic resumed after playback")

            buffer = []
            silence_start = None

            # === Wait for speech ===
            while True:
                if is_playing():
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                
                # Run VAD on raw audio
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Debug: Show audio levels
                if DEBUG_AUDIO_LEVELS:
                    rms = np.sqrt(np.mean(channel_0 ** 2))
                    print(f"[Debug] VAD: {vad_prob:.2f}, RMS: {rms:.4f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    set_transcribing(True)
                    buffer.append(audio_block)
                    break

            # === Continue recording ===
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                buffer.append(audio_block)
                
                # Run VAD for silence detection
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n⏹️ Speech ended (VAD silence: {vad_prob:.2f} < {VAD_SILENCE_THRESHOLD}). Processing...")
                        set_transcribing(False)
                        break
                else:
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Debug: Show audio stats from hardware DSP
            if DEBUG_NOISE_REDUCTION:
                hw_rms = np.sqrt(np.mean(mono_mix ** 2))
                hw_peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 FROM HARDWARE DSP: RMS={hw_rms:.6f}, Peak={hw_peak:.4f}, Length={len(mono_mix)} samples")
            
            # Apply light software AGC boost (ensures consistent 0.10 RMS)
            if USE_SOFTWARE_AGC:
                mono_mix, sw_gain = software_agc_boost(mono_mix)
                if DEBUG_NOISE_REDUCTION:
                    final_rms = np.sqrt(np.mean(mono_mix ** 2))
                    final_peak = np.max(np.abs(mono_mix))
                    print(f"[Audio] 📢 SOFTWARE BOOST: RMS={final_rms:.6f}, Peak={final_peak:.4f}, Gain={sw_gain:.2f}x")
                    print(f"[Audio] 📈 TOTAL: {hw_rms:.6f} → {final_rms:.6f} (×{final_rms/hw_rms if hw_rms > 0 else 0:.2f})")

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)
                continue

            text = transcribe(mono_mix)
            
            if not text:
                set_transcribing(False)
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
