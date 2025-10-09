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
# NOTE: Using HARDWARE AGC on ReSpeaker DSP (configured via tune_respeaker.py)
# Software AGC disabled - hardware is faster and more effective
USE_AUTO_GAIN = False  # Disabled - using hardware AGC instead

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
# NOTE: All processing now done in HARDWARE DSP
# No software processing needed!

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
        
        # Enable noise suppression for ASR
        tuning.write("STATNOISEONOFF_SR", 1)  # Stationary noise (fan, hum)
        tuning.write("NONSTATNOISEONOFF_SR", 1)  # Non-stationary noise (AC, etc)
        
        # Gentle over-subtraction
        tuning.write("GAMMA_NS_SR", 1.0)
        tuning.write("GAMMA_NN_SR", 1.1)
        
        # Disable echo cancellation (not needed)
        tuning.write("ECHOONOFF", 0)
        
        print("[Hardware] ✅ ReSpeaker DSP configured for far-field (8-16 feet)")
        
    except Exception as e:
        print(f"[Hardware] ⚠️  Could not configure ReSpeaker DSP: {e}")
        print(f"[Hardware] ℹ️  Proceeding with default settings...")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")
    
    # Auto-configure ReSpeaker hardware on startup
    configure_respeaker_hardware()
    
    # Show configuration
    print("\n" + "="*70)
    print("[Audio] ✅ Using HARDWARE DSP processing (ReSpeaker XMOS chip)")
    print("[Audio] 🔧 Hardware AGC + Noise Suppression (auto-configured on startup)")
    print("[Audio] ⚡ No software processing - clean audio from hardware")
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
            
            # Debug: Show audio stats (already processed by hardware DSP)
            if DEBUG_NOISE_REDUCTION:
                rms = np.sqrt(np.mean(mono_mix ** 2))
                peak = np.max(np.abs(mono_mix))
                print(f"\n[Audio] 📊 FROM HARDWARE DSP: RMS={rms:.6f}, Peak={peak:.4f}, Length={len(mono_mix)} samples")

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
