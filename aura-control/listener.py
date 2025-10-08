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
from pydub import AudioSegment
from aura_gui import set_transcribing
from scipy import signal

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_TIMEOUT = 0.2  # Time of continuous silence to stop - balance between responsiveness and cutoff prevention
VAD_START_THRESHOLD = 0.25  # Threshold to START detecting speech
VAD_SILENCE_THRESHOLD = 0.10  # Threshold for silence (closer to actual silence ~0.05, minimizes dead air to Whisper)
MIN_AUDIO_SAMPLES = 4000  # Reduced from 8000 to allow shorter utterances

# Audio processing
AUDIO_GAIN = 2.0  # Simple gain multiplier
ENABLE_NOISE_REDUCTION = True  # Enable spectral noise subtraction

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

# Debug: Show audio levels to help diagnose mic issues
DEBUG_AUDIO_LEVELS = True  # Set to False to disable

WELCOME_AUDIO_PATH = os.path.expanduser("~/LedgerAI/assets/voice_samples/audio1.wav")

# Global noise profile (learned continuously during startup)
noise_profile = None
noise_samples = []  # Accumulate noise samples from GUI load until welcome prompt
is_sampling_noise = False  # Flag to control noise sampling

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

def bandpass_filter(audio, lowcut=80, highcut=8000, order=5):
    """
    Apply bandpass filter to focus on human voice frequencies
    - Removes low-frequency fan rumble (< 80 Hz)
    - Removes high-frequency hiss (> 8000 Hz)
    """
    nyquist = SAMPLE_RATE / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Bandpass filter failed: {e}")
        return audio

def spectral_noise_subtraction(audio, noise_profile, strength=0.5):
    """
    Subtract noise spectrum from audio using spectral subtraction
    - noise_profile: FFT of background noise
    - strength: how much noise to subtract (0.0-1.0)
    """
    if noise_profile is None:
        return audio
    
    try:
        # Compute FFT of signal
        fft_signal = np.fft.rfft(audio)
        magnitude = np.abs(fft_signal)
        phase = np.angle(fft_signal)
        
        # Subtract noise profile from magnitude
        magnitude_clean = np.maximum(magnitude - strength * noise_profile, 0)
        
        # Reconstruct signal with cleaned magnitude
        fft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = np.fft.irfft(fft_clean, n=len(audio))
        
        return audio_clean
    except Exception as e:
        print(f"[Audio] ⚠️ Spectral subtraction failed: {e}")
        return audio

def start_noise_sampling():
    """
    Start accumulating background noise samples
    Called as soon as the listener starts (GUI loaded)
    """
    global is_sampling_noise, noise_samples
    noise_samples = []
    is_sampling_noise = True
    print("[Audio] 🔇 Started continuous noise sampling (accumulating until welcome prompt)...")
    print("[Audio] 💡 Fan noise pattern is being learned in the background...")

def stop_noise_sampling_and_compute_profile():
    """
    Stop sampling and compute the final noise profile from all accumulated samples
    Called right before playing the welcome prompt
    """
    global is_sampling_noise, noise_profile, noise_samples
    
    is_sampling_noise = False
    
    if len(noise_samples) == 0:
        print("[Audio] ⚠️ No noise samples collected - noise reduction disabled")
        noise_profile = None
        return
    
    # Concatenate all accumulated noise samples
    noise_audio = np.concatenate(noise_samples)
    
    duration = len(noise_audio) / SAMPLE_RATE
    print(f"[Audio] 🔧 Computing noise profile from {duration:.1f}s of accumulated samples...")
    
    # Compute average noise spectrum (FFT)
    noise_fft = np.fft.rfft(noise_audio)
    noise_magnitude = np.abs(noise_fft)
    
    noise_profile = noise_magnitude
    
    # Clear samples to free memory
    noise_samples = []
    
    print(f"[Audio] ✅ Noise profile captured: {len(noise_magnitude)} frequency bins from {duration:.1f}s of data")
    print(f"[Audio] 🎯 Noise reduction ready (strength=0.5)")

def process_audio(audio):
    """
    Full audio processing pipeline:
    1. Bandpass filter (remove fan rumble and high-freq hiss)
    2. Spectral noise subtraction (remove learned fan noise)
    3. Gain normalization
    """
    # Step 1: Bandpass filter (removes extreme frequencies)
    audio = bandpass_filter(audio, lowcut=80, highcut=8000)
    
    # Step 2: Spectral noise subtraction (removes learned noise pattern)
    if ENABLE_NOISE_REDUCTION and noise_profile is not None:
        audio = spectral_noise_subtraction(audio, noise_profile, strength=0.5)
    
    # Step 3: Apply gain
    audio = np.clip(audio * AUDIO_GAIN, -1.0, 1.0)
    
    return audio

# === Simple Audio Gain (kept for backward compatibility) ===
def apply_gain(audio):
    """Apply simple gain to audio signal"""
    return np.clip(audio * AUDIO_GAIN, -1.0, 1.0)

# === Simple frequency function for GUI border (placeholder) ===
def get_transcription_frequency():
    """Return default frequency for GUI border pulsation"""
    return 0.7  # Moderate pulsation speed

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
        # Step 1: Stop noise sampling and compute final noise profile
        if ENABLE_NOISE_REDUCTION:
            stop_noise_sampling_and_compute_profile()
        
        # Step 2: Play welcome prompt
        print("[Aura] 🔊 Playing welcome prompt...")
        stream.stop()
        subprocess.run(["aplay", WELCOME_AUDIO_PATH])
        time.sleep(0.25)
        stream.start()
        print("[Aura] 🎤 Mic resumed after welcome prompt")
        
        # Signal GUI that setup is complete, welcome is done, and listener is ready
        try:
            from aura_gui import set_setup_complete, set_welcome_played, set_listening_ready
            set_setup_complete()   # Mark setup as complete
            set_welcome_played()   # Welcome prompt finished
            set_listening_ready()  # Listener is ready
            print("[Aura] ✅ Setup complete, listener ready - aura eye now static")
        except ImportError:
            pass
    except Exception as e:
        print(f"[Aura] ❌ Failed to play welcome prompt: {e}")

# === Main Loop ===
def listen():
    find_device_index()
    print("🎤 Listening (6-channel input, VAD on channel 0)...")

    with sd.InputStream(device=DEVICE_INDEX, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        # ✅ Start noise sampling as soon as stream is ready (GUI loaded)
        if ENABLE_NOISE_REDUCTION:
            start_noise_sampling()
        
        # ✅ Play welcome.wav before entering listening loop
        # (This will stop noise sampling and compute the profile)
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
                
                # Accumulate noise samples if we're still sampling background noise
                if is_sampling_noise:
                    noise_samples.append(channel_0.copy())
                
                # Run VAD on raw audio (no processing)
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Debug: Show audio levels
                if DEBUG_AUDIO_LEVELS:
                    rms = np.sqrt(np.mean(channel_0 ** 2))
                    print(f"[Debug] VAD: {vad_prob:.2f}, RMS: {rms:.4f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started (prob={vad_prob:.2f})")
                    set_transcribing(True)  # Notify GUI: transcription started
                    buffer.append(audio_block)  # Store original audio
                    break

            # === Continue recording ===
            # Use VAD with LOWER threshold for silence detection
            # This allows continuous recording through quiet speech/distance variations
            while True:
                if is_playing():
                    print("[Listener] ⏸️ Pausing mic during playback")
                    set_transcribing(False)  # Reset transcribing state if interrupted
                    break

                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                buffer.append(audio_block)  # Store original audio
                
                # Run VAD on raw audio for silence detection
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                # Check if VAD indicates silence (below LOWER threshold)
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n⏹️ Speech ended (VAD silence: {vad_prob:.2f} < {VAD_SILENCE_THRESHOLD}). Processing...")
                        set_transcribing(False)  # Notify GUI: transcription ended
                        break
                else:
                    # Reset silence timer if VAD detects any speech
                    silence_start = None
                print(".", end="", flush=True)

            if is_playing():
                set_transcribing(False)  # Reset transcribing state if interrupted
                continue

            full_audio = np.concatenate(buffer)
            mono_mix = full_audio[:, 0]
            
            # Apply full audio processing pipeline (bandpass + noise reduction + gain)
            mono_mix = process_audio(mono_mix)

            if len(mono_mix) < MIN_AUDIO_SAMPLES:
                print("⚠️ Skipped: too short")
                set_transcribing(False)  # Reset transcribing state
                continue

            text = transcribe(mono_mix)
            if not text:
                set_transcribing(False)  # Reset transcribing state
                continue

            prompt_history.append(text)
            if len(prompt_history) > CONTEXT_DEPTH:
                prompt_history.pop(0)

            context = "\n".join(prompt_history[:-1])
            speak_llm_response(prompt=text, context=context)
