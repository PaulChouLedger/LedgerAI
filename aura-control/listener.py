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
SILENCE_TIMEOUT = 0.2  # 200ms of silence before stopping (responsive)
VAD_START_THRESHOLD = 0.2
VAD_SILENCE_THRESHOLD = 0.1  # Sensitive to detect speech continuation
MIN_AUDIO_SAMPLES = 2000

# Adaptive AGC (compress near-field only)
AGC_NEAR_FIELD_THRESHOLD = 0.02  # RMS above this = near-field (compress)
AGC_TARGET_RMS = 0.015  # Compress near-field down to this level
AGC_ENABLED = True

DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DEVICE_INDEX = None
CONTEXT_DEPTH = 6
prompt_history = []

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

# === Adaptive AGC ===
def adaptive_agc(audio):
    """
    Smart AGC: Compress near-field, leave far-field alone
    - Near-field (RMS > threshold): Compress down to prevent distortion
    - Far-field (RMS < threshold): Leave as-is (already accurate)
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 1e-6:
        return audio, 1.0
    
    # Only compress if too loud (near-field)
    if current_rms > AGC_NEAR_FIELD_THRESHOLD:
        # Reduce gain to bring down to target
        gain = AGC_TARGET_RMS / current_rms
        audio = audio * gain
        audio = np.clip(audio, -0.95, 0.95)
        return audio, gain
    
    # Far-field: leave as-is
    return audio, 1.0

# === Transcribe ===
def transcribe(audio):
    """Send audio to Whisper with adaptive AGC"""
    raw_rms = np.sqrt(np.mean(audio ** 2))
    raw_peak = np.max(np.abs(audio))
    print(f"[Audio] Raw: RMS={raw_rms:.6f}, Peak={raw_peak:.4f}, Duration={len(audio)/SAMPLE_RATE:.2f}s")
    
    # Apply adaptive AGC (compress near-field only)
    if AGC_ENABLED:
        audio, gain = adaptive_agc(audio)
    else:
        gain = 1.0
    
    final_rms = np.sqrt(np.mean(audio ** 2))
    final_peak = np.max(np.abs(audio))
    
    if gain < 1.0:
        print(f"[Audio] AGC: RMS={final_rms:.6f}, Peak={final_peak:.4f}, Gain={gain:.2f}x (near-field compression)")
    else:
        print(f"[Audio] No processing (far-field, already optimal)")
    
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
    channels = find_device_index()
    
    print("\n" + "="*70)
    if AGC_ENABLED:
        print("[Audio] ADAPTIVE AGC MODE")
        print(f"[Audio] Near-field (>{AGC_NEAR_FIELD_THRESHOLD} RMS): Compress to {AGC_TARGET_RMS}")
        print(f"[Audio] Far-field (<{AGC_NEAR_FIELD_THRESHOLD} RMS): Leave as-is")
    else:
        print("[Audio] RAW MODE - No AGC")
    print("="*70 + "\n")
    
    with sd.InputStream(device=DEVICE_INDEX, channels=channels, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        
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
            
            # === Wait for speech ===
            while True:
                if is_playing():
                    break
                
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                except Exception as e:
                    print(f"[Listener] ⚠️  Error: {e}")
                    time.sleep(0.1)
                    continue
                
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                rms = np.sqrt(np.mean(channel_0 ** 2))
                
                print(f"[VAD] {vad_prob:.2f} | RMS {rms:.6f}", end="\r")
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n[VAD] 🔊 Speech started")
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
            
            if len(mono) < MIN_AUDIO_SAMPLES:
                print("⚠️  Too short\n")
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
