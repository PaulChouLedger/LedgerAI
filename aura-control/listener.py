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
SILENCE_TIMEOUT = 0.30
VAD_START_THRESHOLD = 0.2
VAD_SILENCE_THRESHOLD = 0.05
MIN_AUDIO_SAMPLES = 2000

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

# === Transcribe ===
def transcribe(audio):
    """Send raw audio to Whisper (matches find_optimal_rms.py recording)"""
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
    
    try:
        response = requests.post(
            "http://localhost:11434/chat",
            json={"prompt": text, "chat_id": "voice_session", "reset": False},
            timeout=60,
            stream=True
        )
        
        if response.status_code == 200:
            speak_llm_response(response)
        else:
            print(f"[LLM] ❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"[LLM] ❌ {e}")

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
    print("[Audio] RAW MODE - matches find_optimal_rms.py")
    print("[Audio] 6-channel input → channel 0 → raw to Whisper")
    print("="*70 + "\n")
    
    with sd.InputStream(device=DEVICE_INDEX, channels=channels, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        
        play_welcome_prompt(stream)
        
        while True:
            # Pause during TTS
            if is_playing():
                stream.stop()
                while is_playing():
                    time.sleep(0.1)
                stream.start()
                
                # Flush buffer
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
