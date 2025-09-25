import os
from io import BytesIO
from dotenv import load_dotenv
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

# === Load API Key ===
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")

client = ElevenLabs(api_key=api_key)

VOICE_ID = "iy0lEidUIpheWxyur2p8"
MODEL_ID = "eleven_monolingual_v1"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/LedgerAI/assets/voice_samples")

def save_tts_to_wav(text, filename="output.wav"):
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)

    print(f"[Aura/speaker] 💬 Synthesizing and saving to {output_path}...")

    stream = client.text_to_speech.stream(
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        text=text,
        output_format="mp3_44100_128",
        voice_settings={
            "stability": 0.3,
            "similarity_boost": 0.8,
            "style": 0.2,
            "speed": 1.0,
            "use_speaker_boost": True
        }
    )

    audio_bytes = b"".join(stream)
    segment = AudioSegment.from_mp3(BytesIO(audio_bytes))
    segment.export(output_path, format="wav")

    print(f"[Aura/speaker] ✅ File saved to: {output_path}")

# === Example Usage ===
if __name__ == "__main__":
    save_tts_to_wav("AuraVision is setting up, please wait while I initialize.", filename="startup_test.wav")
