# scripts/generate_cached_prompts.py

import os
from io import BytesIO
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

# === Load API Key ===
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise EnvironmentError("ELEVENLABS_API_KEY not set in .env file.")

# === ElevenLabs Client Setup ===
client = ElevenLabs(api_key=api_key)
VOICE_ID = "iy0lEidUIpheWxyur2p8"
VOICE_MODEL = "eleven_monolingual_v1"

# === Output directory ===
PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/prompts"))
os.makedirs(PROMPT_DIR, exist_ok=True)

# === Prompts to generate ===
prompts = {
    "startup": "How you are doing well. AuraVision is setting up, give me a second while I initialize",
    "welcome": "Welcome to AuraVision, it's good to see you. Let's get started, shall we?"
}

# === Generate and save prompts ===
for name, text in prompts.items():
    print(f"Generating '{name}' prompt...")

    try:
        stream = client.text_to_speech.stream(
            voice_id=VOICE_ID,
            model_id=VOICE_MODEL,
            text=text,
            optimize_streaming_latency=1
        )

        audio_bytes = b"".join(stream)
        audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
        wav_path = os.path.join(PROMPT_DIR, f"{name}.wav")
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"❌ Failed to generate '{name}': {e}")

print("✅ Prompts saved to:", PROMPT_DIR)
