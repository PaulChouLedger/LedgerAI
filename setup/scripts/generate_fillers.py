# generate_fillers.py — Create filler phrases using ElevenLabs (v1.2.1 compatible)
import os
from elevenlabs import generate, save, Voice, VoiceSettings
from dotenv import load_dotenv

# Load credentials
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE_ID")

# Filler phrases
phrases = [
    "Let me think about that...",
    "Hmm, good question.",
    "Just a second...",
    "Interesting...",
    "One moment, I’m thinking.",
    "Let’s see...",
    "Alright...",
    "Thinking about that now...",
    "Okay, processing that...",
    "Give me a second to find the best answer."
]

os.makedirs("data/fillers", exist_ok=True)

# Generate and save
for i, text in enumerate(phrases):
    try:
        print(f"🎙️ Generating: {text}")
        audio = generate(
            text=text,
            voice=Voice(voice_id=voice_id),
            model="eleven_monolingual_v1",
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        )
        save(audio, f"data/fillers/filler_{i+1}.mp3")
        print(f"✅ Saved: filler_{i+1}.mp3")
    except Exception as e:
        print(f"❌ Error generating \"{text}\": {e}")
