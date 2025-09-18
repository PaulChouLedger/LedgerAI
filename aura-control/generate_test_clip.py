import os
from elevenlabs.client import ElevenLabs

# === Config ===
api_key = os.getenv("ELEVEN_API_KEY")
voice_id = os.getenv("ELEVEN_VOICE_ID")
output_file = "test_clip.wav"
sample_text = "This is a test clip for calibrating microphone fingerprint suppression."

client = ElevenLabs(api_key=api_key)

# === Generate WAV from ElevenLabs ===
try:
    print("🔊 Requesting TTS clip from ElevenLabs...")

    # Request .wav format instead of raw PCM
    stream = client.text_to_speech.convert(
        text=sample_text,
        voice_id=voice_id,
        output_format="wav"  # ✅ WAV format instead of pcm_22050
    )

    with open(output_file, "wb") as f:
        for chunk in stream:
            f.write(chunk)

    print(f"✅ TTS clip saved to {output_file}")

except Exception as e:
    print(f"❌ Error generating TTS clip: {e}")
