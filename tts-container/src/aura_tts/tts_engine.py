import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY")
)
