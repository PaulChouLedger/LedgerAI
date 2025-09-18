from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play, save  # or import from main module

client = ElevenLabs(api_key="sk_6c8c45407dc8286c8ca901d81543b380f45da249197e4889")


audio = client.text_to_speech.convert(
    voice_id="iy0lEidUIpheWxyur2p8",
    text="LedgeAI for ever!!! Area 31 lives in the shadows...we watch after you, but you won't see us.",
    model_id="eleven_monolingual_v1",  # or another applicable model
    output_format="mp3_44100_128",
    voice_settings={
        "stability": 0.3,
        "similarity_boost": 0.8,
        "style": 0.2,
        "speed": 1.0,
        "use_speaker_boost": True
    }
)

play(audio)  # or save(audio, "output.mp3")
