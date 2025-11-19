#!/usr/bin/env python3
"""
Helper script to generate and play TTS audio for wake word training.
Usage: play_tts.py "text to speak"
"""
import os
import sys
import subprocess
from elevenlabs.client import ElevenLabs

if len(sys.argv) < 2:
    print("Usage: play_tts.py \"text to speak\"", file=sys.stderr)
    sys.exit(1)

text = sys.argv[1]
api_key = os.getenv('ELEVENLABS_API_KEY')
voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'default')

if not api_key:
    print("Error: ELEVENLABS_API_KEY not set", file=sys.stderr)
    sys.exit(1)

try:
    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        output_format="pcm_22050",
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.0,
            "style": 0.0,
            "use_speaker_boost": False
        }
    )
    
    # Play audio using aplay
    proc = subprocess.Popen(
        ["aplay", "-D", "plughw:0,0", "-f", "S16_LE", "-r", "22050", "-c", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    for chunk in audio_stream:
        if chunk:
            proc.stdin.write(chunk)
    
    proc.stdin.close()
    proc.wait()
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

