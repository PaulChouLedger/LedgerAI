# aura-control/http_client.py

import requests
from io import BytesIO

def send_to_whisper(audio_data):
    # ✅ Send audio as a WAV file using multipart/form-data
    files = {
        "audio": ("input.wav", BytesIO(audio_data), "audio/wav")
    }
    r = requests.post("http://localhost:5000/transcribe", files=files)
    r.raise_for_status()
    return r.json()["text"]

def send_to_llm(text):
    r = requests.post("http://localhost:5001/respond", json={"text": text})
    r.raise_for_status()
    return r.json()["response"]

def send_to_tts(text):
    r = requests.post("http://localhost:5002/speak", json={"text": text})
    r.raise_for_status()
    return r.json()["status"]
