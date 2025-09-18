from flask import Flask, request, jsonify, Response
import os
import subprocess
import requests
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv("/app/.env")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
ELEVEN_MODEL = "eleven_turbo_v2"
PIPER_MODEL_PATH = "/models/en_US/en_US-amy-low.onnx"
PIPER_CONFIG_PATH = "/models/en_US/en_US-amy-low.onnx.json"

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "tts"}), 200

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "No text provided"}, 400

    print(f"[Aura-TTS] 🔊 Requested text: {text}")
    print(f"[Aura-TTS] ✅ ELEVEN_API_KEY loaded: {bool(ELEVEN_API_KEY)}, VOICE_ID: {ELEVEN_VOICE_ID}")

    # === Try ElevenLabs TTS ===
    if ELEVEN_API_KEY and ELEVEN_VOICE_ID:
        headers = {
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }

        try:
            print("[Aura-TTS] 🌐 Sending request to ElevenLabs...")
            r = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream",
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            if r.status_code == 200:
                print("[Aura-TTS] ✅ ElevenLabs response received.")
                def generate():
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk
                return Response(generate(), content_type="audio/mpeg")
            else:
                print(f"[Aura-TTS] ⚠️ ElevenLabs error {r.status_code}: {r.text}")

        except Exception as e:
            print(f"[Aura-TTS] ⚠️ ElevenLabs request failed: {e}")

    # === Fallback to Piper TTS ===
    print("[Aura-TTS] 🔁 Falling back to Piper TTS")
    print(f"[Aura-TTS] 📦 Invoking Piper with: {text}")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run([
            "piper",
            "-m", PIPER_MODEL_PATH,
            "-c", PIPER_CONFIG_PATH,
            "-f", tmp_path
        ], input=text.encode("utf-8"), check=True)

        def stream_audio():
            with open(tmp_path, "rb") as f:
                while chunk := f.read(1024):
                    yield chunk
            os.remove(tmp_path)

        return Response(stream_audio(), content_type="audio/wav")

    except Exception as e:
        print(f"[Aura-TTS] ❌ Piper error: {e}")
        return {"error": f"Piper TTS failed: {str(e)}"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
