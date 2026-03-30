#!/usr/bin/env python3
"""
Farsight LLM Offload — Thin proxy from puck /perpetual/chat API to Ollama.

Listens on port 11435, translates requests to Ollama's OpenAI-compatible
API on localhost:11434, and returns responses in the format the puck expects.
"""

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:72b-instruct-q8_0"


@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/v1/models", timeout=3)
        return jsonify({"status": "healthy", "model": OLLAMA_MODEL})
    except Exception:
        return jsonify({"status": "unhealthy"}), 503


@app.route("/perpetual/chat", methods=["POST"])
def perpetual_chat():
    data = request.get_json(silent=True) or {}
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    user_prompt = data.get("prompt", "")
    context = data.get("context", "")
    max_tokens = data.get("max_tokens", 512)

    if not user_prompt:
        return jsonify({"error": "prompt required"}), 400

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Recent conversation:\n{context}"})
    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Ollama HTTP {resp.status_code}"}), 502

        choices = resp.json().get("choices", [])
        if not choices:
            return jsonify({"error": "no response"}), 502

        text = choices[0].get("message", {}).get("content", "").strip()
        return jsonify({"response": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[farsight-llm] Proxy started on port 11435 → Ollama ({OLLAMA_MODEL})")
    app.run(host="0.0.0.0", port=11435, threaded=True)
