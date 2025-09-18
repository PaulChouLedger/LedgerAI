# === container_rest.py — inside aura-llm ===

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os
import time

app = Flask(__name__)
load_dotenv()

# === Model config from .env ===
MODEL_PATH = os.getenv("MODEL_PATH")
CHAT_FORMAT = os.getenv("CHAT_FORMAT")
MODEL_NAME = os.path.basename(MODEL_PATH) if MODEL_PATH else "unknown"

# === Load the model ===
print(f"[Aura-LLM] 🧠 Loading model: {MODEL_NAME}")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=32,
    n_ctx=1024,
    n_threads=4,
    use_mlock=True,
    use_mmap=True,
    verbose=False,
    chat_format=CHAT_FORMAT
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "llm"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    print(f"[Aura-LLM] 💬 Prompt received: {prompt}")
    messages = [{"role": "user", "content": prompt}]

    def generate():
        buffer = ""
        first_token_time = None
        stream = llm.create_chat_completion(messages=messages, stream=True)

        for chunk in stream:
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not token:
                continue

            if not first_token_time:
                first_token_time = time.time()
                print(f"[LLM] ⏱️ First token in: {first_token_time - start_time:.2f}s")

            buffer += token
            print(f"[LLM] 🧠 {token}", end="", flush=True)

            if len(buffer.strip()) >= 8 or any(p in buffer for p in [".", "?", "!"]):
                yield buffer.strip() + "\n"
                buffer = ""

        if buffer.strip():
            yield buffer.strip() + "\n"

    start_time = time.time()
    return Response(stream_with_context(generate()), mimetype="text/plain")

if __name__ == "__main__":
    print("[Aura-LLM] 🚀 LLM API ready on port 11434")
    app.run(host="0.0.0.0", port=11434)
