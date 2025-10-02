# === main.py — Orchestrates Aura modules ===

import time
import subprocess
import threading
import os
import signal
import sys
import requests
from dotenv import dotenv_values   # 👈 load host .env

from aura_gui import launch_gui, run_gui_loop
from listener import listen
import speaker  # ✅ Starts TTS playback loop and queue

os.environ["DISPLAY"] = ":0"

# Load host .env (adjust path if needed)
HOST_ENV = dotenv_values(os.path.expanduser("~/LedgerAI/llm-container/.env"))

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Utility: Stop and remove container if it exists ===
def remove_existing_container(name):
    try:
        subprocess.run(["docker", "rm", "-f", name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[Aura] 🗑️ Removed existing container: {name}")
    except subprocess.CalledProcessError:
        print(f"[Aura] ℹ️ No existing container to remove: {name}")

# === Stream container logs into Aura ===
def stream_container_logs(name):
    def _logs():
        process = subprocess.Popen(
            ["docker", "logs", "-f", name],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        process.wait()
    threading.Thread(target=_logs, daemon=True).start()

# === Health check for container via HTTP ===
def wait_for_container(url, name, timeout=15):
    print(f"[Aura] ⏳ Waiting for {name} to respond (timeout {timeout}s)...")
    for _ in range(timeout * 10):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code in (200, 404):
                print(f"[Aura] ✅ {name} is online.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.1)
    print(f"[Aura] ❌ Timeout waiting for {name}.")
    return False

# === Launch container cleanly with retry ===
def run_container(name, port, image, timeout=15):
    remove_existing_container(name)

    print(f"[Aura] 🚀 Launching {name}...")

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--rm",
        "--network=host",
        "--device", "/dev/snd",
        "--device", "/dev/bus/usb",
    ]

    if name == "aura-llm":
        # Read from host .env, fallback to defaults
        model_path  = HOST_ENV.get("MODEL_PATH", "/models/qwen2.5-1.5b-instruct-q4_0.gguf")
        chat_format = HOST_ENV.get("CHAT_FORMAT", "qwen")
        n_ctx       = HOST_ENV.get("N_CTX", "1024")

        cmd += [
            "-e", f"MODEL_PATH={model_path}",
            "-e", f"CHAT_FORMAT={chat_format}",
            "-e", f"N_CTX={n_ctx}",
            "-v", f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))}:/app/data"  # Mount embeddings data
        ]

    if name == "aura-whisper":
        cmd += [
            "-v", f"{os.path.expanduser('~')}/LedgerAI/whisper-container/cache/whisper:/root/.cache/whisper",
            "-v", f"{os.path.expanduser('~')}/LedgerAI/whisper-container/cache/whisper_trt:/root/.cache/whisper_trt"
        ]


    cmd.append(image)

    for attempt in range(3):
        subprocess.Popen(cmd)
        if wait_for_container(f"http://localhost:{port}", name, timeout=timeout):
            stream_container_logs(name)
            return True
        print(f"[Aura] 🔁 Retry {attempt + 1}/3 for {name}")
        remove_existing_container(name)
        time.sleep(2)

    return False

# === ElevenLabs warm-up ===
def warm_up_tts():
    speaker.warm_up_tts()

# === LLM warm-up ===
def warm_up_llm():
    try:
        print("[Aura] 🧠 Warming up LLM...")
        requests.post("http://localhost:11434/chat", json={"prompt": "..."}, timeout=5)
        print("[Aura] ✅ LLM warm-up complete.")
    except Exception as e:
        print(f"[Aura] ⚠️ LLM warm-up failed: {e}")

# === RAG warm-up ===
def warm_up_rag():
    try:
        print("[Aura] 🔍 Warming up RAG system...")
        # Test RAG stats
        stats_response = requests.get("http://localhost:11434/rag/stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"[Aura] ✅ RAG loaded: {stats.get('chunks_loaded', 0)} medical documents")
        else:
            print("[Aura] ⚠️ RAG stats endpoint not available")
            
        # Test RAG search
        search_response = requests.post(
            "http://localhost:11434/rag/search",
            json={"query": "test", "k": 1},
            timeout=10
        )
        if search_response.status_code == 200:
            print("[Aura] ✅ RAG search working")
        else:
            print("[Aura] ⚠️ RAG search endpoint not working")
            
    except Exception as e:
        print(f"[Aura] ⚠️ RAG warm-up failed: {e}")

# === Start services after GUI is ready ===
def start_services():
    TIMEOUT = 10  # unified timeout for all containers

    whisper_ok = run_container("aura-whisper", 5000, "aura-whisper:latest", timeout=TIMEOUT)
    if not whisper_ok:
        print("[Aura] ❌ Whisper container failed. Aborting.")
        return


    llm_ok = run_container("aura-llm", 11434, "aura-llm-rag:latest", timeout=TIMEOUT)
    if not llm_ok:
        print("[Aura] ❌ LLM container failed. Aborting.")
        return

    warm_up_llm()
    warm_up_rag()
    print("[Aura] 🎙️ Starting listener...")
    threading.Thread(target=listen, daemon=True).start()

# === Main Entrypoint ===
def main():
    print("[Aura] 🌀 Launching Aura GUI...")
    warm_up_tts()
    threading.Thread(target=start_services, daemon=True).start()
    launch_gui()
    run_gui_loop()

if __name__ == "__main__":
    main()
