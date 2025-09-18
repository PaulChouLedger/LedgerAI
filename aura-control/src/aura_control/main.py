# === main.py — Orchestrates Aura modules ===

import time
import threading
import os
import signal
import sys
import requests

from aura_gui import launch_gui, run_gui_loop, gui_is_ready
from listener import listen
from fingerprint import start_fingerprint_monitor
import speaker  # Starts TTS playback loop and queue

os.environ["DISPLAY"] = ":0"

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Health check for container via HTTP ===
def wait_for_container(url, name, timeout=20):
    """Check if a container service is responding."""
    print(f"[Aura] ⏳ Waiting for {name} to respond...")
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
        print("[Aura] 🧠 Warming up RAG system...")
        requests.get("http://localhost:5003/health", timeout=10)
        print("[Aura] ✅ RAG system ready.")
    except Exception as e:
        print(f"[Aura] ⚠️ RAG warm-up failed: {e}")

# === Start services after GUI is ready ===
def start_services():
    # Check if containers are running (managed by docker-compose)
    whisper_ok = wait_for_container("http://localhost:5000", "aura-whisper", timeout=10)
    rapids_ok = wait_for_container("http://localhost:5003", "aura-rapids", timeout=10)
    llm_ok = wait_for_container("http://localhost:11434", "aura-llm", timeout=10)

    if whisper_ok and rapids_ok and llm_ok:
        warm_up_llm()
        warm_up_rag()
        print("[Aura] 🔍 Starting fingerprint monitor...")
        threading.Thread(target=start_fingerprint_monitor, daemon=True).start()

        print("[Aura] 🎙️ Starting listener...")
        threading.Thread(target=listen, daemon=True).start()
    else:
        print("[Aura] ❌ One or more containers not available. Make sure docker-compose is running.")

# === Main Entrypoint ===
def main():
    print("[Aura] 🌀 Launching Aura GUI...")
    warm_up_tts()
    threading.Thread(target=start_services, daemon=True).start()
    launch_gui()         # Must run in main thread
    run_gui_loop()       # Keeps GUI alive

if __name__ == "__main__":
    main()
