# === main.py — Orchestrates Aura modules ===

import time
import subprocess
import threading
import os
import signal
import sys
import requests

from aura_gui import launch_gui, run_gui_loop, gui_is_ready
from listener import listen
from fingerprint import start_fingerprint_monitor
import speaker  # ✅ Starts TTS playback loop and queue

os.environ["DISPLAY"] = ":0"

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Utility: Stop and remove container if it exists ===
def remove_existing_container(name):
    try:
        subprocess.run(["docker", "rm", "-f", name], check=True, stdout=subprocess.DEVNULL)
        print(f"[Aura] 🗑️ Removed existing container: {name}")
    except subprocess.CalledProcessError:
        print(f"[Aura] ℹ️ No existing container to remove: {name}")

# === Health check for container via HTTP ===
def wait_for_container(url, name, timeout=20):
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

# === Launch container cleanly ===
def run_container(name, port, image):
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

    # Inject LLM-specific environment variables
    if name == "aura-llm":
        cmd += [
            "-e", "MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_0.gguf",
            "-e", "CHAT_FORMAT=qwen"
        ]

    # Inject Whisper cache mounts
    if name == "aura-whisper":
        cmd += [
            "-v", f"{os.getcwd()}/cache/whisper:/root/.cache/whisper",
            "-v", f"{os.getcwd()}/cache/whisper_trt:/root/.cache/whisper_trt"
        ]

    cmd.append(image)
    subprocess.Popen(cmd)

    return wait_for_container(f"http://localhost:{port}", name)

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

# === Start services after GUI is ready ===
def start_services():
    whisper_ok = run_container("aura-whisper", 5000, "aura-whisper:latest")
    llm_ok = run_container("aura-llm", 11434, "aura-llm:latest")

    if whisper_ok and llm_ok:
        warm_up_llm()
        print("[Aura] 🔍 Starting fingerprint monitor...")
        threading.Thread(target=start_fingerprint_monitor, daemon=True).start()

        print("[Aura] 🎙️ Starting listener...")
        threading.Thread(target=listen, daemon=True).start()
    else:
        print("[Aura] ❌ One or more containers failed to start. Aborting listener.")

# === Main Entrypoint ===
def main():
    print("[Aura] 🌀 Launching Aura GUI...")
    warm_up_tts()
    threading.Thread(target=start_services, daemon=True).start()
    launch_gui()         # Must run in main thread
    run_gui_loop()       # Keeps GUI alive

if __name__ == "__main__":
    main()
