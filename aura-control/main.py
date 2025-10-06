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
try:
    from web_upload_server import start_upload_server
    UPLOAD_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"[Aura] ⚠️ Upload server not available: {e}")
    print(f"[Aura] 💡 Install Flask: pip install flask")
    UPLOAD_SERVER_AVAILABLE = False

os.environ["DISPLAY"] = ":0"

# Load host .env (adjust path if needed)
HOST_ENV = dotenv_values(os.path.expanduser("~/LedgerAI/llm-container/.env"))

# === Whisper Container Configuration ===
# Using faster-whisper with distil-small.en
WHISPER_IMAGE = "aura-whisper-faster:latest"
WHISPER_CONTAINER_NAME = "aura-whisper"
print(f"[Aura] 🎤 Whisper container: faster-whisper with distil-small.en")

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    cleanup_resources()
    sys.exit(0)

def cleanup_resources():
    """Clean up all resources before exit"""
    print("[Aura] 🧹 Cleaning up resources...")
    
    # Stop and remove containers with proper cleanup
    containers_to_cleanup = ["aura-llm", WHISPER_CONTAINER_NAME]
    for container in containers_to_cleanup:
        try:
            print(f"[Aura] 🧹 Stopping container: {container}")
            # First try graceful stop
            subprocess.run(["docker", "stop", container], 
                          timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            # Then force remove
            remove_existing_container(container)
        except Exception as e:
            print(f"[Aura] ⚠️ Failed to cleanup container {container}: {e}")
    
    # Clear RAG-specific resources
    cleanup_rag_resources()
    
    print("[Aura] ✅ Resource cleanup completed")

def cleanup_rag_resources():
    """Clean up RAG-specific resources"""
    print("[Aura] 🧹 Cleaning up RAG resources...")
    
    try:
        import shutil
        
        # Clear HuggingFace cache (sentence transformers)
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            print(f"[Aura] 🧹 Clearing HuggingFace cache: {hf_cache}")
            shutil.rmtree(hf_cache, ignore_errors=True)
        
        # Clear sentence transformer cache
        st_cache = os.path.expanduser("~/.cache/sentence_transformers")
        if os.path.exists(st_cache):
            print(f"[Aura] 🧹 Clearing sentence transformer cache: {st_cache}")
            shutil.rmtree(st_cache, ignore_errors=True)
        
        # Clear local RAG cache directories
        rag_cache_dirs = [
            "./cache/huggingface",
            "./cache/transformers", 
            "./cache/sentence_transformers"
        ]
        
        for cache_dir in rag_cache_dirs:
            if os.path.exists(cache_dir):
                print(f"[Aura] 🧹 Clearing RAG cache: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
        
        # Clear any temporary FAISS files
        temp_faiss_files = [
            "data/embeddings/index.faiss.tmp",
            "data/embeddings/doc_chunks.npy.tmp"
        ]
        
        for temp_file in temp_faiss_files:
            if os.path.exists(temp_file):
                print(f"[Aura] 🧹 Removing temp file: {temp_file}")
                os.remove(temp_file)
                
    except Exception as e:
        print(f"[Aura] ⚠️ RAG cleanup failed: {e}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)  # Also handle SIGTERM

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
    for i in range(timeout * 10):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code in (200, 404):
                print(f"[Aura] ✅ {name} is online.")
                return True
            else:
                if i % 50 == 0:  # Print every 5 seconds
                    print(f"[Aura] 🔍 {name} responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            if i % 50 == 0:  # Print every 5 seconds
                print(f"[Aura] 🔍 {name} connection error: {e}")
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
    elif name == WHISPER_CONTAINER_NAME:
        # Use built-in model files, no external cache mounting needed
        pass


    cmd.append(image)
    
    # Debug: Print the exact command being run
    print(f"[Aura] 🔍 Container command: {' '.join(cmd)}")

    for attempt in range(3):
        try:
            # Start container in background
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[Aura] ⚠️ Container command error: {e}")
        
        # Use appropriate health check endpoint
        if name == WHISPER_CONTAINER_NAME:
            health_url = f"http://localhost:{port}/health"
            print(f"[Aura] 🔍 Health check URL: {health_url}")
            time.sleep(15)   # Give more time for model loading
        else:
            health_url = f"http://localhost:{port}"
        
        if wait_for_container(health_url, name, timeout=timeout):
            stream_container_logs(name)
            return True
        print(f"[Aura] 🔁 Retry {attempt + 1}/3 for {name}")
        
        # Check if container is actually running
        try:
            result = subprocess.run(["docker", "ps", "--filter", f"name={name}", "--format", "{{.Status}}"], 
                                capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print(f"[Aura] 🔍 Container {name} is running but not responding: {result.stdout.strip()}")
                
                # Try to get container logs to see what's happening
                try:
                    logs_result = subprocess.run(["docker", "logs", "--tail", "10", name], 
                                              capture_output=True, text=True, timeout=5)
                    if logs_result.returncode == 0 and logs_result.stdout.strip():
                        print(f"[Aura] 📋 Container logs (last 10 lines):")
                        for line in logs_result.stdout.strip().split('\n'):
                            print(f"[Aura] 📋 {line}")
                except Exception as log_e:
                    print(f"[Aura] 🔍 Could not get container logs: {log_e}")
            else:
                print(f"[Aura] 🔍 Container {name} is not running")
        except Exception as e:
            print(f"[Aura] 🔍 Could not check container status: {e}")
        
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
        # Try multiple times with increasing timeout
        for attempt in range(3):
            try:
                response = requests.post("http://localhost:11434/chat", 
                                       json={"prompt": "Hello"}, 
                                       timeout=10 + (attempt * 5))
                if response.status_code == 200:
                    print("[Aura] ✅ LLM warm-up complete.")
                    return True
            except requests.exceptions.RequestException as e:
                print(f"[Aura] ⚠️ LLM warm-up attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(3)
        print("[Aura] ⚠️ LLM warm-up failed after 3 attempts")
        return False
    except Exception as e:
        print(f"[Aura] ⚠️ LLM warm-up failed: {e}")
        return False

# === RAG warm-up ===
def initialize_rag_delayed():
    """Initialize RAG system after core services are stable"""
    try:
        print("[Aura] 🔍 RAG initialization starting...")
        
        print("[Aura] 🔍 Initializing RAG system...")
        
        # RAG container should auto-initialize, just wait for it to be ready
        print("[Aura] 🔍 Waiting for RAG container to be ready...")
        
        # Wait for RAG to be fully ready with proper polling
        print("[Aura] ⏳ Waiting for RAG to be fully ready...")
        max_wait_time = 30  # Maximum 30 seconds wait
        poll_interval = 1   # Check every 1 second
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                stats_response = requests.get("http://localhost:11435/rag/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    chunks_loaded = stats.get('chunks_loaded', 0)
                    if chunks_loaded > 0:  # RAG is ready when it has loaded chunks
                        print(f"[Aura] ✅ RAG loaded: {chunks_loaded} medical documents")
                        break
                    else:
                        print(f"[Aura] ⏳ RAG still loading... ({chunks_loaded} chunks)")
                else:
                    print(f"[Aura] ⏳ RAG not ready yet (status: {stats_response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"[Aura] ⏳ RAG not ready yet: {e}")
            
            time.sleep(poll_interval)
        else:
            print("[Aura] ❌ RAG failed to load within 30 seconds")
            
        # RAG is already verified to be working from the polling above
        print("[Aura] ✅ RAG system fully ready")
            
    except Exception as e:
        print(f"[Aura] ⚠️ Delayed RAG initialization failed: {e}")

def warm_up_rag():
    """Legacy function - now calls delayed initialization"""
    initialize_rag_delayed()

# === Startup cleanup ===
def startup_cleanup():
    """Clean up any leftover resources from previous runs"""
    print("[Aura] 🧹 Performing startup cleanup...")
    
    # Clean up any leftover containers
    containers_to_cleanup = ["aura-llm", "aura-whisper", "aura-whisper-faster"]
    for container in containers_to_cleanup:
        try:
            remove_existing_container(container)
        except Exception as e:
            print(f"[Aura] ⚠️ Failed to cleanup container {container}: {e}")
    
    # Reset RAG state if LLM container is running
    print("[Aura] ✅ Startup cleanup completed")

# === Start services after GUI is ready ===
def start_services():
    TIMEOUT = 10  # Reduced timeout for faster startup
    
    print("[Aura] 🚀 Starting Aura services...")
    
    # Step 0: Cleanup any leftover resources
    startup_cleanup()
    
    # Step 1: Start Whisper container
    print(f"[Aura] 🎤 Starting Whisper container (faster-whisper with distil-small.en)...")
    
    whisper_ok = run_container(WHISPER_CONTAINER_NAME, 5000, WHISPER_IMAGE, timeout=10)
    if not whisper_ok:
        print("[Aura] ❌ Whisper container failed. Aborting.")
        return
    
    # Step 2: Start LLM container
    print("[Aura] 🧠 Starting LLM container...")
    llm_ok = run_container("aura-llm", 11434, "aura-llm:latest", timeout=TIMEOUT)
    if not llm_ok:
        print("[Aura] ❌ LLM container failed. Aborting.")
        return
    
    # Step 3: Wait for LLM to be fully ready
    print("[Aura] ⏳ Waiting for LLM to initialize...")
    time.sleep(10)  # Give LLM more time to load model
    
    # Step 4: Warm up LLM (without RAG first)
    if not warm_up_llm():
        print("[Aura] ❌ LLM warm-up failed. Aborting.")
        return
    
    # Step 4.5: Reset RAG state now that LLM container is running
    print("[Aura] 🔄 Resetting RAG state...")
    try:
        reset_response = requests.post("http://localhost:11434/rag/reset", timeout=5)
        if reset_response.status_code == 200:
            print("[Aura] ✅ RAG state reset")
        else:
            print(f"[Aura] ⚠️ RAG reset failed: {reset_response.status_code}")
    except Exception as e:
        print(f"[Aura] ⚠️ RAG reset failed: {e}")
    
    # Step 5: Initialize RAG immediately after LLM warm-up and wait for completion
    print("[Aura] 🔍 Initializing RAG system...")
    initialize_rag_delayed()  # Run synchronously, not in thread
    
    # Wait for RAG to be fully ready before starting listener
    print("[Aura] ⏳ Waiting for RAG to be fully ready...")
    max_wait_time = 60  # Maximum 60 seconds wait for RAG
    poll_interval = 2   # Check every 2 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            stats_response = requests.get("http://localhost:11435/rag/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                health_score = stats.get('health_score', 0)
                chunks_loaded = stats.get('chunks_loaded', 0)
                
                # RAG must be fully functional (100% health) with sentence transformer
                if health_score >= 100.0 and chunks_loaded > 0:
                    print(f"[Aura] ✅ RAG fully ready: {chunks_loaded} chunks, {health_score}% health")
                    break
                else:
                    print(f"[Aura] ⏳ RAG still loading... ({chunks_loaded} chunks, {health_score}% health)")
            else:
                print(f"[Aura] ⏳ RAG not ready yet (status: {stats_response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"[Aura] ⏳ RAG not ready yet: {e}")
        
        time.sleep(poll_interval)
    else:
        print("[Aura] ❌ RAG failed to load within 60 seconds")
        print("[Aura] ⚠️ Starting with partial RAG functionality")
    
    # Step 6: Start file upload server (if available)
    if UPLOAD_SERVER_AVAILABLE:
        print("[Aura] 📁 Starting file upload server...")
        upload_ip, upload_port = start_upload_server(port=5001)
        print(f"[Aura] 📱 Upload server ready at http://{upload_ip}:{upload_port}")
    else:
        print("[Aura] ⚠️ Upload server skipped (Flask not available)")
    
    # Step 7: Start listener (after RAG is initialized)
    print("[Aura] 🎙️ Starting listener...")
    threading.Thread(target=listen, daemon=True).start()
    
    # Step 8: Mark setup as complete and set GUI to fixed mode (listener ready)
    from aura_gui import set_listening_ready, set_setup_complete
    set_setup_complete()
    set_listening_ready()
    
    print("[Aura] ✅ Core services started successfully!")

# === Main Entrypoint ===
def main():
    print("[Aura] 🌀 Launching Aura GUI...")
    
    # Set up signal handlers for graceful shutdown
    import signal
    def signal_handler(signum, frame):
        print(f"\n[Aura] ⛔ Received signal {signum} - requesting shutdown")
        from state import request_shutdown
        request_shutdown()
        from aura_gui import close_gui
        close_gui()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    warm_up_tts()
    threading.Thread(target=start_services, daemon=True).start()
    launch_gui()
    run_gui_loop()

if __name__ == "__main__":
    main()
