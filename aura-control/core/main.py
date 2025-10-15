# === main.py — Orchestrates Aura modules ===

import time
import subprocess
import threading
import os
import signal
import sys
import requests
import concurrent.futures
from dotenv import dotenv_values   # 👈 load host .env

# Import from organized directories
from ..gui.aura_gui import launch_gui, run_gui_loop, is_gui_ready
from .listener import listen
from . import speaker  # ✅ Starts TTS playback loop and queue
try:
    from ..server.web_upload_server import start_upload_server
    UPLOAD_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"[Aura] ⚠️ Upload server not available: {e}")
    print(f"[Aura] 💡 Install Flask: pip install flask")
    UPLOAD_SERVER_AVAILABLE = False

os.environ["DISPLAY"] = ":0"

# Load host .env (adjust path if needed)
HOST_ENV = dotenv_values(os.path.expanduser("~/LedgerAI/llm-container/.env"))

# === Whisper Container Configuration ===
# Using faster-whisper with distil-small.en model
WHISPER_IMAGE = "aura-whisper:latest"
WHISPER_NAME = "aura-whisper"
WHISPER_DESCRIPTION = "faster-whisper with distil-small.en"

print(f"[Aura] 🎤 Whisper container: {WHISPER_DESCRIPTION}")

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Display Setup ===
def setup_display():
    """
    Configure display settings for Aura:
    - Wake screen from sleep
    - Dismiss screensaver/lock screen
    - Keep screen on for at least 5 minutes
    - Hide mouse cursor for clean interface
    """
    print("[Aura] 🖥️  Configuring display...")
    
    # Wake the screen and turn on display
    try:
        subprocess.run(["xset", "dpms", "force", "on"], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Deactivate screensaver (dismisses any screensaver overlay)
    try:
        subprocess.run(["xscreensaver-command", "-deactivate"], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Kill gnome-screensaver if running
    try:
        subprocess.run(["gnome-screensaver-command", "-d"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Simulate user activity to dismiss any lock/blank screen
    try:
        subprocess.run(["xdotool", "key", "shift"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Disable screen blanking and power saving for 5 minutes (300 seconds)
    try:
        subprocess.run(["xset", "s", "off"], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xset", "s", "noblank"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xset", "-dpms"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xset", "+dpms"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xset", "dpms", "300", "300", "300"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Hide mouse cursor (try unclutter first, fallback to xdotool)
    cursor_hidden = False
    try:
        # unclutter hides cursor when idle (best option)
        subprocess.Popen(["unclutter", "-idle", "0.1", "-root"], 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        print("[Aura] ✅ Mouse cursor hidden (unclutter)")
        cursor_hidden = True
    except FileNotFoundError:
        # Fallback: move cursor to corner using xdotool
        try:
            subprocess.run(["xdotool", "mousemove", "0", "0"], check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[Aura] ✅ Mouse cursor moved to corner")
            cursor_hidden = True
        except Exception:
            pass
    
    if not cursor_hidden:
        print("[Aura] 💡 Install cursor tools: sudo apt install unclutter xdotool wmctrl")
    
    print("[Aura] ✅ Display configured: screen awake, no sleep for 5 min")

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
    elif name == WHISPER_NAME:
        # faster-whisper model is baked into the image, no cache mounting needed
        cmd += [
            "--gpus", "all"  # Add GPU support for faster-whisper
        ]
    elif name == "aura-rag":
        # RAG container - mount data directory only (rebuild_embeddings.py is baked into image)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cmd += [
            "-v", f"{workspace_root}/data:/app/data",  # Mount embeddings data
        ]


    cmd.append(image)

    for attempt in range(3):
        try:
            # Start container in background
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[Aura] ⚠️ Container command error: {e}")
        
        # Use appropriate health check endpoint
        if name == WHISPER_NAME:
            health_url = f"http://localhost:{port}/health"
            time.sleep(5)   # Standard time for faster-whisper
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
                response = requests.post("http://localhost:11434/chat-tts", 
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
        
        # Initialize RAG system (longer timeout for CUDA model loading on first boot)
        for attempt in range(3):
            try:
                init_response = requests.post("http://localhost:11435/rag/init", timeout=90)
                if init_response.status_code == 200:
                    result = init_response.json()
                    if result.get("status") == "success":
                        print("[Aura] ✅ RAG system initialized successfully")
                        break
                    else:
                        print(f"[Aura] ⚠️ RAG init attempt {attempt + 1} failed: {result.get('message')}")
                else:
                    print(f"[Aura] ⚠️ RAG init attempt {attempt + 1} failed: {init_response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[Aura] ⚠️ RAG init attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        else:
            print("[Aura] ⚠️ RAG initialization failed after 3 attempts")
            return
        
        # Test RAG stats
        for attempt in range(3):
            try:
                stats_response = requests.get("http://localhost:11435/rag/stats", timeout=30)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    print(f"[Aura] ✅ RAG loaded: {stats.get('chunks_loaded', 0)} medical documents")
                    break
                else:
                    print(f"[Aura] ⚠️ RAG stats attempt {attempt + 1} failed: {stats_response.status_code}")
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    print(f"[Aura] ⚠️ RAG stats attempt {attempt + 1} failed: {e}")
                    time.sleep(5)  # Wait longer between attempts
                else:
                    print(f"[Aura] ⚠️ RAG stats attempt {attempt + 1} failed: {e}")
        else:
            print("[Aura] ⚠️ RAG stats endpoint not available after 3 attempts")
            
        # Test RAG search with CUDA verification
        print("[Aura] 🔍 Verifying RAG search with CUDA vectors...")
        rag_ready = False
        for attempt in range(5):  # More attempts for CUDA loading
            try:
                search_response = requests.post(
                    "http://localhost:11435/rag/search",
                    json={"query": "test", "k": 3},
                    timeout=30
                )
                if search_response.status_code == 200:
                    result = search_response.json()
                    # Verify we got actual results (CUDA vectors loaded)
                    if result.get('results') and len(result['results']) > 0:
                        print(f"[Aura] ✅ RAG search working - {len(result['results'])} results returned")
                        rag_ready = True
                        break
                    else:
                        print(f"[Aura] ⚠️ RAG search returned empty (CUDA loading?), attempt {attempt + 1}/5")
                else:
                    print(f"[Aura] ⚠️ RAG search attempt {attempt + 1}/5 failed: {search_response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[Aura] ⚠️ RAG search attempt {attempt + 1}/5 failed: {e}")
            
            if attempt < 4:
                time.sleep(3)  # Wait for CUDA vectors to load
        
        if not rag_ready:
            print("[Aura] ❌ RAG search not returning results after 5 attempts - listener may have issues")
            print("[Aura] 💡 Continuing anyway, but first queries may fail...")
        else:
            print("[Aura] ✅ RAG fully initialized and ready for queries")
            
    except Exception as e:
        print(f"[Aura] ⚠️ Delayed RAG initialization failed: {e}")

def warm_up_rag():
    """Legacy function - now calls delayed initialization"""
    initialize_rag_delayed()

# === Start services after GUI is ready ===
def start_services():
    TIMEOUT = 10  # Reduced timeout for faster startup
    
    print("[Aura] 🚀 Starting all containers in parallel...")
    
    # Start all containers in parallel using threads
    def start_whisper():
        print(f"[Aura] 🎤 Starting Whisper container ({WHISPER_DESCRIPTION})...")
        return run_container(WHISPER_NAME, 5000, WHISPER_IMAGE, timeout=10)
    
    def start_llm():
        print("[Aura] 🧠 Starting LLM container...")
        return run_container("aura-llm", 11434, "aura-llm:latest", timeout=TIMEOUT)
    
    def start_rag():
        print("[Aura] 🔍 Starting RAG container...")
        # RAG needs longer timeout due to CUDA model pre-loading (30-60s on first boot)
        return run_container("aura-rag", 11435, "aura-rag:latest", timeout=90)
    
    # Start all containers simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        whisper_future = executor.submit(start_whisper)
        llm_future = executor.submit(start_llm)
        rag_future = executor.submit(start_rag)
        
        # Wait for all to complete
        whisper_ok = whisper_future.result()
        llm_ok = llm_future.result()
        rag_ok = rag_future.result()
    
    # Check if all succeeded
    if not whisper_ok:
        print("[Aura] ❌ Whisper container failed. Aborting.")
        return
    if not llm_ok:
        print("[Aura] ❌ LLM container failed. Aborting.")
        return
    if not rag_ok:
        print("[Aura] ❌ RAG container failed. Aborting.")
        return
    
    print("[Aura] ✅ All containers started successfully!")
    
    # Wait a bit for LLM to fully initialize
    print("[Aura] ⏳ Waiting for LLM to initialize...")
    time.sleep(5)  # Reduced from 10s since containers started in parallel
    
    # Step 4: Warm up LLM (without RAG first)
    if not warm_up_llm():
        print("[Aura] ❌ LLM warm-up failed. Aborting.")
        return
    
    # Step 5: Initialize RAG immediately after LLM warm-up
    print("[Aura] 🔍 Initializing RAG system...")
    initialize_rag_delayed()  # Run synchronously, not in thread
    
    # Step 6: Start file upload server (if available)
    if UPLOAD_SERVER_AVAILABLE:
        print("[Aura] 📁 Starting file upload server...")
        upload_ip, upload_port = start_upload_server(port=5001)
        print(f"[Aura] 📱 Upload server ready at http://{upload_ip}:{upload_port}")
    else:
        print("[Aura] ⚠️ Upload server skipped (Flask not available)")
    
    # Step 7: Start auto-ingest monitoring (container extracts, host generates embeddings)
    print("[Aura] 📂 Starting auto-ingest monitoring...")
    
    # Trigger initial ingest scan
    try:
        # Container extracts text from PDFs
        response = requests.post("http://localhost:11435/rag/ingest", timeout=30)
        if response.status_code == 200:
            result = response.json()
            processed = result.get('processed', 0)
            print(f"[Aura] ✅ Text extraction: {processed} processed, {result.get('skipped', 0)} skipped")
            
            # If new files were processed, run rebuild_embeddings on HOST
            if processed > 0:
                print(f"[Aura] 🔧 Running rebuild_embeddings.py on host...")
                import subprocess
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                rebuild_script = os.path.join(workspace_root, 'rag-container', 'rebuild_embeddings.py')
                rebuild_result = subprocess.run(
                    ["python3", rebuild_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=workspace_root
                )
                
                if rebuild_result.returncode == 0:
                    print(f"[Aura] ✅ Host embeddings generated successfully")
                    
                    # Tell container to reload the new index
                    reload_response = requests.post("http://localhost:11435/rag/reload", timeout=10)
                    if reload_response.status_code == 200:
                        reload_result = reload_response.json()
                        print(f"[Aura] ✅ RAG reloaded: {reload_result.get('total_chunks', 0)} chunks")
                    else:
                        print(f"[Aura] ⚠️ RAG reload failed: {reload_response.status_code}")
                else:
                    print(f"[Aura] ❌ Host rebuild failed")
                    print(f"[Aura] 💥 Error: {rebuild_result.stderr[:500]}")
        else:
            print(f"[Aura] ⚠️ Initial ingest failed: {response.status_code}")
    except Exception as e:
        print(f"[Aura] ⚠️ Initial ingest error: {e}")
    
    # Auto-ingest is triggered by file uploads via web server (no periodic polling needed)
    print("[Aura] ✅ Auto-ingest enabled (triggered by file uploads via web server)")
    
    # Step 8: Final RAG ready check before starting listener
    print("[Aura] 🔍 Final RAG ready check before starting listener...")
    try:
        final_check = requests.post(
            "http://localhost:11435/rag/search",
            json={"query": "system check", "k": 1},
            timeout=10
        )
        if final_check.status_code == 200 and final_check.json().get('results'):
            print("[Aura] ✅ RAG confirmed ready - starting listener")
        else:
            print("[Aura] ⚠️ RAG may not be fully ready, but starting listener anyway")
    except Exception as e:
        print(f"[Aura] ⚠️ RAG final check failed: {e}, starting listener anyway")
    
    # Step 9: Start listener (after RAG is confirmed ready)
    print("[Aura] 🎙️ Starting listener...")
    threading.Thread(target=listen, daemon=True).start()
    
    print("[Aura] ✅ Core services started successfully!")

# === Bring GUI to Front ===
def focus_gui_window():
    """
    Bring Aura GUI window to front and focus it
    Called after GUI is launched to ensure it's visible
    """
    try:
        time.sleep(0.5)  # Give GUI time to create window
        
        # Try wmctrl first (most reliable)
        try:
            subprocess.run(["wmctrl", "-a", "Aura"], check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[Aura] ✅ GUI window brought to front (wmctrl)")
            return
        except FileNotFoundError:
            pass
        
        # Fallback: Use xdotool to find and activate window
        try:
            # Find window by class name
            result = subprocess.run(["xdotool", "search", "--class", "aura"],
                                   capture_output=True, text=True, check=False)
            if result.stdout.strip():
                window_id = result.stdout.strip().split()[0]
                subprocess.run(["xdotool", "windowactivate", window_id], check=False)
                print("[Aura] ✅ GUI window activated (xdotool)")
                return
        except FileNotFoundError:
            pass
        
        print("[Aura] ⚠️  Could not auto-focus GUI window")
        
    except Exception as e:
        print(f"[Aura] ⚠️  Window focus warning: {e}")

# === Main Entrypoint ===
def main():
    print("[Aura] 🌀 Launching Aura...")
    
    # FIRST: Wake screen immediately before anything else
    print("[Aura] 🖥️  Waking display...")
    try:
        subprocess.run(["xset", "dpms", "force", "on"], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xscreensaver-command", "-deactivate"], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Setup display fully (cursor hide, etc.)
    setup_display()
    
    # Launch GUI FIRST so user sees something immediately
    launch_gui()
    
    # Wait for GUI to be fully ready (instead of fixed delay)
    print("[Aura] ⏳ Waiting for GUI to be ready...")
    max_wait = 5.0  # Maximum 5 seconds
    start_time = time.time()
    while not is_gui_ready() and (time.time() - start_time) < max_wait:
        time.sleep(0.05)  # Check every 50ms
    
    if is_gui_ready():
        print("[Aura] ✅ GUI ready - starting services")
    else:
        print("[Aura] ⚠️  GUI ready timeout - continuing anyway")

    # Start TTS warm-up and services in background while GUI is visible
    warm_up_tts()
    threading.Thread(target=start_services, daemon=True).start()
    
    # Bring GUI to front after launch
    threading.Thread(target=focus_gui_window, daemon=True).start()
    
    run_gui_loop()

if __name__ == "__main__":
    main()
