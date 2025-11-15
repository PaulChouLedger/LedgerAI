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

# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from organized directories
from gui.aura_gui import launch_gui, run_gui_loop, is_gui_ready
from listener import listen
import speaker  # ✅ Starts TTS playback loop and queue
try:
    from server.web_upload_server import start_upload_server
    UPLOAD_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"[Aura] ⚠️ Upload server not available: {e}")
    print(f"[Aura] 💡 Install Flask: pip install flask")
    UPLOAD_SERVER_AVAILABLE = False

# Set DISPLAY only if not already set (e.g., for SSH X11 forwarding)
# When running as systemd service on the device, it will be :0
# When running via SSH -X, it will be set by SSH (e.g., localhost:10.0)
if "DISPLAY" not in os.environ or not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"

# Load unified .env from workspace root
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
HOST_ENV = dotenv_values(dotenv_path)
print(f"[Aura] 📋 Loading config from: {dotenv_path}")

# === Whisper Container Configuration ===
# Using faster-whisper with distil-small.en model
WHISPER_IMAGE = "aura-whisper:latest"
WHISPER_NAME = "aura-whisper"
WHISPER_DESCRIPTION = "faster-whisper with distil-small.en"

print(f"[Aura] 🎤 Whisper container: {WHISPER_DESCRIPTION}")

# === Graceful Exit on Ctrl+C ===
def signal_handler(sig, frame):
    print("\n[Aura] ⛔ Exiting gracefully...")
    stop_keyboard_monitor()  # Stop monitoring to allow Ubuntu keyboard to restart
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Ubuntu Keyboard Monitor (disable while main.py is running) ===
_keyboard_monitor_running = False
_keyboard_monitor_thread = None

def monitor_and_disable_ubuntu_keyboard():
    """
    Background thread that periodically kills Ubuntu on-screen keyboard processes
    while main.py is running. When main.py exits, this thread stops and Ubuntu
    keyboard can restart normally.
    """
    global _keyboard_monitor_running
    _keyboard_monitor_running = True
    
    while _keyboard_monitor_running:
        try:
            # Kill any running Ubuntu keyboard processes
            subprocess.run(["pkill", "-f", "onboard"], check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "caribou"], check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "matchbox-keyboard"], check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        
        # Check every 2 seconds
        time.sleep(2)

def start_keyboard_monitor():
    """Start the background thread that monitors and disables Ubuntu keyboard"""
    global _keyboard_monitor_thread
    if _keyboard_monitor_thread is None or not _keyboard_monitor_thread.is_alive():
        _keyboard_monitor_thread = threading.Thread(target=monitor_and_disable_ubuntu_keyboard, daemon=True)
        _keyboard_monitor_thread.start()
        print("[Aura] ✅ Ubuntu keyboard monitor started (will disable while running)")

def stop_keyboard_monitor():
    """Stop the background thread (allows Ubuntu keyboard to restart when main.py exits)"""
    global _keyboard_monitor_running
    _keyboard_monitor_running = False
    print("[Aura] ⌨️  Ubuntu keyboard monitor stopped (keyboard can restart)")

# Register cleanup on exit
import atexit
atexit.register(stop_keyboard_monitor)

# === Display Setup ===
def setup_display():
    """
    Configure display settings for Aura:
    - Wake screen from sleep
    - Dismiss screensaver/lock screen
    - Keep screen on for at least 5 minutes
    - Hide mouse cursor for clean interface
    - Disable Ubuntu on-screen keyboard (use custom GUI keyboard instead)
    """
    print("[Aura] 🖥️  Configuring display...")
    
    # Disable Ubuntu on-screen keyboard - will be monitored in background thread
    print("[Aura] ⌨️  Disabling Ubuntu on-screen keyboard (monitored while running)...")
    try:
        # Initial kill of any running on-screen keyboard processes
        subprocess.run(["pkill", "-f", "onboard"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "caribou"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "matchbox-keyboard"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
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
    """Remove container if it exists"""
    try:
        subprocess.run(["docker", "rm", "-f", name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[Aura] 🗑️ Removed existing container: {name}")
    except subprocess.CalledProcessError:
        pass  # Container doesn't exist, that's fine

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
    
    # For LLM container, provide progress updates
    is_llm = "llm" in name.lower()
    
    for i in range(timeout * 10):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code in (200, 404):
                print(f"[Aura] ✅ {name} is online.")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Show progress every 3 seconds for LLM (model loading takes time)
        if is_llm and i % 30 == 0 and i > 0:
            elapsed = i / 10
            print(f"[Aura] ⏳ Still waiting for {name}... ({elapsed:.0f}s - models loading)")
        
        time.sleep(0.1)
    
    print(f"[Aura] ❌ Timeout waiting for {name} after {timeout}s.")
    return False

# === Old run_container function removed - now using Docker Compose ===

# === ElevenLabs warm-up ===
def warm_up_tts():
    speaker.warm_up_tts()

# === LLM warm-up ===
def warm_up_llm():
    """Warm up LLM with a test request - assumes health check already passed"""
    try:
        print("[Aura] 🧪 Testing LLM with warm-up request...")
        print("[Aura] 💡 First request loads adaptive engine (65 guidelines) - may take 15-20s...")
        
        # Single test request with VERY generous timeout (adaptive engine loads on first request)
        # Get the correct LLM port based on mode
        USE_MEDICAL_MODE = os.environ.get('USE_MEDICAL_MODE', 'true').lower() == 'true'
        llm_port = "11434" if USE_MEDICAL_MODE else "11436"
        
        try:
            response = requests.post(
                f"http://localhost:{llm_port}/chat-tts",
                json={"prompt": "Hello", "session_id": "warmup"},
                stream=True,
                timeout=120  # Increased from 45s - adaptive engine needs time to load
            )
            
            if response.status_code == 200:
                # Consume the stream to complete the request
                for _ in response.iter_lines():
                    pass
                print("[Aura] ✅ LLM warm-up complete.")
                return True
            else:
                print(f"[Aura] ❌ LLM returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[Aura] ❌ LLM warm-up failed: {e}")
            print(f"[Aura] 💡 Check LLM logs: docker logs aura-llm")
            return False
            
    except Exception as e:
        print(f"[Aura] ❌ LLM warm-up exception: {e}")
        return False

# === RAG warm-up ===
def initialize_rag_delayed():
    """Initialize RAG system after core services are stable"""
    try:
        # Check RAG mode
        RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
        
        if RAG_MODE == 'GPU':
            # GPU mode: Initialize external RAG container
            print("[Aura] 🔍 RAG initialization starting...")
            print("[Aura] 🔍 Initializing RAG container (RAG_MODE=GPU)...")
            
            # Initialize RAG system (longer timeout for CUDA model loading on first boot)
            for attempt in range(3):
                try:
                    init_response = requests.post("http://localhost:11435/rag/init", timeout=90)
                    if init_response.status_code == 200:
                        result = init_response.json()
                        if result.get("status") == "success":
                            print("[Aura] ✅ RAG container initialized successfully")
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
                    print("[Aura] ⚠️ RAG container initialization failed after 3 attempts")
        else:
            # CPU mode: RAG is initialized within LLM containers, no external initialization needed
            print("[Aura] 🔍 RAG_MODE=CPU - CPU FAISS initialized within LLM containers")
            print("[Aura] ✅ RAG system ready (no external container needed)")
            return
        
        # GPU mode: Continue with RAG container stats and search verification
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
    
    # Check RAG mode (GPU = RAG container, CPU = CPU FAISS in LLM containers)
    RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
    
    # Check which LLM mode to use
    USE_MEDICAL_MODE = os.environ.get('USE_MEDICAL_MODE', 'true').lower() == 'true'
    
    if RAG_MODE == 'GPU':
        print("[Aura] 🚀 Starting all containers in parallel (RAG_MODE=GPU - using RAG container)...")
    else:
        print("[Aura] 🚀 Starting containers (RAG_MODE=CPU - using CPU FAISS in LLM containers)...")
    
    # Start all containers using Docker Compose
    def start_all_containers():
        print("[Aura] 🚀 Starting all containers via Docker Compose...")
        try:
            # Get workspace root (LedgerAI directory)
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            setup_dir = os.path.join(workspace_root, 'setup')
            
            # Stop any existing containers first
            print("[Aura] 🛑 Stopping existing containers...")
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=setup_dir,
                capture_output=True,
                check=False  # Don't fail if nothing to stop
            )
            
            # Determine which LLM container to start based on USE_MEDICAL_MODE
            if USE_MEDICAL_MODE:
                llm_service = "llm-medical"
                print("[Aura] 🏥 Medical Mode - starting llm-medical container")
            else:
                llm_service = "llm-generic"
                print("[Aura] 💬 Generic Mode - starting llm-generic container")
            
            # Determine which services to start
            services_to_start = ["whisper", llm_service]
            if RAG_MODE == 'GPU':
                services_to_start.append("rag")
                print("[Aura] 🔍 RAG_MODE=GPU - starting all containers including RAG container")
            else:
                print("[Aura] ⏭️  RAG_MODE=CPU - starting Whisper and LLM only (CPU FAISS in LLM containers)")
            
            # Start selected services with Docker Compose (different ports for each LLM)
            cmd = ["docker", "compose", "up", "-d"] + services_to_start
            
            result = subprocess.run(
                cmd,
                cwd=setup_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[Aura] ❌ Failed to start containers: {result.stderr}")
                return False
            
            print(f"[Aura] ✅ Started containers: {', '.join(services_to_start)}")
            
            # Wait for containers to be ready
            print("[Aura] ⏳ Waiting for containers to be ready...")
            
            # Check Whisper
            whisper_ready = wait_for_container("http://localhost:5000/health", "Whisper", timeout=15)
            if not whisper_ready:
                return False
            
            # Check LLM (different ports for medical vs generic)
            if USE_MEDICAL_MODE:
                llm_ready = wait_for_container("http://localhost:11434/health", "LLM", timeout=30)
            else:
                llm_ready = wait_for_container("http://localhost:11436/health", "LLM", timeout=30)
            if not llm_ready:
                return False
            
            # Check RAG container if GPU mode
            if RAG_MODE == 'GPU':
                rag_ready = wait_for_container("http://localhost:11435/health", "RAG", timeout=90)
                if not rag_ready:
                    return False
            
            print("[Aura] ✅ All containers are ready!")
            return True
            
        except Exception as e:
            print(f"[Aura] ❌ Error starting containers: {e}")
            return False
    
    # Start all containers using Docker Compose
    if not start_all_containers():
        print("[Aura] ❌ Failed to start containers. Aborting.")
        return
    
    # Wait for LLM Flask API and model to be fully ready
    print("[Aura] ⏳ Waiting for LLM Flask API and model to load...")
    
    # Get model name from health endpoint or environment
    model_name = "LLM model"
    llm_port = "11434" if USE_MEDICAL_MODE else "11436"
    try:
        # Try to get model name from health endpoint
        response = requests.get(f"http://localhost:{llm_port}/health", timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            models = health_data.get("models", {})
            model_path = models.get("simple_path", "")
            if model_path:
                # Extract model name from path (os is already imported at top of file)
                model_name = os.path.basename(model_path).replace('.gguf', '').replace('.ggml', '')
    except:
        # Fallback to environment variable if available
        model_name = HOST_ENV.get("SIMPLE_MODEL_PATH", "LLM model")
        if '/' in model_name:
            # Extract model name from path (os is already imported at top of file)
            model_name = os.path.basename(model_name).replace('.gguf', '').replace('.ggml', '')
    
    print(f"[Aura] 💡 Loading in background: {model_name} (~1s)...")
    
    # Wait up to 60 seconds for model to load
    model_ready = False
    for attempt in range(24):  # 24 attempts * 2.5 seconds = 60 seconds max
        try:
            response = requests.get(f"http://localhost:{llm_port}/health", timeout=2)
            if response.status_code == 200:
                health_data = response.json()
                models = health_data.get("models", {})
                
                # Check if model is loaded
                simple_loaded = models.get("simple_loaded", False)
                
                if simple_loaded:
                    model_ready = True
                    elapsed = (attempt + 1) * 2.5
                    print(f"[Aura] ✅ Model loaded after {elapsed:.1f} seconds")
                    break
                else:
                    if attempt % 4 == 0:  # Print every 10 seconds
                        print(f"[Aura] ⏳ Model still loading... ({(attempt + 1) * 2.5:.1f}s)")
        except:
            if attempt % 4 == 0:
                print(f"[Aura] ⏳ Waiting for API... ({(attempt + 1) * 2.5:.1f}s)")
        
        time.sleep(2.5)
    
    if not model_ready:
        print("[Aura] ❌ Model did not load after 60 seconds")
        print("[Aura] 💡 Check: docker logs aura-llm")
        return
    
    # Step 4: Warm up LLM (without RAG first)
    if not warm_up_llm():
        print("[Aura] ❌ LLM warm-up failed. Aborting.")
        return
    
    # Step 5: Initialize RAG immediately after LLM warm-up (RAG is always enabled, mode determines implementation)
    RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
    print(f"[Aura] 🔍 Initializing RAG system (RAG_MODE={RAG_MODE})...")
    initialize_rag_delayed()  # Run synchronously, not in thread
    
    # Step 6: Start file upload server (if available)
    if UPLOAD_SERVER_AVAILABLE:
        print("[Aura] 📁 Starting file upload server...")
        upload_ip, upload_port = start_upload_server(port=5001)
        print(f"[Aura] 📱 Upload server ready at http://{upload_ip}:{upload_port}")
    else:
        print("[Aura] ⚠️ Upload server skipped (Flask not available)")
    
    # Step 7: Note about data ingestion (handled by convert_and_ingest_all() if needed)
    # This runs automatically in background if new medical guidelines or data files detected
    print("[Aura] ℹ️ Data ingestion: Handled by background process if new files detected")
    print("[Aura] ℹ️ Auto-ingest: Triggered by file uploads via web server")
    
    # Step 8: Final RAG ready check before starting listener (if GPU mode)
    RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
    if RAG_MODE == 'GPU':
        print("[Aura] 🔍 Final RAG container ready check before starting listener...")
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
    else:
        print("[Aura] ✅ Using CPU mode - ready to start listener")
    
    # Step 9: Start listener
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

# === Auto-Convert New Medical Guidelines ===
def check_for_new_guidelines_quick():
    """
    Quick check if new guidelines exist (doesn't convert)
    Returns True if new JSONs found, False otherwise
    """
    try:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        guidelines_dir = os.path.join(workspace_root, 'llm-container', 'medical', 'guidelines')
        output_dir = os.path.join(workspace_root, 'data', 'input')
        
        if not os.path.exists(guidelines_dir):
            return False
        
        json_files = [f for f in os.listdir(guidelines_dir) if f.endswith('.json')]
        
        if not json_files:
            return False
        
        # Check for any missing RAG files
        for json_file in json_files:
            txt_filename = f"GUIDELINE_{json_file.replace('.json', '.txt')}"
            txt_path = os.path.join(output_dir, txt_filename)
            
            if not os.path.exists(txt_path):
                return True  # Found at least one new guideline
        
        # All already converted
        print(f"[Aura] ✅ All {len(json_files)} medical guidelines already converted - skipping rebuild")
        return False
    
    except Exception as e:
        print(f"[Aura] ⚠️ Error checking guidelines: {e}")
        return False


def convert_and_ingest_all():
    """
    Background task to process medical guidelines and ingest all data
    
    Two-stage process:
    1. Convert medical guidelines (JSON → TXT in data/input/)
    2. Wait for RAG container, then ingest all data and rebuild embeddings
    """
    print("[Aura] 🔄 Starting background data processing...")
    
    # STAGE 1: Convert medical guidelines (if any new ones exist)
    # This can run immediately - no container dependencies
    guidelines_converted = convert_medical_guidelines()
    
    # STAGE 2: Ingest ALL files from data/input/ (guidelines + any other files)
    # This waits for RAG container to be ready, then processes ALL file types
    ingest_and_rebuild_embeddings()


def convert_medical_guidelines():
    """
    STAGE 1: Convert medical guidelines (JSON) to RAG-ready format
    
    Checks llm-container/medical/guidelines/ for JSON files
    Converts them to TXT and saves to data/input/
    
    Returns: True if new guidelines were converted, False otherwise
    """
    try:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        guidelines_dir = os.path.join(workspace_root, 'llm-container', 'medical', 'guidelines')
        output_dir = os.path.join(workspace_root, 'data', 'input')
        
        if not os.path.exists(guidelines_dir):
            return False
        
        # Check for new guidelines
        json_files = [f for f in os.listdir(guidelines_dir) if f.endswith('.json')]
        
        if not json_files:
            return False
        
        new_guidelines = []
        for json_file in json_files:
            # Expected output filename
            txt_filename = f"GUIDELINE_{json_file.replace('.json', '.txt')}"
            txt_path = os.path.join(output_dir, txt_filename)
            
            # Check if RAG file exists
            if not os.path.exists(txt_path):
                new_guidelines.append(json_file)
        
        if not new_guidelines:
            print(f"[Aura] ✅ All {len(json_files)} medical guidelines already RAG-ready")
            return False
        
        print(f"[Aura] 📋 Converting {len(new_guidelines)} medical guidelines to RAG format...")
        for guideline in new_guidelines[:5]:  # Show first 5
            print(f"[Aura]    - {guideline}")
        if len(new_guidelines) > 5:
            print(f"[Aura]    ... and {len(new_guidelines) - 5} more")
        
        # Run converter script
        converter_script = os.path.join(workspace_root, 'medical', 'convert_guidelines_to_rag.py')
        
        if not os.path.exists(converter_script):
            print(f"[Aura] ⚠️ Converter script not found: {converter_script}")
            return False
        
        result = subprocess.run(
            ["python3", converter_script],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"[Aura] ✅ Medical guidelines converted and saved to data/input/")
            return True
        else:
            print(f"[Aura] ⚠️ Guideline conversion failed:")
            print(result.stderr[:500])
            return False
    
    except Exception as e:
        print(f"[Aura] ⚠️ Error in guideline conversion: {e}")
        return False


def ingest_and_rebuild_embeddings():
    """
    STAGE 2: Universal data ingestion from data/input/
    
    Waits for RAG container, then:
    1. Triggers /rag/ingest (extracts PDFs, copies TXT files to data/parsed/)
    2. Rebuilds embeddings from data/parsed/
    3. Reloads RAG container with new embeddings
    
    Handles ALL file types: PDFs, TXT, DOCX, etc.
    """
    try:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Check RAG mode
        RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
        
        # Only wait for RAG container if GPU mode
        if RAG_MODE == 'GPU':
            print(f"[Aura] 📂 Waiting for RAG container to be ready for data ingestion...")
            
            import requests
            import time
            
            # Wait up to 30 seconds for RAG container to be ready
            rag_ready = False
            for attempt in range(30):
                try:
                    health_check = requests.get("http://localhost:11435/health", timeout=2)
                    if health_check.status_code == 200:
                        rag_ready = True
                        print(f"[Aura] ✅ RAG container ready")
                        break
                except:
                    if attempt < 29:
                        time.sleep(1)
                    else:
                        print(f"[Aura] ⚠️ RAG container not responding - skipping data ingestion")
                        return
            
            if not rag_ready:
                return
        
        print(f"[Aura] 📂 Checking data/input/ for new files to ingest...")
        
        # Step 1: Trigger ingest based on RAG_MODE setting
        import threading
        
        # Check RAG_MODE from environment (GPU = RAG container, CPU = CPU FAISS)
        RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
        
        def trigger_cpu_rag_medical():
            try:
                cpu_response = requests.post("http://localhost:11434/cpu-faiss/ingest", timeout=30)
                if cpu_response.status_code == 200:
                    print("[Aura] ✅ Medical CPU FAISS ingest triggered")
                else:
                    print(f"[Aura] ⚠️ Medical CPU FAISS ingest failed: HTTP {cpu_response.status_code}")
            except Exception as e:
                print(f"[Aura] ⚠️ Medical CPU FAISS ingest error: {e}")
        
        def trigger_cpu_rag_generic():
            try:
                cpu_response = requests.post("http://localhost:11436/cpu-faiss/ingest", timeout=30)
                if cpu_response.status_code == 200:
                    print("[Aura] ✅ Generic CPU FAISS ingest triggered")
                else:
                    print(f"[Aura] ⚠️ Generic CPU FAISS ingest failed: HTTP {cpu_response.status_code}")
            except Exception as e:
                print(f"[Aura] ⚠️ Generic CPU FAISS ingest error: {e}")
        
        if RAG_MODE == 'GPU':
            # GPU RAG mode: Use RAG container only (skip CPU FAISS)
            print("[Aura] 🚀 RAG_MODE=GPU - using RAG container")
            def trigger_gpu_rag():
                try:
                    ingest_response = requests.post("http://localhost:11435/rag/ingest", timeout=30)
                    if ingest_response.status_code == 200:
                        result = ingest_response.json()
                        print(f"[Aura] ✅ GPU RAG ingest: {result.get('processed', 0)} processed, {result.get('skipped', 0)} skipped")
                        return result
                    else:
                        print(f"[Aura] ⚠️ GPU RAG ingest failed: HTTP {ingest_response.status_code}")
                        return None
                except Exception as e:
                    print(f"[Aura] ⚠️ GPU RAG ingest error: {e}")
                    return None
            
            # Trigger GPU RAG only
            result = trigger_gpu_rag()
            if result:
                ingest_response = type('obj', (object,), {'status_code': 200, 'json': lambda: result})()
            else:
                ingest_response = type('obj', (object,), {'status_code': 500})()
        else:
            # CPU RAG mode: Use CPU FAISS only (skip GPU RAG container)
            print("[Aura] 💻 RAG_MODE=CPU - using CPU FAISS in LLM containers")
            cpu_medical_thread = threading.Thread(target=trigger_cpu_rag_medical, daemon=True)
            cpu_generic_thread = threading.Thread(target=trigger_cpu_rag_generic, daemon=True)
            
            cpu_medical_thread.start()
            cpu_generic_thread.start()
            
            cpu_medical_thread.join(timeout=30)
            cpu_generic_thread.join(timeout=30)
            
            # Dummy response for compatibility (CPU FAISS handles its own state)
            ingest_response = type('obj', (object,), {'status_code': 200, 'json': lambda: {'processed': 0, 'skipped': 0}})()
        
        if ingest_response.status_code == 200:
            ingest_result = ingest_response.json()
            processed = ingest_result.get('processed', 0)
            skipped = ingest_result.get('skipped', 0)
            
            print(f"[Aura] ✅ RAG ingest: {processed} processed, {skipped} skipped")
            
            # Only rebuild if new files were processed
            if processed > 0:
                # Step 2: Rebuild embeddings on host (from data/parsed)
                print(f"[Aura] 🔄 Rebuilding embeddings with new data...")
                embed_script = os.path.join(workspace_root, 'setup', 'scripts', 'rebuild_embeddings_host.py')
                
                if os.path.exists(embed_script):
                    embed_result = subprocess.run(
                        ["python3", embed_script],
                        cwd=workspace_root,
                        capture_output=True,
                        text=True,
                        timeout=180
                    )
                    
                    if embed_result.returncode == 0:
                        print(f"[Aura] ✅ Embeddings rebuilt successfully")
                        
                        # Step 3: Reload RAG container with new embeddings
                        print(f"[Aura] 🔄 Reloading RAG with new embeddings...")
                        reload_response = requests.post("http://localhost:11435/rag/reload", timeout=10)
                        
                        if reload_response.status_code == 200:
                            reload_result = reload_response.json()
                            total_chunks = reload_result.get('total_chunks', 0)
                            print(f"[Aura] ✅ RAG reloaded: {total_chunks} total chunks available")
                        else:
                            print(f"[Aura] ⚠️ RAG reload failed: HTTP {reload_response.status_code}")
                    else:
                        print(f"[Aura] ⚠️ Embedding rebuild failed:")
                        print(embed_result.stderr[:500])
                else:
                    print(f"[Aura] ⚠️ Embedding script not found")
            else:
                print(f"[Aura] ℹ️ No new files to process - embeddings up to date")
        else:
            print(f"[Aura] ⚠️ RAG ingest failed: HTTP {ingest_response.status_code}")
    
    except Exception as e:
        print(f"[Aura] ⚠️ Error in data ingestion: {e}")
        import traceback
        traceback.print_exc()


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
    
    # Start keyboard monitor (disables Ubuntu keyboard while running)
    start_keyboard_monitor()
    
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

    # Start TTS warm-up FIRST (fast, shows system is responsive)
    warm_up_tts()
    
    # Check for new medical guidelines (but don't rebuild embeddings yet if found)
    # This is quick - just checks if new JSONs exist
    new_guidelines_exist = check_for_new_guidelines_quick()
    
    if new_guidelines_exist:
        print("[Aura] 📋 New data detected in medical guidelines or data/input/ - will process in background")
        # Start services first (user can interact immediately)
        threading.Thread(target=start_services, daemon=True).start()
        # Convert and ingest all data in background
        threading.Thread(target=convert_and_ingest_all, daemon=True).start()
    else:
        # No new data - start immediately
        threading.Thread(target=start_services, daemon=True).start()
    
    # Bring GUI to front after launch
    threading.Thread(target=focus_gui_window, daemon=True).start()
    
    run_gui_loop()

if __name__ == "__main__":
    main()
