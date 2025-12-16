#!/usr/bin/env python3
"""
Chatterbox-TTS Container REST API
Provides TTS synthesis via Chatterbox with voice cloning support
"""

from flask import Flask, request, jsonify, send_file, Response
import os
import io
import json
import tempfile
import inspect
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import logging

# Suppress verbose logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)

# Load environment variables
load_dotenv('/app/.env') if os.path.exists('/app/.env') else None

# ============================================================================
# Auto-install torchaudio and torchvision if needed
# ============================================================================

def ensure_torch_extensions():
    """Ensure torchaudio and torchvision are installed (required by s3tokenizer)"""
    try:
        import torch
        import subprocess
        import sys
        
        # Check if torchaudio is installed
        try:
            import torchaudio
            print(f"[Chatterbox] ✅ torchaudio {torchaudio.__version__} already installed")
        except ImportError:
            print("[Chatterbox] ⚠️  torchaudio not found, installing...")
            # Create constraints file with only torch (to prevent PyTorch upgrade)
            torch_ver = torch.__version__
            constraints_file = '/tmp/torch_only_constraints.txt'
            with open(constraints_file, 'w') as f:
                f.write(f"torch=={torch_ver}\n")
            
            # Install torchaudio matching PyTorch version
            result = subprocess.run([
                'pip3', 'install', '--index-url', 'https://pypi.org/simple',
                '--constraint', constraints_file,
                'torchaudio'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[Chatterbox] ❌ Failed to install torchaudio: {result.stderr}")
                return False
            
            # Verify installation
            import torchaudio
            print(f"[Chatterbox] ✅ torchaudio {torchaudio.__version__} installed successfully")
        
        # Check if torchvision is installed
        try:
            import torchvision
            print(f"[Chatterbox] ✅ torchvision {torchvision.__version__} already installed")
        except ImportError:
            print("[Chatterbox] ⚠️  torchvision not found, installing...")
            # Create constraints file with only torch (to prevent PyTorch upgrade)
            torch_ver = torch.__version__
            constraints_file = '/tmp/torch_only_constraints.txt'
            with open(constraints_file, 'w') as f:
                f.write(f"torch=={torch_ver}\n")
            
            # Install torchvision matching PyTorch version
            result = subprocess.run([
                'pip3', 'install', '--index-url', 'https://pypi.org/simple',
                '--constraint', constraints_file,
                'torchvision'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[Chatterbox] ❌ Failed to install torchvision: {result.stderr}")
                return False
            
            # Verify installation
            import torchvision
            print(f"[Chatterbox] ✅ torchvision {torchvision.__version__} installed successfully")
        
        # Verify PyTorch CUDA is still intact
        assert hasattr(torch.version, 'cuda') and torch.version.cuda is not None, \
            f'PyTorch CUDA lost after torchaudio/torchvision install! Version: {torch.__version__}'
        print(f"[Chatterbox] ✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} preserved")
        
        return True
        
    except Exception as e:
        print(f"[Chatterbox] ⚠️  Error ensuring torch extensions: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run auto-installation on startup
print("[Chatterbox] 🔍 Checking for torchaudio and torchvision...")
ensure_torch_extensions()

# Initialize Chatterbox TTS (lazy loading)
_chatterbox_tts = None
_chatterbox_voice_embedding = None
VOICE_CACHE_DIR = "/app/voice_cache"
VOICE_SAMPLES_DIR = "/app/voice_samples"

os.makedirs(VOICE_CACHE_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

def get_chatterbox_tts():
    """Lazy load Chatterbox TTS"""
    global _chatterbox_tts
    if _chatterbox_tts is None:
        try:
            print("[Chatterbox] 🔄 Attempting to import ChatterboxTTS...")
            # Try different import paths
            ChatterboxTTS = None
            try:
                from chatterbox.tts import ChatterboxTTS
                print("[Chatterbox] ✅ Imported from chatterbox.tts")
            except ImportError as e1:
                print(f"[Chatterbox] ⚠️ Import from chatterbox.tts failed: {e1}")
                try:
                    from chatterbox import ChatterboxTTS
                    print("[Chatterbox] ✅ Imported from chatterbox")
                except ImportError as e2:
                    print(f"[Chatterbox] ❌ Import from chatterbox failed: {e2}")
                    raise ImportError(f"Could not import ChatterboxTTS: {e1}, {e2}")
            
            if ChatterboxTTS is None:
                raise ImportError("ChatterboxTTS class not found")
            
            # Detect device
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Chatterbox] 🚀 Initializing ChatterboxTTS on {device}...")
            
            # Try different initialization methods
            _chatterbox_tts = None
            
            # Method 1: Try from_pretrained (recommended)
            if hasattr(ChatterboxTTS, 'from_pretrained'):
                try:
                    print("[Chatterbox] 🔧 Trying from_pretrained(device={device})...")
                    # Check if from_pretrained accepts device parameter
                    import inspect
                    if hasattr(ChatterboxTTS, 'from_pretrained'):
                        sig = inspect.signature(ChatterboxTTS.from_pretrained)
                        params = list(sig.parameters.keys())
                        print(f"[Chatterbox] 📋 from_pretrained signature: {params}")
                        
                        if 'device' in params:
                            _chatterbox_tts = ChatterboxTTS.from_pretrained(device=device)
                        else:
                            _chatterbox_tts = ChatterboxTTS.from_pretrained()
                    else:
                        _chatterbox_tts = ChatterboxTTS.from_pretrained(device=device)
                    print("[Chatterbox] ✅ Initialized using from_pretrained()")
                except Exception as e:
                    print(f"[Chatterbox] ⚠️ from_pretrained() failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Method 2: Try with device parameter
            if _chatterbox_tts is None:
                try:
                    print("[Chatterbox] 🔧 Trying ChatterboxTTS(device=...)...")
                    _chatterbox_tts = ChatterboxTTS(device=device)
                    print("[Chatterbox] ✅ Initialized using device parameter")
                except Exception as e:
                    print(f"[Chatterbox] ⚠️ Constructor with device failed: {e}")
            
            # Method 3: Inspect constructor signature to see what's needed
            if _chatterbox_tts is None:
                try:
                    import inspect
                    sig = inspect.signature(ChatterboxTTS.__init__)
                    params = list(sig.parameters.keys())
                    print(f"[Chatterbox] 🔍 Constructor signature: {params}")
                    
                    # If it only needs device, try that
                    if len(params) == 2 and 'device' in params:  # self + device
                        print("[Chatterbox] 🔧 Trying ChatterboxTTS(device=...)...")
                        _chatterbox_tts = ChatterboxTTS(device=device)
                        print("[Chatterbox] ✅ Initialized using device parameter")
                    else:
                        # Try default constructor
                        print("[Chatterbox] 🔧 Trying default constructor...")
                        _chatterbox_tts = ChatterboxTTS()
                        print("[Chatterbox] ✅ Initialized using default constructor")
                except Exception as e:
                    print(f"[Chatterbox] ❌ Initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"Could not initialize ChatterboxTTS. "
                        f"Error: {e}. "
                        f"Please check ChatterboxTTS documentation for correct initialization."
                    )
            
            print(f"[Chatterbox] ✅ ChatterboxTTS initialized successfully")
        except Exception as e:
            print(f"[Chatterbox] ❌ Failed to initialize ChatterboxTTS: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _chatterbox_tts

def get_voice_embedding(voice_sample_path):
    """Get or create voice embedding from sample"""
    global _chatterbox_voice_embedding
    
    if not os.path.exists(voice_sample_path):
        return None
    
    # Check cache
    cache_key = os.path.basename(voice_sample_path).replace('.wav', '.pkl')
    cache_path = os.path.join(VOICE_CACHE_DIR, cache_key)
    
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Extract embedding
    try:
        chatterbox = get_chatterbox_tts()
        
        # Try different methods to extract voice embedding
        if hasattr(chatterbox, 'extract_voice_embedding'):
            embedding = chatterbox.extract_voice_embedding(voice_sample_path)
        elif hasattr(chatterbox, 'get_voice_embedding'):
            embedding = chatterbox.get_voice_embedding(voice_sample_path)
        else:
            # Use audio file directly
            embedding = voice_sample_path
        
        # Cache embedding
        if embedding is not None and not isinstance(embedding, str):
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        
        return embedding
    except Exception as e:
        print(f"[Chatterbox] ⚠️ Failed to extract voice embedding: {e}")
        return None

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "service": "chatterbox-tts",
        "status": "running",
        "message": "Chatterbox-TTS Container API"
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        import sys
        import os
        
        # Check if chatterbox module can be imported
        can_import = False
        import_error = None
        try:
            try:
                from chatterbox.tts import ChatterboxTTS
                can_import = True
            except ImportError:
                from chatterbox import ChatterboxTTS
                can_import = True
        except ImportError as e:
            import_error = str(e)
        
        chatterbox_loaded = _chatterbox_tts is not None
        
        # Check if source directory exists
        source_exists = os.path.exists("/app/chatterbox")
        
        device = "unknown"
        if _chatterbox_tts:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return jsonify({
            "status": "ok",
            "service": "chatterbox-tts",
            "chatterbox_loaded": chatterbox_loaded,
            "can_import_chatterbox": can_import,
            "import_error": import_error,
            "source_directory_exists": source_exists,
            "device": device
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "chatterbox-tts",
            "error": str(e)
        }), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice_sample = data.get('voice_sample', None)  # Path to voice sample file
        exaggeration = float(data.get('exaggeration', 0.6))
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"[Chatterbox] 💬 Synthesizing: '{text[:50]}...'")
        
        # Get Chatterbox instance
        chatterbox = get_chatterbox_tts()
        
        # Handle voice cloning if voice sample provided
        voice_embedding = None
        if voice_sample:
            if os.path.exists(voice_sample):
                voice_embedding = get_voice_embedding(voice_sample)
                print(f"[Chatterbox] 🎭 Using voice cloning from: {voice_sample}")
            else:
                # Try in voice_samples directory
                sample_path = os.path.join(VOICE_SAMPLES_DIR, voice_sample)
                if os.path.exists(sample_path):
                    voice_embedding = get_voice_embedding(sample_path)
                    print(f"[Chatterbox] 🎭 Using voice cloning from: {sample_path}")
        
        # Generate audio
        try:
            if hasattr(chatterbox, 'generate'):
                sig = inspect.signature(chatterbox.generate)
                params = sig.parameters
                
                if voice_embedding:
                    # Try different parameter names
                    if 'voice_embedding' in params:
                        audio = chatterbox.generate(text, voice_embedding=voice_embedding, exaggeration=exaggeration)
                    elif 'audio_prompt' in params:
                        audio = chatterbox.generate(text, audio_prompt=voice_embedding, exaggeration=exaggeration)
                    else:
                        audio = chatterbox.generate(text, exaggeration=exaggeration)
                else:
                    audio = chatterbox.generate(text, exaggeration=exaggeration)
                    
            elif hasattr(chatterbox, 'synthesize'):
                sig = inspect.signature(chatterbox.synthesize)
                params = sig.parameters
                
                if voice_embedding:
                    if 'voice_embedding' in params:
                        audio = chatterbox.synthesize(text, voice_embedding=voice_embedding)
                    elif 'audio_prompt_path' in params:
                        audio = chatterbox.synthesize(text, audio_prompt_path=voice_sample if os.path.exists(voice_sample) else sample_path)
                    else:
                        audio = chatterbox.synthesize(text)
                else:
                    audio = chatterbox.synthesize(text)
            else:
                return jsonify({'error': 'ChatterboxTTS has no generate or synthesize method'}), 500
                
        except Exception as e:
            print(f"[Chatterbox] ❌ Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Synthesis failed: {str(e)}'}), 500
        
        # Convert audio to WAV format
        if isinstance(audio, np.ndarray):
            # Normalize audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, 22050)  # Chatterbox uses 22.05kHz
            
            return send_file(
                temp_file.name,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='output.wav'
            )
        else:
            return jsonify({'error': 'Unexpected audio format'}), 500
            
    except Exception as e:
        print(f"[Chatterbox] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/voice/embedding', methods=['POST'])
def extract_voice_embedding():
    """Extract voice embedding from audio sample"""
    try:
        data = request.get_json()
        voice_sample_path = data.get('voice_sample_path', '')
        
        if not voice_sample_path or not os.path.exists(voice_sample_path):
            return jsonify({'error': 'Voice sample file not found'}), 400
        
        embedding = get_voice_embedding(voice_sample_path)
        
        if embedding is None:
            return jsonify({'error': 'Failed to extract voice embedding'}), 500
        
        # Return embedding info (not the actual embedding data for security)
        return jsonify({
            'success': True,
            'voice_sample': voice_sample_path,
            'embedding_cached': os.path.exists(
                os.path.join(VOICE_CACHE_DIR, os.path.basename(voice_sample_path).replace('.wav', '.pkl'))
            )
        })
        
    except Exception as e:
        print(f"[Chatterbox] ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    import traceback
    
    try:
        print("[Chatterbox] 🚀 Starting Chatterbox-TTS Container...", flush=True)
        print("[Chatterbox] 📦 Chatterbox installed from source (github.com/resemble-ai/chatterbox)", flush=True)
        
        # CRITICAL: Verify CUDA is available at runtime (NO CPU FALLBACK)
        print("[Chatterbox] 🔍 Checking CUDA availability...", flush=True)
        try:
            import torch
            print(f"[Chatterbox] PyTorch version: {torch.__version__}", flush=True)
            
            # Check if PyTorch has CUDA support compiled in
            has_cuda_build = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
            if not has_cuda_build:
                print(f"❌ FATAL ERROR: PyTorch is CPU-only version!", flush=True)
                print(f"   PyTorch {torch.__version__} does not have CUDA support.", flush=True)
                print(f"   This indicates PyTorch was overwritten with CPU version during installation.", flush=True)
                print(f"   Check Dockerfile - dependencies may have installed CPU PyTorch.", flush=True)
                sys.exit(1)
            
            print(f"[Chatterbox] ✅ PyTorch has CUDA build: {torch.version.cuda}", flush=True)
            
            # Check if CUDA is available at runtime (GPU access)
            if not torch.cuda.is_available():
                print("❌ FATAL ERROR: CUDA is not available at runtime!", flush=True)
                print("   PyTorch has CUDA support, but GPU is not accessible.", flush=True)
                print("   This is a runtime configuration issue, not a PyTorch build issue.", flush=True)
                print("   Ensure:", flush=True)
                print("   1. Container is run with --runtime=nvidia", flush=True)
                print("   2. NVIDIA Docker runtime is installed", flush=True)
                print("   3. GPU is accessible to the container", flush=True)
                print("   4. Check: docker info | grep -i nvidia", flush=True)
                print("   5. Check: nvidia-smi (on host)", flush=True)
                sys.exit(1)
            
            print(f"[Chatterbox] ✅ CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"[Chatterbox] ✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}", flush=True)
        except Exception as e:
            print(f"❌ FATAL ERROR during CUDA check: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)
        
        # Try to import Chatterbox at startup (non-blocking - Flask will start anyway)
        print("[Chatterbox] 🔍 Pre-loading ChatterboxTTS (non-blocking)...", flush=True)
        try:
            # Don't block on this - just try it
            get_chatterbox_tts()
            print("[Chatterbox] ✅ ChatterboxTTS pre-loaded successfully", flush=True)
        except Exception as e:
            print(f"[Chatterbox] ⚠️ Pre-loading failed (will retry on first request): {e}", flush=True)
            print("[Chatterbox] 💡 Container will start but synthesis may fail until Chatterbox loads", flush=True)
            # Don't exit - let Flask start anyway
        
        print("[Chatterbox] 🌐 Starting Flask server on 0.0.0.0:11437...", flush=True)
        print("[Chatterbox] ✅ Flask server is running - ready for requests", flush=True)
        
        # Ensure Flask output is not buffered
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        app.run(host="0.0.0.0", port=11437, threaded=True, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n[Chatterbox] 👋 Shutting down...", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"❌ FATAL ERROR during startup: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

