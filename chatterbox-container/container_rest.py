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
import threading

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

# Check if perth module is working and fix if needed
def check_and_fix_perth_module():
    """Check if perth module can be imported and PerthImplicitWatermarker is available.
    If PerthImplicitWatermarker is None, try to use DummyWatermarker as fallback.
    """
    try:
        import perth
        print(f"[Chatterbox] ✅ perth module imported")
        print(f"[Chatterbox] 📋 perth attributes: {[attr for attr in dir(perth) if not attr.startswith('_')]}")
        
        # Try to get the actual class, checking if it's None
        if hasattr(perth, 'PerthImplicitWatermarker'):
            watermarker_class = getattr(perth, 'PerthImplicitWatermarker')
            print(f"[Chatterbox] 🔍 PerthImplicitWatermarker value: {watermarker_class}")
            print(f"[Chatterbox] 🔍 PerthImplicitWatermarker type: {type(watermarker_class)}")
            
            if watermarker_class is None:
                print("[Chatterbox] ⚠️  PerthImplicitWatermarker is None - attempting to fix...")
                
                # Try importing directly from submodule
                try:
                    from perth.perth_net import PerthImplicitWatermarker as DirectWatermarker
                    if DirectWatermarker is not None:
                        print("[Chatterbox] ✅ Found PerthImplicitWatermarker in perth.perth_net")
                        perth.PerthImplicitWatermarker = DirectWatermarker
                        print("[Chatterbox] 🔧 Patched perth.PerthImplicitWatermarker to use direct import")
                        return True
                except (ImportError, AttributeError) as e:
                    print(f"[Chatterbox] ⚠️  Direct import failed: {e}")
                
                # Fallback: Try to use DummyWatermarker
                if hasattr(perth, 'DummyWatermarker'):
                    dummy_class = getattr(perth, 'DummyWatermarker')
                    if dummy_class is not None:
                        print("[Chatterbox] ✅ DummyWatermarker is available as fallback")
                        # Monkey-patch PerthImplicitWatermarker to use DummyWatermarker
                        perth.PerthImplicitWatermarker = dummy_class
                        print("[Chatterbox] 🔧 Patched PerthImplicitWatermarker to use DummyWatermarker")
                        return True
                    else:
                        print("[Chatterbox] ⚠️  DummyWatermarker is also None")
                
                print("[Chatterbox] ❌ Could not fix PerthImplicitWatermarker - initialization may fail")
                return False
            else:
                print(f"[Chatterbox] ✅ PerthImplicitWatermarker is available: {watermarker_class}")
                return True
        else:
            print("[Chatterbox] ❌ PerthImplicitWatermarker not found in perth module")
            return False
    except ImportError as e:
        print(f"[Chatterbox] ❌ Failed to import perth: {e}")
        print("[Chatterbox] 💡 resemble-perth may not be installed correctly")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[Chatterbox] ⚠️  Error checking perth module: {e}")
        import traceback
        traceback.print_exc()
        return False

print("[Chatterbox] 🔍 Checking and fixing perth module...")
perth_ok = check_and_fix_perth_module()

# Initialize Chatterbox TTS (lazy loading)
_chatterbox_tts = None
_chatterbox_voice_embedding = None
_initialization_lock = threading.Lock()
_initialization_in_progress = False
_initialization_error = None

VOICE_CACHE_DIR = "/app/voice_cache"
VOICE_SAMPLES_DIR = "/app/voice_samples"

os.makedirs(VOICE_CACHE_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

def get_chatterbox_tts():
    """Lazy load Chatterbox TTS with thread-safe initialization"""
    global _chatterbox_tts, _initialization_in_progress, _initialization_error
    
    # If already initialized, return it
    if _chatterbox_tts is not None:
        return _chatterbox_tts
    
    # Use lock to prevent multiple simultaneous initialization attempts
    with _initialization_lock:
        # Double-check after acquiring lock (another thread might have initialized it)
        if _chatterbox_tts is not None:
            return _chatterbox_tts
        
        # If initialization failed previously, raise the error
        if _initialization_error is not None:
            raise RuntimeError(f"ChatterboxTTS initialization previously failed: {_initialization_error}")
        
        # If initialization is in progress, wait until it completes or fails
        if _initialization_in_progress:
            print("[Chatterbox] ⏳ Initialization already in progress, waiting...")
            print("[Chatterbox] 💡 This may take several minutes (loading large models into GPU memory)...")
            import time
            wait_start = time.time()
            max_wait = 600  # Wait up to 10 minutes (model loading can be slow, especially on first load)
            check_interval = 2  # Check every 2 seconds
            
            while time.time() - wait_start < max_wait:
                time.sleep(check_interval)
                if _chatterbox_tts is not None:
                    elapsed = time.time() - wait_start
                    print(f"[Chatterbox] ✅ Initialization completed (waited {elapsed:.1f} seconds)")
                    return _chatterbox_tts
                if _initialization_error is not None:
                    raise RuntimeError(f"ChatterboxTTS initialization failed: {_initialization_error}")
                
                # Show progress every 30 seconds
                elapsed = time.time() - wait_start
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    print(f"[Chatterbox] ⏳ Still waiting for initialization... ({int(elapsed)}s / {max_wait}s)")
            
            raise RuntimeError(f"ChatterboxTTS initialization timed out (waited {max_wait} seconds)")
        
        # Mark initialization as in progress
        _initialization_in_progress = True
        
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
            
            # Use from_pretrained (the recommended and only supported method)
            if not hasattr(ChatterboxTTS, 'from_pretrained'):
                raise RuntimeError(
                    "ChatterboxTTS.from_pretrained() method not found. "
                    "This is the required initialization method."
                )
            
            try:
                print(f"[Chatterbox] 🔧 Initializing with from_pretrained(device={device})...")
                # Check if from_pretrained accepts device parameter
                import inspect
                sig = inspect.signature(ChatterboxTTS.from_pretrained)
                params = list(sig.parameters.keys())
                print(f"[Chatterbox] 📋 from_pretrained signature: {params}")
                
                # Check if models are already cached
                from pathlib import Path
                cache_dir = Path.home() / '.cache' / 'huggingface'
                models_cached = cache_dir.exists() and any(cache_dir.rglob('*'))
                
                if models_cached:
                    cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    print(f"[Chatterbox] ✅ Models found in cache: {cache_size / (1024**3):.2f} GB")
                    print(f"[Chatterbox] 💡 Using cached models from {cache_dir}")
                    print("[Chatterbox] ⏳ Loading models (should be fast - already downloaded)...")
                else:
                    print("[Chatterbox] ⚠️  Models not found in cache - will download from HuggingFace")
                    print(f"[Chatterbox] 💡 Cache directory: {cache_dir}")
                    print("[Chatterbox] ⏳ This may take a while (downloading/loading models)...")
                    print("[Chatterbox] 💡 First download can take several minutes depending on network speed")
                    print("[Chatterbox] 💡 Models will be cached for future use")
                    print("[Chatterbox] 💡 If this hangs, check:")
                    print("[Chatterbox]    - Internet connectivity")
                    print("[Chatterbox]    - Disk space (models are ~2-3GB)")
                    print("[Chatterbox]    - HuggingFace access (may need token if gated)")
                
                # Set HuggingFace cache directory if specified
                custom_cache = os.environ.get('HUGGINGFACE_CACHE_DIR', None)
                if custom_cache:
                    print(f"[Chatterbox] 📦 Using custom HuggingFace cache: {custom_cache}")
                    os.environ['HF_HOME'] = custom_cache
                
                if 'device' in params:
                    print(f"[Chatterbox] 🔄 Calling ChatterboxTTS.from_pretrained(device={device})...")
                    print("[Chatterbox] ⏳ This may take 1-5 minutes (loading models into GPU memory)...")
                    if device == "cpu":
                        print("[Chatterbox] ⚠️  WARNING: Using CPU - synthesis will be VERY slow (10-30x slower than GPU)")
                    import time
                    init_start = time.time()
                    _chatterbox_tts = ChatterboxTTS.from_pretrained(device=device)
                    init_elapsed = time.time() - init_start
                    print(f"[Chatterbox] ✅ from_pretrained(device=...) returned (took {init_elapsed:.1f} seconds)")
                    
                    # Verify device after loading
                    if hasattr(_chatterbox_tts, 'device'):
                        actual_device = str(_chatterbox_tts.device)
                        print(f"[Chatterbox] 🔍 Model device: {actual_device}")
                        if actual_device != device:
                            print(f"[Chatterbox] ⚠️  Device mismatch: requested {device}, got {actual_device}")
                    elif hasattr(_chatterbox_tts, 'model') and hasattr(_chatterbox_tts.model, 'device'):
                        actual_device = str(_chatterbox_tts.model.device)
                        print(f"[Chatterbox] 🔍 Model device: {actual_device}")
                        if actual_device != device:
                            print(f"[Chatterbox] ⚠️  Device mismatch: requested {device}, got {actual_device}")
                else:
                    print("[Chatterbox] 🔄 Calling ChatterboxTTS.from_pretrained()...")
                    _chatterbox_tts = ChatterboxTTS.from_pretrained()
                    print("[Chatterbox] ✅ from_pretrained() returned")
                    # If device parameter not available, try to move model to device manually
                    if hasattr(_chatterbox_tts, 'to'):
                        print(f"[Chatterbox] 🔄 Moving model to device {device}...")
                        _chatterbox_tts = _chatterbox_tts.to(device)
                        print(f"[Chatterbox] ✅ Model moved to device {device}")
                
                print("[Chatterbox] ✅ Successfully initialized using from_pretrained()")
            except KeyboardInterrupt:
                print("[Chatterbox] ⚠️  Initialization interrupted")
                raise
            except Exception as e:
                print(f"[Chatterbox] ❌ from_pretrained() failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Could not initialize ChatterboxTTS using from_pretrained(). "
                    f"Error: {e}. "
                    f"Please check container logs for details."
                )
            
            print(f"[Chatterbox] ✅ ChatterboxTTS initialized successfully")
            return _chatterbox_tts
        except Exception as e:
            print(f"[Chatterbox] ❌ Failed to initialize ChatterboxTTS: {e}")
            import traceback
            traceback.print_exc()
            _initialization_error = str(e)
            _initialization_in_progress = False
            raise
        finally:
            _initialization_in_progress = False
    
    return _chatterbox_tts

def get_voice_embedding(voice_sample_path):
    """Get or create voice embedding from sample"""
    global _chatterbox_voice_embedding
    
    print(f"[Chatterbox] 🔍 Looking for voice sample: {voice_sample_path}")
    if not os.path.exists(voice_sample_path):
        print(f"[Chatterbox] ⚠️  Voice sample not found: {voice_sample_path}")
        return None
    
    print(f"[Chatterbox] ✅ Voice sample found: {voice_sample_path}")
    
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
            print(f"[Chatterbox] 🔧 Attempting to extract voice embedding...")
            if hasattr(chatterbox, 'extract_voice_embedding'):
                print(f"[Chatterbox] 🔧 Using extract_voice_embedding() method")
                embedding = chatterbox.extract_voice_embedding(voice_sample_path)
            elif hasattr(chatterbox, 'get_voice_embedding'):
                print(f"[Chatterbox] 🔧 Using get_voice_embedding() method")
                embedding = chatterbox.get_voice_embedding(voice_sample_path)
            else:
                # Use audio file directly (Chatterbox will load it)
                print(f"[Chatterbox] 🔧 No embedding extraction method - will use file path directly")
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
        
        # Better device detection
        import torch
        device = "unknown"
        if _chatterbox_tts:
            # Check if model is on GPU
            if hasattr(_chatterbox_tts, 'device'):
                device = str(_chatterbox_tts.device)
            elif hasattr(_chatterbox_tts, 'model') and hasattr(_chatterbox_tts.model, 'device'):
                device = str(_chatterbox_tts.model.device)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
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
        
        # Track timing
        import time
        synthesis_start = time.time()
        
        # Get Chatterbox instance
        chatterbox = get_chatterbox_tts()
        
        # Verify device
        import torch
        if torch.cuda.is_available():
            print(f"[Chatterbox] 🚀 CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[Chatterbox] 🚀 CUDA memory: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB allocated")
        else:
            print("[Chatterbox] ⚠️  CUDA not available - using CPU (will be slow!)")
        
        # Handle voice cloning if voice sample provided
        voice_embedding = None
        voice_sample_path_used = None
        
        if voice_sample:
            # Try multiple paths: direct path, voice_samples directory, prompts directory
            possible_paths = [
                voice_sample,  # Direct path (absolute or relative)
                os.path.join(VOICE_SAMPLES_DIR, voice_sample),  # In voice_samples directory
                os.path.join("/app/prompts", voice_sample),  # In prompts directory (if mounted)
            ]
            
            # Also try with just the filename if a full path was provided
            if os.path.sep in voice_sample:
                possible_paths.append(os.path.basename(voice_sample))
            
            for sample_path in possible_paths:
                if os.path.exists(sample_path):
                    voice_embedding = get_voice_embedding(sample_path)
                    voice_sample_path_used = sample_path
                    print(f"[Chatterbox] 🎭 Using voice cloning from: {sample_path}")
                    break
            
            if voice_embedding is None:
                print(f"[Chatterbox] ⚠️  Voice sample '{voice_sample}' not found in any location")
                print(f"[Chatterbox]    Checked paths:")
                for path in possible_paths:
                    exists = os.path.exists(path)
                    print(f"[Chatterbox]      {'✅' if exists else '❌'} {path}")
                print(f"[Chatterbox]    Falling back to default voice (no cloning)")
            elif isinstance(voice_embedding, str):
                # If embedding is just a path string, use it directly as audio_prompt_path
                voice_sample_path_used = voice_embedding
                print(f"[Chatterbox] 🎭 Will use audio_prompt_path for voice cloning")
        
        # Generate audio
        try:
            gen_start = time.time()
            
            # Debug: Print available methods and parameters
            print(f"[Chatterbox] 🔍 Available methods: generate={hasattr(chatterbox, 'generate')}, synthesize={hasattr(chatterbox, 'synthesize')}")
            
            if hasattr(chatterbox, 'generate'):
                sig = inspect.signature(chatterbox.generate)
                params = list(sig.parameters.keys())
                print(f"[Chatterbox] 🔍 generate() parameters: {params}")
                
                if voice_embedding or voice_sample_path_used:
                    # Try audio_prompt_path first (most common in Chatterbox)
                    if voice_sample_path_used and 'audio_prompt_path' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using audio_prompt_path='{voice_sample_path_used}'")
                        audio = chatterbox.generate(text, audio_prompt_path=voice_sample_path_used, exaggeration=exaggeration)
                    # Try audio_prompt (alternative name)
                    elif voice_sample_path_used and 'audio_prompt' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using audio_prompt='{voice_sample_path_used}'")
                        audio = chatterbox.generate(text, audio_prompt=voice_sample_path_used, exaggeration=exaggeration)
                    # Try voice_embedding (if we have an embedding object)
                    elif voice_embedding and not isinstance(voice_embedding, str) and 'voice_embedding' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using voice_embedding object")
                        audio = chatterbox.generate(text, voice_embedding=voice_embedding, exaggeration=exaggeration)
                    # Last resort: try passing path as audio_prompt even if parameter name differs
                    elif voice_sample_path_used:
                        print(f"[Chatterbox] ⚠️  Trying voice cloning with path as positional/kwarg...")
                        try:
                            # Try common parameter variations
                            audio = chatterbox.generate(text, audio_prompt_path=voice_sample_path_used, exaggeration=exaggeration)
                            print(f"[Chatterbox] ✅ Voice cloning succeeded with audio_prompt_path")
                        except TypeError:
                            try:
                                audio = chatterbox.generate(text, audio_prompt=voice_sample_path_used, exaggeration=exaggeration)
                                print(f"[Chatterbox] ✅ Voice cloning succeeded with audio_prompt")
                            except TypeError:
                                print(f"[Chatterbox] ❌ Voice cloning failed - parameter not recognized")
                                print(f"[Chatterbox]    Available parameters: {params}")
                                print(f"[Chatterbox]    Falling back to default voice")
                                audio = chatterbox.generate(text, exaggeration=exaggeration)
                    else:
                        print(f"[Chatterbox] ⚠️  No voice sample path available, using default voice")
                        audio = chatterbox.generate(text, exaggeration=exaggeration)
                else:
                    print(f"[Chatterbox] 🔊 Using default voice (no voice cloning)")
                    audio = chatterbox.generate(text, exaggeration=exaggeration)
                    
            elif hasattr(chatterbox, 'synthesize'):
                sig = inspect.signature(chatterbox.synthesize)
                params = list(sig.parameters.keys())
                print(f"[Chatterbox] 🔍 synthesize() parameters: {params}")
                
                if voice_embedding or voice_sample_path_used:
                    # Try audio_prompt_path first (most common in Chatterbox)
                    if voice_sample_path_used and 'audio_prompt_path' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using audio_prompt_path='{voice_sample_path_used}'")
                        audio = chatterbox.synthesize(text, audio_prompt_path=voice_sample_path_used)
                    # Try audio_prompt (alternative name)
                    elif voice_sample_path_used and 'audio_prompt' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using audio_prompt='{voice_sample_path_used}'")
                        audio = chatterbox.synthesize(text, audio_prompt=voice_sample_path_used)
                    # Try voice_embedding (if we have an embedding object)
                    elif voice_embedding and not isinstance(voice_embedding, str) and 'voice_embedding' in params:
                        print(f"[Chatterbox] 🎭 VOICE CLONING: Using voice_embedding object")
                        audio = chatterbox.synthesize(text, voice_embedding=voice_embedding)
                    # Last resort: try passing path
                    elif voice_sample_path_used:
                        print(f"[Chatterbox] ⚠️  Trying voice cloning with path...")
                        try:
                            audio = chatterbox.synthesize(text, audio_prompt_path=voice_sample_path_used)
                            print(f"[Chatterbox] ✅ Voice cloning succeeded")
                        except TypeError:
                            print(f"[Chatterbox] ❌ Voice cloning failed - parameter not recognized")
                            print(f"[Chatterbox]    Available parameters: {params}")
                            print(f"[Chatterbox]    Falling back to default voice")
                            audio = chatterbox.synthesize(text)
                    else:
                        print(f"[Chatterbox] ⚠️  No voice sample path available, using default voice")
                        audio = chatterbox.synthesize(text)
                else:
                    print(f"[Chatterbox] 🔊 Using default voice (no voice cloning)")
                    audio = chatterbox.synthesize(text)
            else:
                return jsonify({'error': 'ChatterboxTTS has no generate or synthesize method'}), 500
            
            gen_elapsed = time.time() - gen_start
            print(f"[Chatterbox] ⏱️  Generation took {gen_elapsed:.2f} seconds")
                
        except Exception as e:
            print(f"[Chatterbox] ❌ Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Synthesis failed: {str(e)}'}), 500
        
        # Convert audio to WAV format
        # Handle different audio formats (numpy array, torch tensor, etc.)
        print(f"[Chatterbox] 🔍 Audio type: {type(audio)}")
        
        # Convert torch tensor to numpy if needed
        if hasattr(audio, 'cpu'):  # torch.Tensor
            print("[Chatterbox] 🔄 Converting torch tensor to numpy array...")
            audio = audio.cpu().numpy()
        
        # Convert to numpy array if not already
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio)
                print(f"[Chatterbox] 🔄 Converted to numpy array, shape: {audio.shape}, dtype: {audio.dtype}")
            except Exception as e:
                print(f"[Chatterbox] ❌ Could not convert audio to numpy array: {e}")
                return jsonify({'error': f'Unexpected audio format: {type(audio)}'}), 500
        
        # Handle multi-dimensional arrays (flatten if needed)
        if len(audio.shape) > 1:
            if audio.shape[0] == 1:
                audio = audio[0]  # Remove batch dimension
            elif audio.shape[1] == 1:
                audio = audio[:, 0]  # Remove channel dimension if mono
            else:
                # Take first channel if stereo/multi-channel
                audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
            print(f"[Chatterbox] 🔄 Flattened audio shape: {audio.shape}")
        
        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Validate audio is not empty
        if len(audio) == 0:
            print("[Chatterbox] ❌ Audio is empty!")
            return jsonify({'error': 'Generated audio is empty'}), 500
        
        # Normalize amplitude to [-1, 1] range (but preserve relative levels)
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            print("[Chatterbox] ❌ Audio is all zeros!")
            return jsonify({'error': 'Generated audio contains no sound'}), 500
        elif max_val > 1.0:
            audio = audio / max_val
            print(f"[Chatterbox] ✅ Audio normalized (was {max_val:.3f}, now 1.0)")
        elif max_val < 0.01:
            print(f"[Chatterbox] ⚠️  Audio is very quiet (max: {max_val:.6f}) - amplifying...")
            # Amplify quiet audio
            audio = audio / max_val * 0.95  # Normalize to 95% to avoid clipping
            print(f"[Chatterbox] ✅ Audio amplified to 95%")
        else:
            print(f"[Chatterbox] ✅ Audio levels OK (max: {max_val:.3f})")
        
        # Get sample rate from model if available, otherwise use default
        sample_rate = 24000  # Chatterbox typically uses 24kHz, not 22kHz
        if hasattr(chatterbox, 'sr'):
            sample_rate = int(chatterbox.sr)
            print(f"[Chatterbox] 📊 Using model sample rate: {sample_rate}")
        elif hasattr(chatterbox, 'sample_rate'):
            sample_rate = int(chatterbox.sample_rate)
            print(f"[Chatterbox] 📊 Using model sample rate: {sample_rate}")
        elif hasattr(chatterbox, 'config') and hasattr(chatterbox.config, 'sample_rate'):
            sample_rate = int(chatterbox.config.sample_rate)
            print(f"[Chatterbox] 📊 Using config sample rate: {sample_rate}")
        else:
            print(f"[Chatterbox] 📊 Using default sample rate: {sample_rate} (if audio sounds wrong, model may use different rate)")
        
        # Validate audio statistics
        audio_duration = len(audio) / sample_rate
        print(f"[Chatterbox] 📊 Audio stats: {len(audio)} samples, {audio_duration:.2f}s duration, {sample_rate}Hz")
        print(f"[Chatterbox] 📊 Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        try:
            # Use soundfile with explicit format
            sf.write(temp_file.name, audio, sample_rate, format='WAV', subtype='PCM_16')
            total_elapsed = time.time() - synthesis_start
            print(f"[Chatterbox] ✅ Audio saved to {temp_file.name} ({len(audio)/sample_rate:.2f}s)")
            print(f"[Chatterbox] ⏱️  Total synthesis time: {total_elapsed:.2f} seconds")
            
            return send_file(
                temp_file.name,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='output.wav'
            )
        except Exception as e:
            print(f"[Chatterbox] ❌ Error saving audio file: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error saving audio: {str(e)}'}), 500
            
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
        
        # Pre-load ChatterboxTTS at startup (blocking - ensures model is ready)
        # This prevents the 446s delay on first request
        print("[Chatterbox] 🔍 Pre-loading ChatterboxTTS at startup...", flush=True)
        print("[Chatterbox] ⏳ This will take 1-5 minutes but ensures fast synthesis requests...", flush=True)
        
        try:
            import time
            preload_start = time.time()
            print("[Chatterbox] 🔄 Starting ChatterboxTTS initialization...", flush=True)
            get_chatterbox_tts()
            preload_elapsed = time.time() - preload_start
            print(f"[Chatterbox] ✅ ChatterboxTTS pre-loaded successfully (took {preload_elapsed:.1f} seconds)", flush=True)
            print("[Chatterbox] ✅ Model is ready - synthesis requests will be fast now", flush=True)
        except Exception as e:
            print(f"[Chatterbox] ⚠️ Pre-loading failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[Chatterbox] 💡 Container will start but synthesis will be slow on first request", flush=True)
            print("[Chatterbox] 💡 Model will load on first synthesis request", flush=True)
        
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

