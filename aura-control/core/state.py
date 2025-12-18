# core/state.py — Global state management

import os
import sys

# Set up proper imports for organized structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# === Shutdown state ===
shutdown_requested = False

def request_shutdown():
    global shutdown_requested
    shutdown_requested = True

def clear_shutdown():
    global shutdown_requested
    shutdown_requested = False

def should_shutdown():
    return shutdown_requested


# === Playback state ===
_playing = False

def set_playing(value: bool):
    global _playing
    _playing = value

    # Update GUI state for TTS
    try:
        from gui.aura_gui import set_tts_playing
        set_tts_playing(value)
    except ImportError:
        pass  # GUI not available
    except Exception as e:
        print(f"[State] ⚠️ GUI update failed: {e}")

def is_playing():
    return _playing

# === App Settings (LLM mode/model) ===
_settings_file = os.path.expanduser("~/LedgerAI/data/app_settings.json")
_llm_mode = "generic"  # 'medical' or 'generic' - default to generic
_llm_model = ""        # filename or path inside container; empty means default
_wake_word_enabled = True  # Wake word detection enabled/disabled - default enabled
_wake_word_sensitivity = 0.9  # Wake word detection sensitivity (0.0-1.0) - higher = more sensitive (lower threshold)
_wake_word_model_path = None  # Optional path to custom model file
_wake_word_engine = "openwakeword"  # Wake word engine: "openwakeword"
_tts_engine = "elevenlabs"  # TTS engine: "chatterbox" or "elevenlabs"
_chatterbox_voice_cloning_enabled = True  # Enable voice cloning for ChatterboxTTS (adds ~50-100ms latency)
_whisper_model_set_via_gui = False  # Track if whisper_model was explicitly set via GUI

def _get_whisper_model_default_from_container_rest():
    """Read the default Whisper model from container_rest.py (single source of truth)"""
    try:
        # Get workspace root (2 levels up from core/)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        container_rest_path = os.path.join(workspace_root, 'whisper-container', 'container_rest.py')
        
        if not os.path.exists(container_rest_path):
            # Fallback if file doesn't exist
            return "distil-small.en"
        
        # Read container_rest.py and extract MODEL_NAME default
        import re
        with open(container_rest_path, 'r') as f:
            content = f.read()
            # Match: MODEL_NAME = os.environ.get("WHISPER_MODEL", "default-value")
            match = re.search(r'MODEL_NAME\s*=\s*os\.environ\.get\([^,]+,\s*"([^"]+)"\)', content)
            if match:
                return match.group(1)
        
        # Fallback if pattern not found
        return "distil-small.en"
    except Exception as e:
        print(f"[State] ⚠️ Could not read default from container_rest.py: {e}")
        return "distil-small.en"  # Safe fallback

# Whisper model: Initially defaults from container_rest.py, then uses GUI selection from app_settings.json
_whisper_model_default = _get_whisper_model_default_from_container_rest()
_whisper_model = _whisper_model_default  # Will be overridden by app_settings.json if present, otherwise uses container_rest.py default

def _save_settings_to_disk():
    """Save current settings to disk"""
    try:
        import json, os
        os.makedirs(os.path.dirname(_settings_file), exist_ok=True)
        settings_data = {
            "llm_mode": _llm_mode,
            "llm_model": _llm_model,
            "wake_word_enabled": _wake_word_enabled,
            "wake_word_sensitivity": _wake_word_sensitivity,
            "wake_word_engine": _wake_word_engine,
            "tts_engine": _tts_engine,
            "chatterbox_voice_cloning_enabled": _chatterbox_voice_cloning_enabled
        }
        if _wake_word_model_path:
            settings_data["wake_word_model_path"] = _wake_word_model_path
        # Only save whisper_model if it was explicitly set via GUI
        # This allows GUI selection to persist and override container default
        # If not set via GUI, we don't save it, so it always uses container_rest.py default
        if _whisper_model_set_via_gui:
            settings_data["whisper_model"] = _whisper_model
        with open(_settings_file, "w") as f:
            json.dump(settings_data, f, indent=2)
    except Exception as e:
        print(f"[State] ⚠️ Failed to save settings: {e}")

def _load_settings_from_disk():
    """Load settings from disk, creating default file if it doesn't exist"""
    global _llm_mode, _llm_model, _wake_word_enabled, _wake_word_sensitivity, _wake_word_model_path, _wake_word_engine, _tts_engine, _chatterbox_voice_cloning_enabled, _whisper_model, _whisper_model_default, _whisper_model_set_via_gui
    try:
        import json
        with open(_settings_file, "r") as f:
            data = json.load(f)
            _llm_mode = data.get("llm_mode", _llm_mode)
            _llm_model = data.get("llm_model", _llm_model)
            _wake_word_enabled = data.get("wake_word_enabled", _wake_word_enabled)
            _wake_word_sensitivity = float(data.get("wake_word_sensitivity", _wake_word_sensitivity))
            _wake_word_model_path = data.get("wake_word_model_path", _wake_word_model_path)
            _wake_word_engine = data.get("wake_word_engine", _wake_word_engine)
            _tts_engine = data.get("tts_engine", _tts_engine)
            _chatterbox_voice_cloning_enabled = data.get("chatterbox_voice_cloning_enabled", _chatterbox_voice_cloning_enabled)
            # whisper_model: Use GUI selection from app_settings.json if present, otherwise default from container_rest.py
            _whisper_model_default = _get_whisper_model_default_from_container_rest()
            if "whisper_model" in data:
                # GUI selection exists in JSON - use it
                _whisper_model = data["whisper_model"]
                _whisper_model_set_via_gui = True
            else:
                # No GUI selection - use container_rest.py default
            _whisper_model = _whisper_model_default
                _whisper_model_set_via_gui = False
    except FileNotFoundError:
        # File doesn't exist - use defaults and create it
        _whisper_model_default = _get_whisper_model_default_from_container_rest()
        _whisper_model = _whisper_model_default
        _whisper_model_set_via_gui = False
        _save_settings_to_disk()
        print(f"[State] 📝 Created default settings file: {_settings_file}")
    except Exception as e:
        print(f"[State] ⚠️ Failed to load settings: {e}")
        # On error, use container_rest.py default
        _whisper_model_default = _get_whisper_model_default_from_container_rest()
        _whisper_model = _whisper_model_default
        _whisper_model_set_via_gui = False

# Load settings on import
_load_settings_from_disk()

def get_llm_mode() -> str:
    """Return 'medical' or 'generic'."""
    return _llm_mode

def set_llm_mode(mode: str):
    """Set mode: 'medical' or 'generic'."""
    global _llm_mode
    if mode not in ("medical", "generic"):
        return
    _llm_mode = mode
    _save_settings_to_disk()

def get_llm_model() -> str:
    """Return selected model name/path (may be empty for default)."""
    return _llm_model

def set_llm_model(model: str):
    """Set selected model name/path."""
    global _llm_model
    _llm_model = model or ""
    _save_settings_to_disk()

# === Wake Word Settings ===
def get_wake_word_enabled() -> bool:
    """Return whether wake word detection is enabled."""
    return _wake_word_enabled

def set_wake_word_enabled(enabled: bool):
    """Enable or disable wake word detection."""
    global _wake_word_enabled
    _wake_word_enabled = bool(enabled)
    _save_settings_to_disk()

def get_wake_word_sensitivity() -> float:
    """Return wake word detection sensitivity (0.0-1.0)."""
    return _wake_word_sensitivity

def set_wake_word_sensitivity(sensitivity: float):
    """Set wake word detection sensitivity (0.0-1.0)."""
    global _wake_word_sensitivity
    _wake_word_sensitivity = max(0.0, min(1.0, float(sensitivity)))
    _save_settings_to_disk()

def get_wake_word_model_path() -> str:
    """Return path to custom wake word model file (or None)."""
    return _wake_word_model_path

def set_wake_word_model_path(path: str):
    """Set path to custom wake word model file (or None to use built-in)."""
    global _wake_word_model_path
    _wake_word_model_path = path if path else None
    _save_settings_to_disk()

def get_wake_word_engine() -> str:
    """Return wake word engine: 'openwakeword'."""
    return _wake_word_engine

def set_wake_word_engine(engine: str):
    """Set wake word engine: 'openwakeword'."""
    global _wake_word_engine
    if engine == "openwakeword":
        _wake_word_engine = engine
        _save_settings_to_disk()

# === TTS Engine Settings ===
def get_tts_engine() -> str:
    """Return TTS engine: 'chatterbox' or 'elevenlabs'."""
    return _tts_engine

def set_tts_engine(engine: str):
    """Set TTS engine: 'chatterbox' or 'elevenlabs'."""
    global _tts_engine
    if engine in ("chatterbox", "elevenlabs"):
        _tts_engine = engine
        _save_settings_to_disk()

# === ChatterboxTTS Voice Cloning Settings ===
def get_chatterbox_voice_cloning_enabled() -> bool:
    """Return whether voice cloning is enabled for ChatterboxTTS."""
    return _chatterbox_voice_cloning_enabled

def set_chatterbox_voice_cloning_enabled(enabled: bool):
    """Enable or disable voice cloning for ChatterboxTTS (adds ~50-100ms latency)."""
    global _chatterbox_voice_cloning_enabled
    _chatterbox_voice_cloning_enabled = bool(enabled)
    _save_settings_to_disk()

# === Whisper Model Settings ===
def get_whisper_model() -> str:
    """Return Whisper model name.
    
    Priority:
    1. GUI selection from app_settings.json (if set)
    2. Default from container_rest.py (if no GUI selection)
    """
    global _whisper_model, _whisper_model_default
    # If _whisper_model is None or not set, use container_rest.py default
    if _whisper_model is None:
    _whisper_model_default = _get_whisper_model_default_from_container_rest()
        _whisper_model = _whisper_model_default
    return _whisper_model

def set_whisper_model(model: str):
    """Set Whisper model name (saves to app_settings.json).
    
    After setting via GUI, this value will be used as the default.
    If not set, defaults to container_rest.py MODEL_NAME.
    """
    global _whisper_model, _whisper_model_set_via_gui, _whisper_model_default
    if model and model.strip():
        _whisper_model = model.strip()
        _whisper_model_set_via_gui = True  # Mark as set via GUI
        _save_settings_to_disk()
    else:
        # If empty/None, reset to container_rest.py default and remove from JSON
        _whisper_model_default = _get_whisper_model_default_from_container_rest()
        _whisper_model = _whisper_model_default
        _whisper_model_set_via_gui = False  # Mark as not set via GUI (will use container default)
        _save_settings_to_disk()


# === Listener restart trigger ===
restart_listener_flag = False

def request_listener_restart():
    global restart_listener_flag
    restart_listener_flag = True

def clear_listener_restart():
    global restart_listener_flag
    restart_listener_flag = False

def should_restart_listener():
    return restart_listener_flag

# === Last response question tracking ===
_last_response_ended_with_question = False

def set_last_response_ended_with_question(ended_with_question: bool):
    """Track if the last LLM response ended with a question mark."""
    global _last_response_ended_with_question
    _last_response_ended_with_question = ended_with_question

def get_last_response_ended_with_question() -> bool:
    """Return True if the last LLM response ended with a question mark."""
    return _last_response_ended_with_question

def clear_last_response_question_flag():
    """Clear the question flag (call after user responds)."""
    global _last_response_ended_with_question
    _last_response_ended_with_question = False
