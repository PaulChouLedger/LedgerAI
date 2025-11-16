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
_llm_mode = "medical"  # 'medical' or 'generic'
_llm_model = ""        # filename or path inside container; empty means default

def _load_settings_from_disk():
    global _llm_mode, _llm_model
    try:
        import json
        with open(_settings_file, "r") as f:
            data = json.load(f)
            _llm_mode = data.get("llm_mode", _llm_mode)
            _llm_model = data.get("llm_model", _llm_model)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[State] ⚠️ Failed to load settings: {e}")

def _save_settings_to_disk():
    try:
        import json, os
        os.makedirs(os.path.dirname(_settings_file), exist_ok=True)
        with open(_settings_file, "w") as f:
            json.dump({"llm_mode": _llm_mode, "llm_model": _llm_model}, f, indent=2)
    except Exception as e:
        print(f"[State] ⚠️ Failed to save settings: {e}")

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

# Initialize settings at import
_load_settings_from_disk()


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
