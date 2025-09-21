# core/state.py — Global state management

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

def is_playing():
    return _playing


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
