# GUI (Graphical User Interface) Component

## Overview

The GUI component provides a circular, touch-friendly interface for Aura. Built with PyQt5, it displays visual feedback for system states, provides interactive buttons, and manages dialog windows.

## Architecture

### Main GUI Module
- **Location**: `aura-control/gui/aura_gui.py`
- **Framework**: PyQt5
- **Window Size**: 1080x1080 (circular screen)
- **Window Flags**: Frameless, always on top

### Dialog System
- **Base Dialog**: `aura-control/gui/base_dialog.py`
- **Dialog Templates**: `aura-control/gui/DIALOG_TEMPLATE.md`
- **Custom Dialogs**: Settings, Wallet, File Upload, Welcome Setup

## Core Components

### 1. Main Window (AuraGUI)

**Class**: `AuraGUI(QMainWindow)`

**Layout**:
- **Central Widget**: Main widget with circular image
- **Image**: `aura_eye.png` (scaled to 1080x1080)
- **Border Overlay**: Circular borders on top
- **Buttons**: 6 circular buttons around edge

**Window Properties**:
```python
window_size = 1080  # Square window
setFixedSize(window_size, window_size)
setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
move(0, 0)  # Top-left corner
```

**Styling**:
- Black background
- Border radius: 540px (perfect circle)
- Frameless window (no title bar)
- Always on top

### 2. Circular Buttons

**Number**: 6 buttons equally spaced around edge

**Positioning**:
- Equally spaced 60 degrees apart
- Radius: 450px from center
- Size: 80x80 pixels
- Touch-friendly targets

**Button Types**:
1. **Settings**: Opens settings dialog
2. **Wallet**: Opens wallet dialog
3. **Upload**: Opens file upload dialog
4. **Microphone**: Toggles microphone mute
5. **Volume**: Volume control (future)
6. **Power**: Power/shutdown (future)

**Visual Feedback**:
- Highlight on hover
- Press animation
- Icon-based design

### 3. Border Animation System

**Border Types**:
- **White Reference Circle**: Always visible (30% opacity, 8px width)
- **Red Border**: Visible during transcription/wake word (variable opacity)

**States**:
1. **Idle**: White border only, slow pulsing
2. **Wake Word Detected**: Solid red border (high opacity)
3. **Transcribing**: Pulsating red border (speed based on audio frequency)
4. **TTS Playing**: Border animations based on TTS frequency

**Animation**:
- **Pulse Phase**: Sine wave for pulsing effect
- **Speed**: Variable (0.1-0.5 based on audio frequency)
- **Opacity**: Variable (0.7 during active states)

**Implementation**:
```python
def _animate_aura_eye_idle(self):
    # Slow breathing animation
    self.aura_breathing_phase += 0.01
    breathing_intensity = (math.sin(self.aura_breathing_phase) + 1) / 2
    # ... opacity calculation ...
    self.opacity_effect.setOpacity(self.opacity)
```

### 4. State Management

**Global State Variables**:
```python
_gui_ready = False           # GUI initialized
_listening_ready = False     # System ready for transcription
_transcribing = False        # User currently speaking
_wake_word_detected = False  # Wake word detected
_tts_playing = False         # TTS currently playing
_setup_complete = False      # Initial setup complete
_tts_frequency = 0.15        # Current TTS frequency
_microphone_muted = False    # Mic manually muted
```

**State Update Functions**:
- `set_listening_ready(True)`: System ready
- `set_transcribing(True/False)`: User speaking
- `set_wake_word_detected(True/False)`: Wake word state
- `set_tts_playing(True/False)`: TTS playing
- `set_setup_complete()`: Setup complete
- `set_tts_frequency(freq)`: Update TTS pulsation speed

### 5. Opacity Animation

**Purpose**: Visual feedback for system states

**States**:
1. **Idle**: Slow breathing animation (opacity 0.1-0.8)
2. **Wake Word Detected**: Solid opacity (0.7)
3. **Transcribing**: Pulsating opacity (speed based on audio)
4. **TTS Playing**: Pulsating opacity (speed based on TTS frequency)

**Implementation**:
- Uses `QGraphicsOpacityEffect`
- Applied to main image label
- Opacity updated via timer (60 FPS)

**Animation Speed**:
- Based on audio frequency analysis
- Dominant frequency → pulsation speed
- Range: 0.1-0.5 speed
- Default: 0.15

### 6. Border Overlay Widget

**Class**: `BorderOverlayWidget(QWidget)`

**Purpose**: Draws circular borders on top of all widgets

**Features**:
- Transparent background
- Mouse events ignored (pass-through)
- Painted after all other widgets

**Drawing**:
1. **White Reference Circle**:
   - Always visible
   - 30% opacity (alpha: 77/255)
   - 8px width
   - Radius: 535px (edge of screen)

2. **Red Border**:
   - Conditionally visible
   - Variable opacity (0.7 during active states)
   - 10px width
   - Same radius as white circle

**Implementation**:
```python
def paintEvent(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # Draw white reference circle
    # Draw red border if active
```

## Dialog System

### 1. Base Dialog

**Class**: `BaseAuraDialog(QDialog)`

**Features**:
- Consistent styling across dialogs
- Circular border design
- Custom keyboard support
- Touch-friendly layout

**Common Elements**:
- Title bar
- Content area
- Action buttons
- Keyboard integration

### 2. Settings Dialog

**Location**: `aura-control/gui/settings_dialog.py`

**Features**:
- WiFi configuration
- LLM mode selection (Medical/Generic)
- RAG mode selection (CPU/GPU/OFF)
- Wake word enable/disable
- Model selection
- Volume control

**UI Elements**:
- Network status
- Mode toggles
- Model dropdown
- Volume slider

### 3. Wallet Dialog

**Location**: `aura-control/gui/wallet_dialog.py`

**Features**:
- Ethereum wallet connection
- Balance display
- Token information
- Wallet address entry (custom keyboard)
- NFC wallet authentication

**Integration**:
- `aura-control/wallet/` modules
- Ethereum Web3 connection
- Token balance fetching

### 4. File Upload Dialog

**Location**: `aura-control/gui/file_upload_dialog.py`

**Features**:
- File selection
- Upload progress
- Web upload server integration
- Auto-ingestion trigger

**Flow**:
1. User selects file
2. Uploads to web server (port 5001)
3. Server saves to `data/input/`
4. Auto-ingestion processes file
5. RAG index updated

### 5. Welcome Setup Dialog

**Location**: `aura-control/gui/welcome_setup_dialog.py`

**Features**:
- Initial WiFi setup
- System configuration
- Welcome message
- First-time user guide

**Flow**:
1. Shown on first launch
2. WiFi connection setup
3. Configuration wizard
4. Welcome audio playback
5. Proceeds to main interface

### 6. Custom Keyboard

**Location**: `aura-control/gui/custom_keyboard.py`

**Features**:
- Circular key layout
- Touch-friendly
- QWERTY layout
- Special characters
- Voice input option

**Usage**:
- Wallet address entry
- Text input dialogs
- Settings configuration

## Visual Feedback System

### 1. Transcription Feedback

**State**: `_transcribing = True`

**Visual**:
- Red pulsating border
- Speed based on audio frequency
- Opacity: 0.7
- Updates in real-time

**Implementation**:
```python
if _transcribing:
    self.show_red_border = True
    self.red_border_opacity = 0.7
    # Update border animation
```

### 2. Wake Word Feedback

**State**: `_wake_word_detected = True`

**Visual**:
- Solid red border (no pulsation)
- High opacity (0.7)
- Indicates waiting for speech

**Transition**:
- Wake word → Solid red
- Speech detected → Pulsating red
- TTS playing → Border animations

### 3. TTS Playing Feedback

**State**: `_tts_playing = True`

**Visual**:
- Border animations
- Speed based on TTS audio frequency
- Dominant frequency → pulsation speed

**Audio Analysis**:
- FFT on TTS audio chunks
- Finds dominant frequency
- Maps to pulsation speed (0.1-0.5)

### 4. Idle State Feedback

**State**: All inactive

**Visual**:
- White border only
- Slow breathing animation
- Opacity: 0.1-0.8
- Gentle pulsing

## Animation System

### 1. Aura Eye Animation

**Purpose**: Idle breathing animation

**Phases**:
- **Breathing Phase**: Slow sine wave (0.01 increment)
- **Heartbeat Phase**: Quick pulse (0.1 increment)
- **Glow Phase**: Subtle glow (0.02 increment)

**Combined Effect**:
```python
combined_intensity = (
    breathing_intensity * 0.7 + 
    glow_intensity * 0.3
)
opacity = 0.1 + combined_intensity * 0.7
```

### 2. Border Pulse Animation

**Speed Calculation**:
```python
border_pulse_speed = 0.1  # Base speed
eye_pulse_speed = 0.15    # Idle speed
# Speed updates based on audio/TTS frequency
```

**Update Frequency**: 60 FPS (via QTimer)

### 3. Button Animations

**Hover Effect**:
- Slight scale increase
- Color highlight
- Smooth transitions

**Press Effect**:
- Scale down slightly
- Color change
- Immediate feedback

## Integration Points

### 1. State Updates from Listener

**Functions**:
- `set_listening_ready(True)`: Called after welcome prompt
- `set_transcribing(True)`: Called when VAD detects speech
- `set_transcribing(False)`: Called when speech ends

**Source**: `listener.py` → GUI state functions

### 2. State Updates from Speaker

**Functions**:
- `set_tts_playing(True)`: Called when TTS starts
- `set_tts_playing(False)`: Called when TTS ends
- `set_tts_frequency(freq)`: Updates pulsation speed

**Source**: `speaker.py` → GUI state functions

### 3. Wake Word Integration

**Functions**:
- `set_wake_word_detected(True)`: Wake word detected
- `set_wake_word_detected(False)`: Wake word cleared
- `set_wake_word_activated(False)`: Wake word deactivated

**Source**: `listener.py` → GUI state functions

## Window Management

### 1. Display Setup

**Functions** (in `main.py`):
- `setup_display()`: Configures display settings
- `check_x11_display()`: Validates X11 connection
- `focus_gui_window()`: Brings window to front

**Settings**:
- Screen wake: `xset dpms force on`
- Screensaver disable: `xscreensaver-command -deactivate`
- Cursor hide: `unclutter` or `xdotool mousemove 0 0`
- Keyboard disable: Ubuntu keyboard monitoring

### 2. Window Focus

**Bring to Front**:
1. Try `wmctrl -a Aura`
2. Fallback: `xdotool search --class aura`
3. Activate window

**Purpose**: Ensures GUI visible on startup

### 3. Window Flags

**Flags**:
- `Qt.FramelessWindowHint`: No title bar
- `Qt.WindowStaysOnTopHint`: Always visible
- `Qt.WA_TranslucentBackground`: Transparent overlay

## Code Locations

- **Main GUI**: `aura-control/gui/aura_gui.py`
- **Base Dialog**: `aura-control/gui/base_dialog.py`
- **Settings Dialog**: `aura-control/gui/settings_dialog.py`
- **Wallet Dialog**: `aura-control/gui/wallet_dialog.py`
- **File Upload Dialog**: `aura-control/gui/file_upload_dialog.py`
- **Welcome Setup Dialog**: `aura-control/gui/welcome_setup_dialog.py`
- **Custom Keyboard**: `aura-control/gui/custom_keyboard.py`
- **Circular Border**: `aura-control/gui/circular_border.py`

## Dependencies

- `PyQt5`: GUI framework
- `numpy`: Audio frequency analysis
- `scipy`: FFT for frequency analysis
- `xdotool`: Window management (optional)
- `wmctrl`: Window control (optional)
- `unclutter`: Cursor hiding (optional)

## Configuration

### Environment Variables

**Display**:
- `DISPLAY`: X11 display (e.g., `:0`)
- Auto-detected from `/tmp/.X11-unix/`

**Window Size**:
- Hardcoded: 1080x1080
- Matches circular screen size

### State Configuration

**Settings File**: `data/app_settings.json`

**GUI Settings**:
- Wake word enabled
- Volume level
- Display preferences
- Dialog preferences

