# aura_gui.py — AuraVision GUI

import os
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGraphicsOpacityEffect, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QPoint
from file_upload_dialog import show_upload_dialog

_app = None
_window = None
_gui_ready = False
_listening_ready = False  # Tracks when system is ready to transcribe
_transcribing = False  # Tracks when user is speaking (transcription active)
_tts_playing = False  # Tracks when TTS is playing (AI speaking)
_tts_frequency = 0.15  # Current TTS frequency for pulsation speed

# Debug: Print initial state
print(f"[AuraGUI] 🎯 Initial state: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

class AuraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AuraVision")
        # Dynamic styling - will be updated based on state
        self.base_style = """
            QMainWindow {
                background-color: black;
                border-radius: 540px;  /* Half of 1080 for perfect circle */
            }
        """
        self.setStyleSheet(self.base_style)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # === Load and Scale aura_eye.png ===
        img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/aura_eye.png"))
        print(f"[AuraGUI] ✅ Loaded image: {img_path}")
        pixmap = QPixmap(img_path)

        screen = self.screen()
        screen_size = screen.availableGeometry().size()
        min_dim = min(screen_size.width(), screen_size.height())
        scaled_pixmap = pixmap.scaled(min_dim, min_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # === Create main widget with image and circular buttons ===
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image label
        self.label = QLabel()
        self.label.setPixmap(scaled_pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Create dedicated border widget for continuous circular border
        self.border_widget = QLabel()
        self.border_widget.setParent(self)
        self.border_widget.setGeometry(0, 0, 1080, 1080)
        self.border_widget.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: 5px solid #ff0000;
                border-radius: 540px;
            }
        """)
        self.border_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.border_widget.raise_()  # Bring to front
        
        # Store border state for animation
        self.border_width = 5
        
        # Create 6 buttons equally spaced around the circular edge (after central widget is set)
        self.create_circular_buttons()

        # === Pulsation Effect ===
        self.opacity_effect = QGraphicsOpacityEffect()
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity = 1.0
        self.pulse_direction = -1
        
        # Animation state variables
        self.border_pulse_phase = 0.0
        self.eye_pulse_phase = 0.0
        self.border_pulse_speed = 0.1  # Red edge pulsation speed
        self.eye_pulse_speed = 0.15    # Aura eye pulsation speed

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_pulse)
        self.timer.start(50)  # Faster updates for smoother animation
        
        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
    
    def create_circular_buttons(self):
        """Create 6 buttons equally spaced around the circular edge"""
        # Button configurations: (text, icon, function, color)
        button_configs = [
            ("↑", "Upload", self._handle_upload, "#007AFF"),      # Upload files
            ("⚙", "Settings", self._handle_settings, "#FF9500"),  # Settings
            ("📊", "Analytics", self._handle_analytics, "#34C759"), # Analytics
            ("🎤", "Voice", self._handle_voice, "#FF3B30"),        # Voice control
            ("📱", "Mobile", self._handle_mobile, "#5856D6"),     # Mobile sync
            ("ℹ", "Info", self._handle_info, "#8E8E93")          # Information
        ]
        
        # Calculate positions for 6 buttons around a circle
        # Radius should be close to the edge but not touching the red border
        radius = 480  # Distance from center to button (closer to edge)
        center_x = 540  # Center of 1080x1080 screen
        center_y = 540
        
        self.buttons = []
        
        for i, (text, tooltip, handler, color) in enumerate(button_configs):
            # Calculate angle for this button (0° to 300° in 60° increments)
            angle = math.radians(i * 60)  # 0, 60, 120, 180, 240, 300 degrees
            
            # Calculate position
            x = center_x + radius * math.cos(angle) - 50  # -50 to center the 100px button
            y = center_y + radius * math.sin(angle) - 50
            
            # Create button
            btn = QPushButton(text)
            btn.setFixedSize(100, 100)  # Bigger buttons
            btn.setToolTip(tooltip)
            btn.move(int(x), int(y))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 32px;
                    font-weight: bold;
                    border-radius: 50px;
                    border: 3px solid #ffffff;
                }}
                QPushButton:hover {{
                    background-color: {color}CC;
                    border: 4px solid #ffffff;
                }}
                QPushButton:pressed {{
                    background-color: {color}99;
                }}
            """)
            
            # Connect handler
            btn.clicked.connect(handler)
            self.buttons.append(btn)
            
            # Add button to the main widget (positioned absolutely)
            btn.setParent(self.centralWidget())
            btn.move(int(x), int(y))
            btn.show()
    
    def _handle_upload(self):
        """Handle upload button click"""
        print("[AuraGUI] 📤 Upload button clicked")
        show_upload_dialog()
    
    def _handle_settings(self):
        """Handle settings button click"""
        print("[AuraGUI] ⚙️ Settings button clicked")
        # TODO: Call settings script
        pass
    
    def _handle_analytics(self):
        """Handle analytics button click"""
        print("[AuraGUI] 📊 Analytics button clicked")
        # TODO: Call analytics script
        pass
    
    def _handle_voice(self):
        """Handle voice button click"""
        print("[AuraGUI] 🎤 Voice button clicked")
        # TODO: Call voice control script
        pass
    
    def _handle_mobile(self):
        """Handle mobile button click"""
        print("[AuraGUI] 📱 Mobile button clicked")
        # TODO: Call mobile sync script
        pass
    
    def _handle_info(self):
        """Handle info button click"""
        print("[AuraGUI] ℹ️ Info button clicked")
        # TODO: Call info script
        pass

    def showEvent(self, event):
        super().showEvent(event)
        global _gui_ready
        _gui_ready = True
        print("[AuraGUI] 🎯 GUI has fully rendered")
        
        # Ensure border widget is properly positioned and visible
        self.border_widget.setGeometry(0, 0, 1080, 1080)
        self.border_widget.raise_()
        self.border_widget.show()

    def animate_pulse(self):
        global _listening_ready, _transcribing, _tts_playing, _tts_frequency
        
        # Debug: Print current state
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 5 seconds (100 * 50ms)
            print(f"[GUI Debug] State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}, tts_freq={_tts_frequency:.3f}")
        
        # State 1: System not ready - gentle aura eye pulse
        if not _listening_ready:
            delta = 0.04 * self.pulse_direction
            self.opacity += delta
            if self.opacity >= 1.0:
                self.opacity = 1.0
                self.pulse_direction = -1
            elif self.opacity <= 0.3:
                self.opacity = 0.3
                self.pulse_direction = 1
            self.opacity_effect.setOpacity(self.opacity)
            self._update_border_style(static=True)
            
        # State 2: System ready, fixed mode
        elif _listening_ready and not _transcribing and not _tts_playing:
            self.opacity_effect.setOpacity(1.0)
            self._update_border_style(static=True)
            
        # State 3: User speaking (transcription) - red edge pulsation
        elif _transcribing:
            self.opacity_effect.setOpacity(1.0)
            self.border_pulse_phase += self.border_pulse_speed
            pulse_intensity = (math.sin(self.border_pulse_phase) + 1) / 2  # 0 to 1
            self.border_width = int(5 + pulse_intensity * 10)  # 5px to 15px
            self._update_border_style(pulsating=True, width=self.border_width)
            
        # State 4: TTS playing - aura eye pulsation synchronized with speech frequency
        elif _tts_playing:
            # Use dynamic frequency from actual TTS audio
            self.eye_pulse_phase += _tts_frequency
            pulse_intensity = (math.sin(self.eye_pulse_phase) + 1) / 2  # 0 to 1
            self.opacity = 0.3 + pulse_intensity * 0.7  # 0.3 to 1.0
            self.opacity_effect.setOpacity(self.opacity)
            self._update_border_style(static=True)
    
    def _update_border_style(self, static=True, pulsating=False, width=5):
        """Update the border style based on current state"""
        if pulsating:
            # Pulsating red border
            self.border_widget.setStyleSheet(f"""
                QLabel {{
                    background-color: transparent;
                    border: {width}px solid #ff0000;
                    border-radius: 540px;
                }}
            """)
        else:
            # Static red border
            self.border_widget.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: 5px solid #ff0000;
                    border-radius: 540px;
                }
            """)
        
        # Ensure border widget is always on top and properly positioned
        self.border_widget.raise_()
        self.border_widget.setGeometry(0, 0, 1080, 1080)
        self.border_widget.show()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_U and event.modifiers() == Qt.ControlModifier:
            # Ctrl+U: Open upload dialog
            self._handle_upload()
        elif event.key() == Qt.Key_Escape:
            # Escape: Close GUI
            self.close()
        else:
            super().keyPressEvent(event)

# === GUI Control ===
def launch_gui():
    global _app, _window
    _app = QApplication(sys.argv)
    _window = AuraGUI()
    _window.showFullScreen()
    _app.processEvents()  # ✅ Ensure GUI renders before returning

def run_gui_loop():
    global _app
    try:
        print("[AuraGUI] 🌀 Entering event loop...")
        _app.exec_()
    except KeyboardInterrupt:
        print("[AuraGUI] ⛔ GUI interrupted.")
        _app.quit()

def close_gui():
    global _app, _window
    if _window:
        _window.close()
    if _app:
        _app.quit()

def gui_is_ready():
    return _gui_ready

def set_listening_ready():
    global _listening_ready
    _listening_ready = True
    print("[AuraGUI] 🎯 Switched to fixed mode - listener ready")
    print(f"[AuraGUI] 🎯 State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

def set_transcribing(active):
    """Set transcription state - red edge pulsation when user is speaking"""
    global _transcribing
    _transcribing = active
    if active:
        print("[AuraGUI] 🔴 Transcription active - red edge pulsating")
    else:
        print("[AuraGUI] ⚫ Transcription ended - returning to fixed mode")
    print(f"[AuraGUI] 🔴 State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

def set_tts_playing(active):
    """Set TTS state - aura eye pulsation when AI is speaking"""
    global _tts_playing
    _tts_playing = active
    if active:
        print("[AuraGUI] 👁️ TTS playing - aura eye pulsating")
    else:
        print("[AuraGUI] ⚫ TTS ended - returning to fixed mode")
    print(f"[AuraGUI] 👁️ State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

def set_tts_frequency(frequency_speed):
    """Set TTS frequency for synchronized aura eye pulsation"""
    global _tts_frequency
    _tts_frequency = frequency_speed
    print(f"[AuraGUI] 🎵 TTS frequency updated: {frequency_speed:.3f}")
