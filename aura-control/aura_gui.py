# aura_gui.py — AuraVision GUI

import os
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QTransform
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve
from file_upload_dialog import show_upload_dialog

_app = None
_window = None
_gui_ready = False
_listening_ready = False  # Tracks when system is ready to transcribe
_transcribing = False  # Tracks when user is speaking (transcription active)
_tts_playing = False  # Tracks when TTS is playing (AI speaking)
_setup_complete = False  # Tracks when initial setup is complete
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
        
        # Create dedicated border widget for continuous circular border (hidden by default)
        self.border_widget = QLabel()
        self.border_widget.setParent(self)
        self.border_widget.setGeometry(0, 0, 1080, 1080)
        self.border_widget.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: 5px solid rgba(100, 0, 0, 0.6);
                border-radius: 540px;
            }
        """)
        self.border_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.border_widget.raise_()  # Bring to front
        self.border_widget.hide()  # Hide by default
        
        # Store border state for animation
        self.border_width = 5
        self.border_pulse_speed = 0.1  # Will be randomized during transcription
        
        # Create 6 buttons equally spaced around the circular edge (after central widget is set)
        self.create_circular_buttons()

        # === Pulsation Effect ===
        self.opacity = 1.0
        self.pulse_direction = -1
        
        # Add subtle glow effect for more visual appeal (this will be our main effect)
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(20)
        self.glow_effect.setColor(QColor(0, 100, 255, 100))  # Subtle blue glow
        self.glow_effect.setOffset(0, 0)
        self.label.setGraphicsEffect(self.glow_effect)
        
        # Animation state variables
        self.border_pulse_phase = 0.0
        self.eye_pulse_phase = 0.0
        self.border_pulse_speed = 0.1  # Red edge pulsation speed
        self.eye_pulse_speed = 0.15    # Aura eye pulsation speed
        
        # Enhanced aura eye animation variables
        self.aura_breathing_phase = 0.0      # Slow breathing rhythm
        self.aura_heartbeat_phase = 0.0      # Quick heartbeat rhythm
        self.aura_glow_phase = 0.0          # Subtle glow effect
        self.aura_organic_timer = 0.0       # Organic timing variation
        self.aura_intensity_base = 0.3        # Base intensity
        self.aura_intensity_variation = 0.7   # Intensity variation range

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
        # Add 5mm spacing from edge (5mm ≈ 19 pixels at 1080p)
        # Edge radius is 540px, button radius is 50px, so: 540 - 19 - 50 = 471px
        radius = 471  # Distance from center to button (5mm spacing from edge)
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
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {color}FF, stop:0.3 {color}E6, stop:0.7 {color}CC, stop:1 {color}B3);
                    color: white;
                    font-size: 32px;
                    font-weight: bold;
                    border-radius: 50px;
                    border: none;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {color}FF, stop:0.2 {color}F0, stop:0.8 {color}D9, stop:1 {color}C2);
                    transform: scale(1.05);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {color}CC, stop:0.3 {color}B3, stop:0.7 {color}99, stop:1 {color}80);
                    transform: scale(0.95);
                }}
            """)
            
            # Add 3D shadow effect for Apple-style depth
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(15)
            shadow_effect.setColor(QColor(0, 0, 0, 80))  # Subtle black shadow
            shadow_effect.setOffset(0, 4)  # Slight downward offset for depth
            btn.setGraphicsEffect(shadow_effect)
            
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
        
        # Add smooth fade-out animation for main GUI
        self.fade_out_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out_animation.setDuration(150)  # Quick fade-out
        self.fade_out_animation.setStartValue(1.0)
        self.fade_out_animation.setEndValue(0.7)  # Slightly dimmed
        self.fade_out_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_out_animation.start()
        
        show_upload_dialog()
        
        # Add smooth fade-in animation when dialog closes
        QTimer.singleShot(200, self._restore_gui_opacity)
    
    def _restore_gui_opacity(self):
        """Restore main GUI opacity with smooth animation"""
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(200)  # Smooth fade-in
        self.fade_in_animation.setStartValue(0.7)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_in_animation.start()
        
        # Ensure GUI is properly restored after dialog closes
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        # Reset aura eye to original size
        self._reset_aura_eye()
    
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
        global _listening_ready, _transcribing, _tts_playing, _tts_frequency, _setup_complete
        
        # Debug output removed for cleaner console
        
        # State 1: Initial setup - gentle, meditative aura eye
        if not _setup_complete:
            self._animate_aura_eye_setup()
            self.border_widget.hide()  # Hide border during setup
            
        # State 2: Setup complete but not ready - gentle, meditative aura eye
        elif _setup_complete and not _listening_ready:
            self._animate_aura_eye_idle()
            self.border_widget.hide()  # Hide border in initial state
            
        # State 3: System ready, fixed mode - subtle aura eye
        elif _listening_ready and not _transcribing and not _tts_playing:
            self._animate_aura_eye_idle()  # Gentle breathing animation
            self.border_widget.hide()  # Hide border completely
            
        # State 4: User speaking (transcription) - red edge pulsation matching user's speech frequency
        elif _transcribing:
            self.label.setStyleSheet("opacity: 1.0;")
            
            # Get actual transcription frequency from audio analysis
            try:
                from listener import get_transcription_frequency
                speech_freq = get_transcription_frequency()
                # Ensure frequency is in valid range
                speech_freq = max(0.1, min(speech_freq, 1.0))
            except ImportError:
                speech_freq = 0.7  # Higher default for more visible pulsation
            except Exception as e:
                print(f"[GUI] ⚠️ Frequency analysis failed: {e}")
                speech_freq = 0.7  # Fallback to visible pulsation
            
            # Use speech frequency for pulse speed with enhanced responsiveness
            self.border_pulse_speed = speech_freq * 4.0  # 4x scale for more visible effect
            
            # Add some variation based on speech frequency
            frequency_variation = speech_freq * 1.0  # More variation
            self.border_pulse_phase += self.border_pulse_speed + frequency_variation
            
            # Create pulsation that matches speech characteristics
            pulse_intensity = (math.sin(self.border_pulse_phase) + 1) / 2
            
            # Consistent width calculation for all transcriptions
            base_width = 5  # Fixed base width
            variation_width = 3  # Fixed variation
            self.border_width = int(base_width + pulse_intensity * variation_width)
            
            # Ensure consistent minimum and maximum width
            self.border_width = max(self.border_width, 5)
            self.border_width = min(self.border_width, 8)
            
            print(f"[GUI] 🔴 Border: freq={speech_freq:.3f}, width={self.border_width}px, intensity={pulse_intensity:.3f}")
            self._update_border_style(pulsating=True, width=self.border_width)
            
        # State 5: TTS playing - sophisticated aura eye pulsation
        elif _tts_playing:
            self._animate_aura_eye_tts(_tts_frequency)
            self.border_widget.hide()  # Hide border during TTS
    
    def _update_border_style(self, static=True, pulsating=False, width=5):
        """Update the border style based on current state"""
        if pulsating:
            # Show and animate pulsating red border
            self.border_widget.setStyleSheet(f"""
                QLabel {{
                    background-color: transparent;
                    border: {width}px solid rgba(100, 0, 0, 0.6);
                    border-radius: 540px;
                }}
            """)
            self.border_widget.show()
        else:
            # Hide border completely
            self.border_widget.hide()
            return
        
        # Ensure border widget is always on top and properly positioned
        self.border_widget.raise_()
        self.border_widget.setGeometry(0, 0, 1080, 1080)
    
    def _animate_aura_eye_tts(self, tts_frequency):
        """Sophisticated aura eye animation during TTS with organic, natural movement"""
        # Clamp and scale TTS frequency
        tts_freq = max(0.1, min(tts_frequency, 2.0))
        
        # Update organic timing (creates natural variation)
        self.aura_organic_timer += 0.02
        organic_variation = math.sin(self.aura_organic_timer * 0.3) * 0.1
        
        # Multiple animation layers for natural movement
        
        # Layer 1: Breathing rhythm (slow, deep)
        self.aura_breathing_phase += 0.05 + organic_variation
        breathing_intensity = (math.sin(self.aura_breathing_phase) + 1) / 2
        
        # Layer 2: Heartbeat rhythm (quick, responsive to TTS)
        heartbeat_speed = tts_freq * 2.0 + organic_variation
        self.aura_heartbeat_phase += heartbeat_speed
        heartbeat_intensity = (math.sin(self.aura_heartbeat_phase) + 1) / 2
        
        # Layer 3: Glow effect (subtle, continuous)
        self.aura_glow_phase += 0.08 + organic_variation * 0.5
        glow_intensity = (math.sin(self.aura_glow_phase) + 1) / 2
        
        # Layer 4: Micro-variations (very subtle, organic)
        micro_phase = self.aura_organic_timer * 0.7
        micro_intensity = (math.sin(micro_phase) + math.sin(micro_phase * 1.7)) / 4
        
        # Combine all layers with weighted influence
        combined_intensity = (
            breathing_intensity * 0.4 +      # 40% breathing (slow, natural)
            heartbeat_intensity * 0.35 +     # 35% heartbeat (responsive to TTS)
            glow_intensity * 0.15 +          # 15% glow (subtle)
            micro_intensity * 0.1            # 10% micro-variations (organic)
        )
        
        # Apply natural easing and smoothing
        # Use sigmoid-like function for more natural transitions
        smoothed_intensity = 1 / (1 + math.exp(-6 * (combined_intensity - 0.5)))
        
        # Calculate final opacity with more dramatic range for TTS
        base_opacity = 0.2  # Even lower base for more dramatic effect
        variation_range = 0.7  # Much larger variation range
        self.opacity = base_opacity + smoothed_intensity * variation_range  # 0.2 to 0.9 range
        
        # Set opacity through widget style with more dramatic changes
        self.label.setStyleSheet(f"opacity: {self.opacity};")
        
        # Also try setting the widget opacity directly
        self.label.setWindowOpacity(self.opacity)
        
        # Try a more dramatic approach - scale the pixmap slightly
        try:
            if hasattr(self, '_original_pixmap') and self._original_pixmap:
                # Create a scaled version based on intensity
                scale_factor = 0.8 + (smoothed_intensity * 0.4)  # 0.8 to 1.2 scale
                scaled_pixmap = self._original_pixmap.scaled(
                    int(self._original_pixmap.width() * scale_factor),
                    int(self._original_pixmap.height() * scale_factor),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.label.setPixmap(scaled_pixmap)
        except Exception as e:
            pass  # Fallback to opacity only
        
        # Debug: Print TTS animation values
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 20 == 0:  # Print every second during TTS
            print(f"[GUI] 👁️ Aura Eye TTS: opacity={self.opacity:.3f}, breathing={breathing_intensity:.3f}, heartbeat={heartbeat_intensity:.3f}")
        
        # Temporarily disable glow effect to test aura eye pulsation
        # Dynamic glow effect that responds to TTS
        # glow_intensity = (heartbeat_intensity + breathing_intensity) / 2
        # glow_alpha = int(50 + glow_intensity * 100)  # 50-150 alpha
        # glow_color = QColor(0, 100, 255, glow_alpha)
        # self.glow_effect.setColor(glow_color)
        
        # Vary glow radius based on intensity
        # glow_radius = 15 + glow_intensity * 25  # 15-40 radius
        # self.glow_effect.setBlurRadius(int(glow_radius))
        
        # Debug output removed for cleaner console
    
    def _animate_aura_eye_setup(self):
        """Gentle, meditative aura eye animation during initial setup"""
        # Very slow, peaceful breathing during setup
        self.aura_breathing_phase += 0.02
        breathing_intensity = (math.sin(self.aura_breathing_phase) + 1) / 2
        
        # Subtle glow during setup
        self.aura_glow_phase += 0.015
        glow_intensity = (math.sin(self.aura_glow_phase) + 1) / 2
        
        # Combine with gentle weighting
        combined_intensity = breathing_intensity * 0.8 + glow_intensity * 0.2
        
        # Setup opacity range - more visible than idle
        self.opacity = 0.3 + combined_intensity * 0.5  # 0.3 to 0.8 range
        
        # Set opacity through widget style
        self.label.setStyleSheet(f"opacity: {self.opacity};")
        
        # Debug: Print setup animation values
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 5 seconds
            print(f"[GUI] 👁️ Aura Eye Setup: opacity={self.opacity:.3f}, breathing={breathing_intensity:.3f}, glow={glow_intensity:.3f}")
        
        # Gentle glow effect for setup state
        glow_alpha = int(40 + combined_intensity * 50)  # 40-90 alpha
        glow_color = QColor(0, 100, 255, glow_alpha)
        self.glow_effect.setColor(glow_color)
        
        # Subtle glow radius
        glow_radius = 12 + combined_intensity * 15  # 12-27 radius
        self.glow_effect.setBlurRadius(int(glow_radius))
    
    def _animate_aura_eye_idle(self):
        """Gentle, meditative aura eye animation when idle"""
        # Slow, peaceful breathing
        self.aura_breathing_phase += 0.03
        breathing_intensity = (math.sin(self.aura_breathing_phase) + 1) / 2
        
        # Very subtle glow
        self.aura_glow_phase += 0.02
        glow_intensity = (math.sin(self.aura_glow_phase) + 1) / 2
        
        # Combine with gentle weighting
        combined_intensity = breathing_intensity * 0.7 + glow_intensity * 0.3
        
        # More visible opacity range for setup
        self.opacity = 0.2 + combined_intensity * 0.6  # 0.2 to 0.8 range (more dramatic)
        
        # Set opacity through widget style (no scaling to prevent position drift)
        self.label.setStyleSheet(f"opacity: {self.opacity};")
        
        # Debug: Print opacity values occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 5 seconds
            print(f"[GUI] 👁️ Aura Eye Idle: opacity={self.opacity:.3f}, breathing={breathing_intensity:.3f}, glow={glow_intensity:.3f}")
        
        # Gentle glow effect for idle state
        glow_alpha = int(30 + combined_intensity * 40)  # 30-70 alpha
        glow_color = QColor(0, 100, 255, glow_alpha)
        self.glow_effect.setColor(glow_color)
        
        # Subtle glow radius
        glow_radius = 10 + combined_intensity * 10  # 10-20 radius
        self.glow_effect.setBlurRadius(int(glow_radius))
    
    def _reset_aura_eye(self):
        """Reset aura eye to original size"""
        try:
            if hasattr(self, '_original_pixmap') and self._original_pixmap:
                self.label.setPixmap(self._original_pixmap)
        except Exception as e:
            pass
    
    def closeEvent(self, event):
        """Handle application close event"""
        print("[AuraGUI] 🚪 Close event triggered - requesting shutdown")
        from state import request_shutdown
        request_shutdown()
        event.accept()
    
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

def set_setup_complete():
    """Mark initial setup as complete"""
    global _setup_complete
    _setup_complete = True
    print("[AuraGUI] ✅ Setup complete - switching to idle mode")
