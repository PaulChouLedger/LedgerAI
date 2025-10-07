# aura_gui.py — AuraVision GUI

import os
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QTransform, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve, QMetaObject, Q_ARG, pyqtSlot

_app = None
_window = None
_gui_ready = False
_listening_ready = False  # Tracks when system is ready to transcribe
_transcribing = False  # Tracks when user is speaking (transcription active)
_tts_playing = False  # Tracks when TTS is playing (AI speaking)
_setup_complete = False  # Tracks when initial setup is complete
_welcome_played = False  # Tracks when welcome prompt has been played
_tts_frequency = 0.15  # Current TTS frequency for pulsation speed

# Debug: Print initial state
print(f"[AuraGUI] 🎯 Initial state: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

class BorderOverlayWidget(QWidget):
    """Simple transparent overlay to paint borders on top of all other widgets"""
    def __init__(self, parent, size):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.parent_gui = parent
        
    def paintEvent(self, event):
        """Paint borders - this runs AFTER buttons paint, so borders appear on top"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        
        center = 540
        # Move border to the edge: 540 (screen edge) - 4 (half of 8px pen) - 1 (safety) = 535px
        radius = 535  # Right at the edge of the circular screen
        
        # White reference circle (always)
        white_pen = QPen(QColor(255, 255, 255), 8, Qt.SolidLine)
        painter.setPen(white_pen)
        painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
        
        # Red circle (when transcribing)
        if hasattr(self.parent_gui, 'show_red_border') and self.parent_gui.show_red_border:
            red_color = QColor(200, 0, 0)
            red_color.setAlphaF(self.parent_gui.red_border_opacity)
            red_pen = QPen(red_color, self.parent_gui.red_border_width, Qt.SolidLine)
            painter.setPen(red_pen)
            painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
        
        painter.end()

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

        # Simple: use 1080x1080 for circular screen
        window_size = 1080
        scaled_pixmap = pixmap.scaled(window_size, window_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set fixed size and position at top-left
        self.setFixedSize(window_size, window_size)
        self.move(0, 0)
        
        print(f"[AuraGUI] 📐 Window: {window_size}x{window_size} at (0,0)")

        # === Create main widget with image and circular buttons ===
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image label - simple fixed size
        self.label = QLabel()
        self.label.setPixmap(scaled_pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(window_size, window_size)
        self.label.setScaledContents(False)
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        
        # Store original pixmap for scaling effects
        self._original_pixmap = scaled_pixmap
        
        # Create opacity effect for animations (apply to label for direct control)
        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        self.opacity_effect = QGraphicsOpacityEffect(self.label)
        self.opacity_effect.setOpacity(1.0)
        self.label.setGraphicsEffect(self.opacity_effect)
        
        print(f"[GUI] 🎨 Opacity effect created and applied to label")
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Make central widget ignore mouse events so border can be on top
        main_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Keep mouse events for buttons
        
        # Store border state
        self.border_width = 8
        self.border_pulse_speed = 0.1
        self.red_border_width = 10
        self.red_border_opacity = 0.7
        self.show_red_border = False
        
        # Create 6 buttons equally spaced around the circular edge
        self.create_circular_buttons()
        
        # Create simple overlay widget for borders (ON TOP of buttons)
        self.border_overlay = BorderOverlayWidget(self, window_size)
        self.border_overlay.setGeometry(0, 0, window_size, window_size)
        self.border_overlay.raise_()
        self.border_overlay.show()
        print(f"[Border] Created overlay at (0,0), size {window_size}x{window_size}")

        # === Pulsation Effect ===
        self.opacity = 1.0
        self.pulse_direction = -1
        
        # Remove blue glow effect - it was causing issues
        # self.glow_effect = QGraphicsDropShadowEffect()
        # self.glow_effect.setBlurRadius(20)
        # self.glow_effect.setColor(QColor(0, 100, 255, 100))  # Subtle blue glow
        # self.glow_effect.setOffset(0, 0)
        # self.label.setGraphicsEffect(self.glow_effect)
        
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
        # Move buttons further inward to avoid interfering with red border
        # Edge radius is 540px, button radius is 50px, border is 8-12px, so: 540 - 20 - 50 - 20 = 450px
        radius = 450  # Distance from center to button (further from edge to avoid border interference)
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
        
        # Pass self as parent so dialog appears on top properly
        print("[AuraGUI] 📂 Showing upload dialog...")
        from file_upload_dialog import FileUploadDialog
        
        # Create and show dialog with this window as parent
        dialog = FileUploadDialog(parent=self)
        
        # The dialog should now appear on top of this window
        dialog.exec_()
        
        print("[AuraGUI] ✅ Upload dialog closed")
    
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
        
        # Ensure border overlay is on top
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()
            self.border_overlay.show()

    def animate_pulse(self):
        global _listening_ready, _transcribing, _tts_playing, _tts_frequency, _setup_complete, _welcome_played
        
        # Verify opacity effect is still attached
        if not hasattr(self, '_opacity_check_done'):
            if self.label.graphicsEffect() == self.opacity_effect:
                print(f"[GUI] ✅ Opacity effect is properly attached")
            else:
                print(f"[GUI] ❌ Opacity effect is NOT attached! Re-attaching...")
                self.label.setGraphicsEffect(self.opacity_effect)
            self._opacity_check_done = True
        
        # Debug state changes
        # After setup complete: only care about transcribing, tts_playing, and listening_ready
        # During setup: use _welcome_played to control breathing vs static
        current_state = (
            "transcribing" if _transcribing else
            "tts_playing" if _tts_playing else
            "listening_ready" if _setup_complete and _listening_ready else
            "static_ready" if _setup_complete and _welcome_played else
            "setup_breathing" if not _setup_complete else
            "waiting_for_welcome"
        )
        
        if not hasattr(self, '_last_state'):
            self._last_state = None
            
        if self._last_state != current_state:
            print(f"[GUI] 🎭 Animation state changed: {self._last_state} → {current_state}")
            print(f"      _setup_complete={_setup_complete}, _welcome_played={_welcome_played}, _listening_ready={_listening_ready}, _transcribing={_transcribing}, _tts_playing={_tts_playing}")
            print(f"      Current opacity: {self.opacity_effect.opacity():.3f}")
        self._last_state = current_state
        
        # PRIORITY: Check transcription state FIRST to avoid it being hidden by other states
        if _transcribing:
            # Keep aura eye fully visible during transcription
            self.opacity_effect.setOpacity(1.0)
            
            # Border is now drawn via paintEvent - no need to show/hide widget
            
            # Get real-time voice frequency from audio analysis
            try:
                from listener import get_transcription_frequency
                voice_freq = get_transcription_frequency()
                # voice_freq is 0.0 to 1.0 based on amplitude and pitch
            except ImportError:
                voice_freq = 0.5  # Moderate default
            except Exception as e:
                voice_freq = 0.5  # Fallback
            
            # Create organic pulsation based on voice characteristics
            # Use multiple sine waves at different frequencies for natural feel
            
            # Primary pulse (follows voice frequency closely) - VERY FAST for speech dynamics
            primary_speed = 0.4 + (voice_freq * 1.0)  # 0.4 to 1.4 range (4x faster than original)
            self.border_pulse_phase += primary_speed
            primary_pulse = (math.sin(self.border_pulse_phase) + 1) / 2
            
            # Secondary pulse (adds organic variation) - VERY FAST
            if not hasattr(self, 'secondary_phase'):
                self.secondary_phase = 0.0
            self.secondary_phase += 0.25  # Much faster
            secondary_pulse = (math.sin(self.secondary_phase) + 1) / 2
            
            # Tertiary pulse (micro variations for organic feel) - VERY FAST
            if not hasattr(self, 'tertiary_phase'):
                self.tertiary_phase = 0.0
            self.tertiary_phase += 0.45  # Much faster
            tertiary_pulse = (math.sin(self.tertiary_phase * 1.7) + 1) / 2
            
            # Combine pulses with voice frequency weighting
            # Higher voice frequency = more influence from primary pulse
            combined_pulse = (
                primary_pulse * (0.7 + voice_freq * 0.2) +      # 70-90% primary (highly responsive)
                secondary_pulse * (0.2 - voice_freq * 0.05) +   # 20-15% secondary  
                tertiary_pulse * 0.1                             # 10% micro variation
            )
            
            # Calculate border width based on combined pulse and voice intensity
            # Make it MUCH more dynamic and thicker
            base_width = 10  # Increased from 6
            max_variation = 15  # Increased from 10 for dramatic pulsation
            self.border_width = int(base_width + combined_pulse * max_variation * (0.7 + voice_freq))
            
            # Wider range for more dramatic effect
            self.border_width = max(10, min(self.border_width, 25))
            
            # Calculate opacity variation - consistent visibility
            border_opacity = 0.6 + combined_pulse * 0.3  # 0.6 to 0.9 opacity (more visible)
            
            # Debug logging (occasional)
            if hasattr(self, '_border_debug_counter'):
                self._border_debug_counter += 1
            else:
                self._border_debug_counter = 0
            
            if self._border_debug_counter % 20 == 0:  # Print every second
                print(f"[GUI] 🔴 Transcribing: freq={voice_freq:.3f}, width={self.border_width}px, pulse={combined_pulse:.3f}, opacity={border_opacity:.3f}")
            
            # Update red border state and trigger overlay repaint
            self.red_border_width = int(self.border_width)
            self.red_border_opacity = border_opacity
            self.show_red_border = True
            self.border_overlay.raise_()  # Keep on top every frame
            self.border_overlay.update()  # Trigger overlay paintEvent
            
        # State 1: TTS playing - dramatic pulsation (priority after transcribing)
        elif _tts_playing:
            self._animate_aura_eye_tts(_tts_frequency)
            self.show_red_border = False
            
        # State 2: Setup complete AND listening ready - STATIC (normal idle state)
        elif _setup_complete and _listening_ready:
            self._set_aura_eye_static()  # Static, ready for interaction
            self.show_red_border = False
            
        # State 3: Setup complete but welcome not played - continue breathing
        elif _setup_complete and not _welcome_played:
            self._animate_aura_eye_breathing()  # Continue breathing until welcome plays
            self.show_red_border = False
            
        # State 4: Welcome played but listener not ready yet - STATIC
        elif _setup_complete and _welcome_played:
            self._set_aura_eye_static()  # Static while waiting for listener
            self.show_red_border = False
            
        # State 5: Initial setup - breathing animation (faster, more noticeable)
        elif not _setup_complete:
            self._animate_aura_eye_breathing()  # Breathing during setup
            self.show_red_border = False
    
    def _update_border_style(self, static=True, pulsating=False, width=5, opacity=0.7):
        """Update the border style with organic opacity variation"""
        try:
            if pulsating:
                # Show and animate pulsating red border with solid color for consistency
                window_size = self.size()
                border_radius = min(window_size.width(), window_size.height()) // 2
                
                # Use solid RGB with widget opacity instead of RGBA for consistent rendering
                # This prevents opaque sections at button locations
                self.border_widget.setStyleSheet(f"""
                    QWidget {{
                        background-color: transparent;
                        border: {width}px solid rgb(200, 0, 0);
                        border-radius: {border_radius}px;
                    }}
                """)
                
                # Set widget opacity separately for consistent alpha across entire widget
                self.border_widget.setWindowOpacity(opacity)
                
                # Ensure proper geometry
                self.border_widget.setGeometry(0, 0, window_size.width(), window_size.height())
                
                # Force show and raise - critical for visibility
                if not self.border_widget.isVisible():
                    self.border_widget.show()
                    print(f"[GUI] 🔴 Border forced visible in update_style")
                
                # Always raise to ensure it's on top
                self.border_widget.raise_()
                
                # Update the widget to force repaint
                self.border_widget.update()
                
                # Debug output (only once)
                if not hasattr(self, '_border_shown_once'):
                    self._border_shown_once = True
                    print(f"[GUI] 🔴 Border widget shown: visible={self.border_widget.isVisible()}, geometry={self.border_widget.geometry()}, radius={border_radius}px, width={width}px")
            else:
                # Hide border completely
                self.border_widget.hide()
                return
            
            # Ensure border widget is always on top
            self.border_widget.raise_()
        except Exception as e:
            print(f"[GUI] ❌ Border update error: {e}")
            import traceback
            traceback.print_exc()
            # Don't let errors break the border - try to keep it visible
            try:
                if pulsating and not self.border_widget.isVisible():
                    self.border_widget.show()
                    self.border_widget.raise_()
            except:
                pass
    
    def _animate_aura_eye_tts(self, tts_frequency):
        """Dramatic pulsation during TTS - same as red border during transcription"""
        # Use the same multi-layer pulsation as the red border for consistency
        
        # Primary pulse (fast, responsive)
        primary_speed = 0.4 + (tts_frequency * 1.0)  # 0.4 to 1.4 range
        if not hasattr(self, 'tts_pulse_phase'):
            self.tts_pulse_phase = 0.0
        self.tts_pulse_phase += primary_speed
        primary_pulse = (math.sin(self.tts_pulse_phase) + 1) / 2
        
        # Secondary pulse (adds organic variation)
        if not hasattr(self, 'tts_secondary_phase'):
            self.tts_secondary_phase = 0.0
        self.tts_secondary_phase += 0.25
        secondary_pulse = (math.sin(self.tts_secondary_phase) + 1) / 2
        
        # Tertiary pulse (micro variations for organic feel)
        if not hasattr(self, 'tts_tertiary_phase'):
            self.tts_tertiary_phase = 0.0
        self.tts_tertiary_phase += 0.45
        tertiary_pulse = (math.sin(self.tts_tertiary_phase) + 1) / 2
        
        # Combine with weighted influence (60% primary, 25% secondary, 15% tertiary)
        combined_pulse = (
            primary_pulse * 0.60 +
            secondary_pulse * 0.25 +
            tertiary_pulse * 0.15
        )
        
        # Calculate dramatic opacity variation (0.2 to 1.0)
        self.opacity = 0.2 + combined_pulse * 0.8
        
        # Apply opacity using graphics effect
        self.opacity_effect.setOpacity(self.opacity)
        
        # Debug: Print TTS animation values
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 20 == 0:  # Print every second during TTS
            print(f"[GUI] 👁️ Aura Eye TTS: opacity={self.opacity:.3f}, pulse={combined_pulse:.3f}, freq={tts_frequency:.3f}")
        
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
    
    def _animate_aura_eye_breathing(self):
        """Faster, more noticeable breathing animation during setup"""
        # Faster breathing rhythm (2 second cycle - 2x faster than before)
        self.aura_breathing_phase += 0.05  # Doubled from 0.025
        breathing = (math.sin(self.aura_breathing_phase) + 1) / 2
        
        # More dramatic opacity variation (0.2 to 1.0)
        self.opacity = 0.2 + (breathing * 0.8)
        
        # Apply opacity using graphics effect
        self.opacity_effect.setOpacity(self.opacity)
        
        # Debug output
        if not hasattr(self, '_breathing_debug_counter'):
            self._breathing_debug_counter = 0
        self._breathing_debug_counter += 1
        if self._breathing_debug_counter % 20 == 0:
            print(f"[GUI] 🌬️ BREATHING: opacity={self.opacity:.3f}, phase={self.aura_breathing_phase:.3f}, breathing={breathing:.3f}")
    
    def _set_aura_eye_static(self):
        """Set aura eye to static state - no animation, ready for interaction"""
        # Full opacity, no pulsation
        self.opacity = 1.0
        self.opacity_effect.setOpacity(1.0)
        
        # Debug output
        if not hasattr(self, '_static_debug_logged'):
            self._static_debug_logged = True
            print(f"[GUI] 👁️ STATIC: opacity={self.opacity:.3f} (no animation)")
    
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
        self.opacity = 0.1 + combined_intensity * 0.7  # 0.1 to 0.8 range
        
        # Apply opacity using graphics effect
        self.opacity_effect.setOpacity(self.opacity)
        
        # Debug: Print setup animation values
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 5 seconds
            print(f"[GUI] 👁️ Aura Eye Setup: opacity={self.opacity:.3f}, breathing={breathing_intensity:.3f}, glow={glow_intensity:.3f}")
        
        # Remove blue glow effect - was causing issues
        # glow_alpha = int(40 + combined_intensity * 50)  # 40-90 alpha
        # glow_color = QColor(0, 100, 255, glow_alpha)
        # self.glow_effect.setColor(glow_color)
        
        # Subtle glow radius
        # glow_radius = 12 + combined_intensity * 15  # 12-27 radius
        # self.glow_effect.setBlurRadius(int(glow_radius))
    
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
        
        # More visible opacity range for idle
        self.opacity = 0.1 + combined_intensity * 0.7  # 0.1 to 0.8 range (more dramatic)
        
        # Apply opacity using graphics effect
        self.opacity_effect.setOpacity(self.opacity)
        
        # Debug: Print opacity values occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 5 seconds
            print(f"[GUI] 👁️ Aura Eye Idle: opacity={self.opacity:.3f}, breathing={breathing_intensity:.3f}, glow={glow_intensity:.3f}")
        
        # Remove blue glow effect - was causing issues
        # glow_alpha = int(30 + combined_intensity * 40)  # 30-70 alpha
        # glow_color = QColor(0, 100, 255, glow_alpha)
        # self.glow_effect.setColor(glow_color)
        
        # Subtle glow radius
        # glow_radius = 10 + combined_intensity * 10  # 10-20 radius
        # self.glow_effect.setBlurRadius(int(glow_radius))
    
    def _reset_aura_eye(self):
        """Reset aura eye to original size"""
        try:
            if hasattr(self, '_original_pixmap') and self._original_pixmap:
                self.label.setPixmap(self._original_pixmap)
        except Exception as e:
            pass
    
    @pyqtSlot(bool)
    def _update_transcribing_state(self, active):
        """Thread-safe method to update transcribing state (must be called from GUI thread)"""
        global _transcribing
        _transcribing = active
        if active:
            print("[AuraGUI] 🔴 Transcription active - red edge pulsating")
            # Reset animation phases for consistent start
            self.border_pulse_phase = 0.0
            if hasattr(self, 'secondary_phase'):
                self.secondary_phase = 0.0
            if hasattr(self, 'tertiary_phase'):
                self.tertiary_phase = 0.0
            
            # Enable red border (will be drawn in paintEvent)
            self.show_red_border = True
            self.update()
        else:
            print("[AuraGUI] ⚫ Transcription ended")
            # Disable red border
            self.show_red_border = False
            self.update()
        print(f"[AuraGUI] 🔴 State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")
    
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

def set_welcome_played():
    """Signal that welcome prompt has been played - makes aura eye static"""
    global _welcome_played
    _welcome_played = True
    print("[AuraGUI] 👋 Welcome prompt played - aura eye now static and ready")

def set_transcribing(active):
    """Set transcription state - red edge pulsation when user is speaking (thread-safe)"""
    global _window
    if _window:
        # Use Qt's thread-safe mechanism to update GUI from any thread
        QMetaObject.invokeMethod(_window, "_update_transcribing_state",
                                Qt.QueuedConnection,
                                Q_ARG(bool, active))
    else:
        print("[AuraGUI] ⚠️ Window not initialized, cannot update transcribing state")

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
