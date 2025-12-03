# aura_gui.py — AuraVision GUI

import os
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QTransform, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve, QMetaObject, Q_ARG, pyqtSlot

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

_app = None
_window = None
_gui_ready = False
_listening_ready = False  # Tracks when system is ready to transcribe
_transcribing = False  # Tracks when user is speaking (transcription active)
_wake_word_detected = False  # Tracks when wake word is detected (solid red LED)
_tts_playing = False  # Tracks when TTS is playing (AI speaking)
_setup_complete = False  # Tracks when initial setup is complete
_welcome_played = False  # Tracks when welcome prompt has been played
_tts_frequency = 0.15  # Current TTS frequency for pulsation speed
_microphone_muted = False  # Tracks when microphone is muted via button

# Debug: Print initial state
print(f"[AuraGUI] 🎯 Initial state: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

class SafeModeButton(QPushButton):
    """Completely transparent round button that opens safe mode when held for 3 seconds"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(540, 540)  # 50% of 1080x1080 aura eye size
        
        # Make completely transparent - no visual impact
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                background-color: transparent;
            }
        """)
        
        # Ensure it doesn't block visual content but still receives mouse events
        # Don't use WA_NoSystemBackground as it might interfere with mouse events
        
        # Hold gesture tracking
        self.hold_timer = QTimer()
        self.hold_timer.setSingleShot(True)
        self.hold_timer.timeout.connect(self._on_hold_complete)
        self.is_holding = False
        self.hold_start_time = 0
        self.HOLD_DURATION_MS = 3000  # 3 seconds
        
        # Make button circular and transparent - cursor shows it's interactive
        self.setCursor(Qt.PointingHandCursor)
    
    def paintEvent(self, event):
        """Override paint event to ensure button is completely invisible"""
        # Don't paint anything - completely transparent
        pass
        
    def mousePressEvent(self, event):
        """Start hold timer when mouse is pressed"""
        if event.button() == Qt.LeftButton:
            self.is_holding = True
            self.hold_start_time = QTimer().remainingTime() if hasattr(QTimer(), 'remainingTime') else 0
            self.hold_timer.start(self.HOLD_DURATION_MS)
            print("[SafeModeButton] 👆 Hold started... (3 seconds)")
            super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Cancel hold timer when mouse is released"""
        if event.button() == Qt.LeftButton:
            if self.hold_timer.isActive():
                self.hold_timer.stop()
                print("[SafeModeButton] 👆 Hold cancelled")
            self.is_holding = False
            super().mouseReleaseEvent(event)
    
    def _on_hold_complete(self):
        """Open safe mode dialog when hold is complete (same as welcome screen)"""
        print("[SafeModeButton] ✅ Hold complete - opening Safe Mode dialog...")
        self.is_holding = False
        
        # Open safe mode dialog (same as welcome screen)
        try:
            from gui.safe_mode_dialog import SafeModeDialog
            # Get the main window as parent
            parent_window = self.parent()
            while parent_window and not isinstance(parent_window, QMainWindow):
                parent_window = parent_window.parent()
            
            dialog = SafeModeDialog(parent=parent_window)
            dialog.exec_()
            print("[SafeModeButton] ✅ Safe Mode dialog closed")
        except Exception as e:
            print(f"[SafeModeButton] ❌ Error opening Safe Mode: {e}")
            import traceback
            traceback.print_exc()

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
        
        # White reference circle (always) - 70% transparent (30% opacity)
        white_color = QColor(255, 255, 255, 77)  # Alpha: 77/255 = 30% opacity
        white_pen = QPen(white_color, 8, Qt.SolidLine)
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
        img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/aura_eye.png"))
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
        
        # Create safe mode button (50% size of aura eye, centered)
        self.safe_mode_button = SafeModeButton(self)
        button_size = 540  # 50% of 1080
        center_x = (window_size - button_size) // 2
        center_y = (window_size - button_size) // 2
        self.safe_mode_button.setGeometry(center_x, center_y, button_size, button_size)
        self.safe_mode_button.hide()  # Hidden initially, shown when eye is visible
        self.safe_mode_button.setEnabled(True)  # Ensure it can receive mouse events
        print(f"[SafeModeButton] Created at ({center_x},{center_y}), size {button_size}x{button_size}")
        
        # Create simple overlay widget for borders (ON TOP of everything including safe mode button)
        self.border_overlay = BorderOverlayWidget(self, window_size)
        self.border_overlay.setGeometry(0, 0, window_size, window_size)
        self.border_overlay.show()
        self.border_overlay.raise_()  # Always on top
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
            ("↑", "Upload", self._handle_upload, "#4D94D9"),      # Upload files - Muted blue
            ("⚙", "Settings", self._handle_settings, "#4D94D9"),  # Settings - Muted blue
            ("📊", "Analytics", self._handle_analytics, "#4D94D9"), # Analytics - Muted blue
            ("🎤", "Voice", self._handle_voice, "#4D94D9"),        # Voice control - Muted blue
            ("📱", "Mobile", self._handle_mobile, "#4D94D9"),     # Mobile sync - Muted blue
            ("ℹ", "Info", self._handle_info, "#4D94D9")          # Information - Muted blue
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
                    /* Muted radial gradient with softer center */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                        fx:0.5, fy:0.5,
                        stop:0 #E6E6E6, stop:0.6 {color}, stop:1 {color});
                    color: #1A1A1A;
                    font-size: 36px;
                    font-weight: bold;
                    border-radius: 50px;
                    /* Subtle border */
                    border: 2px solid rgba(0, 0, 0, 0.2);
                    padding: 0px;
                }}
                QPushButton:hover {{
                    /* Slightly brighter on hover */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                        fx:0.5, fy:0.5,
                        stop:0 #F2F2F2, stop:0.5 #99BBE6, stop:1 {color});
                    border: 2px solid rgba(255, 255, 255, 0.3);
                }}
                QPushButton:pressed {{
                    /* Darker when pressed */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                        fx:0.5, fy:0.5,
                        stop:0 {color}, stop:1 #002040);
                    border: 2px solid rgba(0, 0, 0, 0.4);
                }}
            """)
            
            # Add depth shadow effect (dark shadow for depth, not colored glow)
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(20)  # Moderate blur for soft shadow
            shadow_effect.setColor(QColor(0, 0, 0, 100))  # Dark shadow with transparency
            shadow_effect.setOffset(0, 3)  # Slight downward offset for depth
            btn.setGraphicsEffect(shadow_effect)
            
            # Connect handler
            btn.clicked.connect(handler)
            self.buttons.append(btn)
            
            # Add button to the main widget (positioned absolutely)
            btn.setParent(self.centralWidget())
            btn.move(int(x), int(y))
            btn.hide()  # Start hidden, show after welcome prompt
    
    def _handle_upload(self):
        """Handle upload button click"""
        print("[AuraGUI] 📤 Upload button clicked")
        
        # Block transcription while dialog is open
        try:
            from listener import block_transcription, unblock_transcription
            block_transcription("Upload dialog open")
        except ImportError:
            print("[AuraGUI] ⚠️ Could not import listener blocking functions")
        
        # Pass self as parent so dialog appears on top properly
        print("[AuraGUI] 📂 Showing memory dialog...")
        from gui.file_upload_dialog import MemoryDialog
        
        # Create and show dialog with this window as parent
        dialog = MemoryDialog(parent=self)
        
        # The dialog should now appear on top of this window
        dialog.exec_()
        
        # Unblock transcription when dialog closes
        try:
            unblock_transcription()
        except:
            pass
        
        print("[AuraGUI] ✅ Memory dialog closed")
    
    def _handle_settings(self):
        """Handle settings button click"""
        print("[AuraGUI] ⚙️ Settings button clicked")
        
        # Block transcription while dialog is open
        try:
            from listener import block_transcription, unblock_transcription
            block_transcription("Settings dialog open")
        except ImportError:
            print("[AuraGUI] ⚠️ Could not import listener blocking functions")
        
        try:
            from gui.settings_dialog import SettingsDialog
            
            # Create and show settings dialog (modal, like upload dialog)
            dialog = SettingsDialog(parent=self)
            dialog.exec_()  # Modal blocking call
            
            print("[AuraGUI] ✅ Settings dialog closed")
        except ImportError as e:
            print(f"[AuraGUI] ❌ Settings dialog not available: {e}")
        except Exception as e:
            print(f"[AuraGUI] ❌ Error opening settings dialog: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always unblock transcription when done
            try:
                unblock_transcription()
            except:
                pass
    
    def _handle_analytics(self):
        """Handle analytics button click - show wallet & token balance"""
        print("[AuraGUI] 📊 Analytics button clicked - opening wallet dialog")
        
        try:
            from gui.wallet_dialog import WalletDialog
            
            # Check if dialog exists and is still valid
            if hasattr(self, "_wallet_dialog") and self._wallet_dialog is not None:
                try:
                    # Check if dialog is still valid (not deleted) and visible
                    # This will raise RuntimeError if the C++ object was deleted
                    if self._wallet_dialog.isVisible():
                        # Dialog is already open, just raise it
                        self._wallet_dialog.raise_()
                        self._wallet_dialog.activateWindow()
                        print("[AuraGUI] ✅ Wallet dialog already open, raised to front")
                        return
                    else:
                        # Dialog exists but is hidden, close it first (safely)
                        try:
                            self._wallet_dialog.close()
                        except RuntimeError:
                            pass  # Already deleted
                        self._wallet_dialog = None
                except (RuntimeError, AttributeError):
                    # Dialog was deleted (Qt.WA_DeleteOnClose), reference is invalid
                    # RuntimeError: wrapped C/C++ object has been deleted
                    self._wallet_dialog = None
                    print("[AuraGUI] 🗑️ Previous wallet dialog was deleted, creating new one")
                except Exception as e:
                    print(f"[AuraGUI] ⚠️ Error checking existing dialog: {e}")
                    self._wallet_dialog = None
            
            # Create new dialog (wrapped in try-except to catch initialization errors)
            try:
                self._wallet_dialog = WalletDialog(parent=self)
                
                # Connect to destroyed signal to clear reference when dialog is deleted
                def on_dialog_destroyed():
                    if hasattr(self, "_wallet_dialog"):
                        self._wallet_dialog = None
                        print("[AuraGUI] 🗑️ Wallet dialog reference cleared")
                
                self._wallet_dialog.destroyed.connect(on_dialog_destroyed)
                
                # Show modal dialog with immediate opacity (skip fade animation for faster response)
                # This matches the pattern used in settings dialogs
                self._wallet_dialog.setWindowOpacity(1.0)  # Show immediately without fade
                self._wallet_dialog.show()
                self._wallet_dialog.raise_()
                self._wallet_dialog.activateWindow()
                QApplication.processEvents()  # Ensure dialog is rendered
                self._wallet_dialog.exec_()  # Use exec_() for modal dialogs
                # Clear reference after dialog closes (exec_() returns)
                self._wallet_dialog = None
                print("[AuraGUI] ✅ Wallet dialog closed and reference cleared")
            except Exception as e:
                print(f"[AuraGUI] ❌ Error creating wallet dialog: {e}")
                import traceback
                traceback.print_exc()
                # Clear invalid reference
                self._wallet_dialog = None
                # Show error to user
                try:
                    QMessageBox.critical(self, "Wallet Dialog Error", 
                                       f"Failed to open wallet dialog:\n{e}\n\n"
                                       "Please check the console for details.")
                except Exception:
                    pass
        except ImportError as e:
            print(f"[AuraGUI] ❌ Wallet dialog not available: {e}")
            print(f"[AuraGUI] 💡 Install web3: pip install web3")
        except Exception as e:
            print(f"[AuraGUI] ❌ Error opening wallet dialog: {e}")
            import traceback
            traceback.print_exc()
            # Clear invalid reference
            if hasattr(self, "_wallet_dialog"):
                self._wallet_dialog = None
    
    def _handle_voice(self):
        """Handle voice button click - toggle transcription blocking"""
        print("[AuraGUI] 🎤 Voice button clicked")
        
        try:
            from listener import toggle_transcription, is_transcription_blocked
            
            # Toggle the transcription state
            now_blocked = toggle_transcription()
            
            # Get the voice button (index 3 in button_configs)
            voice_btn = self.buttons[3] if len(self.buttons) > 3 else None
            
            if now_blocked:
                print("[AuraGUI] 🚫 Microphone MUTED - transcription blocked")
                # Update global state
                set_microphone_muted(True)
                
                # Update button to RED to show muted state
                if voice_btn:
                    voice_btn.setStyleSheet(f"""
                        QPushButton {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                fx:0.5, fy:0.5,
                                stop:0 #FFE6E6, stop:0.6 #DC143C, stop:1 #DC143C);
                            color: #FFFFFF;
                            font-size: 36px;
                            font-weight: bold;
                            border-radius: 50px;
                            border: 3px solid #FF0000;
                            padding: 0px;
                        }}
                        QPushButton:hover {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                fx:0.5, fy:0.5,
                                stop:0 #FFE6E6, stop:0.6 #FF1744, stop:1 #FF1744);
                            border: 3px solid #FF4444;
                        }}
                    """)
            else:
                print("[AuraGUI] ✅ Microphone ACTIVE - transcription enabled")
                # Update global state
                set_microphone_muted(False)
                
                # Update button to BLUE (normal state)
                if voice_btn:
                    voice_btn.setStyleSheet(f"""
                        QPushButton {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                fx:0.5, fy:0.5,
                                stop:0 #E6E6E6, stop:0.6 #4D94D9, stop:1 #4D94D9);
                            color: #1A1A1A;
                            font-size: 36px;
                            font-weight: bold;
                            border-radius: 50px;
                            border: 2px solid rgba(0, 0, 0, 0.2);
                            padding: 0px;
                        }}
                        QPushButton:hover {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                fx:0.5, fy:0.5,
                                stop:0 #F0F0F0, stop:0.6 #5EA5E8, stop:1 #5EA5E8);
                        }}
                    """)
                
        except ImportError:
            print("[AuraGUI] ⚠️ Could not import listener blocking functions")
        except Exception as e:
            print(f"[AuraGUI] ❌ Error toggling transcription: {e}")
    
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
        
        # Ensure border overlay is always visible and on top
        if hasattr(self, 'border_overlay'):
            if not self.border_overlay.isVisible():
                self.border_overlay.show()
            self.border_overlay.raise_()
            # Update border overlay to ensure it repaints
            self.border_overlay.update()
        
        # Verify opacity effect is still attached and enabled
        if not hasattr(self, '_opacity_check_done'):
            if self.label.graphicsEffect() == self.opacity_effect:
                print(f"[GUI] ✅ Opacity effect is properly attached")
                print(f"[GUI]    Effect enabled: {self.opacity_effect.isEnabled()}")
                print(f"[GUI]    Current opacity: {self.opacity_effect.opacity():.3f}")
            else:
                print(f"[GUI] ❌ Opacity effect is NOT attached! Re-attaching...")
                self.label.setGraphicsEffect(self.opacity_effect)
            
            # Force enable it
            if not self.opacity_effect.isEnabled():
                print(f"[GUI] ⚠️ Opacity effect was disabled! Enabling...")
                self.opacity_effect.setEnabled(True)
            
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
        
        # PRIORITY: Check wake word detected state (solid red, no pulsation)
        if _wake_word_detected and not _transcribing:
            # Wake word detected but no speech yet - show solid red LED
            self.opacity_effect.setOpacity(1.0)
            # Solid red border (no pulsation)
            self.red_border_width = 10
            self.red_border_opacity = 0.8  # Solid, visible red
            self.show_red_border = True
            self.border_overlay.raise_()
            self.border_overlay.update()
        
        # PRIORITY: Check transcription state (pulsating red when speech detected)
        elif _transcribing:
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
            # Update red border state and trigger overlay repaint
            self.red_border_width = int(self.border_width)
            self.red_border_opacity = border_opacity
            self.show_red_border = True
            # Ensure border overlay is visible and on top
            if hasattr(self, 'border_overlay'):
                self.border_overlay.show()
                self.border_overlay.raise_()  # Keep on top every frame
                self.border_overlay.update()  # Trigger overlay paintEvent
            
        # State 1: TTS playing - dramatic pulsation (priority after transcribing)
        elif _tts_playing:
            self._animate_aura_eye_tts(_tts_frequency)
            self.show_red_border = False
            
        # State 2: Microphone muted - DIM/GRAYED to show inactive state
        elif _microphone_muted:
            self._set_aura_eye_muted()  # Dim, grayed out
            self.show_red_border = False
            # Ensure border overlay is visible and updates
            if hasattr(self, 'border_overlay'):
                self.border_overlay.show()
                self.border_overlay.raise_()
                self.border_overlay.update()
            
        # State 3: Setup complete AND listening ready - STATIC (normal idle state)
        elif _setup_complete and _listening_ready:
            self._set_aura_eye_static()  # Static, ready for interaction
            self.show_red_border = False
            # Ensure border overlay is visible and updates
            if hasattr(self, 'border_overlay'):
                self.border_overlay.show()
                self.border_overlay.raise_()
                self.border_overlay.update()
            
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
        """Highly organic TTS pulsation - smooth, speech-like, breathing feel"""
        # Initialize phases and smoothing
        if not hasattr(self, 'tts_organic_timer'):
            self.tts_organic_timer = 0.0
        if not hasattr(self, 'tts_slow_wave'):
            self.tts_slow_wave = 0.0
        if not hasattr(self, 'tts_medium_wave'):
            self.tts_medium_wave = 0.0
        if not hasattr(self, 'tts_fast_wave'):
            self.tts_fast_wave = 0.0
        if not hasattr(self, 'tts_breath_wave'):
            self.tts_breath_wave = 0.0
        if not hasattr(self, 'tts_opacity_smooth'):
            self.tts_opacity_smooth = 0.6  # Start at mid-point
        
        # Update organic timer
        self.tts_organic_timer += 0.015
        
        # Create natural drift for all waves (prevents mechanical repetition)
        drift = math.sin(self.tts_organic_timer * 0.2) * 0.08
        
        # Slow wave - like breathing during speech (baseline, faster than before)
        self.tts_slow_wave += 0.045 + drift * 0.5  # Increased from 0.025
        slow = (math.sin(self.tts_slow_wave) + 1) / 2
        
        # Medium wave - like sentence phrasing (faster cadence)
        self.tts_medium_wave += 0.15 + drift  # Increased from 0.08
        medium = (math.sin(self.tts_medium_wave) + 1) / 2
        
        # Fast wave - like word/syllable rhythm (much faster)
        self.tts_fast_wave += 0.40 + math.sin(self.tts_organic_timer * 0.4) * 0.20  # Increased from 0.25
        fast = (math.sin(self.tts_fast_wave) + 1) / 2
        
        # Breathing overlay - faster, more visible
        self.tts_breath_wave += 0.035  # Increased from 0.02
        breath = (math.sin(self.tts_breath_wave) + 1) / 2
        
        # Micro variations - complex, organic detail
        micro1 = math.sin(self.tts_organic_timer * 1.3)
        micro2 = math.sin(self.tts_organic_timer * 2.1)
        micro3 = math.sin(self.tts_organic_timer * 3.7)
        micro_combined = (micro1 + micro2 + micro3) / 12  # Very subtle
        
        # Combine waves with smooth, speech-like weighting
        # Emphasize slower waves for smooth, organic feel
        combined_raw = (
            slow * 0.35 +           # 35% slow wave (smooth baseline)
            medium * 0.30 +         # 30% medium wave (cadence)
            fast * 0.20 +           # 20% fast wave (syllables, reduced)
            breath * 0.10 +         # 10% breathing (life)
            micro_combined * 0.05   # 5% micro (organic detail)
        )
        
        # Apply double smoothing for smooth but responsive transitions
        # First smooth: exponential moving average (less smoothing for faster response)
        smooth_factor = 0.35  # Increased from 0.25 for more responsiveness
        self.tts_opacity_smooth += (combined_raw - self.tts_opacity_smooth) * smooth_factor
        
        # Second smooth: cosine easing
        eased = 0.5 - 0.5 * math.cos(self.tts_opacity_smooth * math.pi)
        
        # Map to opacity range - extremely wide for maximum dramatic effect
        min_opacity = 0.10  # Very low minimum - extreme dimming
        max_opacity = 1.00  # Full brightness - maximum peaks
        target_opacity = min_opacity + eased * (max_opacity - min_opacity)
        
        # Final smoothing pass for smooth but highly responsive transitions
        if not hasattr(self, 'tts_final_opacity'):
            self.tts_final_opacity = target_opacity
        self.tts_final_opacity += (target_opacity - self.tts_final_opacity) * 0.5  # Increased from 0.4 for faster response
        
        self.opacity = self.tts_final_opacity
        
        # Apply opacity using graphics effect
        self.opacity_effect.setOpacity(self.opacity)
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
        
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
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
    
    def _set_aura_eye_static(self):
        """Set aura eye to static state - no animation, ready for interaction"""
        # Full opacity, no pulsation
        self.opacity = 1.0
        self.opacity_effect.setOpacity(1.0)
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
    
    def _set_aura_eye_muted(self):
        """Set aura eye to muted state - dim and grayed out to show microphone is off"""
        # Dim opacity to show inactive/muted state
        self.opacity = 0.3  # 30% opacity (very dim)
        self.opacity_effect.setOpacity(0.3)
        
        # Optional: Add subtle slow pulsation to show it's not frozen, just muted
        if not hasattr(self, 'muted_pulse_phase'):
            self.muted_pulse_phase = 0
        
        self.muted_pulse_phase += 0.01  # Very slow pulse
        pulse = (math.sin(self.muted_pulse_phase) + 1) / 2  # 0 to 1
        dimmed_opacity = 0.25 + (pulse * 0.1)  # Pulse between 0.25 and 0.35
        self.opacity = dimmed_opacity
        self.opacity_effect.setOpacity(dimmed_opacity)
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
    
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
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
        
        # Debug: Print setup animation values
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
        
        # Update safe mode button visibility based on aura eye opacity
        self._update_safe_mode_button_visibility()
        
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
    
    def _update_safe_mode_button_visibility(self):
        """Update safe mode button visibility based on aura eye opacity"""
        if hasattr(self, 'safe_mode_button'):
            # Show button only when aura eye is visible (opacity > 0)
            if self.opacity > 0:
                if not self.safe_mode_button.isVisible():
                    self.safe_mode_button.show()
                    # Position it above the eye but below border overlay
                    self.safe_mode_button.raise_()
                    if hasattr(self, 'border_overlay'):
                        self.border_overlay.raise_()  # Border always on top
            else:
                if self.safe_mode_button.isVisible():
                    self.safe_mode_button.hide()
    
    @pyqtSlot()
    def _show_buttons(self):
        """Thread-safe method to show buttons (must be called from GUI thread)"""
        if hasattr(self, 'buttons') and self.buttons:
            print(f"[AuraGUI] 🔘 Showing {len(self.buttons)} buttons...")
            for i, btn in enumerate(self.buttons):
                btn.show()
                btn.raise_()  # Ensure buttons are on top
                btn.update()  # Force repaint
                print(f"[AuraGUI]   Button {i+1}: visible={btn.isVisible()}, enabled={btn.isEnabled()}")
            
            # Ensure window is active and focused
            self.raise_()
            self.activateWindow()
            
            # Force event processing to ensure visibility
            QApplication.processEvents()
            print("[AuraGUI] 🔘 Buttons should now be visible")
        else:
            print("[AuraGUI] ⚠️ No buttons to show (buttons list is empty or missing)")
    
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
    
    @pyqtSlot(bool)
    def _update_wake_word_detected_state(self, active):
        """Thread-safe method to update wake word detected state (must be called from GUI thread)"""
        global _wake_word_detected
        _wake_word_detected = active
        if active:
            print("[AuraGUI] 🔴 Wake word detected - red edge solid (waiting for speech)")
            # Show solid red border (no pulsation)
            self.red_border_width = 10
            self.red_border_opacity = 0.8
            self.show_red_border = True
            self.update()
        else:
            print("[AuraGUI] ⚫ Wake word state cleared")
            # Only clear if not transcribing
            if not _transcribing:
                self.show_red_border = False
            self.update()
        print(f"[AuraGUI] 🔴 State: wake_word_detected={_wake_word_detected}, transcribing={_transcribing}, tts_playing={_tts_playing}")
    
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
    # Display diagnostics
    display = os.environ.get("DISPLAY", "NOT SET")
    print(f"[AuraGUI] 🎨 Launching GUI with DISPLAY={display}")
    
    _app = QApplication(sys.argv)
    _window = AuraGUI()
    
    # Ensure window is visible and on top
    _window.showFullScreen()
    _window.raise_()
    _window.activateWindow()
    
    # Force event processing to ensure window renders
    _app.processEvents()
    
    # Additional processing to ensure everything is rendered
    import time
    time.sleep(0.1)  # Small delay to let window fully render
    _app.processEvents()
    
    print(f"[AuraGUI] ✅ Window created and shown (size: {_window.width()}x{_window.height()})")

def is_gui_ready():
    """Check if GUI is fully initialized and visible"""
    return _gui_ready

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
    print("[AuraGUI] 🎯 Listener is now READY - aura eye should be STATIC")
    print(f"[AuraGUI] 🎯 State: listening_ready={_listening_ready}, transcribing={_transcribing}, tts_playing={_tts_playing}")

def set_welcome_played():
    """Signal that welcome prompt has been played - makes aura eye static and shows buttons"""
    global _welcome_played, _window
    _welcome_played = True
    print("[AuraGUI] 👋 Welcome prompt played - aura eye now static and ready")
    
    # Show buttons using thread-safe Qt mechanism
    if _window:
        QMetaObject.invokeMethod(_window, "_show_buttons",
                                Qt.QueuedConnection)

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

def set_wake_word_detected(active):
    """Set wake word detected state - solid red edge when wake word detected (thread-safe)"""
    global _window
    if _window:
        # Use Qt's thread-safe mechanism to update GUI from any thread
        QMetaObject.invokeMethod(_window, "_update_wake_word_detected_state",
                                Qt.QueuedConnection,
                                Q_ARG(bool, active))
    else:
        print("[AuraGUI] ⚠️ Window not initialized, cannot update wake word detected state")

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

def set_microphone_muted(muted):
    """Set microphone muted state - dims aura eye when muted"""
    global _microphone_muted
    _microphone_muted = muted
    if muted:
        print("[AuraGUI] 🔇 Microphone MUTED - aura eye dimmed")
    else:
        print("[AuraGUI] 🔊 Microphone ACTIVE - aura eye normal")

def set_setup_complete():
    """Mark initial setup as complete"""
    global _setup_complete
    _setup_complete = True
    print("[AuraGUI] ✅ Setup complete - switching to idle mode")
