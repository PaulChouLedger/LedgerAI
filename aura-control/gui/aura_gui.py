# aura_gui.py — AuraVision GUI

import os
import sys
import math
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsDropShadowEffect, QTextEdit, QScrollBar
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QTransform, QPainter, QPen, QFont, QFontMetrics
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
_debug_overlay_enabled = True  # Debug overlay enabled by default
_transcription_overlay_enabled = True  # Transcription overlay enabled by default

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

class DebugOverlayWidget(QWidget):
    """Debug text overlay that displays initialization messages during setup"""
    def __init__(self, parent, window_size):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.parent_gui = parent
        self.window_size = window_size
        self.debug_log_path = os.path.expanduser("~/LedgerAI/data/aura_init_debug.log")
        self.last_file_position = 0
        self.max_lines = 12  # Show last 12 lines of debug messages
        
        # Create debug text widget
        self.debug_text = QTextEdit(self)
        self.debug_text.setReadOnly(True)
        self.debug_text.setFrameShape(QTextEdit.NoFrame)
        
        # Styling for console-like appearance
        self.debug_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.75);
                color: #00ff00;
                border: none;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        
        # Position at bottom of screen (below Aura eye) - CENTERED
        overlay_height = 200
        overlay_y = window_size - overlay_height - 40  # 40px from bottom
        overlay_width = window_size - 100  # 50px margins on each side
        overlay_x = (window_size - overlay_width) // 2  # Centered horizontally
        
        self.setGeometry(overlay_x, overlay_y, overlay_width, overlay_height)
        self.debug_text.setGeometry(0, 0, overlay_width, overlay_height)
        
        # Initially hidden - will be shown during initialization
        self.hide()
        
        # Timer to poll debug log file
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._update_debug_messages)
        self.poll_timer.start(500)  # Poll every 500ms
        
        # Initialize debug log file if it doesn't exist
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
    
    def _update_debug_messages(self):
        """Read new messages from debug log file and display them"""
        global _setup_complete, _debug_overlay_enabled
        
        # Only read file if we're visible and enabled (during initialization)
        if not self.isVisible() or _setup_complete or not _debug_overlay_enabled:
            return
        
        try:
            if not os.path.exists(self.debug_log_path):
                return
            
            # Read new content from file
            with open(self.debug_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_file_position)
                new_content = f.read()
                self.last_file_position = f.tell()
            
            if new_content:
                # Add new messages
                lines = new_content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        # Strip emoji and format for display
                        clean_line = self._clean_debug_line(line)
                        self.debug_text.append(clean_line)
                        
                        # Limit to max_lines
                        document = self.debug_text.document()
                        block_count = document.blockCount()
                        if block_count > self.max_lines:
                            cursor = self.debug_text.textCursor()
                            cursor.movePosition(cursor.Start)
                            cursor.movePosition(cursor.Down, cursor.MoveAnchor, block_count - self.max_lines)
                            cursor.movePosition(cursor.StartOfBlock)
                            cursor.movePosition(cursor.End, cursor.KeepAnchor)
                            cursor.removeSelectedText()
                            cursor.deletePreviousChar()  # Remove extra newline
                        
                        # Auto-scroll to bottom
                        scrollbar = self.debug_text.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())
        
        except Exception as e:
            # Silently handle errors - don't break GUI
            pass
    
    def _clean_debug_line(self, line):
        """Clean debug line for display - strip excessive formatting but keep readability"""
        # Remove common timestamp prefixes like [2025-12-05 18:20:01]
        import re
        line = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', line)
        
        # Keep emojis but ensure line is readable
        return line.strip()
    
    def reset_position(self):
        """Reset file position (when starting new initialization)"""
        self.last_file_position = 0
        self.debug_text.clear()

class CircularProgressWidget(QWidget):
    """Circular progress indicator around the aura eye showing initialization progress"""
    def __init__(self, parent, window_size):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.parent_gui = parent
        self.window_size = window_size
        self.progress = 0.0  # 0.0 to 1.0
        self.start_time = None
        self.estimated_duration = 30.0  # Estimated 30 seconds for initialization
        
        # Set transparent background
        self.setStyleSheet("background: transparent;")
        
        # Position to cover entire screen (will draw circle around aura eye)
        self.setGeometry(0, 0, window_size, window_size)
        
        # Initially hidden - will be shown during initialization
        self.hide()
        
        # Timer to update progress
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_progress)
        self.update_timer.start(100)  # Update every 100ms for smooth animation
        
        print(f"[CircularProgress] 🔧 Initializing circular progress widget")
    
    def start(self):
        """Start the progress indicator"""
        self.start_time = time.time()
        self.progress = 0.0
        self.show()
        self.update_timer.start()
        self.raise_()  # Ensure it's above aura eye but below border
        print(f"[CircularProgress] ▶️ Started progress indicator")
    
    def _update_progress(self):
        """Update progress based on actual initialization milestones - no time-based jumping"""
        global _setup_complete, _welcome_played
        
        # Hide when setup is complete (before buttons show)
        if _setup_complete or _welcome_played:
            # Complete the progress and hide immediately
            self.progress = 1.0
            self.update()
            self.hide()
            self.update_timer.stop()
            return
        
        if self.start_time is None:
            return
        
        # Get progress ONLY from milestones - no time-based jumping
        milestone_progress = self._get_milestone_progress()
        
        # Only use milestone progress - this ensures it grows as modules actually load
        # Don't let it jump ahead based on time
        if milestone_progress > self.progress:
            # Only allow progress to increase, never decrease
            # Smooth the increase to avoid sudden jumps
            self.progress = min(1.0, self.progress + (milestone_progress - self.progress) * 0.3)
        elif milestone_progress == 0 and self.progress == 0:
            # Very slow initial progress if no milestones detected yet
            elapsed = time.time() - self.start_time
            if elapsed > 2.0:  # Only start showing progress after 2 seconds
                self.progress = min(0.05, (elapsed - 2.0) * 0.01)  # Very slow initial crawl
        
        self.update()
    
    def _get_milestone_progress(self):
        """Estimate progress based on initialization milestones in debug log - sequential, conservative"""
        try:
            debug_log_path = os.path.expanduser("~/LedgerAI/data/aura_init_debug.log")
            if not os.path.exists(debug_log_path):
                return 0.0
            
            with open(debug_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Sequential milestones - must be detected in order, very conservative progress values
            # Each milestone only adds a small amount of progress
            milestones = [
                # Very early setup (low progress)
                ("Loading config", 0.03),
                ("Starting services", 0.06),
                
                # Container health checks (containers are starting)
                ("Waiting for.*whisper", 0.10),  # Whisper container starting
                ("Waiting for.*llm", 0.15),      # LLM container starting
                ("health", 0.18),                # Health checks passing
                
                # Containers actually responding
                ("whisper.*respond", 0.22),      # Whisper responding
                ("llm.*respond", 0.28),          # LLM responding
                
                # Model loading (this takes time)
                ("Model loaded", 0.35),
                ("simple_loaded.*true", 0.40),   # Model actually loaded
                
                # Warm-ups (these happen after containers are up)
                ("Testing LLM", 0.48),
                ("LLM warm-up", 0.52),
                ("warm-up complete", 0.56),
                
                # RAG initialization (after LLM is ready)
                ("RAG initialization", 0.62),
                ("RAG container initialized", 0.68),
                ("RAG.*ready", 0.72),
                
                # Listener/audio (near the end)
                ("Listener.*initializing", 0.78),
                ("listener ready", 0.84),
                ("listener is now READY", 0.90),
                
                # Final completion
                ("Setup complete", 0.95),
            ]
            
            # Check milestones - use regex for pattern matching
            import re
            detected_milestones = []
            for milestone_text, progress_value in milestones:
                if re.search(milestone_text.lower(), content.lower(), re.IGNORECASE):
                    detected_milestones.append((milestone_text, progress_value))
            
            if len(detected_milestones) == 0:
                return 0.0
            
            # Get the highest milestone
            max_progress = max(m[1] for m in detected_milestones)
            
            # Safety checks to prevent false positives:
            # 1. If we see late milestones (>50%) but no container milestones, cap progress
            early_container_milestones = [m for m in detected_milestones if "whisper" in m[0].lower() or "llm" in m[0].lower() or "health" in m[0].lower()]
            late_milestones = [m for m in detected_milestones if m[1] >= 0.5]
            
            if len(late_milestones) > 0 and len(early_container_milestones) == 0:
                # Late milestones detected but no containers - likely false positive
                max_progress = min(max_progress, 0.20)  # Cap at 20%
            
            # 2. Exclude TTS initialization from triggering high progress
            # TTS happens early but shouldn't indicate containers are loaded
            if "TTS" in content and "initialization" in content.lower():
                # If we see TTS but no container milestones, cap progress
                if len(early_container_milestones) == 0:
                    max_progress = min(max_progress, 0.15)  # Cap at 15% if only TTS
            
            # 3. Require at least one container milestone before allowing progress > 30%
            if max_progress > 0.30 and len(early_container_milestones) == 0:
                max_progress = min(max_progress, 0.25)  # Cap at 25% without containers
            
            return max_progress
        except Exception:
            return 0.0
    
    def set_progress(self, value):
        """Set progress manually (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, value))
        self.update()
    
    def paintEvent(self, event):
        """Draw circular progress ring around the aura eye - hugs the eye, matches white perimeter"""
        if not self.isVisible() or self.progress <= 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Center of screen (same as aura eye)
        center_x, center_y = 540, 540
        
        # Progress ring parameters - hug the aura eye closely, 20% smaller diameter
        # Original: 350px radius, new: 280px radius (20% reduction)
        # Match white perimeter: 8px thickness, white with 30% opacity (77/255)
        ring_radius = 280  # 20% smaller - closer to aura eye
        ring_thickness = 8  # Match white perimeter thickness
        
        # White color matching the perimeter (30% opacity)
        white_color = QColor(255, 255, 255, 77)  # Same as white perimeter
        
        # Draw progress ring (filled portion) - grows from top clockwise
        if self.progress > 0:
            # Calculate angle for progress (start at top, go clockwise)
            progress_angle = int(self.progress * 360 * 16)  # Convert to 1/16th degree units
            
            # Draw progress arc with white color matching perimeter
            progress_pen = QPen(white_color, ring_thickness, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(progress_pen)
            painter.setBrush(Qt.NoBrush)
            
            # Draw arc (Qt: 0° = 3 o'clock, we want 0° = 12 o'clock, so start at -90°)
            start_angle = -90 * 16  # Start at top (12 o'clock)
            painter.drawArc(
                center_x - ring_radius,
                center_y - ring_radius,
                ring_radius * 2,
                ring_radius * 2,
                start_angle,
                progress_angle
            )
        
        painter.end()

class TranscriptionOverlayWidget(QWidget):
    """Transcription text overlay that displays real-time transcription"""
    def __init__(self, parent, window_size):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Ensure widget is transparent and can be painted
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.parent_gui = parent
        self.window_size = window_size
        self.current_text = ""
        self.max_lines = 3  # Show last 3 lines of transcription
        
        # Set transparent background for the widget itself
        self.setStyleSheet("background: transparent;")
        
        print(f"[TranscriptionOverlay] 🔧 Initializing transcription overlay widget")
        
        # Create transcription text widget
        self.transcription_text = QTextEdit(self)
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setFrameShape(QTextEdit.NoFrame)
        
        # Styling for transcription display - fully transparent background with clean border
        # No border dots - using solid border with proper rendering
        self.transcription_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.0);
                color: #ffffff;
                border: none;
                border-radius: 15px;
                padding: 12px 16px;
                font-family: 'Arial', sans-serif;
                font-size: 15pt;
                font-weight: 500;
            }
        """)
        
        # Add a custom paint method to draw a clean border without dots
        # We'll override paintEvent to draw a smooth border
        print(f"[TranscriptionOverlay] 🎨 Styling applied: transparent background, subtle border")
        
        # Strategic positioning to avoid aura eye, buttons, and stay within circular perimeter
        # Layout analysis:
        # - Screen: 1080x1080 circular, center at (540, 540), radius 535px
        # - Aura eye: centered, roughly 300-400px radius
        # - Buttons: at radius 430px from center, size 140x140, at angles 0°, 60°, 120°, 180°, 240°, 300°
        # 
        # Calculate exact button positions:
        # 0° (top): x=540, y=110, extends y:40-180
        # 60° (top-right): x≈912, y≈325, extends x:842-982, y:255-395
        # 120° (bottom-right/analytics): x≈325, y≈912, extends x:255-395, y:842-982
        # 180° (bottom): x=540, y=970, extends y:900-1040
        # 240° (bottom-left): x≈325, y≈168, extends x:255-395, y:98-238
        # 300° (top-left): x≈168, y≈325, extends x:98-238, y:255-395
        #
        # Safe zones within circular perimeter:
        # - Top area: y < 200, x centered (avoids top button, clear of aura)
        # - Left edge: x < 150, y centered (avoids top-left and bottom-left buttons)
        # - Right edge: x > 930, y centered (avoids top-right and bottom-right buttons)
        
        overlay_height = 140  # Slightly shorter to fit better
        overlay_width = 380   # Slightly wider for better readability
        
        # Position lower, closer to center (but still above aura eye)
        # Safe zone: Below top button (ends at y=180), above aura eye center (starts ~y=240)
        overlay_x = (window_size - overlay_width) // 2  # Centered horizontally
        overlay_y = 240  # Position adjusted
        
        # Verify position is within circular perimeter
        # Distance from center (540, 540) to overlay corners:
        center_x, center_y = 540, 540
        perimeter_radius = 535
        
        # Check if overlay fits (using farthest corner)
        farthest_x = max(abs(overlay_x - center_x), abs(overlay_x + overlay_width - center_x))
        farthest_y = max(abs(overlay_y - center_y), abs(overlay_y + overlay_height - center_y))
        farthest_dist = math.sqrt(farthest_x**2 + farthest_y**2)
        
        if farthest_dist > perimeter_radius - 20:  # 20px safety margin
            # Adjust if too close to edge - move slightly up and narrower
            overlay_width = 360
            overlay_x = (window_size - overlay_width) // 2
            overlay_y = 240  # Position adjusted
        
        self.setGeometry(overlay_x, overlay_y, overlay_width, overlay_height)
        self.transcription_text.setGeometry(0, 0, overlay_width, overlay_height)
        
        print(f"[TranscriptionOverlay] 📐 Created at ({overlay_x}, {overlay_y}), size {overlay_width}x{overlay_height}")
        print(f"[TranscriptionOverlay] 📐 Text widget geometry: {self.transcription_text.geometry()}")
        
        # Initially hidden - will be shown when transcription is active
        self.hide()
        print(f"[TranscriptionOverlay] 👁️ Initial state: visible={self.isVisible()}")
    
    def update_transcription(self, text):
        """Update transcription text"""
        if not text or not text.strip():
            print(f"[TranscriptionOverlay] ⚠️ update_transcription called with empty text")
            return
        
        self.current_text = text.strip()
        print(f"[TranscriptionOverlay] 📝 Updating transcription: '{self.current_text[:50]}...'")
        
        # Ensure widget and text widget are visible
        if not self.isVisible():
            self.show()
        if not self.transcription_text.isVisible():
            self.transcription_text.show()
        
        # Display transcription text
        self.transcription_text.clear()
        self.transcription_text.append(self.current_text)
        
        # Force update to ensure text is visible
        self.transcription_text.update()
        self.transcription_text.repaint()
        self.update()
        self.repaint()
        
        # Auto-scroll to bottom
        scrollbar = self.transcription_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        print(f"[TranscriptionOverlay] ✅ Transcription text updated in widget")
        print(f"[TranscriptionOverlay] 📐 Widget state: visible={self.isVisible()}, text_widget_visible={self.transcription_text.isVisible()}, geometry={self.geometry()}")
        print(f"[TranscriptionOverlay] 📐 Text widget has content: {len(self.transcription_text.toPlainText())} chars")
    
    def clear_transcription(self):
        """Clear transcription text"""
        print(f"[TranscriptionOverlay] 🧹 Clearing transcription text")
        self.current_text = ""
        self.transcription_text.clear()
    
    def paintEvent(self, event):
        """Custom paint event - completely transparent, no border"""
        super().paintEvent(event)
        # No border drawing - completely transparent overlay

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
        
        # Create debug overlay widget for initialization messages
        self.debug_overlay = DebugOverlayWidget(self, window_size)
        print(f"[DebugOverlay] Created debug overlay widget")
        
        # Create transcription overlay widget for real-time transcription
        self.transcription_overlay = TranscriptionOverlayWidget(self, window_size)
        print(f"[TranscriptionOverlay] Created transcription overlay widget")
        
        # Create circular progress indicator for initialization
        self.circular_progress = CircularProgressWidget(self, window_size)
        print(f"[CircularProgress] Created circular progress widget")
        
        # Ensure proper z-order: border overlay always on top
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()  # Border always on top

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
        
        # Load overlay settings from app_settings.json
        self._load_overlay_settings()
        
        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _load_overlay_settings(self):
        """Load overlay settings from app_settings.json"""
        global _debug_overlay_enabled, _transcription_overlay_enabled
        try:
            import json
            settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings_data = json.load(f) or {}
                    _debug_overlay_enabled = settings_data.get("debug_overlay_enabled", True)
                    _transcription_overlay_enabled = settings_data.get("transcription_overlay_enabled", True)
            print(f"[AuraGUI] 📋 Overlay settings loaded: debug={_debug_overlay_enabled}, transcription={_transcription_overlay_enabled}")
        except Exception as e:
            print(f"[AuraGUI] ⚠️ Failed to load overlay settings: {e}")
            # Use defaults
            _debug_overlay_enabled = True
            _transcription_overlay_enabled = True
    
    def _load_overlay_settings(self):
        """Load overlay settings from app_settings.json"""
        global _debug_overlay_enabled, _transcription_overlay_enabled
        try:
            import json
            settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings_data = json.load(f) or {}
                    _debug_overlay_enabled = settings_data.get("debug_overlay_enabled", True)
                    _transcription_overlay_enabled = settings_data.get("transcription_overlay_enabled", True)
            print(f"[AuraGUI] 📋 Overlay settings loaded: debug={_debug_overlay_enabled}, transcription={_transcription_overlay_enabled}")
        except Exception as e:
            print(f"[AuraGUI] ⚠️ Failed to load overlay settings: {e}")
            # Use defaults
            _debug_overlay_enabled = True
            _transcription_overlay_enabled = True
    
    def create_circular_buttons(self):
        """Create 6 buttons equally spaced around the circular edge - iPhone-style with bigger, multicolored icons"""
        # Button configurations: (text, icon, function, color, hover_color, pressed_color)
        # Using vibrant iPhone-inspired colors
        button_configs = [
            ("↑", "Upload", self._handle_upload, "#007AFF", "#0051D5", "#003D9E"),      # iOS Blue
            ("⚙", "Settings", self._handle_settings, "#8E8E93", "#6E6E73", "#4E4E53"),  # iOS Gray
            ("📊", "Analytics", self._handle_analytics, "#FF3B30", "#D32F2F", "#B71C1C"), # iOS Red (swapped from Green)
            ("🎤", "Voice", self._handle_voice, "#34C759", "#28A745", "#1E7E34"),        # iOS Green (swapped from Red)
            ("📱", "Mobile", self._handle_mobile, "#AF52DE", "#9C27B0", "#7B1FA2"),     # iOS Purple
            ("ℹ", "Info", self._handle_info, "#FF9500", "#F57C00", "#E65100")          # iOS Orange
        ]
        
        # Calculate positions for 6 buttons around a circle
        # Move buttons further inward to avoid interfering with red border
        # Edge radius is 540px, button radius is now 70px (for 140px button), border is 8-12px
        # Adjusted radius: 540 - 20 - 70 - 20 = 430px
        radius = 430  # Distance from center to button (adjusted for larger buttons)
        center_x = 540  # Center of 1080x1080 screen
        center_y = 540
        
        self.buttons = []
        button_size = 140  # Bigger iPhone-style buttons (was 100)
        
        for i, (text, tooltip, handler, color, hover_color, pressed_color) in enumerate(button_configs):
            # Calculate angle for this button (0° to 300° in 60° increments)
            angle = math.radians(i * 60)  # 0, 60, 120, 180, 240, 300 degrees
            
            # Calculate position (centered on button)
            x = center_x + radius * math.cos(angle) - (button_size // 2)
            y = center_y + radius * math.sin(angle) - (button_size // 2)
            
            # Create button
            btn = QPushButton(text)
            btn.setFixedSize(button_size, button_size)
            btn.setToolTip(tooltip)
            btn.move(int(x), int(y))
            
            # iPhone-style styling - clean, high-quality iOS graphics
            btn.setStyleSheet(f"""
                QPushButton {{
                    /* Clean iOS-style solid color with subtle highlight */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                        fx:0.45, fy:0.45,
                        stop:0 rgba(255, 255, 255, 0.2),
                        stop:0.5 {color},
                        stop:1 {color});
                    color: #FFFFFF;
                    font-size: 56px;
                    font-weight: bold;
                    border-radius: {button_size // 2}px;
                    border: none;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    /* Brighter on hover */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                        fx:0.45, fy:0.45,
                        stop:0 rgba(255, 255, 255, 0.3),
                        stop:0.5 {hover_color},
                        stop:1 {hover_color});
                    border: none;
                }}
                QPushButton:pressed {{
                    /* Darker when pressed */
                    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                        fx:0.45, fy:0.45,
                        stop:0 rgba(255, 255, 255, 0.1),
                        stop:0.5 {pressed_color},
                        stop:1 {pressed_color});
                    border: none;
                }}
            """)
            
            # Clean iOS-style shadow effect
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(20)  # Moderate blur
            shadow_effect.setColor(QColor(0, 0, 0, 100))  # Subtle shadow
            shadow_effect.setOffset(0, 3)  # Subtle offset
            btn.setGraphicsEffect(shadow_effect)
            
            # Enable high-quality rendering attributes
            btn.setAttribute(Qt.WA_TranslucentBackground, False)  # Solid background for better rendering
            
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
                
                # Update button to RED for muted state (muted = red)
                if voice_btn:
                    voice_btn.setStyleSheet(f"""
                        QPushButton {{
                            /* Vibrant red for muted state */
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.25),
                                stop:0.3 #FF3B30,
                                stop:1 #FF3B30);
                            color: #FFFFFF;
                            font-size: 56px;
                            font-weight: bold;
                            border-radius: 70px;
                            border: none;
                            padding: 0px;
                        }}
                        QPushButton:hover {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.35),
                                stop:0.3 #D32F2F,
                                stop:1 #D32F2F);
                            border: none;
                        }}
                        QPushButton:pressed {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.15),
                                stop:0.3 #B71C1C,
                                stop:1 #B71C1C);
                            border: none;
                        }}
                    """)
            else:
                print("[AuraGUI] ✅ Microphone ACTIVE - transcription enabled")
                # Update global state
                set_microphone_muted(False)
                
                # Update button to GREEN for active state (active = green, more visible)
                if voice_btn:
                    voice_btn.setStyleSheet(f"""
                        QPushButton {{
                            /* Vibrant green for active voice state - high quality, highly visible */
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.25),
                                stop:0.3 #34C759,
                                stop:1 #34C759);
                            color: #FFFFFF;
                            font-size: 56px;
                            font-weight: bold;
                            border-radius: 70px;
                            border: none;
                            padding: 0px;
                        }}
                        QPushButton:hover {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.35),
                                stop:0.3 #28A745,
                                stop:1 #28A745);
                            border: none;
                        }}
                        QPushButton:pressed {{
                            background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                                fx:0.45, fy:0.45,
                                stop:0 rgba(255, 255, 255, 0.15),
                                stop:0.3 #1E7E34,
                                stop:1 #1E7E34);
                            border: none;
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
        global _gui_ready, _setup_complete
        _gui_ready = True
        print("[AuraGUI] 🎯 GUI has fully rendered")
        
        # Ensure proper z-order: border overlay always on top, then progress, then other elements
        if hasattr(self, 'circular_progress') and not _setup_complete:
            # Start progress indicator if initialization is in progress
            if not self.circular_progress.isVisible():
                self.circular_progress.start()
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()  # Border always on top
            self.border_overlay.show()
        if hasattr(self, 'circular_progress'):
            # Progress should be above aura eye but below border
            self.circular_progress.raise_()

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
            # Show circular progress indicator during initialization
            if hasattr(self, 'circular_progress'):
                if not self.circular_progress.isVisible():
                    self.circular_progress.start()  # Start progress indicator
                self.circular_progress.raise_()  # Ensure it's visible (but below border)
            # Show debug overlay during initialization (if enabled)
            global _debug_overlay_enabled
            if hasattr(self, 'debug_overlay') and _debug_overlay_enabled:
                if not self.debug_overlay.isVisible():
                    self.debug_overlay.reset_position()  # Reset to start reading from beginning
                    self.debug_overlay.show()
                self.debug_overlay.raise_()  # Ensure it's visible (but below border)
            elif hasattr(self, 'debug_overlay') and not _debug_overlay_enabled:
                if self.debug_overlay.isVisible():
                    self.debug_overlay.hide()
        else:
            # Hide debug overlay and progress indicator once initialization is complete
            if hasattr(self, 'debug_overlay'):
                if self.debug_overlay.isVisible():
                    self.debug_overlay.hide()
            if hasattr(self, 'circular_progress'):
                if self.circular_progress.isVisible():
                    self.circular_progress.hide()
    
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
    
    @pyqtSlot(str)
    def _update_transcription_text(self, text):
        """Update transcription text in overlay (called from listener thread)"""
        global _transcription_overlay_enabled
        print(f"[TranscriptionOverlay] 🔔 _update_transcription_text called: text='{text}', enabled={_transcription_overlay_enabled}")
        
        if hasattr(self, 'transcription_overlay'):
            print(f"[TranscriptionOverlay] ✅ Overlay widget exists: visible={self.transcription_overlay.isVisible()}, geometry={self.transcription_overlay.geometry()}")
            
            if _transcription_overlay_enabled and text and text.strip():
                print(f"[TranscriptionOverlay] 📝 Updating transcription (enabled=True, has_text=True)")
                self.transcription_overlay.update_transcription(text)
                
                # Ensure parent window is visible
                if not self.isVisible():
                    print(f"[TranscriptionOverlay] ⚠️ Parent window not visible, showing it")
                    self.show()
                
                # Show and raise the overlay widget
                if not self.transcription_overlay.isVisible():
                    print(f"[TranscriptionOverlay] 👁️ Showing overlay (was hidden)")
                    self.transcription_overlay.show()
                    self.transcription_overlay.setAttribute(Qt.WA_ShowWithoutActivating, True)
                
                # Ensure text widget is visible
                if hasattr(self.transcription_overlay, 'transcription_text'):
                    if not self.transcription_overlay.transcription_text.isVisible():
                        print(f"[TranscriptionOverlay] 👁️ Showing text widget (was hidden)")
                        self.transcription_overlay.transcription_text.show()
                else:
                    print(f"[TranscriptionOverlay] 👁️ Overlay already visible")
                
                # Raise transcription overlay, then ensure border stays on top
                print(f"[TranscriptionOverlay] ⬆️ Raising transcription overlay")
                self.transcription_overlay.raise_()
                if hasattr(self.transcription_overlay, 'transcription_text'):
                    self.transcription_overlay.transcription_text.raise_()
                
                # Force update and repaint to ensure visibility
                self.transcription_overlay.update()
                self.transcription_overlay.repaint()
                if hasattr(self.transcription_overlay, 'transcription_text'):
                    self.transcription_overlay.transcription_text.update()
                    self.transcription_overlay.transcription_text.repaint()
                
                if hasattr(self, 'border_overlay'):
                    print(f"[TranscriptionOverlay] ⬆️ Raising border overlay (keep on top)")
                    self.border_overlay.raise_()  # Border always on top
                
                # Force main window update
                self.update()
                self.repaint()
                QApplication.processEvents()  # Force Qt to process events
                
                print(f"[TranscriptionOverlay] ✅ Final state: visible={self.transcription_overlay.isVisible()}, geometry={self.transcription_overlay.geometry()}")
                if hasattr(self.transcription_overlay, 'transcription_text'):
                    print(f"[TranscriptionOverlay] ✅ Text widget visible={self.transcription_overlay.transcription_text.isVisible()}, text_length={len(self.transcription_overlay.transcription_text.toPlainText())}")
            else:
                # Clear and hide if disabled or text is empty
                reason = "disabled" if not _transcription_overlay_enabled else "empty text"
                print(f"[TranscriptionOverlay] 🚫 Clearing/hiding overlay: {reason}")
                if hasattr(self.transcription_overlay, 'current_text'):
                    self.transcription_overlay.current_text = ""
                self.transcription_overlay.clear_transcription()
                if self.transcription_overlay.isVisible():
                    print(f"[TranscriptionOverlay] 👁️ Hiding overlay")
                    self.transcription_overlay.hide()
        else:
            print(f"[TranscriptionOverlay] ❌ Overlay widget does not exist!")
    
    def _clear_transcription_text(self):
        """Clear transcription text in overlay"""
        if hasattr(self, 'transcription_overlay'):
            self.transcription_overlay.clear_transcription()
            if self.transcription_overlay.isVisible():
                self.transcription_overlay.hide()
    
    @pyqtSlot()
    def _update_debug_overlay_visibility(self):
        """Update debug overlay visibility based on enabled flag (called from settings)"""
        global _debug_overlay_enabled, _setup_complete
        if hasattr(self, 'debug_overlay'):
            if _debug_overlay_enabled and not _setup_complete:
                # Show if enabled and during initialization
                if not self.debug_overlay.isVisible():
                    self.debug_overlay.reset_position()
                    self.debug_overlay.show()
                self.debug_overlay.raise_()
            else:
                # Hide if disabled or initialization complete
                if self.debug_overlay.isVisible():
                    self.debug_overlay.hide()
    
    @pyqtSlot()
    def _update_transcription_overlay_visibility(self):
        """Update transcription overlay visibility based on enabled flag (called from settings)"""
        global _transcription_overlay_enabled
        if hasattr(self, 'transcription_overlay'):
            if _transcription_overlay_enabled:
                # Show if enabled and there's current transcription text
                if hasattr(self.transcription_overlay, 'current_text') and self.transcription_overlay.current_text:
                    if not self.transcription_overlay.isVisible():
                        self.transcription_overlay.show()
                    # Raise transcription overlay, then ensure border stays on top
                    self.transcription_overlay.raise_()
                    if hasattr(self, 'border_overlay'):
                        self.border_overlay.raise_()  # Border always on top
                else:
                    # Hide if no transcription text
                    if self.transcription_overlay.isVisible():
                        self.transcription_overlay.hide()
            else:
                # Hide immediately if disabled
                if self.transcription_overlay.isVisible():
                    self.transcription_overlay.clear_transcription()
                    self.transcription_overlay.hide()
                # Also clear current text
                if hasattr(self.transcription_overlay, 'current_text'):
                    self.transcription_overlay.current_text = ""
    
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
    
    # Hide circular progress indicator before showing buttons
    if _window and hasattr(_window, 'circular_progress'):
        if _window.circular_progress.isVisible():
            _window.circular_progress.hide()
            _window.circular_progress.update_timer.stop()
    
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

def update_transcription_text(text):
    """Update transcription text in GUI overlay (thread-safe)"""
    global _window
    print(f"[AuraGUI] 📝 update_transcription_text called: text='{text}', window={_window is not None}")
    
    if _window:
        # Use Qt's thread-safe mechanism to update GUI from any thread
        print(f"[AuraGUI] 📝 Invoking _update_transcription_text via QMetaObject")
        success = QMetaObject.invokeMethod(_window, "_update_transcription_text",
                                          Qt.QueuedConnection,
                                          Q_ARG(str, text))
        print(f"[AuraGUI] 📝 QMetaObject.invokeMethod result: {success}")
        if not success:
            print(f"[AuraGUI] ⚠️ QMetaObject.invokeMethod returned False, using QTimer fallback")
            # Fallback: use QTimer to call method directly in main thread
            QTimer.singleShot(0, lambda: _window._update_transcription_text(text))
    else:
        print("[AuraGUI] ⚠️ Window not initialized, cannot update transcription text")

def clear_transcription_text():
    """Clear transcription text in GUI overlay (thread-safe)"""
    global _window
    print(f"[AuraGUI] 🧹 clear_transcription_text called, window={_window is not None}")
    
    if _window:
        # Use Qt's thread-safe mechanism to update GUI from any thread
        QMetaObject.invokeMethod(_window, "_clear_transcription_text",
                                Qt.QueuedConnection)
    else:
        print("[AuraGUI] ⚠️ Window not initialized, cannot clear transcription text")

def set_setup_complete():
    """Mark initial setup as complete"""
    global _setup_complete, _window
    _setup_complete = True
    
    # Complete and hide circular progress indicator
    if _window and hasattr(_window, 'circular_progress'):
        _window.circular_progress.set_progress(1.0)  # Complete the progress
        QTimer.singleShot(500, lambda: _window.circular_progress.hide() if hasattr(_window, 'circular_progress') else None)
    
    # Close debug log file when initialization completes
    try:
        from core.main import end_initialization_phase
        end_initialization_phase()  # Stop debug logging
    except Exception:
        pass  # Silently fail if import fails
    
    print("[AuraGUI] ✅ Setup complete - switching to idle mode")
