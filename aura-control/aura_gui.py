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

class AuraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AuraVision")
        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
                border: 5px solid #ff0000;
                border-radius: 540px;  /* Half of 1080 for perfect circle */
            }
        """)
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
        
        # Create 6 buttons equally spaced around the circular edge
        self.create_circular_buttons()
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # === Pulsation Effect ===
        self.opacity_effect = QGraphicsOpacityEffect()
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity = 1.0
        self.pulse_direction = -1

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_pulse)
        self.timer.start(100)
        
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
        radius = 400  # Distance from center to button
        center_x = 540  # Center of 1080x1080 screen
        center_y = 540
        
        self.buttons = []
        
        for i, (text, tooltip, handler, color) in enumerate(button_configs):
            # Calculate angle for this button (0° to 300° in 60° increments)
            angle = math.radians(i * 60)  # 0, 60, 120, 180, 240, 300 degrees
            
            # Calculate position
            x = center_x + radius * math.cos(angle) - 30  # -30 to center the 60px button
            y = center_y + radius * math.sin(angle) - 30
            
            # Create button
            btn = QPushButton(text)
            btn.setFixedSize(60, 60)
            btn.setToolTip(tooltip)
            btn.move(int(x), int(y))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    border-radius: 30px;
                    border: 2px solid #ffffff;
                }}
                QPushButton:hover {{
                    background-color: {color}CC;
                    border: 3px solid #ffffff;
                }}
                QPushButton:pressed {{
                    background-color: {color}99;
                }}
            """)
            
            # Connect handler
            btn.clicked.connect(handler)
            self.buttons.append(btn)
            
            # Add button to the main widget
            self.centralWidget().layout().addWidget(btn)
    
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

    def animate_pulse(self):
        global _listening_ready
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
        else:
            self.opacity_effect.setOpacity(1.0)
    
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
