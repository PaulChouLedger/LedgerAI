# aura_gui.py — AuraVision GUI

import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGraphicsOpacityEffect, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QTimer
from file_upload_dialog import show_upload_dialog

_app = None
_window = None
_gui_ready = False
_listening_ready = False  # Tracks when system is ready to transcribe

class AuraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AuraVision")
        self.setStyleSheet("background-color: black;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # === Load and Scale aura_eye.png ===
        img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/aura_eye.png"))
        print(f"[AuraGUI] ✅ Loaded image: {img_path}")
        pixmap = QPixmap(img_path)

        screen = self.screen()
        screen_size = screen.availableGeometry().size()
        min_dim = min(screen_size.width(), screen_size.height())
        scaled_pixmap = pixmap.scaled(min_dim, min_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # === Create main widget with image and button ===
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Image label
        self.label = QLabel()
        self.label.setPixmap(scaled_pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # Upload button
        self.upload_btn = QPushButton("↑")  # Circular upload button with up arrow
        self.upload_btn.setFixedSize(80, 80)  # Circular button
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 40px;  /* Perfect circle */
                border: none;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
            QPushButton:pressed {
                background-color: rgba(25, 25, 25, 200);
            }
        """)
        # Add debounce to prevent multiple rapid clicks
        self.upload_btn.clicked.connect(self._debounced_upload_click)
        layout.addWidget(self.upload_btn)
        
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
            show_upload_dialog()
        elif event.key() == Qt.Key_Escape:
            # Escape: Close GUI
            self.close()
        else:
            super().keyPressEvent(event)
    
    def _debounced_upload_click(self):
        """Debounced upload button click to prevent multiple rapid clicks"""
        # Show upload dialog immediately
        show_upload_dialog()
        
        # Disable button temporarily to prevent rapid clicking
        self.upload_btn.setEnabled(False)
        self.upload_btn.setText("⏳ Opening...")
        
        # Use QTimer to re-enable button after a short delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, self._reenable_upload_button)
    
    def _reenable_upload_button(self):
        """Re-enable upload button after debounce delay"""
        self.upload_btn.setEnabled(True)
        self.upload_btn.setText("↑")  # Circular upload button with up arrow

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
