# custom_keyboard.py — On-Screen Keyboard for Circular Touch Display

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from gui.base_dialog import BaseAuraDialog

class CircularKeyboard(BaseAuraDialog):
    """Custom on-screen keyboard optimized for circular 1080x1080 touchscreen"""
    
    # Signal emitted when text is confirmed
    text_confirmed = pyqtSignal(str)
    
    # Signal emitted when microphone is requested
    voice_input_requested = pyqtSignal()
    
    def __init__(self, parent=None, initial_text=""):
        self.current_text = initial_text
        # Initialize base dialog with proper centering
        super().__init__(parent, title="Enter Wallet Address", size=(1080, 1080), modal=True)
        
        # Add additional styles for keyboard
        additional_styles = """
            QLabel {
                color: #ffffff;
            }
        """
        base_stylesheet = self.styleSheet()
        self.setStyleSheet(base_stylesheet + additional_styles)
        
    def _setup_ui(self):
        """Setup keyboard UI - compact to fit within white circular perimeter"""
        main_layout = QVBoxLayout()
        # Use base class default margins to ensure content fits within white perimeter
        margins = BaseAuraDialog.get_default_margins()
        main_layout.setContentsMargins(*margins)
        main_layout.setSpacing(BaseAuraDialog.get_default_spacing())
        
        # Add vertical centering stretch
        main_layout.addStretch(1)
        
        # Title - smaller for space
        title = QLabel("Enter Ethereum Address")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 3px;")
        main_layout.addWidget(title)
        
        # Display area showing current input - more compact
        self.display = QLabel(self.format_display_text())
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setWordWrap(True)
        self.display.setMaximumHeight(60)
        self.display.setMinimumHeight(60)
        self.display.setStyleSheet("""
            QLabel {
                background-color: rgba(44, 44, 46, 0.9);
                color: #00ff00;
                border-radius: 12px;
                padding: 12px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(self.display)
        
        # Character count - smaller
        self.char_count = QLabel(f"{len(self.current_text)}/42 characters")
        self.char_count.setAlignment(Qt.AlignCenter)
        self.char_count.setStyleSheet("color: #8e8e93; font-size: 9pt; margin: 3px;")
        main_layout.addWidget(self.char_count)
        
        # Keyboard grid - hexadecimal characters for Ethereum addresses
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout()
        keyboard_layout.setSpacing(4)  # Reduced spacing between buttons
        
        # Hex keyboard layout (0-9, a-f, x for 0x prefix)
        keys = [
            ['1', '2', '3', '4', '5'],
            ['6', '7', '8', '9', '0'],
            ['a', 'b', 'c', 'd', 'e'],
            ['f', 'x', '←', '0x', 'Clear']
        ]
        
        button_style = """
            QPushButton {
                background-color: rgba(142, 142, 147, 0.3);
                color: #ffffff;
                font-size: 13pt;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
                border: 2px solid rgba(142, 142, 147, 0.5);
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: rgba(142, 142, 147, 0.5);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.7);
            }
        """
        
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                btn = QPushButton(key)
                btn.setStyleSheet(button_style)
                
                if key == '←':
                    # Backspace button
                    btn.clicked.connect(self.backspace)
                    btn.setStyleSheet(button_style.replace("0.3", "0.4"))  # Slightly different
                elif key == '0x':
                    # Quick 0x prefix button
                    btn.clicked.connect(self.add_0x_prefix)
                elif key == 'Clear':
                    # Clear all button
                    btn.clicked.connect(self.clear_all)
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: rgba(255, 59, 48, 0.7);
                            color: #ffffff;
                            font-size: 11pt;
                            font-weight: bold;
                            padding: 8px;
                            border-radius: 8px;
                            border: 2px solid rgba(255, 59, 48, 0.9);
                            min-height: 40px;
                            max-height: 40px;
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 59, 48, 0.85);
                        }
                        QPushButton:pressed {
                            background-color: rgba(255, 59, 48, 1.0);
                        }
                    """)
                else:
                    # Regular character button
                    btn.clicked.connect(lambda checked, k=key: self.add_character(k))
                
                keyboard_layout.addWidget(btn, row_idx, col_idx)
        
        keyboard_widget.setLayout(keyboard_layout)
        main_layout.addWidget(keyboard_widget)
        
        # Action buttons row - more compact
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        
        # Voice input button
        voice_btn = QPushButton("🎤 Voice")
        voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #4D94D9;
                color: #ffffff;
                font-size: 10pt;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 12px;
                border: none;
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: #5DA4E9;
            }
            QPushButton:pressed {
                background-color: #3D84C9;
            }
        """)
        voice_btn.clicked.connect(self.request_voice_input)
        action_layout.addWidget(voice_btn)
        
        # Paste button (for testing/development)
        paste_btn = QPushButton("📋 Paste")
        paste_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(142, 142, 147, 0.4);
                color: #ffffff;
                font-size: 10pt;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 12px;
                border: none;
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: rgba(142, 142, 147, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.8);
            }
        """)
        paste_btn.clicked.connect(self.paste_from_clipboard)
        action_layout.addWidget(paste_btn)
        
        main_layout.addLayout(action_layout)
        
        # Confirm/Cancel buttons - more compact
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 12px;
                border: none;
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("✅ Confirm")
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 12px;
                border: none;
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: #30B350;
            }
            QPushButton:pressed {
                background-color: #2A9D47;
            }
        """)
        confirm_btn.clicked.connect(self.confirm_text)
        button_layout.addWidget(confirm_btn)
        
        main_layout.addLayout(button_layout)
        
        # Bottom stretch for vertical centering
        main_layout.addStretch(1)
        
        self.setLayout(main_layout)
    
    def format_display_text(self):
        """Format display text with line breaks for readability"""
        if not self.current_text:
            return "0x..."
        
        # Show address with line break after 0x and every 20 chars
        if len(self.current_text) <= 22:
            return self.current_text
        else:
            # Split into multiple lines for long addresses
            return self.current_text[:22] + "\n" + self.current_text[22:]
    
    def add_character(self, char: str):
        """Add a character to the input"""
        # Ethereum addresses are 42 characters (0x + 40 hex digits)
        if len(self.current_text) < 42:
            self.current_text += char.lower()
            self.update_display()
    
    def backspace(self):
        """Remove last character"""
        if self.current_text:
            self.current_text = self.current_text[:-1]
            self.update_display()
    
    def clear_all(self):
        """Clear all input"""
        self.current_text = ""
        self.update_display()
    
    def add_0x_prefix(self):
        """Add 0x prefix if not present"""
        if not self.current_text.startswith("0x"):
            self.current_text = "0x" + self.current_text
            self.update_display()
    
    def paste_from_clipboard(self):
        """Paste from clipboard"""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        
        # Validate it looks like an Ethereum address
        if text.startswith("0x") and len(text) <= 42:
            self.current_text = text[:42]  # Limit to 42 chars
            self.update_display()
    
    def request_voice_input(self):
        """Request voice input from main system"""
        print("[Keyboard] 🎤 Voice input requested")
        self.voice_input_requested.emit()
    
    def set_text(self, text: str):
        """Set the current text (called after voice input)"""
        self.current_text = text[:42]  # Limit to 42 chars
        self.update_display()
    
    def update_display(self):
        """Update the display label"""
        self.display.setText(self.format_display_text())
        self.char_count.setText(f"{len(self.current_text)}/42 characters")
    
    def confirm_text(self):
        """Confirm and emit the entered text"""
        if self.current_text:
            self.text_confirmed.emit(self.current_text)
            self.accept()
        else:
            # Show error - no text entered
            self.display.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 59, 48, 0.3);
                    color: #ff0000;
                    border-radius: 15px;
                    padding: 15px;
                    font-family: 'Courier New', monospace;
                    font-size: 11pt;
                    font-weight: bold;
                }
            """)
            self.display.setText("❌ Please enter an address")
            
            # Reset style after 2 seconds
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.display.setStyleSheet("""
                QLabel {
                    background-color: rgba(44, 44, 46, 0.9);
                    color: #00ff00;
                    border-radius: 15px;
                    padding: 15px;
                    font-family: 'Courier New', monospace;
                    font-size: 11pt;
                    font-weight: bold;
                }
            """))
            QTimer.singleShot(2000, self.update_display)
    
        self.activateWindow()
    
    def get_text(self):
        """Get the current text (for use after exec_())"""
        return self.current_text

