# password_keyboard.py — On-Screen Password Keyboard for Circular Touch Display

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from gui.base_dialog import BaseAuraDialog

class PasswordKeyboard(BaseAuraDialog):
    """Custom on-screen keyboard for password entry with full alphanumeric support"""
    
    # Signal emitted when text is confirmed
    text_confirmed = pyqtSignal(str)
    
    def __init__(self, parent=None, initial_text="", title="Enter Password"):
        self.current_text = initial_text
        self.shift_active = False  # Track shift state
        self.letter_buttons = []  # Store letter buttons for shift toggle
        # Initialize base dialog with proper centering
        super().__init__(parent, title=title, size=(1080, 1080), modal=True)
        
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
        
        main_layout.addStretch(1)
        
        # Title - smaller
        title = QLabel(self.windowTitle())
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 5px;")
        main_layout.addWidget(title)
        
        # Display area showing masked password - more compact
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
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(self.display)
        
        # Character count - smaller
        self.char_count = QLabel(f"{len(self.current_text)} characters")
        self.char_count.setAlignment(Qt.AlignCenter)
        self.char_count.setStyleSheet("color: #8e8e93; font-size: 9pt; margin: 3px;")
        main_layout.addWidget(self.char_count)
        
        # Keyboard grid - full alphanumeric - more compact
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout()
        keyboard_layout.setSpacing(4)  # Reduced spacing between buttons
        
        # Full keyboard layout (numbers, letters, common symbols)
        # Row 1: Numbers
        keys_row1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        # Row 2: QWERTY top row
        keys_row2 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
        # Row 3: QWERTY middle row
        keys_row3 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
        # Row 4: QWERTY bottom row
        keys_row4 = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
        # Row 5: Special characters and actions
        keys_row5 = ['@', '#', '$', '%', '&', '*', '-', '_', '+', '=']
        
        button_style = """
            QPushButton {
                background-color: rgba(142, 142, 147, 0.3);
                color: #ffffff;
                font-size: 12pt;
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
        
        # Add number row
        for col_idx, key in enumerate(keys_row1):
            btn = QPushButton(key)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(lambda checked, k=key: self.add_character(k))
            keyboard_layout.addWidget(btn, 0, col_idx)
        
        # Add QWERTY rows - store buttons for shift toggle
        for row_idx, row in enumerate([keys_row2, keys_row3, keys_row4], start=1):
            for col_idx, key in enumerate(row):
                btn = QPushButton(key.upper())
                btn.setStyleSheet(button_style)
                btn.clicked.connect(lambda checked, k=key: self.add_character(k))
                keyboard_layout.addWidget(btn, row_idx, col_idx)
                # Store letter buttons for shift toggle
                self.letter_buttons.append((btn, key))
        
        # Add special characters row
        for col_idx, key in enumerate(keys_row5):
            btn = QPushButton(key)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(lambda checked, k=key: self.add_character(k))
            keyboard_layout.addWidget(btn, 4, col_idx)
        
        # Action buttons row
        action_row = 5
        # Shift button
        self.shift_btn = QPushButton("⇧ Shift")
        shift_style = """
            QPushButton {
                background-color: rgba(142, 142, 147, 0.4);
                color: #ffffff;
                font-size: 11pt;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
                border: 2px solid rgba(142, 142, 147, 0.6);
                min-height: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: rgba(142, 142, 147, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.8);
            }
        """
        self.shift_btn.setStyleSheet(shift_style)
        self.shift_btn.clicked.connect(self.toggle_shift)
        keyboard_layout.addWidget(self.shift_btn, action_row, 0, 1, 2)
        
        # Backspace
        backspace_btn = QPushButton("←")
        backspace_btn.setStyleSheet(button_style.replace("0.3", "0.4"))
        backspace_btn.clicked.connect(self.backspace)
        keyboard_layout.addWidget(backspace_btn, action_row, 2, 1, 2)
        
        # Space
        space_btn = QPushButton("Space")
        space_btn.setStyleSheet(button_style)
        space_btn.clicked.connect(lambda: self.add_character(' '))
        keyboard_layout.addWidget(space_btn, action_row, 4, 1, 2)
        
        # Clear
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
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
                background-color: rgba(255, 59, 48, 0.9);
            }
        """)
        clear_btn.clicked.connect(self.clear_all)
        keyboard_layout.addWidget(clear_btn, action_row, 6, 1, 2)
        
        keyboard_widget.setLayout(keyboard_layout)
        main_layout.addWidget(keyboard_widget)
        
        # Confirm/Cancel buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
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
            }
            QPushButton:hover {
                background-color: #D70015;
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
            }
            QPushButton:hover {
                background-color: #30B350;
            }
        """)
        confirm_btn.clicked.connect(self.confirm_text)
        button_layout.addWidget(confirm_btn)
        
        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)
        
        self.setLayout(main_layout)
    
    def format_display_text(self):
        """Format display text - show as masked password"""
        if not self.current_text:
            return "Enter password..."
        # Show as asterisks for security
        return "*" * len(self.current_text)
    
    def toggle_shift(self):
        """Toggle shift state and update button labels"""
        self.shift_active = not self.shift_active
        
        # Update shift button appearance
        if self.shift_active:
            self.shift_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 122, 255, 0.6);
                    color: #ffffff;
                    font-size: 11pt;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 2px solid rgba(0, 122, 255, 0.9);
                    min-height: 40px;
                    max-height: 40px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 122, 255, 0.8);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 122, 255, 1.0);
                }
            """)
        else:
            self.shift_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(142, 142, 147, 0.4);
                    color: #ffffff;
                    font-size: 11pt;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 2px solid rgba(142, 142, 147, 0.6);
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
        
        # Update letter button labels
        for btn, key in self.letter_buttons:
            if self.shift_active:
                btn.setText(key.upper())
            else:
                btn.setText(key.lower())
    
    def add_character(self, char: str):
        """Add a character to the input"""
        # WiFi passwords can be up to 63 characters
        if len(self.current_text) < 63:
            # Use uppercase if shift is active, otherwise use the char as provided
            if char.isalpha():
                char_to_add = char.upper() if self.shift_active else char.lower()
            else:
                char_to_add = char
            self.current_text += char_to_add
            self.update_display()
            # Auto-disable shift after adding a character (like iOS)
            if self.shift_active:
                self.toggle_shift()
    
    def backspace(self):
        """Remove last character"""
        if self.current_text:
            self.current_text = self.current_text[:-1]
            self.update_display()
    
    def clear_all(self):
        """Clear all input"""
        self.current_text = ""
        self.update_display()
    
    def update_display(self):
        """Update the display label"""
        self.display.setText(self.format_display_text())
        self.char_count.setText(f"{len(self.current_text)} characters")
    
    def confirm_text(self):
        """Confirm and emit the entered text"""
        if self.current_text:
            self.text_confirmed.emit(self.current_text)
            self.accept()
        else:
            # Show error
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
            self.display.setText("❌ Please enter a password")
            
            # Reset style after 2 seconds
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.display.setStyleSheet("""
                QLabel {
                    background-color: rgba(44, 44, 46, 0.9);
                    color: #00ff00;
                    border-radius: 12px;
                    padding: 15px;
                    font-family: 'Courier New', monospace;
                    font-size: 14pt;
                    font-weight: bold;
                }
            """))
            QTimer.singleShot(2000, self.update_display)
    
    def get_text(self):
        """Get the current text (for use after exec_())"""
        return self.current_text

