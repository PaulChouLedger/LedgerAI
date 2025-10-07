#!/usr/bin/env python3
"""
Example button script for AuraVision circular GUI
Shows how to create a new feature that integrates with the fixed circular border

This template can be copied for any of the 6 buttons:
- Settings ⚙
- Analytics 📊
- Voice 🎤
- Mobile 📱
- Info ℹ
- Or any future features
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from circular_border import (
    CircularBorderConfig,
    create_circular_dialog,
    get_centered_layout_margins,
    center_dialog_in_parent
)


def show_example_dialog(parent=None):
    """
    Example function showing how to create a circular dialog
    This is called when a button is clicked on the main GUI
    
    Args:
        parent: Parent widget (usually the main AuraGUI window)
    """
    
    # === Method 1: Using helper function (recommended) ===
    dialog = create_circular_dialog(parent, "My Feature")
    
    # Create layout with standard safe margins
    layout = QVBoxLayout()
    left, top, right, bottom = get_centered_layout_margins()
    layout.setContentsMargins(left, top, right, bottom)
    layout.setSpacing(20)
    
    # Add top stretch for vertical centering
    layout.addStretch(1)
    
    # === Add your content here ===
    
    # Title
    title = QLabel("🎯 My Feature")
    title.setFont(QFont("Arial", 18, QFont.Bold))
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("color: #ffffff; margin: 10px;")
    layout.addWidget(title)
    
    # Description
    desc = QLabel("This is an example feature.\nContent is automatically centered within the circular border.")
    desc.setAlignment(Qt.AlignCenter)
    desc.setStyleSheet("color: #8e8e93; font-size: 12px; margin: 10px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Some content area
    content = QLabel("Main content goes here.\nIt will fit perfectly within the circular safe area.")
    content.setAlignment(Qt.AlignCenter)
    content.setStyleSheet("""
        QLabel {
            background-color: rgba(44, 44, 46, 0.8);
            color: white;
            border-radius: 15px;
            padding: 20px;
            font-size: 12px;
        }
    """)
    content.setWordWrap(True)
    layout.addWidget(content)
    
    # Buttons
    button_layout = QHBoxLayout()
    
    ok_btn = QPushButton("✅ OK")
    ok_btn.clicked.connect(dialog.accept)
    ok_btn.setStyleSheet("""
        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 15px;
            padding: 10px 20px;
            font-size: 12px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #0056CC;
        }
    """)
    button_layout.addWidget(ok_btn)
    
    cancel_btn = QPushButton("❌ Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    cancel_btn.setStyleSheet("""
        QPushButton {
            background-color: rgba(142, 142, 147, 0.3);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 10px 20px;
            font-size: 12px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: rgba(142, 142, 147, 0.5);
        }
    """)
    button_layout.addWidget(cancel_btn)
    
    layout.addLayout(button_layout)
    
    # Add bottom stretch for vertical centering
    layout.addStretch(1)
    
    # Set layout
    dialog.setLayout(layout)
    
    # Center dialog within parent
    center_dialog_in_parent(dialog, parent)
    
    # Show dialog
    result = dialog.exec_()
    
    return result == QDialog.Accepted


# === Method 2: Manual configuration (advanced) ===

def show_custom_dialog(parent=None):
    """
    Example showing manual configuration for more control
    """
    from PyQt5.QtWidgets import QDialog
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("Custom Dialog")
    
    # Use shared configuration
    dialog.setFixedSize(
        CircularBorderConfig.SCREEN_SIZE,
        CircularBorderConfig.SCREEN_SIZE
    )
    dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    dialog.setStyleSheet(CircularBorderConfig.get_dialog_stylesheet())
    
    # Create layout with safe margins
    layout = QVBoxLayout()
    layout.setContentsMargins(*CircularBorderConfig.get_safe_margins())
    
    # Your content here...
    layout.addStretch(1)
    layout.addWidget(QLabel("Custom content"))
    layout.addStretch(1)
    
    dialog.setLayout(layout)
    center_dialog_in_parent(dialog, parent)
    dialog.exec_()


# === Access configuration values ===

def get_screen_info():
    """Example: Access screen configuration"""
    print(f"Screen size: {CircularBorderConfig.SCREEN_SIZE}x{CircularBorderConfig.SCREEN_SIZE}")
    print(f"Screen radius: {CircularBorderConfig.SCREEN_RADIUS}px")
    print(f"Safe content area: {CircularBorderConfig.SAFE_CONTENT_SIZE}x{CircularBorderConfig.SAFE_CONTENT_SIZE}")
    print(f"Safe margins: {CircularBorderConfig.get_safe_margins()}")
    print(f"Fixed border: {CircularBorderConfig.FIXED_BORDER_WIDTH}px, {CircularBorderConfig.FIXED_BORDER_COLOR}")


# === Integration with button handler ===

def handle_my_button_click():
    """
    This is what you'd call from the button's click handler in aura_gui.py
    
    In aura_gui.py:
        def _handle_my_feature(self):
            from my_button_script import show_example_dialog
            show_example_dialog(self)
    """
    pass


if __name__ == "__main__":
    """Test the dialog standalone"""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    result = show_example_dialog()
    print(f"Dialog result: {'Accepted' if result else 'Rejected'}")
    sys.exit(0)

