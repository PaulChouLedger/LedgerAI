#!/usr/bin/env python3
"""
Circular Border System for AuraVision
Provides fixed transparent reference border for all GUI components

This creates a consistent circular boundary that:
- Always visible (subtle white ring)
- Defines safe content area
- Used by all dialogs and screens
- Ensures consistent UX across all features
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt

class CircularBorderConfig:
    """Shared configuration for all circular UI elements"""
    
    # Screen configuration
    SCREEN_SIZE = 1080  # 1080x1080 circular screen
    SCREEN_RADIUS = 540  # Half of screen size
    
    # Fixed border configuration
    FIXED_BORDER_WIDTH = 3  # Thin reference border
    FIXED_BORDER_COLOR = "rgba(255, 255, 255, 1.0)"  # SOLID WHITE for visibility/testing
    FIXED_BORDER_RADIUS = 538  # 540 - 2 (account for border width)
    
    # Safe content area (inside fixed border)
    SAFE_AREA_MARGIN = 120  # Minimum margin from edge
    SAFE_CONTENT_SIZE = SCREEN_SIZE - (SAFE_AREA_MARGIN * 2)  # 840x840
    
    # Dynamic border configuration (for transcription, etc.)
    DYNAMIC_BORDER_MIN = 10
    DYNAMIC_BORDER_MAX = 25
    DYNAMIC_BORDER_COLOR = "rgb(200, 0, 0)"  # Bright red
    
    # Dialog configuration
    DIALOG_BORDER_WIDTH = 8
    DIALOG_BORDER_COLOR = "#ff0000"  # Red
    DIALOG_BORDER_RADIUS = 536  # Account for border width
    DIALOG_BACKGROUND = "rgba(28, 28, 30, 0.95)"
    
    @staticmethod
    def get_safe_margins():
        """Get symmetric margins for content to fit within circular area"""
        return (
            CircularBorderConfig.SAFE_AREA_MARGIN,  # left
            CircularBorderConfig.SAFE_AREA_MARGIN,  # top
            CircularBorderConfig.SAFE_AREA_MARGIN,  # right
            CircularBorderConfig.SAFE_AREA_MARGIN   # bottom
        )
    
    @staticmethod
    def get_dialog_stylesheet():
        """Get standard stylesheet for circular dialogs"""
        return f"""
            QDialog {{
                background-color: {CircularBorderConfig.DIALOG_BACKGROUND};
                color: white;
                border: {CircularBorderConfig.DIALOG_BORDER_WIDTH}px solid {CircularBorderConfig.DIALOG_BORDER_COLOR};
                border-radius: {CircularBorderConfig.DIALOG_BORDER_RADIUS}px;
            }}
            QLabel {{
                color: white;
            }}
        """


class FixedCircularBorder(QWidget):
    """
    Fixed transparent circular border widget
    Always visible to provide consistent reference for all UI elements
    """
    
    def __init__(self, parent, size=None):
        super().__init__(parent)
        
        # Use provided size or default config
        if size is None:
            size = CircularBorderConfig.SCREEN_SIZE
        
        # Set geometry
        self.setGeometry(0, 0, size, size)
        
        # Apply styling
        border_radius = size // 2
        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                border: {CircularBorderConfig.FIXED_BORDER_WIDTH}px solid {CircularBorderConfig.FIXED_BORDER_COLOR};
                border-radius: {border_radius}px;
            }}
        """)
        
        # Attributes for proper rendering
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # Allow transparent background
        
        # Always on top but below dynamic borders
        self.raise_()
        self.show()
        
        print(f"[CircularBorder] ⚪ Fixed border created: {size}x{size}, radius={border_radius}px")
    
    def update_geometry(self, size):
        """Update border geometry (e.g., after window resize)"""
        self.setGeometry(0, 0, size, size)
        border_radius = size // 2
        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                border: {CircularBorderConfig.FIXED_BORDER_WIDTH}px solid {CircularBorderConfig.FIXED_BORDER_COLOR};
                border-radius: {border_radius}px;
            }}
        """)
        self.raise_()


class DynamicCircularBorder(QWidget):
    """
    Dynamic pulsating circular border widget
    Shows during specific activities (transcription, alerts, etc.)
    """
    
    def __init__(self, parent, size=None, color=None):
        super().__init__(parent)
        
        # Use provided values or defaults
        if size is None:
            size = CircularBorderConfig.SCREEN_SIZE
        if color is None:
            color = CircularBorderConfig.DYNAMIC_BORDER_COLOR
        
        self.color = color
        self.current_width = CircularBorderConfig.DYNAMIC_BORDER_MIN
        self.current_opacity = 0.7
        
        # Set geometry
        self.setGeometry(0, 0, size, size)
        self.border_radius = size // 2
        
        # Attributes for proper rendering
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        # Don't set window flags for child widgets - it breaks rendering
        # Just use raise_() to control z-order
        self.hide()  # Hidden by default
        
        print(f"[CircularBorder] 🔴 Dynamic border created: {size}x{size}, color={color}")
    
    def update_style(self, width, opacity):
        """Update border width and opacity for animation"""
        self.current_width = width
        self.current_opacity = opacity
        
        # Use rgba for opacity in the stylesheet instead of setWindowOpacity
        # setWindowOpacity only works for top-level windows, not child widgets
        rgba_color = self._color_with_opacity(self.color, opacity)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                border: {width}px solid {rgba_color};
                border-radius: {self.border_radius}px;
            }}
        """)
        
        # Ensure visibility and z-order
        was_visible = self.isVisible()
        if not was_visible:
            self.show()
            print(f"[CircularBorder] 🔴 SHOWING border: width={width}px, opacity={opacity:.2f}, color={rgba_color}")
        self.raise_()
        self.update()
        
        # Debug every 30 frames
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:
            print(f"[CircularBorder] 🔴 Border update: visible={self.isVisible()}, width={width}px, opacity={opacity:.2f}, geometry={self.geometry()}")
    
    def _color_with_opacity(self, rgb_color, opacity):
        """Convert rgb(r,g,b) to rgba(r,g,b,opacity)"""
        # rgb(200, 0, 0) -> rgba(200, 0, 0, 0.7)
        if rgb_color.startswith('rgb(') and rgb_color.endswith(')'):
            rgb_values = rgb_color[4:-1]  # Extract "200, 0, 0"
            return f"rgba({rgb_values}, {opacity})"
        return rgb_color  # Fallback
    
    def show_border(self):
        """Show the dynamic border"""
        self.show()
        self.raise_()
        print(f"[CircularBorder] 🔴 Dynamic border shown")
    
    def hide_border(self):
        """Hide the dynamic border"""
        self.hide()
        print(f"[CircularBorder] 🔴 Dynamic border hidden")


# === Helper functions for dialogs ===

def create_circular_dialog(parent, title="Dialog"):
    """
    Create a standard circular dialog with proper styling
    
    Args:
        parent: Parent widget
        title: Dialog title
        
    Returns:
        Configured QDialog
    """
    from PyQt5.QtWidgets import QDialog
    
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setFixedSize(
        CircularBorderConfig.SCREEN_SIZE,
        CircularBorderConfig.SCREEN_SIZE
    )
    dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    dialog.setStyleSheet(CircularBorderConfig.get_dialog_stylesheet())
    
    return dialog


def get_centered_layout_margins():
    """
    Get standard margins for centering content within circular border
    
    Returns:
        Tuple of (left, top, right, bottom) margins
    """
    return CircularBorderConfig.get_safe_margins()


def center_dialog_in_parent(dialog, parent):
    """
    Center a dialog within its parent window
    
    Args:
        dialog: QDialog to center
        parent: Parent window/widget
    """
    if parent:
        parent_rect = parent.geometry()
        x = parent_rect.x() + (parent_rect.width() - dialog.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - dialog.height()) // 2
        dialog.move(x, y)
        print(f"[CircularBorder] 🎯 Dialog centered: position=({x}, {y})")
    else:
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - dialog.width()) // 2
        y = (screen.height() - dialog.height()) // 2
        dialog.move(x, y)


# === Usage Examples ===

"""
Example 1: Create a new button script with circular dialog
--------------------------------------------------------------

from circular_border import (
    CircularBorderConfig, 
    create_circular_dialog,
    get_centered_layout_margins,
    center_dialog_in_parent
)

def my_button_action(parent):
    # Create circular dialog
    dialog = create_circular_dialog(parent, "My Feature")
    
    # Create layout with standard margins
    layout = QVBoxLayout()
    left, top, right, bottom = get_centered_layout_margins()
    layout.setContentsMargins(left, top, right, bottom)
    
    # Add your content
    layout.addStretch(1)
    layout.addWidget(QLabel("My Content"))
    layout.addStretch(1)
    
    dialog.setLayout(layout)
    
    # Center the dialog
    center_dialog_in_parent(dialog, parent)
    
    # Show dialog
    dialog.exec_()


Example 2: Access configuration values
--------------------------------------------------------------

from circular_border import CircularBorderConfig

# Get screen size
size = CircularBorderConfig.SCREEN_SIZE  # 1080

# Get safe content area
safe_size = CircularBorderConfig.SAFE_CONTENT_SIZE  # 840

# Get margins
margins = CircularBorderConfig.get_safe_margins()  # (120, 120, 120, 120)


Example 3: Create custom border
--------------------------------------------------------------

from circular_border import FixedCircularBorder, DynamicCircularBorder

class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Add fixed reference border
        self.fixed_border = FixedCircularBorder(self)
        
        # Add custom dynamic border (e.g., blue for alerts)
        self.alert_border = DynamicCircularBorder(
            self, 
            color="rgb(0, 100, 255)"  # Blue
        )
        
        # Show alert
        self.alert_border.update_style(width=12, opacity=0.8)
        self.alert_border.show_border()
"""

