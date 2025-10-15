# Circular Border System - Developer Guide

## Overview

The **Circular Border System** provides a unified framework for creating consistent circular UI elements across all AuraVision GUI components. It ensures perfect centering, consistent styling, and professional appearance for the 1080x1080 circular touchscreen.

## Architecture

### **Two-Border Design**

```
┌─────────────────────────────────┐
│                                 │
│  Layer 1: Content (center)      │
│  Layer 2: Buttons (edge)        │
│                                 │
│  Layer 3: ⚪ Fixed Border       │
│           - 2px thin            │
│           - White, 15% opacity  │
│           - Always visible      │
│           - Reference point     │
│                                 │
│  Layer 4: 🔴 Dynamic Border     │
│           - 10-25px thick       │
│           - Red, variable       │
│           - Shows on activity   │
│           - Pulsates organically│
│                                 │
└─────────────────────────────────┘
```

### **Key Files**

| File | Purpose |
|------|---------|
| `circular_border.py` | **Core system** - Border classes and configuration |
| `aura_gui.py` | **Main GUI** - Uses borders for main screen |
| `file_upload_dialog.py` | **Upload dialog** - Uses borders for uploads |
| `example_button_script.py` | **Template** - Copy for new features |

## Configuration (`CircularBorderConfig`)

### **Screen Settings**
```python
SCREEN_SIZE = 1080        # 1080x1080 circular screen
SCREEN_RADIUS = 540       # Half of screen size
```

### **Fixed Border (Always Visible)**
```python
FIXED_BORDER_WIDTH = 2                      # Thin border
FIXED_BORDER_COLOR = "rgba(255, 255, 255, 0.15)"  # Subtle white
FIXED_BORDER_RADIUS = 538                   # Account for width
```

### **Safe Content Area**
```python
SAFE_AREA_MARGIN = 120     # Minimum margin from edge
SAFE_CONTENT_SIZE = 840    # 1080 - (120 * 2)
```

### **Dynamic Border (Activity)**
```python
DYNAMIC_BORDER_MIN = 10          # Minimum width
DYNAMIC_BORDER_MAX = 25          # Maximum width
DYNAMIC_BORDER_COLOR = "rgb(200, 0, 0)"  # Bright red
```

## Classes

### **`FixedCircularBorder`** ⚪

**Purpose**: Always-visible reference border

**Usage**:
```python
from circular_border import FixedCircularBorder

class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Add fixed border
        self.fixed_border = FixedCircularBorder(self, size=1080)
```

**Properties**:
- Always visible
- 2px thin white border
- 15% opacity (subtle)
- Mouse-transparent (clicks pass through)
- Defines circular boundary

### **`DynamicCircularBorder`** 🔴

**Purpose**: Activity-responsive pulsating border

**Usage**:
```python
from circular_border import DynamicCircularBorder

# Create dynamic border
self.border = DynamicCircularBorder(self, size=1080)

# Update during animation
self.border.update_style(width=15, opacity=0.8)

# Show/hide
self.border.show_border()
self.border.hide_border()
```

**Features**:
- Variable width (10-25px)
- Variable opacity (0.6-0.9)
- Color customizable
- Shows only during activity

## Creating New Features

### **Quick Start (3 steps)**

#### **1. Import the system**
```python
from circular_border import (
    CircularBorderConfig,
    create_circular_dialog,
    get_centered_layout_margins,
    center_dialog_in_parent
)
```

#### **2. Create your dialog**
```python
def my_feature_dialog(parent=None):
    # Create dialog with automatic styling
    dialog = create_circular_dialog(parent, "My Feature")
    
    # Create layout with safe margins
    layout = QVBoxLayout()
    layout.setContentsMargins(*get_centered_layout_margins())
    
    # Add content
    layout.addStretch(1)
    layout.addWidget(QLabel("Your content here"))
    layout.addStretch(1)
    
    dialog.setLayout(layout)
    
    # Center and show
    center_dialog_in_parent(dialog, parent)
    dialog.exec_()
```

#### **3. Connect to button**
In `aura_gui.py`:
```python
def _handle_my_feature(self):
    from my_feature_script import my_feature_dialog
    my_feature_dialog(self)
```

### **Full Example: Settings Dialog**

Create `aura-control/settings_dialog.py`:

```python
#!/usr/bin/env python3
"""Settings dialog for AuraVision"""

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from circular_border import create_circular_dialog, get_centered_layout_margins, center_dialog_in_parent

def show_settings_dialog(parent=None):
    """Show settings dialog"""
    
    # Create circular dialog (automatic styling)
    dialog = create_circular_dialog(parent, "⚙ Settings")
    
    # Layout with safe margins
    layout = QVBoxLayout()
    layout.setContentsMargins(*get_centered_layout_margins())
    layout.setSpacing(15)
    
    # Vertical centering
    layout.addStretch(1)
    
    # Title
    title = QLabel("⚙ Settings")
    title.setFont(QFont("Arial", 18, QFont.Bold))
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("color: #ffffff; margin-bottom: 20px;")
    layout.addWidget(title)
    
    # Volume slider
    volume_label = QLabel("🔊 Volume")
    volume_label.setStyleSheet("color: #8e8e93; font-size: 12px;")
    layout.addWidget(volume_label)
    
    volume_slider = QSlider(Qt.Horizontal)
    volume_slider.setRange(0, 100)
    volume_slider.setValue(70)
    volume_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            background: rgba(142, 142, 147, 0.3);
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #007AFF;
            width: 20px;
            height: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }
    """)
    layout.addWidget(volume_slider)
    
    # Brightness slider
    brightness_label = QLabel("☀️ Brightness")
    brightness_label.setStyleSheet("color: #8e8e93; font-size: 12px; margin-top: 15px;")
    layout.addWidget(brightness_label)
    
    brightness_slider = QSlider(Qt.Horizontal)
    brightness_slider.setRange(0, 100)
    brightness_slider.setValue(80)
    brightness_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            background: rgba(142, 142, 147, 0.3);
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #34C759;
            width: 20px;
            height: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }
    """)
    layout.addWidget(brightness_slider)
    
    # Save button
    save_btn = QPushButton("💾 Save Settings")
    save_btn.clicked.connect(dialog.accept)
    save_btn.setStyleSheet("""
        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 18px;
            padding: 12px 30px;
            font-size: 13px;
            font-weight: 500;
            margin-top: 20px;
        }
        QPushButton:hover {
            background-color: #0056CC;
        }
    """)
    layout.addWidget(save_btn, alignment=Qt.AlignCenter)
    
    # Vertical centering
    layout.addStretch(1)
    
    # Set layout and center
    dialog.setLayout(layout)
    center_dialog_in_parent(dialog, parent)
    
    # Show
    dialog.exec_()
```

Then in `aura_gui.py`, connect the button:
```python
def _handle_settings(self):
    """Handle settings button click"""
    from settings_dialog import show_settings_dialog
    show_settings_dialog(self)
```

## Benefits for Each Button

### **Current Buttons:**

#### **1. Upload ↑** (Already implemented)
- ✅ Uses fixed border reference
- ✅ Content centered within safe area
- ✅ QR dialog matches border system

#### **2. Settings ⚙** (Example above)
- Volume, brightness, preferences
- Perfectly centered sliders and controls
- Consistent with other dialogs

#### **3. Analytics 📊** (Future)
```python
from circular_border import create_circular_dialog
dialog = create_circular_dialog(parent, "📊 Analytics")
# Add charts, graphs, stats
# Auto-centered within circular border
```

#### **4. Voice 🎤** (Future)
```python
# Voice settings, wake word config
# All controls fit within safe circular area
```

#### **5. Mobile 📱** (Future)
```python
# Mobile app pairing, sync settings
# QR codes, connection status
```

#### **6. Info ℹ** (Future)
```python
# About, help, system info
# Scrollable content within circle
```

## Best Practices

### **✅ DO:**
1. **Always use `create_circular_dialog()`** for new features
2. **Always use `get_centered_layout_margins()`** for layouts
3. **Add top and bottom stretches** for vertical centering
4. **Use `center_dialog_in_parent()`** before showing
5. **Test on actual circular screen** for perfect fit

### **❌ DON'T:**
1. ~~Hardcode margins~~ - Use `get_centered_layout_margins()`
2. ~~Hardcode sizes~~ - Use `CircularBorderConfig.SCREEN_SIZE`
3. ~~Skip stretches~~ - Content won't center vertically
4. ~~Use square layouts~~ - Design for circular from start
5. ~~Forget parent parameter~~ - Needed for proper centering

## Customization

### **Change Border Appearance**

Edit `circular_border.py`:

```python
# Thicker fixed border
FIXED_BORDER_WIDTH = 3

# More visible fixed border
FIXED_BORDER_COLOR = "rgba(255, 255, 255, 0.25)"

# Larger safe area margins
SAFE_AREA_MARGIN = 150

# Different dynamic border color
DYNAMIC_BORDER_COLOR = "rgb(0, 150, 255)"  # Blue instead of red
```

### **Per-Feature Customization**

```python
# Create dialog with custom color
from circular_border import DynamicCircularBorder

alert_border = DynamicCircularBorder(
    self,
    color="rgb(255, 150, 0)"  # Orange for alerts
)
```

## Migration Guide

### **Converting Existing Dialogs**

**Before**:
```python
dialog = QDialog(parent)
dialog.setFixedSize(1080, 1080)
dialog.setStyleSheet("background: black; border: 8px solid red; border-radius: 536px;")
layout.setContentsMargins(100, 80, 100, 120)  # Asymmetric, hardcoded
```

**After**:
```python
from circular_border import create_circular_dialog, get_centered_layout_margins

dialog = create_circular_dialog(parent, "My Dialog")
layout.setContentsMargins(*get_centered_layout_margins())  # Automatic, symmetric
```

**Benefits**:
- ✅ Less code (3 lines → 2 lines)
- ✅ Automatic styling
- ✅ Guaranteed centering
- ✅ Consistent with other dialogs

## Testing

### **Visual Test**
```bash
cd aura-control
python3 example_button_script.py
```

Should show:
- ⚪ Subtle white ring (fixed border)
- 🔴 Dialog with red border
- ✅ Content perfectly centered
- ✅ No clipping at edges

### **Integration Test**
In main GUI, each button should:
1. Open dialog
2. Dialog has red border
3. Content fits within fixed white ring
4. No overlap with screen edges

## Future Enhancements

### **Planned Features**

1. **Color Themes**
   - Different border colors per feature
   - Theme system (dark, light, custom)

2. **Animation Presets**
   - Pulse animations for different states
   - Transition effects between screens

3. **Responsive Sizing**
   - Auto-adjust for different screen sizes
   - Support for non-circular displays

4. **Accessibility**
   - High contrast mode
   - Larger touch targets
   - Voice feedback integration

## Summary

The Circular Border System provides:

✅ **Consistency** - All dialogs look and feel the same  
✅ **Simplicity** - 2-line integration for new features  
✅ **Reliability** - Guaranteed centering and layout  
✅ **Maintainability** - Single source of truth  
✅ **Scalability** - Easy to add new features  
✅ **Professional** - Polished, cohesive UI  

**Every button script can now create perfectly centered circular dialogs with just a few lines of code!** 🎯

---

## Quick Reference

```python
# Import
from circular_border import create_circular_dialog, get_centered_layout_margins, center_dialog_in_parent

# Create
dialog = create_circular_dialog(parent, "Title")

# Layout
layout = QVBoxLayout()
layout.setContentsMargins(*get_centered_layout_margins())
layout.addStretch(1)

# Content
layout.addWidget(your_content)

# Finish
layout.addStretch(1)
dialog.setLayout(layout)
center_dialog_in_parent(dialog, parent)
dialog.exec_()
```

**That's it!** Your feature is now integrated with the circular border system. 🚀

