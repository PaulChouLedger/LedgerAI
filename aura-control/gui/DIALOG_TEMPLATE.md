# Dialog Template Guide

This document explains how to use the `BaseAuraDialog` template for creating new dialogs in Aura.

## BaseAuraDialog Features

The `BaseAuraDialog` class provides:
- ✅ Proper z-ordering without artifacts
- ✅ Smooth fade-in/fade-out animations
- ✅ Automatic transcription blocking/unblocking
- ✅ Parent window reactivation on close
- ✅ Proper cleanup with `Qt.WA_DeleteOnClose`
- ✅ Consistent window flags for modal behavior

## Usage Example

```python
from gui.base_dialog import BaseAuraDialog

class MyDialog(BaseAuraDialog):
    def __init__(self, parent=None):
        # Initialize your attributes first
        self.my_data = []
        
        # Initialize base dialog
        super().__init__(
            parent=parent,
            title="My Dialog Title",
            size=(1080, 1080),  # Standard Aura dialog size
            modal=True  # or False for non-modal
        )
        
        # Set your stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border-radius: 536px;
            }
        """)
    
    def _setup_ui(self):
        """Set up dialog UI (called by base class)"""
        # Your UI setup code here
        layout = QVBoxLayout()
        # ... add widgets ...
        self.setLayout(layout)
    
    def _on_show(self):
        """Optional: Additional logic when dialog opens"""
        # Base class already blocks transcription, but you can add more here
        pass
    
    def _on_close(self):
        """Optional: Cleanup when dialog closes"""
        # Base class already unblocks transcription and reactivates parent
        # Add your cleanup here (e.g., stop timers, disconnect signals)
        pass
```

## Key Methods to Override

### `_setup_ui(self)`
**Required.** Called by base class during initialization. Set up all your UI widgets here.

### `_on_show(self)`
**Optional.** Called when dialog is shown. Base class already:
- Blocks transcription
- Handles fade-in animation
- Raises and activates the dialog

Override to add custom show logic (e.g., refresh data, start timers).

### `_on_close(self)`
**Optional.** Called when dialog is closing (before final close). Base class already:
- Unblocks transcription
- Reactivates parent window
- Handles fade-out animation

Override to add cleanup (e.g., stop timers, disconnect signals, save data).

## Window Flags

The base class automatically sets:
- `Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint` when parent exists
- `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint` when no parent

This ensures proper z-ordering above parent dialogs.

## Animation Behavior

- **Modal dialogs with parent**: Opacity set to 1.0 immediately (no fade) to avoid rendering artifacts
- **Standalone dialogs**: Smooth 400ms fade-in with `OutCubic` easing
- **Close animation**: 300ms fade-out with `InCubic` easing (for non-modal dialogs)
- **Modal dialogs**: Close immediately (no fade) to avoid blocking

## Transcription Blocking

The base class automatically:
- Blocks transcription when dialog opens (via `_on_show`)
- Unblocks transcription when dialog closes (via `closeEvent`)

You don't need to manually call `block_transcription`/`unblock_transcription` unless you have special requirements.

## Migration from Old Dialogs

To migrate an existing dialog:

1. Change `class MyDialog(QDialog):` to `class MyDialog(BaseAuraDialog):`
2. Update `__init__` to call `super().__init__(parent, title, size, modal)`
3. Move UI setup to `_setup_ui()` method
4. Remove custom `showEvent` and `closeEvent` (use `_on_show` and `_on_close` hooks instead)
5. Remove manual window flags, opacity, and centering code (handled by base class)
6. Remove manual transcription blocking/unblocking (handled by base class)

## Standard Pattern

All dialogs should follow this exact pattern:

1. **Inherit from `BaseAuraDialog`**
2. **Call `super().__init__()` with parent, title, size, and modal parameters**
3. **Override `_setup_ui()` for UI setup** (required)
4. **Override `_on_show()` for show-time logic** (optional)
5. **Override `_on_close()` for cleanup** (optional)
6. **Only override `closeEvent()` if you need to prevent closing** (then call `super().closeEvent(event)`)

**DO NOT:**
- Override `showEvent()` - use `_on_show()` instead
- Override `closeEvent()` unless you need to prevent closing - use `_on_close()` instead
- Implement custom `center_dialog()` - base class handles this
- Implement custom fade animations - base class handles this
- Manually block/unblock transcription - base class handles this

## Examples

All dialogs now follow this pattern:
- `file_upload_dialog.py` - File upload dialog
- `wallet_dialog.py` - Wallet connection dialog
- `settings_dialog.py` - Settings dialog
- `welcome_setup_dialog.py` - Welcome/WiFi setup dialog
- `payment_dialog.py` - Payment dialog
- `metamask_payment_dialog.py` - MetaMask payment dialog

