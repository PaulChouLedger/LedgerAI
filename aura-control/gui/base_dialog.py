# base_dialog.py — Base Dialog Template for Aura Dialogs
# This template ensures consistent behavior across all dialogs:
# - Proper z-ordering without artifacts
# - Smooth animations
# - Transcription blocking/unblocking
# - Parent window reactivation
# - Proper cleanup

from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve


class BaseAuraDialog(QDialog):
    """Base dialog class for all Aura dialogs with consistent behavior"""
    
    def __init__(self, parent=None, title="Aura Dialog", size=(1080, 1080), modal=True):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setFixedSize(size[0], size[1])
        
        # Window flags for proper z-ordering
        if parent:
            # Use Window flag instead of Dialog to ensure proper z-ordering above parent
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setModal(modal)
        else:
            # If no parent, stay on top
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Ensure resources are freed on close
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        # Set base stylesheet with white border to match white perimeter
        # White border: 8px solid white, radius 536px (540 - 4 to account for border width)
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border: 8px solid white;
                border-radius: 536px;
            }
        """)
        
        # Initialize opacity to 0 for fade-in animation
        self.setWindowOpacity(0.0)
        
        # Override in subclass to set up UI
        self._setup_ui()
        
        # Center the dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Override in subclass to set up dialog UI"""
        pass
    
    def _center_dialog(self):
        """Position dialog to align with white perimeter reference circle"""
        if self.parent():
            # Align with parent window position to match white perimeter
            # Parent window is at (0, 0) with 1080x1080 size
            # Dialog should be at same position to align borders perfectly
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x()  # Match parent's x position
            y = parent_geometry.y()  # Match parent's y position
            self.move(x, y)
            print(f"[BaseAuraDialog] 🎯 Dialog aligned with parent: position=({x}, {y})")
        else:
            # No parent: position at (0, 0) to match main window position
            self.move(0, 0)
            print(f"[BaseAuraDialog] 🎯 Dialog positioned at (0, 0)")
    
    def showEvent(self, event):
        """Handle dialog show event with smooth fade-in animation"""
        super().showEvent(event)
        
        # Ensure dialog is positioned and ready
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        
        # For sub-dialogs opened from parent, skip fade animation to avoid rendering issues
        if self.isModal() and self.parent():
            self.setWindowOpacity(1.0)
        else:
            # Only use fade animation for standalone dialogs
            self.fade_in = QPropertyAnimation(self, b"windowOpacity")
            self.fade_in.setDuration(400)
            self.fade_in.setStartValue(0.0)
            self.fade_in.setEndValue(1.0)
            self.fade_in.setEasingCurve(QEasingCurve.OutCubic)
            self.fade_in.start()
        
        # Call subclass hook for additional show logic
        self._on_show()
    
    def _on_show(self):
        """Override in subclass for additional show logic (e.g., blocking transcription)"""
        pass
    
    def closeEvent(self, event):
        """Handle dialog close event with smooth fade-out animation"""
        # Always ensure transcription is unblocked when dialog closes
        try:
            self._unblock_transcription()
        except Exception:
            pass
        
        # Reactivate parent window immediately to prevent freezing
        if self.parent():
            try:
                self.parent().raise_()
                self.parent().activateWindow()
                QApplication.processEvents()
            except (RuntimeError, AttributeError):
                pass  # Parent already deleted
        
        # Only animate if we're actually closing (not just hiding)
        if event.spontaneous() or not self.isVisible():
            # Still call cleanup even if accepting immediately
            try:
                self._on_close()
            except Exception:
                pass
            event.accept()
            return
        
        # Cancel fade-in if still running
        if hasattr(self, 'fade_in') and self.fade_in:
            try:
                if self.fade_in.state() == QPropertyAnimation.Running:
                    self.fade_in.stop()
            except (RuntimeError, AttributeError):
                pass  # Animation already deleted
        
        # For modal dialogs opened from home screen, accept immediately to avoid blocking
        if self.isModal() and self.parent():
            # Still call cleanup
            try:
                self._on_close()
            except Exception:
                pass
            event.accept()
            return
        
        # Non-modal: use fade animation
        try:
            self.fade_out = QPropertyAnimation(self, b"windowOpacity")
            self.fade_out.setDuration(300)  # Slightly longer for smoother exit
            self.fade_out.setStartValue(self.windowOpacity())
            self.fade_out.setEndValue(0.0)
            self.fade_out.setEasingCurve(QEasingCurve.InCubic)  # Smooth ease-in for exit
            
            # Connect finished signal to actually close the dialog
            def _finalize():
                # Call subclass cleanup hook (with error handling)
                try:
                    self._on_close()
                except Exception as e:
                    print(f"[BaseAuraDialog] ⚠️ Error in _on_close: {e}")
                
                event.accept()
                
                # Ensure parent is reactivated after close
                if self.parent():
                    try:
                        self.parent().raise_()
                        self.parent().activateWindow()
                        QApplication.processEvents()
                    except (RuntimeError, AttributeError):
                        pass  # Parent already deleted
                
                try:
                    self.dialog_closed.emit()
                except (RuntimeError, AttributeError):
                    pass  # Dialog already deleted
            
            self.fade_out.finished.connect(_finalize)
            self.fade_out.start()
            
            # Prevent immediate close
            event.ignore()
        except (RuntimeError, AttributeError) as e:
            # If we can't create animation (dialog already being deleted), just accept
            print(f"[BaseAuraDialog] ⚠️ Cannot animate close: {e}")
            try:
                self._on_close()
            except Exception:
                pass
            event.accept()
    
    def _on_close(self):
        """Override in subclass for additional close logic (e.g., thread cleanup)"""
        pass
    
    def _block_transcription(self, reason="Dialog open"):
        """Block transcription when dialog opens"""
        try:
            from listener import block_transcription
            block_transcription(reason)
            print(f"[{self.__class__.__name__}] 🚫 Transcription blocked: {reason}")
        except ImportError:
            print(f"[{self.__class__.__name__}] ⚠️ Could not import listener blocking functions")
        except Exception as e:
            print(f"[{self.__class__.__name__}] ⚠️ Error blocking transcription: {e}")
    
    def _unblock_transcription(self):
        """Unblock transcription when dialog closes"""
        try:
            from listener import unblock_transcription
            unblock_transcription()
            print(f"[{self.__class__.__name__}] ✅ Transcription unblocked")
        except ImportError:
            print(f"[{self.__class__.__name__}] ⚠️ Could not import listener unblocking functions")
        except Exception as e:
            print(f"[{self.__class__.__name__}] ⚠️ Error unblocking transcription: {e}")

