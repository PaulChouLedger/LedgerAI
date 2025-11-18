# base_dialog.py — Base Dialog Template for Aura Dialogs
# This template ensures consistent behavior across all dialogs:
# - Proper z-ordering without artifacts
# - Smooth animations
# - Transcription blocking/unblocking
# - Parent window reactivation
# - Proper cleanup

from PyQt5.QtWidgets import QDialog, QApplication, QWidget
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QColor, QPen


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
        
        # Set base stylesheet - NO CSS border (border is painted in paintEvent to match home screen exactly)
        # Border is drawn in paintEvent with exact same color/transparency as home screen
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border: none;
                border-radius: 535px;
            }
        """)
        
        # Initialize opacity to 0 for fade-in animation
        self.setWindowOpacity(0.0)
        
        # Override in subclass to set up UI
        self._setup_ui()
        
        # Create white border overlay widget (like home screen BorderOverlayWidget)
        # This ensures border is always on top of all child widgets
        self._create_border_overlay()
        
        # Center the dialog
        self._center_dialog()
    
    def _create_border_overlay(self):
        """Create transparent overlay widget to draw white border on top of all widgets"""
        from PyQt5.QtWidgets import QWidget
        
        class BorderOverlay(QWidget):
            """Transparent overlay widget that draws the white border"""
            def __init__(self, parent_dialog):
                super().__init__(parent_dialog)
                self.parent_dialog = parent_dialog
                self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self.setStyleSheet("background: transparent;")
                self.setGeometry(0, 0, 1080, 1080)
            
            def paintEvent(self, event):
                """Paint white border - EXACT match to BorderOverlayWidget in aura_gui.py"""
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setBrush(Qt.NoBrush)
                
                center = 540
                # Move border to the edge: 540 (screen edge) - 4 (half of 8px pen) - 1 (safety) = 535px
                radius = 535  # Right at the edge of the circular screen
                
                # White reference circle (always) - 70% transparent (30% opacity)
                # EXACT match to home screen: QColor(255, 255, 255, 77) where 77/255 = 30% opacity
                white_color = QColor(255, 255, 255, 77)  # Alpha: 77/255 = 30% opacity
                white_pen = QPen(white_color, 8, Qt.SolidLine)
                painter.setPen(white_pen)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                
                painter.end()
        
        self.border_overlay = BorderOverlay(self)
        self.border_overlay.raise_()  # Always on top
        self.border_overlay.show()
    
    def _setup_ui(self):
        """Override in subclass to set up dialog UI"""
        pass
    
    def paintEvent(self, event):
        """Base paintEvent - border is drawn in border_overlay widget"""
        super().paintEvent(event)
        # Ensure border overlay is always on top after repaint
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()
    
    def _center_dialog(self):
        """Center dialog so its center aligns with white border center (540, 540)"""
        try:
            # White border is drawn at center (540, 540) relative to parent window
            # Dialog center should align with this point
            
            if self.parent():
                # Get parent window position
                parent_geometry = self.parent().geometry()
                parent_x = parent_geometry.x()
                parent_y = parent_geometry.y()
                
                # White border center in screen coordinates
                white_border_center_x = parent_x + 540
                white_border_center_y = parent_y + 540
            else:
                # No parent: use screen center or (540, 540) if screen is 1080x1080
                app = QApplication.instance()
                if app and app.primaryScreen():
                    screen = app.primaryScreen().geometry()
                    if screen.width() == 1080 and screen.height() == 1080:
                        white_border_center_x = 540
                        white_border_center_y = 540
                    else:
                        white_border_center_x = screen.width() // 2
                        white_border_center_y = screen.height() // 2
                else:
                    white_border_center_x = 540
                    white_border_center_y = 540
            
            # Dialog dimensions
            dialog_width = 1080  # Always 1080x1080
            dialog_height = 1080
            
            # Position dialog so its center (540, 540) aligns with white border center
            x = white_border_center_x - 540
            y = white_border_center_y - 540
            
            # Move dialog to calculated position
            self.move(x, y)
            QApplication.processEvents()
            
            # Verify alignment
            actual_pos = self.pos()
            dialog_center_x = actual_pos.x() + 540
            dialog_center_y = actual_pos.y() + 540
            
        except Exception:
            pass  # Silently fail if centering error
    
    def showEvent(self, event):
        """Handle dialog show event with smooth fade-in animation"""
        super().showEvent(event)
        
        # Ensure dialog is shown and ready
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        
        # Reposition dialog after showing to ensure correct centering
        # This is needed because geometry might not be accurate until after show
        # Use a timer to ensure positioning happens after window is fully shown
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, self._center_dialog)  # Center after window is fully shown
        QTimer.singleShot(50, lambda: self.border_overlay.raise_() if hasattr(self, 'border_overlay') else None)  # Ensure border is on top
        QTimer.singleShot(50, lambda: self.border_overlay.update() if hasattr(self, 'border_overlay') else None)  # Force border repaint
        
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

