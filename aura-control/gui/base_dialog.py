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
        print(f"[BaseAuraDialog] 🚀 __init__ called: title={title}, size={size}, parent={parent}")
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
        # White border: 8px solid white, radius 535px to match main window's white border radius
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border: 8px solid white;
                border-radius: 535px;
            }
        """)
        print(f"[BaseAuraDialog] ✅ Stylesheet set, size={self.size()}")
        
        # Initialize opacity to 0 for fade-in animation
        self.setWindowOpacity(0.0)
        
        # Override in subclass to set up UI
        self._setup_ui()
        
        # Create debug green border overlay to verify alignment
        print("[BaseAuraDialog] 🔧 About to create debug border...")
        try:
            self._create_debug_border()
            print("[BaseAuraDialog] ✅ Debug border creation completed")
        except Exception as e:
            print(f"[BaseAuraDialog] ❌ Error creating debug border: {e}")
            import traceback
            traceback.print_exc()
        
        # Center the dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Override in subclass to set up dialog UI"""
        pass
    
    def paintEvent(self, event):
        """Override paintEvent to draw green debug border directly on dialog"""
        super().paintEvent(event)
        
        # Draw green debug border directly on the dialog
        try:
            painter = QPainter(self)
            if painter.isActive():
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setBrush(Qt.NoBrush)
                
                # Green border at dialog center (540, 540) with radius 535
                center = 540
                radius = 535
                
                # Bright green border - very thick and fully opaque
                green_color = QColor(0, 255, 0, 255)  # Fully opaque bright green
                green_pen = QPen(green_color, 16, Qt.SolidLine)  # Very thick for visibility
                painter.setPen(green_pen)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                
                print(f"[BaseAuraDialog] 🟢 ✅ Painted green border in paintEvent: center=({center}, {center}), radius={radius}")
                painter.end()
        except Exception as e:
            print(f"[BaseAuraDialog] ❌ Error painting green border: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_debug_border(self):
        """Create a green debug border overlay to verify alignment with white border"""
        print("[BaseAuraDialog] 🟢 Starting _create_debug_border()...")
        
        class DebugBorderWidget(QWidget):
            """Green border overlay to show dialog center alignment"""
            def __init__(self, parent):
                print("[DebugBorder] 🔧 Initializing DebugBorderWidget...")
                super().__init__(parent)
                print(f"[DebugBorder] 🔧 Parent: {parent}, parent type: {type(parent)}")
                
                self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self.setAttribute(Qt.WA_NoSystemBackground, True)
                self.setStyleSheet("background: transparent;")
                
                # Cover entire dialog
                self.setGeometry(0, 0, 1080, 1080)
                print(f"[DebugBorder] 🟢 Widget created: geometry={self.geometry()}, visible={self.isVisible()}, parent={self.parent()}")
            
            def paintEvent(self, event):
                """Paint green debug border at dialog center"""
                print(f"[DebugBorder] 🎨 paintEvent called! rect={event.rect()}")
                try:
                    painter = QPainter(self)
                    if not painter.isActive():
                        print("[DebugBorder] ❌ Painter not active!")
                        return
                    
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    painter.setBrush(Qt.NoBrush)
                    
                    # Green border at dialog center (540, 540) with radius 535
                    # This should align with the white border if centering is correct
                    center = 540
                    radius = 535
                    
                    # Bright green border for visibility - make it thicker and more opaque
                    green_color = QColor(0, 255, 0, 255)  # Fully opaque bright green
                    green_pen = QPen(green_color, 12, Qt.SolidLine)  # Thicker for visibility
                    painter.setPen(green_pen)
                    painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                    
                    print(f"[DebugBorder] 🟢 ✅ Painted green border: center=({center}, {center}), radius={radius}, pen_width=12")
                    painter.end()
                except Exception as e:
                    print(f"[DebugBorder] ❌ Error in paintEvent: {e}")
                    import traceback
                    traceback.print_exc()
            
            def showEvent(self, event):
                """Override showEvent to ensure visibility"""
                super().showEvent(event)
                print(f"[DebugBorder] 👁️ showEvent called, visible={self.isVisible()}")
                self.update()
        
        # Create and show debug border overlay
        print("[BaseAuraDialog] 🔧 Creating DebugBorderWidget instance...")
        try:
            self.debug_border = DebugBorderWidget(self)
            print(f"[BaseAuraDialog] ✅ DebugBorderWidget created: {self.debug_border}")
            
            # Ensure it's visible
            self.debug_border.setParent(self)
            self.debug_border.raise_()  # Ensure it's on top
            self.debug_border.show()
            self.debug_border.setVisible(True)
            self.debug_border.update()  # Force immediate repaint
            
            # Force a repaint
            QApplication.processEvents()
            self.debug_border.repaint()
            
            print(f"[BaseAuraDialog] 🟢 Debug green border created: geometry={self.debug_border.geometry()}, visible={self.debug_border.isVisible()}, parent={self.debug_border.parent()}")
        except Exception as e:
            print(f"[BaseAuraDialog] ❌ Error creating debug border widget: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            print(f"[BaseAuraDialog] 🎯 Centered: white_border=({white_border_center_x}, {white_border_center_y}), dialog_pos=({x}, {y}), dialog_center=({dialog_center_x}, {dialog_center_y}), offset=({dialog_center_x - white_border_center_x}, {dialog_center_y - white_border_center_y})")
            
        except Exception as e:
            print(f"[BaseAuraDialog] ❌ Error centering: {e}")
            import traceback
            traceback.print_exc()
    
    def showEvent(self, event):
        """Handle dialog show event with smooth fade-in animation"""
        super().showEvent(event)
        
        # Ensure dialog is shown and ready
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        
        # Ensure debug border is visible
        if hasattr(self, 'debug_border'):
            self.debug_border.raise_()
            self.debug_border.show()
            self.debug_border.update()
            print(f"[BaseAuraDialog] 🟢 Debug border shown: visible={self.debug_border.isVisible()}, geometry={self.debug_border.geometry()}")
        
        # Reposition dialog after showing to ensure correct centering
        # This is needed because geometry might not be accurate until after show
        # Use a timer to ensure positioning happens after window is fully shown
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(10, self._center_dialog)
        QTimer.singleShot(50, self._center_dialog)  # Double-check after a short delay
        QTimer.singleShot(100, lambda: self.debug_border.update() if hasattr(self, 'debug_border') else None)  # Force debug border repaint
        
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

