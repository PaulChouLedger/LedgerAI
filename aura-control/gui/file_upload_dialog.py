# file_upload_dialog.py — File Upload Dialog for Document Ingestion

import os
import sys
import shutil
import urllib.parse
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem, QInputDialog,
                             QLineEdit, QWidget, QApplication, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
import requests
import json
import socket
import webbrowser

# Import base dialog template
from gui.base_dialog import BaseAuraDialog

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

class FileUploadDialog(BaseAuraDialog):
    def __init__(self, parent=None):
        # Initialize attributes first
        self.uploaded_files = []
        self.touch_coordinates = []
        
        print("[Upload] 🔧 Initializing upload dialog...")
        
        # Initialize base dialog
        super().__init__(
            parent=parent,
            title="Document Upload - AuraVision",
            size=(1080, 1080),
            modal=True
        )
        
        # Set stylesheet before creating UI (base class already sets white border)
        print("[Upload] 👁️ Dialog initialized and ready")
        # Add additional styles while preserving base white border
        additional_styles = """
            /* Remove red border from message boxes */
            QMessageBox {
                background-color: rgba(28, 28, 30, 1.0);
                border: none !important;
                border-radius: 10px;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 60px;
            }
            QMessageBox QPushButton:hover {
                background-color: #0056CC;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            /* Global QPushButton styles removed - each button will have its own styling */
            /* All global QPushButton styles removed - each button will have its own styling */
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
                padding: 8px;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """
        # Combine with base stylesheet (preserving white border from base class)
        base_stylesheet = self.styleSheet()
        self.setStyleSheet(base_stylesheet + additional_styles)
    
    def _setup_ui(self):
        """Set up dialog UI (called by base class)"""
        self.setup_ui()
    
    def _on_show(self):
        """Block transcription when dialog opens"""
        self._block_transcription("File upload dialog open")
        
    def mousePressEvent(self, event):
        """Capture mouse/touch coordinates for debugging"""
        x, y = event.x(), event.y()
        self.touch_coordinates.append((x, y))
        
        # Make output more prominent
        print("=" * 60)
        print(f"[Upload] 🖱️ TOUCH DETECTED!")
        print(f"[Upload] 📍 Dialog-relative coordinates: ({x}, {y})")
        print(f"[Upload] 📐 Dialog size: {self.width()} x {self.height()}")
        print(f"[Upload] 📐 Dialog position: ({self.x()}, {self.y()})")
        
        # Calculate center offset
        dialog_center_x = self.width() // 2
        dialog_center_y = self.height() // 2
        offset_x = x - dialog_center_x
        offset_y = y - dialog_center_y
        print(f"[Upload] 📐 Dialog center: ({dialog_center_x}, {dialog_center_y})")
        print(f"[Upload] 📐 Offset from dialog center: ({offset_x}, {offset_y})")
        
        # Calculate screen coordinates
        screen_x = self.x() + x
        screen_y = self.y() + y
        print(f"[Upload] 🌍 ABSOLUTE SCREEN COORDINATES: ({screen_x}, {screen_y})")
        
        # Check if this is near the dialog center
        if abs(offset_x) < 50 and abs(offset_y) < 50:
            print(f"[Upload] 🎯 TOUCHED DIALOG CENTER!")
            print(f"[Upload] 💡 Screen center should be around: ({screen_x}, {screen_y})")
            print(f"[Upload] 💡 Dialog should be positioned at: ({screen_x - 540}, {screen_y - 540})")
        
        # Check if this is near the screen center (assuming 1080x1080 screen)
        screen_center_x, screen_center_y = 540, 540
        screen_offset_x = screen_x - screen_center_x
        screen_offset_y = screen_y - screen_center_y
        if abs(screen_offset_x) < 50 and abs(screen_offset_y) < 50:
            print(f"[Upload] 🎯 TOUCHED SCREEN CENTER!")
            print(f"[Upload] 📊 Screen offset from center: ({screen_offset_x}, {screen_offset_y})")
        
        print("=" * 60)
        print()
        
        super().mousePressEvent(event)
        
    def center_dialog(self):
        """Center dialog to align white border with home screen white perimeter"""
        if self.parent():
            # Center dialog within parent window so white borders align
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            print(f"[Upload] 🎯 Dialog centered on parent: position=({x}, {y})")
        else:
            # No parent: center on screen
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            print(f"[Upload] 🎯 Dialog centered on screen: position=({x}, {y})")
        
        self.move(x, y)
        self.raise_()
        self.activateWindow()
    
    def close_dialog(self):
        """Close dialog with smooth fade-out animation"""
        print("[Upload] 🔄 Closing dialog with fade-out animation...")
        
        # Cancel fade-in if still running
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running:
            self.fade_in.stop()
        
        # Create optimized fade-out animation
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(300)  # Slightly longer for smoother exit
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InCubic)  # Smooth ease-in for exit
        
        # Connect finished signal to actually close the dialog
        self.fade_out.finished.connect(self._final_close)
        self.fade_out.start()
    
    def _final_close(self):
        """Final step to close the dialog"""
        print("[Upload] ✅ Dialog closing completely...")
        self.accept()
    
    def _on_close(self):
        """Additional cleanup when dialog closes (called by base class)"""
        pass

    def center_dialog_manually(self, screen_center_x, screen_center_y):
        """Manually center the dialog based on screen center coordinates"""
        # Calculate new position to center the dialog
        new_x = screen_center_x - 540  # 540 is half of 1080
        new_y = screen_center_y - 540
        self.move(new_x, new_y)
        print(f"[Upload] 🎯 Manually centered dialog at: ({new_x}, {new_y})")
        print(f"[Upload] 📐 Screen center was: ({screen_center_x}, {screen_center_y})")
        
    def setup_ui(self):
        # Create main layout with no margins for full screen
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # No margins for full screen
        main_layout.setSpacing(0)
        
        # Create content widget that fills the dialog (behind the red border)
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(28, 28, 30, 0.95);
                border-radius: 532px;  /* Adjusted to account for dialog border and margins */
                border: none;
            }
        """)
        
        # Add content widget to main layout (fills entire dialog)
        main_layout.addWidget(content_widget)
        
        # Add edge buttons for future functions - will be added after layout is set
        
        # Create layout for circular content - centered within red border
        self.content_layout = QVBoxLayout()
        # Symmetric margins to center content within the circular area
        # Red border is 8px, so safe area starts at ~8px inset
        # Use symmetric margins for proper centering
        self.content_layout.setContentsMargins(120, 100, 120, 100)  # Equal top/bottom for vertical centering
        self.content_layout.setSpacing(20)  # Compact spacing
        
        # Add top spacer for vertical centering
        self.content_layout.addStretch(1)
        
        # Title centered (no close button)
        title_layout = QHBoxLayout()
        
        # Spacer to center title
        title_layout.addStretch()
        
        # Title
        title = QLabel("📄 Document Upload")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        title_layout.addWidget(title)
        
        # Spacer to balance layout
        title_layout.addStretch()
        
        self.content_layout.addLayout(title_layout)
        
        # Description - more compact
        desc = QLabel("Upload docs to data/input - auto-ingest processes them")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 5px;")
        self.content_layout.addWidget(desc)
        
        # File selection area - optimized for circular screen
        file_layout = QHBoxLayout()
        file_layout.setSpacing(20)
        
        self.file_list = QListWidget()
        # Remove fixed height constraints - let it match button layout height
        self.file_list.setMaximumWidth(550)  # 10% bigger (500 * 1.1 = 550)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 15px;
                border: none;
                padding: 10px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 8px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: rgba(0, 122, 255, 0.3);
            }
            QListWidget::item:hover {
                background-color: rgba(142, 142, 147, 0.2);
            }
        """)
        file_layout.addWidget(self.file_list, 1)  # Add stretch factor
        
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)  # More spacing for larger buttons
        
        # Keep 4 main options (removed URL option)
        self.select_files_btn = QPushButton("📁 Select Files")
        self.select_files_btn.clicked.connect(self.select_files)
        button_layout.addWidget(self.select_files_btn)
        
        self.gdrive_btn = QPushButton("☁️ Google Drive")
        self.gdrive_btn.clicked.connect(self.add_google_drive)
        button_layout.addWidget(self.gdrive_btn)
        
        self.qr_btn = QPushButton("📱 Show QR Code")
        self.qr_btn.clicked.connect(self.show_qr_code)
        button_layout.addWidget(self.qr_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(self.clear_btn)
        
        # Apply Apple-style styling to all action buttons (compact for 4 buttons)
        action_button_style = """
            QPushButton {
                background-color: rgba(142, 142, 147, 0.2);
                color: #ffffff;
                font-size: 22px;
                font-weight: 600;
                padding: 15px 20px;
                min-height: 50px;
                border-radius: 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(142, 142, 147, 0.4);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.6);
            }
        """
        
        for button in [self.select_files_btn, self.gdrive_btn, self.qr_btn, self.clear_btn]:
            button.setStyleSheet(action_button_style)
        
        file_layout.addLayout(button_layout)
        self.content_layout.addLayout(file_layout)
        
        # Upload progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.content_layout.addWidget(self.progress_bar)
        
        # Status log - made thinner for circular screen
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)  # Much thinner
        self.status_log.setMinimumHeight(80)   # Much thinner
        self.status_log.setMaximumWidth(500)   # Narrower for circular screen
        self.status_log.setReadOnly(True)
        self.status_log.setPlaceholderText("Upload status will appear here...")
        self.status_log.setStyleSheet("""
            QTextEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 15px;
                font-size: 11px;
                border: none;
                padding: 10px;
            }
        """)
        self.content_layout.addWidget(self.status_log)
        
        # Upload and Close buttons
        button_layout = QHBoxLayout()
        
        # Upload button
        self.upload_btn = QPushButton("📤 Upload Files")
        self.upload_btn.setEnabled(False)  # Disabled until files are selected
        self.upload_btn.clicked.connect(self.upload_files)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
            QPushButton:disabled {
                background-color: rgba(142, 142, 147, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        
        # Close button
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
        """)
        
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.close_btn)
        self.content_layout.addLayout(button_layout)
        
        # Add bottom spacer to maintain proper spacing (smaller since we shifted content up)
        self.content_layout.addStretch(2)  # Use integer, not float
        
        print(f"[Upload] 🔴 Red border should show circular screen boundaries")
        
        # Set layout to content widget
        content_widget.setLayout(self.content_layout)
        
        # Set main layout to dialog
        self.setLayout(main_layout)
        
        # Debug: Check dialog properties
        print(f"[Upload] 🔍 Dialog size: {self.size()}")
        print(f"[Upload] 🔍 Dialog geometry: {self.geometry()}")
        print(f"[Upload] 🔍 Dialog is visible: {self.isVisible()}")
        print(f"[Upload] 🔍 Dialog is shown: {self.isVisible()}")
        print(f"[Upload] 🔍 Content widget size: {content_widget.size()}")
        print(f"[Upload] 🔍 Content widget geometry: {content_widget.geometry()}")
        print(f"[Upload] 🔍 Content widget is visible: {content_widget.isVisible()}")
        
        
        # Edge buttons removed - separate GUI functions will have their own scripts
    
    def select_files(self):
        """Open file dialog to select multiple files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents to Upload",
            "",
            "All Supported (*.pdf *.txt *.docx *.md *.xlsx *.xls);;PDF Files (*.pdf);;Text Files (*.txt);;Word Documents (*.docx);;Markdown (*.md);;Excel Files (*.xlsx *.xls)"
        )
        
        if files:
            for file_path in files:
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
                    filename = os.path.basename(file_path)
                    # Choose icon based on file type
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.xlsx', '.xls']:
                        icon = "📊"
                    elif file_ext == '.pdf':
                        icon = "📕"
                    elif file_ext in ['.docx', '.doc']:
                        icon = "📘"
                    elif file_ext in ['.md']:
                        icon = "📝"
                    else:
                        icon = "📄"
                    item = QListWidgetItem(f"{icon} {filename}")
                    item.setData(Qt.UserRole, file_path)
                    self.file_list.addItem(item)
            
            self.upload_btn.setEnabled(len(self.uploaded_files) > 0)
            self.log_status(f"Selected {len(files)} file(s)")
    
    def clear_files(self):
        """Clear all files from the upload list"""
        self.uploaded_files.clear()
        self.file_list.clear()
        self.upload_btn.setEnabled(False)
        self.log_status("Cleared all files")
        
    def add_google_drive(self):
        """Add document from Google Drive"""
        # Show options for Google Drive integration
        options = [
            "📋 Paste Google Drive Share Link",
            "🔑 Authenticate with Google Drive API",
            "📁 Browse Google Drive (requires auth)"
        ]
        
        option, ok = QInputDialog.getItem(
            self,
            "Google Drive Integration",
            "Choose how to access Google Drive:",
            options,
            0,
            False
        )
        
        if ok and option:
            if "Share Link" in option:
                self.add_google_drive_link()
            elif "Authenticate" in option:
                self.authenticate_google_drive()
            elif "Browse" in option:
                self.browse_google_drive()
    
    def add_google_drive_link(self):
        """Add document from Google Drive share link"""
        url, ok = QInputDialog.getText(
            self,
            "Google Drive Share Link",
            "Paste Google Drive share link:",
            text="https://drive.google.com/file/d/"
        )
        
        if ok and url.strip():
            # Convert Google Drive share link to direct download link
            if "drive.google.com/file/d/" in url:
                file_id = url.split("drive.google.com/file/d/")[1].split("/")[0]
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                self.uploaded_files.append(direct_url)
                filename = f"gdrive_document_{file_id[:8]}.pdf"
                
                item = QListWidgetItem(f"☁️ {filename} (Google Drive)")
                item.setData(Qt.UserRole, direct_url)
                self.file_list.addItem(item)
                
                self.upload_btn.setEnabled(len(self.uploaded_files) > 0)
                self.log_status(f"Added Google Drive file: {filename}")
            else:
                QMessageBox.warning(self, "Invalid Link", "Please provide a valid Google Drive share link.")
    
    def authenticate_google_drive(self):
        """Authenticate with Google Drive API"""
        QMessageBox.information(
            self,
            "Google Drive Authentication",
            "Google Drive API authentication requires:\n\n"
            "1. Google Cloud Console project\n"
            "2. Drive API enabled\n"
            "3. OAuth2 credentials\n\n"
            "This feature will be implemented in a future update.\n"
            "For now, use the 'Share Link' option."
        )
    
    def browse_google_drive(self):
        """Browse Google Drive files"""
        QMessageBox.information(
            self,
            "Google Drive Browser",
            "Google Drive file browser requires authentication.\n\n"
            "This feature will be implemented in a future update.\n"
            "For now, use the 'Share Link' option."
        )
    
    def show_qr_code(self):
        """Show QR code for web upload interface"""
        local_ip = get_local_ip()
        upload_url = f"http://{local_ip}:5001"
        
        # Check if upload server is running
        try:
            response = requests.get(f"{upload_url}/api/status", timeout=2)
            if response.status_code == 200:
                # Server is running, show QR code - optimized for circular screen
                # Generate QR code locally instead of using external API
                try:
                    import qrcode
                    from PIL import Image
                    import io
                    
                    # Create QR code
                    qr = qrcode.QRCode(
                        version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_L,
                        box_size=10,
                        border=4,
                    )
                    qr.add_data(upload_url)
                    qr.make(fit=True)
                    
                    # Create QR code image
                    qr_image = qr.make_image(fill_color="black", back_color="white")
                    
                    # Convert to QPixmap
                    from PyQt5.QtGui import QPixmap
                    import tempfile
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        qr_image.save(tmp_file.name)
                        qr_pixmap = QPixmap(tmp_file.name)
                        os.unlink(tmp_file.name)  # Clean up temp file
                        
                except ImportError:
                    QMessageBox.warning(self, "QR Code", "QR code library not available. Please install: pip install qrcode[pil]")
                    return
                
                # Create a new dialog to show QR code - full screen for circular screen
                qr_dialog = QDialog(self)
                qr_dialog.setWindowTitle("📱 Mobile Upload QR Code")
                # Use Window flag to ensure proper z-ordering above parent dialog
                qr_dialog.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # Full screen
                qr_dialog.setStyleSheet("""
                    QDialog {
                        background-color: rgba(28, 28, 30, 1.0);
                        color: white;
                        border: none;
                        border-radius: 535px;
                    }
                    QLabel {
                        color: white;
                        font-size: 12px;
                    }
                    QPushButton {
                        background-color: rgba(142, 142, 147, 0.2);
                        color: #ffffff;
                        font-size: 12px;
                        font-weight: 500;
                        padding: 10px 15px;
                        min-height: 35px;
                        border-radius: 18px;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: rgba(142, 142, 147, 0.3);
                    }
                    QPushButton:pressed {
                        background-color: rgba(142, 142, 147, 0.4);
                    }
                """)
                
                layout = QVBoxLayout()
                # Symmetric margins to center content within circular red border
                # Account for 8px red border on the QR dialog itself
                layout.setContentsMargins(120, 120, 120, 120)  # Centered within circular area
                layout.setSpacing(15)  # Compact spacing
                
                # Add top stretch for vertical centering
                layout.addStretch(1)
                
                # Title - compact for circular screen
                title = QLabel("📱 Mobile Upload")
                title.setFont(QFont("Arial", 16, QFont.Bold))
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 15px; font-size: 16px;")
                layout.addWidget(title)
                
                # Instructions - compact
                instructions = QLabel("Scan QR code with your phone:")
                instructions.setAlignment(Qt.AlignCenter)
                instructions.setStyleSheet("color: #8e8e93; font-size: 13px; margin: 8px;")
                layout.addWidget(instructions)
                
                # QR Code (display the generated QR code)
                qr_label = QLabel()
                qr_label.setAlignment(Qt.AlignCenter)
                qr_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(44, 44, 46, 0.8);
                        border-radius: 15px;
                        border: none;
                        padding: 20px;
                    }
                """)
                qr_label.setPixmap(qr_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                layout.addWidget(qr_label)
                
                # URL
                url_label = QLabel(f"🌐 {upload_url}")
                url_label.setAlignment(Qt.AlignCenter)
                url_label.setStyleSheet("color: #007AFF; font-size: 14px; font-weight: 500; margin: 10px;")
                layout.addWidget(url_label)
                
                # Buttons
                button_layout = QHBoxLayout()
                
                open_btn = QPushButton("🌐 Open in Browser")
                open_btn.clicked.connect(lambda: webbrowser.open(upload_url))
                open_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #007AFF;
                        color: white;
                        font-size: 12px;
                        font-weight: 600;
                        padding: 8px 16px;
                        border-radius: 15px;
                        border: none;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #0056CC;
                    }
                """)
                button_layout.addWidget(open_btn)
                
                close_btn = QPushButton("❌ Close")
                close_btn.clicked.connect(qr_dialog.accept)
                close_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FF3B30;
                        color: white;
                        font-size: 12px;
                        font-weight: 600;
                        padding: 8px 16px;
                        border-radius: 15px;
                        border: none;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #D70015;
                    }
                """)
                button_layout.addWidget(close_btn)
                
                layout.addLayout(button_layout)
                
                # Add bottom stretch for vertical centering
                layout.addStretch(1)
                
                qr_dialog.setLayout(layout)
                
                # Center the QR dialog within the circular display
                # Get parent window geometry for proper centering
                if self.parent():
                    parent_rect = self.parent().geometry()
                    dialog_size = min(parent_rect.width(), parent_rect.height())
                    qr_dialog.setFixedSize(dialog_size, dialog_size)
                    
                    # Center relative to parent
                    x = parent_rect.x() + (parent_rect.width() - dialog_size) // 2
                    y = parent_rect.y() + (parent_rect.height() - dialog_size) // 2
                else:
                    # No parent, center on screen
                    screen = QApplication.primaryScreen().availableGeometry()
                    dialog_size = min(screen.width(), screen.height())
                    qr_dialog.setFixedSize(dialog_size, dialog_size)
                    x = (screen.width() - dialog_size) // 2
                    y = (screen.height() - dialog_size) // 2
                
                qr_dialog.move(x, y)
                print(f"[Upload] 🔍 QR dialog centered: size={dialog_size}x{dialog_size}, position=({x}, {y})")
                qr_dialog.show()
                qr_dialog.raise_()
                qr_dialog.activateWindow()
                
                qr_dialog.exec_()
                
            else:
                QMessageBox.warning(self, "Upload Server", "Upload server is not responding properly.")
        except requests.exceptions.RequestException:
            QMessageBox.warning(self, "Upload Server", 
                              "Upload server is not running.\n\n"
                              "To enable web upload:\n"
                              "1. Install Flask: pip install flask\n"
                              "2. Restart Aura system\n"
                              "3. The upload server will start automatically")
    
    def clear_files(self):
        """Clear all selected files"""
        self.uploaded_files.clear()
        self.file_list.clear()
        self.upload_btn.setEnabled(False)
        self.log_status("Cleared all files")
    
    def log_status(self, message):
        """Add message to status log"""
        self.status_log.append(f"[{self.get_timestamp()}] {message}")
        self.status_log.ensureCursorVisible()
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def upload_files(self):
        """Upload selected files to the system"""
        if not self.uploaded_files:
            QMessageBox.warning(self, "No Files", "Please select files to upload.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.uploaded_files))
        self.progress_bar.setValue(0)
        self.upload_btn.setEnabled(False)
        
        # Start upload process
        self.upload_worker = UploadWorker(self.uploaded_files)
        self.upload_worker.progress.connect(self.progress_bar.setValue)
        self.upload_worker.status.connect(self.log_status)
        self.upload_worker.finished.connect(self.upload_finished)
        self.upload_worker.start()
    
    def upload_finished(self, success_count, error_count):
        """Handle upload completion"""
        self.progress_bar.setVisible(False)
        self.upload_btn.setEnabled(True)
        
        if success_count > 0:
            self.log_status(f"✅ Successfully uploaded {success_count} file(s)")
        if error_count > 0:
            self.log_status(f"❌ Failed to upload {error_count} file(s)")
        
        # Show completion message
        if success_count > 0:
            QMessageBox.information(
                self, 
                "Upload Complete", 
                f"Successfully uploaded {success_count} file(s).\n"
                f"Documents are being processed and will be available in the RAG system shortly."
            )
        
        # Clear files after successful upload
        if success_count > 0:
            self.clear_files()

class UploadWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    
    def __init__(self, files):
        super().__init__()
        self.files = files
    
    def run(self):
        """Upload files to the data/input directory"""
        success_count = 0
        error_count = 0
        
        # Ensure data/input directory exists at workspace root
        # From aura-control/gui/ we need to go up 2 levels to workspace root
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        input_dir = os.path.join(workspace_root, 'data', 'input')
        os.makedirs(input_dir, exist_ok=True)
        
        for i, file_path in enumerate(self.files):
            try:
                if file_path.startswith(('http://', 'https://')):
                    # Download from URL
                    self.status.emit(f"🌐 Downloading from URL...")
                    response = requests.get(file_path, timeout=30, stream=True)
                    response.raise_for_status()
                    
                    # Get filename from URL or use default
                    filename = os.path.basename(urllib.parse.urlparse(file_path).path)
                    if not filename or '.' not in filename:
                        # Try to get filename from Content-Disposition header
                        content_disposition = response.headers.get('Content-Disposition', '')
                        if 'filename=' in content_disposition:
                            filename = content_disposition.split('filename=')[1].strip('"')
                        else:
                            filename = "downloaded_document.pdf"
                    
                    dest_path = os.path.join(input_dir, filename)
                    
                    # Download file
                    with open(dest_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    self.status.emit(f"🌐 Downloaded {filename} from URL")
                    success_count += 1
                    
                else:
                    # Copy local file
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(input_dir, filename)
                    
                    shutil.copy2(file_path, dest_path)
                    self.status.emit(f"📄 Copied {filename} to data/input")
                    success_count += 1
                
            except Exception as e:
                self.status.emit(f"❌ Error processing {file_path}: {str(e)}")
                error_count += 1
            
            self.progress.emit(i + 1)
        
        # Auto-ingest will handle the processing automatically
        self.status.emit("✅ Files uploaded to data/input - auto-ingest will process them automatically")
        
        self.finished.emit(success_count, error_count)

# Global variable to prevent multiple dialogs
_current_dialog = None

def show_upload_dialog():
    """Show the file upload dialog"""
    global _current_dialog
    
    # Prevent multiple dialogs
    if _current_dialog is not None:
        print("[Upload] ⚠️ Dialog already open, bringing to front...")
        _current_dialog.raise_()
        _current_dialog.activateWindow()
        return
    
    print("[Upload] 🚀 Opening upload dialog...")
    
    try:
        _current_dialog = FileUploadDialog()
        print("[Upload] 📱 Dialog created, showing...")
        
        # Ensure dialog is visible and on top
        _current_dialog.show()
        _current_dialog.raise_()
        _current_dialog.activateWindow()
        
        # Process events to ensure dialog is rendered
        QApplication.processEvents()
        
        print("[Upload] 👁️ Dialog is now visible, waiting for user interaction...")
        _current_dialog.exec_()
        print("[Upload] ✅ Dialog closed")
    except Exception as e:
        print(f"[Upload] ❌ Dialog error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _current_dialog = None
