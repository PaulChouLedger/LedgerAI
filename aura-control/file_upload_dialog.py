# file_upload_dialog.py — File Upload Dialog for Document Ingestion

import os
import sys
import shutil
import urllib.parse
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem, QInputDialog,
                             QLineEdit, QWidget, QApplication, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import requests
import json
import socket
import webbrowser

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

class FileUploadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        print("[Upload] 🔧 Initializing upload dialog...")
        self.setWindowTitle("Document Upload - AuraVision")
        # Full screen for 5-inch 1080x1080 circular screen
        self.setFixedSize(1080, 1080)  # Full screen size
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # Frameless, stay on top
        
        # Center the dialog on the actual screen
        screen = self.screen()
        # Show dialog first, then center it after it's rendered
        self.show()  # Show the dialog first
        self.raise_()  # Bring to front
        self.activateWindow()  # Activate the window
        
        # Center dialog after it's been rendered
        QApplication.processEvents()  # Process events to ensure dialog is rendered
        self.center_dialog()
        
        # Debug: Print dialog dimensions and position
        print(f"[Upload] 📐 Dialog geometry: {self.geometry()}")
        print(f"[Upload] 📐 Dialog size: {self.size()}")
        print(f"[Upload] 📐 Dialog position: {self.pos()}")
        print("[Upload] 👁️ Dialog should now be visible with transparent background")
        
        # Add interactive touch coordinates for debugging
        self.touch_coordinates = []
        self.setMouseTracking(True)
        print("[Upload] 🖱️ Touch the center of the screen to get coordinates for debugging")
        print("[Upload] 📝 Touch coordinates will appear in the console output")
        print("[Upload] 🎯 Touch center, top, right, bottom, left edges to get all coordinates")
        self.setStyleSheet("""
            QDialog {
                background-color: transparent;  /* Transparent background */
                color: white;
                border: none;  /* No border */
                border-radius: 540px;  /* Circular border to match 5-inch screen */
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
        """)
        
        self.setup_ui()
        self.uploaded_files = []
        
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
        """Center the dialog dynamically on the screen using proper PyQt methods"""
        from PyQt5.QtWidgets import QDesktopWidget
        
        # Get screen geometry
        screen = QDesktopWidget().screenGeometry()
        print(f"[Upload] 🔍 Screen geometry: {screen.width()}x{screen.height()}")
        
        # Get dialog size (use fixed size since we set it to 1080x1080)
        dialog_width = 1080
        dialog_height = 1080
        print(f"[Upload] 🔍 Dialog size: {dialog_width}x{dialog_height}")
        
        # Calculate center position dynamically
        x = (screen.width() - dialog_width) // 2
        y = (screen.height() - dialog_height) // 2
        
        # Ensure dialog doesn't go off-screen
        x = max(0, x)
        y = max(0, y)
        
        print(f"[Upload] 📐 Calculated center position: ({x}, {y})")
        
        # Move to center
        self.move(x, y)
        print(f"[Upload] ✅ Dialog centered at ({x}, {y})")
        
        # Verify final position
        final_pos = self.pos()
        print(f"[Upload] 📐 Final dialog position: ({final_pos.x()}, {final_pos.y()})")
        
        # Ensure dialog is visible and active
        self.raise_()
        self.activateWindow()

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
        
        # Create circular content container
        content_widget = QWidget()
        content_widget.setFixedSize(1078, 1078)  # 10% bigger (980 * 1.1 = 1078)
        content_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(28, 28, 30, 0.95);
                border-radius: 539px;
                border: none;
            }
        """)
        
        # Use simple centering with equal stretch
        main_layout.addStretch()
        main_layout.addWidget(content_widget, 0, Qt.AlignCenter)
        main_layout.addStretch()
        
        # Add edge buttons for future functions - will be added after layout is set
        
        # Create layout for circular content
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(110, 110, 110, 110)  # 10% bigger margins (100 * 1.1 = 110)
        self.content_layout.setSpacing(28)  # 10% bigger spacing (25 * 1.1 = 27.5, rounded to 28)
        
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
        self.file_list.setMaximumHeight(308)  # 10% bigger (280 * 1.1 = 308)
        self.file_list.setMinimumHeight(253)  # 10% bigger (230 * 1.1 = 253)
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
        file_layout.addWidget(self.file_list)
        
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)  # Compact spacing for circular screen
        
        self.select_files_btn = QPushButton("📁 Select Files")
        self.select_files_btn.clicked.connect(self.select_files)
        button_layout.addWidget(self.select_files_btn)
        
        self.url_btn = QPushButton("🌐 Add URL")
        self.url_btn.clicked.connect(self.add_url)
        button_layout.addWidget(self.url_btn)
        
        self.gdrive_btn = QPushButton("☁️ Google Drive")
        self.gdrive_btn.clicked.connect(self.add_google_drive)
        button_layout.addWidget(self.gdrive_btn)
        
        self.qr_btn = QPushButton("📱 Show QR Code")
        self.qr_btn.clicked.connect(self.show_qr_code)
        button_layout.addWidget(self.qr_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(self.clear_btn)
        
        # Apply Apple-style styling to all action buttons
        action_button_style = """
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
                background-color: rgba(142, 142, 147, 0.4);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.6);
            }
        """
        
        for button in [self.select_files_btn, self.url_btn, self.gdrive_btn, self.qr_btn, self.clear_btn]:
            button.setStyleSheet(action_button_style)
        
        file_layout.addLayout(button_layout)
        self.content_layout.addLayout(file_layout)
        
        # Upload progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.content_layout.addWidget(self.progress_bar)
        
        # Status log - increased for even larger circle
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(154)  # 10% bigger (140 * 1.1 = 154)
        self.status_log.setMinimumHeight(132)  # 10% bigger (120 * 1.1 = 132)
        self.status_log.setMaximumWidth(660)  # 10% bigger (600 * 1.1 = 660)
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
        
        # Simple working buttons
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.clicked.connect(self.upload_files)
        self.upload_btn.setEnabled(False)
        self.content_layout.addWidget(self.upload_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.content_layout.addWidget(self.close_btn)
        
        # Set layout to content widget
        content_widget.setLayout(self.content_layout)
        
        # Set main layout to dialog
        self.setLayout(main_layout)
        
        
        # Edge buttons removed - separate GUI functions will have their own scripts
    
        
    def select_files(self):
        """Open file dialog to select multiple files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents to Upload",
            "",
            "All Supported (*.pdf *.txt *.docx *.md);;PDF Files (*.pdf);;Text Files (*.txt);;Word Documents (*.docx);;Markdown (*.md)"
        )
        
        if files:
            for file_path in files:
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
                    filename = os.path.basename(file_path)
                    item = QListWidgetItem(f"📄 {filename}")
                    item.setData(Qt.UserRole, file_path)
                    self.file_list.addItem(item)
            
            self.upload_btn.setEnabled(len(self.uploaded_files) > 0)
            self.log_status(f"Selected {len(files)} file(s)")
    
    def add_url(self):
        """Add document from URL"""
        url, ok = QInputDialog.getText(
            self, 
            "Add Document from URL", 
            "Enter document URL:",
            text="https://"
        )
        
        if ok and url.strip():
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Add to list immediately
            self.uploaded_files.append(url)
            filename = os.path.basename(urllib.parse.urlparse(url).path) or "document"
            if not filename or '.' not in filename:
                filename += ".pdf"  # Default extension
            
            item = QListWidgetItem(f"🌐 {filename} ({url[:50]}...)")
            item.setData(Qt.UserRole, url)
            self.file_list.addItem(item)
            
            self.upload_btn.setEnabled(len(self.uploaded_files) > 0)
            self.log_status(f"Added URL: {url}")
    
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
                qr_dialog.setFixedSize(1080, 1080)  # Full screen for circular screen
                qr_dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # Full screen
                qr_dialog.setStyleSheet("""
                    QDialog {
                        background-color: rgba(28, 28, 30, 1.0);
                        color: white;
                        border-radius: 540px;
                        border: none;
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
                layout.setContentsMargins(80, 80, 80, 80)  # Larger margins for full screen
                layout.setSpacing(20)
                
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
                button_layout.addWidget(open_btn)
                
                close_btn = QPushButton("❌ Close")
                close_btn.clicked.connect(qr_dialog.accept)
                button_layout.addWidget(close_btn)
                
                layout.addLayout(button_layout)
                qr_dialog.setLayout(layout)
                
                # Center the QR dialog using the same method as main dialog
                qr_dialog.show()
                QApplication.processEvents()
                
                # Center dynamically
                from PyQt5.QtWidgets import QDesktopWidget
                screen = QDesktopWidget().screenGeometry()
                x = (screen.width() - 1080) // 2
                y = (screen.height() - 1080) // 2
                x = max(0, x)
                y = max(0, y)
                qr_dialog.move(x, y)
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
        
        # Ensure data/input directory exists
        input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'input'))
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
        _current_dialog.exec_()
        print("[Upload] ✅ Dialog closed")
    except Exception as e:
        print(f"[Upload] ❌ Dialog error: {e}")
    finally:
        _current_dialog = None
