# file_upload_dialog.py — File Upload Dialog for Document Ingestion

import os
import sys
import shutil
import urllib.parse
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem, QInputDialog,
                             QLineEdit)
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
        # Test with smaller size first
        self.setFixedSize(800, 800)  # Smaller size for testing
        self.setWindowFlags(Qt.FramelessWindowHint)  # Remove title bar but allow normal positioning
        self.move(100, 100)  # Position away from edge
        print("[Upload] 📐 Dialog size set to 1080x1080, positioned at (0,0)")
        self.show()  # Show the dialog
        print("[Upload] 👁️ Dialog should now be visible")
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: white;
                border-radius: 400px;  /* Circle for 800x800 */
                border: 5px solid #4CAF50;  /* Green border for visibility */
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                padding: 12px 20px;
                color: white;
                border-radius: 25px;  /* More rounded for circular theme */
                font-weight: bold;
                font-size: 13px;
                min-height: 45px;
                min-width: 100px;
                max-width: 150px;  /* Limit width for circular layout */
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
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
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(80, 80, 80, 80)  # Larger margins for circular screen
        layout.setSpacing(20)  # Spacing for full screen
        
        # Title with close button for full screen
        title_layout = QHBoxLayout()
        
        # Close button in top-right
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(40, 40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        
        # Spacer to push title to center
        title_layout.addStretch()
        
        # Title
        title = QLabel("📄 Document Upload")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin: 10px;")
        title_layout.addWidget(title)
        
        # Spacer to balance layout
        title_layout.addStretch()
        
        # Invisible spacer to balance the close button
        invisible_spacer = QLabel("")
        invisible_spacer.setFixedSize(40, 40)
        title_layout.addWidget(invisible_spacer)
        
        layout.addLayout(title_layout)
        
        # Description - more compact
        desc = QLabel("Upload docs to data/input - auto-ingest processes them")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #aaa; font-size: 11px; margin: 5px;")
        layout.addWidget(desc)
        
        # File selection area - optimized for circular screen
        file_layout = QHBoxLayout()
        file_layout.setSpacing(20)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(200)  # Reduced for circular screen
        self.file_list.setMinimumHeight(150)
        self.file_list.setMaximumWidth(400)  # Limit width for circular layout
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
        
        file_layout.addLayout(button_layout)
        layout.addLayout(file_layout)
        
        # Upload progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status log - compact for circular screen
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)  # Reduced for circular screen
        self.status_log.setMinimumHeight(80)
        self.status_log.setMaximumWidth(500)  # Limit width
        self.status_log.setReadOnly(True)
        self.status_log.setPlaceholderText("Upload status will appear here...")
        self.status_log.setStyleSheet("border-radius: 15px; font-size: 11px;")
        layout.addWidget(self.status_log)
        
        # Action buttons - optimized for circular screen
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)  # Compact spacing
        
        self.upload_btn = QPushButton("🚀 Upload")
        self.upload_btn.clicked.connect(self.upload_files)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-size: 14px;
                padding: 15px 25px;
                min-height: 50px;
                min-width: 120px;
                max-width: 180px;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        button_layout.addWidget(self.upload_btn)
        
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                font-size: 14px;
                padding: 15px 25px;
                min-height: 50px;
                min-width: 120px;
                max-width: 150px;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
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
                qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={upload_url}"
                
                # Create a new dialog to show QR code - full screen for circular screen
                qr_dialog = QDialog(self)
                qr_dialog.setWindowTitle("📱 Mobile Upload QR Code")
                qr_dialog.setFixedSize(1080, 1080)  # Full screen for circular screen
                qr_dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # Full screen
                qr_dialog.showMaximized()  # Ensure it fills the screen
                qr_dialog.setStyleSheet("""
                    QDialog {
                        background-color: #1a1a1a;
                        color: white;
                        border-radius: 540px;  /* Perfect circle for 1080x1080 */
                        border: none;  /* No border for full screen */
                    }
                    QLabel {
                        color: white;
                        font-size: 12px;
                    }
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                
                layout = QVBoxLayout()
                layout.setContentsMargins(80, 80, 80, 80)  # Larger margins for full screen
                layout.setSpacing(20)
                
                # Title - compact for circular screen
                title = QLabel("📱 Mobile Upload")
                title.setFont(QFont("Arial", 14, QFont.Bold))
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("color: #4CAF50; margin: 10px;")
                layout.addWidget(title)
                
                # Instructions - compact
                instructions = QLabel("Scan QR code with your phone:")
                instructions.setAlignment(Qt.AlignCenter)
                instructions.setStyleSheet("color: #aaa; font-size: 11px; margin: 5px;")
                layout.addWidget(instructions)
                
                # QR Code (using a label with HTML to display image)
                qr_label = QLabel()
                qr_label.setAlignment(Qt.AlignCenter)
                qr_label.setStyleSheet("border: 2px solid #333; border-radius: 10px; margin: 10px;")
                qr_label.setText(f'<img src="{qr_url}" width="300" height="300">')
                layout.addWidget(qr_label)
                
                # URL
                url_label = QLabel(f"🌐 {upload_url}")
                url_label.setAlignment(Qt.AlignCenter)
                url_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold; margin: 10px;")
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

def show_upload_dialog():
    """Show the file upload dialog"""
    print("[Upload] 🚀 Opening upload dialog...")
    dialog = FileUploadDialog()
    print("[Upload] 📱 Dialog created, showing...")
    dialog.exec_()
    print("[Upload] ✅ Dialog closed")
