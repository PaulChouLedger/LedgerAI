# file_upload_dialog.py — File Upload Dialog for Document Ingestion

import os
import sys
import shutil
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import requests
import json

class FileUploadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Document Upload - AuraVision")
        self.setFixedSize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 8px;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
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
        
        # Title
        title = QLabel("📄 Document Upload for RAG System")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Upload documents to data/input directory - auto-ingest will process them automatically")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(desc)
        
        # File selection area
        file_layout = QHBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_layout.addWidget(self.file_list)
        
        button_layout = QVBoxLayout()
        
        self.select_files_btn = QPushButton("📁 Select Files")
        self.select_files_btn.clicked.connect(self.select_files)
        button_layout.addWidget(self.select_files_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(self.clear_btn)
        
        file_layout.addLayout(button_layout)
        layout.addLayout(file_layout)
        
        # Upload progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status log
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(120)
        self.status_log.setPlaceholderText("Upload status will appear here...")
        layout.addWidget(self.status_log)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("🚀 Upload & Process")
        self.upload_btn.clicked.connect(self.upload_files)
        self.upload_btn.setEnabled(False)
        button_layout.addWidget(self.upload_btn)
        
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.clicked.connect(self.close)
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
                filename = os.path.basename(file_path)
                dest_path = os.path.join(input_dir, filename)
                
                # Copy file to data/input
                shutil.copy2(file_path, dest_path)
                
                self.status.emit(f"📄 Copied {filename} to data/input")
                success_count += 1
                
            except Exception as e:
                self.status.emit(f"❌ Error copying {os.path.basename(file_path)}: {str(e)}")
                error_count += 1
            
            self.progress.emit(i + 1)
        
        # Auto-ingest will handle the processing automatically
        self.status.emit("✅ Files uploaded to data/input - auto-ingest will process them automatically")
        
        self.finished.emit(success_count, error_count)

def show_upload_dialog():
    """Show the file upload dialog"""
    dialog = FileUploadDialog()
    dialog.exec_()
