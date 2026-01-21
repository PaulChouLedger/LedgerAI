# file_upload_dialog.py — Memory Dialog for Conversation Management

import os
import sys
import shutil
import urllib.parse
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem, QInputDialog,
                             QLineEdit, QWidget, QApplication, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor
import requests
import json
import socket
import webbrowser

# Import base dialog template
from gui.base_dialog import BaseAuraDialog

class RAGFilesDialog(BaseAuraDialog):
    """Dialog showing files actively being used by RAG"""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="📚 RAG Files Status", size=(1080, 1080), modal=True)
        # Initialize file item mapping
        self.file_item_map = {}
    
    def _setup_ui(self):
        """Setup UI - called by BaseAuraDialog"""
        layout = QVBoxLayout()
        # Use base class default margins to ensure content fits within white perimeter
        margins = BaseAuraDialog.get_default_margins()
        layout.setContentsMargins(*margins)
        layout.setSpacing(BaseAuraDialog.get_default_spacing())
        
        # Add top stretch to center content vertically within white perimeter
        layout.addStretch(1)
        
        # Title
        title = QLabel("📚 Files in RAG System")
        title.setFont(QFont("Arial", 15, QFont.Bold))  # Further reduced font size
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 5px; font-size: 15px;")  # Reduced margins
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Files currently being used by RAG")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 3px;")  # Shorter text, smaller font
        layout.addWidget(desc)
        
        # File list (selectable for deletion)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.setMaximumHeight(450)  # Reduced to make room for remove button
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(44, 44, 46, 0.8);
                border: 1px solid #555;
                border-radius: 10px;
                color: white;
                font-size: 12px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        layout.addWidget(self.file_list)
        
        # Remove button (hidden until file is selected)
        self.remove_btn = QPushButton("🗑️ Remove Selected File")
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 30px;
                border-radius: 15px;
                border: none;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B00012;
            }
            QPushButton:disabled {
                background-color: rgba(142, 142, 147, 0.3);
                color: rgba(142, 142, 147, 0.5);
            }
        """)
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._remove_selected_file)
        layout.addWidget(self.remove_btn)
        
        # Load RAG files (file_item_map is initialized in __init__)
        self._load_rag_files()
        
        # Close button - centered and wider
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("❌ Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 15px 40px;
                border-radius: 20px;
                border: none;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B00012;
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Add bottom stretch to center content vertically within white perimeter
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def _load_rag_files(self):
        """Load files that are actively being used by RAG"""
        # Clear existing mapping and list
        self.file_item_map.clear()
        self.file_list.clear()
        
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        embeddings_dir = os.path.join(workspace_root, 'data', 'embeddings')
        input_dir = os.path.join(workspace_root, 'data', 'input')
        
        files_in_rag = {}
        total_chunks = 0
        
        # Get RAG_MODE from settings file first, then fall back to environment variable
        RAG_MODE = None
        try:
            settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
            if os.path.exists(settings_path):
                import json
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    RAG_MODE = settings.get('rag_mode', '').upper()
        except:
            pass
        
        # Fall back to environment variable if not in settings
        if not RAG_MODE:
            RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
        
        # Try to get RAG stats from API first (for GPU mode only, CPU mode reads from metadata)
        if RAG_MODE == 'GPU':
            try:
                import requests
                response = requests.get("http://localhost:11435/rag/stats", timeout=2)
                if response.status_code == 200:
                    stats = response.json()
                    total_chunks = stats.get('chunks_loaded', 0)
                    # Try to get file list from metadata
                    if 'files' in stats:
                        for file_info in stats['files']:
                            filename = file_info.get('name', 'Unknown')
                            chunks = file_info.get('chunks', 0)
                            files_in_rag[filename] = chunks
            except:
                pass
        
        # Check local metadata files (for CPU mode, this is the primary source)
        # For GPU mode, only use this if API didn't return file list
        metadata_file = os.path.join(embeddings_dir, "metadata.pkl")
        if os.path.exists(metadata_file) and (RAG_MODE == 'CPU' or not files_in_rag):
            try:
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    
                    # Reset counters to avoid double-counting
                    if RAG_MODE == 'CPU':
                        files_in_rag = {}
                        total_chunks = 0
                    
                    # CPU FAISS stores metadata as a dict with 'chunks' and 'metadata' keys
                    if isinstance(metadata, dict):
                        chunk_metadata = metadata.get('metadata', [])
                        if chunk_metadata:
                            # Count chunks per file correctly
                            for chunk_meta in chunk_metadata:
                                if isinstance(chunk_meta, dict):
                                    # Try different possible keys for file path/name
                                    file_path = chunk_meta.get('file_path', '')
                                    doc_name = chunk_meta.get('document_name', '') or chunk_meta.get('guideline_name', '')
                                    
                                    if file_path:
                                        filename = os.path.basename(file_path)
                                    elif doc_name:
                                        filename = doc_name
                                    else:
                                        continue
                                    
                                    # Count chunks per file
                                    if filename not in files_in_rag:
                                        files_in_rag[filename] = 0
                                    files_in_rag[filename] += 1
                            
                            # Total chunks is the length of chunk_metadata (each entry is one chunk)
                            total_chunks = len(chunk_metadata)
                    # Legacy format: metadata is a list
                    elif isinstance(metadata, list):
                        # Reset if CPU mode
                        if RAG_MODE == 'CPU':
                            files_in_rag = {}
                            total_chunks = 0
                        
                        for chunk_meta in metadata:
                            if isinstance(chunk_meta, dict):
                                source = chunk_meta.get('source') or chunk_meta.get('file_path') or chunk_meta.get('document_name', 'Unknown')
                                filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
                                if filename != 'Unknown':
                                    if filename not in files_in_rag:
                                        files_in_rag[filename] = 0
                                    files_in_rag[filename] += 1
                        
                        # Total chunks is the length of the metadata list
                        total_chunks = len(metadata)
            except Exception as e:
                print(f"[RAGFilesDialog] ⚠️ Error reading metadata: {e}")
                import traceback
                traceback.print_exc()
        
        # Display files
        if files_in_rag:
            # Sort by chunk count (descending)
            sorted_files = sorted(files_in_rag.items(), key=lambda x: x[1], reverse=True)
            
            for filename, chunks in sorted_files:
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Choose icon based on file type
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
                
                # Get file size if file exists
                file_path = os.path.join(input_dir, filename)
                size_str = ""
                if os.path.exists(file_path):
                    try:
                        size_bytes = os.path.getsize(file_path)
                        if size_bytes < 1024:
                            size_str = f" ({size_bytes}B)"
                        elif size_bytes < 1024 * 1024:
                            size_str = f" ({size_bytes / 1024:.1f}KB)"
                        else:
                            size_str = f" ({size_bytes / (1024 * 1024):.1f}MB)"
                    except:
                        pass
                
                # Truncate long filenames to fit within dialog
                max_filename_len = 30  # Reduced for better fit
                display_filename = filename if len(filename) <= max_filename_len else filename[:max_filename_len-3] + "..."
                # Compact item text
                item_text = f"{icon} {display_filename}{size_str} - {chunks} chunks"
                item = QListWidgetItem(item_text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # Selectable
                item.setToolTip(filename)  # Show full filename on hover
                self.file_list.addItem(item)
                # Store mapping from item to filename
                self.file_item_map[item] = filename
            
            # Add summary - compact format (not selectable)
            summary_item = QListWidgetItem(f"\n📊 {len(files_in_rag)} file(s), {total_chunks} chunk(s)")
            summary_item.setFlags(Qt.NoItemFlags)
            summary_item.setForeground(QColor(142, 142, 147))  # Gray color
            self.file_list.addItem(summary_item)
            
            # Store files dict for later use
            self.files_in_rag = files_in_rag
        else:
            # No files in RAG
            no_files_item = QListWidgetItem("📭 No files currently in RAG system")
            no_files_item.setFlags(Qt.NoItemFlags)
            self.file_list.addItem(no_files_item)
            
            info_item = QListWidgetItem("Upload files via Google Drive or QR code to add them to RAG")
            info_item.setFlags(Qt.NoItemFlags)
            info_item.setForeground(QColor(142, 142, 147))
            self.file_list.addItem(info_item)
    
    def _trigger_ingestion(self):
        """Manually trigger RAG ingestion for files in data/input"""
        try:
            import requests
            
            # Get RAG_MODE from settings file first, then fall back to environment variable
            RAG_MODE = None
            try:
                settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
                if os.path.exists(settings_path):
                    import json
                    with open(settings_path, 'r') as f:
                        settings = json.load(f)
                        RAG_MODE = settings.get('rag_mode', '').upper()
            except:
                pass
            
            # Fall back to environment variable if not in settings
            if not RAG_MODE:
                RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
            
            # Show processing message
            processing_item = QListWidgetItem("🔄 Processing files...")
            processing_item.setFlags(Qt.NoItemFlags)
            processing_item.setForeground(QColor(255, 193, 7))  # Yellow
            self.file_list.insertItem(0, processing_item)
            self.file_list.scrollToTop()
            
            # Trigger ingestion based on RAG_MODE
            if RAG_MODE == 'CPU':
                # CPU mode: Use CPU FAISS in LLM container (both medical and generic use port 11434)
                response = requests.post("http://localhost:11434/cpu-faiss/ingest", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    processed = result.get('processed', 0)
                    skipped = result.get('skipped', 0)
                    self.file_list.takeItem(0)  # Remove processing message
                    self._load_rag_files()  # Reload the file list
                    QMessageBox.information(self, "Processing Complete", 
                        f"✅ Processed {processed} file(s), skipped {skipped} file(s)\n"
                        f"Files are being processed in the background.")
                else:
                    self.file_list.takeItem(0)  # Remove processing message
                    QMessageBox.warning(self, "Processing Failed", 
                        f"Failed to trigger ingestion: HTTP {response.status_code}")
            else:
                # GPU mode: Use RAG container
                response = requests.post("http://localhost:11435/rag/ingest", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    processed = result.get('processed', 0)
                    skipped = result.get('skipped', 0)
                    
                    # Rebuild embeddings on host
                    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    host_script = os.path.join(workspace_root, 'setup', 'scripts', 'rebuild_embeddings_host.py')
                    import subprocess
                    rebuild_result = subprocess.run(
                        ["python3", host_script],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=workspace_root
                    )
                    if rebuild_result.returncode == 0:
                        # Reload RAG
                        reload_response = requests.post("http://localhost:11435/rag/reload", timeout=10)
                        if reload_response.status_code == 200:
                            self.file_list.takeItem(0)  # Remove processing message
                            self._load_rag_files()  # Reload the file list
                            QMessageBox.information(self, "Processing Complete", 
                                f"✅ Processed {processed} file(s), skipped {skipped} file(s)\n"
                                f"Embeddings built and RAG reloaded.")
                        else:
                            QMessageBox.warning(self, "Processing Incomplete", 
                                f"Files processed but RAG reload failed.")
                    else:
                        QMessageBox.warning(self, "Processing Incomplete", 
                            f"Files processed but embedding build failed.")
                else:
                    QMessageBox.warning(self, "Processing Failed", 
                        f"Failed to trigger ingestion: HTTP {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to trigger ingestion: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_file_selected(self):
        """Handle file selection - enable/disable remove button"""
        selected_items = self.file_list.selectedItems()
        # Enable remove button only if a file (not summary) is selected
        if selected_items and selected_items[0] in self.file_item_map:
            self.remove_btn.setEnabled(True)
        else:
            self.remove_btn.setEnabled(False)
    
    def _remove_selected_file(self):
        """Remove the selected file from RAG system"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        selected_item = selected_items[0]
        if selected_item not in self.file_item_map:
            # Selected item is not a file (e.g., summary)
            return
        
        filename = self.file_item_map[selected_item]
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to remove '{filename}' from the RAG system?\n\n"
            "This will:\n"
            "• Delete the file from data/input/\n"
            "• Remove its chunks from embeddings\n"
            "• Update the RAG index",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            input_dir = os.path.join(workspace_root, 'data', 'input')
            file_path = os.path.join(input_dir, filename)
            
            # Delete the file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[RAGFilesDialog] ✅ Deleted file: {file_path}")
            else:
                print(f"[RAGFilesDialog] ⚠️ File not found: {file_path}")
            
            # Trigger ingestion to remove chunks from embeddings
            # The ingestion system will detect the missing file and remove its chunks
            self._trigger_ingestion_for_file_removal(filename)
            
            # Reload file list
            self.file_list.clear()
            self.file_item_map.clear()
            self._load_rag_files()
            
            # Clear selection
            self.remove_btn.setEnabled(False)
            
            QMessageBox.information(
                self,
                "File Removed",
                f"✅ '{filename}' has been removed from the RAG system.\n\n"
                "The file and its embeddings are being removed in the background."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove file: {e}")
            import traceback
            traceback.print_exc()
    
    def _trigger_ingestion_for_file_removal(self, removed_filename):
        """Trigger ingestion after file removal to clean up embeddings"""
        try:
            import requests
            
            # Get RAG_MODE from settings file first, then fall back to environment variable
            RAG_MODE = None
            try:
                settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
                if os.path.exists(settings_path):
                    import json
                    with open(settings_path, 'r') as f:
                        settings = json.load(f)
                        RAG_MODE = settings.get('rag_mode', '').upper()
            except:
                pass
            
            # Fall back to environment variable if not in settings
            if not RAG_MODE:
                RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
            
            # Trigger ingestion based on RAG_MODE
            # The ingestion system will detect missing files and remove their chunks
            if RAG_MODE == 'CPU':
                # CPU mode: Use CPU FAISS in LLM container
                response = requests.post("http://localhost:11434/cpu-faiss/ingest", timeout=30)
                if response.status_code == 200:
                    print(f"[RAGFilesDialog] ✅ Triggered ingestion to remove chunks for {removed_filename}")
                else:
                    print(f"[RAGFilesDialog] ⚠️ Failed to trigger ingestion: HTTP {response.status_code}")
            else:
                # GPU mode: Use RAG container
                response = requests.post("http://localhost:11435/rag/ingest", timeout=30)
                if response.status_code == 200:
                    # Also trigger rebuild and reload
                    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    host_script = os.path.join(workspace_root, 'setup', 'scripts', 'rebuild_embeddings_host.py')
                    import subprocess
                    subprocess.run(
                        ["python3", host_script],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=workspace_root
                    )
                    # Reload RAG
                    reload_response = requests.post("http://localhost:11435/rag/reload", timeout=10)
                    if reload_response.status_code == 200:
                        print(f"[RAGFilesDialog] ✅ Triggered ingestion and reload to remove chunks for {removed_filename}")
                    else:
                        print(f"[RAGFilesDialog] ⚠️ Failed to reload RAG: HTTP {reload_response.status_code}")
                else:
                    print(f"[RAGFilesDialog] ⚠️ Failed to trigger ingestion: HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"[RAGFilesDialog] ⚠️ Error triggering ingestion: {e}")

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

class ConversationsDialog(BaseAuraDialog):
    """Dialog showing stored conversations with ability to select, delete selected, or delete all"""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="💬 Conversations", size=(1080, 1080), modal=True)
        print("[Conversations] 🔧 Initializing conversations dialog...")
        
        # Add additional styles
        additional_styles = """
            QLabel {
                color: white;
                font-size: 14px;
            }
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
            QPushButton {
                background-color: rgba(70, 130, 180, 0.25);
                color: #ffffff;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 0.45);
            }
            QPushButton:pressed {
                background-color: rgba(70, 130, 180, 0.65);
            }
        """
        base_stylesheet = self.styleSheet()
        self.setStyleSheet(base_stylesheet + additional_styles)
    
    def _setup_ui(self):
        """Setup UI - called by BaseAuraDialog"""
        self.setup_ui()
    
    def setup_ui(self):
        """Setup conversations dialog UI"""
        layout = QVBoxLayout()
        # Use base class default margins to ensure content fits within white perimeter
        margins = BaseAuraDialog.get_default_margins()
        layout.setContentsMargins(*margins)
        layout.setSpacing(BaseAuraDialog.get_default_spacing())
        
        # Add top stretch to center content vertically within white perimeter
        layout.addStretch(1)
        
        # Title
        title = QLabel("💬 Stored Conversations")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select conversations to delete or delete all")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #8e8e93; font-size: 12px; margin: 5px;")
        layout.addWidget(desc)
        
        # Conversations list (multi-select)
        # Limit height to ensure buttons fit within white perimeter
        self.conversations_list = QListWidget()
        self.conversations_list.setSelectionMode(QListWidget.MultiSelection)
        self.conversations_list.setMaximumHeight(500)  # Taller list for better conversation visibility
        self.conversations_list.setStyleSheet("""
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
        """)
        layout.addWidget(self.conversations_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Delete Selected button
        self.delete_selected_btn = QPushButton("🗑️ Delete Selected")
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        self.delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9500;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #E58500;
            }
            QPushButton:pressed {
                background-color: #CC7500;
            }
            QPushButton:disabled {
                background-color: rgba(255, 149, 0, 0.3);
                color: #999;
            }
        """)
        self.delete_selected_btn.setEnabled(False)
        button_layout.addWidget(self.delete_selected_btn)
        
        # Delete All button
        self.delete_all_btn = QPushButton("🗑️ Delete All")
        self.delete_all_btn.clicked.connect(self._delete_all)
        self.delete_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
            QPushButton:disabled {
                background-color: rgba(255, 59, 48, 0.3);
                color: #999;
            }
        """)
        # Will be enabled after conversations are loaded
        self.delete_all_btn.setEnabled(False)
        button_layout.addWidget(self.delete_all_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #8E8E93;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #6E6E73;
            }
            QPushButton:pressed {
                background-color: #5E5E63;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Add bottom stretch to center content vertically within white perimeter
        layout.addStretch(1)
        
        self.setLayout(layout)
        
        # Initialize conversation data list
        self.conversation_data = []
        
        # Connect selection changed to enable/disable delete selected button
        self.conversations_list.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Load conversations after UI is set up
        QTimer.singleShot(100, self._load_conversations)
    
    def _load_conversations(self):
        """Load conversations from memory container"""
        try:
            import requests
            response = requests.get("http://localhost:11438/recent?hours=8760&limit=1000", timeout=5)
            if response.status_code == 200:
                data = response.json()
                conversations = data.get("conversations", [])
                
                self.conversations_list.clear()
                self.conversation_data = []
                
                for conv in conversations:
                    # Get brief description (first 100 chars of text or summary)
                    text = conv.get("text", "")
                    summary = conv.get("summary", "")
                    description = summary if summary else (text[:100] + "..." if len(text) > 100 else text)
                    
                    # Get timestamp
                    timestamp = conv.get("timestamp", 0)
                    from datetime import datetime
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M")
                    else:
                        time_str = "Unknown"
                    
                    # Create item text
                    item_text = f"💬 {time_str}\n{description}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, conv)
                    self.conversations_list.addItem(item)
                    self.conversation_data.append(conv)
                
                if not conversations:
                    no_conv_item = QListWidgetItem("📭 No conversations stored yet")
                    no_conv_item.setFlags(Qt.NoItemFlags)
                    self.conversations_list.addItem(no_conv_item)
                    if hasattr(self, 'delete_all_btn'):
                        self.delete_all_btn.setEnabled(False)
                else:
                    # Enable delete all button if conversations exist
                    if hasattr(self, 'delete_all_btn'):
                        self.delete_all_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", f"Failed to load conversations: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            QMessageBox.warning(self, "Error", f"Could not connect to memory container: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading conversations: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_selection_changed(self):
        """Enable/disable delete selected button based on selection"""
        selected = self.conversations_list.selectedItems()
        self.delete_selected_btn.setEnabled(len(selected) > 0)
    
    def _delete_selected(self):
        """Delete selected conversations"""
        selected = self.conversations_list.selectedItems()
        if not selected:
            return
        
        reply = QMessageBox.question(
            self,
            "Delete Selected",
            f"Delete {len(selected)} selected conversation(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                import requests
                # Get conversation IDs to delete
                conv_ids = []
                for item in selected:
                    conv = item.data(Qt.UserRole)
                    if conv and "id" in conv:
                        conv_ids.append(conv["id"])
                
                if not conv_ids:
                    QMessageBox.warning(self, "Error", "Could not extract conversation IDs")
                    return
                
                # Try to delete via API endpoint (if it exists)
                # For now, we'll try POST to /delete endpoint
                try:
                    response = requests.post(
                        "http://localhost:11438/delete",
                        json={"conversation_ids": conv_ids},
                        timeout=10
                    )
                    if response.status_code == 200:
                        QMessageBox.information(self, "Success", 
                            f"Successfully deleted {len(conv_ids)} conversation(s).")
                        self._load_conversations()
                    else:
                        QMessageBox.warning(self, "Error", 
                            f"Delete endpoint returned HTTP {response.status_code}.\n"
                            "Deletion may not be implemented in memory container yet.\n\n"
                            "To enable deletion, add a DELETE endpoint to memory-container/container_rest.py")
                except requests.exceptions.Timeout:
                    QMessageBox.warning(self, "Timeout", 
                        f"Delete request timed out for {len(conv_ids)} conversation(s).\n"
                        "The deletion may have still succeeded. Reloading conversation list...")
                    self._load_conversations()  # Check if deletion actually happened
                except requests.exceptions.ConnectionError:
                    QMessageBox.warning(self, "Connection Error", 
                        "Could not connect to memory container.\n"
                        "Please ensure it's running on port 11438.")
                except requests.exceptions.RequestException as e:
                    # Request failed, but deletion may have still succeeded
                    QMessageBox.warning(self, "Request Error", 
                        f"Error during delete request: {str(e)}\n\n"
                        "The deletion may have still succeeded on the server.\n"
                        "Reloading conversation list to verify...")
                    self._load_conversations()  # Check if deletion actually happened
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete conversations: {e}")
                import traceback
                traceback.print_exc()
    
    def _delete_all(self):
        """Delete all conversations"""
        reply = QMessageBox.question(
            self,
            "Delete All",
            "Delete ALL stored conversations? This cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                import requests
                # Try to delete all via API endpoint
                try:
                    response = requests.post(
                        "http://localhost:11438/delete-all",
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        deleted_count = result.get('deleted', 0)
                        QMessageBox.information(self, "Success", 
                            f"Successfully deleted all {deleted_count} conversation(s).")
                        self._load_conversations()
                    else:
                        # Try to get error message from response
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', f'HTTP {response.status_code}')
                        except:
                            error_msg = f'HTTP {response.status_code}'
                        QMessageBox.warning(self, "Error", 
                            f"Delete-all endpoint returned: {error_msg}.\n"
                            "Deletion may have failed.")
                except requests.exceptions.Timeout:
                    QMessageBox.warning(self, "Timeout", 
                        "Delete request timed out. The deletion may still be in progress.\n"
                        "Reloading conversation list to check...")
                    self._load_conversations()  # Check if deletion actually happened
                except requests.exceptions.ConnectionError:
                    QMessageBox.warning(self, "Connection Error", 
                        "Could not connect to memory container.\n"
                        "Please ensure it's running on port 11438.")
                except requests.exceptions.RequestException as e:
                    # Other request errors - deletion may have still succeeded
                    QMessageBox.warning(self, "Request Error", 
                        f"Error during delete request: {str(e)}\n\n"
                        "The deletion may have succeeded. Reloading conversation list...")
                    self._load_conversations()  # Check if deletion actually happened
                except Exception as e:
                    # Catch any other errors (JSON parsing, etc.)
                    QMessageBox.warning(self, "Error", 
                        f"Unexpected error: {str(e)}\n\n"
                        "The deletion may have still succeeded on the server.\n"
                        "Reloading conversation list to verify...")
                    self._load_conversations()  # Reload to check if deletion actually happened
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete all conversations: {e}")
                import traceback
                traceback.print_exc()

class MemoryDialog(BaseAuraDialog):
    def __init__(self, parent=None):
        print("[Memory] 🔧 Initializing memory dialog...")
        
        # Initialize base dialog
        super().__init__(
            parent=parent,
            title="Memory - AuraVision",
            size=(1080, 1080),
            modal=True
        )
        
        # Set stylesheet before creating UI (base class already sets white border)
        print("[Memory] 👁️ Dialog initialized and ready")
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
        # Ensure border overlay is on top after UI is set up
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()
    
    def _on_show(self):
        """Block transcription when dialog opens"""
        self._block_transcription("Memory dialog open")
        # Ensure border overlay is on top when dialog is shown
        if hasattr(self, 'border_overlay'):
            self.border_overlay.raise_()
            self.border_overlay.update()
        
    def _on_close(self):
        """Additional cleanup when dialog closes (called by base class)"""
        pass
        
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
        # Use base class default margins to ensure content fits within white perimeter
        margins = BaseAuraDialog.get_default_margins()
        self.content_layout.setContentsMargins(*margins)
        self.content_layout.setSpacing(BaseAuraDialog.get_default_spacing())
        
        # Add top spacer for vertical centering
        self.content_layout.addStretch(1)
        
        # Title centered
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        
        # Title
        title = QLabel("💾 Memory")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        self.content_layout.addLayout(title_layout)
        
        # Description
        desc = QLabel("Manage stored conversations and RAG files")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #8e8e93; font-size: 12px; margin: 5px;")
        self.content_layout.addWidget(desc)
        
        # Buttons layout
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)
        
        # Conversations button
        self.conversations_btn = QPushButton("💬 Conversations")
        self.conversations_btn.clicked.connect(self.show_conversations)
        button_layout.addWidget(self.conversations_btn)
        
        # RAG Files button
        self.rag_files_btn = QPushButton("📚 RAG Files")
        self.rag_files_btn.clicked.connect(self.show_rag_files)
        button_layout.addWidget(self.rag_files_btn)
        
        # Google Drive button
        self.gdrive_btn = QPushButton("☁️ Google Drive")
        self.gdrive_btn.clicked.connect(self.add_google_drive)
        button_layout.addWidget(self.gdrive_btn)
        
        # QR Code button
        self.qr_btn = QPushButton("📱 Show QR Code")
        self.qr_btn.clicked.connect(self.show_qr_code)
        button_layout.addWidget(self.qr_btn)
        
        # Apply Apple-style styling to all action buttons
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
        
        for button in [self.conversations_btn, self.rag_files_btn, self.gdrive_btn, self.qr_btn]:
            button.setStyleSheet(action_button_style)
        
        self.content_layout.addLayout(button_layout)
        
        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        
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
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
        """)
        
        close_layout.addWidget(self.close_btn)
        close_layout.addStretch()
        self.content_layout.addLayout(close_layout)
        
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
    
    def show_conversations(self):
        """Show conversations dialog"""
        dialog = ConversationsDialog(self)
        dialog.exec_()
    
    def show_rag_files(self):
        """Show dialog with files actively being used by RAG"""
        dialog = RAGFilesDialog(self)
        dialog.exec_()
        
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
                
                # Create QR code dialog using BaseAuraDialog for proper centering
                try:
                    class QRCodeDialog(BaseAuraDialog):
                        def __init__(self, parent, qr_pixmap, upload_url):
                            # Set attributes BEFORE calling super().__init__() because
                            # BaseAuraDialog.__init__() calls _setup_ui() which needs these
                            self.qr_pixmap = qr_pixmap
                            self.upload_url = upload_url
                            super().__init__(parent, title="📱 Mobile Upload QR Code", size=(1080, 1080), modal=True)
                        
                        def _setup_ui(self):
                            """Setup UI - called by BaseAuraDialog"""
                            layout = QVBoxLayout()
                            layout.setContentsMargins(120, 100, 120, 100)
                            layout.setSpacing(15)
                            layout.addStretch(1)
                            
                            # Title
                            title = QLabel("📱 Mobile Upload")
                            title.setFont(QFont("Arial", 16, QFont.Bold))
                            title.setAlignment(Qt.AlignCenter)
                            title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 15px; font-size: 16px;")
                            layout.addWidget(title)
                            
                            # Instructions
                            instructions = QLabel("Scan QR code with your phone:")
                            instructions.setAlignment(Qt.AlignCenter)
                            instructions.setStyleSheet("color: #8e8e93; font-size: 13px; margin: 8px;")
                            layout.addWidget(instructions)
                            
                            # QR Code
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
                            qr_label.setPixmap(self.qr_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                            layout.addWidget(qr_label)
                            
                            # URL
                            url_label = QLabel(f"🌐 {self.upload_url}")
                            url_label.setAlignment(Qt.AlignCenter)
                            url_label.setStyleSheet("color: #007AFF; font-size: 14px; font-weight: 500; margin: 10px;")
                            layout.addWidget(url_label)
                            
                            # Buttons
                            button_layout = QHBoxLayout()
                            
                            open_btn = QPushButton("🌐 Open in Browser")
                            open_btn.clicked.connect(lambda: webbrowser.open(self.upload_url))
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
                            close_btn.clicked.connect(self.accept)
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
                            layout.addStretch(1)
                            self.setLayout(layout)
                    
                    qr_dialog = QRCodeDialog(self, qr_pixmap, upload_url)
                    print(f"[Upload] 📱 Showing QR code dialog for {upload_url}")
                    qr_dialog.show()  # Explicitly show before exec_()
                    qr_dialog.raise_()
                    qr_dialog.activateWindow()
                    qr_dialog.exec_()
                except Exception as e:
                    print(f"[Upload] ❌ Error showing QR code dialog: {e}")
                    import traceback
                    traceback.print_exc()
                    QMessageBox.warning(self, "QR Code Error", f"Failed to show QR code dialog: {e}")
                
            else:
                QMessageBox.warning(self, "Upload Server", "Upload server is not responding properly.")
        except requests.exceptions.RequestException:
            QMessageBox.warning(self, "Upload Server", 
                              "Upload server is not running.\n\n"
                              "To enable web upload:\n"
                              "1. Install Flask: pip install flask\n"
                              "2. Restart Aura system\n"
                              "3. The upload server will start automatically")
    

# Global variable to prevent multiple dialogs
_current_dialog = None

def show_memory_dialog():
    """Show the memory dialog"""
    global _current_dialog
    
    # Prevent multiple dialogs
    if _current_dialog is not None:
        print("[Memory] ⚠️ Dialog already open, bringing to front...")
        _current_dialog.raise_()
        _current_dialog.activateWindow()
        return
    
    print("[Memory] 🚀 Opening memory dialog...")
    
    try:
        _current_dialog = MemoryDialog()
        print("[Memory] 📱 Dialog created, showing...")
        
        # Ensure dialog is visible and on top
        _current_dialog.show()
        _current_dialog.raise_()
        _current_dialog.activateWindow()
        
        # Process events to ensure dialog is rendered
        QApplication.processEvents()
        
        print("[Memory] 👁️ Dialog is now visible, waiting for user interaction...")
        _current_dialog.exec_()
        print("[Memory] ✅ Dialog closed")
    except Exception as e:
        print(f"[Memory] ❌ Dialog error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _current_dialog = None

# Backward compatibility alias
def show_upload_dialog():
    """Backward compatibility - redirects to memory dialog"""
    show_memory_dialog()
