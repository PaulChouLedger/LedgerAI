#!/usr/bin/env python3
"""
Web-based file upload server for Aura
Allows easy file uploads from laptops, phones, or any device on the network
"""

import os
import sys
import time
import threading
import webbrowser
from flask import Flask, request, render_template_string, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import socket

# Configuration
# Upload folder should be at workspace root: LedgerAI/data/input
# From aura-control/server/ we need to go up 2 levels to workspace root
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER = os.path.join(_workspace_root, 'data', 'input')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'rtf', 'odt', 'xlsx', 'xls', 'wav', 'mp3', 'mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max file size

# Get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'aura-upload-secret-key'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML template for the upload interface
UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura File Upload</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .upload-area {
            border: 3px dashed rgba(255, 255, 255, 0.5);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: rgba(255, 255, 255, 0.8);
            background: rgba(255, 255, 255, 0.1);
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: rgba(76, 175, 80, 0.2);
        }
        .file-input {
            display: none;
        }
        .upload-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .file-list {
            margin-top: 30px;
        }
        .file-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            text-align: center;
        }
        .success {
            background: rgba(76, 175, 80, 0.3);
            border: 1px solid #4CAF50;
        }
        .error {
            background: rgba(244, 67, 54, 0.3);
            border: 1px solid #f44336;
        }
        .info {
            background: rgba(33, 150, 243, 0.3);
            border: 1px solid #2196F3;
        }
        .qr-code {
            text-align: center;
            margin: 20px 0;
        }
        .qr-code img {
            max-width: 200px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📁 Aura File Upload</h1>
        
        <div class="info">
            <strong>📱 Access from any device:</strong><br>
            <strong>Desktop:</strong> <a href="http://{{ ip }}:{{ port }}" style="color: #4CAF50;">http://{{ ip }}:{{ port }}</a><br>
            <strong>Mobile:</strong> Scan QR code or use the link above
        </div>
        
        <div class="qr-code">
            <img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=http://{{ ip }}:{{ port }}" alt="QR Code">
        </div>
        
        <form method="post" action="/upload" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📤 Drop files here or click to select</h3>
                <p>Supported formats: PDF, DOC, DOCX, TXT, MD, RTF, ODT, XLSX, XLS, WAV, MP3, MP4, AVI, MOV, PNG, JPG, GIF</p>
                <input type="file" id="fileInput" name="files" multiple class="file-input" onchange="handleFiles(this.files)">
                <div id="fileStatus"></div>
            </div>
            <div style="text-align: center; margin: 20px 0;">
                <button type="submit" class="upload-btn">🚀 Upload Files</button>
            </div>
        </form>
        
        {% if files %}
        <div class="file-list">
            <h3>📋 Recent Files:</h3>
            {% for file in files %}
            <div class="file-item">
                <span>📄 {{ file }}</span>
                <span style="color: #4CAF50;">✅ Uploaded</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="status {{ 'success' if 'success' in message else 'error' }}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>
    
    <script>
        const uploadArea = document.querySelector('.upload-area');
        const fileInput = document.getElementById('fileInput');
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            fileInput.files = files;
            handleFiles(files);
        });
        
        function handleFiles(files) {
            const fileList = Array.from(files);
            const fileNames = fileList.map(f => f.name).join(', ');
            const fileStatus = document.getElementById('fileStatus');
            fileStatus.innerHTML = `
                <div style="margin-top: 20px; padding: 15px; background: rgba(76, 175, 80, 0.2); border-radius: 10px;">
                    <h3>✅ ${fileList.length} file(s) selected</h3>
                    <p style="font-size: 0.9em; margin: 5px 0;">${fileNames}</p>
                    <p style="font-size: 0.9em; color: #4CAF50;">Ready to upload! Click the button below.</p>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    # Get list of recent files
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        files = sorted(os.listdir(UPLOAD_FOLDER), key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)[:10]
    
    return render_template_string(UPLOAD_TEMPLATE, 
                                ip=get_local_ip(), 
                                port=app.config.get('PORT', 5001),
                                files=files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    uploaded_count = 0
    upload_errors = []
    
    # Ensure upload folder exists
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        abs_path = os.path.abspath(UPLOAD_FOLDER)
        print(f"[Aura-Upload] 📁 Upload folder: {abs_path}")
        
        # Verify the folder is writable
        if not os.access(abs_path, os.W_OK):
            raise PermissionError(f"Upload folder is not writable: {abs_path}")
    except Exception as e:
        print(f"[Aura-Upload] ❌ Failed to create/access upload folder: {e}")
        flash(f'Upload folder error: {e}', 'error')
        return redirect(url_for('index'))
    
    for file in files:
        if file.filename == '':
            continue
        
        if not file:
            print(f"[Aura-Upload] ⚠️ Empty file object for {file.filename}")
            continue
            
        if not allowed_file(file.filename):
            error_msg = f'File {file.filename} has an invalid extension'
            print(f"[Aura-Upload] ⚠️ {error_msg}")
            flash(error_msg, 'error')
            upload_errors.append(error_msg)
            continue
        
        try:
            filename = secure_filename(file.filename)
            if not filename:
                error_msg = f'Invalid filename: {file.filename}'
                print(f"[Aura-Upload] ⚠️ {error_msg}")
                flash(error_msg, 'error')
                upload_errors.append(error_msg)
                continue
            
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Handle duplicate filenames
            counter = 1
            original_filename = filename
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                counter += 1
            
            # Save the file
            file.save(filepath)
            
            # Verify file was actually saved
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                uploaded_count += 1
                print(f"[Aura-Upload] ✅ Uploaded: {filename} ({file_size} bytes) to {filepath}")
            else:
                error_msg = f'File save failed: {filename}'
                print(f"[Aura-Upload] ❌ {error_msg}")
                flash(error_msg, 'error')
                upload_errors.append(error_msg)
        except Exception as e:
            error_msg = f'Error uploading {file.filename}: {e}'
            print(f"[Aura-Upload] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            flash(error_msg, 'error')
            upload_errors.append(error_msg)
    
    if uploaded_count > 0:
        flash(f'Successfully uploaded {uploaded_count} file(s)', 'success')
        
        # Trigger ingest pipeline: container extracts → rebuild embeddings → ready
        try:
            import requests
            import subprocess
            
            print(f"[Aura-Upload] 🔄 Processing {uploaded_count} new file(s)...")
            
            # Step 1: Container extracts text from PDFs/TXT/DOCX
            print(f"[Aura-Upload] 🔄 Step 1: Extracting text...")
            # Trigger ingest based on RAG_MODE setting
            # Note: os and threading are already imported at top of file
            
            # Check RAG_MODE from environment (GPU = RAG container, CPU = CPU FAISS)
            RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()
            USE_MEDICAL_MODE = os.environ.get('USE_MEDICAL_MODE', 'false').lower() == 'true'
            
            def trigger_cpu_rag_medical():
                try:
                    response = requests.post("http://localhost:11434/cpu-faiss/ingest", timeout=30)
                    if response.status_code == 200:
                        print("[Upload] ✅ Medical CPU FAISS ingest triggered")
                    else:
                        print(f"[Upload] ⚠️ Medical CPU FAISS ingest failed: HTTP {response.status_code}")
                except Exception as e:
                    print(f"[Upload] ⚠️ Medical CPU FAISS ingest error: {e}")
            
            def trigger_cpu_rag_generic():
                try:
                    response = requests.post("http://localhost:11436/cpu-faiss/ingest", timeout=30)
                    if response.status_code == 200:
                        print("[Upload] ✅ Generic CPU FAISS ingest triggered")
                    else:
                        print(f"[Upload] ⚠️ Generic CPU FAISS ingest failed: HTTP {response.status_code}")
                except Exception as e:
                    print(f"[Upload] ⚠️ Generic CPU FAISS ingest error: {e}")
            
            if RAG_MODE == 'GPU':
                # GPU RAG mode: Use RAG container only (skip CPU FAISS)
                print("[Upload] 🚀 RAG_MODE=GPU - using RAG container")
                try:
                    response = requests.post("http://localhost:11435/rag/ingest", timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"[Upload] ✅ GPU RAG ingest: {result.get('processed', 0)} processed, {result.get('skipped', 0)} skipped")
                    else:
                        print(f"[Upload] ⚠️ GPU RAG ingest failed: HTTP {response.status_code}")
                        response = type('obj', (object,), {'status_code': response.status_code})()
                        response.json = lambda: {'processed': 0, 'skipped': 0}
                except Exception as e:
                    print(f"[Upload] ⚠️ GPU RAG ingest error: {e}")
                    response = type('obj', (object,), {'status_code': 500})()
                    response.json = lambda: {'processed': 0, 'skipped': 0}
                
                if response.status_code == 200:
                    result = response.json()
                    processed = result.get('processed', 0)
                    print(f"[Aura-Upload] ✅ Text extracted: {processed} files")
                    
                    # Step 2: Rebuild embeddings on HOST (container's FAISS has SWIG issues)
                    print(f"[Aura-Upload] 🔄 Step 2: Building embeddings on HOST...")
                    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    host_script = os.path.join(workspace_root, 'setup', 'scripts', 'rebuild_embeddings_host.py')
                    
                    rebuild_result = subprocess.run(
                        ["python3", host_script],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=workspace_root
                    )
                    
                    if rebuild_result.returncode == 0:
                        print(f"[Aura-Upload] ✅ Embeddings rebuilt - files ready in RAG!")
                        
                        # Step 3: Reload RAG to use new index
                        print(f"[Aura-Upload] 🔄 Step 3: Reloading RAG index...")
                        reload_response = requests.post("http://localhost:11435/rag/reload", timeout=10)
                        if reload_response.status_code == 200:
                            reload_result = reload_response.json()
                            total_chunks = reload_result.get('total_chunks', 0)
                            print(f"[Aura-Upload] ✅ RAG reloaded: {total_chunks} total chunks")
                            flash(f'Successfully uploaded and indexed {uploaded_count} file(s) - {total_chunks} chunks available', 'success')
                        else:
                            print(f"[Aura-Upload] ⚠️ RAG reload failed: {reload_response.status_code}")
                            flash(f'Files uploaded but RAG reload failed', 'warning')
                    else:
                        print(f"[Aura-Upload] ❌ Embedding rebuild failed")
                        flash('Files uploaded but indexing failed', 'warning')
                else:
                    print(f"[Aura-Upload] ❌ Text extraction failed")
                    flash('File uploaded but processing failed', 'warning')
                    return redirect(url_for('index'))
            else:
                # CPU RAG mode: Use CPU FAISS only (skip GPU RAG container)
                print("[Upload] 💻 RAG_MODE=CPU - using CPU FAISS in LLM containers")
                
                # Only trigger the container that's actually running
                threads = []
                if USE_MEDICAL_MODE:
                    cpu_medical_thread = threading.Thread(target=trigger_cpu_rag_medical, daemon=True)
                    cpu_medical_thread.start()
                    threads.append(cpu_medical_thread)
                
                cpu_generic_thread = threading.Thread(target=trigger_cpu_rag_generic, daemon=True)
                cpu_generic_thread.start()
                threads.append(cpu_generic_thread)
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join(timeout=30)
                
                print(f"[Aura-Upload] ✅ CPU FAISS ingest triggered for {len(threads)} container(s)")
                # CPU FAISS handles its own processing internally - no host embedding rebuild needed
                print(f"[Aura-Upload] ✅ Files processed by CPU FAISS in LLM containers")
                flash(f'Successfully uploaded {uploaded_count} file(s) - processing by CPU FAISS', 'success')
        except Exception as e:
            print(f"[Aura-Upload] ⚠️ Processing error: {e}")
    
    return redirect(url_for('index'))

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'upload_folder': UPLOAD_FOLDER,
        'files_count': len(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else 0
    })

def start_upload_server(port=5001):
    """Start the upload server"""
    app.config['PORT'] = port
    local_ip = get_local_ip()
    
    print(f"[Aura-Upload] 🚀 Starting upload server...")
    print(f"[Aura-Upload] 📱 Access from any device:")
    print(f"[Aura-Upload] 🌐 Desktop: http://{local_ip}:{port}")
    print(f"[Aura-Upload] 📱 Mobile: Scan QR code or use the link above")
    print(f"[Aura-Upload] 📁 Upload folder: {UPLOAD_FOLDER}")
    
    # Start server in a separate thread
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    return local_ip, port

if __name__ == '__main__':
    start_upload_server()
