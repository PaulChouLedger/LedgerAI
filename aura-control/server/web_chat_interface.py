#!/usr/bin/env python3
"""
Aura Web UI - Open WebUI-inspired Interface
Modern web-based chat interface with streaming, session management, and advanced features
Inspired by https://github.com/open-webui/open-webui
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context
import requests
from typing import Dict, List, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Configuration
DEFAULT_LLM_PORT = os.getenv("LLM_PORT", "11434")
CHAT_SERVER_PORT = int(os.getenv("CHAT_SERVER_PORT", "5001"))
ENABLE_DARK_MODE = os.getenv("ENABLE_DARK_MODE", "true").lower() == "true"

# Session management (in-memory, can be replaced with Redis for production)
sessions: Dict[str, Dict] = {}

def detect_llm_port():
    """Detect which LLM container is running by checking health endpoints"""
    for port in [11434, 11436]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"[Aura WebUI] ✅ Detected LLM container on port {port}")
                return port
        except:
            continue
    
    print(f"[Aura WebUI] ⚠️ Could not detect LLM container, defaulting to port {DEFAULT_LLM_PORT}")
    return DEFAULT_LLM_PORT

LLM_PORT = detect_llm_port()
CHAT_URL = f"http://localhost:{LLM_PORT}/chat-tts"

app = Flask(__name__)

# HTML Template - Open WebUI-inspired design
WEBUI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura WebUI - AI Chat Interface</title>
    <style>
        :root {
            --primary: #667eea;
            --primary-dark: #5568d3;
            --secondary: #764ba2;
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border: #dee2e6;
            --success: #28a745;
            --error: #dc3545;
            --warning: #ffc107;
            --shadow: rgba(0, 0, 0, 0.1);
        }
        
        [data-theme="dark"] {
            --bg-primary: #1a1d29;
            --bg-secondary: #252936;
            --bg-tertiary: #2d3142;
            --text-primary: #e9ecef;
            --text-secondary: #adb5bd;
            --border: #495057;
            --shadow: rgba(0, 0, 0, 0.3);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            transition: background 0.3s, color 0.3s;
        }
        
        .header {
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: var(--bg-primary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn:hover {
            background: var(--bg-tertiary);
            border-color: var(--primary);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
        }
        
        .btn-primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .main-container {
            flex: 1;
            display: flex;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem;
            gap: 1rem;
        }
        
        .sidebar {
            width: 280px;
            background: var(--bg-primary);
            border-radius: 0.75rem;
            padding: 1rem;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }
        
        .sidebar-section {
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }
        
        .sidebar-section:last-child {
            border-bottom: none;
        }
        
        .sidebar-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .chat-list {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .chat-item {
            padding: 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.875rem;
        }
        
        .chat-item:hover {
            background: var(--bg-tertiary);
        }
        
        .chat-item.active {
            background: var(--bg-tertiary);
            border-left: 3px solid var(--primary);
        }
        
        .chat-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
            border-radius: 0.75rem;
            border: 1px solid var(--border);
            overflow: hidden;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .message {
            display: flex;
            gap: 1rem;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }
        
        .message.assistant .message-avatar {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }
        
        .message-content {
            flex: 1;
            padding: 1rem;
            border-radius: 0.75rem;
            background: var(--bg-secondary);
            line-height: 1.6;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            margin-left: auto;
            max-width: 80%;
        }
        
        .message.assistant .message-content {
            background: var(--bg-secondary);
            color: var(--text-primary);
            max-width: 85%;
        }
        
        .message.streaming .message-content {
            border-left: 3px solid var(--primary);
        }
        
        .message-content p {
            margin: 0.5em 0;
        }
        
        .message-content p:first-child {
            margin-top: 0;
        }
        
        .message-content p:last-child {
            margin-bottom: 0;
        }
        
        .message-content ol,
        .message-content ul {
            margin: 0.8em 0;
            padding-left: 1.5em;
        }
        
        .message-content li {
            margin: 0.4em 0;
        }
        
        .chat-input-container {
            padding: 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--bg-primary);
        }
        
        .input-wrapper {
            display: flex;
            gap: 0.75rem;
            align-items: flex-end;
            background: var(--bg-secondary);
            border: 2px solid var(--border);
            border-radius: 1rem;
            padding: 0.75rem;
            transition: border-color 0.2s;
        }
        
        .input-wrapper:focus-within {
            border-color: var(--primary);
        }
        
        .chat-input {
            flex: 1;
            border: none;
            background: transparent;
            color: var(--text-primary);
            font-size: 1rem;
            resize: none;
            outline: none;
            max-height: 200px;
            min-height: 24px;
            line-height: 1.5;
        }
        
        .chat-input::placeholder {
            color: var(--text-secondary);
        }
        
        .send-button {
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .send-button:hover:not(:disabled) {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .status-bar {
            padding: 0.5rem 1.5rem;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-dot.error {
            background: var(--error);
            animation: none;
        }
        
        .settings-panel {
            position: fixed;
            top: 0;
            right: -400px;
            width: 400px;
            height: 100vh;
            background: var(--bg-primary);
            border-left: 1px solid var(--border);
            box-shadow: -4px 0 12px var(--shadow);
            transition: right 0.3s;
            z-index: 1000;
            overflow-y: auto;
            padding: 2rem;
        }
        
        .settings-panel.open {
            right: 0;
        }
        
        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        .settings-title {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .settings-section {
            margin-bottom: 2rem;
        }
        
        .settings-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .settings-input,
        .settings-select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.875rem;
        }
        
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 48px;
            height: 24px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: var(--bg-tertiary);
            transition: 0.3s;
            border-radius: 24px;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }
        
        input:checked + .toggle-slider {
            background-color: var(--primary);
        }
        
        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        
        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .empty-state-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .empty-state-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }
        
        .empty-state-text {
            font-size: 1rem;
            max-width: 500px;
        }
        
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            
            .main-container {
                padding: 0.5rem;
            }
        }
    </style>
</head>
<body data-theme="{{ theme }}">
    <div class="header">
        <div class="header-left">
            <div class="logo">✨ Aura WebUI</div>
        </div>
        <div class="header-right">
            <button class="btn" onclick="toggleTheme()">
                <span id="themeIcon">🌙</span> <span id="themeText">Dark</span>
            </button>
            <button class="btn" onclick="toggleSettings()">⚙️ Settings</button>
            <button class="btn btn-primary" onclick="newChat()">+ New Chat</button>
        </div>
    </div>
    
    <div class="main-container">
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">Chats</div>
                <div class="chat-list" id="chatList">
                    <div class="chat-item active" data-session="default">
                        <span>New Chat</span>
                    </div>
                </div>
            </div>
            <div class="sidebar-section">
                <div class="sidebar-title">Model</div>
                <select class="settings-select" id="modelSelect" onchange="updateModel()">
                    <option value="medical">Medical (Qwen2.5-1.5B)</option>
                    <option value="generic">Generic (Qwen2.5-1.5B)</option>
                </select>
            </div>
        </div>
        
        <div class="chat-content">
            <div class="chat-messages" id="chatMessages">
                <div class="empty-state">
                    <div class="empty-state-icon">💬</div>
                    <div class="empty-state-title">Welcome to Aura WebUI</div>
                    <div class="empty-state-text">
                        Start a conversation by typing a message below. I'm here to help with medical triage, 
                        general questions, and more!
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <div class="input-wrapper">
                    <textarea 
                        class="chat-input" 
                        id="userInput" 
                        placeholder="Type your message here..."
                        rows="1"
                        onkeydown="handleKeyDown(event)"
                        oninput="autoResize(this)"
                    ></textarea>
                    <button class="send-button" id="sendButton" onclick="sendMessage()" disabled>
                        <span>Send</span>
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Ready</span>
        </div>
        <div id="modelInfo">Model: Medical</div>
    </div>
    
    <div class="settings-panel" id="settingsPanel">
        <div class="settings-header">
            <div class="settings-title">Settings</div>
            <button class="btn" onclick="toggleSettings()">✕</button>
        </div>
        
        <div class="settings-section">
            <label class="settings-label">Theme</label>
            <label class="toggle-switch">
                <input type="checkbox" id="darkModeToggle" onchange="toggleTheme()">
                <span class="toggle-slider"></span>
            </label>
            <span style="margin-left: 1rem;">Dark Mode</span>
        </div>
        
        <div class="settings-section">
            <label class="settings-label">LLM Port</label>
            <input type="number" class="settings-input" id="llmPort" value="{{ llm_port }}" placeholder="11434">
        </div>
        
        <div class="settings-section">
            <label class="settings-label">Streaming</label>
            <label class="toggle-switch">
                <input type="checkbox" id="streamingToggle" checked>
                <span class="toggle-slider"></span>
            </label>
            <span style="margin-left: 1rem;">Enable Streaming</span>
        </div>
    </div>
    
    <script>
        let currentSession = 'default';
        let isStreaming = false;
        let currentMessageDiv = null;
        let theme = localStorage.getItem('theme') || '{{ theme }}';
        
        // Initialize
        document.documentElement.setAttribute('data-theme', theme);
        updateThemeUI();
        updateSendButton();
        
        // Auto-resize textarea
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        
        // Handle Enter key
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        // Update send button state
        function updateSendButton() {
            const input = document.getElementById('userInput');
            const button = document.getElementById('sendButton');
            button.disabled = !input.value.trim() || isStreaming;
        }
        
        document.getElementById('userInput').addEventListener('input', updateSendButton);
        
        // Theme toggle
        function toggleTheme() {
            theme = theme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            updateThemeUI();
        }
        
        function updateThemeUI() {
            const icon = document.getElementById('themeIcon');
            const text = document.getElementById('themeText');
            const toggle = document.getElementById('darkModeToggle');
            
            if (theme === 'dark') {
                icon.textContent = '☀️';
                text.textContent = 'Light';
                if (toggle) toggle.checked = true;
            } else {
                icon.textContent = '🌙';
                text.textContent = 'Dark';
                if (toggle) toggle.checked = false;
            }
        }
        
        // Settings panel
        function toggleSettings() {
            const panel = document.getElementById('settingsPanel');
            panel.classList.toggle('open');
        }
        
        // New chat
        function newChat() {
            currentSession = 'session_' + Date.now();
            document.getElementById('chatMessages').innerHTML = '';
            updateStatus('Ready', 'success');
        }
        
        // Update model
        function updateModel() {
            const select = document.getElementById('modelSelect');
            document.getElementById('modelInfo').textContent = 'Model: ' + select.value;
        }
        
        // Status update
        function updateStatus(text, type = 'success') {
            document.getElementById('statusText').textContent = text;
            const dot = document.getElementById('statusDot');
            dot.className = 'status-dot ' + (type === 'error' ? 'error' : '');
        }
        
        // Format message
        function formatMessage(text) {
            if (!text) return '';
            
            // Clean markdown
            text = text.replace(/^#{1,6}\s+/gm, '');
            text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
            
            // Format lists
            const lines = text.split('\\n');
            let html = '';
            let inList = false;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const listMatch = line.match(/^(\\d+)\\.\\s+(.+)$/);
                
                if (listMatch) {
                    if (!inList) {
                        html += '<ol>';
                        inList = true;
                    }
                    html += '<li>' + listMatch[2] + '</li>';
                } else {
                    if (inList) {
                        html += '</ol>';
                        inList = false;
                    }
                    if (line.trim()) {
                        html += '<p>' + line + '</p>';
                    }
                }
            }
            
            if (inList) html += '</ol>';
            
            return html || text.replace(/\\n/g, '<br>');
        }
        
        // Add message
        function addMessage(role, text, streaming = false) {
            const messagesDiv = document.getElementById('chatMessages');
            
            // Remove empty state
            const emptyState = messagesDiv.querySelector('.empty-state');
            if (emptyState) emptyState.remove();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}${streaming ? ' streaming' : ''}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? 'U' : 'A';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            if (role === 'assistant' && !streaming) {
                content.innerHTML = formatMessage(text);
            } else {
                content.textContent = text;
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            messagesDiv.appendChild(messageDiv);
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            if (streaming) {
                currentMessageDiv = messageDiv;
            }
            
            return messageDiv;
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message || isStreaming) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            autoResize(input);
            updateSendButton();
            
            isStreaming = true;
            updateStatus('Streaming...', 'success');
            
            // Create assistant message div
            const assistantMsg = addMessage('assistant', '', true);
            let accumulated = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: message,
                        session_id: currentSession,
                        stream: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));
                                accumulated = data.response || accumulated;
                                
                                const content = assistantMsg.querySelector('.message-content');
                                content.textContent = accumulated;
                                
                                if (data.done) {
                                    content.innerHTML = formatMessage(accumulated);
                                    assistantMsg.classList.remove('streaming');
                                    updateStatus('Ready', 'success');
                                    isStreaming = false;
                                    updateSendButton();
                                    input.focus();
                                    return;
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                updateStatus('Error: ' + error.message, 'error');
                const content = assistantMsg.querySelector('.message-content');
                content.textContent = 'Sorry, I encountered an error: ' + error.message;
                assistantMsg.classList.remove('streaming');
            } finally {
                isStreaming = false;
                updateSendButton();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web UI"""
    return render_template_string(
        WEBUI_TEMPLATE,
        theme='dark' if ENABLE_DARK_MODE else 'light',
        llm_port=LLM_PORT
    )

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat with streaming support"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        session_id = data.get('session_id', 'default')
        stream = data.get('stream', True)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Initialize session if needed
        if session_id not in sessions:
            sessions[session_id] = {
                'created_at': datetime.now().isoformat(),
                'messages': []
            }
        
        # Add user message to session
        sessions[session_id]['messages'].append({
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        })
        
        if stream:
            return Response(
                stream_with_context(generate_stream(prompt, session_id)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive'
                }
            )
        else:
            # Non-streaming response
            response = requests.post(
                CHAT_URL,
                json={'prompt': prompt, 'chat_id': session_id, 'stream': False},
                timeout=60
            )
            
            if response.status_code != 200:
                return jsonify({'error': f'LLM error: {response.status_code}'}), 500
            
            result = response.json()
            response_text = result.get('response', '')
            
            # Add assistant message to session
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': response_text,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({'response': response_text, 'done': True})
            
    except Exception as e:
        print(f"[Aura WebUI] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_stream(prompt: str, session_id: str):
    """Generate streaming response"""
    try:
        # Check LLM health
        try:
            health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
            if health_check.status_code != 200:
                yield f"data: {json.dumps({'response': 'LLM container not responding', 'done': True})}\\n\\n"
                return
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'response': f'Cannot connect to LLM on port {LLM_PORT}', 'done': True})}\\n\\n"
            return
        
        # Forward to LLM container
        response = requests.post(
            CHAT_URL,
            json={'prompt': prompt, 'chat_id': session_id, 'stream': True},
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            yield f"data: {json.dumps({'response': f'LLM error: {response.status_code}', 'done': True})}\\n\\n"
            return
        
        accumulated = ''
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.strip():
                    # Forward SSE format
                    if decoded.startswith('data: '):
                        yield decoded + '\n'
                        try:
                            data = json.loads(decoded[6:])
                            accumulated = data.get('response', accumulated)
                        except:
                            pass
                    else:
                        # Plain text line - convert to SSE
                        accumulated += decoded + '\n'
                        yield f"data: {json.dumps({'response': accumulated})}\\n"
        
        # Final message
        sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': accumulated,
            'timestamp': datetime.now().isoformat()
        })
        
        yield f"data: {json.dumps({'response': accumulated, 'done': True})}\\n\\n"
        
    except Exception as e:
        print(f"[Aura WebUI] ❌ Stream error: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'response': f'Error: {str(e)}', 'done': True})}\\n\\n"

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    return jsonify({
        'sessions': {
            sid: {
                'id': sid,
                'created_at': s['created_at'],
                'message_count': len(s.get('messages', []))
            }
            for sid, s in sessions.items()
        }
    })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific session"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'session': sessions[session_id]
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    llm_healthy = False
    try:
        health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
        llm_healthy = health_check.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'ok' if llm_healthy else 'degraded',
        'service': 'aura-webui',
        'llm_port': LLM_PORT,
        'llm_connected': llm_healthy,
        'sessions': len(sessions)
    })

def main():
    """Start the Aura WebUI server"""
    print("=" * 80)
    print("  ✨ Aura WebUI - Open WebUI-inspired Interface")
    print("=" * 80)
    print(f"🌐 Server: http://localhost:{CHAT_SERVER_PORT}")
    print(f"🔗 LLM endpoint: {CHAT_URL} (port {LLM_PORT})")
    print(f"📱 Access from network: http://0.0.0.0:{CHAT_SERVER_PORT}")
    
    # Verify LLM container
    try:
        health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
        if health_check.status_code == 200:
            print(f"✅ LLM container is accessible")
        else:
            print(f"⚠️  LLM container returned status {health_check.status_code}")
    except:
        print(f"⚠️  WARNING: Cannot connect to LLM container on port {LLM_PORT}")
        print(f"   Start LLM container: docker-compose up llm-medical")
    
    print("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=CHAT_SERVER_PORT,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()
