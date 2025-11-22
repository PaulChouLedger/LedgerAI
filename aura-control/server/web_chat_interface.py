#!/usr/bin/env python3
"""
Web Chat Interface - Test streaming chatbot
Provides a web-based chat interface to test the streaming /chat-tg endpoint
"""

import os
import sys
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context
import requests
import json
import threading

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Configuration
# Both medical and generic containers use port 11434
DEFAULT_LLM_PORT = os.getenv("LLM_PORT", "11434")
CHAT_SERVER_PORT = int(os.getenv("CHAT_SERVER_PORT", "5001"))

def detect_llm_port():
    """Detect which LLM container is running by checking health endpoints"""
    # Try port 11434 first (both containers use this port)
    for port in [11434, 11436]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"[WebChat] ✅ Detected LLM container on port {port}")
                return port
        except:
            continue
    
    # Default to 11434 if detection fails
    print(f"[WebChat] ⚠️ Could not detect LLM container, defaulting to port {DEFAULT_LLM_PORT}")
    return DEFAULT_LLM_PORT

LLM_PORT = detect_llm_port()
CHAT_URL = f"http://localhost:{LLM_PORT}/chat-tg"

app = Flask(__name__)

# HTML template for chat interface
CHAT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura Chat - Streaming Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            width: 100%;
            max-width: 800px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            max-width: 75%;
            padding: 14px 18px;
            border-radius: 20px;
            word-wrap: break-word;
            word-break: break-word;
            line-height: 1.5;
            animation: fadeIn 0.3s ease-in;
            margin-bottom: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 6px;
            margin-left: auto;
            box-shadow: 0 2px 12px rgba(102, 126, 234, 0.3);
        }
        
        .message.assistant {
            background: #ffffff;
            color: #2c3e50;
            align-self: flex-start;
            border-bottom-left-radius: 6px;
            border: 1px solid #e0e0e0;
            margin-right: auto;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            font-size: 15px;
        }
        
        .message.streaming {
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }
        
        .message.assistant p {
            margin: 0.6em 0;
            line-height: 1.6;
        }
        
        .message.assistant p:first-child {
            margin-top: 0;
        }
        
        .message.assistant p:last-child {
            margin-bottom: 0;
        }
        
        .message.assistant strong {
            color: #667eea;
            font-weight: 600;
        }
        
        .message.assistant ol,
        .message.assistant ul {
            margin: 0.8em 0;
            padding-left: 1.8em;
            line-height: 1.7;
        }
        
        .message.assistant ol li,
        .message.assistant ul li {
            margin: 0.5em 0;
            padding-left: 0.3em;
        }
        
        .message.assistant ol {
            list-style-type: decimal;
        }
        
        .message.assistant ul {
            list-style-type: disc;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 12px 24px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .send-button:active {
            transform: translateY(0);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            padding: 10px;
            text-align: center;
            font-size: 14px;
            color: #666;
        }
        
        .status.streaming {
            color: #667eea;
            font-weight: bold;
        }
        
        .typing-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite;
            margin-left: 5px;
        }
        
        .typing-indicator:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
            30% { transform: translateY(-10px); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            💬 Aura Chat - Streaming Test
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message assistant">
                Hello! I'm Aura. Ask me anything and watch the response stream in real-time! 🚀
            </div>
        </div>
        
        <div class="status" id="status">Ready</div>
        
        <div class="chat-input-container">
            <input 
                type="text" 
                class="chat-input" 
                id="userInput" 
                placeholder="Type your message here..."
                autocomplete="off"
            >
            <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');
        const status = document.getElementById('status');
        let isStreaming = false;
        
        // Allow Enter key to send
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        function formatMessage(text) {
            if (!text) return '';
            
            let formatted = text;
            
            // Split text into sections (before numbered list, numbered list, after)
            // Detect numbered list pattern: lines starting with "1. ", "2. ", etc.
            const lines = formatted.split('\\n');
            const sections = [];
            let currentSection = { type: 'text', content: [] };
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const listMatch = line.match(/^(\\d+)\\.\\s+(.+)$/);
                
                if (listMatch) {
                    // If we were in text mode, save it
                    if (currentSection.type === 'text' && currentSection.content.length > 0) {
                        sections.push(currentSection);
                        currentSection = { type: 'list', items: [] };
                    }
                    // Ensure we're in list mode
                    if (currentSection.type !== 'list') {
                        currentSection = { type: 'list', items: [] };
                    }
                    currentSection.items.push(listMatch[2]);
                } else {
                    // If we were in list mode, save it
                    if (currentSection.type === 'list' && currentSection.items.length > 0) {
                        sections.push(currentSection);
                        currentSection = { type: 'text', content: [] };
                    }
                    currentSection.content.push(line);
                }
            }
            // Add final section
            if ((currentSection.type === 'text' && currentSection.content.length > 0) ||
                (currentSection.type === 'list' && currentSection.items.length > 0)) {
                sections.push(currentSection);
            }
            
            // Build HTML from sections
            let html = '';
            for (const section of sections) {
                if (section.type === 'list') {
                    html += '<ol>';
                    for (const item of section.items) {
                        // Process bold text in list items
                        const itemFormatted = item.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
                        html += '<li>' + itemFormatted + '</li>';
                    }
                    html += '</ol>';
                } else {
                    const textContent = section.content.join('\\n').trim();
                    if (textContent) {
                        // Process bold text
                        const textFormatted = textContent.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
                        // Split by double line breaks for paragraphs
                        const paragraphs = textFormatted.split('\\n\\n+').filter(p => p.trim());
                        for (const para of paragraphs) {
                            html += '<p>' + para.replace(/\\n/g, '<br>') + '</p>';
                        }
                    }
                }
            }
            
            return html || formatted.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>').replace(/\\n/g, '<br>');
        }
        
        function addMessage(role, text, isStreaming = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}${isStreaming ? ' streaming' : ''}`;
            
            if (role === 'assistant' && !isStreaming) {
                // Format assistant messages with HTML
                messageDiv.innerHTML = formatMessage(text);
            } else {
                // User messages and streaming messages use plain text
                messageDiv.textContent = text;
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return messageDiv;
        }
        
        function updateStatus(text, streaming = false) {
            status.textContent = text;
            status.className = streaming ? 'status streaming' : 'status';
        }
        
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isStreaming) return;
            
            // Add user message to chat
            addMessage('user', message);
            userInput.value = '';
            sendButton.disabled = true;
            isStreaming = true;
            updateStatus('Streaming response...', true);
            
            // Create assistant message element for streaming
            const responseDiv = addMessage('assistant', '', true);
            
            try {
                // Use fetch with streaming
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: message })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                // Read streaming response using EventSource-like parsing
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let accumulated = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    
                    // Keep last incomplete line in buffer
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));
                                accumulated = data.response || accumulated;
                                
                                // Update text content while streaming (plain text)
                                responseDiv.textContent = accumulated;
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                                
                                if (data.done) {
                                    // Format final message with HTML
                                    responseDiv.innerHTML = formatMessage(accumulated);
                                    responseDiv.classList.remove('streaming');
                                    updateStatus('Ready');
                                    sendButton.disabled = false;
                                    isStreaming = false;
                                    userInput.focus();
                                    return;
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e, line);
                            }
                        }
                    }
                }
                
                // Handle any remaining buffer
                if (buffer.trim()) {
                    if (buffer.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(buffer.substring(6));
                            accumulated = data.response || accumulated;
                            responseDiv.textContent = accumulated;
                            if (data.done) {
                                responseDiv.classList.remove('streaming');
                                updateStatus('Ready');
                                sendButton.disabled = false;
                                isStreaming = false;
                                userInput.focus();
                            }
                        } catch (e) {
                            console.error('Error parsing final SSE data:', e);
                        }
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                responseDiv.textContent = 'Sorry, I encountered an error: ' + error.message;
                responseDiv.classList.remove('streaming');
                updateStatus('Error occurred');
                sendButton.disabled = false;
                isStreaming = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template_string(CHAT_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Proxy chat requests to LLM container with streaming"""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    session_id = data.get('session_id', 'web_chat')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    def generate():
        try:
            # Check if LLM container is available
            try:
                health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
                if health_check.status_code != 200:
                    yield f"data: {json.dumps({'response': 'LLM container is not responding. Please ensure the LLM container is running.', 'done': True})}\\n\\n"
                    return
            except requests.exceptions.ConnectionError:
                yield f"data: {json.dumps({'response': f'Cannot connect to LLM container on port {LLM_PORT}. Please ensure the LLM container is running.', 'done': True})}\\n\\n"
                return
            
            # Forward to LLM container with streaming enabled
            response = requests.post(
                CHAT_URL,
                json={
                    'prompt': prompt,
                    'chat_id': session_id,
                    'stream': True  # Enable streaming
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                yield f"data: {json.dumps({'response': f'Error from LLM container: HTTP {response.status_code}', 'done': True})}\\n\\n"
                return
            
            # Stream the SSE response from LLM container
            for line in response.iter_lines():
                if line:
                    # Forward the SSE line as-is
                    decoded_line = line.decode('utf-8')
                    if decoded_line.strip():  # Only yield non-empty lines
                        yield decoded_line + '\n'
                    
        except Exception as e:
            print(f"[WebChat] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'response': 'Error: ' + str(e), 'done': True})}\\n\\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check LLM container health
    llm_healthy = False
    try:
        health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
        llm_healthy = health_check.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'ok' if llm_healthy else 'degraded',
        'service': 'web-chat-interface',
        'llm_url': CHAT_URL,
        'llm_port': LLM_PORT,
        'llm_connected': llm_healthy
    })

def main():
    """Start the web chat server"""
    print("=" * 80)
    print("  💬 Aura Web Chat Interface - Streaming Test")
    print("=" * 80)
    print(f"🌐 Server starting on http://localhost:{CHAT_SERVER_PORT}")
    print(f"🔗 LLM endpoint: {CHAT_URL} (port {LLM_PORT})")
    
    # Verify LLM container is accessible
    try:
        health_check = requests.get(f"http://localhost:{LLM_PORT}/health", timeout=2)
        if health_check.status_code == 200:
            print(f"✅ LLM container is accessible on port {LLM_PORT}")
        else:
            print(f"⚠️  LLM container returned status {health_check.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  WARNING: Cannot connect to LLM container on port {LLM_PORT}")
        print(f"   Please ensure the LLM container is running:")
        print(f"   - Medical mode: docker-compose up llm-medical")
        print(f"   - Generic mode: docker-compose up llm-generic")
    except Exception as e:
        print(f"⚠️  Error checking LLM container: {e}")
    
    print(f"📝 Open your browser and navigate to: http://localhost:{CHAT_SERVER_PORT}")
    print("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=CHAT_SERVER_PORT,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()

