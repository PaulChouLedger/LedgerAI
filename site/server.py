"""
LedgerAI website server — serves static files + proxies /api/chat to Ollama.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
MODEL = os.environ.get('OLLAMA_MODEL', 'mistral:latest')
CHAT_LOG = Path(__file__).parent.parent / 'data' / 'site_chats.jsonl'
CHAT_LOG.parent.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    system = data.get('system', '')
    messages = data.get('messages', [])

    # Build Ollama messages format
    ollama_messages = [{'role': 'system', 'content': system}]
    for msg in messages:
        ollama_messages.append({'role': msg['role'], 'content': msg['content']})

    try:
        resp = requests.post(
            f'{OLLAMA_URL}/api/chat',
            json={
                'model': MODEL,
                'messages': ollama_messages,
                'stream': False,
                'options': {'num_predict': 500},
            },
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        reply = result.get('message', {}).get('content', '')
        # Log the exchange
        try:
            user_msg = messages[-1]['content'] if messages else ''
            entry = {
                'ts': datetime.now(timezone.utc).isoformat(),
                'ip': request.headers.get('CF-Connecting-IP', request.remote_addr),
                'user': user_msg,
                'aura': reply,
                'turns': len(messages),
            }
            with open(CHAT_LOG, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        return jsonify({'reply': reply})
    except requests.Timeout:
        return jsonify({'reply': 'I took too long to think. Try again.'}), 504
    except Exception as e:
        print(f'[server] Ollama error: {e}')
        return jsonify({'reply': 'Could not reach my brain. Try again in a moment.'}), 502

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8888))
    print(f'[LedgerAI Site] Serving on port {port}, Ollama at {OLLAMA_URL}, model={MODEL}')
    app.run(host='0.0.0.0', port=port, debug=False)
