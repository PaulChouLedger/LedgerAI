"""
LedgerAI website server — serves static files + proxies /api/chat to Ollama.
"""

import os
import re
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

# Telegram live feed — reads from shared log written by the TG bot
import threading
TG_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7958014896:AAHwEVb2ef230LxZ5JwTXRP-FFLo3d4QoPU')
TG_GROUP_ID = -1002111119265
TG_API = f'https://api.telegram.org/bot{TG_TOKEN}'
TG_FEED_FILE = Path(__file__).parent.parent / 'data' / 'tg_feed.jsonl'
tg_messages = []
tg_feed_pos = 0

def tg_feed_reader():
    """Tail the feed file for new messages."""
    global tg_messages, tg_feed_pos
    # On startup, seek to end minus ~20KB to load recent messages only
    if TG_FEED_FILE.exists():
        size = TG_FEED_FILE.stat().st_size
        tg_feed_pos = max(0, size - 20000)
    while True:
        try:
            if TG_FEED_FILE.exists():
                with open(TG_FEED_FILE) as f:
                    f.seek(tg_feed_pos)
                    if tg_feed_pos > 0:
                        f.readline()  # skip partial line after seek
                    new_lines = f.readlines()
                    tg_feed_pos = f.tell()
                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get('chat_id') == TG_GROUP_ID:
                                tg_messages.append(entry)
                                if len(tg_messages) > 100:
                                    tg_messages = tg_messages[-100:]
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f'[TG feed reader] error: {e}')
        time.sleep(2)


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
                'options': {'num_predict': 150},
            },
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        reply = result.get('message', {}).get('content', '')
        # Hallucination filter — catch fabricated partnerships/claims
        KNOWN_ENTITIES = {'auravision', 'ledgerai', 'ledgerx', 'goldman sachs', 'binance',
                          'sprinklr', 'petra capital', 'alphacityai', 'coinmarketcap',
                          'coingecko', 'dextools', 'nvidia', 'jetson', 'ethereum',
                          'paul chou', 'bob carella', 'david lara', 'jorge guinovart',
                          'qwen', 'piper', 'faiss', 'whisper', 'seeed', 'xvf3800'}
        HALLUCINATION_MARKERS = [
            r'collaborat\w*\s+with', r'partner\w*\s+with', r'partnership\s+with',
            r'work\w*\s+with\s+.{3,30}\s+to', r'work\w*\s+together\s+with',
            r'working\s+with\s+the', r'teamed\s+up',
            r'raised\s+\$', r'series\s+[A-C]', r'funding\s+round',
            r'press\s+release', r'prnewswire', r'announced\s+a',
            r'signed\s+a\s+deal', r'agreement\s+with', r'contract\s+with',
            r'invested\s+in', r'acquired', r'merged\s+with',
        ]
        # Also catch any company/org names that aren't in our known list
        FAKE_ORG_MARKERS = [
            r'mayo clinic', r'nhs', r'google', r'apple', r'microsoft', r'amazon',
            r'openai', r'meta\b', r'tesla', r'samsung', r'hospitals?',
            r'university', r'research\s+(center|institute|lab)',
            r'mistral ai', r'created by mistral', r'made by mistral',
        ]
        reply_lower = reply.lower()
        caught = False
        # Check for fabricated partnerships
        for pattern in HALLUCINATION_MARKERS:
            if re.search(pattern, reply_lower):
                context = reply_lower
                if not any(e in context for e in KNOWN_ENTITIES if len(e) > 5):
                    caught = True
                    break
                # Even if known entities present, check if unknown orgs are mentioned
                for org in FAKE_ORG_MARKERS:
                    if re.search(org, reply_lower):
                        caught = True
                        break
            if caught:
                break
        # Also catch direct mentions of fake orgs regardless of partnership language
        if not caught:
            for org in FAKE_ORG_MARKERS:
                if re.search(org, reply_lower):
                    # Check if reply is ABOUT that org (not just mentioning it in passing)
                    if re.search(r'(we|aura|auravision).{0,30}' + org, reply_lower):
                        caught = True
                        break
        if caught:
            reply = "Honestly, I'm not sure about that. I don't want to give you bad info. What I can tell you about is how I work, the team behind AuraVision, or the $LEDGER token — ask me anything on those."
        # Log the exchange and write to unified feed
        try:
            user_msg = messages[-1]['content'] if messages else ''
            ts_now = int(time.time())
            entry = {
                'ts': datetime.now(timezone.utc).isoformat(),
                'ip': request.headers.get('CF-Connecting-IP', request.remote_addr),
                'user': user_msg,
                'aura': reply,
                'turns': len(messages),
            }
            with open(CHAT_LOG, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            # Write to TG feed so it shows in the unified chat stream
            with open(TG_FEED_FILE, 'a') as f:
                f.write(json.dumps({'name': 'Visitor', 'text': user_msg, 'ts': ts_now,
                                    'is_bot': False, 'chat_id': TG_GROUP_ID, 'source': 'web'}) + '\n')
                f.write(json.dumps({'name': 'Aura', 'text': reply, 'ts': ts_now + 1,
                                    'is_bot': True, 'chat_id': TG_GROUP_ID, 'source': 'web'}) + '\n')
        except Exception:
            pass
        return jsonify({'reply': reply})
    except requests.Timeout:
        return jsonify({'reply': 'I took too long to think. Try again.'}), 504
    except Exception as e:
        print(f'[server] Ollama error: {e}')
        return jsonify({'reply': 'Could not reach my brain. Try again in a moment.'}), 502

@app.route('/api/feed')
def feed():
    return jsonify(list(tg_messages))

if __name__ == '__main__':
    t = threading.Thread(target=tg_feed_reader, daemon=True)
    t.start()
    print('[TG feed reader] started')
    port = int(os.environ.get('PORT', 8888))
    print(f'[LedgerAI Site] Serving on port {port}, Ollama at {OLLAMA_URL}, model={MODEL}')
    app.run(host='0.0.0.0', port=port, debug=False)
