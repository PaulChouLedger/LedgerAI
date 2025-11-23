#!/usr/bin/env python3
"""
Memory Container REST API
Proactive AI brain component for continuous conversation analysis and suggestions
"""

import os
import sys
import time
import threading
import logging
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Import memory components
from memory_manager import MemoryManager
from proactive_analyzer import ProactiveAnalyzer
from background_listener import BackgroundListener

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable debug logging for memory components
if LOG_LEVEL == "DEBUG":
    logging.getLogger("memory_manager").setLevel(logging.DEBUG)
    logging.getLogger("proactive_analyzer").setLevel(logging.DEBUG)
    logging.getLogger("background_listener").setLevel(logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Service configuration
SERVICE_NAME = "memory-container"
WHISPER_SERVICE_URL = os.environ.get("WHISPER_SERVICE_URL", "http://localhost:5000")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:11434")
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://localhost:11437")  # If TTS has REST API
SPEAKER_ENDPOINT = os.environ.get("SPEAKER_ENDPOINT", None)  # Custom endpoint for TTS

# Memory configuration
MEMORY_DIR = os.environ.get("MEMORY_DIR", "/app/data/memory")
DEVICE_NAME = os.environ.get("AUDIO_DEVICE_NAME", "reSpeaker")

# Global instances
memory_manager: Optional[MemoryManager] = None
analyzer: Optional[ProactiveAnalyzer] = None
listener: Optional[BackgroundListener] = None

# State
listener_enabled = False
last_conversation_text = ""

def initialize_service():
    """Initialize memory service"""
    global memory_manager, analyzer, listener
    
    try:
        logger.info(f"[{SERVICE_NAME}] 🚀 Initializing Memory Container...")
        
        # Initialize memory manager
        logger.info(f"[{SERVICE_NAME}] 🔧 Initializing MemoryManager...")
        memory_manager = MemoryManager(memory_dir=MEMORY_DIR)
        logger.info(f"[{SERVICE_NAME}] ✅ MemoryManager initialized")
        
        # Initialize proactive analyzer
        logger.info(f"[{SERVICE_NAME}] 🔧 Initializing ProactiveAnalyzer...")
        analyzer = ProactiveAnalyzer(
            memory_manager=memory_manager,
            llm_service_url=LLM_SERVICE_URL
        )
        logger.info(f"[{SERVICE_NAME}] ✅ ProactiveAnalyzer initialized")
        
        # Initialize background listener
        logger.info(f"[{SERVICE_NAME}] 🔧 Initializing BackgroundListener...")
        listener = BackgroundListener(
            memory_manager=memory_manager,
            whisper_service_url=WHISPER_SERVICE_URL,
            device_name=DEVICE_NAME,
            on_transcription=_on_transcription_callback
        )
        logger.info(f"[{SERVICE_NAME}] ✅ BackgroundListener initialized")
        
        # Start background listener automatically (always listening)
        logger.info(f"[{SERVICE_NAME}] 🎙️ Starting background listener (always listening)...")
        try:
            listener.start()
            global listener_enabled
            # Check if listener actually started (device might not be found)
            if listener.running:
                listener_enabled = True
                logger.info(f"[{SERVICE_NAME}] ✅ Background listener started - continuously transcribing all conversations")
            else:
                listener_enabled = False
                logger.warning(f"[{SERVICE_NAME}] ⚠️ Background listener failed to start (audio device not found)")
                logger.warning(f"[{SERVICE_NAME}] 💡 Memory container will still receive transcriptions via /store API (wake word forwarding)")
        except Exception as e:
            logger.warning(f"[{SERVICE_NAME}] ⚠️ Failed to start background listener: {e}")
            logger.warning(f"[{SERVICE_NAME}] 💡 Memory container will still receive transcriptions via /store API (wake word forwarding)")
            listener_enabled = False
        
        logger.info(f"[{SERVICE_NAME}] ✅ Memory Container initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False

def _on_transcription_callback(text: str):
    """Callback when transcription is received from background listener"""
    global last_conversation_text
    
    last_conversation_text = text
    logger.info(f"[{SERVICE_NAME}] 📝 Received transcription from background listener: '{text[:80]}...'")
    
    # Store the conversation (background listener transcriptions are always stored)
    if memory_manager:
        try:
            conv_id = memory_manager.store_conversation(
                text=text,
                source="background",
                metadata={"from_background_listener": True}
            )
            logger.info(f"[{SERVICE_NAME}] ✅ Stored background conversation (ID: {conv_id})")
        except Exception as e:
            logger.error(f"[{SERVICE_NAME}] ❌ Failed to store background conversation: {e}")
    
    # Analyze and generate suggestion (proactive suggestions are separate from wake word TTS)
    if analyzer:
        logger.debug(f"[{SERVICE_NAME}] 🔍 Analyzing conversation for proactive suggestions...")
        suggestion = analyzer.analyze_and_suggest(text)
        if suggestion:
            logger.info(f"[{SERVICE_NAME}] 💡 Generated proactive suggestion: '{suggestion[:100]}...'")
            # Send suggestion to TTS (this is separate from wake word TTS responses)
            _speak_suggestion(suggestion)
        else:
            logger.debug(f"[{SERVICE_NAME}] ℹ️ No suggestion generated (cooldown or no insights)")

def _speak_suggestion(suggestion: str):
    """Send suggestion to TTS system"""
    try:
        # Try to use speaker module directly if available (same process)
        try:
            import sys
            # Add parent directory to path to import speaker
            sys.path.insert(0, '/app/../aura-control/core')
            from speaker import enqueue_tts_chunk
            enqueue_tts_chunk(suggestion)
            logger.info(f"[{SERVICE_NAME}] ✅ Suggestion sent to TTS via speaker module")
            return
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Try to send to speaker endpoint if available
        if SPEAKER_ENDPOINT:
            response = requests.post(
                SPEAKER_ENDPOINT,
                json={"text": suggestion, "source": "memory_container"},
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"[{SERVICE_NAME}] ✅ Suggestion sent to TTS")
                return
        
        # Fallback: Write to shared file for main process to pick up
        try:
            suggestion_file = "/shared/memory_suggestion.txt"
            with open(suggestion_file, 'w') as f:
                f.write(suggestion)
            logger.info(f"[{SERVICE_NAME}] ✅ Suggestion written to shared file")
            return
        except Exception as e:
            logger.warning(f"[{SERVICE_NAME}] Failed to write suggestion file: {e}")
        
        # If no TTS endpoint, log the suggestion
        logger.info(f"[{SERVICE_NAME}] 💡 Suggestion (no TTS): {suggestion}")
        
    except Exception as e:
        logger.warning(f"[{SERVICE_NAME}] Failed to send suggestion to TTS: {e}")

# === REST API Endpoints ===

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    stats = memory_manager.get_stats() if memory_manager else {}
    return jsonify({
        "status": "healthy",
        "service": SERVICE_NAME,
        "listener_enabled": listener_enabled,
        "memory_stats": stats
    })

@app.route('/start', methods=['POST'])
def start_listener():
    """Start background listener"""
    global listener_enabled
    
    if not listener:
        return jsonify({"error": "Listener not initialized"}), 500
    
    if listener_enabled:
        return jsonify({"status": "already_running"})
    
    try:
        listener.start()
        listener_enabled = True
        logger.info(f"[{SERVICE_NAME}] ✅ Background listener started")
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to start listener: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_listener():
    """Stop background listener"""
    global listener_enabled
    
    if not listener:
        return jsonify({"error": "Listener not initialized"}), 500
    
    if not listener_enabled:
        return jsonify({"status": "already_stopped"})
    
    try:
        listener.stop()
        listener_enabled = False
        logger.info(f"[{SERVICE_NAME}] ✅ Background listener stopped")
        return jsonify({"status": "stopped"})
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to stop listener: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/store', methods=['POST'])
def store_conversation():
    """Manually store a conversation"""
    if not memory_manager:
        return jsonify({"error": "MemoryManager not initialized"}), 500
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        source = data.get("source", "manual")
        metadata = data.get("metadata", {})
        
        logger.info(f"[{SERVICE_NAME}] 📥 Received conversation to store (source: {source})")
        logger.debug(f"[{SERVICE_NAME}] 📝 Text: '{text[:100]}...'")
        
        if not text:
            logger.warning(f"[{SERVICE_NAME}] ⚠️ Empty text received, skipping")
            return jsonify({"error": "Text is required"}), 400
        
        logger.debug(f"[{SERVICE_NAME}] 💾 Storing conversation in memory manager...")
        conv_id = memory_manager.store_conversation(
            text=text,
            source=source,
            metadata=metadata
        )
        logger.info(f"[{SERVICE_NAME}] ✅ Conversation stored (ID: {conv_id})")
        
        # Analyze for suggestions
        if analyzer:
            logger.debug(f"[{SERVICE_NAME}] 🔍 Analyzing stored conversation for suggestions...")
            suggestion = analyzer.analyze_and_suggest(text)
            if suggestion:
                logger.info(f"[{SERVICE_NAME}] 💡 Suggestion generated: '{suggestion[:80]}...'")
                _speak_suggestion(suggestion)
            else:
                logger.debug(f"[{SERVICE_NAME}] ℹ️ No suggestion generated")
        
        return jsonify({
            "status": "stored",
            "conversation_id": conv_id
        })
        
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] ❌ Failed to store conversation: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_conversations():
    """Search for similar conversations"""
    if not memory_manager:
        return jsonify({"error": "MemoryManager not initialized"}), 500
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        k = data.get("k", 5)
        threshold = data.get("threshold", 0.5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = memory_manager.search_similar(query, k=k, threshold=threshold)
        
        return jsonify({
            "results": [
                {
                    "text": r["conversation"].get("text", ""),
                    "score": r["score"],
                    "timestamp": r["conversation"].get("timestamp"),
                    "source": r["conversation"].get("source")
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recent', methods=['GET'])
def get_recent():
    """Get recent conversations"""
    if not memory_manager:
        return jsonify({"error": "MemoryManager not initialized"}), 500
    
    try:
        hours = request.args.get("hours", 24, type=int)
        limit = request.args.get("limit", 50, type=int)
        
        recent = memory_manager.search_recent(hours=hours, limit=limit)
        
        return jsonify({
            "conversations": recent
        })
        
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to get recent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get memory statistics"""
    if not memory_manager:
        return jsonify({"error": "MemoryManager not initialized"}), 500
    
    try:
        stats = memory_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to get stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_conversation():
    """Manually trigger analysis of a conversation"""
    if not analyzer:
        return jsonify({"error": "Analyzer not initialized"}), 500
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        suggestion = analyzer.analyze_and_suggest(text)
        
        return jsonify({
            "suggestion": suggestion,
            "has_suggestion": suggestion is not None
        })
        
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] Failed to analyze: {e}")
        return jsonify({"error": str(e)}), 500

# === Main ===

if __name__ == "__main__":
    # Initialize service
    if not initialize_service():
        logger.error(f"[{SERVICE_NAME}] ❌ Failed to initialize, exiting")
        sys.exit(1)
    
    # Start Flask app
    port = int(os.environ.get("PORT", 11438))
    logger.info(f"[{SERVICE_NAME}] 🚀 Starting REST API on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)

