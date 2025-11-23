"""
Memory Container Integration
Helper functions to forward transcriptions to memory container
"""

import os
import time
import requests
import logging

logger = logging.getLogger(__name__)

# Memory container configuration
MEMORY_CONTAINER_URL = os.environ.get("MEMORY_CONTAINER_URL", "http://localhost:11438")

def _load_memory_enabled():
    """Load memory enabled setting from app_settings.json or environment variable"""
    # First check environment variable (highest priority)
    env_value = os.environ.get("MEMORY_ENABLED")
    if env_value is not None:
        return env_value.lower() == "true"
    
    # Then check app_settings.json
    try:
        settings_path = os.path.expanduser("~/LedgerAI/data/app_settings.json")
        if os.path.exists(settings_path):
            import json
            with open(settings_path, "r") as f:
                data = json.load(f) or {}
            memory_enabled = data.get("memory_enabled")
            if memory_enabled is not None:
                return bool(memory_enabled)
    except Exception as e:
        logger.debug(f"[Memory] Could not load memory_enabled from settings: {e}")
    
    # Default to True if not set
    return True

MEMORY_ENABLED = _load_memory_enabled()

def reload_memory_enabled():
    """Reload memory enabled setting (called when setting changes)"""
    global MEMORY_ENABLED
    MEMORY_ENABLED = _load_memory_enabled()
    logger.info(f"[Memory] Memory enabled setting reloaded: {MEMORY_ENABLED}")

def forward_to_memory(text: str, source: str = "wake_word", metadata: dict = None):
    """
    Forward transcription to memory container for storage and analysis
    
    Args:
        text: Transcribed text
        source: Source of transcription (e.g., "wake_word", "background")
        metadata: Additional metadata
    """
    if not MEMORY_ENABLED:
        logger.debug("[Memory] Memory forwarding disabled (MEMORY_ENABLED=false)")
        return
    
    if not text or not text.strip():
        logger.debug("[Memory] Empty text, skipping forward")
        return
    
    logger.info(f"[Memory] 📤 Forwarding transcription to memory container (source: {source})")
    logger.debug(f"[Memory] 📝 Text: '{text[:80]}...'")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{MEMORY_CONTAINER_URL}/store",
            json={
                "text": text.strip(),
                "source": source,
                "metadata": metadata or {}
            },
            timeout=2  # Short timeout to avoid blocking
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            conv_id = result.get("conversation_id", "unknown")
            logger.info(f"[Memory] ✅ Forwarded to memory container (ID: {conv_id}, {elapsed:.3f}s)")
        else:
            logger.warning(f"[Memory] ⚠️ Memory container returned status {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.warning(f"[Memory] ⏱️ Memory container request timeout (>2s)")
    except requests.exceptions.RequestException as e:
        # Log warning instead of debug - make it more visible
        logger.warning(f"[Memory] ⚠️ Memory container unavailable: {e}")
        logger.warning(f"[Memory] 💡 Check if memory container is running: curl http://localhost:11438/health")
    except Exception as e:
        logger.warning(f"[Memory] Failed to forward to memory container: {e}")

def check_memory_suggestion():
    """
    Check for proactive suggestions from memory container
    Reads from shared file if memory container writes suggestions there
    
    Returns:
        Suggestion text if available, None otherwise
    """
    try:
        suggestion_file = "/shared/memory_suggestion.txt"
        if os.path.exists(suggestion_file):
            with open(suggestion_file, 'r') as f:
                suggestion = f.read().strip()
            
            # Delete file after reading
            os.remove(suggestion_file)
            
            if suggestion:
                logger.info(f"[Memory] 💡 Received suggestion: {suggestion[:50]}...")
                return suggestion
    except Exception as e:
        logger.debug(f"[Memory] No suggestion available: {e}")
    
    return None

