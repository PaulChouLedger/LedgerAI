"""
Memory Container Integration
Helper functions to forward transcriptions to memory container
"""

import os
import requests
import logging

logger = logging.getLogger(__name__)

# Memory container configuration
MEMORY_CONTAINER_URL = os.environ.get("MEMORY_CONTAINER_URL", "http://localhost:11438")
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "true").lower() == "true"

def forward_to_memory(text: str, source: str = "wake_word", metadata: dict = None):
    """
    Forward transcription to memory container for storage and analysis
    
    Args:
        text: Transcribed text
        source: Source of transcription (e.g., "wake_word", "background")
        metadata: Additional metadata
    """
    if not MEMORY_ENABLED:
        return
    
    if not text or not text.strip():
        return
    
    try:
        response = requests.post(
            f"{MEMORY_CONTAINER_URL}/store",
            json={
                "text": text.strip(),
                "source": source,
                "metadata": metadata or {}
            },
            timeout=2  # Short timeout to avoid blocking
        )
        
        if response.status_code == 200:
            logger.debug(f"[Memory] ✅ Forwarded transcription to memory container")
        else:
            logger.warning(f"[Memory] ⚠️ Memory container returned status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        # Silently fail - memory container might not be running
        logger.debug(f"[Memory] Memory container unavailable: {e}")
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

