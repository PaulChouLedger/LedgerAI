#!/usr/bin/env python3
"""
Simple cache for LLM-generated triage outcomes
Speeds up responses for common condition/severity combinations
"""

from typing import Optional, Dict
import hashlib

# In-memory cache (could be Redis/memcached in production)
_outcome_cache: Dict[str, str] = {}

def get_cache_key(condition: str, severity: str, json_outcome: str) -> str:
    """Generate cache key from condition, severity, and JSON outcome"""
    # Use hash to handle long JSON outcomes
    content = f"{condition}:{severity}:{json_outcome}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def get_cached_outcome(condition: str, severity: str, json_outcome: str) -> Optional[str]:
    """
    Retrieve cached outcome if available
    
    Args:
        condition: Medical condition
        severity: Severity level
        json_outcome: JSON-defined outcome
        
    Returns:
        Cached outcome string or None
    """
    key = get_cache_key(condition, severity, json_outcome)
    outcome = _outcome_cache.get(key)
    
    if outcome:
        print(f"[Cache] ⚡ HIT: Using cached outcome for {condition}/{severity}")
        return outcome
    
    print(f"[Cache] ❌ MISS: No cached outcome for {condition}/{severity}")
    return None

def cache_outcome(condition: str, severity: str, json_outcome: str, outcome: str) -> None:
    """
    Cache an LLM-generated outcome
    
    Args:
        condition: Medical condition
        severity: Severity level
        json_outcome: JSON-defined outcome
        outcome: Generated outcome to cache
    """
    key = get_cache_key(condition, severity, json_outcome)
    _outcome_cache[key] = outcome
    print(f"[Cache] ✅ CACHED: {condition}/{severity} ({len(_outcome_cache)} total)")

def clear_cache() -> None:
    """Clear all cached outcomes"""
    global _outcome_cache
    count = len(_outcome_cache)
    _outcome_cache = {}
    print(f"[Cache] 🗑️ Cleared {count} cached outcomes")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics"""
    return {
        "size": len(_outcome_cache),
        "total_outcomes": len(_outcome_cache)
    }

