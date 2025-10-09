#!/usr/bin/env python3
"""
Aura Conversation Router - Intelligent Mode Selection

Routes incoming prompts to the appropriate conversation mode:
1. CASUAL - Simple greetings
2. THINKER - Knowledge/information queries  
3. TRIAGE - Hardcoded medical diagnostic system (current baseline)
4. CLINICIAN - RAG-powered intelligent diagnosis (future)

Priority:
1. Check active session state (continue current mode)
2. Classify new prompt and route to appropriate mode
"""

import re
from typing import Tuple, Optional

# Mode trigger imports
from casual import is_casual_trigger
from thinker import is_thinker_trigger
from clinician import is_clinician_trigger

# Feature flags
USE_CLINICIAN_MODE = False  # Set to True when ready to test
CLINICIAN_FALLBACK_TO_TRIAGE = True  # If clinician fails, use triage


class ConversationMode:
    """Enum for conversation modes"""
    CASUAL = "casual"
    THINKER = "thinker"
    TRIAGE = "triage"
    CLINICIAN = "clinician"


def route_prompt(prompt: str, state: dict, session_id: str) -> Tuple[str, dict]:
    """
    Determine which conversation mode to use
    
    Args:
        prompt: User's input (normalized/lowercase)
        state: Current session state
        session_id: Unique session identifier
        
    Returns:
        Tuple of (mode_name, updated_state)
    """
    # Check for active session first (continuation)
    active_mode = get_active_mode(state)
    
    if active_mode:
        print(f"[Router] 🔄 Continuing active session: {active_mode.upper()}")
        return active_mode, state
    
    # No active session - classify new prompt
    print(f"[Router] 🎯 Routing new prompt: '{prompt[:60]}...'")
    
    # Priority order for new prompts:
    
    # 1. CASUAL - Simple greetings (highest priority for UX)
    if is_casual_trigger(prompt):
        print(f"[Router] 💬 → CASUAL mode (greeting)")
        state['mode'] = ConversationMode.CASUAL
        return ConversationMode.CASUAL, state
    
    # 2. THINKER - Knowledge/information queries
    if is_thinker_trigger(prompt):
        print(f"[Router] 🧠 → THINKER mode (knowledge query)")
        state['mode'] = ConversationMode.THINKER
        return ConversationMode.THINKER, state
    
    # 3. CLINICIAN - RAG-powered diagnosis (if enabled)
    if USE_CLINICIAN_MODE and is_clinician_trigger(prompt):
        print(f"[Router] 🩺 → CLINICIAN mode (RAG-powered diagnosis)")
        state.update({
            'mode': ConversationMode.CLINICIAN,
            'chief_complaint': prompt,
            'clinician_history': []
        })
        return ConversationMode.CLINICIAN, state
    
    # 4. TRIAGE - Hardcoded diagnostic system (fallback/baseline)
    # Import here to avoid circular dependency
    from container_rest import detect_condition
    
    condition = detect_condition(prompt, session_id)
    
    if condition:
        print(f"[Router] 🏥 → TRIAGE mode (condition: {condition})")
        state.update({
            'mode': ConversationMode.TRIAGE,
            'condition': condition,
            'step_index': 0,
            'answers': [],
            'flags': {}
        })
        return ConversationMode.TRIAGE, state
    
    # 5. Default to THINKER for anything else (general conversation with potential RAG)
    print(f"[Router] 🧠 → THINKER mode (default for general queries)")
    state['mode'] = ConversationMode.THINKER
    return ConversationMode.THINKER, state


def get_active_mode(state: dict) -> Optional[str]:
    """
    Check if there's an active conversation mode in session state
    
    Args:
        state: Session state dictionary
        
    Returns:
        Active mode name or None
    """
    # Check for active CLINICIAN session
    if state.get('mode') == ConversationMode.CLINICIAN:
        return ConversationMode.CLINICIAN
    
    # Check for active TRIAGE session
    if state.get('condition'):
        return ConversationMode.TRIAGE
    
    # Check for explicit mode marker
    if state.get('mode') in [ConversationMode.CASUAL, ConversationMode.THINKER]:
        # These are stateless, don't persist
        return None
    
    return None


def format_mode_info(mode: str) -> dict:
    """
    Get display information for a mode
    
    Args:
        mode: Mode name
        
    Returns:
        Dict with mode metadata
    """
    mode_info = {
        ConversationMode.CASUAL: {
            'name': 'Casual',
            'icon': '💬',
            'description': 'Simple greetings and small talk',
            'uses_rag': False
        },
        ConversationMode.THINKER: {
            'name': 'Thinker',
            'icon': '🧠',
            'description': 'Knowledge queries with RAG search',
            'uses_rag': True
        },
        ConversationMode.TRIAGE: {
            'name': 'Triage',
            'icon': '🏥',
            'description': 'Structured medical triage (hardcoded)',
            'uses_rag': False
        },
        ConversationMode.CLINICIAN: {
            'name': 'Clinician',
            'icon': '🩺',
            'description': 'Intelligent diagnosis with RAG-powered medical guidelines',
            'uses_rag': True
        }
    }
    
    return mode_info.get(mode, {
        'name': 'Unknown',
        'icon': '❓',
        'description': 'Unknown mode',
        'uses_rag': False
    })

