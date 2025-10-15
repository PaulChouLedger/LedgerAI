#!/usr/bin/env python3
"""
Aura Conversation Router - Intelligent Mode Selection

Routes incoming prompts to the appropriate conversation mode:
1. CASUAL - Simple greetings and general conversation
2. THINKER - Non-medical knowledge queries with RAG
3. UNIFIED_MEDICAL - All medical interactions (symptoms + knowledge)
4. TRIAGE - Fallback medical diagnostic system

Priority:
1. Check active session state (continue current mode)
2. Classify new prompt and route to appropriate mode
"""

import re
from typing import Tuple, Optional

# Mode trigger imports
from casual import is_casual_trigger
from thinker import is_thinker_trigger
from unified_medical_mode import is_unified_medical_trigger

# Feature flags
USE_CLINICIAN_MODE = True  # Enable enhanced clinician mode for medical symptoms
CLINICIAN_FALLBACK_TO_TRIAGE = True  # If clinician fails, use triage
ENABLE_MEDICAL_SYMPTOM_ROUTING = True  # Route medical symptoms to clinician instead of triage


class ConversationMode:
    """Enum for conversation modes"""
    CASUAL = "casual"
    THINKER = "thinker"
    TRIAGE = "triage"
    UNIFIED_MEDICAL = "unified_medical"


def route_prompt(prompt: str, state: dict, session_id: str, llm_chat_fn=None) -> Tuple[str, dict]:
    """
    Determine which conversation mode to use
    
    Supports dynamic mode switching:
    - CASUAL ↔ THINKER: Can switch freely based on query type
    - TRIAGE: Locked until completion (must finish all questions)
    - CLINICIAN: Locked until completion (future)
    
    Args:
        prompt: User's input (normalized/lowercase)
        state: Current session state
        session_id: Unique session identifier
        
    Returns:
        Tuple of (mode_name, updated_state)
    """
    # Debug: Show state info
    print(f"[Router] 🔍 Current state: condition={state.get('condition')}, step_index={state.get('step_index')}, mode={state.get('mode')}")
    
    # PRIORITY 1: TRIAGE mode is LOCKED - must complete before switching
    active_condition = state.get('condition')
    if active_condition:
        print(f"[Router] 🔒 TRIAGE mode locked (condition={active_condition}) - must complete before switching")
        state['mode'] = ConversationMode.TRIAGE
        return ConversationMode.TRIAGE, state
    
    # PRIORITY 2: UNIFIED_MEDICAL mode is LOCKED - must complete before switching
    if state.get('mode') == ConversationMode.UNIFIED_MEDICAL:
        print(f"[Router] 🔒 UNIFIED_MEDICAL mode locked - must complete before switching")
        return ConversationMode.UNIFIED_MEDICAL, state
    
    # PRIORITY 3: Check for unified medical mode (handles both symptoms and medical knowledge)
    if is_unified_medical_trigger(prompt):
        print(f"[Router] 🩺 → UNIFIED_MEDICAL mode (medical query detected)")
        state['mode'] = ConversationMode.UNIFIED_MEDICAL
        return ConversationMode.UNIFIED_MEDICAL, state

    # PRIORITY 4: Dynamic mode switching for CASUAL ↔ THINKER
    # Check if user is asking a knowledge query (regardless of current mode)
    if is_thinker_trigger(prompt):
        print(f"[Router] 🧠 → THINKER mode (knowledge query detected)")
        state['mode'] = ConversationMode.THINKER
        return ConversationMode.THINKER, state
    
    # PRIORITY 5: Check for NEW medical condition (start triage) - only if unified medical didn't catch it
    from triage import detect_condition
    condition = detect_condition(prompt, session_id, llm_chat_fn)
    if condition:
        print(f"[Router] 🏥 → TRIAGE mode (NEW condition: {condition})")
        state.update({
            'mode': ConversationMode.TRIAGE,
            'condition': condition,
            'step_index': 0,
            'answers': [],
            'flags': {},
            'is_new_triage': True
        })
        return ConversationMode.TRIAGE, state

    # PRIORITY 7: Simple greetings → CASUAL mode
    if is_casual_trigger(prompt):
        print(f"[Router] 💬 → CASUAL mode (greeting)")
        state['mode'] = ConversationMode.CASUAL
        return ConversationMode.CASUAL, state

    # PRIORITY 8: Default to CASUAL for general conversation
    print(f"[Router] 💬 → CASUAL mode (general conversation)")
    state['mode'] = ConversationMode.CASUAL
    return ConversationMode.CASUAL, state


def get_active_mode(state: dict) -> Optional[str]:
    """
    Check if there's an active conversation mode in session state
    
    Args:
        state: Session state dictionary
        
    Returns:
        Active mode name or None
    """
    # Check for active UNIFIED_MEDICAL session
    if state.get('mode') == ConversationMode.UNIFIED_MEDICAL:
        return ConversationMode.UNIFIED_MEDICAL
    
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
            'description': 'Structured medical triage (fallback)',
            'uses_rag': False
        },
        ConversationMode.UNIFIED_MEDICAL: {
            'name': 'Medical Assistant',
            'icon': '🩺',
            'description': 'Comprehensive physician-like medical assistance (symptoms + knowledge)',
            'uses_rag': True
        }
    }
    
    return mode_info.get(mode, {
        'name': 'Unknown',
        'icon': '❓',
        'description': 'Unknown mode',
        'uses_rag': False
    })

