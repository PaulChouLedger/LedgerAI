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
USE_CLINICIAN_MODE = True  # Enable enhanced clinician mode for medical symptoms
CLINICIAN_FALLBACK_TO_TRIAGE = True  # If clinician fails, use triage
ENABLE_MEDICAL_SYMPTOM_ROUTING = True  # Route medical symptoms to clinician instead of triage


class ConversationMode:
    """Enum for conversation modes"""
    CASUAL = "casual"
    THINKER = "thinker"
    TRIAGE = "triage"
    CLINICIAN = "clinician"


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
    
    # PRIORITY 2: CLINICIAN mode is LOCKED - must complete before switching
    if state.get('mode') == ConversationMode.CLINICIAN:
        print(f"[Router] 🔒 CLINICIAN mode locked - must complete before switching")
        return ConversationMode.CLINICIAN, state
    
    # PRIORITY 3: Dynamic mode switching for CASUAL ↔ THINKER
    # Check if user is asking a knowledge query (regardless of current mode)
    if is_thinker_trigger(prompt):
        print(f"[Router] 🧠 → THINKER mode (knowledge query detected)")
        state['mode'] = ConversationMode.THINKER
        return ConversationMode.THINKER, state
    
    # PRIORITY 4: Check for medical symptoms (route to clinician instead of triage)
    if ENABLE_MEDICAL_SYMPTOM_ROUTING and is_clinician_trigger(prompt):
        print(f"[Router] 🩺 → CLINICIAN mode (medical symptom detected)")
        state.update({
            'mode': ConversationMode.CLINICIAN,
            'chief_complaint': prompt,
            'is_new_clinician': True
        })
        return ConversationMode.CLINICIAN, state

    # PRIORITY 5: Check for NEW medical condition (start triage) - only if not using clinician mode
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
    
    # PRIORITY 6: Simple greetings → CASUAL mode
    if is_casual_trigger(prompt):
        print(f"[Router] 💬 → CASUAL mode (greeting)")
        state['mode'] = ConversationMode.CASUAL
        return ConversationMode.CASUAL, state

    # PRIORITY 7: Default to CASUAL for general conversation
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

