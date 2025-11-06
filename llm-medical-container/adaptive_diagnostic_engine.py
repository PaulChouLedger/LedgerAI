#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - Minimal Universal Version

UNIVERSAL FLOW:
1. Chief complaint → Unified function with chief complaint synonyms → Match category → Narrow guidelines
2. Parse prompt → Detect answered OLDCARTS elements
3. For each element:
   - Location: Extract location → Filter using medical_rules.json → Parse user response array → 
     Compare to structured_oldcarts → Generate clarifying question if missing → Use unified function to score
   - Other elements: Use unified function directly
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import modular RAG client
from rag import get_rag_client
import numpy as np
import faiss

# Import fuzzy medical matcher for typo correction
from fuzzy_medical_matcher import FuzzyMedicalMatcher

class RAGEmbeddingAPI:
    """Wrapper for RAG client's embedding service"""
    
    def __init__(self, rag_url: str = "http://localhost:11435"):
        self.rag_client = get_rag_client()
    
    def encode(self, texts: List[str]) -> List:
        embeddings = self.rag_client.embed(texts)
        if embeddings:
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        else:
            raise RuntimeError(f"RAG embedding failed")

# RAG client availability check
try:
    rag_api = RAGEmbeddingAPI()
    test_embedding = rag_api.encode(["test"])
    RAG_API_AVAILABLE = True
    rag_client = get_rag_client()
    print(f"[Engine] ✅ RAG client available - using {rag_client.get_mode()}")
except Exception as e:
    RAG_API_AVAILABLE = False
    print(f"[Engine] ⚠️ RAG client not available: {e}")


class AdaptiveDiagnosticEngine:
    """
    Adaptive Diagnostic Engine - Organized by Assessment Flow
    
    ASSESSMENT ALGORITHM SECTIONS:
    1. CONFIGURATION (Top) - All thresholds, LLM rules, weights for easy tuning
    2. INITIALIZATION - Setup and loading
    3. CHIEF COMPLAINT - Category matching and narrowing
    4. DEMOGRAPHICS - Age, sex, chronicity extraction
    5. OLDCARTS PROCESSING - Location, onset, duration, character, etc.
    6. SCORING - Guideline scoring and ranking
    7. QUESTION GENERATION - Asking next questions
    8. UTILITIES - Helper functions (including all debug functions)
    """
    
    # ============================================================================
    # SECTION 1: CONFIGURATION (Top - Easy Tuning)
    # ============================================================================
    
    # ===== THRESHOLD CONFIGURATION =====
    # FAISS semantic matching thresholds (for easy tuning)
    FAISS_SEMANTIC_THRESHOLD = 0.75  # Main threshold for FAISS semantic matching (location, character, etc.)
    FAISS_ASSOCIATED_THRESHOLD = 0.70  # Threshold for associated symptoms matching
    FAISS_RADIATION_THRESHOLD = 0.45  # Threshold for radiation matching
    # REMOVED: FAISS_STRICT_THRESHOLD - only used in removed _parse_prompt_against_structured_oldcarts function
    
    # Chief complaint matching thresholds
    CHIEF_COMPLAINT_FAISS_THRESHOLD = 0.75  # FAISS threshold for chief complaint matching
    CHIEF_COMPLAINT_NEAR_MISS_LOWER = 0.5  # Lower bound for near-miss candidates
    CHIEF_COMPLAINT_NEAR_MISS_UPPER = 0.6  # Upper bound for near-miss candidates (fuzzy matching)
    CHIEF_COMPLAINT_FUZZY_THRESHOLD = 0.8  # Fuzzy matching threshold for typos/near-misses
    
    # ML learning confidence threshold
    ML_CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence for ML learning recording
    
    # ===== LLM RULES & GUIDELINES =====
    # All LLM prompts, system messages, and guidance text for easy tuning
    
    # Clarification Question Generation
    LLM_CLARIFICATION_SYSTEM_MSG = "You are a medical assistant conducting a telehealth interview. Generate a natural, grammatically correct clarification question that flows well. Use proper grammar with 'and' and 'or' to connect options naturally. Return ONLY the question - no explanations, no reasoning, no additional text."
    
    LLM_CLARIFICATION_LOCATION_RULES = """CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You MUST use ONLY the terms from the list above - do NOT create new terms like "one side", "both sides", "upper", "lower" unless they are in the list
2. If the patient already mentioned a location (like "right side"), do NOT ask about that same location again
3. Use "or" to connect the options naturally with proper grammar
4. You MUST include at least 3-5 of the provided options in your question - do NOT ask "Can you be more specific?" without including specific options
5. Format: "Can you be more specific? For example, is it located at [option1], [option2], [option3], or [option4]?" - the options list is REQUIRED, not optional
6. Return ONLY the question - no explanations, no reasoning, no "Here's a question:", no "Alternatively:", no additional text"""
    
    LLM_CLARIFICATION_GENERAL_RULES = """CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You MUST use ONLY the terms from the list above - do NOT create new terms
2. If the patient already mentioned something, do NOT ask about that same thing again
3. Use "or" to connect the options naturally with proper grammar
4. You MUST include at least 3-5 of the provided options in your question - do NOT ask "Can you be more specific?" without including specific options
5. Format: "Can you be more specific? For example, is it [option1], [option2], [option3], or [option4]?" - the options list is REQUIRED, not optional
6. Return ONLY the question - no explanations, no reasoning, no "Here's a question:", no "Alternatively:", no additional text"""
    
    # OLDCARTS Question Generation
    LLM_OLDCARTS_SYSTEM_MSG = "You are a medical assistant conducting a telehealth interview. Generate a simple, direct question following the example exactly. Use the chief complaint and conversation context to make the question relevant. Do NOT add assumptions, examples, or extra details. Keep it short and open-ended."
    
    LLM_OLDCARTS_STRICT_INSTRUCTIONS = """CRITICAL RULES:
- Follow the example question structure EXACTLY
- Use the chief complaint context to make it relevant
- Keep it simple and direct
- Do NOT add assumptions or specific examples
- Do NOT mention body parts unless asking about location
- Use simple language
- Return ONLY the question, no explanation"""
    
    LLM_OLDCARTS_COMPONENT_GUIDANCE = {
        'location': "Ask ONLY 'Where exactly is the pain located?' or similar. Do NOT mention body parts or give examples. Do NOT ask about intensity or duration.",
        'severity': "Ask ONLY the EXACT question 'On a scale of 1 to 10, how would you rate this?' or very similar wording. Do NOT ask about location or other qualities. Do NOT return just a number.",
        'aggravating': "Ask ONLY 'What makes it worse?' or similar. Do NOT assume specific activities or body parts. Do NOT use words like 'triggers' or 'causes'. Keep it simple.",
        'relieving': "Ask ONLY 'What helps or makes it better?' or similar. Do NOT assume specific treatments or positions. Keep it simple.",
        'timing': "Ask ONLY 'Is it constant or does it come and go?' or similar. Do NOT add details.",
        'duration': "Ask ONLY 'How long does each episode typically last?' or similar. Do NOT add details."
    }
    
    # Character Component Analysis
    LLM_CHARACTER_DEFAULT_QUESTION = "What does it feel like?"
    LLM_CHARACTER_DEFAULT_GUIDANCE = "Ask ONLY 'What does it feel like?' or similar. Do NOT mention any specific qualities like 'sharp', 'sharpness', 'burning', etc. Do NOT ask about location, intensity, or duration. Keep it completely open-ended."
    
    LLM_CHARACTER_DESCRIPTIVE_AND_SENSORY_GUIDANCE = "Ask about both appearance/description AND how it feels. You can ask 'Can you describe what it looks like?' or 'What does it look like?' for visual/descriptive characteristics, and also 'What does it feel like?' for sensory qualities. Do NOT mention specific examples like 'bright red', 'coffee ground', 'sharp', 'dull', etc. Keep it open-ended."
    
    LLM_CHARACTER_DESCRIPTIVE_ONLY_GUIDANCE = "Ask about appearance or description. You can ask 'What does it look like?' or 'Can you describe what you see?' or similar. Do NOT mention specific examples like 'bright red', 'coffee ground', 'black tarry', etc. Keep it open-ended and focused on visual/descriptive characteristics."
    
    # Empathetic Response Generation
    LLM_EMPATHETIC_SYSTEM_MSG = """You are a compassionate medical assistant. The patient is expressing significant distress with severe symptoms. 
Generate a brief (1-2 sentences), empathetic response that:
1. Acknowledges their distress and validates their feelings
2. Reassures them you're taking this seriously
3. Shows urgency and concern

Be warm, professional, and reassuring. Do NOT ask any questions - just acknowledge and reassure. The system will ask the appropriate clinical question next."""
    
    LLM_EMPATHETIC_USER_TEMPLATE = """{chief_complaint_context}{conversation_context}

Patient just said: "{user_answer}"

Distress detected: severity {severity:.1f}/10

Generate an empathetic response that acknowledges their distress and reassures them. Do NOT ask any questions - just provide emotional support and reassurance."""
    
    # Confirmation Message Generation
    LLM_CONFIRMATION_SYSTEM_MSG = "You are a medical assistant. Generate a brief confirmation message paraphrasing what the patient just told you to show you understand."
    
    LLM_CONFIRMATION_USER_TEMPLATE = "{chief_complaint_context}\n\nPatient just said: '{user_answer}'\n\nGenerate a brief confirmation message (1-2 sentences) that paraphrases what they told you to confirm understanding. Make it natural and empathetic. Return only the confirmation message, no other text."
    
    # Chronicity Question Generation
    LLM_CHRONICITY_SYSTEM_MSG = "You are a medical assistant. Generate a concise question to ask if the patient's problem is new or ongoing."
    LLM_CHRONICITY_USER_MSG = "Is this a new problem or an ongoing issue?"
    
    # Question Acknowledgment Generation
    LLM_QUESTION_ACK_SYSTEM_MSG = "You are a compassionate medical assistant. Generate a brief, natural acknowledgment (1 sentence) for a patient's question. Be warm and reassuring."
    LLM_QUESTION_ACK_USER_TEMPLATE = "Patient asked: '{user_input}'\n\nGenerate a brief acknowledgment:"
    
    # Comment Acknowledgment Generation
    LLM_COMMENT_ACK_SYSTEM_MSG = "You are a compassionate medical assistant. Generate a brief, natural acknowledgment (1 sentence) for a patient's comment or emotional expression. Be warm and reassuring, then naturally transition back to gathering information."
    LLM_COMMENT_ACK_USER_TEMPLATE = "Patient said: '{user_input}'\n\nGenerate a brief acknowledgment:"
    
    # Category to organ system mapping (reused throughout)
    CATEGORY_TO_SYSTEM = {
        'gastrointestinal': 'GI', 'cardiovascular': 'CARDIO',
        'respiratory': 'PULMONARY', 'neurological': 'NEURO',
        'musculoskeletal': 'MSK', 'renal': 'RENAL',
        'genitourinary': 'GU', 'gynecological': 'GYN',
        'dermatological': 'DERM'
    }
    
    @staticmethod
    def get_oldcarts_element_weight(category: str, oldcarts_element: str) -> float:
        """
        Get the weight/impact of an OLDCARTS element on scoring for a given category.
        
        Higher weights mean the element has more impact on the final score.
        Weights are used in the formula: new_score = (old_score * (1 - weight)) + (similarity * weight)
        
        Returns a weight between 0.0 and 1.0. Default is 0.3 (standard impact).
        
        Args:
            category: Category name (e.g., 'gastrointestinal', 'cardiovascular')
            oldcarts_element: OLDCARTS element name (e.g., 'location', 'character', 'onset')
        
        Returns:
            Weight value (0.0-1.0) indicating how much this element affects scoring
        """
        # Default weights (can be overridden per category)
        default_weight = 0.3
        
        # Category-specific element weights
        # Format: {category: {element: weight}}
        weights = {
            'gastrointestinal': {
                'location': 0.65,      # Location is critical for GI (RUQ vs RLQ vs epigastric)
                'character': 0.20,      # Character is moderately important
                'aggravating': 0.30,    # Aggravating factors (food, movement)
                'relieving': 0.30,      # Relieving factors
                'onset': 0.25,          # Onset timing
                'timing': 0.25,         # Timing (constant vs episodic)
                'duration': 0.25,       # Duration
                'severity': 0.20,       # Severity less critical
                'associated': 0.25      # Associated symptoms
            },
            'cardiovascular': {
                'character': 0.65,      # Character is critical (heavy, crushing, pressure vs sharp, stabbing)
                'location': 0.30,       # Location (chest, substernal, left side)
                'aggravating': 0.65,    # Aggravating (exertion, stress)
                'relieving': 0.35,      # Relieving (rest, nitroglycerin)
                'onset': 0.30,          # Onset timing
                'timing': 0.25,         # Timing
                'duration': 0.30,       # Duration (important for angina vs MI)
                'severity': 0.25,       # Severity
                'associated': 0.30      # Associated (SOB, diaphoresis, nausea)
            },
            'respiratory': {
                'character': 0.35,      # Character (sharp, pleuritic, dull)
                'location': 0.30,       # Location (chest, unilateral, bilateral)
                'aggravating': 0.30,     # Aggravating (breathing, coughing)
                'relieving': 0.25,      # Relieving
                'onset': 0.30,          # Onset
                'timing': 0.25,          # Timing
                'duration': 0.25,        # Duration
                'severity': 0.25,       # Severity
                'associated': 0.35      # Associated (cough, SOB, fever) - very important
            },
            'neurological': {
                'character': 0.35,      # Character (throbbing, sharp, pressure)
                'location': 0.30,        # Location (unilateral, bilateral, frontal, occipital)
                'aggravating': 0.30,     # Aggravating (light, sound, movement)
                'relieving': 0.30,      # Relieving
                'onset': 0.35,          # Onset (sudden vs gradual) - important for stroke
                'timing': 0.25,          # Timing
                'duration': 0.30,        # Duration
                'severity': 0.25,        # Severity
                'associated': 0.35      # Associated (nausea, photophobia, neurological deficits)
            },
            'musculoskeletal': {
                'location': 0.35,        # Location (specific joint, limb)
                'character': 0.30,       # Character (sharp, dull, aching)
                'aggravating': 0.35,     # Aggravating (movement, weight-bearing) - critical
                'relieving': 0.35,       # Relieving (rest, ice, elevation)
                'onset': 0.30,           # Onset (trauma vs insidious)
                'timing': 0.25,          # Timing
                'duration': 0.25,        # Duration
                'severity': 0.25,        # Severity
                'associated': 0.25       # Associated
            },
            'renal': {
                'location': 0.35,         # Location (flank, unilateral, bilateral)
                'character': 0.30,        # Character (colicky, sharp, dull)
                'aggravating': 0.25,     # Aggravating
                'relieving': 0.25,       # Relieving
                'onset': 0.30,           # Onset
                'timing': 0.25,          # Timing
                'duration': 0.30,         # Duration
                'severity': 0.30,        # Severity
                'associated': 0.35       # Associated (hematuria, dysuria, fever)
            },
            'genitourinary': {
                'location': 0.30,        # Location
                'character': 0.30,       # Character
                'aggravating': 0.25,     # Aggravating
                'relieving': 0.25,       # Relieving
                'onset': 0.30,           # Onset
                'timing': 0.25,          # Timing
                'duration': 0.30,         # Duration
                'severity': 0.30,        # Severity
                'associated': 0.35       # Associated (very important for GU)
            },
            'gynecological': {
                'location': 0.30,        # Location (pelvic, lower abdomen)
                'character': 0.30,        # Character
                'aggravating': 0.25,     # Aggravating
                'relieving': 0.25,       # Relieving
                'onset': 0.30,           # Onset
                'timing': 0.30,          # Timing (relation to menses)
                'duration': 0.25,        # Duration
                'severity': 0.25,        # Severity
                'associated': 0.35       # Associated (very important)
            },
            'dermatological': {
                'location': 0.35,        # Location (specific body region)
                'character': 0.30,        # Character (itching, burning, pain)
                'aggravating': 0.30,     # Aggravating (sun, heat, contact)
                'relieving': 0.30,       # Relieving
                'onset': 0.30,           # Onset
                'timing': 0.25,          # Timing
                'duration': 0.30,         # Duration
                'severity': 0.25,        # Severity
                'associated': 0.30       # Associated
            }
        }
        
        # Get category-specific weights, or use default
        category_weights = weights.get(category, {})
        return category_weights.get(oldcarts_element, default_weight)
    
    # ============================================================================
    # SECTION 2: INITIALIZATION - Setup and Loading
    # ============================================================================
    
    def __init__(self, guidelines_dir: str = None, llm_chat_fn=None, embedding_model=None, llm_chat_simple_fn=None):
        # Auto-detect guidelines directory
        if guidelines_dir is None:
            if os.path.exists("/app/medical/guidelines"):
                guidelines_dir = "/app/medical/guidelines"
            elif os.path.exists("medical/guidelines"):
                guidelines_dir = "medical/guidelines"
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "medical", "guidelines")):
                guidelines_dir = os.path.join(os.path.dirname(__file__), "medical", "guidelines")
            else:
                raise RuntimeError("Could not find medical guidelines directory")
        
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn
        self.llm_chat_simple_fn = llm_chat_simple_fn or llm_chat_fn
        self.embedding_model = embedding_model
        
        # Configuration - read all LLM settings from environment
        self.temperature = float(os.environ.get('LLM_TEMPERATURE_SIMPLE'))
        self.top_p = float(os.environ.get('LLM_TOP_P'))
        self.top_k = int(os.environ.get('LLM_TOP_K'))
        self.repeat_penalty = float(os.environ.get('LLM_REPEAT_PENALTY'))
        self.presence_penalty = float(os.environ.get('LLM_PRESENCE_PENALTY'))
        self.frequency_penalty = float(os.environ.get('LLM_FREQUENCY_PENALTY'))
        self.num_predict = int(os.environ.get('LLM_NUM_PREDICT'))
        
        # Stop sequences
        stop_env = os.environ.get('LLM_STOP', '').strip()
        self.stop_sequences = [s for s in stop_env.split(',') if s] if stop_env else None
        
        # Initialize debug capture
        self._captured_debug_output = []
        self.current_category = None
        
        def _get_llm_kwargs(override_max_tokens=None):
            """Get all LLM parameters for generation"""
            kwargs = {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'repeat_penalty': self.repeat_penalty,
                'presence_penalty': self.presence_penalty,
                'frequency_penalty': self.frequency_penalty,
            }
            if override_max_tokens:
                kwargs['max_tokens'] = override_max_tokens
            elif self.num_predict:
                kwargs['max_tokens'] = self.num_predict
            if self.stop_sequences:
                kwargs['stop'] = self.stop_sequences
            return kwargs
        
        self._get_llm_kwargs = _get_llm_kwargs
        
        # Initialize Medical Rule Engine
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
            self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized")
            
            # No need for separate anatomical FAISS index - medical_rules.json handles this
        except ImportError:
            try:
                import sys
                sys.path.append('/app/ml')
                from medical_rule_engine import MedicalRuleEngine
                self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
                self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized (alternative path)")
                
                # No need for separate anatomical FAISS index - medical_rules.json handles this
            except ImportError:
                self.medical_rule_engine = None
                self._capture_debug(f"[Engine] ⚠️ Medical Rule Engine not available")
        
        # Initialize fuzzy matcher
        self.fuzzy_matcher = FuzzyMedicalMatcher()
        
        # RAG API
        self.rag_api = RAGEmbeddingAPI() if RAG_API_AVAILABLE else None
        self.use_rag_api = RAG_API_AVAILABLE
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # No separate anatomical FAISS index needed - medical_rules.json handles this
        
        # Pre-build chief complaint trigger index for category matching
        self.chief_complaint_triggers_index = None
        self.chief_complaint_triggers_data = []  # List of {trigger, category, condition}
        self._build_chief_complaint_triggers_index()
        
        # REMOVED: ML learning feature - disabled and not used in production
        
        # Initialize assessment state
        self.demographics_optional = False  # Reserved for future use (not currently used to skip questions)
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files, filtered by ENABLED_MEDICAL_CATEGORIES"""
        if not self.guidelines_dir.exists():
            return
        
        # Get enabled categories from environment variable (comma-separated, e.g., "GI" or "GI,CARDIO")
        # Default to "GI" if not set (user has only curated GI so far)
        enabled_categories_env = os.environ.get('ENABLED_MEDICAL_CATEGORIES', 'GI').strip()
        enabled_categories = [cat.strip().upper() for cat in enabled_categories_env.split(',') if cat.strip()]
        
        if enabled_categories:
            self._capture_debug(f"[Engine] 📋 Loading guidelines for categories: {', '.join(enabled_categories)}")
        else:
            self._capture_debug(f"[Engine] ⚠️ No enabled categories specified - loading all guidelines")
        
        loaded_count = 0
        skipped_count = 0
        
        for json_file in sorted(self.guidelines_dir.glob("**/*.json")):
            try:
                # Extract organ system from directory structure
                organ_system = json_file.parent.name if json_file.parent != self.guidelines_dir else "Other"
                
                # Filter by enabled categories (if any are specified)
                if enabled_categories and organ_system.upper() not in enabled_categories:
                    skipped_count += 1
                    continue
                
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    name = guideline.get('condition', json_file.stem)
                    # Store organ system from directory structure
                    guideline['organ_system'] = organ_system  # Store for filtering
                    self.all_guidelines[name] = guideline
                    loaded_count += 1
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
    
        if enabled_categories:
            self._capture_debug(f"[Engine] ✅ Loaded {loaded_count} guidelines ({skipped_count} skipped from disabled categories)")
        else:
            self._capture_debug(f"[Engine] ✅ Loaded {loaded_count} guidelines")
    
    
    # ============================================================================
    # SECTION 3: CHIEF COMPLAINT - Category Matching and Narrowing
    # ============================================================================
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        UNIVERSAL FLOW - Simplified:
        1. Chief complaint → Apply fuzzy matching (typo correction) → Semantic similarity matching to chief complaint triggers from all guidelines
        2. Match category
        3. Narrow guidelines
        
        Note: OLDCARTS parsing from chief complaint is removed for simplicity (can be added later)
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (SIMPLIFIED FLOW)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # STEP 0: Apply fuzzy matching to correct typos in chief complaint (ALWAYS RUN)
        if self.fuzzy_matcher:
            original_complaint = chief_complaint
            chief_complaint = self.fuzzy_matcher.fuzzy_correct_medical_terms(chief_complaint)
            if chief_complaint != original_complaint:
                self._capture_debug(f"[Fuzzy] 🔄 Corrected chief complaint typos: '{original_complaint}' → '{chief_complaint}'")
        
        # STEP 1: Semantic similarity matching to chief complaint triggers → match category
        category = self._match_chief_complaint_to_category(chief_complaint)
        self.current_category = category
        self._capture_debug(f"[Engine] 🎯 Category: {category}")
        
        # Switch FAISS indexes to category-specific once category is determined
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'set_active_category'):
            self._capture_debug(f"[Engine] 🔀 Switching FAISS indexes to {category} category...")
            self.medical_rule_engine.set_active_category(category)
            self._capture_debug(f"[Engine] ✅ FAISS indexes switched to {category} category")
        
        # STEP 2: Narrow down guidelines to matched category
        matched_guidelines = self._get_all_guidelines_in_category(category)
        self._capture_debug(f"[Engine] 📊 Found {len(matched_guidelines)} guidelines in {category}")
        self._capture_debug(f"[Guideline Load] 📚 Conditions: {[g.get('name', 'Unknown') for g in matched_guidelines[:10]]}")
        
        # Initialize assessment
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        self.active_guidelines = matched_guidelines[:self.MAX_ACTIVE]
        self.reserve_pool = matched_guidelines[self.MAX_ACTIVE:]
        
        self._capture_debug(f"[Initial Pool] 🎯 Active: {len(self.active_guidelines)}, Reserve: {len(self.reserve_pool)}")
        self._capture_debug(f"[Initial Pool] 🏆 Active: {[g.get('name', 'Unknown') for g in self.active_guidelines]}")
        
        # Start with empathetic statement + chronicity question
        empathetic_msg = self._generate_empathetic_statement()
        
        self.conversation_history.append({
            'type': 'statement',
            'message': empathetic_msg
        })
        
        # Generate chronicity question immediately
        chronicity_response = self._generate_ml_first_question_with_demographics()
        chronicity_question = chronicity_response.get('message', '')
        
        return {
            'success': True,
            'message': empathetic_msg,
            'question': chronicity_question,
            'status': 'questioning',
            'has_pause': True,  # Pause between statement and question
            'debug': {
                'engine': self._format_engine_debug("[Engine] 🧠 Generating first question with chronicity..."),
                'internal': self._get_debug_info()
            }
        }
    
    def _build_chief_complaint_triggers_index(self):
        """Pre-build FAISS index for chief_complaint_triggers from all guidelines"""
        if not self.embedding_model:
            self._capture_debug("[Engine] ⚠️ No embedding model for chief complaint triggers index")
            return
        
        try:
            triggers = []
            for name, guideline in self.all_guidelines.items():
                triggers_list = guideline.get('chief_complaint_triggers', [])
                category = self._get_guideline_category(guideline)
                
                for trigger in triggers_list:
                    self.chief_complaint_triggers_data.append({
                        'trigger': trigger,
                        'category': category,
                        'condition': name
                    })
                    triggers.append(trigger)
            
            if triggers:
                # Build FAISS index for all triggers
                embeddings = self.embedding_model.encode(triggers)
                dimension = len(embeddings[0])
                self.chief_complaint_triggers_index = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                embeddings_np = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_np)
                self.chief_complaint_triggers_index.add(embeddings_np)
                
                self._capture_debug(f"[Engine] ✅ Built chief complaint triggers index: {len(triggers)} triggers from {len(set(g['category'] for g in self.chief_complaint_triggers_data))} categories")
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Failed to build chief complaint triggers index: {e}")
            self.chief_complaint_triggers_index = None
    
    def _get_guideline_category(self, guideline: Dict) -> str:
        """Extract category from guideline (from directory structure or organ_system)"""
        # Try to get from organ_system stored during loading
        organ_system = guideline.get('organ_system', '')
        if organ_system:
            # Reverse map: organ_system -> category
            for cat, sys in self.CATEGORY_TO_SYSTEM.items():
                if sys == organ_system:
                    return cat
        
        # Fallback: try to infer from guideline name/path
        return 'gastrointestinal'  # Default
    
    def _match_chief_complaint_to_category(self, chief_complaint: str) -> str:
        """
        Match chief complaint to category using semantic similarity matching to chief complaint triggers.
        
        Algorithm:
        - User says "I have a tummy ache"
        - Semantically compare to all chief_complaint_triggers from all guidelines
        - Match to highest similarity → return category
        - Keep it simple: direct semantic comparison, no synonym normalization
        """
        triggers_index_missing = not self.chief_complaint_triggers_index
        embedding_model_missing = not self.embedding_model
        triggers_data_empty = len(self.chief_complaint_triggers_data) == 0
        
        if triggers_index_missing or embedding_model_missing or triggers_data_empty:
            raise ValueError("Chief complaint triggers index not available. Cannot match category.")
        
        try:
            # Direct semantic similarity matching to chief complaint triggers (no synonym normalization)
            # Encode chief complaint directly
            query_embedding = self.embedding_model.encode([chief_complaint.lower().strip()])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index (get top matches - use more for fuzzy filtering)
            k = min(10, len(self.chief_complaint_triggers_data))  # Get more candidates for fuzzy filtering
            similarities, indices = self.chief_complaint_triggers_index.search(query_embedding, k)
            
            # Find best matching category (threshold 0.6) and track near-misses for fuzzy matching
            category_scores = {}
            near_miss_candidates = []  # Triggers that scored 0.5-0.6 (close but below threshold)
            
            for idx, sim in zip(indices[0], similarities[0]):
                if idx < len(self.chief_complaint_triggers_data):
                    trigger_data = self.chief_complaint_triggers_data[idx]
                    
                    if sim >= self.CHIEF_COMPLAINT_NEAR_MISS_UPPER:
                        # Above threshold - use for category matching
                        category = trigger_data['category']
                        if category not in category_scores or sim > category_scores[category]:
                            category_scores[category] = sim
                    elif sim >= self.CHIEF_COMPLAINT_NEAR_MISS_LOWER:
                        # Close to threshold - candidate for fuzzy matching (typo detection)
                        near_miss_candidates.append((trigger_data, sim))
            
            # If FAISS didn't find matches above threshold, try fuzzy matching only on near-misses
            if not category_scores and near_miss_candidates:
                self._capture_debug(f"[Engine] ⚠️ FAISS found no matches above {self.CHIEF_COMPLAINT_NEAR_MISS_UPPER}, trying fuzzy matching on {len(near_miss_candidates)} near-miss candidates...")
                try:
                    from difflib import SequenceMatcher
                    
                    chief_complaint_lower = chief_complaint.lower()
                    
                    # OPTIMIZATION: Extract key medical terms from chief complaint for faster matching
                    # Remove common phrases like "I have", "I'm experiencing", etc.
                    key_terms = chief_complaint_lower
                    for phrase in ["i have", "i'm having", "i've got", "i feel", "i'm experiencing"]:
                        if key_terms.startswith(phrase):
                            key_terms = key_terms[len(phrase):].strip()
                            break
                    
                    key_terms_len = len(key_terms)
                    best_fuzzy_match = None
                    best_fuzzy_score = 0.0
                    best_fuzzy_category = None
                    
                    # Only fuzzy match against near-miss candidates (already filtered by FAISS)
                    for trigger_data, faiss_score in near_miss_candidates:
                        trigger_text = trigger_data.get('trigger', '').lower()
                        if not trigger_text:
                            continue
                        
                        # Fast length check first (skip if lengths too different)
                        trigger_len = len(trigger_text)
                        length_ratio = min(key_terms_len, trigger_len) / max(key_terms_len, trigger_len)
                        if length_ratio < 0.5:  # More than 50% length difference = skip
                            continue
                        
                        # Fast substring check (if one is substring of other, high similarity)
                        key_terms_in_trigger = key_terms in trigger_text
                        trigger_in_key_terms = trigger_text in key_terms
                        
                        if key_terms_in_trigger or trigger_in_key_terms:
                            similarity = 0.9  # High score for substring match
                        else:
                            # Only do expensive SequenceMatcher if length check passed
                            similarity = SequenceMatcher(None, key_terms, trigger_text).ratio()
                        
                        similarity_greater_than_best = similarity > best_fuzzy_score
                        similarity_meets_threshold = similarity >= self.CHIEF_COMPLAINT_FUZZY_THRESHOLD  # Fuzzy threshold (stricter than FAISS for typos)
                        
                        if similarity_greater_than_best and similarity_meets_threshold:
                            best_fuzzy_score = similarity
                            best_fuzzy_match = trigger_text
                            best_fuzzy_category = trigger_data['category']
                    
                    if best_fuzzy_match and best_fuzzy_category:
                        self._capture_debug(f"[Engine] ✅ Fuzzy match found: '{chief_complaint}' → '{best_fuzzy_match}' ({best_fuzzy_category}, FAISS: {faiss_score:.3f} → fuzzy: {best_fuzzy_score:.3f})")
                        return best_fuzzy_category
                    else:
                        self._capture_debug(f"[Engine] ❌ Fuzzy matching on near-misses found no matches (threshold: 0.8)")
                except Exception as fuzzy_e:
                    self._capture_debug(f"[Engine] ⚠️ Fuzzy matching error: {fuzzy_e}")
            
            # If no matches found (either above threshold or via fuzzy on near-misses)
            if not category_scores:
                raise ValueError(f"No category match found for chief complaint: '{chief_complaint}' (FAISS threshold: {self.CHIEF_COMPLAINT_NEAR_MISS_UPPER}, fuzzy on near-misses {self.CHIEF_COMPLAINT_NEAR_MISS_LOWER}-{self.CHIEF_COMPLAINT_NEAR_MISS_UPPER})")
            
            # Sort categories by score
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            best_category, best_score = sorted_categories[0]
            
            # Check for cross-organ system matches (if multiple categories have similar scores)
            # If second-best category is within 0.1 of best, it might be crossover
            if len(sorted_categories) > 1:
                second_score = sorted_categories[1][1]
                score_difference_small = best_score - second_score < 0.1
                second_score_meets_threshold = second_score >= 0.6
                
                if score_difference_small and second_score_meets_threshold:
                    second_category = sorted_categories[1][0]
                    self._capture_debug(f"[Engine] 🎯 Multiple categories detected (crossover): {best_category} ({best_score:.3f}) vs {second_category} ({second_score:.3f})")
                    # For now, return best - could be extended to return both categories
            
            self._capture_debug(f"[Engine] 🎯 Category matched via chief_complaint_triggers: {best_category} (score: {best_score:.3f})")
            return best_category
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ FAISS chief complaint matching failed: {e}")
            raise ValueError(f"Failed to match chief complaint to category: {e}")
    
    def _get_all_guidelines_in_category(self, category: str) -> List[Dict]:
        """Get all guidelines in category"""
        relevant_guidelines = self._get_guidelines_by_category(category)
        
        matched_guidelines = []
        for name, guideline in relevant_guidelines.items():
            matched_guidelines.append({
                'name': name,
                'score': 0.5,
                'data': guideline
            })
        
        # Sort by prevalence
        matched_guidelines.sort(key=lambda x: x['data'].get('prevalence_score', 0), reverse=True)
        return matched_guidelines
    
    def _get_guidelines_by_category(self, category: str) -> Dict:
        """Get guidelines filtered by category using directory structure"""
        if category == 'ALL':
            return self.all_guidelines
        
        target_organ = self.CATEGORY_TO_SYSTEM.get(category.lower(), category.upper())
        filtered = {}
        
        for name, guideline in self.all_guidelines.items():
            organ_system = guideline.get('organ_system', '')
            # Match by directory name (e.g., 'GI', 'CARDIO', etc.)
            if organ_system == target_organ or target_organ in organ_system.upper():
                filtered[name] = guideline
        
        if not filtered:
            self._capture_debug(f"[Engine] ⚠️ No guidelines found for {category}, using all guidelines")
            return self.all_guidelines
        
        self._capture_debug(f"[Engine] ✅ Filtered {len(self.all_guidelines)} → {len(filtered)} guidelines for {category}")
        return filtered
    
    
    # ============================================================================
    # SECTION 4: DEMOGRAPHICS - Age, Sex, Chronicity Extraction
    # ============================================================================
    
    def _generate_ml_first_question_with_demographics(self) -> Dict[str, Any]:
        """Generate demographics questions in order: chronicity, age, sex"""
        # Note: Demographics are always collected unless it's a severe emergency (911/ER case)
        # Distress alone does NOT skip demographics - only severe emergencies skip
        
        # STEP 1: Chronicity question (first after empathetic statement)
        if 'chronicity' not in self.demographics:
            if not self.llm_chat_simple_fn:
                raise ValueError("LLM not available for question generation")
            
            system_msg = self.LLM_CHRONICITY_SYSTEM_MSG
            user_msg = self.LLM_CHRONICITY_USER_MSG
            
            llm_kwargs = self._get_llm_kwargs()
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                **llm_kwargs
            )
            response_is_empty = not response
            response_stripped_is_empty = not response.strip() if response else True
            
            if response_is_empty or response_stripped_is_empty:
                raise ValueError("LLM returned empty response for chronicity question")
            
            question = response.strip()
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'chronicity'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'New Problem', 'callback_data': 'chronicity_new'},
                    {'text': 'Ongoing Issue', 'callback_data': 'chronicity_recurring'}
                ],
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Demographics question generated"),
                    'internal': self._get_debug_info()
                }
            }
        
        # STEP 2: Age question
        if 'age' not in self.demographics:
            # Use hardcoded question for consistency
            question = "Can you please tell me your age so I can update our medical records?"
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'age'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Demographics question generated"),
                    'internal': self._get_debug_info()
                }
            }
        
        # STEP 3: Sex question
        if 'sex' not in self.demographics:
            # Use simple, direct question instead of LLM for consistency
            question = "What is your biological sex?"
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'oldcarts': 'demographics',
                'focus': 'sex'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'Male', 'callback_data': 'sex_male'},
                    {'text': 'Female', 'callback_data': 'sex_female'}
                ],
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Demographics question generated"),
                    'internal': self._get_debug_info()
                }
            }
        
        # All demographics collected, start OLDCARTS
        self._capture_debug("[Engine] 📋 Demographics complete, transitioning to clinical questions")
        
        # Initialize oldcarts_analysis if not already set
        if not self.oldcarts_analysis:
            # Standard order: timing before duration
            standard_order = ['onset', 'location', 'timing', 'duration', 'progression', 'character', 'aggravating', 'relieving', 'severity', 'associated']
            self.oldcarts_analysis = {
                'answered_components': {},
                'missing_components': standard_order.copy(),
                'anatomical_analysis': {}
            }
            self._capture_debug(f"[Engine] ✅ Initialized OLDCARTS analysis with missing_components: {self.oldcarts_analysis['missing_components']}")
        
        self._capture_debug(f"[Engine] Current OLDCARTS analysis: {self.oldcarts_analysis}")
        return self._ask_next_clinical_question()
    
    def _generate_empathetic_statement(self) -> str:
        """Generate empathetic opening statement using LLM"""
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for empathetic statement generation")
        
        system_msg = "You are a compassionate medical assistant. Generate a brief, empathetic statement acknowledging the patient's concern."
        user_msg = f"Patient reported: '{self.chief_complaint}'\n\nGenerate a brief, empathetic acknowledgment (1-2 sentences). Acknowledge their concern, show compassion, and express that you're here to help. Do NOT ask questions. End with a period. Return only the statement, no other text."
        
        # Use all LLM settings from environment
        llm_kwargs = self._get_llm_kwargs()
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        response_is_empty = not response
        response_stripped_is_empty = not response.strip() if response else True
        
        if response_is_empty or response_stripped_is_empty:
            raise ValueError("LLM returned empty response for empathetic statement")
        
        return response.strip()
    
    def _extract_key_features_from_guidelines(self) -> List[Dict]:
        """Extract key positives and negatives from top 2 active guidelines"""
        key_features_list = []
        
        # Get top 2 active guidelines only
        top_guidelines = self.active_guidelines[:2]
        
        for g in top_guidelines:
            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
            key_features = g.get('data', {}).get('key_features', {})
            
            # Extract from structured key_positives if available
            key_positives = key_features.get('key_positives', [])
            for pos in key_positives[:3]:  # Limit to top 3 positives per condition
                if isinstance(pos, str):
                    key_features_list.append({
                        'type': 'positive',
                        'condition': condition_name,
                        'feature': pos,
                        'guideline': g
                    })
                elif isinstance(pos, dict):
                    key_features_list.append({
                        'type': 'positive',
                        'condition': condition_name,
                        'feature': pos.get('feature', pos.get('medical', '')),
                        'guideline': g
                    })
            
            # Extract from structured key_negatives if available
            key_negatives = key_features.get('key_negatives', [])
            for neg in key_negatives[:2]:  # Limit to top 2 negatives per condition
                if isinstance(neg, str):
                    key_features_list.append({
                        'type': 'negative',
                        'condition': condition_name,
                        'feature': neg,
                        'guideline': g
                    })
                elif isinstance(neg, dict):
                    key_features_list.append({
                        'type': 'negative',
                        'condition': condition_name,
                        'feature': neg.get('feature', neg.get('medical', '')),
                        'guideline': g
                    })
        
        return key_features_list
    
    def _start_key_features_phase(self) -> Dict[str, Any]:
        """Start asking about key positives and negatives"""
        self.key_features_phase = True
        self.key_features_list = self._extract_key_features_from_guidelines()
        self.key_features_index = 0
        
        if not self.key_features_list:
            # No key features found - move to red flag screening
            return self._start_red_flag_phase()
        
        return self._ask_next_key_feature()
    
    def _ask_next_key_feature(self) -> Dict[str, Any]:
        """Ask next key feature question"""
        if self.key_features_index >= len(self.key_features_list):
            # All key features asked - move to red flag screening
            if not self.red_flag_phase:
                return self._start_red_flag_phase()
            else:
                return self._ask_next_red_flag()
        
        feature_data = self.key_features_list[self.key_features_index]
        feature_type = feature_data['type']
        feature_text = feature_data['feature']
        condition_name = feature_data['condition']
        
        # Generate question using LLM
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for key feature question generation")
        
        system_msg = "You are a medical assistant. Generate a simple, patient-friendly question about a key clinical feature."
        if feature_type == 'positive':
            user_msg = f"Key positive finding: '{feature_text}'\n\nFor condition: {condition_name}\n\nGenerate a simple yes/no or open-ended question to ask the patient about this finding. Keep it short and natural."
        else:
            user_msg = f"Key negative finding: '{feature_text}'\n\nFor condition: {condition_name}\n\nGenerate a simple yes/no or open-ended question to ask the patient about this finding. If present, it would make this condition less likely. Keep it short and natural."
        
        llm_kwargs = self._get_llm_kwargs()
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        
        if not response or not response.strip():
            raise ValueError("LLM returned empty response for key feature question")
        question = response.strip()
        
        # Remove prefixes if present
        prefixes = ["Here is the question:", "Q:", "Question:"]
        for prefix in prefixes:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'focus': 'key_feature',
            'feature_data': feature_data
        })
        
        self._capture_debug(f"[Engine] 🔍 Asking key feature ({feature_type}): {feature_text}")
        self._capture_debug(f"[Engine] Question: {question}")
        
        return {
            'success': True,
            'question': question,
            'status': 'key_feature',
            'debug': {
                'engine': self._format_engine_debug(f"[Engine] 🔍 Key feature question ({self.key_features_index + 1}/{len(self.key_features_list)})"),
                'internal': self._get_debug_info()
            }
        }
    
    def _process_key_feature_answer(self, user_answer: str, feature_data: Dict) -> None:
        """Process answer to key feature question and update scores"""
        feature_type = feature_data['type']
        feature_text = feature_data['feature']
        condition_name = feature_data['condition']
        guideline = feature_data['guideline']
        
        # Normalize answer
        answer_lower = user_answer.lower().strip()
        
        # Check for positive indicators (yes, present, etc.)
        positive_indicators = ['yes', 'yep', 'yeah', 'y', 'true', 'present', 'have', 'has', 'do', 'does', 'am', 'is', 'are']
        is_positive = any(indicator in answer_lower for indicator in positive_indicators)
        
        # Update score based on key feature answer
        old_score = guideline['score']
        if feature_type == 'positive' and is_positive:
            # Positive finding present - boost score
            boost = 0.1
            new_score = min(1.0, old_score + boost)
            self._capture_debug(f"[Key Feature] ✅ Positive finding '{feature_text}' present - boosting {condition_name}: {old_score:.3f} → {new_score:.3f}")
        elif feature_type == 'negative' and is_positive:
            # Negative finding present - penalize score
            penalty = -0.15
            new_score = max(0.0, old_score + penalty)
            self._capture_debug(f"[Key Feature] ❌ Negative finding '{feature_text}' present - penalizing {condition_name}: {old_score:.3f} → {new_score:.3f}")
        else:
            # No change (positive absent or negative absent)
            new_score = old_score
            self._capture_debug(f"[Key Feature] ⚪ No change for {condition_name}: {old_score:.3f}")
        
        guideline['score'] = new_score
        
        # Re-rank after score update
        all_guidelines = self.active_guidelines + self.reserve_pool
        previous_active = len(self.active_guidelines)
        self._rerank_and_pool_guidelines(all_guidelines, previous_active)
    
    def _get_all_missing_elements(self) -> Dict[str, Any]:
        """
        Get comprehensive list of ALL missing elements (OLDCARTS + demographics).
        This tracks what still needs to be collected at any point in the conversation.
        
        Returns:
            Dict with:
            - 'missing_oldcarts': List of missing OLDCARTS elements
            - 'missing_demographics': List of missing demographics
            - 'next_element': The next element to ask about (priority order)
            - 'has_missing': bool
        """
        missing_oldcarts = []
        missing_demographics = []
        
        # Check OLDCARTS coverage
        if hasattr(self, 'oldcarts_analysis') and self.oldcarts_analysis:
            missing_oldcarts = self.oldcarts_analysis.get('missing_components', [])
        
        # Check demographics
        if 'chronicity' not in self.demographics:
            missing_demographics.append('chronicity')
        # Always collect demographics unless it's a severe emergency
        if True:  # Always collect demographics
            # Age and sex only required if not distressed
            if 'age' not in self.demographics:
                missing_demographics.append('age')
            if 'sex' not in self.demographics:
                missing_demographics.append('sex')
        
        # Determine next element based on priority
        # Priority: Demographics first (if not distressed), then OLDCARTS
        next_element = None
        next_type = None
        
        if missing_demographics:
            # Demographics priority: chronicity > age > sex
            if 'chronicity' in missing_demographics:
                next_element = 'chronicity'
                next_type = 'demographics'
            elif 'age' in missing_demographics:
                next_element = 'age'
                next_type = 'demographics'
            elif 'sex' in missing_demographics:
                next_element = 'sex'
                next_type = 'demographics'
        elif missing_oldcarts:
            # OLDCARTS priority order
            priority_order = ['onset', 'location', 'timing', 'duration', 'progression', 'character', 
                            'aggravating', 'relieving', 'severity', 'associated']
            for element in priority_order:
                if element in missing_oldcarts:
                    next_element = element
                    next_type = 'oldcarts'
                    break
            
            # If no priority match, use first missing
            if not next_element and missing_oldcarts:
                next_element = missing_oldcarts[0]
                next_type = 'oldcarts'
        
        has_missing = len(missing_oldcarts) > 0 or len(missing_demographics) > 0
        
        return {
            'missing_oldcarts': missing_oldcarts,
            'missing_demographics': missing_demographics,
            'next_element': next_element,
            'next_type': next_type,
            'has_missing': has_missing
        }
    
    def _return_to_next_missing_element(self, acknowledgment_msg: str = None, last_user_input: str = None) -> Dict[str, Any]:
        """
        After handling a comment/question/distress, intelligently return to the next missing element.
        This ensures we don't lose track of what needs to be collected.
        
        IMPORTANT: Checks for emergency/distress at EVERY transition point, not just at the start.
        
        Args:
            acknowledgment_msg: Optional acknowledgment message to prepend
            last_user_input: The user's last input (to check for emergency/distress)
            
        Returns:
            Response dict with next question or None if assessment complete
        """
        # CRITICAL: Check for emergency/distress at EVERY transition point
        # This catches cases where emergency wasn't detected initially but should be checked again
        if last_user_input:
            # Re-check the user's input for emergency/distress
            red_flag_info = self._detect_red_flags_in_input(last_user_input)
            distress_check = self._detect_distress(last_user_input)
            severity_score = distress_check.get('severity', 0.0)
            is_distressed = distress_check.get('is_distressed', False)
            
            red_flag_count = red_flag_info.get('red_flag_count', 0)
            has_severity_language = red_flag_info.get('has_severity_language', False)
            
            # Check if this is a severe emergency (same restrictive criteria)
            life_threatening_keywords = [
                'chest pain', 'heart attack', 'can\'t breathe', 'shortness of breath', 'difficulty breathing',
                'unconscious', 'passed out', 'fainting', 'seizure', 'convulsion',
                'severe bleeding', 'vomiting blood', 'blood in stool', 'black stool'
            ]
            last_input_lower = last_user_input.lower()
            has_life_threatening = any(keyword in last_input_lower for keyword in life_threatening_keywords)
            
            is_severe_emergency = (
                severity_score >= 9.0 or
                (has_life_threatening and severity_score >= 7.0) or
                red_flag_count >= 3 or
                (has_life_threatening and red_flag_count >= 1)
            )
            
            if is_severe_emergency:
                self._capture_debug(f"[Engine] 🚨 EMERGENCY detected in _return_to_next_missing_element: severity={severity_score:.1f}, red_flags={red_flag_count}")
                return self._generate_emergency_response(last_user_input, red_flag_info, distress_check)
        
        # Note: Demographics are always collected unless it's a severe emergency (911/ER case)
        # Distress alone does NOT skip demographics - only severe emergencies skip
        
        missing_info = self._get_all_missing_elements()
        
        if not missing_info['has_missing']:
            # No missing elements - check if assessment is complete
            # This would trigger key features or red flags phase
            next_response = self._ask_next_clinical_question()
            if acknowledgment_msg and next_response and next_response.get('success'):
                next_msg = next_response.get('message') or next_response.get('question', '')
                combined_msg = f"{acknowledgment_msg}\n\n{next_msg}"
                return {
                    'success': True,
                    'message': combined_msg,
                    'status': next_response.get('status', 'questioning'),
                    'debug': next_response.get('debug', {})
                }
            return next_response
        
        # Determine what to ask next
        next_element = missing_info['next_element']
        next_type = missing_info['next_type']
        
        if next_type == 'demographics':
            # Return to demographics collection
            if next_element == 'chronicity':
                return self._generate_ml_first_question_with_demographics()
            elif next_element == 'age':
                question = "Can you please tell me your age so I can update our medical records?"
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'focus': 'age'
                })
                response = {
                    'success': True,
                    'message': question,
                    'status': 'questioning',
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] 📋 Returning to collect missing age"),
                        'internal': self._get_debug_info()
                    }
                }
                if acknowledgment_msg:
                    response['message'] = f"{acknowledgment_msg}\n\n{question}"
                return response
            elif next_element == 'sex':
                question = "What is your biological sex?"
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'focus': 'sex'
                })
                response = {
                    'success': True,
                    'message': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'Male', 'callback_data': 'sex_male'},
                        {'text': 'Female', 'callback_data': 'sex_female'}
                    ],
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] 📋 Returning to collect missing sex"),
                        'internal': self._get_debug_info()
                    }
                }
                if acknowledgment_msg:
                    response['message'] = f"{acknowledgment_msg}\n\n{question}"
                return response
        
        elif next_type == 'oldcarts':
            # Return to OLDCARTS collection
            next_response = self._ask_next_clinical_question()
            if acknowledgment_msg and next_response and next_response.get('success'):
                next_msg = next_response.get('message') or next_response.get('question', '')
                combined_msg = f"{acknowledgment_msg}\n\n{next_msg}"
                return {
                    'success': True,
                    'message': combined_msg,
                    'status': next_response.get('status', 'questioning'),
                    'debug': next_response.get('debug', {})
                }
            return next_response
        
        # No fallback - raise error if no valid response
        raise ValueError("No valid response generated for missing demographics check")
    
    def _check_and_collect_missing_demographics(self) -> Dict[str, Any]:
        """
        Check for missing demographics and collect them if needed.
        Called after urgent assessment is complete but before final diagnosis.
        """
        missing_demographics = []
        if 'age' not in self.demographics:
            missing_demographics.append('age')
        if 'sex' not in self.demographics:
            missing_demographics.append('sex')
        
        if not missing_demographics:
            return None  # No missing demographics
        
        # Ask for the first missing demographic
        missing = missing_demographics[0]
        
        if missing == 'age':
            question = "To complete your medical record, can you please tell me your age?"
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'age',
                'post_assessment': True  # Flag that this is being asked after assessment
            })
            return {
                'success': True,
                'message': question,
                'status': 'collecting_demographics',
                'missing_demographics': missing_demographics,
                'debug': {
                    'engine': self._format_engine_debug("[Engine] 📋 Collecting missing demographics after assessment"),
                    'internal': self._get_debug_info()
                }
            }
        elif missing == 'sex':
            question = "To complete your medical record, what is your biological sex?"
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'sex',
                'post_assessment': True  # Flag that this is being asked after assessment
            })
            return {
                'success': True,
                'message': question,
                'status': 'collecting_demographics',
                'buttons': [
                    {'text': 'Male', 'callback_data': 'sex_male'},
                    {'text': 'Female', 'callback_data': 'sex_female'}
                ],
                'missing_demographics': missing_demographics,
                'debug': {
                    'engine': self._format_engine_debug("[Engine] 📋 Collecting missing demographics after assessment"),
                    'internal': self._get_debug_info()
                }
            }
        
        return None
    
    def _generate_completion_message(self) -> str:
        """Generate completion message with diagnosis, urgency, and recommendation"""
        if not self.active_guidelines:
            diagnosis = "Insufficient information for specific diagnosis"
            urgency_level = "ROUTINE"
            urgency_score = 3.0
        else:
            # Get top condition
            top_condition = self.active_guidelines[0]
            condition_name = top_condition.get('data', {}).get('condition', top_condition.get('name', 'Unknown condition'))
            condition_urgency = top_condition.get('urgency') or top_condition.get('data', {}).get('urgency', 'routine')
            
            # Get top 2-3 conditions for differential
            top_conditions = []
            for i, g in enumerate(self.active_guidelines[:3]):
                cond_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
                score = g.get('score', 0.0)
                top_conditions.append(f"{cond_name} (confidence: {score:.1%})")
            
            if len(top_conditions) == 1:
                diagnosis = f"MOST LIKELY DIAGNOSIS: {top_conditions[0]}"
            elif len(top_conditions) == 2:
                diagnosis = f"MOST LIKELY DIAGNOSIS: {top_conditions[0]}\n\nAlternative consideration: {top_conditions[1]}"
            else:
                diagnosis = f"MOST LIKELY DIAGNOSIS: {top_conditions[0]}\n\nAlternative considerations: {', '.join(top_conditions[1:])}"
            
            # Map urgency to score and level
            urgency_map = {
                'emergent': (9.0, 'EMERGENT'),
                'urgent': (7.0, 'URGENT'),
                'semi-urgent': (5.0, 'SEMI-URGENT'),
                'routine': (3.0, 'ROUTINE')
            }
            urgency_score, urgency_level = urgency_map.get(condition_urgency.lower(), (3.0, 'ROUTINE'))
        
        # Generate recommendation based on urgency
        if urgency_score >= 8.0:
            recommendation = "Call 911 or go to emergency room immediately"
        elif urgency_score >= 6.0:
            recommendation = "Go to emergency room within 1-2 hours"
        elif urgency_score >= 4.0:
            recommendation = "See doctor today or within 24 hours"
        else:
            recommendation = "Schedule appointment with doctor within 1 week"
        
        # Format final message
        completion_msg = f"""{diagnosis}

URGENCY LEVEL: {urgency_level} ({urgency_score:.1f}/10)

RECOMMENDATION: {recommendation}"""
        
        return completion_msg
    
    def _start_red_flag_phase(self) -> Dict[str, Any]:
        """Start asking about red flags for top winning condition"""
        if not self.active_guidelines:
            # No active guidelines - assessment complete
            # Check if we need to collect missing demographics first
            missing_demo_response = self._check_and_collect_missing_demographics()
            if missing_demo_response:
                # Still need demographics - return diagnosis but ask for missing info
                completion_msg = self._generate_completion_message()
                combined_msg = f"""{completion_msg}

---

{missing_demo_response.get('message', '')}"""
                return {
                    'success': True,
                    'status': 'completed_with_demographics',
                    'message': combined_msg,
                    'missing_demographics': missing_demo_response.get('missing_demographics', []),
                    'buttons': missing_demo_response.get('buttons'),
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - collecting missing demographics"),
                        'internal': self._get_debug_info()
                    }
                }
            
            # Assessment complete, all demographics collected
            completion_msg = self._generate_completion_message()
            return {
                'success': True,
                'status': 'completed',
                'message': completion_msg,
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Assessment complete"),
                    'internal': self._get_debug_info()
                }
            }
        
        # Get top winning condition
        top_condition = self.active_guidelines[0]
        condition_name = top_condition.get('data', {}).get('condition', top_condition.get('name', ''))
        red_flags = top_condition.get('data', {}).get('red_flags', [])
        
        if not red_flags:
            # No red flags for this condition - assessment complete
            # Check if we need to collect missing demographics first
            missing_demo_response = self._check_and_collect_missing_demographics()
            if missing_demo_response:
                # Still need demographics - return diagnosis but ask for missing info
                completion_msg = self._generate_completion_message()
                combined_msg = f"""{completion_msg}

---

{missing_demo_response.get('message', '')}"""
                return {
                    'success': True,
                    'status': 'completed_with_demographics',
                    'message': combined_msg,
                    'missing_demographics': missing_demo_response.get('missing_demographics', []),
                    'buttons': missing_demo_response.get('buttons'),
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - collecting missing demographics"),
                        'internal': self._get_debug_info()
                    }
                }
            
            # Assessment complete, all demographics collected
            completion_msg = self._generate_completion_message()
            return {
                'success': True,
                'status': 'completed',
                'message': completion_msg,
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Assessment complete"),
                    'internal': self._get_debug_info()
                }
            }
        
        self.red_flag_phase = True
        self.red_flags_list = [
            {'condition': condition_name, 'red_flag': flag, 'guideline': top_condition}
            for flag in red_flags
        ]
        self.red_flag_index = 0
        
        return self._ask_next_red_flag()
    
    def _ask_next_red_flag(self) -> Dict[str, Any]:
        """Ask next red flag question"""
        # Skip red flags that were already mentioned in conversation
        while self.red_flag_index < len(self.red_flags_list):
            red_flag_data = self.red_flags_list[self.red_flag_index]
            red_flag_text = red_flag_data['red_flag']
            
            # Check if this red flag was already mentioned in conversation history
            already_mentioned = False
            if hasattr(self, 'conversation_history') and self.conversation_history:
                # Check all previous answers for mentions of this red flag
                red_flag_lower = red_flag_text.lower()
                for item in self.conversation_history:
                    item_type_is_answer = item.get('type') == 'answer'
                    if item_type_is_answer:
                        answer_text = item.get('answer', item.get('message', '')).lower()
                        # Direct substring match
                        red_flag_in_answer = red_flag_lower in answer_text
                        # Check if key words from red flag are in answer
                        red_flag_words = set(red_flag_lower.split())
                        answer_words = set(answer_text.split())
                        # Remove common words
                        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'i', 'am', 'you', 'your', 'my', 'me', 'we', 'they', 'this', 'that', 'these', 'those'}
                        red_flag_key_words = red_flag_words - common_words
                        answer_key_words = answer_words - common_words
                        key_words_match = len(red_flag_key_words & answer_key_words) > 0
                        
                        # FAISS semantic check if available
                        semantic_match = False
                        if self.medical_rule_engine and red_flag_key_words:
                            try:
                                # Use FAISS to check if red flag is semantically similar to answer
                                semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(
                                    answer_text, 'associated', threshold=self.FAISS_ASSOCIATED_THRESHOLD
                                )
                                red_flag_in_matches = any(red_flag_lower in match.lower() or match.lower() in red_flag_lower 
                                                         for match in semantic_matches)
                                semantic_match = red_flag_in_matches
                            except:
                                pass
                        
                        if red_flag_in_answer or key_words_match or semantic_match:
                            already_mentioned = True
                            self._capture_debug(f"[Red Flag] ⏭️ Skipping '{red_flag_text}' - already mentioned in conversation")
                            break
            
            if not already_mentioned:
                # This red flag hasn't been mentioned - ask about it
                break
            
            # Skip this red flag and move to next
            self.red_flag_index += 1
        
        if self.red_flag_index >= len(self.red_flags_list):
            # All red flags asked - assessment complete
            # Check if we need to collect missing demographics first
            missing_demo_response = self._check_and_collect_missing_demographics()
            if missing_demo_response:
                # Still need demographics - return diagnosis but ask for missing info
                completion_msg = self._generate_completion_message()
                combined_msg = f"""{completion_msg}

---

{missing_demo_response.get('message', '')}"""
                return {
                    'success': True,
                    'status': 'completed_with_demographics',
                    'message': combined_msg,
                    'missing_demographics': missing_demo_response.get('missing_demographics', []),
                    'buttons': missing_demo_response.get('buttons'),
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - collecting missing demographics"),
                        'internal': self._get_debug_info()
                    }
                }
            
            # Assessment complete, all demographics collected
            completion_msg = self._generate_completion_message()
            return {
                'success': True,
                'status': 'completed',
                'message': completion_msg,
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ✅ Assessment complete"),
                    'internal': self._get_debug_info()
                }
            }
        
        red_flag_data = self.red_flags_list[self.red_flag_index]
        red_flag_text = red_flag_data['red_flag']
        condition_name = red_flag_data['condition']
        
        # Generate question using LLM
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for red flag question generation")
        
        # Build conversation context to avoid asking about things already mentioned
        conversation_context = ""
        if hasattr(self, 'conversation_history') and self.conversation_history:
            recent_items = [item for item in self.conversation_history[-5:] if item.get('type') == 'answer']
            if recent_items:
                mentioned_symptoms = [item.get('answer', item.get('message', ''))[:50] for item in recent_items]
                conversation_context = f"\n\nIMPORTANT: The patient has already mentioned: {', '.join(mentioned_symptoms)}. Do NOT ask about symptoms already mentioned. Only ask about the specific red flag: '{red_flag_text}'."
        
        system_msg = "You are a medical assistant screening for urgent medical conditions. Generate ONLY a simple, patient-friendly question about a red flag symptom. Do NOT include any explanations, reasoning, or prefixes. Return ONLY the question text. CRITICAL: Use second person ('Are you experiencing...') NOT third person ('Is the patient experiencing...')."
        user_msg = f"Red flag: '{red_flag_text}'\n\nFor condition: {condition_name}{conversation_context}\n\nGenerate a simple yes/no or open-ended question to screen for this urgent symptom. Use second person ('Are you experiencing...', 'Do you have...', 'Have you noticed...'). Do NOT use third person ('Is the patient experiencing...', 'Does the patient have...'). Keep it short, clear, and direct. Return ONLY the question - no explanations, no prefixes, no reasoning. Just the question itself."
        
        llm_kwargs = self._get_llm_kwargs()
        # Override max_tokens for red flag questions to prevent cutoff
        # Increase from default 120 to 200 to allow for complete questions
        if 'max_tokens' in llm_kwargs:
            llm_kwargs['max_tokens'] = 200
        else:
            llm_kwargs['max_tokens'] = 200
        
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        
        question = response.strip() if response else f"About {red_flag_text.lower()}:"
        
        # Filter out internal reasoning and extract just the question
        # Remove common prefixes and reasoning patterns
        prefixes_to_remove = [
            "Here is the question:",
            "Here's a simple question to screen for",
            "Here's a simple question to screen for the",
            "Here's a simple question to screen for the urgent symptom of",
            "Here's a simple question to screen for the red flag symptom of",
            "Q:",
            "Question:",
            "The question is:",
            "This question is designed to",
        ]
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
                # Remove any leading punctuation or whitespace
                question = question.lstrip('.,:;').strip()
        
        # Extract just the question if there's reasoning before it
        # Look for patterns like "This question is designed to..." followed by the actual question
        # The question usually starts with a capital letter or quote
        if 'designed to' in question.lower() or 'screen for' in question.lower():
            # Look for quoted text (most common format for questions)
            quoted_match = re.search(r'["\']([^"\']+[?])["\']', question)
            if quoted_match:
                question = quoted_match.group(1).strip()
            else:
                # Look for text after colon that ends with ?
                colon_match = re.search(r':\s*([A-Z][^.!?]+[?])', question)
                if colon_match:
                    question = colon_match.group(1).strip()
                else:
                    # Find the last sentence that ends with ? (usually the actual question)
                    sentences = re.split(r'([.!?]\s+)', question)
                    question_parts = []
                    for i in range(len(sentences) - 1, -1, -1):
                        if '?' in sentences[i]:
                            question_parts.insert(0, sentences[i])
                            # Also include preceding sentence if it's part of the question
                            if i > 0 and sentences[i-1].strip():
                                question_parts.insert(0, sentences[i-1])
                            break
                    if question_parts:
                        question = ''.join(question_parts).strip()
                    else:
                        # Last resort: extract text after "question:" or similar
                        question_match = re.search(r'(?:question|question to|symptom):\s*([^.!]+[?])', question, re.IGNORECASE)
                        if question_match:
                            question = question_match.group(1).strip()
        
        # Replace third-person with second-person (safety check)
        # Common patterns: "Is the patient experiencing" -> "Are you experiencing"
        third_person_patterns = [
            (r'Is the patient (experiencing|having|noticing)', r'Are you \1'),
            (r'Does the patient (have|experience|notice)', r'Do you \1'),
            (r'Has the patient (experienced|had|noticed)', r'Have you \1'),
            (r'the patient (is|has|does|experiences|has|notices)', r'you \1'),
            (r'patient\'s', r'your'),
        ]
        for pattern, replacement in third_person_patterns:
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        # Final cleanup: ensure we have a proper question
        if not question or len(question) < 10:
            question = f"Are you experiencing {red_flag_text.lower()}?"
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'focus': 'red_flag',
            'red_flag_data': red_flag_data
        })
        
        self._capture_debug(f"[Engine] 🚨 Asking red flag: {red_flag_text}")
        self._capture_debug(f"[Engine] Question: {question}")
        
        return {
            'success': True,
            'question': question,
            'status': 'red_flag',
            'debug': {
                'engine': self._format_engine_debug(f"[Engine] 🚨 Red flag question ({self.red_flag_index + 1}/{len(self.red_flags_list)})"),
                'internal': self._get_debug_info()
            }
        }
    
    def _process_red_flag_answer(self, user_answer: str, red_flag_data: Dict) -> None:
        """Process answer to red flag question - if positive, mark as urgent case"""
        red_flag_text = red_flag_data['red_flag']
        condition_name = red_flag_data['condition']
        
        # Normalize answer
        answer_lower = user_answer.lower().strip()
        
        # Check for positive indicators (yes, present, etc.)
        positive_indicators = ['yes', 'yep', 'yeah', 'y', 'true', 'present', 'have', 'has', 'do', 'does', 'am', 'is', 'are']
        is_positive = any(indicator in answer_lower for indicator in positive_indicators)
        
        if is_positive:
            # Red flag present - URGENT CASE DETECTED
            self.red_flags_present.append({
                'condition': condition_name,
                'red_flag': red_flag_text,
                'severity': 'urgent'
            })
            self._capture_debug(f"[Red Flag] 🚨 URGENT: '{red_flag_text}' present for {condition_name}")
        else:
            self._capture_debug(f"[Red Flag] ✅ No red flag: '{red_flag_text}' absent for {condition_name}")
    
    def _interpret_patient_response(self, user_input: str, expected_element: str = None, last_q: Dict = None) -> Dict[str, Any]:
        """
        Intelligently interpret patient response to categorize it and extract relevant information.
        Detects comments, questions, distress, or direct answers.
        
        Args:
            user_input: The patient's response
            expected_element: Optional OLDCARTS element that was expected (for context)
            
        Returns:
            Dict with:
            - 'type': 'direct_answer', 'comment', 'question', 'distress', 'distress_question', 'mixed'
            - 'is_distressed': bool
            - 'distress_info': dict from _detect_distress
            - 'is_question': bool
            - 'is_comment': bool
            - 'needs_acknowledgment': bool
            - 'acknowledgment_message': str or None
            - 'extracted_info': str or None (clinical info extracted from response)
        """
        user_lower = user_input.lower().strip()
        
        # Detect distress
        distress_info = self._detect_distress(user_input)
        is_distressed = distress_info.get('is_distressed', False)
        
        # Detect if user is asking a question
        is_question = self._is_user_asking_question(user_input)
        
        # Detect if it's a comment/exclamation (short emotional expressions)
        is_comment = self._is_comment_or_exclamation(user_input)
        
        # Try to extract clinical information (if answer contains both comment and clinical info)
        # ALWAYS try to extract, even if there's a comment/distress/question
        extracted_info = None
        
        # For chronicity (new/recurring), use Jaccard similarity with reference phrases
        if expected_element == 'chronicity' or expected_element == 'demographics' or (last_q and last_q.get('focus') == 'chronicity'):
            extracted_info = self._extract_chronicity_with_jaccard(user_input)
        
        # If simple extraction didn't work, try LLM extraction
        if not extracted_info and self.llm_chat_simple_fn and expected_element:
            # Use LLM to extract just the clinical info if there's a comment mixed in
            try:
                system_msg = f"You are a medical assistant. Extract ONLY the clinical information relevant to: {expected_element}. Remove all emotional comments, questions, or other non-clinical text. Return ONLY the clinical information, or 'none' if no clinical info found."
                user_msg = f"Patient said: '{user_input}'\n\nExtract clinical information about {expected_element} only:"
                
                llm_kwargs = self._get_llm_kwargs(override_max_tokens=50)
                response = self.llm_chat_simple_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    **llm_kwargs
                )
                
                extracted = response.strip().lower() if response else ''
                if extracted and extracted != 'none' and len(extracted) > 2:
                    extracted_info = extracted
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to extract clinical info: {e}")
        
        # Determine response type
        response_type = 'direct_answer'
        if is_distressed and is_question:
            response_type = 'distress_question'
        elif is_distressed:
            response_type = 'distress'
        elif is_question:
            response_type = 'question'
        elif is_comment:
            response_type = 'comment'
        elif extracted_info and (is_comment or is_question):
            response_type = 'mixed'  # Contains both comment/question and clinical info
        
        # Determine if acknowledgment is needed
        # ALWAYS acknowledge distress, comments, or questions
        needs_acknowledgment = False
        acknowledgment_message = None
        
        if is_distressed or is_comment or is_question:
            needs_acknowledgment = True
            # Generate acknowledgment message - prioritize distress
            if is_distressed:
                # Distress ALWAYS needs empathetic acknowledgment
                acknowledgment_message = self._generate_empathetic_response(user_input, distress_info)
                self._capture_debug(f"[Engine] 💬 Generated empathetic acknowledgment for distress (severity={distress_info.get('severity', 0):.1f})")
            elif is_question:
                # Acknowledge question - generate with LLM if available, otherwise raise error
                if self.llm_chat_simple_fn:
                    try:
                        system_msg = self.LLM_QUESTION_ACK_SYSTEM_MSG
                        user_msg = self.LLM_QUESTION_ACK_USER_TEMPLATE.format(user_input=user_input)
                        
                        llm_kwargs = self._get_llm_kwargs(override_max_tokens=40)
                        response = self.llm_chat_simple_fn(
                            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                            **llm_kwargs
                        )
                        acknowledgment_message = response.strip() if response else None
                        if not acknowledgment_message:
                            raise ValueError("LLM returned empty acknowledgment for question")
                    except Exception as e:
                        self._capture_debug(f"[Engine] ⚠️ Failed to generate question acknowledgment: {e}")
                        raise ValueError(f"Cannot generate acknowledgment for question: {e}")
                else:
                    raise ValueError("LLM not available for question acknowledgment generation")
            elif is_comment:
                # Acknowledge comment
                if self.llm_chat_simple_fn:
                    try:
                        system_msg = self.LLM_COMMENT_ACK_SYSTEM_MSG
                        user_msg = self.LLM_COMMENT_ACK_USER_TEMPLATE.format(user_input=user_input)
                        
                        llm_kwargs = self._get_llm_kwargs(override_max_tokens=40)
                        response = self.llm_chat_simple_fn(
                            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                            **llm_kwargs
                        )
                        acknowledgment_message = response.strip() if response else "I understand. "
                    except Exception as e:
                        self._capture_debug(f"[Engine] ⚠️ Failed to generate acknowledgment: {e}")
                        acknowledgment_message = "I understand. "
                else:
                    acknowledgment_message = "I understand. "
        
        return {
            'type': response_type,
            'is_distressed': is_distressed,
            'distress_info': distress_info,
            'is_question': is_question,
            'is_comment': is_comment,
            'needs_acknowledgment': needs_acknowledgment,
            'acknowledgment_message': acknowledgment_message,
            'extracted_info': extracted_info
        }
    
    def _is_comment_or_exclamation(self, user_input: str) -> bool:
        """Detect if user input is a short comment or exclamation rather than a direct answer"""
        user_lower = user_input.lower().strip()
        
        # Check for location/clinical terms that should NOT be treated as comments
        location_terms = ['right', 'left', 'upper', 'lower', 'side', 'center', 'middle', 
                         'abdomen', 'chest', 'back', 'pain', 'hurt', 'ache']
        has_location_term = any(term in user_lower for term in location_terms)
        
        # If contains location/clinical terms, likely a direct answer, not a comment
        if has_location_term:
            return False
        
        # Very short inputs are often comments
        if len(user_input.split()) <= 3:
            # Check for common comment patterns (removed "right" since it's ambiguous)
            comment_patterns = [
                'ok', 'okay', 'sure', 'yeah', 'yep', 'got it',
                'oh', 'wow', 'really', 'hmm', 'um', 'ah', 'i see',
                'thanks', 'thank you', 'please', 'help', 'scared', 'worried'
            ]
            if user_lower in comment_patterns:
                return True
        
        # Emotional expressions
        emotional_words = ['scared', 'worried', 'afraid', 'nervous', 'anxious', 'frightened', 
                          'terrified', 'panicking', 'please', 'help', 'ok', 'okay']
        if any(word in user_lower for word in emotional_words) and len(user_input.split()) <= 5:
            return True
        
        # Exclamations
        if user_input.endswith('!') and len(user_input.split()) <= 5:
            return True
        
        return False
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts.
        Jaccard similarity = |intersection| / |union| of word sets.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize: lowercase and split into word sets
        words1 = set(re.sub(r'[^\w\s]', ' ', text1.lower()).split())
        words2 = set(re.sub(r'[^\w\s]', ' ', text2.lower()).split())
        
        # Remove empty strings
        words1 = {w for w in words1 if w}
        words2 = {w for w in words2 if w}
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_chronicity_with_jaccard(self, user_input: str) -> Optional[str]:
        """
        Extract chronicity (new/recurring) from user input using Jaccard similarity.
        
        Args:
            user_input: User's response
            
        Returns:
            'new', 'recurring', or None if no match found
        """
        # Reference phrases for "new" problem
        new_phrases = [
            "its new",
            "it's new",
            "this is new",
            "new problem",
            "new issue",
            "new pain",
            "new symptom",
            "never had this",
            "never had pain",
            "never experienced",
            "never felt",
            "first time",
            "brand new",
            "just started",
            "just began",
            "i have never had",
            "have never had"
        ]
        
        # Reference phrases for "recurring" problem
        recurring_phrases = [
            "recurring",
            "ongoing",
            "chronic",
            "for months",
            "for years",
            "always",
            "frequent",
            "often",
            "again",
            "returned",
            "keeps coming back",
            "persistent",
            "long term"
        ]
        
        # Normalize user input
        user_normalized = re.sub(r'[^\w\s]', ' ', user_input.lower())
        
        # Compute Jaccard similarity with all reference phrases
        new_scores = [self._jaccard_similarity(user_normalized, phrase) for phrase in new_phrases]
        recurring_scores = [self._jaccard_similarity(user_normalized, phrase) for phrase in recurring_phrases]
        
        max_new = max(new_scores) if new_scores else 0.0
        max_recurring = max(recurring_scores) if recurring_scores else 0.0
        
        # Threshold for matching (adjust as needed)
        threshold = 0.2  # 20% word overlap is reasonable
        
        # Return the category with highest similarity, if above threshold
        if max_new > max_recurring and max_new >= threshold:
            self._capture_debug(f"[Engine] ✅ Extracted chronicity 'new' via Jaccard (similarity={max_new:.3f})")
            return 'new'
        elif max_recurring > max_new and max_recurring >= threshold:
            self._capture_debug(f"[Engine] ✅ Extracted chronicity 'recurring' via Jaccard (similarity={max_recurring:.3f})")
            return 'recurring'
        
        return None
    
    def _detect_distress(self, user_answer: str) -> Dict[str, Any]:
        """
        Detect patient distress/urgency from their response
        
        Returns:
            Dict with 'is_distressed' (bool), 'severity' (float 0-10), 'urgency_boost' (float)
        """
        user_lower = user_answer.lower()
        
        # High severity indicators
        high_severity_terms = [
            'very severe', 'extremely severe', 'severe pain', 'severe', 'excruciating',
            'unbearable', 'worst pain', 'can\'t stand', 'can\'t bear', 'can\'t handle',
            '10/10', '9/10', '8/10', 'worst ever', 'never felt', 'dying', 'dying pain',
            'excrucitating'  # Common typo
        ]
        
        # Urgent language
        urgent_language = [
            'help me', 'please help', 'need help', 'emergency', 'urgent', 'immediately',
            'right now', 'asap', 'now', 'can\'t wait', 'not well', 'very unwell',
            'really bad', 'really sick', 'critical', 'serious'
        ]
        
        # Emotional distress
        emotional_distress = [
            'scared', 'afraid', 'frightened', 'worried', 'terrified', 'panicking',
            'anxious', 'fear', 'concerned', 'please', 'desperate'
        ]
        
        severity_score = 0.0
        urgency_boost = 0.0
        
        # Check for high severity
        for term in high_severity_terms:
            if term in user_lower:
                severity_score += 2.0
                urgency_boost += 0.5
                break
        
        # Check for urgent language
        for term in urgent_language:
            if term in user_lower:
                severity_score += 1.5
                urgency_boost += 0.3
        
        # Check for emotional distress
        for term in emotional_distress:
            if term in user_lower:
                severity_score += 1.0
                urgency_boost += 0.2
        
        # Check for multiple distress indicators (compound effect)
        distress_count = sum([
            any(term in user_lower for term in high_severity_terms),
            any(term in user_lower for term in urgent_language),
            any(term in user_lower for term in emotional_distress)
        ])
        
        if distress_count >= 2:
            severity_score += 1.0  # Bonus for multiple indicators
            urgency_boost += 0.2
        
        # Cap severity at 10
        severity_score = min(10.0, severity_score)
        
        is_distressed = severity_score >= 3.0  # Threshold for distress
        
        return {
            'is_distressed': is_distressed,
            'severity': severity_score,
            'urgency_boost': urgency_boost,
            'distress_count': distress_count
        }
    
    def _generate_empathetic_response(self, user_answer: str, distress_info: Dict) -> str:
        """Generate empathetic response acknowledging patient distress"""
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for empathetic response generation")
        
        # Get chief complaint context
        chief_complaint_context = f"Patient's chief complaint: {self.chief_complaint}" if self.chief_complaint else "No chief complaint recorded yet"
        
        # Get recent conversation context (what's already been discussed)
        conversation_context = ""
        if hasattr(self, 'conversation_history') and self.conversation_history:
            recent_items = []
            for item in self.conversation_history[-5:]:  # Last 5 items
                if item.get('type') == 'question':
                    recent_items.append(f"Asked: {item.get('question', '')}")
                elif item.get('type') == 'answer':
                    recent_items.append(f"Patient said: {item.get('answer', '')[:100]}")  # Truncate long answers
            if recent_items:
                conversation_context = "\n\nRecent conversation:\n" + "\n".join(recent_items[-3:])  # Last 3 items
        
        system_msg = self.LLM_EMPATHETIC_SYSTEM_MSG
        
        user_msg = self.LLM_EMPATHETIC_USER_TEMPLATE.format(
            chief_complaint_context=chief_complaint_context,
            conversation_context=conversation_context,
            user_answer=user_answer,
            severity=distress_info['severity']
        )
        
        llm_kwargs = self._get_llm_kwargs(override_max_tokens=60)
        response = self.llm_chat_simple_fn(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            **llm_kwargs
        )
        
        return response.strip()
    
    def _detect_red_flags_in_input(self, user_input: str) -> Dict[str, Any]:
        """
        Detect red flag symptoms mentioned in user input.
        Checks for common critical symptoms that warrant immediate ER/911.
        
        Returns:
            Dict with 'red_flags_detected' (list), 'red_flag_count' (int), 'is_severe' (bool)
        """
        user_lower = user_input.lower()
        
        # Common red flag symptoms across conditions
        red_flag_keywords = [
            # Cardiovascular/Shock
            'chest pain', 'heart attack', 'can\'t breathe', 'shortness of breath', 'difficulty breathing',
            'dizzy', 'lightheaded', 'fainting', 'passed out', 'unconscious',
            'rapid heart', 'fast heart', 'heart racing', 'palpitations',
            
            # Neurological
            'severe headache', 'worst headache', 'confused', 'disoriented', 'altered mental',
            'seizure', 'convulsion', 'unresponsive',
            
            # Abdominal emergencies
            'rigid abdomen', 'board-like abdomen', 'hard abdomen', 'guarding',
            'peritonitis', 'rebound tenderness',
            
            # Severe pain indicators
            'excruciating', 'excrucitating', 'unbearable', 'worst pain ever', '10/10', '9/10',
            'can\'t move', 'can\'t stand', 'doubled over',
            
            # High fever/infection
            'high fever', 'fever over 103', 'fever 103', 'fever 104', 'fever 105',
            'chills', 'shaking chills', 'rigors',
            
            # Bleeding
            'severe bleeding', 'heavy bleeding', 'bleeding that won\'t stop',
            'blood in stool', 'black stool', 'tarry stool', 'vomiting blood',
            
            # Trauma/Injury
            'penetrating injury', 'stab wound', 'gunshot', 'severe trauma',
            'head injury', 'loss of consciousness', 'unconscious',
            
            # Shock signs
            'cold', 'clammy', 'sweating profusely', 'very pale', 'gray',
            'rapid pulse', 'weak pulse', 'thready pulse',
            
            # Pregnancy-related
            'pregnant', 'ectopic', 'ruptured', 'bleeding during pregnancy',
            
            # Other critical
            'severe allergic reaction', 'anaphylaxis', 'swelling', 'throat closing',
            'severe dehydration', 'can\'t keep fluids down', 'severe vomiting'
        ]
        
        detected_red_flags = []
        for keyword in red_flag_keywords:
            if keyword in user_lower:
                detected_red_flags.append(keyword)
        
        # Also check for severity + multiple symptoms combination
        severity_indicators = ['severe', 'extreme', 'worst', 'excruciating', 'unbearable', '10/10', '9/10']
        has_severity = any(indicator in user_lower for indicator in severity_indicators)
        
        # Multiple symptoms mentioned together = more concerning
        symptom_indicators = ['pain', 'fever', 'nausea', 'vomiting', 'dizzy', 'bleeding', 'shortness of breath']
        symptom_count = sum(1 for symptom in symptom_indicators if symptom in user_lower)
        
        # Severe case if:
        # 1. Multiple red flags detected (2+)
        # 2. One red flag + high severity language
        # 3. One red flag + multiple symptoms (3+)
        is_severe = (
            len(detected_red_flags) >= 2 or
            (len(detected_red_flags) >= 1 and has_severity) or
            (len(detected_red_flags) >= 1 and symptom_count >= 3)
        )
        
        return {
            'red_flags_detected': detected_red_flags,
            'red_flag_count': len(detected_red_flags),
            'is_severe': is_severe,
            'has_severity_language': has_severity,
            'symptom_count': symptom_count
        }
    
    def _check_emergent_tags_in_input(self, user_input: str) -> list:
        """
        Check if user input matches OLDCARTS terms with emergent tags.
        Returns list of matched emergent terms.
        """
        if not self.active_guidelines:
            return []
        
        user_lower = user_input.lower()
        emergent_terms = []
        
        # Check all active guidelines for emergent-tagged terms
        for guideline in self.active_guidelines:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            if not structured:
                continue
            
            # Check each OLDCARTS element
            for element_name, element_data in structured.items():
                if not isinstance(element_data, dict) or 'includes' not in element_data:
                    continue
                
                includes = element_data.get('includes', [])
                for term_obj in includes:
                    if not isinstance(term_obj, dict):
                        continue
                    
                    # Check if term has emergent tag
                    if not term_obj.get('emergent', False):
                        continue
                
                    # Check if user input matches this term
                    medical = term_obj.get('medical', '').lower()
                    patient_friendly = term_obj.get('patient_friendly', '').lower()
                    
                    # Match if medical or patient_friendly term appears in user input
                    if medical and medical in user_lower:
                        emergent_terms.append(f"{medical} ({element_name})")
                    elif patient_friendly and patient_friendly in user_lower:
                        emergent_terms.append(f"{patient_friendly} ({element_name})")
        
        return emergent_terms
    
    def _generate_emergency_response(self, user_input: str, red_flag_info: Dict = None, distress_info: Dict = None) -> Dict[str, Any]:
        """
        Generate immediate emergency response (911/ER) when severe distress or multiple red flags detected.
        Skips all questions and goes straight to emergency recommendation.
        """
        # Build emergency message
        emergency_msg = """🚨 EMERGENCY ASSESSMENT

Based on your symptoms, you need immediate medical attention.

URGENCY LEVEL: EMERGENT (10/10)

RECOMMENDATION: Call 911 or go to the emergency room immediately.

Please do not wait. Your symptoms indicate a potentially serious condition that requires immediate evaluation by emergency medical professionals."""
        
        # Add specific red flags if detected
        if red_flag_info and red_flag_info.get('red_flags_detected'):
            red_flags = red_flag_info['red_flags_detected'][:3]  # Top 3
            emergency_msg += f"\n\nDetected critical symptoms: {', '.join(red_flags)}"
        
        # Mark assessment as complete with emergency status
        return {
            'success': True,
            'status': 'emergency',
            'message': emergency_msg,
            'skip_questions': True,  # Flag to skip all further questions
            'debug': {
                'engine': self._format_engine_debug("[Engine] 🚨 EMERGENCY: Severe distress/multiple red flags detected - immediate 911/ER recommendation"),
                'internal': self._get_debug_info()
            }
        }
    
    def _check_and_handle_deviating_comment(self, user_answer: str, expected_element: str = None, last_q: Dict = None) -> Dict[str, Any]:
        """
        Unified function to check for and handle deviating comments/questions/distress.
        Called at the start of processing any user input.
        
        Returns:
            Dict with response if comment/question/distress detected and handled, None otherwise
            Also returns interpretation dict for use in further processing
        """
        # Interpret the response (detects comments, questions, distress, or direct answers)
        response_interpretation = self._interpret_patient_response(user_answer, expected_element, last_q=last_q)
        
        # Extract info
        distress_info = response_interpretation.get('distress_info', {})
        is_distressed = response_interpretation.get('is_distressed', False)
        needs_acknowledgment = response_interpretation.get('needs_acknowledgment', False)
        acknowledgment_msg = response_interpretation.get('acknowledgment_message')
        is_question = response_interpretation.get('is_question', False)
        extracted_info = response_interpretation.get('extracted_info')
        
        # CRITICAL: Check for severe distress or multiple red flags FIRST
        # If detected, skip all questions and recommend 911/ER immediately
        red_flag_info = self._detect_red_flags_in_input(user_answer)
        severity_score = distress_info.get('severity', 0.0)
        
        # Check for emergent tags in matched OLDCARTS terms
        emergent_terms_detected = self._check_emergent_tags_in_input(user_answer)
        emergent_term_count = len(emergent_terms_detected) if emergent_terms_detected else 0
        
        # SEVERE EMERGENCY criteria (very restrictive - only truly life-threatening cases):
        # 1. Very high distress severity (9.0+)
        # 2. Multiple red flags (3+) indicating multiple critical systems affected
        # 3. Life-threatening red flags (chest pain, can't breathe, unconscious, etc.)
        # 4. Severe red flag + very high severity (8.0+)
        # 5. Multiple emergent OLDCARTS terms matched (2+) - indicates emergent presentation
        # 6. One emergent term + high severity (7.0+)
        red_flag_count = red_flag_info.get('red_flag_count', 0)
        has_severity_language = red_flag_info.get('has_severity_language', False)
        
        # Check for life-threatening red flags
        life_threatening_keywords = [
            'chest pain', 'heart attack', 'can\'t breathe', 'shortness of breath', 'difficulty breathing',
            'unconscious', 'passed out', 'fainting', 'seizure', 'convulsion',
            'severe bleeding', 'vomiting blood', 'blood in stool', 'black stool'
        ]
        user_lower = user_answer.lower()
        has_life_threatening = any(keyword in user_lower for keyword in life_threatening_keywords)
        
        is_severe_emergency = (
            severity_score >= 9.0 or  # Very high threshold
            (has_life_threatening and severity_score >= 7.0) or  # Life-threatening + high severity
            red_flag_count >= 3 or  # Multiple critical symptoms
            (has_life_threatening and red_flag_count >= 1) or  # Life-threatening symptom + any other red flag
            emergent_term_count >= 2 or  # Multiple emergent OLDCARTS terms matched
            (emergent_term_count >= 1 and severity_score >= 7.0)  # Emergent term + high severity
        )
        
        if is_severe_emergency:
            self._capture_debug(f"[Engine] 🚨 SEVERE EMERGENCY DETECTED: severity={severity_score:.1f}, red_flags={red_flag_info.get('red_flag_count', 0)}, emergent_terms={emergent_term_count}")
            if emergent_term_count > 0:
                self._capture_debug(f"[Engine] 🚨 Emergent OLDCARTS terms matched: {emergent_terms_detected}")
            self._capture_debug(f"[Engine] 🚨 Skipping all questions - recommending immediate 911/ER")
            return self._generate_emergency_response(user_answer, red_flag_info, distress_info), response_interpretation
        
        # Handle distress if detected (but not severe enough for immediate emergency)
        # For distress, acknowledge naturally and continue conversation - do NOT skip questions
        if is_distressed:
            self._capture_debug(f"[Engine] 🚨 DISTRESS DETECTED (moderate): severity={distress_info['severity']:.1f}, urgency_boost={distress_info['urgency_boost']:.2f}")
            # Do NOT skip demographics - only emergency cases skip questions
            # Boost urgency for active guidelines (proportional to severity)
            if distress_info.get('urgency_boost', 0) > 0 and self.active_guidelines:
                for guideline in self.active_guidelines:
                    current_score = guideline.get('score', 0.0)
                    guideline['score'] = min(1.0, current_score + distress_info['urgency_boost'])
                self._capture_debug(f"[Engine] ⚡ Urgency boost applied: +{distress_info['urgency_boost']:.2f}")
        
        # Handle pure questions (with no clinical info)
        # Acknowledge naturally and continue conversation
        if is_question and not extracted_info:
            # Pure question - acknowledge intelligently and return to next missing element
            if needs_acknowledgment:
                # Natural acknowledgment - return to next missing element
                return self._return_to_next_missing_element(acknowledgment_msg, last_user_input=user_answer), response_interpretation
            return self._handle_user_question(user_answer), response_interpretation
        
        # Handle comments/distress (with or without extractable clinical info)
        # CRITICAL: Distress ALWAYS needs acknowledgment, regardless of extracted_info
        # If there's extracted info, process it first, then acknowledge in next question
        # If no extracted info, acknowledge and return to next missing element
        if needs_acknowledgment:
            # CRITICAL: For distress, ALWAYS ensure acknowledgment is stored and will be included
            if is_distressed:
                # Distress detected - ensure acknowledgment is stored
                if not acknowledgment_msg:
                    # Generate it if not already generated
                    acknowledgment_msg = self._generate_empathetic_response(user_answer, distress_info)
                    self._capture_debug(f"[Engine] 💬 Generated distress acknowledgment on the fly")
                
                # Store for later use
                self._pending_acknowledgment = acknowledgment_msg
                self._capture_debug(f"[Engine] 💬 Distress acknowledgment stored: '{acknowledgment_msg[:60]}...'")
            
            if extracted_info:
                # Comment/distress WITH clinical info - process answer first, acknowledgment will be included in next question
                if is_distressed:
                    self._capture_debug(f"[Engine] 💬 Distress WITH clinical info detected - will process answer and acknowledge in next question")
                else:
                    self._capture_debug(f"[Engine] 💬 Comment with clinical info detected - will process answer and acknowledge in next question")
                # Don't return here - let normal processing handle the extracted info
                # The acknowledgment is already stored in _pending_acknowledgment AND will be passed to processing
            else:
                # Pure comment/distress - acknowledge naturally and return to next missing element
                if is_distressed:
                    self._capture_debug(f"[Engine] 💬 Pure distress detected - acknowledging naturally and returning to next missing element")
                    # For distress: show acknowledgment + transition message, then continue normal flow
                    transition_msg = "Let me ask you some questions to help figure out what's going on."
                    combined_msg = f"{acknowledgment_msg}\n\n{transition_msg}"
                    return self._return_to_next_missing_element(combined_msg, last_user_input=user_answer), response_interpretation
                else:
                    self._capture_debug(f"[Engine] 💬 Pure comment detected - acknowledging naturally and returning to next missing element")
                    return self._return_to_next_missing_element(acknowledgment_msg, last_user_input=user_answer), response_interpretation
        
        # If we get here, either:
        # 1. No comment/question/distress (normal answer)
        # 2. Comment/question with extractable clinical info (process answer, then acknowledge)
        # Return None to continue normal processing
        return None, response_interpretation
    
    def process_answer(self, user_answer: str) -> Dict[str, Any]:
        """
        Process user answer and continue assessment
        
        ALWAYS checks for comments/distress/questions FIRST, like a human would,
        before processing any specific answer type.
        """
        # STEP 0: Apply fuzzy matching to correct typos in user response (ALWAYS RUN)
        if self.fuzzy_matcher:
            original_answer = user_answer
            user_answer = self.fuzzy_matcher.fuzzy_correct_medical_terms(user_answer)
            if user_answer != original_answer:
                self._capture_debug(f"[Fuzzy] 🔄 Corrected typos: '{original_answer}' → '{user_answer}'")
        
        # Get context about what we're expecting
        last_q = None
        for item in reversed(self.conversation_history):
            if item.get('type') in ['question', 'statement']:
                last_q = item
                break
                        
        expected_element = last_q.get('oldcarts') if last_q else None
        
        # ALWAYS FIRST: Check for and handle deviating comments/questions/distress
        # Pass last_q to help with extraction context
        comment_response, response_interpretation = self._check_and_handle_deviating_comment(user_answer, expected_element, last_q=last_q)
        if comment_response:
            # Comment/question/distress was handled - return the response
            return comment_response
        
        # Extract info from interpretation (for use in processing)
        is_distressed = response_interpretation.get('is_distressed', False)
        needs_acknowledgment = response_interpretation.get('needs_acknowledgment', False)
        acknowledgment_msg = response_interpretation.get('acknowledgment_message')
        extracted_info = response_interpretation.get('extracted_info')
        
        # Store acknowledgment for later if answer contains clinical info AND needs acknowledgment
        # CRITICAL: Always store acknowledgment if distress/comment/question detected
        if needs_acknowledgment:
            self._pending_acknowledgment = acknowledgment_msg
            self._capture_debug(f"[Engine] 💬 Acknowledgment stored for next question: {response_interpretation['type']}")
            if is_distressed:
                self._capture_debug(f"[Engine] 💬 Distress acknowledgment stored: '{acknowledgment_msg[:50]}...'")
        
        # Record answer (with original text for context)
        self.conversation_history.append({
            'type': 'answer',
            'answer': user_answer,
            'interpretation': response_interpretation  # Store interpretation for debugging
        })
        
        # Use extracted info if available (cleaner than full response with comments)
        processing_answer = extracted_info if extracted_info else user_answer
        
        # Note: Distress is acknowledged but does NOT skip questions - only severe emergencies skip
        # The acknowledgment will be included in the next question response
        
        # Get last question
        last_q = None
        for item in reversed(self.conversation_history):
            item_is_question = item['type'] == 'question'
            item_is_statement = item['type'] == 'statement'
            
            if item_is_question or item_is_statement:
                last_q = item
                break
        
        self._capture_debug(f"[Engine] 🔍 Last question: {last_q}")
        self._capture_debug(f"[Engine] 🔍 User answer: '{user_answer}'")
        self._capture_debug(f"[Engine] 🔍 Conversation history length: {len(self.conversation_history)}")
        self._capture_debug(f"[Engine] 🔍 Demographics: {self.demographics}")
        
        # Handle age answers - check if we just asked an age question
        # Note: Interpretation already happened at start of process_answer
        self._capture_debug(f"[Engine] 🔍 Checking age processing: last_q={last_q.get('focus') if last_q else None}, age in demographics: {'age' in self.demographics}")
        if (last_q and last_q.get('type') == 'question' and last_q.get('focus') == 'age' and 
            'age' not in self.demographics):
            # Process age answer (use processing_answer which may have extracted info)
            age_str = processing_answer.strip()
            self._capture_debug(f"[Engine] 🔍 Processing age: '{age_str}'")
            if age_str.isdigit():
                age = int(age_str)
                if 0 <= age <= 150:
                    self.demographics['age'] = age
                    self._capture_debug(f"[Engine] ✅ Age set to: {age}")
                    self._capture_debug(f"[Engine] 🔍 Demographics after age: {self.demographics}")
                    return self._generate_ml_first_question_with_demographics()
                else:
                    self._capture_debug(f"[Engine] ❌ Age out of range: {age}")
            else:
                self._capture_debug(f"[Engine] ❌ Age not numeric: '{age_str}'")
                # No fallback - return error with the correct age question
                return {
                    'success': False,
                    'message': 'Can you please tell me your age so I can update our medical records?',
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] ❌ Age validation failed - no fallback"),
                        'internal': self._get_debug_info(last_answer=user_answer)
                    }
                }
        
        # If we reach here without processing age, continue to other handlers
        # (Don't return here - let other handlers process the answer)
        
        
        # Handle red flag answers
        if last_q and last_q.get('focus') == 'red_flag':
            red_flag_data = last_q.get('red_flag_data')
            if red_flag_data:
                self._process_red_flag_answer(user_answer, red_flag_data)
                self.red_flag_index += 1
                return self._ask_next_red_flag()
            else:
                self._capture_debug("[Engine] ⚠️ Missing red_flag_data in red flag question")
                self.red_flag_index += 1
                return self._ask_next_red_flag()
        
        # Handle key feature answers
        if last_q and last_q.get('focus') == 'key_feature':
            feature_data = last_q.get('feature_data')
            if feature_data:
                self._process_key_feature_answer(user_answer, feature_data)
                self.key_features_index += 1
                return self._ask_next_key_feature()
            else:
                self._capture_debug("[Engine] ⚠️ Missing feature_data in key feature question")
                self.key_features_index += 1
                return self._ask_next_key_feature()
        
        # Handle other demographics (sex, chronicity)
        # Note: Interpretation already happened at start of process_answer
        if last_q and last_q.get('focus') == 'sex':
            # Simple sex validation - accept direct answers
            # Use processing_answer (may have extracted info)
            sex_str = processing_answer.strip().lower()
            if sex_str in ['male', 'female', 'm', 'f']:
                if sex_str in ['m', 'f']:
                    sex_str = 'male' if sex_str == 'm' else 'female'
                self.demographics['sex'] = sex_str
                
                # Check if this was asked post-assessment
                if last_q and last_q.get('post_assessment'):
                    # After assessment - check if there are more missing demographics
                    missing_demo_response = self._check_and_collect_missing_demographics()
                    if missing_demo_response:
                        return missing_demo_response
                    # All demographics collected - assessment truly complete
                    completion_msg = self._generate_completion_message()
                    return {
                        'success': True,
                        'status': 'completed',
                        'message': completion_msg,
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - all demographics collected"),
                            'internal': self._get_debug_info()
                        }
                    }
                
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'sex_male':
                self.demographics['sex'] = 'male'
                
                # Check if this was asked post-assessment
                if last_q and last_q.get('post_assessment'):
                    missing_demo_response = self._check_and_collect_missing_demographics()
                    if missing_demo_response:
                        return missing_demo_response
                    completion_msg = self._generate_completion_message()
                    return {
                        'success': True,
                        'status': 'completed',
                        'message': completion_msg,
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - all demographics collected"),
                            'internal': self._get_debug_info()
                        }
                    }
                
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'sex_female':
                self.demographics['sex'] = 'female'
                
                # Check if this was asked post-assessment
                if last_q and last_q.get('post_assessment'):
                    missing_demo_response = self._check_and_collect_missing_demographics()
                    if missing_demo_response:
                        return missing_demo_response
                    completion_msg = self._generate_completion_message()
                    return {
                        'success': True,
                        'status': 'completed',
                        'message': completion_msg,
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - all demographics collected"),
                            'internal': self._get_debug_info()
                        }
                    }
                
                return self._generate_ml_first_question_with_demographics()
            
            # Fallback to LLM if simple validation fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Extract the patient's biological sex from their response. Return ONLY 'male', 'female', or 'invalid'."
                user_msg = f"Patient said: '{processing_answer}'\n\nExtract biological sex (male/female) only:"
                
                llm_kwargs = self._get_llm_kwargs(override_max_tokens=10)
                response = self.llm_chat_simple_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    **llm_kwargs
                )
                
                sex_str = response.strip().lower()
                if sex_str in ['male', 'female']:
                    self.demographics['sex'] = sex_str
                    
                    # Check if this was asked post-assessment
                    if last_q and last_q.get('post_assessment'):
                        # After assessment - check if there are more missing demographics
                        missing_demo_response = self._check_and_collect_missing_demographics()
                        if missing_demo_response:
                            return missing_demo_response
                        # All demographics collected - assessment truly complete
                        completion_msg = self._generate_completion_message()
                        return {
                            'success': True,
                            'status': 'completed',
                            'message': completion_msg,
                            'debug': {
                                'engine': self._format_engine_debug("[Engine] ✅ Assessment complete - all demographics collected"),
                                'internal': self._get_debug_info()
                            }
                        }
                    
                    return self._generate_ml_first_question_with_demographics()
            
            return {
                'success': False,
                'message': 'Please specify your biological sex (male or female)',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ❌ Sex validation failed"),
                    'internal': self._get_debug_info(last_answer=user_answer)
                }
            }
        
        self._capture_debug(f"[Engine] 🔍 Checking chronicity processing: last_q focus={last_q.get('focus') if last_q else None}, chronicity in demographics: {'chronicity' in self.demographics}")
        if last_q and last_q.get('focus') == 'chronicity':
            # Note: Interpretation already happened at start of process_answer
            # Distress is acknowledged but does NOT skip questions - only severe emergencies skip
            # Use the processing_answer (which may have extracted info) for keyword matching
            
            # Check for button callbacks first
            if user_answer == 'chronicity_new':
                self.demographics['chronicity'] = 'new'
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'chronicity_recurring':
                self.demographics['chronicity'] = 'recurring'
                return self._generate_ml_first_question_with_demographics()
            
            # Simple keyword matching first (more reliable than LLM)
            # Use processing_answer (may have extracted info, or original if no extraction)
            processing_answer_lower = processing_answer.lower().strip()
            
            # Use extracted_info if available (from _interpret_patient_response)
            self._capture_debug(f"[Engine] 🔍 Chronicity extraction - extracted_info: {extracted_info}, processing_answer: {processing_answer}")
            chronicity_value = None
            
            if extracted_info and extracted_info in ['new', 'recurring']:
                # Extracted info already found - use it directly
                chronicity_value = extracted_info
                self._capture_debug(f"[Engine] ✅ Chronicity extracted from extracted_info: {chronicity_value}")
            else:
                # Try Jaccard similarity extraction
                chronicity_result = self._extract_chronicity_with_jaccard(processing_answer)
                self._capture_debug(f"[Engine] 🔍 Jaccard chronicity result: {chronicity_result}")
                if chronicity_result:
                    chronicity_value = chronicity_result
                    self._capture_debug(f"[Engine] ✅ Chronicity extracted from Jaccard: {chronicity_value}")
                else:
                    # No fallback - if we can't determine chronicity, return error
                    self._capture_debug(f"[Engine] ❌ Could not determine chronicity from answer")
                    return {
                        'success': False,
                        'message': 'I need to know if this is a new problem or something you\'ve had before. Please tell me if it\'s new or recurring.',
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ❌ Chronicity extraction failed - no fallback"),
                            'internal': self._get_debug_info(last_answer=user_answer)
                        }
                    }
            
            # Save chronicity value (should always have a value here if we didn't return above)
            if not chronicity_value:
                self._capture_debug(f"[Engine] ❌ CRITICAL: No chronicity value to save!")
                return {
                    'success': False,
                    'message': 'I need to know if this is a new problem or something you\'ve had before. Please tell me if it\'s new or recurring.',
                    'debug': {
                        'engine': self._format_engine_debug("[Engine] ❌ Chronicity extraction failed - no value to save"),
                        'internal': self._get_debug_info(last_answer=user_answer)
                    }
                }
            
            self.demographics['chronicity'] = chronicity_value
            self._capture_debug(f"[Engine] ✅ Chronicity saved to demographics: {chronicity_value}, demographics now: {self.demographics}")
            
            # If chronicity was successfully set, proceed to next missing element
            if 'chronicity' in self.demographics:
                # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                # Pass user_answer to check for emergency/distress again
                # CRITICAL: Always include acknowledgment if there was distress/comment/question
                # Priority: 1) acknowledgment_msg from interpretation, 2) _pending_acknowledgment
                final_acknowledgment = None
                if needs_acknowledgment and acknowledgment_msg:
                    # Use the acknowledgment message from interpretation (most current)
                    final_acknowledgment = acknowledgment_msg
                    # If distress detected, add transition message
                    if is_distressed:
                        transition_msg = "Let me ask you some questions to help figure out what's going on."
                        final_acknowledgment = f"{final_acknowledgment}\n\n{transition_msg}"
                    self._capture_debug(f"[Engine] 💬 Using acknowledgment from interpretation")
                elif hasattr(self, '_pending_acknowledgment') and self._pending_acknowledgment:
                    # Use stored acknowledgment (no fallback - this is the stored value)
                    final_acknowledgment = self._pending_acknowledgment
                    self._pending_acknowledgment = None
                    # If distress was detected, add transition message
                    # Check if distress was detected in the conversation
                    distress_detected = is_distressed or getattr(self, 'demographics_optional', False)
                    if distress_detected:
                        transition_msg = "Let me ask you some questions to help figure out what's going on."
                        final_acknowledgment = f"{final_acknowledgment}\n\n{transition_msg}"
                    self._capture_debug(f"[Engine] 💬 Using stored pending acknowledgment")
                
                if final_acknowledgment:
                    self._capture_debug(f"[Engine] ✅ Chronicity processed - returning with acknowledgment: '{final_acknowledgment[:50]}...'")
                else:
                    self._capture_debug(f"[Engine] ✅ Chronicity processed - no acknowledgment needed")
                
                return self._return_to_next_missing_element(
                    final_acknowledgment,
                    last_user_input=user_answer
                )
            
            # No LLM fallback - if chronicity wasn't set above, return error
            # (This should not happen if extraction worked, but if it does, we fail)
            return {
                'success': False,
                'message': 'I need to know if this is a new problem or something you\'ve had before. Please tell me if it\'s new or recurring.',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ❌ Chronicity processing failed - no fallback"),
                    'internal': self._get_debug_info(last_answer=user_answer)
                }
            }
        
        # If no demographics handler matched and demographics incomplete, check if we need to ask demographics
        # Demographics are always collected unless it's a severe emergency (911/ER case)
        required_demographics = ['chronicity', 'age', 'sex']  # Always need all demographics
        
        if not all(key in self.demographics for key in required_demographics):
            # Demographics incomplete - ask next demographic question
            return self._generate_ml_first_question_with_demographics()
        
        # Handle clinical answers (use processing_answer which may have extracted info)
        return self._process_clinical_answer(processing_answer)
    
    def _is_user_asking_question(self, user_input: str) -> bool:
        """Detect if user is asking a question instead of answering"""
        question_indicators = [
            'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'can you', 'could you', 'would you', 'please explain',
            'what do you mean', 'what does that mean', 'i don\'t understand',
            'i\'m confused', 'explain', 'clarify', '?'
        ]
        
        user_lower = user_input.lower().strip()
        
        # Check for question marks
        if '?' in user_lower:
            return True
        
        # Check for question indicators at start
        for indicator in question_indicators:
            if user_lower.startswith(indicator):
                return True
        
        # Check for question patterns
        question_indicators = ['what do you mean', 'what does that mean', 'i don\'t understand']
        has_question_indicator = any(indicator in user_lower for indicator in question_indicators)
        
        if has_question_indicator:
            return True
        
        return False
    
    def _handle_user_question(self, user_question: str) -> Dict[str, Any]:
        """Handle user questions - just repeat the last question (no LLM generation)"""
        # Find the last question asked
        last_q = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_q = item
                break
        if last_q and last_q.get('focus') == 'clinical' and last_q.get('oldcarts') == 'location':
            # Use existing _generate_clarifying_question method instead of duplicating logic
            # Get the last patient answer for context
            last_answer = ""
            for item in reversed(self.conversation_history):
                if item.get('type') == 'answer':
                    last_answer = item.get('answer', '')
                    break
            
            # Get missing location terms from active guidelines (medical terms)
            missing_terms = []
            for guideline in self.active_guidelines:
                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                location_data = structured.get('location', {})
                if isinstance(location_data, dict):
                    includes = location_data.get('includes', [])
                    for term_obj in includes:
                        if isinstance(term_obj, dict):
                            medical_term = term_obj.get('medical', '')
                            if medical_term and medical_term not in missing_terms:
                                missing_terms.append(medical_term)
                                # No limit - process all terms
                                if len(missing_terms) >= 20:  # Reasonable limit to prevent excessive processing
                                    break
                    if len(missing_terms) >= 20:
                        break
            
            if not missing_terms:
                raise ValueError("No location options available for clarification question")
            
            # Use the existing method
            msg = self._generate_clarifying_question('location', last_answer or "right side", 0, missing_terms)
            # Track as clarification question
            self.conversation_history.append({
                'type': 'question',
                'question': msg,
                'oldcarts': 'location',
                'is_clarification': True
            })
            return {
                'success': True,
                'message': msg,
                'status': 'questioning'
            }
        
        # Generate clarifying question using missing terms from guidelines
        if last_q:
            oldcarts_element = last_q.get('oldcarts', '')
            if not oldcarts_element:
                raise ValueError("Cannot generate clarification - no OLDCARTS element found in last question")
            
            # Get the last patient answer for context
            last_answer = ""
            for item in reversed(self.conversation_history):
                if item.get('type') == 'answer':
                    last_answer = item.get('answer', '')
                    break
            
            # Get missing terms from active guidelines for this element
            missing_terms = []
            for guideline in self.active_guidelines:
                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                element_data = structured.get(oldcarts_element, {})
                if isinstance(element_data, dict):
                    includes = element_data.get('includes', [])
                    for term_obj in includes:
                        if isinstance(term_obj, dict):
                            medical_term = term_obj.get('medical', '')
                            if medical_term and medical_term not in missing_terms:
                                missing_terms.append(medical_term)
                                if len(missing_terms) >= 20:  # Reasonable limit
                                    break
                    if len(missing_terms) >= 20:
                        break
            
            if not missing_terms:
                raise ValueError(f"No {oldcarts_element} options available from guidelines for clarification question")
            
            # Use the existing method to generate clarifying question with missing terms
            msg = self._generate_clarifying_question(oldcarts_element, last_answer or "", 0, missing_terms)
            
            # Track as clarification question
            self.conversation_history.append({
                'type': 'question',
                'question': msg,
                'oldcarts': oldcarts_element,
                'is_clarification': True
            })
            return {
                'success': True,
                'message': msg,
                'status': 'questioning'
            }
        else:
            raise ValueError("Cannot generate clarification - no previous question found in conversation history")
    
    # ============================================================================
    # SECTION 5: OLDCARTS PROCESSING - Location, Onset, Duration, Character, etc.
    # ============================================================================
    
    # REMOVED: _parse_prompt_against_structured_oldcarts - only used in test files, not in production
    
    def _get_structured_oldcarts(self, guideline: Dict) -> Dict:
        """Helper to extract structured_oldcarts from guideline (handles both data wrapper and direct)"""
        structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
        if not structured:
            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
        return structured
    
    def _get_active_condition_names(self, guidelines: List[Dict] = None) -> set:
        """Helper to extract active condition names from guidelines"""
        if guidelines is None:
            guidelines = self.active_guidelines
        active_condition_names = set()
        for g in guidelines:
            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
            if condition_name:
                active_condition_names.add(condition_name)
        return active_condition_names
    
    def _extract_patient_friendly_from_includes(self, includes: List) -> List[str]:
        """Helper to extract patient_friendly terms from includes list"""
        patient_friendly_terms = []
        for term_obj in includes:
            if isinstance(term_obj, dict):
                pf_term = term_obj.get('patient_friendly', '')
                if pf_term and isinstance(pf_term, str) and pf_term.strip():
                    patient_friendly_terms.append(pf_term.strip())
            elif isinstance(term_obj, str):
                # Fallback: use string as patient_friendly term if no dict structure
                if term_obj.strip():
                    patient_friendly_terms.append(term_obj.strip())
        return patient_friendly_terms
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """Score guidelines using unified similarity function"""
        # Apply fuzzy matching to correct typos (ALWAYS RUN)
        if self.fuzzy_matcher:
            original_answer = answer
            answer = self.fuzzy_matcher.fuzzy_correct_medical_terms(answer)
            if answer != original_answer:
                self._capture_debug(f"[Fuzzy] 🔄 Corrected clinical answer typos: '{original_answer}' → '{answer}'")
        
        # Get context about what we're expecting
        last_q = None
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question':
                last_q = item
                break
        
        expected_element = last_q.get('oldcarts') if last_q else None
        
        # PRIORITY: Intelligently interpret the response (comments, questions, distress, or direct answer)
        response_interpretation = self._interpret_patient_response(answer, expected_element)
        
        # Extract info for processing
        distress_info = response_interpretation.get('distress_info', {})
        distress_detected = response_interpretation.get('is_distressed', False)
        
        # Store for later use
        self._last_distress_info = distress_info if distress_detected else None
        
        # Store acknowledgment if response needs it
        if response_interpretation['needs_acknowledgment']:
            self._pending_acknowledgment = response_interpretation['acknowledgment_message']
        
        # Use extracted info if available (might be cleaner than full response)
        if response_interpretation.get('extracted_info'):
            answer = response_interpretation['extracted_info']
        
        if distress_detected:
            self._capture_debug(f"[Engine] 🚨 DISTRESS DETECTED in clinical answer: severity={distress_info['severity']:.1f}, urgency_boost={distress_info['urgency_boost']:.2f}")
            
            # Note: Distress does NOT skip questions - only severe emergencies skip
            # Boost urgency for active guidelines
            if distress_info['urgency_boost'] > 0 and self.active_guidelines:
                for guideline in self.active_guidelines:
                    current_score = guideline.get('score', 0.0)
                    guideline['score'] = min(1.0, current_score + distress_info['urgency_boost'])
                self._capture_debug(f"[Engine] ⚡ Urgency boost applied: +{distress_info['urgency_boost']:.2f} to active guidelines")
        
        # Get last question
        last_q = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_q = item
                break
        
        oldcarts_element = last_q.get('oldcarts') if last_q else None
        
        if not oldcarts_element:
            return self._ask_next_with_distress_handling(answer)
        
        # Handle demographics
        if last_q.get('focus') == 'demographics':
            if 'sex' not in self.demographics:
                question = "What is your biological sex?"
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'oldcarts': 'demographics',
                    'focus': 'demographics'
                })
                return {
                    'success': True,
                    'message': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'Male', 'callback_data': 'sex_male'},
                        {'text': 'Female', 'callback_data': 'sex_female'}
                    ]
                }
                return self._ask_next_clinical_question()
        
        # Handle radiation (separate question after location)
        if oldcarts_element == 'radiation':
            self.radiation_answered = True
            self._capture_debug(f"[Engine] 📍 Radiation answer: '{answer}'")
            
            # Score guidelines based on radiation answer using radiation section
            if self.medical_rule_engine:
                # Track previous active/reserve state for promotion/demotion logging
                previous_active = set(g.get('data', {}).get('condition', g.get('name')) for g in self.active_guidelines)
                all_guidelines = self.active_guidelines + self.reserve_pool
                
                # Detect organ system from category
                organ_system = self.CATEGORY_TO_SYSTEM.get(self.current_category or 'gastrointestinal', 'GI')
                
                # Score radiation using patient_friendly matching (use raw answer, no normalization)
                for g in all_guidelines:
                    structured_oldcarts = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                    condition_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
                    element_data = structured_oldcarts.get('radiation')
                    
                    # Score radiation: Use patient_friendly matching (radiation is not location)
                    if element_data:
                        similarity = self._match_to_patient_friendly_terms(answer, element_data, 'radiation')
                    else:
                        similarity = 0.0
                    
                    old_score = g.get('score', 0.5)
                    new_score = (old_score * 0.7) + (similarity * 0.3)
                    g['score'] = new_score
                    self._capture_debug(f"[Scoring] 📊 {condition_name}: old={old_score:.3f}, radiation={similarity:.3f}, new={new_score:.3f}")
                
                # Re-rank after radiation scoring
                self._rerank_and_pool_guidelines(all_guidelines, previous_active)
            
            # Continue to next question
            return self._ask_next_clinical_question()
        
        # Handle onset, duration, timing, severity, associated (documentation only - no clarification needed)
        if oldcarts_element in ['onset', 'progression', 'duration', 'timing', 'severity', 'associated']:
            # Mark element as covered and store the answer
            element_map = {'onset': 'O', 'progression': 'P', 'duration': 'D', 'timing': 'T', 'severity': 'S', 'associated': 'AS'}
            if oldcarts_element in element_map:
                self.oldcarts_covered[element_map[oldcarts_element]] = True
                # Update missing_components list to remove this element
                if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                    if oldcarts_element in self.oldcarts_analysis['missing_components']:
                        self.oldcarts_analysis['missing_components'].remove(oldcarts_element)
                # Also update answered_components if it exists
                if self.oldcarts_analysis and 'answered_components' in self.oldcarts_analysis:
                    if oldcarts_element not in self.oldcarts_analysis['answered_components']:
                        self.oldcarts_analysis['answered_components'][oldcarts_element] = []
                    self.oldcarts_analysis['answered_components'][oldcarts_element].append(answer)
            self._capture_debug(f"[Engine] ✅ {oldcarts_element} marked as complete (no clarification)")
            # Include pending acknowledgment if available
            next_response = self._ask_next_clinical_question()
            if hasattr(self, '_pending_acknowledgment') and self._pending_acknowledgment and next_response.get('success'):
                acknowledgment = self._pending_acknowledgment
                self._pending_acknowledgment = None
                next_msg = next_response.get('message') or next_response.get('question', '')
                combined_msg = f"{acknowledgment}\n\n{next_msg}"
                return {
                    'success': True,
                    'message': combined_msg,
                    'status': next_response.get('status', 'questioning'),
                    'debug': next_response.get('debug', {})
                }
            return next_response
        
        # Score guidelines (strict: require embeddings, no fallbacks)
        if not self.embedding_model:
            return {'success': False, 'message': 'Embedding model not available'}
        
        # Track previous active/reserve state for promotion/demotion logging
        previous_active = set(g.get('data', {}).get('condition', g.get('name')) for g in self.active_guidelines)
        all_guidelines = self.active_guidelines + self.reserve_pool
        
        # STEP 1: Filter guidelines using medical_rules.json (location only)
        category = self.current_category or 'gastrointestinal'
        organ_system = self.CATEGORY_TO_SYSTEM.get(category, 'GI')
        
        if oldcarts_element == 'location' and self.medical_rule_engine:
            self._capture_debug(f"[Engine] 🏥 Filtering guidelines using medical_rules.json")
            # Debug: Show scores BEFORE filtering
            scores_before_filter = [(g.get('data', {}).get('condition', g.get('name', 'Unknown')), id(g), g.get('score', 'MISSING')) for g in all_guidelines[:3]]
            self._capture_debug(f"[Engine] 🔍 Before filtering (first 3): {scores_before_filter}")
            all_guidelines = self.medical_rule_engine.filter_guidelines_by_location(answer, all_guidelines, organ_system)
            # Debug: Show scores AFTER filtering
            scores_after_filter = [(g.get('data', {}).get('condition', g.get('name', 'Unknown')), id(g), g.get('score', 'MISSING')) for g in all_guidelines[:3]]
            self._capture_debug(f"[Engine] 🔍 After filtering (first 3): {scores_after_filter}")
        
        # STEP 2: Score all guidelines using patient_friendly semantic matching
        self._capture_debug(f"[Scoring] 🔍 Scoring {len(all_guidelines)} guidelines for element: {oldcarts_element}")
        self._capture_debug(f"[Scoring] 📝 Patient answer: '{answer}'")
        
        # Prepare guideline data for scoring
        # For all elements (including location): collect element_data for patient_friendly matching
        guideline_data = []
        for g in all_guidelines:
            structured_oldcarts = self._get_structured_oldcarts(g)
            element_data = structured_oldcarts.get(oldcarts_element, {})
            
            guideline_data.append({
                'guideline': g,
                'condition_name': g.get('data', {}).get('condition', g.get('name', 'Unknown')),
                'structured_oldcarts': structured_oldcarts,
                'element_data': element_data
            })
        
        # Track which guidelines got scored (for debugging)
        scored_guidelines = set()
        
        # Debug: Show what we're about to score
        self._capture_debug(f"[Scoring] 🔍 About to score {len(guideline_data)} guidelines from {len(all_guidelines)} total")
        for idx, g_data in enumerate(guideline_data[:3]):
            g = g_data['guideline']
            condition_name = g_data['condition_name']
            g_id = id(g)
            g_score = g.get('score', 'MISSING')
            self._capture_debug(f"[Scoring] 🔍 Guideline[{idx}]: {condition_name}, id={g_id}, score={g_score}")
        
        # Pass 2: Score each guideline
        for idx, g_data in enumerate(guideline_data):
            g = g_data['guideline']
            condition_name = g_data['condition_name']
            structured_oldcarts = g_data['structured_oldcarts']
            
            # Score using patient_friendly matching for all elements (including location)
            # Use raw answer directly for semantic matching (no normalization)
            if self.medical_rule_engine:
                element_data = g_data.get('element_data') or structured_oldcarts.get(oldcarts_element)
                similarity = self._match_to_patient_friendly_terms(answer, element_data, oldcarts_element)
                word_match_boost = 0.0  # No word match boost for simplified matching
                
                # Record for ML learning (both matched and unmatched)
                # REMOVED: ML learning code - feature disabled and not used in production
            else:
                similarity = 0.5
                word_match_boost = 0.0
            
            # Update score using category-specific element weights
            old_score = g.get('score', 0.5)
            category = self.current_category or 'gastrointestinal'
            element_weight = self.get_oldcarts_element_weight(category, oldcarts_element)
            
            # Formula: new_score = (old_score * (1 - weight)) + (similarity * weight)
            # Higher weight = this element has more impact on the score
            new_score = (old_score * (1.0 - element_weight)) + (similarity * element_weight)
            g['score'] = new_score
            scored_guidelines.add(condition_name)
            
            self._capture_debug(f"[Scoring] 📊 {condition_name}: old={old_score:.3f}, similarity={similarity:.3f} (boost={word_match_boost:.3f}), weight={element_weight:.2f} ({oldcarts_element}), new={new_score:.3f}")
        
        # Debug: Verify scores were actually updated in the objects
        self._capture_debug(f"[Scoring] 🔍 Verifying scores after update (first 3 from guideline_data):")
        for idx, g_data in enumerate(guideline_data[:3]):
            g = g_data['guideline']
            condition_name = g_data['condition_name']
            g_id = id(g)
            g_score = g.get('score', 'MISSING')
            self._capture_debug(f"[Scoring] 🔍 Guideline[{idx}]: {condition_name}, id={g_id}, score={g_score}")
        
        # Debug: Also check all_guidelines list to see if scores propagate
        self._capture_debug(f"[Scoring] 🔍 Checking all_guidelines list (first 3) after scoring:")
        for idx, g in enumerate(all_guidelines[:3]):
            cond_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
            g_id = id(g)
            g_score = g.get('score', 'MISSING')
            self._capture_debug(f"[Scoring] 🔍 all_guidelines[{idx}]: {cond_name}, id={g_id}, score={g_score}")
        
        # Check if any guidelines weren't scored (missing sections)
        unscored = []
        for g in all_guidelines:
            cond_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
            if cond_name not in scored_guidelines:
                unscored.append(cond_name)
                # Keep old score (don't reset to 0.5)
                if 'score' not in g:
                    g['score'] = 0.5
        
        if unscored:
            self._capture_debug(f"[Scoring] ⚠️ {len(unscored)} guidelines not scored (missing {oldcarts_element} section): {unscored[:5]}")
        
        # Re-rank and pool guidelines
        self._rerank_and_pool_guidelines(all_guidelines, previous_active)
        
        # Check if this is an answer to a clarifying question
        is_clarifying_answer = False
        missing_patient_friendly_terms = []  # Patient_friendly terms from the clarifying question
        
        # Check if last question was a clarification
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question' and item.get('is_clarification') and item.get('oldcarts') == oldcarts_element:
                is_clarifying_answer = True
                # Get missing medical terms from the clarifying question (stored in conversation history)
                missing_medical_terms = item.get('missing_terms', [])
                self._capture_debug(f"[Clarification Answer] 🔍 Detected clarifying question answer. Missing medical terms: {missing_medical_terms}")
                
                # Map missing medical terms back to patient_friendly terms from remaining guidelines
                all_guidelines_for_clarification = self.active_guidelines + self.reserve_pool
                for g in all_guidelines_for_clarification:
                    structured = self._get_structured_oldcarts(g)
                    element_data = structured.get(oldcarts_element, {})
                    if isinstance(element_data, dict):
                        includes = element_data.get('includes', [])
                        for t in includes:
                            if isinstance(t, dict):
                                med = t.get('medical', '')
                                pf = t.get('patient_friendly', '')
                                if med and med.strip().lower() in [mt.lower() for mt in missing_medical_terms]:
                                    if pf and pf.strip() and pf.strip() not in missing_patient_friendly_terms:
                                        missing_patient_friendly_terms.append(pf.strip())
                            elif isinstance(t, str):
                                # Plain string - check if it matches any missing medical term
                                if t.strip().lower() in [mt.lower() for mt in missing_medical_terms]:
                                    if t.strip() not in missing_patient_friendly_terms:
                                        missing_patient_friendly_terms.append(t.strip())
                
                self._capture_debug(f"[Clarification Answer] 🔍 Mapped to patient_friendly terms: {missing_patient_friendly_terms}")
                break
        
        # If this is a clarifying answer, run FAISS again on missing patient_friendly terms
        if is_clarifying_answer and missing_patient_friendly_terms and self.medical_rule_engine:
            self._capture_debug(f"[Clarification Answer] 🔄 Running FAISS on clarifying answer against {len(missing_patient_friendly_terms)} patient_friendly terms")
            
            # Get all condition names from remaining guidelines
            all_condition_names = set()
            for g in all_guidelines:
                condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                if condition_name:
                    all_condition_names.add(condition_name)
            
            # Run FAISS on user response against missing patient_friendly terms
            # This will find which patient_friendly terms match the user's clarifying answer
            faiss_clarification_matches = self.medical_rule_engine.find_matching_terms_faiss(
                answer, oldcarts_element, threshold=self.FAISS_SEMANTIC_THRESHOLD,
                return_scores=True, active_condition_names=all_condition_names
            )
            
            # Get FAISS scores for clarification matching
            clarification_faiss_scores = {}
            if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
                raw_faiss_scores = self.medical_rule_engine._last_faiss_scores
                # Filter to only include missing patient_friendly terms
                clarification_faiss_scores = {term: score for term, score in raw_faiss_scores.items()
                                             if term.lower() in [t.lower() for t in missing_patient_friendly_terms]}
            
            self._capture_debug(f"[Clarification Answer] 🔍 FAISS matches: {faiss_clarification_matches}")
            self._capture_debug(f"[Clarification Answer] 🔍 FAISS scores: {clarification_faiss_scores}")
            
            # Update guideline scores based on clarification FAISS matches
            # Use the highest matching score for each guideline
            for g in all_guidelines:
                structured = self._get_structured_oldcarts(g)
                element_data = structured.get(oldcarts_element, {})
                if isinstance(element_data, dict):
                    includes = element_data.get('includes', [])
                    max_clarification_score = 0.0
                    
                    # Check which patient_friendly terms from this guideline matched
                    for t in includes:
                        if isinstance(t, dict):
                            pf = t.get('patient_friendly', '')
                        elif isinstance(t, str):
                            pf = t
                        else:
                            continue
                        
                        if pf and pf.strip().lower() in [m.lower() for m in faiss_clarification_matches]:
                            score = clarification_faiss_scores.get(pf.strip(), 
                                                                    clarification_faiss_scores.get(pf.strip().lower(), 0.0))
                            max_clarification_score = max(max_clarification_score, score)
                    
                    # Update score using clarification match
                    if max_clarification_score > 0:
                        old_score = g.get('score', 0.5)
                        category = self.current_category or 'gastrointestinal'
                        element_weight = self.get_oldcarts_element_weight(category, oldcarts_element)
                        
                        # Use clarification score to update (similar to initial scoring)
                        new_score = (old_score * (1.0 - element_weight * 0.5)) + (max_clarification_score * element_weight * 0.5)
                        g['score'] = new_score
                        condition_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
                        self._capture_debug(f"[Clarification Answer] 📊 {condition_name}: clarification_score={max_clarification_score:.3f}, old={old_score:.3f}, new={new_score:.3f}")
            
            # Re-rank after clarification scoring
            self._rerank_and_pool_guidelines(all_guidelines, previous_active)
        
        # Check if further clarification needed
        clarification_count = sum(1 for item in self.conversation_history 
                                 if item.get('oldcarts') == oldcarts_element 
                                 and item.get('is_clarification'))
        
        clarification_needed = False
        missing_terms = []  # Missing terms for the current element
        
        if self.active_guidelines:
            try:
                # Get missing terms and satisfied terms - this already does the expensive matching
                missing_terms, satisfied_medical_terms = self._analyze_missing_information(answer, oldcarts_element)
                self._capture_debug(f"[Clarification] 📊 Missing terms: {missing_terms}")
                self._capture_debug(f"[Clarification] ✅ Satisfied medical terms: {satisfied_medical_terms} (count: {len(satisfied_medical_terms)})")
                
                # Decision logic:
                # 1. If satisfied array contains a single satisfied term → no need for clarifying question, move on
                # 2. If satisfied array contains 2+ items → already deduplicated and filtered anatomically in STEP 3
                #    - If 2+ items remain after processing → generate clarifying question with satisfied array context
                # 3. If no satisfied terms (0 items) → generate clarifying question with missing terms
                
                if len(satisfied_medical_terms) == 1:
                    # Single satisfied term - no clarification needed, move on
                    self._capture_debug(f"[Clarification] ✅ Have exactly 1 satisfied medical term - moving on")
                    clarification_needed = False
                elif len(satisfied_medical_terms) >= 2:
                    # 2+ satisfied terms after filtering - need clarification with satisfied array context
                    clarification_needed = True
                    self._capture_debug(f"[Clarification] 🔍 {len(satisfied_medical_terms)} satisfied medical terms after filtering - generating clarifying question with satisfied context")
                    # Use satisfied terms for the clarifying question (user needs to choose which one)
                    question = self._generate_clarifying_question(oldcarts_element, answer, clarification_count, satisfied_medical_terms, satisfied_context=True)
                    self.conversation_history.append({
                        'type': 'question',
                        'question': question,
                        'oldcarts': oldcarts_element,
                        'is_clarification': True,
                        'satisfied_terms': satisfied_medical_terms  # Store satisfied medical terms for later use
                    })
                    return {
                        'success': True,
                        'question': question,
                        'status': 'questioning',
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ⏳ Clarification requested (2+ satisfied terms)") + "\n\n" + self._format_rankings_debug(),
                            'internal': self._get_debug_info(last_answer=answer)
                        }
                    }
                elif len(satisfied_medical_terms) == 0:
                    # No satisfied terms - use missing terms for clarifying question
                    clarification_needed = True
                    self._capture_debug(f"[Clarification] 🔍 No satisfied terms - generating clarifying question with missing terms")
                    question = self._generate_clarifying_question(oldcarts_element, answer, clarification_count, missing_terms)
                    self.conversation_history.append({
                        'type': 'question',
                        'question': question,
                        'oldcarts': oldcarts_element,
                        'is_clarification': True,
                        'missing_terms': missing_terms  # Store missing medical terms for later use
                    })
                    return {
                        'success': True,
                        'question': question,
                        'status': 'questioning',
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ⏳ Clarification requested (no satisfied terms)") + "\n\n" + self._format_rankings_debug(),
                            'internal': self._get_debug_info(last_answer=answer)
                        }
                    }
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Clarification check failed: {e}")
        
        # Only mark element as covered if NO clarification needed
        if not clarification_needed:
            element_map = {'onset': 'O', 'progression': 'P', 'location': 'L', 'timing': 'T', 'duration': 'D',
                          'character': 'C', 'aggravating': 'A', 'relieving': 'R', 'severity': 'S', 'associated': 'AS'}
            if oldcarts_element in element_map:
                self.oldcarts_covered[element_map[oldcarts_element]] = True
                # Update missing_components list to remove this element
                if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                    if oldcarts_element in self.oldcarts_analysis['missing_components']:
                        self.oldcarts_analysis['missing_components'].remove(oldcarts_element)
                self._capture_debug(f"[Engine] ✅ {oldcarts_element} marked as complete")
                
                # Special handling: If timing is answered as constant, mark duration as covered (redundant)
                # EXCEPTION: Still ask duration if guidelines have comparison operators (>, <, >=, <=) to differentiate conditions
                if oldcarts_element == 'timing':
                    answer_lower = answer.lower()
                    constant_words = ['constant', 'continuous', 'always', 'all the time', 'never stops']
                    has_constant_word = any(word in answer_lower for word in constant_words)
                    
                    if has_constant_word:
                        # Check if any active guidelines have duration terms with comparison operators
                        has_duration_comparison = False
                        for guideline in self.active_guidelines:
                            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                            duration_data = structured.get('duration', {})
                            if isinstance(duration_data, dict):
                                includes = duration_data.get('includes', [])
                                for term in includes:
                                    medical_term = term.get('medical', '') if isinstance(term, dict) else term
                                    is_medical_term_string = isinstance(medical_term, str)
                                    comparison_operators = ['>', '<', '>=', '<=']
                                    has_comparison_operator = any(op in medical_term for op in comparison_operators) if is_medical_term_string else False
                                    
                                    if is_medical_term_string and has_comparison_operator:
                                        has_duration_comparison = True
                                        break
                            if has_duration_comparison:
                                break
                        
                        # Only mark duration as covered if NO comparison operators found
                        if not has_duration_comparison:
                            if 'duration' in element_map:
                                self.oldcarts_covered[element_map['duration']] = True
                                if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                                    if 'duration' in self.oldcarts_analysis['missing_components']:
                                        self.oldcarts_analysis['missing_components'].remove('duration')
                                        self._capture_debug(f"[Engine] ⏭️ Duration marked as covered (timing is constant, no comparison operators)")
                        else:
                            self._capture_debug(f"[Engine] ⚠️ Duration still needed (timing is constant but duration has comparison operators for differentiation)")
                
                # If distress was detected and no clarification needed, acknowledge it
                if distress_detected and not clarification_needed:
                    return self._ask_next_with_distress_handling(answer)
                
                # Special handling: After location is satisfied, check if any guidelines have radiation section
                if oldcarts_element == 'location' and not self.radiation_asked:
                    # Check if any active guidelines have a radiation section
                    has_radiation_section = False
                    for guideline in self.active_guidelines:
                        structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                        if structured.get('radiation') and structured['radiation'].get('includes'):
                            has_radiation_section = True
                            break
                    
                    if has_radiation_section:
                        self._capture_debug(f"[Engine] 📍 Location satisfied - radiation section found in guidelines")
                        return self._ask_about_radiation()
        else:
            self._capture_debug(f"[Engine] ⏳ {oldcarts_element} needs clarification - not marked complete")
        
        # Continue to next question
        # Check if there's a pending acknowledgment and return to next missing element intelligently
        if hasattr(self, '_pending_acknowledgment') and self._pending_acknowledgment:
            acknowledgment_msg = self._pending_acknowledgment
            self._pending_acknowledgment = None
            # Get last user input from conversation history for emergency check
            last_user_input = None
            for item in reversed(self.conversation_history):
                if item.get('type') == 'answer':
                    last_user_input = item.get('answer', '')
                    break
            return self._return_to_next_missing_element(acknowledgment_msg, last_user_input=last_user_input)
        
        return self._ask_next_clinical_question()
    
    
    # ============================================================================
    # SECTION 6: SCORING - Guideline Scoring and Ranking
    # ============================================================================
    
    def _rerank_and_pool_guidelines(self, all_guidelines: list, previous_active: set):
        """Re-rank guidelines by score and update active/reserve pools"""
        # Debug: Show scores BEFORE sorting with IDs to detect object reference issues
        scores_before = [(g.get('data', {}).get('condition', g.get('name', 'Unknown')), id(g), g.get('score', 'MISSING')) for g in all_guidelines[:5]]
        self._capture_debug(f"[Ranking] 🔍 Scores before sorting (first 5): {scores_before}")
        
        # Re-rank
        all_guidelines.sort(key=lambda x: x.get('score', 0.5), reverse=True)
        scores_after = [(g.get('data', {}).get('condition', g.get('name', 'Unknown')), g.get('score', 'MISSING')) for g in all_guidelines[:5]]
        self._capture_debug(f"[Ranking] 🎯 Top 5 after scoring: {scores_after}")
        
        # Rule out low scores
        remaining = []
        ruled_out_count = 0
        for g in all_guidelines:
            if g['score'] >= self.RULE_OUT_THRESHOLD:
                remaining.append(g)
            else:
                self.ruled_out.append(g)
                ruled_out_count += 1
                self._capture_debug(f"[Rule Out] ❌ {g.get('data', {}).get('condition', g.get('name', 'Unknown'))}: score={g['score']:.3f} < threshold={self.RULE_OUT_THRESHOLD:.3f}")
        
        self._capture_debug(f"[Rule Out] 📉 Ruled out {ruled_out_count} guidelines, {len(remaining)} remaining")
        
        remaining.sort(key=lambda x: x['score'], reverse=True)
        self.active_guidelines = remaining[:self.MAX_ACTIVE]
        self.reserve_pool = remaining[self.MAX_ACTIVE:]
        
        # Track promotions and demotions
        current_active = set(g.get('data', {}).get('condition', g.get('name', 'Unknown')) for g in self.active_guidelines)
        promoted = [g for g in self.active_guidelines if g.get('data', {}).get('condition', g.get('name', 'Unknown')) not in previous_active]
        demoted = [g for g in self.reserve_pool if g.get('data', {}).get('condition', g.get('name', 'Unknown')) in previous_active]
        
        if promoted:
            self._capture_debug(f"\n[Engine] 🔼 PROMOTED to active:")
            for g in promoted:
                score_val = g.get('score', 0.0)
                pct = round(score_val * 100, 1)
                self._capture_debug(f"[Engine]   ↑ {g.get('data', {}).get('condition', g.get('name', 'Unknown'))} (score: {pct}%)")
        
        if demoted:
            self._capture_debug(f"\n[Engine] 🔽 DEMOTED to reserve:")
            for g in demoted:
                score_val = g.get('score', 0.0)
                pct = round(score_val * 100, 1)
                self._capture_debug(f"[Engine]   ↓ {g.get('data', {}).get('condition', g.get('name', 'Unknown'))} (score: {pct}%)")
        
        self._capture_debug(f"\n[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g.get('data', {}).get('urgency') == 'emergent' else "⚠️" if g.get('data', {}).get('urgency') == 'urgent' else "📋"
            score = g.get('score', 0.0)
            pct = round(score * 100, 1)  # Show 1 decimal place for precision
            self._capture_debug(f"[Engine]   {i}. {g.get('data', {}).get('condition', g.get('name', 'Unknown'))}: {pct}% {urgency_emoji}")
            
            # ML Progress Tracking - Top Conditions
            self._capture_debug(f"[Scoring] 🏆 Top {i}: {g.get('data', {}).get('condition', g.get('name', 'Unknown'))}")
            self._capture_debug(f"[Scoring]   📊 Score: {pct}%")
            self._capture_debug(f"[Scoring]   📋 Prevalence: {g.get('prevalence', 'unknown')}")
            self._capture_debug(f"[Scoring]   🎯 ML Confidence: High similarity match")
            self._capture_debug(f"[Scoring]   🚨 Urgency: {g.get('data', {}).get('urgency', 'standard')}")
        
        # Always show pool statistics
        self._capture_debug(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
        # ML Progress Tracking - Final Statistics
        self._capture_debug(f"[Scoring] 📊 Final statistics:")
        self._capture_debug(f"[Scoring]   🎯 Active Conditions: {len(self.active_guidelines)}")
        self._capture_debug(f"[Scoring]   📋 Reserve Conditions: {len(self.reserve_pool)}")
        self._capture_debug(f"[Scoring]   ❌ Ruled Out: {len(self.ruled_out)}")
        self._capture_debug(f"[Scoring]   📈 Total Processed: {len(all_guidelines)}")
        self._capture_debug(f"[Scoring]   🧠 ML System: Fully operational")
    
    def _analyze_missing_information(self, answer: str, oldcarts_element: str) -> tuple:
        """Analyze what information is missing using unified function with FAISS semantic matching
        Returns: (missing_medical_terms, satisfied_medical_terms)
        
        For location element:
        - missing_medical_terms: Array of medical terms (deduplicated, filtered anatomically) for clarifying questions
        - satisfied_medical_terms: Array of medical terms (deduplicated, filtered anatomically) that matched
        
        For non-location elements:
        - missing_medical_terms: Array of patient_friendly terms that weren't satisfied
        - satisfied_medical_terms: Empty list (use satisfied_terms set for decision making)
        
        For location element, follows specific flow:
        1. Apply anatomical mismatch using anatomical opposites from medical rules (filter guidelines)
        2. Raw semantic match to patient_friendly terms (FAISS)
        3. Build satisfied medical terms array, deduplicate, filter anatomically
        4. Extract missing medical terms, remove duplicates, filter with anatomical mismatch
        5. Generate missing terms array for LLM clarifying question
        """
        # Get all guidelines (active + reserve) to check against all possible terms
        all_guidelines_to_check = self.active_guidelines + self.reserve_pool
        
        if not all_guidelines_to_check:
            return [], set()
        
        # STEP 1: For location, apply anatomical mismatch FIRST to filter guidelines
        if oldcarts_element == 'location' and self.medical_rule_engine:
            # Extract patient components for anatomical filtering
            answer_lower = answer.lower()
            patient_components = self.medical_rule_engine._extract_anatomical_components(answer_lower)
            
            if patient_components:
                # Filter guidelines by anatomical compatibility
                organ_system = self.CATEGORY_TO_SYSTEM.get(self.current_category or 'gastrointestinal', 'GI')
                filtered_guidelines = self.medical_rule_engine.filter_guidelines_by_location(
                    answer, all_guidelines_to_check, organ_system
                )
                self._capture_debug(f"[Location Analysis] 🎯 STEP 1: Anatomical filtering: {len(all_guidelines_to_check)} → {len(filtered_guidelines)} guidelines")
                all_guidelines_to_check = filtered_guidelines
        
        # STEP 2: Collect patient_friendly terms from filtered guidelines (for FAISS matching)
        # Also collect medical terms for later extraction (steps 4-5)
        all_includes_patient_friendly = set()  # Patient-friendly terms for FAISS matching
        all_medical_terms = []  # Medical terms array (will deduplicate in step 4)
        term_to_guidelines = {}  # Map patient_friendly term -> list of guideline names
        medical_to_patient_friendly = {}  # Map medical term -> patient_friendly term
        
        for g in all_guidelines_to_check:
            condition_name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for t in includes:
                    if isinstance(t, dict):
                        med = t.get('medical', '')
                        pf = t.get('patient_friendly', '')
                        if isinstance(pf, str) and pf.strip():
                            pf_key = pf.strip().lower()
                            all_includes_patient_friendly.add(pf.strip())  # Keep original case
                            if pf_key not in term_to_guidelines:
                                term_to_guidelines[pf_key] = []
                            term_to_guidelines[pf_key].append(condition_name)
                            if isinstance(med, str) and med.strip():
                                all_medical_terms.append(med.strip())  # Keep original case
                                medical_to_patient_friendly[med.strip().lower()] = pf.strip()
                    elif isinstance(t, str):
                        # Plain string - treat as both medical and patient_friendly
                        term_key = t.strip().lower()
                        all_includes_patient_friendly.add(t.strip())
                        all_medical_terms.append(t.strip())
                        medical_to_patient_friendly[term_key] = t.strip()
                        if term_key not in term_to_guidelines:
                            term_to_guidelines[term_key] = []
                        term_to_guidelines[term_key].append(condition_name)
        
        # Use patient_friendly terms for FAISS matching
        all_includes = all_includes_patient_friendly
        
        # Also collect active-only patient_friendly terms for debug (to distinguish active vs reserve)
        active_includes = set()
        for g in self.active_guidelines:
            structured = self._get_structured_oldcarts(g)
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for t in includes:
                    if isinstance(t, dict):
                        pf = t.get('patient_friendly', '')
                        if isinstance(pf, str) and pf.strip():
                            active_includes.add(pf.strip())
                    elif isinstance(t, str):
                        active_includes.add(t.strip())
        
        self._capture_debug(f"[Location Analysis] 📍 Checking satisfaction against {len(all_guidelines_to_check)} guidelines (after anatomical filtering)")
        self._capture_debug(f"[Location Analysis] 📍 Patient_friendly terms from {len(all_guidelines_to_check)} guidelines: {sorted(all_includes)}")
        self._capture_debug(f"[Location Analysis] 📍 Medical terms from {len(all_guidelines_to_check)} guidelines: {sorted(set(t.lower() for t in all_medical_terms))}")
        self._capture_debug(f"[Location Analysis] 📝 Patient answer: '{answer}'")
        
        if not all_includes:
            self._capture_debug(f"[Location Analysis] ⚠️ No includes terms found for {oldcarts_element}")
            return []
        
        # STEP 2: Raw semantic match to patient_friendly terms (FAISS)
        satisfied_terms = set()
        answer_lower = answer.lower()
        semantic_matches_set = set()
        faiss_scores = {}
        
        # Get ALL condition names from filtered guidelines
        all_condition_names = set()
        for g in all_guidelines_to_check:
            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
            if condition_name:
                all_condition_names.add(condition_name)
        
        # Use FAISS index to find patient_friendly term matches
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
            try:
                # FAISS call returns patient_friendly terms (indexed terms are patient_friendly)
                semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(
                    answer, oldcarts_element, threshold=self.FAISS_SEMANTIC_THRESHOLD, 
                    return_scores=True, active_condition_names=all_condition_names
                )
                semantic_matches_set = set(t.lower() for t in semantic_matches)
                
                # Use FAISS scores directly (FAISS already computed semantic similarity on patient_friendly terms)
                if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
                    raw_faiss_scores = self.medical_rule_engine._last_faiss_scores
                    # Filter to only include terms that are in guidelines' includes (patient_friendly terms)
                    faiss_scores = {term: score for term, score in raw_faiss_scores.items() 
                                   if term.lower() in [t.lower() for t in all_includes]}
                    self._capture_debug(f"[Location Analysis] 🔍 STEP 2: FAISS found {len(semantic_matches)} patient_friendly matches above threshold: {semantic_matches}")
                    self._capture_debug(f"[Location Analysis] 🔍 FAISS similarity scores (patient_friendly terms): {faiss_scores}")
            except Exception as e:
                self._capture_debug(f"[Location Analysis] ⚠️ FAISS error: {e}")
                pass
        
        # OPTIMIZATION: Extract anatomical components ONCE from patient answer (for location only)
        # IMPORTANT: Use original answer, not normalized_answer, to avoid adding components that weren't mentioned
        if oldcarts_element == 'location' and self.medical_rule_engine:
            patient_components = self.medical_rule_engine._extract_anatomical_components(answer_lower)
            
            # Filter out anatomically opposite terms based on patient's answer
            # Use medical_rule_engine._are_anatomical_opposites to check ALL mismatch rules (not just horizontal)
            if patient_components:
                original_count = len(all_includes)
                filtered_includes = set()
                
                # Pre-extract components for all terms to check anatomical compatibility
                for term in all_includes:
                    term_guidelines = term_to_guidelines.get(term, [])
                    term_components = {}
                    
                    # Try to get components from guideline anatomical_type first
                    if term_guidelines:
                        for guideline_name in term_guidelines:
                            for g in all_guidelines_to_check:
                                condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                                if condition_name == guideline_name:
                                    anatomical_type = self.medical_rule_engine._get_anatomical_type_from_guideline(g)
                                    if anatomical_type:
                                        term_components = self.medical_rule_engine._map_anatomical_type_to_components(anatomical_type)
                                        break
                            if term_components:
                                break
                    
                    # Fallback to keyword extraction if no guideline found
                    if not term_components:
                        term_components = self.medical_rule_engine._extract_anatomical_components(term)
                    
                    # Check if term is anatomically opposite using medical rules
                    if term_components:
                        is_opposite = self.medical_rule_engine._are_anatomical_opposites(patient_components, term_components)
                        if is_opposite:
                            self._capture_debug(f"[Location Analysis] ❌ Filtered out anatomically opposite term '{term}' (patient: {patient_components}, term: {term_components})")
                            continue
                    
                    # Term is compatible (not anatomically opposite or no components specified)
                    filtered_includes.add(term)
                
                # Update all_includes to only include filtered terms
                all_includes = filtered_includes
                filtered_count = original_count - len(all_includes)
                self._capture_debug(f"[Location Analysis] 🔍 Filtered {filtered_count} anatomically opposite terms, {len(all_includes)} remaining")
        
        # OPTIMIZATION: Pre-extract anatomical components for ALL terms ONCE (not in loop)
        # PRIMARY METHOD: Use guideline anatomical_type when term is associated with a guideline
        # FALLBACK: Use keyword-based extraction only if term has no associated guideline
        term_components_cache = {}
        if oldcarts_element == 'location' and self.medical_rule_engine:
            for term in all_includes:
                components = {}
                term_guidelines = term_to_guidelines.get(term, [])
                
                if term_guidelines:
                    # PRIMARY: Use guideline's anatomical_type (when available)
                    for guideline_name in term_guidelines:
                        for g in all_guidelines_to_check:
                            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                            if condition_name == guideline_name:
                                anatomical_type = self.medical_rule_engine._get_anatomical_type_from_guideline(g)
                                if anatomical_type:
                                    # Map anatomical_type to components using medical_rules.json
                                    components = self.medical_rule_engine._map_anatomical_type_to_components(anatomical_type)
                                    self._capture_debug(f"[Location Analysis] 📍 '{term}' components from guideline anatomical_type '{anatomical_type}': {components}")
                                    break
                        if components:
                            break
                else:
                    # FALLBACK: Use keyword-based extraction only if term has no associated guideline
                    components = self.medical_rule_engine._extract_anatomical_components(term)
                    if components:
                        self._capture_debug(f"[Location Analysis] 📍 '{term}' components from keyword extraction: {components}")
                
                term_components_cache[term] = components
        
        # STEP 3: Use FAISS scores to determine satisfied terms
        # Build satisfied patient_friendly terms array (before anatomical filtering)
        satisfied_pf_terms_raw = []
        for term in all_includes:
            if term.lower() in semantic_matches_set:
                satisfied_pf_terms_raw.append(term)
        
        self._capture_debug(f"[Location Analysis] 📊 STEP 3: Found {len(satisfied_pf_terms_raw)} satisfied patient_friendly terms (from FAISS matches above threshold): {satisfied_pf_terms_raw}")
        
        # For location: Build satisfied medical terms array, deduplicate, filter anatomically
        # For non-location: satisfied_medical_terms will remain empty (we use satisfied_terms set instead)
        satisfied_medical_terms = []
        if oldcarts_element == 'location' and self.medical_rule_engine:
            # Map satisfied patient_friendly terms to medical terms
            for pf_term in satisfied_pf_terms_raw:
                for g in all_guidelines_to_check:
                    structured = self._get_structured_oldcarts(g)
                    element_data = structured.get(oldcarts_element, {})
                    if isinstance(element_data, dict):
                        includes = element_data.get('includes', [])
                        for t in includes:
                            if isinstance(t, dict):
                                med = t.get('medical', '')
                                pf = t.get('patient_friendly', '')
                                if pf and pf.strip().lower() == pf_term.lower() and med:
                                    satisfied_medical_terms.append(med.strip())
                                    break
                        if med:
                            break
            
            self._capture_debug(f"[Location Analysis] 🔍 STEP 3: Extracted {len(satisfied_medical_terms)} medical terms from satisfied patient_friendly terms")
            
            # Remove duplicates
            seen_medical = set()
            unique_satisfied_medical = []
            for med_term in satisfied_medical_terms:
                med_lower = med_term.lower()
                if med_lower not in seen_medical:
                    seen_medical.add(med_lower)
                    unique_satisfied_medical.append(med_term)
            
            self._capture_debug(f"[Location Analysis] 🔍 STEP 3: After deduplication: {len(unique_satisfied_medical)} unique satisfied medical terms: {unique_satisfied_medical}")
            
            # Filter anatomically (remove opposite side terms)
            if patient_components:
                filtered_satisfied_medical = []
                for med_term in unique_satisfied_medical:
                    # Extract components for this medical term
                    med_components = self.medical_rule_engine._extract_anatomical_components(med_term.lower())
                    if not med_components:
                        # Try to get from guideline anatomical_type
                        for g in all_guidelines_to_check:
                            structured = self._get_structured_oldcarts(g)
                            element_data = structured.get(oldcarts_element, {})
                            if isinstance(element_data, dict):
                                includes = element_data.get('includes', [])
                                for t in includes:
                                    if isinstance(t, dict) and t.get('medical', '').strip().lower() == med_term.lower():
                                        anatomical_type = self.medical_rule_engine._get_anatomical_type_from_guideline(g)
                                        if anatomical_type:
                                            med_components = self.medical_rule_engine._map_anatomical_type_to_components(anatomical_type)
                                            break
                    
                    if med_components:
                        is_opposite = self.medical_rule_engine._are_anatomical_opposites(patient_components, med_components)
                        if not is_opposite:
                            filtered_satisfied_medical.append(med_term)
                            self._capture_debug(f"[Location Analysis] ✅ STEP 3: '{med_term}' included in satisfied (anatomically compatible)")
                        else:
                            self._capture_debug(f"[Location Analysis] ❌ STEP 3: '{med_term}' excluded from satisfied (anatomical opposite)")
                    else:
                        # No components found - include it (might be vague/bilateral)
                        filtered_satisfied_medical.append(med_term)
                        self._capture_debug(f"[Location Analysis] ✅ STEP 3: '{med_term}' included in satisfied (no components, assumed compatible)")
                
                unique_satisfied_medical = filtered_satisfied_medical
                self._capture_debug(f"[Location Analysis] 🔍 STEP 3: After anatomical filtering: {len(unique_satisfied_medical)} satisfied medical terms: {unique_satisfied_medical}")
            
            # Store satisfied medical terms for later use
            satisfied_medical_terms = unique_satisfied_medical
            
            # Build satisfied_terms set from filtered satisfied medical terms (for backward compatibility)
            # Map back to patient_friendly for satisfied_terms set
            for med_term in satisfied_medical_terms:
                for g in all_guidelines_to_check:
                    structured = self._get_structured_oldcarts(g)
                    element_data = structured.get(oldcarts_element, {})
                    if isinstance(element_data, dict):
                        includes = element_data.get('includes', [])
                        for t in includes:
                            if isinstance(t, dict):
                                med = t.get('medical', '')
                                pf = t.get('patient_friendly', '')
                                if med and med.strip().lower() == med_term.lower() and pf:
                                    satisfied_terms.add(pf.strip().lower())
                                    break
                        if pf:
                            break
        else:
            # For non-location, just use patient_friendly terms directly
            for term in satisfied_pf_terms_raw:
                satisfied_terms.add(term.lower())
        
        self._capture_debug(f"[Location Analysis] 📊 STEP 3: Final satisfied patient_friendly terms: {sorted(satisfied_terms)}")
        
        # Missing terms: Include ALL unsatisfied terms (from active + reserve) that are anatomically compatible
        # Anatomical compatibility takes precedence over active/reserve distinction
        # This ensures we ask about all relevant locations, not just those from top-scoring guidelines
        all_missing_candidates = [term for term in all_includes if term not in satisfied_terms]
        
        # DEBUG: Show which unsatisfied terms are from reserve pool (will be included if anatomically compatible)
        unsatisfied_from_reserve = [term for term in all_missing_candidates if term not in active_includes]
        if unsatisfied_from_reserve:
            reserve_sources = {}
            for term in unsatisfied_from_reserve:
                sources = term_to_guidelines.get(term, [])
                if sources:
                    reserve_sources[term] = sources
            if reserve_sources:
                self._capture_debug(f"[Location Analysis] 📋 Unsatisfied terms from RESERVE pool (will be included if anatomically compatible): {reserve_sources}")
        
        # STEP 4 & 5: After analysis, decide which array to build
        # If 1+ elements were satisfied → use satisfied array (skip missing)
        # If 0 elements satisfied → build missing array
        missing = []
        
        # Check if any elements were satisfied (for location: check satisfied_medical_terms, for others: check satisfied_terms)
        has_satisfied_elements = False
        if oldcarts_element == 'location' and self.medical_rule_engine:
            # For location, check satisfied_medical_terms array (already built in STEP 3)
            has_satisfied_elements = len(satisfied_medical_terms) > 0
        else:
            # For non-location, check satisfied_terms set
            has_satisfied_elements = len(satisfied_terms) > 0
        
        if has_satisfied_elements:
            # 1+ elements satisfied → use satisfied array, skip missing array
            if oldcarts_element == 'location' and self.medical_rule_engine:
                self._capture_debug(f"[Location Analysis] ✅ STEP 4-5: Using satisfied array ({len(satisfied_medical_terms)} elements), skipping missing array")
            else:
                self._capture_debug(f"[Location Analysis] ✅ STEP 4-5: Using satisfied terms ({len(satisfied_terms)} elements), skipping missing array")
            missing = []  # Return empty missing array
        else:
            # 0 elements satisfied → build missing array
            self._capture_debug(f"[Location Analysis] 🔍 STEP 4-5: No satisfied elements, building missing array")
            
            if oldcarts_element == 'location' and self.medical_rule_engine:
                
                # Extract patient components if not already done
                if not patient_components:
                    patient_components = self.medical_rule_engine._extract_anatomical_components(answer_lower)
                
                # Get unsatisfied patient_friendly terms (all terms that weren't satisfied)
                unsatisfied_pf_terms = [t for t in all_includes if t.lower() not in satisfied_terms]
                
                self._capture_debug(f"[Location Analysis] 🔍 STEP 4: Found {len(unsatisfied_pf_terms)} unsatisfied patient_friendly terms")
                
                # Map ALL unsatisfied patient_friendly terms back to medical terms
                # Collect from ALL guidelines (not just first match) to ensure we get all medical terms
                unsatisfied_medical_terms = []
                seen_pf_med_pairs = set()  # Track (pf_term, medical_term) to avoid duplicates
                
                for pf_term in unsatisfied_pf_terms:
                    # Find medical term(s) corresponding to this patient_friendly term from ALL guidelines
                    for g in all_guidelines_to_check:
                        structured = self._get_structured_oldcarts(g)
                        element_data = structured.get(oldcarts_element, {})
                        if isinstance(element_data, dict):
                            includes = element_data.get('includes', [])
                            for t in includes:
                                if isinstance(t, dict):
                                    med = t.get('medical', '')
                                    pf = t.get('patient_friendly', '')
                                    if pf and pf.strip().lower() == pf_term.lower() and med:
                                        med_stripped = med.strip()
                                        pair_key = (pf_term.lower(), med_stripped.lower())
                                        if pair_key not in seen_pf_med_pairs:
                                            unsatisfied_medical_terms.append(med_stripped)
                                            seen_pf_med_pairs.add(pair_key)
                                elif isinstance(t, str):
                                    # Plain string - treat as both medical and patient_friendly
                                    if t.strip().lower() == pf_term.lower():
                                        pair_key = (pf_term.lower(), t.strip().lower())
                                        if pair_key not in seen_pf_med_pairs:
                                            unsatisfied_medical_terms.append(t.strip())
                                            seen_pf_med_pairs.add(pair_key)
                
                self._capture_debug(f"[Location Analysis] 🔍 STEP 4: Extracted {len(unsatisfied_medical_terms)} medical terms from unsatisfied patient_friendly terms")
                
                # Remove duplicates (keep first occurrence, preserve original case)
                seen_medical = set()
                unique_medical_terms = []
                for med_term in unsatisfied_medical_terms:
                    med_lower = med_term.lower()
                    if med_lower not in seen_medical:
                        seen_medical.add(med_lower)
                        unique_medical_terms.append(med_term)
                
                self._capture_debug(f"[Location Analysis] 🔍 STEP 4: After deduplication: {len(unique_medical_terms)} unique medical terms: {unique_medical_terms}")
                
                # Filter with anatomical mismatch to remove opposite side terms
                if patient_components:
                    filtered_medical_terms = []
                    for med_term in unique_medical_terms:
                        # Extract components for this medical term
                        med_components = self.medical_rule_engine._extract_anatomical_components(med_term.lower())
                        if not med_components:
                            # Try to get from guideline anatomical_type
                            for g in all_guidelines_to_check:
                                structured = self._get_structured_oldcarts(g)
                                element_data = structured.get(oldcarts_element, {})
                                if isinstance(element_data, dict):
                                    includes = element_data.get('includes', [])
                                    for t in includes:
                                        if isinstance(t, dict) and t.get('medical', '').strip().lower() == med_term.lower():
                                            anatomical_type = self.medical_rule_engine._get_anatomical_type_from_guideline(g)
                                            if anatomical_type:
                                                med_components = self.medical_rule_engine._map_anatomical_type_to_components(anatomical_type)
                                                break
                        
                        if med_components:
                            is_opposite = self.medical_rule_engine._are_anatomical_opposites(patient_components, med_components)
                            if not is_opposite:
                                filtered_medical_terms.append(med_term)
                                self._capture_debug(f"[Location Analysis] ✅ STEP 4: '{med_term}' included (anatomically compatible)")
                            else:
                                self._capture_debug(f"[Location Analysis] ❌ STEP 4: '{med_term}' excluded (anatomical opposite)")
                        else:
                            # No components found - include it (might be vague/bilateral)
                            filtered_medical_terms.append(med_term)
                            self._capture_debug(f"[Location Analysis] ✅ STEP 4: '{med_term}' included (no components, assumed compatible)")
                    
                    unique_medical_terms = filtered_medical_terms
                    self._capture_debug(f"[Location Analysis] 🔍 STEP 4: After anatomical filtering: {len(unique_medical_terms)} medical terms: {unique_medical_terms}")
                
                # STEP 5: Generate missing terms array for LLM clarifying question
                missing = unique_medical_terms
                self._capture_debug(f"[Location Analysis] 📋 STEP 5: Missing terms array for LLM: {missing}")
            else:
                # For non-location elements, use simpler logic
                all_missing_candidates = [t for t in all_includes if t.lower() not in satisfied_terms]
                missing = all_missing_candidates
        
        self._capture_debug(f"[Location Analysis] ✅ Satisfied terms (checked against ALL {len(all_guidelines_to_check)} guidelines): {sorted(satisfied_terms)}")
        missing_count = len(missing) if missing else 0
        self._capture_debug(f"[Location Analysis] ❌ Missing terms from ALL guidelines: {missing_count} total: {sorted(missing) if missing else []}")
        
        # Return: (missing_medical_terms, satisfied_medical_terms)
        # For location: satisfied_medical_terms is already deduplicated and filtered anatomically
        # For non-location: satisfied_medical_terms is empty list (use satisfied_terms set instead)
        if oldcarts_element == 'location' and self.medical_rule_engine:
            # Return satisfied medical terms array (already processed)
            satisfied_return = satisfied_medical_terms if 'satisfied_medical_terms' in locals() else []
        else:
            # For non-location, return empty list (satisfied_terms set is used for decision making)
            satisfied_return = []
        
        return missing, satisfied_return
    
    def _get_satisfied_terms(self, answer: str, oldcarts_element: str) -> set:
        """Get terms that are satisfied using unified function logic"""
        if not self.active_guidelines:
            return set()
        
        # Collect all includes terms from active guidelines (normalize dicts)
        all_includes = set()
        for g in self.active_guidelines:
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for t in includes:
                    if isinstance(t, dict):
                        med = t.get('medical')
                        if isinstance(med, str) and med.strip():
                            all_includes.add(med.strip().lower())
                    elif isinstance(t, str):
                        all_includes.add(t.strip().lower())
        
        if not all_includes:
            return set()
        
        # Use semantic matching only (no normalization, no substring matching)
        satisfied_terms = set()
        
        # Use FAISS semantic matching only
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
            # Get active condition names for filtering
            active_condition_names = None
            if hasattr(self, 'active_guidelines') and self.active_guidelines:
                active_condition_names = self._get_active_condition_names()
            
            semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(
                answer, oldcarts_element, threshold=self.FAISS_SEMANTIC_THRESHOLD, active_condition_names=active_condition_names
            )
            semantic_matches_lower = set(t.lower() for t in semantic_matches)
            
            # Check each term using semantic matching only
            for term in all_includes:
                term_lower = term.lower()
                if term_lower in semantic_matches_lower:
                    satisfied_terms.add(term)
        
        return satisfied_terms

    def _collect_patient_friendly_options(self, oldcarts_element: str, limit: int = 3) -> list:
        """Collect unique patient-friendly options for an OLDCARTS element from active guidelines."""
        options = []
        seen = set()
        for g in self.active_guidelines:
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for term in includes:
                    if isinstance(term, dict):
                        pf = term.get('patient_friendly') or term.get('medical')
                        if isinstance(pf, str):
                            key = pf.strip().lower()
                            if key and key not in seen:
                                options.append(pf.strip())
                                seen.add(key)
                    elif isinstance(term, str):
                        key = term.strip().lower()
                        if key and key not in seen:
                            options.append(term.strip())
                            seen.add(key)
            if len(options) >= limit:
                break
        return options[:limit]
    
    def _build_conversation_context(self, recent_items: int = 3, char_limit: int = 100, include_answered: bool = False) -> str:
        """Build conversation context string for LLM prompts"""
        conversation_context = ""
        if hasattr(self, 'conversation_history') and self.conversation_history:
            recent_conversation = []
            for item in self.conversation_history[-recent_items:]:
                if item.get('type') == 'answer':
                    recent_conversation.append(f"Patient: {item.get('answer', '')[:char_limit]}")
                elif item.get('type') == 'question':
                    recent_conversation.append(f"Asked: {item.get('question', '')[:char_limit]}")
            
            if include_answered:
                answered_elements = []
                for item in self.conversation_history:
                    if item.get('type') == 'answer' and item.get('oldcarts'):
                        answered_elements.append(item.get('oldcarts'))
                if answered_elements:
                    conversation_context = f"\n\nAlready answered: {', '.join(set(answered_elements))}"
            
            if recent_conversation:
                if conversation_context:
                    conversation_context += f"\n\nRecent conversation:\n" + "\n".join(recent_conversation[-2:])
                else:
                    conversation_context = f"\n\nRecent conversation:\n" + "\n".join(recent_conversation)
        
        return conversation_context
    
    def _get_chief_complaint_context(self) -> str:
        """Get chief complaint context string for LLM prompts"""
        return f"Chief complaint: {self.chief_complaint}" if self.chief_complaint else "No chief complaint recorded"
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response (remove double question marks, strip whitespace, remove reasoning)"""
        if response and response.strip():
            cleaned = response.strip()
            
            # Remove double question marks
            if cleaned.endswith('??'):
                cleaned = cleaned[:-1]
            
            # Remove common reasoning patterns that might appear anywhere before the question
            # Look for patterns like "Here's a question:" or "The question is:" etc.
            reasoning_patterns = [
                r'here\'s.*?:',
                r'here is.*?:',
                r'the question.*?:',
                r'a clarification question.*?:',
                r'clarification question.*?:',
                r'this question.*?:',
                r'i would ask.*?:',
                r'you could ask.*?:',
                r'alternatively.*?:',
                r'this uses.*?:',
                r'it also.*?:',
                r'which are.*?:',
            ]
            
            # Remove reasoning patterns (not just at start)
            for pattern in reasoning_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Extract ONLY the question part - find the first question mark and extract from there
            if '?' in cleaned:
                # Find the first question mark
                first_q_index = cleaned.find('?')
                # Find the sentence start (look for sentence boundaries before the question)
                sentence_start = 0
                for i in range(first_q_index - 1, -1, -1):
                    if cleaned[i] in ['.', '!', '\n']:
                        sentence_start = i + 1
                        break
                    # Also check for common reasoning phrase endings
                    if i > 10 and cleaned[max(0, i-20):i+1].lower() in ['for example:', 'such as:', 'like:']:
                        sentence_start = i + 1
                        break
                
                # Extract from sentence start to first question mark
                cleaned = cleaned[sentence_start:first_q_index + 1].strip()
                
                # Remove any trailing explanations after the question mark
                if len(cleaned) > first_q_index - sentence_start + 1:
                    # There's more text after the question - remove it
                    q_mark_pos = cleaned.find('?')
                    if q_mark_pos >= 0:
                        cleaned = cleaned[:q_mark_pos + 1]
            else:
                # No question mark found - return empty (will trigger error)
                return ""
            
            return cleaned.strip()
        return ""
    
    
    # ============================================================================
    # SECTION 7: QUESTION GENERATION - Asking Next Questions
    # ============================================================================
    
    def _generate_clarifying_question(self, oldcarts_element: str, patient_answer: str,
                                     clarification_count: int, terms: list, satisfied_context: bool = False) -> str:
        """Generate clarifying question using LLM with patient-friendly terms from guidelines
        
        Args:
            oldcarts_element: The OLDCARTS element being clarified
            patient_answer: The patient's previous answer
            clarification_count: Number of times we've asked for clarification
            terms: List of medical terms (either missing_terms or satisfied_terms)
            satisfied_context: If True, terms are satisfied terms that need disambiguation. If False, terms are missing terms.
        """
        if not terms:
            context_type = "satisfied" if satisfied_context else "missing"
            raise ValueError(f"Cannot generate clarifying question for {oldcarts_element} - no {context_type} terms")
        
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for clarification question generation")
        
        context_label = "satisfied" if satisfied_context else "missing"
        self._capture_debug(f"[Clarification] 🔍 {context_label.capitalize()} medical terms ({oldcarts_element}): {terms[:8]}")
        
        # Get patient-friendly terms directly from guidelines (deduplicate to avoid same friendly term from different medical terms)
        patient_friendly_terms = []
        seen_friendly_terms = set()  # Track unique patient-friendly terms to avoid duplicates
        medical_to_friendly_map = {}
        
        # Process ALL terms (no limit)
        # Skip terms that can't be found (e.g., from reserve guidelines without patient_friendly terms)
        for term in terms:
            try:
                friendly_term = self._get_patient_friendly_from_guidelines(term, oldcarts_element)
                medical_to_friendly_map[term] = friendly_term
                
                # Debug output showing the mapping for each term as we process it
                self._capture_debug(f"[Clarification] 📝 '{term}' → '{friendly_term}'")
                
                # Only add non-empty, unique terms
                if friendly_term and friendly_term.strip():
                    friendly_lower = friendly_term.strip().lower()
                    if friendly_lower not in seen_friendly_terms:
                        patient_friendly_terms.append(friendly_term.strip())
                        seen_friendly_terms.add(friendly_lower)
            except ValueError as e:
                # Skip terms that can't be found - log but continue processing other terms
                self._capture_debug(f"[Clarification] ⚠️ Skipping '{term}': {e}")
                continue
        
        # If no good terms found, raise error
        if not patient_friendly_terms:
            raise ValueError(f"Cannot generate clarifying question for {oldcarts_element} - no patient-friendly terms found")
        
        # Use LLM to generate a natural, properly structured clarification question
        system_msg = "You are a medical assistant conducting a telehealth interview. Generate a natural, grammatically correct clarification question that flows well. Use proper grammar with 'and' and 'or' to connect options naturally. Return ONLY the question - no explanations, no reasoning, no additional text."
        
        # Get context using helper functions
        chief_complaint_context = self._get_chief_complaint_context()
        conversation_context = self._build_conversation_context(recent_items=3, char_limit=100, include_answered=False)
        
        # Build instructions based on element type
        # Take first 5 terms for the question (to keep it manageable)
        terms_to_use = patient_friendly_terms[:5]
        options_list = ", ".join(terms_to_use)
        
        if oldcarts_element == 'location':
            # Example format to make it crystal clear
            example_question = f"Can you be more specific? For example, is it located at {terms_to_use[0]}, {terms_to_use[1]}, {terms_to_use[2]}, or {terms_to_use[3] if len(terms_to_use) > 3 else terms_to_use[0]}?"
            
            if satisfied_context:
                # Context: Patient said something that matches multiple locations, need to disambiguate
                clarification_text = f"""The patient already said: "{patient_answer}"

This matches multiple possible locations. We need to clarify WHICH ONE is correct. Here are the locations that matched (you MUST use ONLY these exact terms):
{options_list}

IMPORTANT: The patient's answer matched these locations, so ask them to choose which one is correct."""
            else:
                # Context: Missing information, need to ask about locations not mentioned
                clarification_text = f"""The patient already said: "{patient_answer}"

We need to clarify the location. Here are the ONLY possible locations from the medical guidelines (you MUST use ONLY these exact terms):
{options_list}"""
            
            user_msg = f"""{chief_complaint_context}{conversation_context}

{clarification_text}

{self.LLM_CLARIFICATION_LOCATION_RULES}

EXAMPLE of correct format: "{example_question}"

Generate a clarification question using EXACTLY this format with the terms provided above. The question MUST include at least 3-4 specific options from the list."""
        else:
            # Example format for non-location elements
            example_question = f"Can you be more specific? For example, is it {terms_to_use[0]}, {terms_to_use[1]}, {terms_to_use[2]}, or {terms_to_use[3] if len(terms_to_use) > 3 else terms_to_use[0]}?"
            
            user_msg = f"""{chief_complaint_context}{conversation_context}

The patient already said: "{patient_answer}"

We need to clarify the {oldcarts_element}. Here are the ONLY possible options from the medical guidelines (you MUST use ONLY these exact terms):
{options_list}

{self.LLM_CLARIFICATION_GENERAL_RULES}

EXAMPLE of correct format: "{example_question}"

Generate a clarification question using EXACTLY this format with the terms provided above. The question MUST include at least 3-4 specific options from the list."""
        
        llm_kwargs = self._get_llm_kwargs(override_max_tokens=100)
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        
        cleaned_response = self._clean_llm_response(response)
        if not cleaned_response:
            raise ValueError(f"LLM returned empty response for clarification question ({oldcarts_element})")
        
        # Validate that the response includes at least one of the patient-friendly terms
        # If it doesn't, it's likely a generic fallback like "Can you be more specific?" without terms
        response_lower = cleaned_response.lower()
        terms_to_use = patient_friendly_terms[:5]  # Use same terms we sent to LLM
        has_term = any(term.lower() in response_lower for term in terms_to_use)
        
        if not has_term:
            # Check if it's just a generic question without terms
            generic_patterns = [
                "can you be more specific",
                "can you tell me more",
                "where exactly",
                "what do you mean"
            ]
            is_generic = any(pattern in response_lower for pattern in generic_patterns)
            if is_generic:
                raise ValueError(f"LLM generated generic question without missing terms: '{cleaned_response}'. Expected question with terms from: {terms_to_use}")
        
        return cleaned_response
    
    def _get_patient_friendly_from_guidelines(self, medical_term: str, oldcarts_element: str) -> str:
        """Get patient-friendly term directly from guidelines (case-insensitive match)
        Checks ALL guidelines (active + reserve) since missing terms can come from reserve pool
        """
        medical_term_lower = medical_term.lower().strip()
        # Check ALL guidelines (active + reserve) since missing terms can come from reserve pool
        all_guidelines_to_check = self.active_guidelines + self.reserve_pool
        for guideline in all_guidelines_to_check:
            # Try both possible structures (with and without 'data' wrapper)
            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
            if not structured:
                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for term_obj in includes:
                    if isinstance(term_obj, dict):
                        med = term_obj.get('medical', '')
                        if isinstance(med, str) and med.lower().strip() == medical_term_lower:
                            return term_obj.get('patient_friendly', medical_term)
                    elif isinstance(term_obj, str) and term_obj.lower().strip() == medical_term_lower:
                        # Handle old format where terms are just strings
                        return medical_term
        
        # No fallback - raise error if term not found
        raise ValueError(f"Medical term '{medical_term}' not found in synonym mappings for {oldcarts_element}")
    
    def _ask_about_radiation(self) -> Dict[str, Any]:
        """Ask about radiation as a separate question after location is satisfied"""
        self.radiation_asked = True
        
        # Collect all radiation terms with patient-friendly versions from radiation section
        radiation_options = []
        for guideline in self.active_guidelines:
            structured = self._get_structured_oldcarts(guideline)
            radiation_data = structured.get('radiation', {})
            if isinstance(radiation_data, dict):
                includes = radiation_data.get('includes', [])
                for term_obj in includes:
                    medical_term = None
                    patient_friendly = None
                    if isinstance(term_obj, dict):
                        medical_term = term_obj.get('medical', '')
                        patient_friendly = term_obj.get('patient_friendly', medical_term)
                    elif isinstance(term_obj, str):
                        medical_term = term_obj
                        patient_friendly = term_obj
                    
                    if medical_term and patient_friendly:
                        if patient_friendly not in radiation_options:
                            radiation_options.append(patient_friendly)
        
        if not radiation_options:
            # No radiation terms found, continue normally
            return self._ask_next_clinical_question()
        
        # Generate question with radiation options
        if len(radiation_options) == 1:
            question = f"Does the pain spread or radiate anywhere? For example, {radiation_options[0]}?"
        else:
            options = ", ".join(radiation_options[:3])
            question = f"Does the pain spread or radiate anywhere? For example, {options}?"
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'oldcarts': 'radiation',
            'focus': 'clinical'
        })
        
        self._capture_debug(f"[Engine] 📍 Asking about radiation: {question}")
        
        return {
            'success': True,
            'question': question,
            'status': 'questioning',
            'debug': {
                'engine': self._format_engine_debug("[Engine] 📍 Radiation question generated") + "\n\n" + self._format_rankings_debug(),
                'internal': self._get_debug_info()
            }
        }
    
    def _ask_next_clinical_question(self) -> Dict[str, Any]:
        """Ask next OLDCARTS question - standard order"""
        self._capture_debug(f"[Engine] 🔍 Asking next clinical question")
        self._capture_debug(f"[Engine] Has oldcarts_analysis: {hasattr(self, 'oldcarts_analysis')}")
        if hasattr(self, 'oldcarts_analysis'):
            self._capture_debug(f"[Engine] OLDCARTS analysis: {self.oldcarts_analysis}")
        
        has_oldcarts_analysis_attr = hasattr(self, 'oldcarts_analysis')
        oldcarts_analysis_missing = not has_oldcarts_analysis_attr or not self.oldcarts_analysis
        
        if oldcarts_analysis_missing:
            return {
                'success': False,
                'message': 'No OLDCARTS analysis available',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ❌ No OLDCARTS analysis available"),
                    'internal': self._get_debug_info()
                }
            }
        
        missing = self.oldcarts_analysis.get('missing_components', [])
        self._capture_debug(f"[Engine] Missing components: {missing}")
        if not missing:
            # All OLDCARTS complete - now ask about key positives/negatives
            key_features_not_active = not self.key_features_phase
            red_flag_not_active = not self.red_flag_phase
            
            if key_features_not_active and red_flag_not_active:
                return self._start_key_features_phase()
            elif self.key_features_phase:
                # Continue asking about key features (will be handled in process_answer)
                return self._ask_next_key_feature()
            elif self.red_flag_phase:
                # Continue asking about red flags (will be handled in process_answer)
                return self._ask_next_red_flag()
        
        # OPTIMIZATION: Reorder to prioritize timing before duration
        # If timing is answered as constant, skip duration (redundant - already constant since onset)
        # EXCEPTION: Still ask duration if guidelines have comparison operators (>, <, >=, <=) to differentiate conditions
        priority_order = ['onset', 'location', 'timing', 'duration', 'progression', 'character', 'aggravating', 'relieving', 'severity', 'associated']
        reordered_missing = []
        skip_duration = False
        
        # Check if timing is already answered as constant
        if 'timing' not in missing:
            # Timing was already answered - check if it's constant
            if hasattr(self, 'conversation_history'):
                for item in reversed(self.conversation_history):
                    item_oldcarts_is_timing = item.get('oldcarts') == 'timing'
                    item_type_is_answer = item.get('type') == 'answer'
                    
                    if item_oldcarts_is_timing and item_type_is_answer:
                        answer = item.get('answer', item.get('message', '')).lower()
                        constant_words = ['constant', 'continuous', 'always', 'all the time', 'never stops']
                        has_constant_word = any(word in answer for word in constant_words)
                        
                        if has_constant_word:
                            # Check if any active guidelines have duration terms with comparison operators
                            has_duration_comparison = False
                            for guideline in self.active_guidelines:
                                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                                duration_data = structured.get('duration', {})
                                if isinstance(duration_data, dict):
                                    includes = duration_data.get('includes', [])
                                    for term in includes:
                                        medical_term = term.get('medical', '') if isinstance(term, dict) else term
                                        is_medical_term_string = isinstance(medical_term, str)
                                        comparison_operators = ['>', '<', '>=', '<=']
                                        has_comparison_operator = any(op in medical_term for op in comparison_operators) if is_medical_term_string else False
                                        
                                        if is_medical_term_string and has_comparison_operator:
                                            has_duration_comparison = True
                                            break
                                if has_duration_comparison:
                                    break
                            
                            # Only skip duration if NO comparison operators found
                            if not has_duration_comparison:
                                skip_duration = True
                                self._capture_debug(f"[Engine] ⏭️ Skipping duration - timing already answered as constant (no comparison operators)")
                            else:
                                self._capture_debug(f"[Engine] ⚠️ Duration still needed - timing is constant but duration has comparison operators for differentiation")
                        break
        
        # Build reordered list: priority order, but only include missing elements
        for element in priority_order:
            if element in missing:
                # Skip duration if timing is constant AND no comparison operators
                if element == 'duration' and skip_duration:
                    continue
                reordered_missing.append(element)
        
        # Add any remaining elements not in priority order
        for element in missing:
            if element not in reordered_missing:
                reordered_missing.append(element)
        
        # Standard OLDCARTS order (now reordered)
        next_element = reordered_missing[0] if reordered_missing else missing[0]
        self._capture_debug(f"[Engine] Next element to ask: {next_element}")
        question = self._generate_oldcarts_question_for_component(next_element)
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'oldcarts': next_element,
            'focus': 'clinical'
        })
        
        return {
            'success': True,
            'message': question,
            'status': 'questioning',
            'debug': {
                'engine': self._format_engine_debug(f"[Engine] 🔍 Next element to ask: {next_element}") + "\n\n" + self._format_rankings_debug(),
                'internal': self._get_debug_info()
            }
        }
    
    def _generate_confirmation_message(self, user_answer: str) -> str:
        """Generate a confirmation/paraphrase message to show we understand what the patient said"""
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for confirmation message generation")
        
        # Get context about chief complaint
        chief_complaint_context = f"Patient's chief complaint: {self.chief_complaint}"
        has_conversation_history_attr = hasattr(self, 'conversation_history')
        conversation_history_exists = has_conversation_history_attr and self.conversation_history
        
        if conversation_history_exists:
            # Include recent conversation context
            recent_msgs = [item.get('message', item.get('question', item.get('answer', ''))) 
                         for item in self.conversation_history[-3:] if item.get('type') in ['statement', 'question', 'answer']]
            context = " ".join(recent_msgs[-2:])  # Last 2 messages
        else:
            context = ""
        
        system_msg = "You are a medical assistant. Generate a brief confirmation message paraphrasing what the patient just told you to show you understand."
        user_msg = f"{chief_complaint_context}\n\nPatient just said: '{user_answer}'\n\nGenerate a brief confirmation message (1-2 sentences) that paraphrases what they told you to confirm understanding. Make it natural and empathetic. Return only the confirmation message, no other text."
        
        llm_kwargs = self._get_llm_kwargs()
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        response_is_empty = not response
        response_stripped_is_empty = not response.strip() if response else True
        
        if response_is_empty or response_stripped_is_empty:
            raise ValueError("LLM returned empty response for confirmation message")
        
        return response.strip()
    
    def _analyze_character_terms(self) -> dict:
        """Analyze character terms from active guidelines to determine question type"""
        if not self.active_guidelines:
            return {'has_descriptive': False, 'has_sensory': False, 'sample_question': self.LLM_CHARACTER_DEFAULT_QUESTION, 'guidance': self.LLM_CHARACTER_DEFAULT_GUIDANCE}
        
        # Collect all character terms from active guidelines
        all_character_terms = []
        for guideline in self.active_guidelines:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            character_data = structured.get('character', {})
            if isinstance(character_data, dict):
                includes = character_data.get('includes', [])
                for term_obj in includes:
                    if isinstance(term_obj, dict):
                        medical = term_obj.get('medical', '').lower()
                        patient_friendly = term_obj.get('patient_friendly', '').lower()
                        all_character_terms.extend([medical, patient_friendly])
        
        # Check for descriptive/visual terms (colors, appearances, visual characteristics)
        descriptive_keywords = ['red', 'blood', 'bright', 'dark', 'black', 'coffee', 'ground', 'tarry', 'sticky', 
                              'clots', 'tissue', 'mixed', 'separate', 'look', 'appear', 'color', 'appearance']
        
        # Check for sensory/feeling terms (pain qualities, sensations)
        sensory_keywords = ['sharp', 'dull', 'aching', 'burning', 'stabbing', 'throbbing', 'pressure', 'cramping',
                           'colicky', 'gnawing', 'squeezing', 'tight', 'feel', 'sensation', 'pain']
        
        has_descriptive = any(keyword in term for term in all_character_terms for keyword in descriptive_keywords)
        has_sensory = any(keyword in term for term in all_character_terms for keyword in sensory_keywords)
        
        # Determine question type based on what's in the guidelines
        if has_descriptive and has_sensory:
            # Both types present - ask about description/appearance
            sample_question = "Can you describe what it looks like or how it appears?"
            guidance = self.LLM_CHARACTER_DESCRIPTIVE_AND_SENSORY_GUIDANCE
        elif has_descriptive:
            # Only descriptive terms - ask about appearance/description
            sample_question = "Can you describe what it looks like?"
            guidance = self.LLM_CHARACTER_DESCRIPTIVE_ONLY_GUIDANCE
        else:
            # Only sensory terms or default - ask about feeling
            sample_question = "What does it feel like?"
            guidance = self.LLM_CHARACTER_DEFAULT_GUIDANCE
        
        return {
            'has_descriptive': has_descriptive,
            'has_sensory': has_sensory,
            'sample_question': sample_question,
            'guidance': guidance
        }
    
    def _generate_oldcarts_question_for_component(self, component: str) -> str:
        """Generate question for OLDCARTS component using LLM with chief complaint and conversation context"""
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for question generation")
        
        # For character component, analyze terms from active guidelines to determine question type
        if component == 'character':
            character_analysis = self._analyze_character_terms()
            sample_question = character_analysis['sample_question']
            component_guidance_text = character_analysis['guidance']
        else:
            # Sample questions for each OLDCARTS element as guidance (ONLY reference)
            sample_questions = {
                'onset': "When did this start?",
                'progression': "Did it come on gradually or suddenly?",
                'location': "Where exactly is the pain located?",
                'timing': "Is it constant or does it come and go?",
                'duration': "How long does each episode typically last?",
                'associated': "Are there any other symptoms you're experiencing?",
                'character': "What does it feel like?",  # Default, but will be overridden if character
                'aggravating': "What makes it worse?",
                'relieving': "What helps or makes it better?",
                'severity': "On a scale of 1 to 10, how would you rate this?"
            }
            if component not in sample_questions:
                raise ValueError(f"Unknown OLDCARTS component: {component}. Must be one of: {list(sample_questions.keys())}")
            sample_question = sample_questions[component]
        
        # Component-specific guidance to prevent mixing elements - STRICT and explicit
        component_guidance = self.LLM_OLDCARTS_COMPONENT_GUIDANCE
        
        # Use character-specific guidance if character component, otherwise use standard guidance
        if component == 'character':
            guidance_text = component_guidance_text
        else:
            guidance_text = component_guidance.get(component, "")
        
        system_msg = self.LLM_OLDCARTS_SYSTEM_MSG
        
        # Get context using helper functions
        chief_complaint_context = self._get_chief_complaint_context()
        conversation_context = self._build_conversation_context(recent_items=4, char_limit=80, include_answered=True)
        
        # Make guidance more explicit and strict
        strict_instructions = self.LLM_OLDCARTS_STRICT_INSTRUCTIONS
        
        if guidance_text:
            user_msg = f"""{chief_complaint_context}{conversation_context}

Component: {component.upper()}
Example question: {sample_question}

{guidance_text}

{strict_instructions}

Generate a question about {component} for this patient:"""
        else:
            user_msg = f"""{chief_complaint_context}{conversation_context}

Component: {component.upper()}
Example question: {sample_question}

{strict_instructions}

Generate a question about {component} for this patient:"""
        
        llm_kwargs = self._get_llm_kwargs()
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        
        # Clean response to remove reasoning
        cleaned_response = self._clean_llm_response(response)
        if cleaned_response:
            generated_question = cleaned_response
            
            # VALIDATION: Ensure severity question is actually a question, not just a number
            if component == 'severity':
                # Check if response is just a number or too short (likely LLM misinterpreted as answer)
                if len(generated_question) < 15 or generated_question.strip().replace('/', '').replace('-', '').isdigit():
                    raise ValueError(f"LLM returned invalid severity question: '{generated_question}' - expected a question, got a number or too short response")
            
            return generated_question
        
        # If LLM returns empty, raise error instead of using fallback
        raise ValueError(f"LLM returned empty response for {component} question")
    
    def _check_conversation_for_distress(self) -> bool:
        """Check conversation history for distress indicators"""
        # Note: This is now only used for informational purposes, not to skip demographics
        # Check recent answers for distress
        for item in self.conversation_history[-5:]:
            if item.get('type') == 'answer':
                answer = item.get('answer', '')
                distress_info = self._detect_distress(answer)
                if distress_info['is_distressed']:
                    return True
        return False
    
    
    # ============================================================================
    # SECTION 8: UTILITIES - Helper Functions (Including All Debug Functions)
    # ============================================================================
    
    def _match_to_patient_friendly_terms(self, patient_answer: str, element_data: Dict, oldcarts_element: str) -> float:
        """
        Match patient answer to patient_friendly terms from guidelines using semantic similarity.
        
        For non-location OLDCARTS elements, this replaces the unified function.
        Returns the highest similarity score between patient answer and any patient_friendly term.
        
        Args:
            patient_answer: The patient's response
            element_data: The structured OLDCARTS element data (with 'includes' list)
            oldcarts_element: The OLDCARTS element being matched (e.g., 'character', 'onset')
            
        Returns:
            float: Highest similarity score (0.0-1.0) between patient answer and patient_friendly terms
        """
        if not self.medical_rule_engine or not self.medical_rule_engine.embedding_model:
            return 0.5  # Default score if no embedding model
        
        if not element_data or not isinstance(element_data, dict):
            return 0.0
        
        includes = element_data.get('includes', [])
        if not includes:
            return 0.0
        
        # Collect all patient_friendly terms from this element
        patient_friendly_terms = self._extract_patient_friendly_from_includes(includes)
        
        if not patient_friendly_terms:
            return 0.0
        
        try:
            # Encode patient answer (raw, no normalization) and all patient_friendly terms
            texts_to_encode = [patient_answer] + patient_friendly_terms
            embeddings = self.medical_rule_engine.embedding_model.encode(texts_to_encode)
            embeddings = np.asarray(embeddings, dtype='float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Get patient answer embedding (first one)
            patient_emb = embeddings[0]
            
            # Calculate similarity with each patient_friendly term
            max_similarity = 0.0
            for i, pf_term in enumerate(patient_friendly_terms):
                pf_emb = embeddings[i + 1]  # +1 because first is patient answer
                similarity = float(np.dot(patient_emb, pf_emb))
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
        except Exception as e:
            self._capture_debug(f"[Scoring] ⚠️ Error matching to patient_friendly terms: {e}")
            return 0.0
        
    def _capture_debug(self, message: str):
        """Capture debug output"""
        self._captured_debug_output.append(message)
        print(message)
    
    def reset_assessment(self):
        """Reset for new patient"""
        self.active_guidelines = []
        self.reserve_pool = []
        self.ruled_out = []
        self.chief_complaint = None
        self.status = "idle"
        self.conversation_history = []
        self.demographics = {}
        self.demographics_optional = False  # Reset distress flag
        self.oldcarts_covered = {'O': False, 'L': False, 'T': False, 'D': False, 'C': False, 'A': False, 'R': False, 'S': False, 'AS': False}
        self.oldcarts_analysis = None
        self.clarification_count = {}
        self.diagnosed_condition = None
        self.radiation_asked = False  # Track if radiation question has been asked
        self.radiation_answered = False  # Track if radiation has been answered
        self.red_flags_present = []
        self.red_flag_index = 0
        self.red_flag_phase = False  # Track if we're in red flag screening phase
        self.red_flags_list = []  # List of red flags to ask about
        self.key_features_phase = False  # Track if we're in key positives/negatives phase
        self.key_features_index = 0  # Track which key feature we're asking about
        self.key_features_list = []  # List of key features to ask about
        self.MAX_ACTIVE = 5
        self.RULE_OUT_THRESHOLD = 0.05
    
    def _get_debug_info(self, last_answer: str = None) -> Dict:
        """Build debug information"""
        num_questions = len([item for item in self.conversation_history if item['type'] == 'question'])
        covered_count = sum(self.oldcarts_covered.values())
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        
        return {
            'demographics': self.demographics,
            'question_number': num_questions,
            'oldcarts_coverage': coverage_str,
            'active_differentials': [
                {'rank': i+1, 'name': g['name'], 'score': g['score']}
                for i, g in enumerate(self.active_guidelines[:5])
            ]
        }

    def _format_engine_debug(self, prefix_note: str = None) -> str:
        """Return formatted debug banner similar to Telegram output."""
        lines = []
        lines.append("="*80)
        lines.append("[Telegram] 🧠 ENGINE DEBUG OUTPUT")
        lines.append("="*80)
        lines.append(f"[Engine] 🎯 Structured guidelines: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}")
        if prefix_note:
            lines.append(prefix_note)
        # OLDCARTS coverage
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        lines.append(f"[Engine] 📋 OLDCARTS: {coverage_str} ({sum(self.oldcarts_covered.values())}/8)")
        # Differentials
        lines.append("[Engine] 📊 ACTIVE DIFFERENTIALS:")
        for i, g in enumerate(self.active_guidelines[:5], start=1):
            severity = g.get('urgency', 'routine')
            name = g.get('name', 'Unknown')
            score_val = g.get('score', 0.5) if isinstance(g.get('score', None), (int, float)) else 0.5
            score = round(score_val * 100, 1)  # Show 1 decimal place for precision
            sev_icon = '⚠️' if 'urgent' in str(severity).lower() else '📋'
            lines.append(f"  {i}. {name}: {score}% ({severity}) {sev_icon}")
        lines.append(f"[Engine] 🔄 Pool: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        return "\n".join(lines)

    def _format_rankings_debug(self) -> str:
        """Return formatted UPDATED RANKINGS block and pool statistics."""
        def urgency_icon(u):
            u_str = str(u or 'routine').lower()
            if 'emerg' in u_str:
                return '🚨'
            if 'urgent' in u_str:
                return '⚠️'
            return '📋'

        lines = []
        lines.append("[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines[:5], start=1):
            name = g.get('data', {}).get('condition', g.get('name', 'Unknown'))
            score = g.get('score') or 0.0
            pct = round(score * 100, 1)  # Show 1 decimal place for precision
            urg = g.get('urgency') or g.get('data', {}).get('urgency', 'routine')
            icon = urgency_icon(urg)
            lines.append(f"[Engine]   {i}. {name}: {pct}% {icon}")
            lines.append(f"[Scoring] 🏆 Top {i}: {name}")
            lines.append(f"[Scoring]   📊 Score: {pct}%")
            prev = g.get('data', {}).get('prevalence', 'unknown')
            lines.append(f"[Scoring]   📋 Prevalence: {prev}")
            lines.append(f"[Scoring]   🎯 ML Confidence: High similarity match")
            lines.append(f"[Scoring]   🚨 Urgency: {urg if urg else 'routine'}")
        lines.append("")
        lines.append(f"[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        lines.append("[Scoring] 📊 Final statistics:")
        lines.append(f"[Scoring]   🎯 Active Conditions: {len(self.active_guidelines)}")
        lines.append(f"[Scoring]   📋 Reserve Conditions: {len(self.reserve_pool)}")
        lines.append(f"[Scoring]   ❌ Ruled Out: {len(self.ruled_out)}")
        total_processed = len(self.active_guidelines) + len(self.reserve_pool) + len(self.ruled_out)
        lines.append(f"[Scoring]   📈 Total Processed: {total_processed}")
        lines.append(f"[Scoring]   🧠 ML System: Fully operational")
        return "\n".join(lines)
    
    
