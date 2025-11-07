#!/usr/bin/env python3
"""
Advanced Medical Navigator - Hybrid LLM/RAG/FAISS Medical Assistant

A medical assistant that combines:
- LLM for natural conversation and question generation
- Medical guidelines with structured OLDCARTS
- RAG with FAISS for semantic similarity matching
- Rolling ranking system of top 5 conditions
- Dynamic question selection based on condition differentiation

Features:
- Natural, human-like conversations
- Evidence-based diagnosis support
- Dynamic condition ranking
- Context-aware question selection

ASSESSMENT ALGORITHM SECTIONS:
1. CONFIGURATION (Top) - All thresholds, LLM rules, weights for easy tuning
2. INITIALIZATION - Setup and loading
3. GREETING HANDLING - Greeting detection and responses (before chief complaint)
4. CHIEF COMPLAINT - Category matching and narrowing
5. DEMOGRAPHICS - Age, sex, chronicity extraction (if needed)
6. ASSESSMENT - OLDCARTS processing, scoring, question generation
7. UTILITIES - Helper functions
8. DEBUGGING - Debug functions (last)
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict


class AdvancedMedicalNavigator:
    """
    Advanced Medical Navigator - Hybrid LLM/RAG/FAISS medical assistant.
    
    Combines:
    - LLM for natural conversation
    - Medical guidelines with structured OLDCARTS
    - RAG with FAISS for semantic matching
    - Rolling condition ranking system
    """
    
    # ============================================================================
    # SECTION 1: CONFIGURATION (Top - Easy Tuning)
    # ============================================================================
    
    # ===== THRESHOLD CONFIGURATION =====
    # FAISS semantic matching thresholds
    FAISS_SEMANTIC_THRESHOLD = 0.65  # Main threshold for FAISS semantic matching (OLDCARTS elements)
    CHIEF_COMPLAINT_MATCHING_THRESHOLD = 0.6  # Threshold for chief complaint to trigger matching
    
    # ===== LLM RULES & GUIDELINES =====
    # LLM prompts and system messages
    
    # Chief Complaint Extraction
    LLM_CHIEF_COMPLAINT_SYSTEM_MSG = """You are a medical assistant. Extract the chief complaint (main symptom or concern) from the patient's message.
Return ONLY the chief complaint in simple terms, or 'none' if unclear."""
    
    # OLDCARTS Element Extraction
    LLM_OLDCARTS_EXTRACTION_SYSTEM_MSG = """You are a medical assistant. Determine which OLDCARTS element (Onset, Location, Duration, Character, Aggravating, Relieving, Timing, Severity, Associated) was answered in the patient's response.

Return ONLY the element name (lowercase), or 'none' if unclear.

OLDCARTS elements:
- onset: When did it start?
- location: Where is it?
- duration: How long does it last?
- character: What does it feel like?
- aggravating: What makes it worse?
- relieving: What makes it better?
- timing: Constant or intermittent?
- severity: How bad (1-10)?
- associated: Other symptoms?"""
    
    # Demographics Questions (matching adaptive engine)
    LLM_CHRONICITY_SYSTEM_MSG = "You are a medical assistant. Generate a concise question to ask if the patient's problem is new or ongoing."
    LLM_CHRONICITY_USER_MSG = "Is this a new problem or an ongoing issue?"
    
    # Empathetic Response (matching adaptive engine)
    LLM_EMPATHETIC_SYSTEM_MSG = """You are a compassionate medical assistant. The patient is expressing significant distress with severe symptoms. 
Generate a brief (1-2 sentences), empathetic response that:
1. Acknowledges their distress and validates their feelings
2. Reassures them you're taking this seriously
3. Shows urgency and concern

Be warm, professional, and reassuring. Do NOT ask any questions - just acknowledge and reassure. The system will ask the appropriate clinical question next."""
    
    # Question/Comment Acknowledgment (matching adaptive engine)
    LLM_QUESTION_ACK_SYSTEM_MSG = "You are a compassionate medical assistant. Generate a brief, natural acknowledgment (1 sentence) for a patient's question. Be warm and reassuring."
    LLM_COMMENT_ACK_SYSTEM_MSG = "You are a compassionate medical assistant. Generate a brief, natural acknowledgment (1 sentence) for a patient's comment or emotional expression. Be warm and reassuring, then naturally transition back to gathering information."
    
    # Greeting Detection
    LLM_GREETING_DETECTION_SYSTEM_MSG = """You are a medical assistant. Determine if the patient's message is a greeting or a medical concern.

Return ONLY 'greeting' or 'medical'.

Examples of GREETINGS:
- "hello", "hi", "hey"
- "good morning", "good afternoon"
- "how are you"
- Any casual greeting or small talk

Examples of MEDICAL:
- "I have chest pain"
- "My stomach hurts"
- "I'm feeling nauseous"
- Any symptom description or medical question"""
    
    # Greeting Response
    LLM_GREETING_SYSTEM_MSG = """You are Aura, a friendly and helpful medical AI assistant.
Respond to greetings warmly and briefly. Mention that you're here to help with medical questions or symptom assessment.
Keep it conversational and inviting. Respond in 1-2 short sentences."""
    
    # Chief Complaint Acknowledgment
    LLM_CHIEF_COMPLAINT_ACK_SYSTEM_MSG = """You are a medical assistant. The patient just described their chief complaint.
Respond with:
1. A brief, empathetic acknowledgment (1 sentence)
2. A natural first question to start the assessment

For the first question, ask about either:
- Location: "Where exactly is the [symptom]?" (if it's a pain/discomfort)
- Character: "What does it feel like?" or "Can you describe it?" (if it's a sensation)

Keep it conversational and human. Don't be overly formal."""
    
    # Assessment Question Generation
    LLM_ASSESSMENT_SYSTEM_MSG_TEMPLATE = """You are a medical assistant conducting a symptom assessment using the OLDCARTS framework.

OLDCARTS stands for:
- Onset: When did it start? Was it sudden or gradual?
- Location: Where exactly is it located? Can you point to it?
- Duration: How long does it last? Has it been constant, or does it come and go?
- Character: Can you describe it? What does it feel like? (sharp, dull, pressure, burning, etc.)
- Aggravating: What makes it worse? (e.g., activity, movement, breathing)
- Relieving: What makes it better? (e.g., rest, medication, position)
- Timing: Is it constant or does it come and go? How long does each episode last?
- Severity: On a scale of 1 to 10, how severe is it?
- Associated: Are there any other symptoms you've noticed?

CRITICAL INSTRUCTIONS:
1. You will be asked about a SPECIFIC patient's chief complaint
2. Generate a question about the element: {next_element}
3. The question MUST be about the PATIENT'S ACTUAL COMPLAINT (not the examples)
4. Use the examples ONLY for formatting style (HOW to ask), NOT for content (WHAT to ask about)
5. Be conversational, empathetic, and specific to the patient's situation

DO NOT:
- Use topics from the examples (e.g., if examples mention "chest pain" but patient has "abdominal pain", ask about ABDOMINAL pain)
- Ask multiple questions at once
- Use medical jargon
- Sound robotic or formal

Return ONLY the question, no explanations:"""
    
    # ===== OLDCARTS QUESTION TEMPLATES =====
    # Comprehensive examples for question formatting (style guide)
    # Organized by OLDCARTS element with examples from different medical systems
    # The LLM uses these as style guides to generate similar questions tailored to the patient's complaint
    OLDCARTS_QUESTION_TEMPLATES = {
        'onset': [
            # Cardiovascular
            'When did the chest pain begin, and did it start suddenly?',
            # Pulmonary
            'When did your cough and fever start?',
            # GI
            'When did the abdominal pain start, and did it start suddenly around your belly button?',
            # MSK
            'When exactly did the injury happen, and what were you doing?',
            # Dermatology
            'When did you first notice the redness and swelling begin?',
            # Renal
            'When did this sudden, severe flank (side) pain start?',
            # GU/GYN
            'When did you first notice the burning with urination begin?'
        ],
        'location': [
            # Cardiovascular
            'Can you point to exactly where the pain is? Does it move to your jaw, neck, or arm?',
            # Pulmonary
            'Are you feeling any pain in your chest, and can you point to where?',
            # GI
            'Did the pain start in the middle and then move to your lower right side?',
            # MSK
            'Can you show me precisely where on your ankle the pain is the worst?',
            # Dermatology
            'Which part of your leg is affected?',
            # Renal
            'Does the pain start in your back/side and move down towards your groin?',
            # GU/GYN
            'Do you feel pain in your lower abdomen, pelvis, or during urination?'
        ],
        'duration': [
            # Cardiovascular
            'How long has this pain lasted? Is it constant or intermittent?',
            # Pulmonary
            'Have you had this cough for a few days, or is it a chronic issue?',
            # GI
            'Is the pain constant now, or does it come and go?',
            # MSK
            'Is the pain constant since the injury, or does it only hurt when you try to move it?',
            # Dermatology
            'Has the redness been constant since you noticed it, or does it fluctuate?',
            # Renal
            'Is the pain constant, or does it come in waves?',
            # GU/GYN
            'Have these symptoms been constant for the last few days?'
        ],
        'character': [
            # Cardiovascular
            'How would you describe the pain? Is it a pressure, squeezing, or a sharp pain?',
            # Pulmonary
            'How would you describe the cough? Is it dry, or are you coughing up phlegm (sputum)?',
            # GI
            'How would you describe the pain? Is it a dull ache or a sharp, stabbing sensation?',
            # MSK
            'How would you describe the pain? Is it sharp, or a constant deep ache?',
            # Dermatology
            'How does the skin feel? Is it hot, tight, tender, or itchy?',
            # Renal
            'How would you describe the pain? Is it a sharp, intense, cramping pain?',
            # GU/GYN
            'How would you describe the pain? Is it a sharp burning feeling, or a dull pelvic ache?'
        ],
        'aggravating': [
            # Cardiovascular
            'Does physical activity, like walking, make the pain worse?',
            # Pulmonary
            'Does taking a deep breath make the chest pain or cough worse?',
            # GI
            'Does moving around, coughing, or going over bumps in a car make the pain worse?',
            # MSK
            'Does putting any weight on your foot or trying to walk make the pain worse?',
            # Dermatology
            'Does wearing tight clothing or walking for a long time make the area hurt more?',
            # Renal
            'Does movement or trying to find a comfortable position make the pain worse?',
            # GU/GYN
            'Does going to the bathroom or having intercourse make the pain worse?'
        ],
        'relieving': [
            # Cardiovascular
            'Does rest or medication (like nitroglycerin) make the pain better?',
            # Pulmonary
            'Does rest, a change in position, or any medication make you feel better?',
            # GI
            'Does lying perfectly still or anything else make it feel better?',
            # MSK
            'Does resting, elevating your foot, or putting ice on it help reduce the pain?',
            # Dermatology
            'Does elevating your leg or putting a cool compress on it help with the discomfort?',
            # Renal
            'Does anything make the pain better? Are you unable to find a comfortable position?',
            # GU/GYN
            'Does sitting down or taking a warm bath help ease the pain?'
        ],
        'timing': [
            # Cardiovascular
            'Has the pain been getting worse steadily over time?',
            # Pulmonary
            'Is the cough worse at night?',
            # GI
            'Has the pain been getting steadily worse over the last few hours/day?',
            # MSK
            'Has the swelling or pain increased since the initial injury?',
            # Dermatology
            'Has the red area been growing in size over the last 24 hours?',
            # Renal
            'Do the waves of pain seem to come closer together over time?',
            # GU/GYN
            'Are you needing to go to the bathroom much more frequently than usual?'
        ],
        'severity': [
            # Cardiovascular
            'On a scale of 1 to 10, how would you rate this pain?',
            # Pulmonary
            'How severe is your shortness of breath? Are you able to speak in full sentences?',
            # GI
            'On a scale of 1 to 10, how intense is the pain currently?',
            # MSK
            'On a scale of 1 to 10, how bad is the pain right now? Can you bear any weight at all?',
            # Dermatology
            'How much pain are you in? Is it severe enough to keep you from walking normally?',
            # Renal
            'On a scale of 1 to 10, how severe is this pain?',
            # GU/GYN
            'How uncomfortable are you? Is the pain affecting your ability to perform daily activities?'
        ],
        'associated': [
            # Cardiovascular
            'Are you experiencing any shortness of breath, nausea, or sweating?',
            # Pulmonary
            'Are you experiencing any fever, chills, or chest tightness?',
            # GI
            'Are you experiencing any nausea, vomiting, or changes in your appetite?',
            # MSK
            'Are you able to move the joint, or is it completely locked up?',
            # Dermatology
            'Have you noticed any fever, chills, or spreading of the redness?',
            # Renal
            'Are you experiencing any nausea, vomiting, or blood in your urine?',
            # GU/GYN
            'Are you experiencing any fever, discharge, or changes in your menstrual cycle?'
        ]
    }
    
    # ===== OLDCARTS PRIORITY ORDER =====
    # Order in which to ask OLDCARTS questions (if not using differentiation)
    OLDCARTS_PRIORITY_ORDER = ['location', 'character', 'timing', 'severity', 'duration', 
                               'onset', 'aggravating', 'relieving', 'associated']
    
    # ===== TOP CONDITIONS LIMIT =====
    TOP_CONDITIONS_LIMIT = 5  # Number of top conditions to track
    
    # ===== GREETING DETECTION =====
    # LLM-based greeting detection (no hardcoded patterns needed)
    
    # ===== CATEGORY MAPPING =====
    # Map directory names (GI, CARDIO) to full category names (gastrointestinal, cardiovascular)
    # This matches the medical_rule_engine's category naming
    CATEGORY_DIR_TO_FULL_NAME = {
        'GI': 'gastrointestinal',
        'CARDIO': 'cardiovascular',
        'PULMONARY': 'respiratory',
        'NEURO': 'neurological',
        'MSK': 'musculoskeletal',
        'RENAL': 'renal',
        'GU': 'genitourinary',
        'GYN': 'gynecological',
        'DERM': 'dermatological'
    }
    
    # ============================================================================
    # SECTION 2: INITIALIZATION
    # ============================================================================
    
    def __init__(self, llm_chat_fn=None, medical_rule_engine=None, embedding_model=None):
        """
        Initialize Advanced Medical Navigator.
        
        Args:
            llm_chat_fn: Single LLM function for all interactions
            medical_rule_engine: MedicalRuleEngine instance for guidelines and FAISS
            embedding_model: Embedding model for semantic matching
        """
        self.llm_chat_fn = llm_chat_fn
        self.medical_rule_engine = medical_rule_engine
        self.embedding_model = embedding_model
        
        # Load LLM configuration from environment (no hardcoded parameters)
        self.temperature = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.7'))
        self.top_p = float(os.environ.get('LLM_TOP_P', '0.9'))
        self.top_k = int(os.environ.get('LLM_TOP_K', '40'))
        self.repeat_penalty = float(os.environ.get('LLM_REPEAT_PENALTY', '1.1'))
        self.presence_penalty = float(os.environ.get('LLM_PRESENCE_PENALTY', '0.0'))
        self.frequency_penalty = float(os.environ.get('LLM_FREQUENCY_PENALTY', '0.0'))
        self.num_predict = int(os.environ.get('LLM_NUM_PREDICT', '100'))
        
        # Stop sequences
        stop_env = os.environ.get('LLM_STOP', '').strip()
        self.stop_sequences = [s.strip() for s in stop_env.split(',') if s.strip()] if stop_env else None
        
        # Helper function to get LLM kwargs
        def _get_llm_kwargs(override_max_tokens=None, override_temperature=None):
            """Get all LLM parameters for generation from environment"""
            kwargs = {
                'temperature': override_temperature if override_temperature is not None else self.temperature,
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
        
        # Guidelines loaded on-demand based on chief complaint (for latency optimization)
        self.all_guidelines: Dict[str, Dict] = {}
        self.guidelines_path = os.path.join(
            os.path.dirname(__file__),
            'medical', 'guidelines'
        )
        
        # Category mapping and chief complaint triggers (for quick lookup)
        self.category_to_guidelines: Dict[str, List[str]] = {}
        self.all_chief_complaint_triggers: List[Dict] = []  # [{trigger, category, condition, filepath}, ...]
        self._build_category_mapping()
        self._build_chief_complaint_triggers_index()
        
        # Note: All FAISS indexes are built at startup by MedicalRuleEngine
        # We only load guidelines and activate category indexes based on chief complaint
        
        # Active sessions
        self.sessions: Dict[str, 'AdvancedMedicalNavigator.MedicalSession'] = {}
        
        # Debug output capture
        self._captured_debug_output = []
        
        if not self.llm_chat_fn:
            raise ValueError("LLM function is required for Advanced Medical Navigator")
        
        print("[Navigator] ✅ Advanced Medical Navigator initialized (hybrid LLM/RAG/FAISS mode)")
        print(f"[Navigator] 📋 All FAISS indexes built at startup. Guidelines loaded on-demand based on chief complaint")
    
    def _build_category_mapping(self):
        """Build mapping of categories to guideline files (without loading full guidelines)"""
        if not os.path.exists(self.guidelines_path):
            print(f"[Navigator] ⚠️ Guidelines directory not found: {self.guidelines_path}")
            return
        
        # Get enabled categories from environment variable (same as medical_rule_engine)
        enabled_categories_env = os.environ.get('ENABLED_MEDICAL_CATEGORIES', 'GI').strip()
        enabled_categories = [cat.strip().upper() for cat in enabled_categories_env.split(',') if cat.strip()]
        
        # Scan only enabled category directories
        for category_dir in os.listdir(self.guidelines_path):
            category_path = os.path.join(self.guidelines_path, category_dir)
            if not os.path.isdir(category_path):
                continue
            
            # Filter by enabled categories
            if enabled_categories and category_dir.upper() not in enabled_categories:
                continue
            
            self.category_to_guidelines[category_dir] = []
            for filename in os.listdir(category_path):
                if filename.endswith('.json'):
                    self.category_to_guidelines[category_dir].append(filename)
        
        print(f"[Navigator] 📂 Found {len(self.category_to_guidelines)} enabled categories: {', '.join(self.category_to_guidelines.keys())}")
        if enabled_categories:
            print(f"[Navigator] 🔍 Filtering to enabled categories: {', '.join(enabled_categories)}")
    
    def _build_chief_complaint_triggers_index(self):
        """Build index of all chief complaint triggers from all guidelines (lightweight scan)"""
        if not os.path.exists(self.guidelines_path):
            return
        
        # Scan all guidelines to collect chief complaint triggers
        for category_dir in self.category_to_guidelines.keys():
            category_path = os.path.join(self.guidelines_path, category_dir)
            
            for filename in self.category_to_guidelines[category_dir]:
                filepath = os.path.join(category_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        guideline = json.load(f)
                        triggers = guideline.get('chief_complaint_triggers', [])
                        condition_name = guideline.get('condition', filename.replace('.json', ''))
                        
                        for trigger in triggers:
                            self.all_chief_complaint_triggers.append({
                                'trigger': trigger,
                                'category': category_dir,
                                'condition': condition_name,
                                'filepath': filepath
                            })
                except Exception as e:
                    print(f"[Navigator] ⚠️ Failed to scan guideline {filename}: {e}")
        
        print(f"[Navigator] 📋 Indexed {len(self.all_chief_complaint_triggers)} chief complaint triggers from all guidelines")
    
    def _load_single_guideline(self, category: str, filename: str) -> Optional[Dict]:
        """Load a single guideline file (lightweight)"""
        filepath = os.path.join(self.guidelines_path, category, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Navigator] ⚠️ Failed to load guideline {filename}: {e}")
            return None
    
    # ============================================================================
    # SECTION 3: GREETING HANDLING (Before Chief Complaint)
    # ============================================================================
    
    def _is_greeting(self, message: str) -> bool:
        """
        Detect if message is a greeting using LLM (before chief complaint matching).
        """
        if not self.llm_chat_fn:
            raise ValueError("LLM function required for greeting detection")
        
        # Use LLM to intelligently detect greetings
        llm_kwargs = self._get_llm_kwargs(override_max_tokens=10, override_temperature=0.1)
        response = self.llm_chat_fn(
            [{"role": "system", "content": self.LLM_GREETING_DETECTION_SYSTEM_MSG}, 
             {"role": "user", "content": f"Patient message: {message}\n\nIs this a greeting or medical? Return ONLY 'greeting' or 'medical':"}],
            **llm_kwargs
        )
        
        result = response.strip().lower()
        is_greeting = 'greeting' in result and 'medical' not in result
        
        if is_greeting:
            self._capture_debug(f"[Navigator] 👋 Detected greeting: '{message}'")
        
        return is_greeting
    
    # ============================================================================
    # SECTION 4: CHIEF COMPLAINT
    # ============================================================================
    
    def _match_chief_complaint_to_categories(self, chief_complaint: str) -> List[str]:
        """
        Match chief complaint to categories by semantically comparing to all chief_complaint_triggers.
        
        Algorithm:
        1. Semantically compare chief complaint to all triggers from all guidelines
        2. Find matching guidelines (above threshold)
        3. Extract categories from matching guidelines
        4. Return categories (single or multiple if overlap)
        """
        if not self.embedding_model or not self.all_chief_complaint_triggers:
            raise ValueError("Embedding model and chief complaint triggers required for category matching")
        
        # Semantically compare chief complaint to all triggers
        threshold = self.CHIEF_COMPLAINT_MATCHING_THRESHOLD
        matched_guidelines = []  # [{category, condition, score}, ...]
        
        import numpy as np
        
        # Encode chief complaint
        chief_complaint_embedding = self.embedding_model.encode([chief_complaint])[0]
        chief_emb = np.array(chief_complaint_embedding).reshape(1, -1)
        chief_norm = chief_emb / np.linalg.norm(chief_emb)
        
        # Compare to all triggers
        for trigger_data in self.all_chief_complaint_triggers:
            trigger = trigger_data['trigger']
            
            # Encode trigger
            trigger_embedding = self.embedding_model.encode([trigger])[0]
            trigger_emb = np.array(trigger_embedding).reshape(1, -1)
            trigger_norm = trigger_emb / np.linalg.norm(trigger_emb)
            
            # Calculate cosine similarity
            similarity = float(np.dot(trigger_norm, chief_norm.T)[0][0])
            
            if similarity >= threshold:
                matched_guidelines.append({
                    'category': trigger_data['category'],
                    'condition': trigger_data['condition'],
                    'filepath': trigger_data['filepath'],
                    'score': similarity
                })
        
        # Extract unique categories from matched guidelines
        matched_categories = list(set([g['category'] for g in matched_guidelines]))
        
        if matched_guidelines:
            self._capture_debug(f"[Navigator] 🔍 Chief complaint '{chief_complaint}' matched {len(matched_guidelines)} guidelines in categories: {matched_categories}")
        else:
            self._capture_debug(f"[Navigator] ⚠️ No semantic match found for chief complaint: '{chief_complaint}' (threshold: {threshold}) - will use general LLM knowledge with OLDCARTS")
        
        # Return empty list if no matches - caller will use general LLM knowledge
        return matched_categories
    
    def _load_guidelines_for_categories(self, categories: List[str], chief_complaint: str = None):
        """
        Load only guidelines that matched the chief complaint (on-demand).
        Indexes are already built at startup - we just activate them.
        
        Args:
            categories: Categories to load from (determined by chief complaint matching)
            chief_complaint: Chief complaint to match against (for loading only matching guidelines)
        """
        if not categories:
            # No categories to load - this is expected when using general LLM
            return
        
        loaded_count = 0
        
        # Find which specific guidelines matched the chief complaint
        matched_guideline_files = set()  # {(category, filename), ...}
        
        if chief_complaint and self.all_chief_complaint_triggers:
            # Semantically compare chief complaint to all triggers to find matching guidelines
            import numpy as np
            chief_emb = np.array(self.embedding_model.encode([chief_complaint])[0])
            chief_norm = chief_emb / np.linalg.norm(chief_emb)
            
            for trigger_data in self.all_chief_complaint_triggers:
                # Only check triggers from matched categories
                if trigger_data['category'] not in categories:
                    continue
                
                trigger_emb = np.array(self.embedding_model.encode([trigger_data['trigger']])[0])
                trigger_norm = trigger_emb / np.linalg.norm(trigger_emb)
                similarity = float(np.dot(trigger_norm, chief_norm))
                
                if similarity >= self.CHIEF_COMPLAINT_MATCHING_THRESHOLD:
                    # Extract filename from filepath
                    filepath = trigger_data['filepath']
                    filename = os.path.basename(filepath)
                    matched_guideline_files.add((trigger_data['category'], filename))
        
        # Load only matching guidelines
        for category in categories:
            if category not in self.category_to_guidelines:
                continue
            
            for filename in self.category_to_guidelines[category]:
                # If we have matched files, only load those specific guidelines
                if matched_guideline_files is not None and (category, filename) not in matched_guideline_files:
                    continue
                
                condition_name = filename.replace('.json', '').replace(f'{category}_', '')
                
                # Skip if already loaded
                if condition_name in self.all_guidelines:
                    continue
                
                guideline = self._load_single_guideline(category, filename)
                if guideline:
                    self.all_guidelines[condition_name] = guideline
                    loaded_count += 1
        
        self._capture_debug(f"[Navigator] 📋 Loaded {loaded_count} matching guidelines for categories: {', '.join(categories)}")
        
        # Activate FAISS indexes for these categories (indexes already built at startup)
        # This ensures FAISS searches are limited to relevant categories (latency optimization)
        if self.medical_rule_engine and categories:
            # Translate directory names (GI, CARDIO) to full names (gastrointestinal, cardiovascular)
            # Medical rule engine uses full names for indexing
            full_category_names = [self.CATEGORY_DIR_TO_FULL_NAME.get(cat, cat.lower()) for cat in categories]
            
            if len(full_category_names) > 1:
                # Multiple categories - activate merged indexes
                self.medical_rule_engine.set_active_category(full_category_names)
                self._capture_debug(f"[Navigator] 🔍 Activated merged FAISS indexes for categories: {', '.join(full_category_names)}")
            elif len(full_category_names) == 1:
                # Single category - activate category-specific index
                self.medical_rule_engine.set_active_category(full_category_names[0])
                self._capture_debug(f"[Navigator] 🔍 Activated FAISS index for category: {full_category_names[0]}")
            
            # Note: FAISS searches are already element-specific via the 'element' parameter
            # This ensures we only search the pertinent OLDCARTS element section
    
    # ============================================================================
    # SECTION 5: DEMOGRAPHICS
    # ============================================================================
    
    def _needs_demographics(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> str:
        """Check which demographic info is missing (chronicity, age, sex)"""
        demographics = session.context.get('demographics', {})
        
        if 'chronicity' not in demographics:
            return 'chronicity'
        if 'age' not in demographics:
            return 'age'
        if 'sex' not in demographics:
            return 'sex'
        
        return None  # All demographics collected
    
    def _handle_demographics(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> Dict[str, Any]:
        """Handle demographics questions (chronicity, age, sex) before OLDCARTS"""
        missing_demo = self._needs_demographics(session)
        
        if not missing_demo:
            return None  # All demographics collected
        
        if missing_demo == 'chronicity':
            question = "Is this a new problem or an ongoing issue you've been dealing with?"
            self._capture_debug(f"[Navigator] 📋 Demographics: Asking chronicity")
            return {
                'response': question,
                'status': 'demographics',
                'metadata': {'asking': 'chronicity'}
            }
        elif missing_demo == 'age':
            question = "Can you please tell me your age so I can better assist you?"
            self._capture_debug(f"[Navigator] 📋 Demographics: Asking age")
            return {
                'response': question,
                'status': 'demographics',
                'metadata': {'asking': 'age'}
            }
        elif missing_demo == 'sex':
            question = "What is your biological sex?"
            self._capture_debug(f"[Navigator] 📋 Demographics: Asking sex")
            return {
                'response': question,
                'status': 'demographics',
                'metadata': {'asking': 'sex'}
            }
        
        return None
    
    def _extract_demographics(self, session: 'AdvancedMedicalNavigator.MedicalSession', user_message: str):
        """Extract demographics (chronicity, age, sex) from user response"""
        if 'demographics' not in session.context:
            session.context['demographics'] = {}
        
        demographics = session.context['demographics']
        
        # Extract chronicity
        if 'chronicity' not in demographics:
            user_lower = user_message.lower().strip()
            new_keywords = ['new', 'first time', 'never had', 'just started', 'just began', 'brand new']
            recurring_keywords = ['ongoing', 'chronic', 'recurring', 'always', 'frequent', 'often', 'again', 'returned', 'keeps coming back', 'persistent']
            
            if any(kw in user_lower for kw in new_keywords):
                demographics['chronicity'] = 'new'
                self._capture_debug(f"[Navigator] ✅ Chronicity extracted: new")
            elif any(kw in user_lower for kw in recurring_keywords):
                demographics['chronicity'] = 'recurring'
                self._capture_debug(f"[Navigator] ✅ Chronicity extracted: recurring")
        
        # Extract age
        if 'age' not in demographics:
            import re
            age_match = re.search(r'\b(\d{1,3})\b', user_message)
            if age_match:
                age = int(age_match.group(1))
                if 0 <= age <= 150:
                    demographics['age'] = age
                    self._capture_debug(f"[Navigator] ✅ Age extracted: {age}")
        
        # Extract sex
        if 'sex' not in demographics:
            user_lower = user_message.lower().strip()
            if any(word in user_lower for word in ['male', 'man', 'boy']):
                demographics['sex'] = 'male'
                self._capture_debug(f"[Navigator] ✅ Sex extracted: male")
            elif any(word in user_lower for word in ['female', 'woman', 'girl']):
                demographics['sex'] = 'female'
                self._capture_debug(f"[Navigator] ✅ Sex extracted: female")
    
    # ============================================================================
    # SECTION 6: ASSESSMENT - OLDCARTS Processing, Scoring, Question Generation
    # ============================================================================
    
    def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Unique session identifier
            user_message: User's message
        
        Returns:
            {
                'response': str,
                'status': str,  # 'greeting', 'assessment', 'complete', etc.
                'metadata': dict
            }
        """
        # Get or create session
        session = self._get_or_create_session(session_id)
        session.add_message('user', user_message)
        
        # Extract information from user's answer (if in assessment phase)
        if session.context.get('chief_complaint'):
            # Extract demographics first (chronicity, age, sex)
            self._extract_demographics(session, user_message)
            
            # Extract OLDCARTS info
            self._extract_oldcarts_info(session, user_message)
            
            # Score patient answer against guidelines using FAISS (only if using guidelines)
            if not session.context.get('use_general_llm', False):
                if self.medical_rule_engine and self.embedding_model:
                    self._score_patient_answer(session, user_message)
                    
                    # Update condition rankings (top 5)
                    self._update_condition_rankings(session)
        
        # Check if message is a greeting FIRST (before any other processing)
        if self._is_greeting(user_message) and not session.context.get('chief_complaint'):
            response = self._handle_greeting(session, user_message)
        else:
            # Determine conversation phase
            phase = self._determine_phase(session)
            
            # Generate response based on phase
            if phase == "greeting":
                response = self._handle_greeting(session, user_message)
            elif phase == "chief_complaint":
                response = self._handle_chief_complaint(session, user_message)
            elif phase == "assessment":
                # Check if demographics are complete first
                demo_response = self._handle_demographics(session)
                if demo_response:
                    response = demo_response
                else:
                    response = self._handle_assessment(session, user_message)
            else:
                response = self._handle_followup(session, user_message)
        
        # Add assistant response to session
        session.add_message('assistant', response['response'])
        
        return response
    
    def _determine_phase(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> str:
        """Determine current conversation phase"""
        # If no chief complaint yet, check if this is first exchange or subsequent
        if not session.context.get('chief_complaint'):
            # First exchange (user message only) = greeting
            # Second exchange (user + assistant) = chief complaint extraction
            if len(session.messages) <= 1:  # Only user message, no assistant response yet
                return "greeting"
            return "chief_complaint"
        return "assessment"
    
    def _handle_greeting(self, session: 'AdvancedMedicalNavigator.MedicalSession', message: str) -> Dict[str, Any]:
        """Handle greeting phase"""
        if not self.llm_chat_fn:
            raise ValueError("LLM function required for greeting handling")
        
        llm_kwargs = self._get_llm_kwargs(override_max_tokens=100)
        response = self.llm_chat_fn(
            [{"role": "system", "content": self.LLM_GREETING_SYSTEM_MSG}, {"role": "user", "content": message}],
            **llm_kwargs
        )
        greeting = response.strip() if response else "Hello! How can I help you today?"
        
        return {
            'response': greeting,
            'status': 'greeting',
            'metadata': {}
        }
    
    def _handle_chief_complaint(self, session: 'AdvancedMedicalNavigator.MedicalSession', message: str) -> Dict[str, Any]:
        """Handle chief complaint extraction and load relevant guidelines"""
        if not self.llm_chat_fn:
            chief_complaint = message.lower()
        else:
            # Extract chief complaint using LLM
            llm_kwargs = self._get_llm_kwargs(override_max_tokens=50, override_temperature=0.1)
            response = self.llm_chat_fn(
                [{"role": "system", "content": self.LLM_CHIEF_COMPLAINT_SYSTEM_MSG}, {"role": "user", "content": message}],
                **llm_kwargs
            )
            
            chief_complaint = response.strip().lower()
            if chief_complaint == 'none' or not chief_complaint:
                chief_complaint = message.lower()
        
        session.context['chief_complaint'] = chief_complaint
        self._capture_debug(f"[Navigator] 🩺 Chief Complaint: {chief_complaint}")
        
        # Match chief complaint to categories by semantically comparing to all triggers
        matched_categories = self._match_chief_complaint_to_categories(chief_complaint)
        
        if matched_categories:
            # Semantic match found - load matching guidelines and activate category indexes
            self._load_guidelines_for_categories(matched_categories, chief_complaint)
            session.context['matched_categories'] = matched_categories
            session.context['use_general_llm'] = False
            self._capture_debug(f"[Navigator] ✅ Using guideline-based assessment for categories: {matched_categories}")
            self._capture_debug(f"[Navigator] 📋 Loaded {len(self.all_guidelines)} matching guidelines")
        else:
            # No semantic match - use general LLM knowledge with OLDCARTS structure
            session.context['matched_categories'] = []
            session.context['use_general_llm'] = True
            self._capture_debug(f"[Navigator] 📚 Using general LLM knowledge with OLDCARTS structure (no guideline match)")
        
        # Generate empathetic response (mimic adaptive engine's approach)
        if self.llm_chat_fn:
            system_msg = "You are a compassionate medical assistant. Generate a brief, empathetic statement acknowledging the patient's concern."
            user_msg = f"Patient reported: '{chief_complaint}'\n\nGenerate a brief, empathetic acknowledgment (1-2 sentences). Acknowledge their concern, show compassion, and express that you're here to help. Do NOT ask questions. End with a period. Return only the statement, no other text."
            
            llm_kwargs = self._get_llm_kwargs()
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, 
                 {"role": "user", "content": user_msg}],
                **llm_kwargs
            )
            
            acknowledgment = response.strip() if response else f"I understand you're experiencing {chief_complaint}."
        else:
            raise ValueError("LLM function required for chief complaint handling")
        
        # Ask chronicity question next (hardcoded for consistency)
        chronicity_question = "Is this a new problem or an ongoing issue you've been dealing with?"
        combined_response = f"{acknowledgment}\n\n{chronicity_question}"
        
        return {
            'response': combined_response,
            'status': 'demographics',
            'metadata': {
                'chief_complaint': chief_complaint,
                'asking': 'chronicity'
            }
        }
    
    def _extract_oldcarts_info(self, session: 'AdvancedMedicalNavigator.MedicalSession', user_message: str):
        """Extract OLDCARTS information from user's answer using LLM"""
        if not self.llm_chat_fn:
            return
        
        # Get the last question asked to understand what we're extracting
        last_question = None
        for msg in reversed(session.messages):
            if msg['role'] == 'assistant':
                last_question = msg['content']
                break
        
        if not last_question:
            return
        
        # Use LLM to determine which OLDCARTS element was answered
        user_msg = f"""Question asked: {last_question}
Patient answered: {user_message}

Which OLDCARTS element was answered? Return ONLY the element name:"""
        
        try:
            llm_kwargs = self._get_llm_kwargs(override_max_tokens=20, override_temperature=0.1)
            response = self.llm_chat_fn(
                [{"role": "system", "content": self.LLM_OLDCARTS_EXTRACTION_SYSTEM_MSG}, {"role": "user", "content": user_msg}],
                **llm_kwargs
            )
            
            element = response.strip().lower()
            if element and element != 'none' and element in session.context['oldcarts_covered']:
                session.context['oldcarts_covered'][element] = True
                self._capture_debug(f"[Navigator] ✅ Extracted {element} from patient answer")
        except Exception as e:
            print(f"[Navigator] ⚠️ Failed to extract OLDCARTS info: {e}")
    
    def _score_patient_answer(self, session: 'AdvancedMedicalNavigator.MedicalSession', patient_answer: str):
        """Score patient answer against guidelines using FAISS semantic matching (element-specific)"""
        if not self.medical_rule_engine:
            return
        
        # Get the last question asked to determine which OLDCARTS element was answered
        last_question = None
        for msg in reversed(session.messages):
            if msg['role'] == 'assistant':
                last_question = msg['content']
                break
        
        if not last_question:
            return
        
        # Determine which OLDCARTS element was asked about
        element = self._infer_oldcarts_element_from_question(last_question)
        if not element:
            return
        
        # Score against ONLY loaded guidelines for this SPECIFIC element (optimized search)
        # FAISS search is element-specific - only searches terms for this OLDCARTS element
        # Category-specific indexes ensure we only search relevant categories (latency optimization)
        threshold = self.FAISS_SEMANTIC_THRESHOLD
        matches = self.medical_rule_engine.find_matching_terms_faiss(
            prompt=patient_answer,
            element=element,  # CRITICAL: Only searches this element's terms (e.g., 'location', 'character')
            threshold=threshold,
            return_scores=True
        )
        
        self._capture_debug(f"[Navigator] 🔍 FAISS search: element='{element}', matches={len(matches)}, categories={session.context.get('matched_categories', [])}")
        
        # Score only loaded guidelines (not all guidelines)
        for condition_name, guideline in self.all_guidelines.items():
            score = self._calculate_guideline_score(guideline, element, patient_answer, matches)
            session.condition_scores[condition_name] += score
    
    def _calculate_guideline_score(self, guideline: Dict, element: str, patient_answer: str, matches: List[str]) -> float:
        """Calculate score for a guideline based on FAISS matches"""
        # Get structured OLDCARTS from guideline
        structured_oldcarts = guideline.get('key_features', {}).get('structured_oldcarts', {})
        element_data = structured_oldcarts.get(element, {})
        
        if not element_data:
            return 0.0
        
        # Check if any patient_friendly terms from guideline match
        includes = element_data.get('includes', [])
        score = 0.0
        
        for term_data in includes:
            if isinstance(term_data, dict):
                patient_friendly = term_data.get('patient_friendly', '').lower()
                if patient_friendly in [m.lower() for m in matches]:
                    score += 1.0  # Match found
            elif isinstance(term_data, str):
                if term_data.lower() in [m.lower() for m in matches]:
                    score += 1.0
        
        return score
    
    def _update_condition_rankings(self, session: 'AdvancedMedicalNavigator.MedicalSession'):
        """Update top 5 condition rankings based on current scores"""
        # Sort conditions by score
        sorted_conditions = sorted(
            session.condition_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top N conditions
        session.condition_rankings = []
        for condition_name, score in sorted_conditions[:self.TOP_CONDITIONS_LIMIT]:
            guideline = self.all_guidelines.get(condition_name)
            if guideline:
                session.condition_rankings.append({
                    'condition': condition_name,
                    'score': score,
                    'guideline': guideline
                })
        
        self._capture_debug(f"[Navigator] 📊 Top conditions: {[r['condition'] for r in session.condition_rankings[:3]]}")
    
    def _handle_assessment(self, session: 'AdvancedMedicalNavigator.MedicalSession', message: str) -> Dict[str, Any]:
        """Handle assessment phase - generate next question based on condition ranking or general LLM"""
        if not self.llm_chat_fn:
            raise ValueError("LLM function required for assessment")
        
        # Log current assessment state
        demographics = session.context.get('demographics', {})
        oldcarts_covered = session.context.get('oldcarts_covered', {})
        missing_oldcarts = self._get_missing_oldcarts_info(session)
        covered_oldcarts = [elem for elem, val in oldcarts_covered.items() if val]
        
        self._capture_debug(f"[Navigator] 📊 Assessment Status:")
        self._capture_debug(f"[Navigator]   Demographics: chronicity={demographics.get('chronicity', '?')}, age={demographics.get('age', '?')}, sex={demographics.get('sex', '?')}")
        self._capture_debug(f"[Navigator]   OLDCARTS Covered ({len(covered_oldcarts)}/9): {', '.join(covered_oldcarts) if covered_oldcarts else 'none'}")
        self._capture_debug(f"[Navigator]   OLDCARTS Missing ({len(missing_oldcarts)}/9): {', '.join(missing_oldcarts) if missing_oldcarts else 'none'}")
        
        use_general_llm = session.context.get('use_general_llm', False)
        chief_complaint = session.context.get('chief_complaint', 'symptoms')
        
        if use_general_llm:
            # Use general LLM knowledge with OLDCARTS structure (no guidelines)
            next_element, examples = self._select_best_next_question(session)
            
            if not next_element:
                # All information gathered - provide summary
                next_question = self._generate_assessment_summary(session)
            else:
                # Get generic template examples (style guide)
                template_examples = self._get_oldcarts_examples(chief_complaint, next_element)
                
                # Use LLM to generate next question with OLDCARTS structure
                system_msg = self.LLM_ASSESSMENT_SYSTEM_MSG_TEMPLATE.format(next_element=next_element)
                
                # Build conversation context
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in session.get_recent_messages(8)
                ])
                
                user_msg = f"""PATIENT'S ACTUAL CHIEF COMPLAINT: {chief_complaint}

IMPORTANT: Ask about {next_element} for the patient's {chief_complaint}, NOT about any topics from the examples below.

Information already gathered from this patient:
{self._format_covered_info(session)}

Next element to ask about: {next_element}

STYLE GUIDE ONLY (use these for HOW to format your question, NOT WHAT to ask about):
{template_examples}

Recent conversation with this patient:
{conversation_text}

Generate a question about {next_element} specifically for this patient's {chief_complaint}. 
CRITICAL: Use the patient's actual complaint ({chief_complaint}), not any topics from the examples.
Be conversational and specific to their situation."""
                
                llm_kwargs = self._get_llm_kwargs(override_max_tokens=100)
                response = self.llm_chat_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    **llm_kwargs
                )
                
                next_question = response.strip()
                if not next_question or len(next_question) < 10:
                    raise ValueError(f"LLM failed to generate valid question for element: {next_element}")
        else:
            # Use guideline-based assessment with condition rankings
            next_element, examples = self._select_best_next_question(session)
            
            if not next_element:
                # All information gathered - provide summary
                next_question = self._generate_assessment_summary(session)
            else:
                # Get generic template examples (style guide)
                template_examples = self._get_oldcarts_examples(chief_complaint, next_element)
                
                # Use LLM to generate next question with structured guidance
                system_msg = self.LLM_ASSESSMENT_SYSTEM_MSG_TEMPLATE.format(next_element=next_element)
                
                # Build conversation context
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in session.get_recent_messages(8)
                ])
                
                # Build context about top conditions for differentiation
                top_conditions_context = ""
                if session.condition_rankings:
                    top_conditions_context = "\n\nTop conditions being considered:\n"
                    for i, ranking in enumerate(session.condition_rankings[:3], 1):
                        top_conditions_context += f"{i}. {ranking['condition']} (score: {ranking['score']:.2f})\n"
                
                user_msg = f"""PATIENT'S ACTUAL CHIEF COMPLAINT: {chief_complaint}

IMPORTANT: Ask about {next_element} for the patient's {chief_complaint}, NOT about any topics from the examples below.

Information already gathered from this patient:
{self._format_covered_info(session)}

Next element to ask about: {next_element}

STYLE GUIDE ONLY (use these for HOW to format your question, NOT WHAT to ask about):
{template_examples}
{top_conditions_context}

Recent conversation with this patient:
{conversation_text}

Generate a question about {next_element} specifically for this patient's {chief_complaint}. 
CRITICAL: Use the patient's actual complaint ({chief_complaint}), not any topics from the examples.
Be conversational and specific to their situation."""
                
                llm_kwargs = self._get_llm_kwargs(override_max_tokens=100)
                response = self.llm_chat_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    **llm_kwargs
                )
                
                next_question = response.strip()
                if not next_question or len(next_question) < 10:
                    raise ValueError(f"LLM failed to generate valid question for element: {next_element}")
        
        # Get missing info for metadata
        missing_info = self._get_missing_oldcarts_info(session)
        
        return {
            'response': next_question,
            'status': 'assessment',
            'metadata': {
                'conversation_length': len(session.messages),
                'missing_info': missing_info,
                'next_element': missing_info[0] if missing_info else None
            }
        }
    
    def _select_best_next_question(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> tuple:
        """Select best next question based on condition rankings"""
        # Get missing OLDCARTS elements
        missing_info = self._get_missing_oldcarts_info(session)
        
        if not missing_info:
            return None, None
        
        # If we have condition rankings, find element that best differentiates top conditions
        if session.condition_rankings and len(session.condition_rankings) >= 2:
            # Find OLDCARTS element that differs most between top 2 conditions
            differentiating_element = self._find_differentiating_element(
                session.condition_rankings[:2],
                missing_info
            )
            if differentiating_element:
                chief_complaint = session.context.get('chief_complaint', 'symptoms')
                examples = self._get_oldcarts_examples(chief_complaint, differentiating_element)
                return differentiating_element, examples
        
        # Use priority order if no differentiating element found
        next_element = missing_info[0]
        chief_complaint = session.context.get('chief_complaint', 'symptoms')
        examples = self._get_oldcarts_examples(chief_complaint, next_element)
        return next_element, examples
    
    def _find_differentiating_element(self, top_conditions: List[Dict], missing_elements: List[str]) -> Optional[str]:
        """Find OLDCARTS element that best differentiates between top conditions"""
        if len(top_conditions) < 2:
            return None
        
        condition1 = top_conditions[0]['guideline']
        condition2 = top_conditions[1]['guideline']
        
        # Compare structured OLDCARTS between conditions
        oldcarts1 = condition1.get('key_features', {}).get('structured_oldcarts', {})
        oldcarts2 = condition2.get('key_features', {}).get('structured_oldcarts', {})
        
        # Find element where they differ most
        for element in missing_elements:
            data1 = oldcarts1.get(element, {}).get('includes', [])
            data2 = oldcarts2.get(element, {}).get('includes', [])
            
            # Extract patient_friendly terms
            terms1 = set()
            terms2 = set()
            
            for term in data1:
                if isinstance(term, dict):
                    terms1.add(term.get('patient_friendly', '').lower())
                elif isinstance(term, str):
                    terms1.add(term.lower())
            
            for term in data2:
                if isinstance(term, dict):
                    terms2.add(term.get('patient_friendly', '').lower())
                elif isinstance(term, str):
                    terms2.add(term.lower())
            
            # If they differ significantly, this is a good differentiating element
            if terms1 and terms2 and not terms1.intersection(terms2):
                return element
        
        return None
    
    def _generate_assessment_summary(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> str:
        """Generate assessment summary with top conditions"""
        if not session.condition_rankings:
            return "Thank you for providing all that information. I'll review everything and get back to you."
        
        top_condition = session.condition_rankings[0]
        return f"Based on your symptoms, the most likely condition is {top_condition['condition']}. However, this is not a diagnosis - please consult with a healthcare provider for proper evaluation."
    
    def _handle_followup(self, session: 'AdvancedMedicalNavigator.MedicalSession', message: str) -> Dict[str, Any]:
        """Handle follow-up questions"""
        return self._handle_assessment(session, message)
    
    # ============================================================================
    # SECTION 7: UTILITIES - Helper Functions
    # ============================================================================
    
    # ===== SESSION MANAGEMENT CLASS =====
    
    class MedicalSession:
        """Session management for medical conversations with condition ranking"""
        
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.messages = []  # Conversation history
            self.context = {
                'chief_complaint': None,
                'demographics': {},
                'symptoms': [],
                'questions_asked': [],
                # Track what clinical information has been gathered
                'oldcarts_covered': {
                    'onset': False,      # When did it start?
                    'location': False,   # Where is it?
                    'duration': False,    # How long?
                    'character': False,   # What does it feel like?
                    'aggravating': False, # What makes it worse?
                    'relieving': False,   # What makes it better?
                    'timing': False,      # Constant or intermittent?
                    'severity': False,    # How bad is it (1-10)?
                    'associated': False   # Any other symptoms?
                },
                # Store extracted OLDCARTS data
                'oldcarts_data': {
                    'onset': None,
                    'location': None,
                    'duration': None,
                    'character': None,
                    'aggravating': None,
                    'relieving': None,
                    'timing': None,
                    'severity': None,
                    'associated': None
                }
            }
            # Condition ranking (top 5, updated after each question)
            self.condition_rankings: List[Dict[str, Any]] = []  # [{condition, score, guideline}, ...]
            self.condition_scores: Dict[str, float] = defaultdict(float)  # Track scores per condition
            self.created_at = datetime.now()
        
        def add_message(self, role: str, content: str):
            """Add message to conversation"""
            self.messages.append({
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
        
        def get_conversation_summary(self) -> str:
            """Get a brief summary of the conversation"""
            if not self.messages:
                return "New conversation"
            
            summary_parts = []
            if self.context.get('chief_complaint'):
                summary_parts.append(f"Chief complaint: {self.context['chief_complaint']}")
            if self.context.get('demographics'):
                demo = self.context['demographics']
                if demo.get('age'):
                    summary_parts.append(f"Age: {demo['age']}")
                if demo.get('sex'):
                    summary_parts.append(f"Sex: {demo['sex']}")
            
            return ". ".join(summary_parts) if summary_parts else "New conversation"
        
        def get_recent_messages(self, n: int = 6) -> List[Dict]:
            """Get last N messages for context"""
            return self.messages[-n:]
    
    # ===== SESSION MANAGEMENT FUNCTIONS =====
    
    def _get_or_create_session(self, session_id: str) -> 'AdvancedMedicalNavigator.MedicalSession':
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self.MedicalSession(session_id)
        return self.sessions[session_id]
    
    def _get_missing_oldcarts_info(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> List[str]:
        """Get list of OLDCARTS elements that haven't been covered yet"""
        oldcarts = session.context.get('oldcarts_covered', {})
        missing = []
        
        # Use priority order
        for element in self.OLDCARTS_PRIORITY_ORDER:
            if not oldcarts.get(element, False):
                missing.append(element)
        
        return missing
    
    def _format_covered_info(self, session: 'AdvancedMedicalNavigator.MedicalSession') -> str:
        """Format what information has been covered"""
        oldcarts = session.context.get('oldcarts_covered', {})
        covered = []
        
        element_names = {
            'onset': 'When it started',
            'location': 'Location',
            'duration': 'Duration',
            'character': 'Character/description',
            'aggravating': 'What makes it worse',
            'relieving': 'What makes it better',
            'timing': 'Timing (constant/intermittent)',
            'severity': 'Severity',
            'associated': 'Associated symptoms'
        }
        
        for element, covered_bool in oldcarts.items():
            if covered_bool:
                covered.append(element_names.get(element, element))
        
        if covered:
            return ", ".join(covered)
        return "None yet"
    
    def _infer_oldcarts_element_from_question(self, question: str) -> Optional[str]:
        """Infer which OLDCARTS element a question is asking about"""
        question_lower = question.lower()
        
        # Simple keyword matching
        if any(word in question_lower for word in ['where', 'location', 'point']):
            return 'location'
        elif any(word in question_lower for word in ['when', 'start', 'begin', 'onset']):
            return 'onset'
        elif any(word in question_lower for word in ['how long', 'duration', 'last']):
            return 'duration'
        elif any(word in question_lower for word in ['describe', 'feel', 'character', 'what does it']):
            return 'character'
        elif any(word in question_lower for word in ['worse', 'aggravating', 'makes it worse']):
            return 'aggravating'
        elif any(word in question_lower for word in ['better', 'relieving', 'helps', 'relief']):
            return 'relieving'
        elif any(word in question_lower for word in ['constant', 'intermittent', 'come and go', 'timing']):
            return 'timing'
        elif any(word in question_lower for word in ['scale', '1 to 10', 'severe', 'severity', 'how bad']):
            return 'severity'
        elif any(word in question_lower for word in ['other', 'associated', 'additional', 'else']):
            return 'associated'
        
        return None
    
    def _get_oldcarts_examples(self, chief_complaint: str, element: str) -> str:
        """
        Get comprehensive template examples for question formatting (style guide).
        
        Returns multiple examples from different medical systems to help the LLM
        understand how to adapt questions to different types of complaints.
        """
        templates = self.OLDCARTS_QUESTION_TEMPLATES.get(element, [])
        
        if not templates:
            return f'Ask about {element} in a natural, conversational way.'
        
        # Return all examples formatted as a list for the LLM to see patterns
        # The LLM will use these as style guides to generate similar questions
        examples_text = "Here are example questions from different medical systems (use as style guide):\n"
        for i, example in enumerate(templates, 1):
            examples_text += f"{i}. {example}\n"
        
        return examples_text.strip()
    
    def reset_session(self, session_id: str):
        """Reset a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"[Navigator] 🔄 Session {session_id} reset")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'message_count': len(session.messages),
            'context': session.context,
            'created_at': session.created_at.isoformat()
        }
    
    # ============================================================================
    # SECTION 8: DEBUGGING - Debug Functions (Last)
    # ============================================================================
    
    def _capture_debug(self, message: str):
        """Capture debug output"""
        self._captured_debug_output.append(message)
        print(message)
    
    def _get_debug_info(self, session_id: str = None, last_answer: str = None) -> Dict:
        """Build debug information"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            num_questions = len([msg for msg in session.messages if msg['role'] == 'assistant'])
            covered_count = sum(session.context['oldcarts_covered'].values())
            coverage_str = ''.join([k[0].upper() if v else '_' for k, v in session.context['oldcarts_covered'].items()])
            
            return {
                'demographics': session.context.get('demographics', {}),
                'question_number': num_questions,
                'oldcarts_coverage': coverage_str,
                'active_differentials': [
                    {'rank': i+1, 'name': r['condition'], 'score': r['score']}
                    for i, r in enumerate(session.condition_rankings[:5])
                ],
                'chief_complaint': session.context.get('chief_complaint'),
                'matched_categories': session.context.get('matched_categories', [])
            }
        return {}
    
    def _format_engine_debug(self, session_id: str = None, prefix_note: str = None) -> str:
        """Return formatted debug banner similar to Telegram output."""
        lines = []
        lines.append("="*80)
        lines.append("[Navigator] 🧠 ENGINE DEBUG OUTPUT")
        lines.append("="*80)
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            lines.append(f"[Navigator] 🎯 Loaded guidelines: {len(self.all_guidelines)}")
            if prefix_note:
                lines.append(prefix_note)
            
            # OLDCARTS coverage
            coverage_str = ''.join([k[0].upper() if v else '_' for k, v in session.context['oldcarts_covered'].items()])
            covered_count = sum(session.context['oldcarts_covered'].values())
            lines.append(f"[Navigator] 📋 OLDCARTS: {coverage_str} ({covered_count}/9)")
            
            # Top conditions
            lines.append("[Navigator] 📊 TOP CONDITIONS:")
            for i, ranking in enumerate(session.condition_rankings[:5], start=1):
                condition_name = ranking.get('condition', 'Unknown')
                score = ranking.get('score', 0.0)
                score_pct = round(score * 10, 1) if score > 0 else 0.0  # Convert to percentage-like display
                urgency = ranking.get('guideline', {}).get('urgency', 'routine')
                sev_icon = '🚨' if 'emerg' in str(urgency).lower() else '⚠️' if 'urgent' in str(urgency).lower() else '📋'
                lines.append(f"  {i}. {condition_name}: {score_pct}% ({urgency}) {sev_icon}")
            
            lines.append(f"[Navigator] 🔄 Categories active: {', '.join(session.context.get('matched_categories', []))}")
        else:
            lines.append("[Navigator] ⚠️ No active session")
        
        return "\n".join(lines)
    
    def _format_rankings_debug(self, session_id: str = None) -> str:
        """Return formatted UPDATED RANKINGS block and statistics."""
        def urgency_icon(u):
            u_str = str(u or 'routine').lower()
            if 'emerg' in u_str:
                return '🚨'
            if 'urgent' in u_str:
                return '⚠️'
            return '📋'
        
        lines = []
        lines.append("[Navigator] 📊 UPDATED RANKINGS:")
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            for i, ranking in enumerate(session.condition_rankings[:5], start=1):
                condition_name = ranking.get('condition', 'Unknown')
                score = ranking.get('score', 0.0)
                score_pct = round(score * 10, 1) if score > 0 else 0.0
                guideline = ranking.get('guideline', {})
                urgency = guideline.get('urgency', 'routine')
                prevalence = guideline.get('prevalence', 'unknown')
                icon = urgency_icon(urgency)
                
                lines.append(f"[Navigator]   {i}. {condition_name}: {score_pct}% {icon}")
                lines.append(f"[Scoring] 🏆 Top {i}: {condition_name}")
                lines.append(f"[Scoring]   📊 Score: {score_pct}%")
                lines.append(f"[Scoring]   📋 Prevalence: {prevalence}")
                lines.append(f"[Scoring]   🎯 ML Confidence: High similarity match")
                lines.append(f"[Scoring]   🚨 Urgency: {urgency}")
            
            lines.append("")
            lines.append(f"[Navigator] 🔄 Loaded guidelines: {len(self.all_guidelines)}")
            lines.append("[Scoring] 📊 Final statistics:")
            lines.append(f"[Scoring]   🎯 Active Conditions: {len(session.condition_rankings)}")
            lines.append(f"[Scoring]   📋 Categories: {', '.join(session.context.get('matched_categories', []))}")
        else:
            lines.append("[Navigator] ⚠️ No active session")
        
        return "\n".join(lines)
    
    def get_debug_output(self) -> List[str]:
        """Get captured debug output"""
        return self._captured_debug_output.copy()
    
    def clear_debug_output(self):
        """Clear captured debug output"""
        self._captured_debug_output = []
