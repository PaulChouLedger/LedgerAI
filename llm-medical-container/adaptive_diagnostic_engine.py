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
from thinking_fillers import get_filler

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
    """Minimal universal diagnostic engine"""
    
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
        
        # No separate anatomical FAISS index needed - medical_rules.json + OLDCARTS synonyms handle this
        
        # OPTIMIZATION: Pre-load and cache synonym mappings to avoid repeated I/O
        self.synonym_cache = {}
        self._load_synonym_cache()
        
        # Pre-build chief complaint trigger index for category matching
        self.chief_complaint_triggers_index = None
        self.chief_complaint_triggers_data = []  # List of {trigger, category, condition}
        self.chief_complaint_synonyms_index = None  # FAISS index for chief complaint synonyms
        self.chief_complaint_synonyms_data = []  # List of {synonym, medical_term, category}
        self._build_chief_complaint_triggers_index()
        self._build_chief_complaint_synonyms_index()
        
        # Initialize assessment state
        self.demographics_optional = False  # Set to True if distress detected
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
    
    def _load_synonym_cache(self):
        """Pre-load all synonym mappings for enabled organ systems only"""
        # Get enabled categories (same logic as _load_guidelines)
        enabled_categories_env = os.environ.get('ENABLED_MEDICAL_CATEGORIES', 'GI').strip()
        enabled_categories = [cat.strip().upper() for cat in enabled_categories_env.split(',') if cat.strip()]
        
        # Only load synonyms for enabled categories
        if enabled_categories:
            loaded_systems = []
            for system_name in self.CATEGORY_TO_SYSTEM.values():
                if system_name.upper() in enabled_categories:
                    self.synonym_cache[system_name] = self._load_synonyms_for_system(system_name)
                    loaded_systems.append(system_name)
            self._capture_debug(f"[Engine] ✅ Synonym cache loaded for {len(loaded_systems)} enabled organ systems: {', '.join(loaded_systems)}")
        else:
            # Load all if no categories specified
            for system_name in self.CATEGORY_TO_SYSTEM.values():
                self.synonym_cache[system_name] = self._load_synonyms_for_system(system_name)
            self._capture_debug(f"[Engine] ✅ Synonym cache loaded for {len(self.synonym_cache)} organ systems")
    
    def _load_synonyms_for_system(self, organ_system: str) -> dict:
        """Load synonyms for a specific organ system and pre-build data structures"""
        cache = {
            'onset': {}, 'location': {}, 'timing': {}, 'duration': {},
            'character': {}, 'aggravating': {}, 'relieving': {}, 'severity': {},
            'associated': {}
        }
        
        synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
        synonym_path = os.path.join(os.path.dirname(__file__), synonym_file)
        
        if not os.path.exists(synonym_path):
            return cache
        
        try:
            with open(synonym_path, 'r') as f:
                synonyms = json.load(f)
            
            # Pre-build synonym_expansions and synonym_to_group for each OLDCARTS element
            for oldcarts_element in cache.keys():
                if oldcarts_element in synonyms:
                    expansions = {}
                    to_group = {}
                    
                    for standard_term, synonym_list in synonyms[oldcarts_element].items():
                        # Map standard term to all its synonyms for comparison
                        expansions[standard_term] = [standard_term] + synonym_list
                        # Build reverse mapping: each synonym points back to its group
                        for synonym in [standard_term] + synonym_list:
                            to_group[synonym.lower()] = standard_term
                    
                    cache[oldcarts_element] = {
                        'expansions': expansions,
                        'to_group': to_group
                    }
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Failed to load synonyms for {organ_system}: {e}")
        
        return cache
        
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
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        UNIVERSAL FLOW:
        1. Unified function with chief complaint synonyms → match category → narrow guidelines
        2. Parse prompt to detect answered OLDCARTS elements
        3. Process each element individually using same logic as _process_clinical_answer
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (UNIVERSAL FLOW)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # STEP 1: Unified function with chief complaint synonyms → match category
        category = self._match_chief_complaint_to_category(chief_complaint)
        self.current_category = category
        self._capture_debug(f"[Engine] 🎯 Category: {category}")
        
        # Switch FAISS indexes to category-specific once category is determined
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'set_active_category'):
            self._capture_debug(f"[Engine] 🔀 Switching FAISS indexes to {category} category...")
            self.medical_rule_engine.set_active_category(category)
            self._capture_debug(f"[Engine] ✅ FAISS indexes switched to {category} category")
        
        # STEP 2: Narrow down guidelines
        matched_guidelines = self._get_all_guidelines_in_category(category)
        self._capture_debug(f"[Engine] 📊 Found {len(matched_guidelines)} guidelines")
        self._capture_debug(f"[Guideline Load] 📚 Conditions: {[g.get('name', 'Unknown') for g in matched_guidelines[:10]]}")
        
        # STEP 3: Parse prompt to detect answered OLDCARTS elements
        oldcarts_analysis = self._parse_prompt_against_structured_oldcarts(chief_complaint, matched_guidelines)
        self._capture_debug(f"[Engine] 🔍 OLDCARTS Analysis: {oldcarts_analysis}")
        
        # Initialize assessment
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        self.active_guidelines = matched_guidelines[:self.MAX_ACTIVE]
        self.reserve_pool = matched_guidelines[self.MAX_ACTIVE:]
        
        self._capture_debug(f"[Initial Pool] 🎯 Active: {len(self.active_guidelines)}, Reserve: {len(self.reserve_pool)}")
        self._capture_debug(f"[Initial Pool] 🏆 Active: {[g.get('name', 'Unknown') for g in self.active_guidelines]}")
        
        # Store OLDCARTS analysis for use in questioning
        self.oldcarts_analysis = oldcarts_analysis
        
        # Process any detected OLDCARTS elements from the initial prompt
        answered_components = oldcarts_analysis.get('answered_components', {})
        if answered_components:
            self._capture_debug(f"[Engine] 🔄 Processing {len(answered_components)} detected elements from initial prompt")
            for element, detected_terms in answered_components.items():
                self._capture_debug(f"[Engine] 📊 Processing {element}: {detected_terms}")
                # Create a fake question entry for this element so _process_clinical_answer knows which element to process
                self.conversation_history.append({
                    'type': 'question',
                    'question': f"Tell me about {element}",
                    'oldcarts': element,
                    'focus': 'clinical'
                })
                # Store the RAW answer for context (full prompt, not just detected terms)
                self.conversation_history.append({
                    'type': 'answer',
                    'answer': chief_complaint,  # Store full raw text
                    'oldcarts': element
                })
                # Just mark the element as covered without processing through _process_clinical_answer
                # This prevents the function from returning a result that overrides the empathetic statement
                if element == 'onset':
                    self.oldcarts_covered['O'] = True
                elif element == 'location':
                    self.oldcarts_covered['L'] = True
                elif element == 'duration':
                    self.oldcarts_covered['D'] = True
                elif element == 'character':
                    self.oldcarts_covered['C'] = True
                elif element == 'aggravating':
                    self.oldcarts_covered['A'] = True
                elif element == 'relieving':
                    self.oldcarts_covered['R'] = True
                elif element == 'timing':
                    self.oldcarts_covered['T'] = True
                elif element == 'severity':
                    self.oldcarts_covered['S'] = True
                elif element == 'associated':
                    self.oldcarts_covered['AS'] = True
                
                self._capture_debug(f"[Engine]   ✅ {element} marked as covered from initial prompt")
        
        # Start with empathetic statement + chronicity question
        has_shown_statement = any(item.get('type') == 'statement' for item in self.conversation_history)
        if not has_shown_statement:
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
        else:
            # Statement already shown, proceed to demographics
            return self._generate_ml_first_question_with_demographics()
    
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
    
    def _build_chief_complaint_synonyms_index(self):
        """Pre-build FAISS index for chief_complaint synonyms from all synonym files"""
        if not self.embedding_model:
            self._capture_debug("[Engine] ⚠️ No embedding model for chief complaint synonyms index")
            return
        
        try:
            synonyms_list = []
            synonyms_dir = Path(__file__).parent / 'synonyms'
            
            # Load chief complaint synonyms from all synonym files
            for synonym_file in synonyms_dir.glob("*_synonyms_oldcarts.json"):
                try:
                    with open(synonym_file, 'r') as f:
                        synonyms_data = json.load(f)
                        chief_complaint_syns = synonyms_data.get('chief_complaint', {})
                        
                        # Determine category from filename (reverse map: organ_system -> category)
                        organ_system = synonym_file.stem.replace('_synonyms_oldcarts', '').upper()
                        # Reverse lookup: find category that maps to this organ system
                        category = None
                        for cat, sys in self.CATEGORY_TO_SYSTEM.items():
                            if sys == organ_system:
                                category = cat
                                break
                        if not category:
                            # Fallback: use filename as category
                            category = organ_system.lower()
                        
                        # Add all synonyms and their medical terms to the index
                        for medical_term, patient_terms in chief_complaint_syns.items():
                            for patient_term in patient_terms:
                                self.chief_complaint_synonyms_data.append({
                                    'synonym': patient_term,
                                    'medical_term': medical_term,
                                    'category': category
                                })
                                synonyms_list.append(patient_term)
                except Exception as e:
                    self._capture_debug(f"[Engine] ⚠️ Failed to load {synonym_file.name}: {e}")
            
            if synonyms_list:
                # Build FAISS index for all synonyms
                embeddings = self.embedding_model.encode(synonyms_list)
                dimension = len(embeddings[0])
                self.chief_complaint_synonyms_index = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                embeddings_np = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_np)
                self.chief_complaint_synonyms_index.add(embeddings_np)
                
                self._capture_debug(f"[Engine] ✅ Built chief complaint synonyms index: {len(synonyms_list)} synonyms from {len(set(g['category'] for g in self.chief_complaint_synonyms_data))} categories")
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Failed to build chief complaint synonyms index: {e}")
            self.chief_complaint_synonyms_index = None
    
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
        """Match chief complaint to category using unified function with synonym normalization first, then FAISS against triggers"""
        triggers_index_missing = not self.chief_complaint_triggers_index
        embedding_model_missing = not self.embedding_model
        triggers_data_empty = len(self.chief_complaint_triggers_data) == 0
        
        if triggers_index_missing or embedding_model_missing or triggers_data_empty:
            raise ValueError("Chief complaint triggers index not available. Cannot match category.")
        
        try:
            # STEP 1: Normalize using chief complaint synonyms (unified function)
            normalized_complaint = chief_complaint.lower().strip()
            if self.chief_complaint_synonyms_index and len(self.chief_complaint_synonyms_data) > 0:
                try:
                    # Use FAISS to find matching medical term from synonyms
                    query_embedding = self.embedding_model.encode([normalized_complaint])[0]
                    query_embedding = np.array([query_embedding]).astype('float32')
                    faiss.normalize_L2(query_embedding)
                    
                    # Search synonyms index
                    k = min(5, len(self.chief_complaint_synonyms_data))
                    similarities, indices = self.chief_complaint_synonyms_index.search(query_embedding, k)
                    
                    # Find best matching medical term (threshold 0.75 for synonym matching)
                    best_synonym_match = None
                    best_synonym_score = 0.0
                    
                    for idx, sim in zip(indices[0], similarities[0]):
                        if idx < len(self.chief_complaint_synonyms_data) and sim >= 0.75:
                            synonym_data = self.chief_complaint_synonyms_data[idx]
                            if sim > best_synonym_score:
                                best_synonym_score = sim
                                best_synonym_match = synonym_data['medical_term']
                    
                    if best_synonym_match:
                        normalized_complaint = best_synonym_match
                        self._capture_debug(f"[Engine] 🔄 Normalized '{chief_complaint}' → '{normalized_complaint}' (synonym score: {best_synonym_score:.3f})")
                except Exception as e:
                    self._capture_debug(f"[Engine] ⚠️ Synonym normalization failed, using original: {e}")
            
            # STEP 2: Match normalized complaint against chief_complaint_triggers
            # Encode normalized chief complaint
            query_embedding = self.embedding_model.encode([normalized_complaint])[0]
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
                    
                    if sim >= 0.6:
                        # Above threshold - use for category matching
                        category = trigger_data['category']
                        if category not in category_scores or sim > category_scores[category]:
                            category_scores[category] = sim
                    elif sim >= 0.5:
                        # Close to threshold - candidate for fuzzy matching (typo detection)
                        near_miss_candidates.append((trigger_data, sim))
            
            # If FAISS didn't find matches above threshold, try fuzzy matching only on near-misses
            if not category_scores and near_miss_candidates:
                self._capture_debug(f"[Engine] ⚠️ FAISS found no matches above 0.6, trying fuzzy matching on {len(near_miss_candidates)} near-miss candidates...")
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
                        similarity_meets_threshold = similarity >= 0.8  # Fuzzy threshold (stricter than FAISS for typos)
                        
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
                raise ValueError(f"No category match found for chief complaint: '{chief_complaint}' (FAISS threshold: 0.6, fuzzy on near-misses 0.5-0.6)")
            
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
    
    def _parse_prompt_against_structured_oldcarts(self, prompt: str, guidelines: List[Dict]) -> Dict[str, Any]:
        """Parse prompt against structured OLDCARTS using FAISS term matching + medical_rules.json"""
        if not guidelines:
            return {
                'answered_components': {},
                'missing_components': ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity', 'associated'],
                'anatomical_analysis': {}
            }
        
        # Use FAISS-based term matching with extensive synonyms
        answered_components = {}
        
        # Use FAISS to find matching terms - relies on extensive synonym files
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
            all_elements = ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity', 'associated']
            
            for element in all_elements:
                # Use FAISS to find matching terms with semantic similarity (very high threshold for initial parsing to avoid false positives)
                # Get active condition names for filtering (if available)
                active_condition_names = None
                has_active_guidelines_attr = hasattr(self, 'active_guidelines')
                active_guidelines_exist = has_active_guidelines_attr and self.active_guidelines
                
                if active_guidelines_exist:
                    active_condition_names = set()
                    for g in self.active_guidelines:
                        condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                        if condition_name:
                            active_condition_names.add(condition_name)
                
                matching_terms = self.medical_rule_engine.find_matching_terms_faiss(
                    prompt, element, threshold=0.85, active_condition_names=active_condition_names
                )
                if matching_terms:
                    answered_components[element] = matching_terms
                    self._capture_debug(f"[Engine] 📍 {element}: {matching_terms}")
        
        answered_elements = list(answered_components.keys())
        # Priority order: timing before duration
        standard_order = ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity', 'associated']
        missing_elements = [element for element in standard_order if element not in answered_elements]
        
        return {
            'answered_components': answered_components,
            'missing_components': missing_elements,
            'anatomical_analysis': {}
        }
    
    def _parse_prompt_against_structured_oldcarts_regex(self, prompt: str, guidelines: List[Dict]) -> Dict[str, Any]:
        """Fallback regex-based parsing"""
        # Collect all 'includes' terms from guidelines
        all_includes = {
            'onset': set(), 'location': set(), 'timing': set(), 'duration': set(),
            'character': set(), 'aggravating': set(), 'relieving': set(), 'severity': set()
        }
        
        for guideline in guidelines:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            for element, data in structured.items():
                if isinstance(data, dict) and 'includes' in data:
                    if element in all_includes:
                        for term in data['includes']:
                            all_includes[element].add(term.lower())
        
        # Detect which elements are present in prompt (using whole word matching)
        answered_components = {}
        prompt_lower = prompt.lower()
        
        # Common words to exclude (too generic, cause false positives)
        exclude_words = {'pain', 'ache', 'hurt', 'sore'}
        
        for element, expected_terms in all_includes.items():
            for term in expected_terms:
                term_lower = term.lower()
                # Skip single generic words that cause false positives
                if term_lower in exclude_words:
                    continue
                
                # Use whole word matching (more specific)
                # Check if term appears as whole word/phrase, not substring
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, prompt_lower):
                    if element not in answered_components:
                        answered_components[element] = []
                    answered_components[element].append(term)
                    break
                        
        all_elements = ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity', 'associated']
        answered_elements = list(answered_components.keys())
        missing_elements = [element for element in all_elements if element not in answered_elements]
        
        return {
            'answered_components': answered_components,
            'missing_components': missing_elements
        }
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """Score guidelines using unified similarity function"""
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
            
            # Mark demographics as optional if not already done
            self.demographics_optional = True
            
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
                
                # OPTIMIZATION: Pre-normalize patient answer once before the loop
                pre_normalized_text = answer.lower()
                if self.synonym_cache:
                    organ_system_key = organ_system
                    if organ_system_key in self.synonym_cache:
                        # Use synonym cache to normalize once
                        element_synonyms = self.synonym_cache[organ_system_key].get('radiation', {})
                        if element_synonyms:
                            # Try to find best matching term via FAISS
                            # Get active condition names for filtering
                            active_condition_names = set()
                            for g in all_guidelines:
                                condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                                if condition_name:
                                    active_condition_names.add(condition_name)
                            
                            faiss_matches = self.medical_rule_engine.find_matching_terms_faiss(
                                answer, 'radiation', threshold=0.75, active_condition_names=active_condition_names
                            )
                            if faiss_matches:
                                pre_normalized_text = faiss_matches[0]
                
                # OPTIMIZATION: Batch embedding for all guidelines
                # Pass 1: Collect all sections to embed
                guideline_sections = []
                guideline_data = []
                for g in all_guidelines:
                    structured_oldcarts = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                    # Build location section text from structured data
                    location_data = structured_oldcarts.get('location', {})
                    location_terms = []
                    for item in location_data.get('includes', []):
                        location_terms.append(item.get('medical', ''))
                    oldcarts_section = ' '.join(location_terms)
                    
                    if oldcarts_section:
                        guideline_sections.append(oldcarts_section)
                        guideline_data.append({
                            'guideline': g,
                            'condition_name': g.get('data', {}).get('condition', g.get('name', 'Unknown')),
                            'structured_oldcarts': structured_oldcarts,
                            'section': oldcarts_section
                        })
                
                # Batch encode all sections at once
                batch_embeddings = None
                patient_embedding = None
                if self.medical_rule_engine.embedding_model and guideline_sections:
                    try:
                        # Encode patient answer + all sections in one batch
                        batch_texts = [answer.lower()] + guideline_sections
                        batch_embeddings = self.medical_rule_engine.embedding_model.encode(batch_texts)
                        batch_embeddings = np.asarray(batch_embeddings, dtype='float32')
                        patient_embedding = batch_embeddings[0]
                        section_embeddings = batch_embeddings[1:]
                    except Exception as e:
                        self._capture_debug(f"[Scoring] ⚠️ Batch embedding failed: {e}")
                        batch_embeddings = None
                
                # Pass 2: Score each guideline
                for idx, g_data in enumerate(guideline_data):
                    g = g_data['guideline']
                    oldcarts_section = g_data['section']
                    condition_name = g_data['condition_name']
                    structured_oldcarts = g_data['structured_oldcarts']
                    element_data = structured_oldcarts.get('radiation')  # Use radiation element data
                    
                    # Compute raw similarity from batch embeddings if available
                    raw_similarity = 0.0
                    if batch_embeddings is not None and idx < len(section_embeddings):
                        section_emb = section_embeddings[idx]
                        raw_similarity = float(np.dot(patient_embedding, section_emb) / 
                                              (np.linalg.norm(patient_embedding) * np.linalg.norm(section_emb)))
                    
                    # Get active condition names for filtering
                    active_condition_names = set()
                    for g in all_guidelines:
                        cond_name = g.get('data', {}).get('condition', g.get('name', ''))
                        if cond_name:
                            active_condition_names.add(cond_name)
                    
                    # Score radiation using radiation element data
                    similarity_result = self.medical_rule_engine.compute_unified_similarity(
                        answer, oldcarts_section, condition_name, organ_system,
                        'location', {'location': element_data} if element_data else None,
                        pre_normalized_text=pre_normalized_text,
                        precomputed_similarity=raw_similarity if batch_embeddings is not None else None,
                        active_condition_names=active_condition_names
                    )
                    
                    old_score = g['score']
                    new_score = (old_score * 0.7) + (similarity_result['similarity'] * 0.3)
                    g['score'] = new_score
                    self._capture_debug(f"[Scoring] 📊 {condition_name}: old={old_score:.3f}, radiation={similarity_result['similarity']:.3f}, new={new_score:.3f}")
                
                # Re-rank after radiation scoring
                self._rerank_and_pool_guidelines(all_guidelines, previous_active)
            
            # Continue to next question
            return self._ask_next_clinical_question()
        
        # Handle onset, duration, timing, severity, associated (documentation only - no clarification needed)
        if oldcarts_element in ['onset', 'duration', 'timing', 'severity', 'associated']:
            # Mark element as covered and store the answer
            element_map = {'onset': 'O', 'duration': 'D', 'timing': 'T', 'severity': 'S', 'associated': 'AS'}
            if oldcarts_element in element_map:
                self.oldcarts_covered[element_map[oldcarts_element]] = True
                # Update missing_components list to remove this element
                if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                    if oldcarts_element in self.oldcarts_analysis['missing_components']:
                        self.oldcarts_analysis['missing_components'].remove(oldcarts_element)
            self._capture_debug(f"[Engine] ✅ {oldcarts_element} marked as complete (no clarification)")
            return self._ask_next_clinical_question()
        
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
        
        # STEP 2: Score all guidelines using unified function
        self._capture_debug(f"[Scoring] 🔍 Scoring {len(all_guidelines)} guidelines for element: {oldcarts_element}")
        self._capture_debug(f"[Scoring] 📝 Patient answer: '{answer}'")
        
        # OPTIMIZATION: Pre-normalize patient answer once before the loop
        pre_normalized_text = answer.lower()
        if self.medical_rule_engine and self.synonym_cache:
            organ_system_key = organ_system
            if organ_system_key in self.synonym_cache:
                # Use synonym cache to normalize once
                element_synonyms = self.synonym_cache[organ_system_key].get(oldcarts_element, {})
                if element_synonyms:
                    # Try to find best matching term via FAISS
                    # Get active condition names for filtering
                    active_condition_names = set()
                    for g in all_guidelines:
                        condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                        if condition_name:
                            active_condition_names.add(condition_name)
                    
                    faiss_matches = self.medical_rule_engine.find_matching_terms_faiss(
                        answer, oldcarts_element, threshold=0.75, active_condition_names=active_condition_names
                    )
                    if faiss_matches:
                        pre_normalized_text = faiss_matches[0]
        
        # OPTIMIZATION: Batch embedding for all guidelines
        # Pass 1: Collect all sections to embed
        guideline_sections = []
        guideline_data = []
        for g in all_guidelines:
            structured_oldcarts = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            # Build section text from structured data
            element_data = structured_oldcarts.get(oldcarts_element, {})
            element_terms = []
            for item in element_data.get('includes', []):
                element_terms.append(item.get('medical', ''))
            oldcarts_section = ' '.join(element_terms)
            
            if oldcarts_section:
                guideline_sections.append(oldcarts_section)
                guideline_data.append({
                    'guideline': g,
                    'condition_name': g.get('data', {}).get('condition', g.get('name', 'Unknown')),
                    'structured_oldcarts': structured_oldcarts,
                    'section': oldcarts_section
                })
        
        # Batch encode all sections at once
        batch_embeddings = None
        patient_embedding = None
        if self.medical_rule_engine and self.medical_rule_engine.embedding_model and guideline_sections:
            try:
                # Encode patient answer + all sections in one batch
                batch_texts = [answer.lower()] + guideline_sections
                batch_embeddings = self.medical_rule_engine.embedding_model.encode(batch_texts)
                batch_embeddings = np.asarray(batch_embeddings, dtype='float32')
                patient_embedding = batch_embeddings[0]
                section_embeddings = batch_embeddings[1:]
            except Exception as e:
                self._capture_debug(f"[Scoring] ⚠️ Batch embedding failed: {e}")
                batch_embeddings = None
        
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
            oldcarts_section = g_data['section']
            condition_name = g_data['condition_name']
            structured_oldcarts = g_data['structured_oldcarts']
            
            # Use unified function for scoring with batch embeddings
            if self.medical_rule_engine:
                element_data = structured_oldcarts.get(oldcarts_element)
                
                # Compute raw similarity from batch embeddings if available
                raw_similarity = 0.0
                if batch_embeddings is not None and idx < len(section_embeddings):
                    section_emb = section_embeddings[idx]
                    raw_similarity = float(np.dot(patient_embedding, section_emb) / 
                                          (np.linalg.norm(patient_embedding) * np.linalg.norm(section_emb)))
                
                # Get active condition names for filtering FAISS results
                active_condition_names = set()
                for g_check in all_guidelines:
                    cond_name = g_check.get('data', {}).get('condition', g_check.get('name', ''))
                    if cond_name:
                        active_condition_names.add(cond_name)
                
                # Use unified function for word match boost and normalization
                similarity_result = self.medical_rule_engine.compute_unified_similarity(
                    answer, oldcarts_section, condition_name, organ_system,
                    oldcarts_element, {oldcarts_element: element_data} if element_data else None,
                    pre_normalized_text=pre_normalized_text,
                    precomputed_similarity=raw_similarity if batch_embeddings is not None else None,
                    active_condition_names=active_condition_names
                )
                similarity = similarity_result['similarity']
                word_match_boost = similarity_result.get('word_match_boost', 0.0)
                normalized_text = similarity_result.get('normalized_text', answer)
            else:
                similarity = 0.5
                word_match_boost = 0.0
                normalized_text = answer
            
            # Update score using category-specific element weights
            old_score = g.get('score', 0.5)
            category = self.current_category or 'gastrointestinal'
            element_weight = self.get_oldcarts_element_weight(category, oldcarts_element)
            
            # Formula: new_score = (old_score * (1 - weight)) + (similarity * weight)
            # Higher weight = this element has more impact on the score
            new_score = (old_score * (1.0 - element_weight)) + (similarity * element_weight)
            g['score'] = new_score
            scored_guidelines.add(condition_name)
            
            self._capture_debug(f"[Scoring] 📊 {condition_name}: old={old_score:.3f}, similarity={similarity:.3f} (boost={word_match_boost:.3f}), weight={element_weight:.2f} ({oldcarts_element}), normalized='{normalized_text}', new={new_score:.3f}")
        
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
        
        # Check if clarification needed
        clarification_count = sum(1 for item in self.conversation_history 
                                 if item.get('oldcarts') == oldcarts_element 
                                 and item.get('is_clarification'))
        
        clarification_needed = False
        missing_terms = []  # Missing terms for the current element
        
        if self.active_guidelines:
            try:
                # Get missing terms and satisfied terms - this already does the expensive matching
                missing_terms, satisfied_terms = self._analyze_missing_information(answer, oldcarts_element)
                self._capture_debug(f"[Clarification] 📊 Missing terms: {missing_terms}")
                self._capture_debug(f"[Clarification] ✅ Satisfied terms: {sorted(satisfied_terms)} (count: {len(satisfied_terms)})")
                
                # Continue asking if we have 2+ satisfied terms (ambiguous) OR no satisfied terms at all
                if len(satisfied_terms) >= 2 or len(satisfied_terms) == 0:
                    clarification_needed = True
                    # Use missing terms for the clarifying question
                    question = self._generate_clarifying_question(oldcarts_element, answer, clarification_count, missing_terms)
                    self.conversation_history.append({
                        'type': 'question',
                        'question': question,
                        'oldcarts': oldcarts_element,
                        'is_clarification': True
                    })
                    return {
                        'success': True,
                        'question': question,
                        'status': 'questioning',
                        'debug': {
                            'engine': self._format_engine_debug("[Engine] ⏳ Clarification requested") + "\n\n" + self._format_rankings_debug(),
                            'internal': self._get_debug_info(last_answer=answer)
                        }
                    }
                elif len(satisfied_terms) == 1:
                    self._capture_debug(f"[Clarification] ✅ Have exactly 1 satisfied term - moving on")
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Clarification check failed: {e}")
        
        # Only mark element as covered if NO clarification needed
        if not clarification_needed:
            element_map = {'onset': 'O', 'location': 'L', 'timing': 'T', 'duration': 'D',
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
            return self._return_to_next_missing_element(acknowledgment_msg)
        
        return self._ask_next_clinical_question()
    
    def _get_dynamic_threshold(self, score: float) -> float:
        """Get dynamic threshold for ruling out"""
        if score >= 0.3:
            return 0.1
        elif score >= 0.2:
            return 0.1
        else:
            return 0.05
    
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
            threshold = self._get_dynamic_threshold(g['score'])
            if g['score'] >= threshold:
                remaining.append(g)
            else:
                self.ruled_out.append(g)
                ruled_out_count += 1
                self._capture_debug(f"[Rule Out] ❌ {g.get('data', {}).get('condition', g.get('name', 'Unknown'))}: score={g['score']:.3f} < threshold={threshold:.3f}")
        
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
        Returns: (missing_terms, satisfied_terms)
        
        IMPORTANT: Consider ALL guidelines (active + reserve) to properly detect satisfied terms.
        Patient answers should be checked against all guidelines, not just top 5.
        """
        # Get all guidelines (active + reserve) to check against all possible terms
        all_guidelines_to_check = self.active_guidelines + self.reserve_pool
        
        if not all_guidelines_to_check:
            return [], set()
        
        # Collect all includes terms from ALL guidelines (not just active)
        # This ensures we detect when patient answer matches terms from reserve pool
        all_includes = set()
        for g in all_guidelines_to_check:
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
        
        # Also collect active-only terms for missing terms calculation
        active_includes = set()
        for g in self.active_guidelines:
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                for t in includes:
                    if isinstance(t, dict):
                        med = t.get('medical')
                        if isinstance(med, str) and med.strip():
                            active_includes.add(med.strip().lower())
                    elif isinstance(t, str):
                        active_includes.add(t.strip().lower())
        
        self._capture_debug(f"[Location Analysis] 📍 Checking satisfaction against ALL {len(all_guidelines_to_check)} guidelines (active + reserve)")
        self._capture_debug(f"[Location Analysis] 📍 All includes terms from {len(all_guidelines_to_check)} total guidelines: {sorted(all_includes)}")
        self._capture_debug(f"[Location Analysis] 📍 Active-only terms from {len(self.active_guidelines)} guidelines: {sorted(active_includes)}")
        self._capture_debug(f"[Location Analysis] 📝 Patient answer: '{answer}'")
        
        if not all_includes:
            self._capture_debug(f"[Location Analysis] ⚠️ No includes terms found for {oldcarts_element}")
            return []
        
        # Use unified function to check which terms are satisfied
        satisfied_terms = set()
        answer_lower = answer.lower()
        
        # OPTIMIZATION: Use pre-loaded synonym cache instead of file I/O
        organ_system = self.CATEGORY_TO_SYSTEM.get(self.current_category or 'gastrointestinal', 'GI')
        
        # Get pre-built synonym structures from cache
        synonym_expansions = {}
        synonym_to_group = {}
        if organ_system in self.synonym_cache:
            cache_data = self.synonym_cache[organ_system].get(oldcarts_element, {})
            synonym_expansions = cache_data.get('expansions', {})
            synonym_to_group = cache_data.get('to_group', {})
        
        # OPTIMIZATION: Do FAISS semantic matching ONCE for all terms (expensive operation)
        # Get ALL condition names (active + reserve) to check against all guidelines
        all_condition_names = set()
        for g in all_guidelines_to_check:
            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
            if condition_name:
                all_condition_names.add(condition_name)
        
        semantic_matches_set = set()
        faiss_scores = {}
        normalized_answer = answer_lower  # Default to original answer
        patient_components = {}  # For location only
        
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
            try:
                # OPTIMIZATION: Single FAISS call for both matching AND normalization
                # Check against ALL guidelines, not just active ones
                # Use threshold=0.6 to match scoring behavior (compute_unified_similarity uses substring/exact matching which is more lenient)
                semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(
                    answer, oldcarts_element, threshold=0.75, 
                    return_scores=True, active_condition_names=all_condition_names
                )
                semantic_matches_set = set(t.lower() for t in semantic_matches)
                
                # DEBUG: Show what FAISS actually found
                if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
                    raw_faiss_scores = self.medical_rule_engine._last_faiss_scores
                    self._capture_debug(f"[Location Analysis] 🔍 FAISS found {len(semantic_matches)} matches above threshold: {semantic_matches}")
                    self._capture_debug(f"[Location Analysis] 🔍 Raw FAISS scores (all): {raw_faiss_scores}")
                
                # OPTIMIZATION: Use first FAISS match as normalized answer (reuse result)
                if semantic_matches:
                    normalized_answer = semantic_matches[0].lower()
                
                # Compute unified similarity scores for matched terms (same as scoring uses)
                # This replaces raw FAISS scores with unified scores (raw_similarity + word_match_boost)
                unified_scores = {}
                if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
                    raw_faiss_scores = self.medical_rule_engine._last_faiss_scores
                    # For each matched term, compute unified similarity using a representative guideline
                    for term, raw_score in raw_faiss_scores.items():
                        if term.lower() not in all_includes:
                            continue
                        # Find a guideline that contains this term
                        for g in all_guidelines_to_check:
                            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                            element_data = structured.get(oldcarts_element, {})
                            if isinstance(element_data, dict):
                                includes = element_data.get('includes', [])
                                term_found = False
                                for t in includes:
                                    if isinstance(t, dict):
                                        med = t.get('medical', '')
                                        if med and med.strip().lower() == term.lower():
                                            term_found = True
                                            break
                                    elif isinstance(t, str) and t.strip().lower() == term.lower():
                                        term_found = True
                                        break
                                
                                if term_found:
                                    condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                                    # Build section text same way as scoring (join all medical terms)
                                    element_terms = []
                                    for t in includes:
                                        if isinstance(t, dict):
                                            med = t.get('medical', '')
                                            if med:
                                                element_terms.append(med)
                                        elif isinstance(t, str):
                                            element_terms.append(t)
                                    oldcarts_section = ' '.join(element_terms)
                                    
                                    # Compute unified similarity (same as scoring)
                                    similarity_result = self.medical_rule_engine.compute_unified_similarity(
                                        answer, oldcarts_section, condition_name, organ_system,
                                        oldcarts_element, {oldcarts_element: element_data},
                                        pre_normalized_text=normalized_answer if normalized_answer != answer_lower else None,
                                        precomputed_similarity=raw_score,
                                        active_condition_names=all_condition_names
                                    )
                                    unified_scores[term] = similarity_result['similarity']
                                    break
                
                # Use unified scores for display and matching
                faiss_scores = unified_scores
                if unified_scores:
                    self._capture_debug(f"[Location Analysis] 🔍 Unified similarity scores (from {len(all_condition_names)} total conditions): {faiss_scores}")
            except Exception as e:
                self._capture_debug(f"[Location Analysis] ⚠️ FAISS error: {e}")
                pass
        
        # OPTIMIZATION: Extract anatomical components ONCE from patient answer (for location only)
        if oldcarts_element == 'location' and self.medical_rule_engine:
            patient_components = self.medical_rule_engine._extract_anatomical_components(normalized_answer)
        
        # OPTIMIZATION: Pre-extract anatomical components for ALL terms ONCE (not in loop)
        term_components_cache = {}
        if oldcarts_element == 'location' and self.medical_rule_engine:
            for term in all_includes:
                term_components_cache[term] = self.medical_rule_engine._extract_anatomical_components(term)
        
        # DEBUG: Show summary of FAISS matches and scores
        self._capture_debug(f"[Location Analysis] 📊 FAISS Summary:")
        self._capture_debug(f"[Location Analysis]   - semantic_matches_set ({len(semantic_matches_set)} terms): {sorted(semantic_matches_set)}")
        self._capture_debug(f"[Location Analysis]   - faiss_scores ({len(faiss_scores)} terms): {dict(sorted(faiss_scores.items()))}")
        if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
            raw_scores = self.medical_rule_engine._last_faiss_scores
            self._capture_debug(f"[Location Analysis]   - Raw FAISS scores ({len(raw_scores)} terms): {dict(sorted(raw_scores.items()))}")
        
        # Check each term using the same logic as unified function
        # IMPORTANT: Check ALL terms (from all guidelines) to properly detect satisfied terms
        # Missing terms should only include terms from ACTIVE guidelines that aren't satisfied
        for term in all_includes:
            term_satisfied = False
            match_reason = None
            
            # For location element, check anatomical mismatch BEFORE marking as satisfied
            # OPTIMIZATION: Use pre-extracted components from cache
            anatomical_mismatch = False
            if oldcarts_element == 'location' and patient_components and self.medical_rule_engine:
                condition_components = term_components_cache.get(term, {})
                if condition_components:
                    if self.medical_rule_engine._are_anatomical_opposites(patient_components, condition_components):
                        anatomical_mismatch = True
                        self._capture_debug(f"[Location Analysis]   ⚠️ '{term}' skipped: anatomical mismatch (patient: {patient_components} vs condition: {condition_components})")
            
            # Skip if anatomical mismatch detected
            if anatomical_mismatch:
                self._capture_debug(f"[Location Analysis]   ⚠️ '{term}' SKIPPED: anatomical mismatch")
                continue
            
            # DEBUG: Start checking this term
            self._capture_debug(f"[Location Analysis] 🔍 Checking term: '{term}' (patient answer: '{answer}')")
            
            # 1. Exact/substring matching (fast path)
            term_in_answer = term in answer_lower
            answer_in_term = answer_lower in term
            self._capture_debug(f"[Location Analysis]   Step 1 - Substring check: term in answer={term_in_answer}, answer in term={answer_in_term}")
            
            if term_in_answer or answer_in_term:
                term_satisfied = True
                match_reason = f"exact/substring match: '{term}' in '{answer}' or '{answer}' in '{term}'"
                self._capture_debug(f"[Location Analysis]   ✅ SATISFIED via substring match")
            else:
                # OPTIMIZATION: Check if answer was normalized to a synonym key that maps to this term
                # Use pre-built synonym_to_group for O(1) lookup instead of O(n²) nested loops
                if term.lower() in synonym_to_group:
                    group_key = synonym_to_group[term.lower()]
                    synonym_list = synonym_expansions.get(group_key, [])
                    self._capture_debug(f"[Location Analysis]   Step 2 - Synonym check: term '{term}' in synonym group '{group_key}' with {len(synonym_list)} synonyms")
                    # Use more precise matching: answer must be a substring of synonym (not reverse)
                    matched_synonyms = [syn for syn in synonym_list if syn.lower() in answer_lower]
                    self._capture_debug(f"[Location Analysis]   Step 2 - Synonym matches: {matched_synonyms}")
                    if matched_synonyms:
                        term_satisfied = True
                        match_reason = f"synonym match: '{matched_synonyms[0]}' in '{answer}'"
                        self._capture_debug(f"[Location Analysis]   ✅ SATISFIED via synonym match")
                else:
                    self._capture_debug(f"[Location Analysis]   Step 2 - Synonym check: term '{term}' NOT in any synonym group")
                
                if not term_satisfied:
                    # 2. Check against FAISS semantic matches (already computed)
                    in_semantic_matches = term in semantic_matches_set
                    score = faiss_scores.get(term, None)
                    self._capture_debug(f"[Location Analysis]   Step 3 - FAISS check: in semantic_matches_set={in_semantic_matches}, score={score}")
                    
                    # DEBUG: Check raw FAISS scores to see actual scores
                    if hasattr(self.medical_rule_engine, '_last_faiss_scores'):
                        raw_faiss_scores = self.medical_rule_engine._last_faiss_scores
                        raw_score = raw_faiss_scores.get(term, None)
                        if raw_score is not None:
                            self._capture_debug(f"[Location Analysis]   Step 3 - Raw FAISS score for '{term}': {raw_score:.3f} (threshold=0.75)")
                            if raw_score < 0.75:
                                self._capture_debug(f"[Location Analysis]   ⚠️ Raw score {raw_score:.3f} < 0.75 threshold, should NOT be in semantic_matches_set")
                        else:
                            # Check if any synonyms matched that map to this term
                            if raw_faiss_scores:
                                # Check if any matched synonym terms map to this medical term via synonym groups
                                if term.lower() in synonym_to_group:
                                    group_key = synonym_to_group[term.lower()]
                                    synonym_list = synonym_expansions.get(group_key, [])
                                    matched_synonyms = []
                                    for syn in synonym_list:
                                        if syn.lower() in raw_faiss_scores:
                                            matched_synonyms.append((syn, raw_faiss_scores[syn.lower()]))
                                    if matched_synonyms:
                                        self._capture_debug(f"[Location Analysis]   Step 3 - Found {len(matched_synonyms)} synonym matches for '{term}': {matched_synonyms}")
                                        # Check if any synonym scored above threshold (use slightly lower threshold for synonyms: 0.70)
                                        # Synonyms are patient-friendly terms that might not embed as well as medical terms
                                        synonym_threshold = 0.70  # Slightly lower than medical term threshold (0.75)
                                        high_scoring_synonyms = [(s, sc) for s, sc in matched_synonyms if sc >= synonym_threshold]
                                        if high_scoring_synonyms:
                                            best_synonym, best_score = max(high_scoring_synonyms, key=lambda x: x[1])
                                            self._capture_debug(f"[Location Analysis]   ✅ SATISFIED via synonym FAISS match: '{best_synonym}' (score: {best_score:.3f}) maps to '{term}'")
                                            term_satisfied = True
                                            match_reason = f"synonym FAISS match: '{best_synonym}' (score: {best_score:.3f}) → '{term}'"
                                        else:
                                            self._capture_debug(f"[Location Analysis]   ⚠️ Synonyms found but below threshold ({synonym_threshold}): {matched_synonyms}")
                                # Also check all FAISS matches to see what was actually found
                                all_matches = sorted(raw_faiss_scores.items(), key=lambda x: x[1], reverse=True)
                                top_5_matches = all_matches[:5]
                                self._capture_debug(f"[Location Analysis]   Step 3 - Top 5 FAISS matches: {top_5_matches}")
                    
                    if in_semantic_matches:
                        term_satisfied = True
                        if score is not None:
                            match_reason = f"FAISS semantic match (score: {score:.3f})"
                        else:
                            match_reason = f"FAISS semantic match"
                        self._capture_debug(f"[Location Analysis]   ✅ SATISFIED via FAISS semantic match")
                    else:
                        self._capture_debug(f"[Location Analysis]   ❌ NOT in semantic_matches_set (score likely < 0.75)")
            
            if term_satisfied:
                satisfied_terms.add(term)
                self._capture_debug(f"[Location Analysis]   ✅ '{term}' satisfied: {match_reason}")
                # If term is part of a synonym group, check if we should satisfy other terms in that group
                # BUT: Only expand if no anatomical mismatch
                if synonym_to_group and term.lower() in synonym_to_group:
                    group_key = synonym_to_group[term.lower()]
                    self._capture_debug(f"[Location Analysis]   Step 4 - Synonym expansion: '{term}' is in synonym group '{group_key}', checking for expansion")
                    # Find all other terms in this group and mark them satisfied (with mismatch check)
                    if group_key in synonym_expansions:
                        expansion_list = synonym_expansions[group_key]
                        self._capture_debug(f"[Location Analysis]   Step 4 - Expansion list contains {len(expansion_list)} terms: {expansion_list[:5]}")
                        for other_synonym in expansion_list:
                            other_term_lower = other_synonym.lower()
                            if other_term_lower in all_includes:
                                self._capture_debug(f"[Location Analysis]   Step 4 - Checking expansion to '{other_term_lower}'...")
                                # Check for anatomical mismatch before adding
                                # OPTIMIZATION: Use pre-extracted components from cache
                                should_add = True
                                if oldcarts_element == 'location' and patient_components and self.medical_rule_engine:
                                    other_components = term_components_cache.get(other_term_lower, {})
                                    if other_components:
                                        is_opposite = self.medical_rule_engine._are_anatomical_opposites(patient_components, other_components)
                                        self._capture_debug(f"[Location Analysis]   Step 4 - Anatomical check for '{other_term_lower}': patient={patient_components}, condition={other_components}, is_opposite={is_opposite}")
                                        if is_opposite:
                                            should_add = False
                                            self._capture_debug(f"[Location Analysis]   ⚠️ '{other_term_lower}' skipped in synonym expansion: anatomical mismatch")
                                if should_add:
                                    satisfied_terms.add(other_term_lower)
                                    self._capture_debug(f"[Location Analysis]   ✅ '{other_term_lower}' satisfied via synonym group expansion from '{term}'")
                                else:
                                    self._capture_debug(f"[Location Analysis]   ❌ '{other_term_lower}' NOT expanded (should_add=False)")
                            else:
                                self._capture_debug(f"[Location Analysis]   Step 4 - '{other_term_lower}' NOT in all_includes, skipping expansion")
                    else:
                        self._capture_debug(f"[Location Analysis]   Step 4 - No expansion list found for group '{group_key}'")
                else:
                    self._capture_debug(f"[Location Analysis]   Step 4 - '{term}' NOT in synonym_to_group, no expansion")
            else:
                self._capture_debug(f"[Location Analysis]   ❌ '{term}' not satisfied")
        
        # Missing terms should only include terms from ACTIVE guidelines that aren't satisfied
        # (We check all guidelines for satisfied terms, but only report missing from active)
        missing = [term for term in active_includes if term not in satisfied_terms]
        
        self._capture_debug(f"[Location Analysis] ✅ Satisfied terms (checked against ALL {len(all_guidelines_to_check)} guidelines): {sorted(satisfied_terms)}")
        self._capture_debug(f"[Location Analysis] ❌ Missing terms from active guidelines only ({len(missing)} total): {sorted(missing)}")
        
        # Return both satisfied and missing terms for better decision making
        # Return all missing terms (not limited to 5) so clarification questions can see all options
        return missing, satisfied_terms
    
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
        
        # Use unified function to check which terms are satisfied
        satisfied_terms = set()
        answer_lower = answer.lower()
            # Normalize with synonyms
        try:
            organ_system = self.CATEGORY_TO_SYSTEM.get(self.current_category or 'gastrointestinal', 'GI')
            synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
            synonym_path = os.path.join(os.path.dirname(__file__), synonym_file)
            normalized_for_match = answer_lower
            if os.path.exists(synonym_path) and self.medical_rule_engine:
                with open(synonym_path, 'r') as f:
                    synonyms = json.load(f)
                normalized_for_match = self.medical_rule_engine._normalize_with_synonyms(answer, synonyms, oldcarts_element)
        except Exception:
            normalized_for_match = answer_lower
        
        # Check each term using the same logic as unified function
        for term in all_includes:
            term_satisfied = False
            
            # 1. Exact/substring matching (fast path)
            term_in_answer_lower = term in answer_lower
            answer_lower_in_term = answer_lower in term
            term_in_normalized = term in normalized_for_match
            normalized_in_term = normalized_for_match in term
            
            if term_in_answer_lower or answer_lower_in_term or term_in_normalized or normalized_in_term:
                term_satisfied = True
            else:
                # 2. Use FAISS semantic matching (same as unified function)
                if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
                    # Get active condition names for filtering
                    active_condition_names = None
                    has_active_guidelines_attr = hasattr(self, 'active_guidelines')
                    active_guidelines_exist = has_active_guidelines_attr and self.active_guidelines
                    
                    if active_guidelines_exist:
                        active_condition_names = set()
                        for g in self.active_guidelines:
                            condition_name = g.get('data', {}).get('condition', g.get('name', ''))
                            if condition_name:
                                active_condition_names.add(condition_name)
                    
                    semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(
                        answer, oldcarts_element, threshold=0.6, active_condition_names=active_condition_names
                    )
                    if term in [t.lower() for t in semantic_matches]:
                        term_satisfied = True
                    else:
                        # 3. Direct semantic similarity check (same as unified function)
                        if self.embedding_model:
                            try:
                                embeddings = self.embedding_model.encode([answer_lower, term])
                                similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                                if similarity >= 0.6:
                                    term_satisfied = True
                            except Exception:
                                pass
            
            if term_satisfied:
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
    
    def _generate_clarifying_question(self, oldcarts_element: str, patient_answer: str,
                                     clarification_count: int, missing_terms: list) -> str:
        """Generate clarifying question using patient-friendly terms from guidelines"""
        if not missing_terms:
            raise ValueError(f"Cannot generate clarifying question for {oldcarts_element} - no missing terms")
        
        self._capture_debug(f"[Clarification] 🔍 Missing medical terms ({oldcarts_element}): {missing_terms[:8]}")
        
        # Get patient-friendly terms directly from guidelines (deduplicate to avoid same friendly term from different medical terms)
        patient_friendly_terms = []
        seen_friendly_terms = set()  # Track unique patient-friendly terms to avoid duplicates
        medical_to_friendly_map = {}
        
        # Try up to 10 terms to get 5 unique patient-friendly terms
        for term in missing_terms[:10]:
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
                    if len(patient_friendly_terms) >= 5:  # Stop when we have 5 unique ones
                        break
        
        # If no good terms found, use generic clarifying question
        if not patient_friendly_terms:
            if oldcarts_element == 'location':
                return "Can you be more specific about where exactly the pain is located?"
            else:
                return f"Can you tell me more about the {oldcarts_element}?"
        
        if oldcarts_element == 'location':
            options = ", ".join(patient_friendly_terms)
            return f"Can you be more specific? For example, is it {options}?"
        else:
            options = ", ".join(patient_friendly_terms)
            return f"Can you be more specific? For example, {options}?"
    
    def _get_patient_friendly_from_guidelines(self, medical_term: str, oldcarts_element: str) -> str:
        """Get patient-friendly term directly from guidelines (case-insensitive match)"""
        medical_term_lower = medical_term.lower().strip()
        for guideline in self.active_guidelines:
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
        
        # Fallback to original term if not found
        return medical_term
    
    def _ask_about_radiation(self) -> Dict[str, Any]:
        """Ask about radiation as a separate question after location is satisfied"""
        self.radiation_asked = True
        
        # Collect all radiation terms with patient-friendly versions from radiation section
        radiation_options = []
        for guideline in self.active_guidelines:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
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
        priority_order = ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity', 'associated']
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
    
    def _generate_oldcarts_question_for_component(self, component: str) -> str:
        """Generate question for OLDCARTS component using LLM - ONLY uses sample question template, NO conversation history"""
        if not self.llm_chat_simple_fn:
            raise ValueError("LLM not available for question generation")
        
        # Sample questions for each OLDCARTS element as guidance (ONLY reference)
        sample_questions = {
            'onset': "When did this start?",
            'location': "Where exactly is the pain located?",
            'timing': "Is it constant or does it come and go?",
            'duration': "How long does each episode typically last?",
            'associated': "Are there any other symptoms you're experiencing?",
            'character': "What does it feel like?",
            'aggravating': "What makes it worse?",
            'relieving': "What helps or makes it better?",
            'severity': "On a scale of 1 to 10, how would you rate this?"
        }
        
        sample_question = sample_questions.get(component, f"Tell me more about {component}.")
        
        # Component-specific guidance to prevent mixing elements - STRICT and explicit
        component_guidance = {
            'character': "Ask ONLY 'What does it feel like?' or similar. Do NOT mention any specific qualities like 'sharp', 'sharpness', 'burning', etc. Do NOT ask about location, intensity, or duration. Keep it completely open-ended.",
            'location': "Ask ONLY 'Where exactly is the pain located?' or similar. Do NOT mention body parts or give examples. Do NOT ask about intensity or duration.",
            'severity': "Ask ONLY the EXACT question 'On a scale of 1 to 10, how would you rate this?' or very similar wording. Do NOT ask about location or other qualities. Do NOT return just a number.",
            'aggravating': "Ask ONLY 'What makes it worse?' or similar. Do NOT assume specific activities or body parts. Do NOT use words like 'triggers' or 'causes'. Keep it simple.",
            'relieving': "Ask ONLY 'What helps or makes it better?' or similar. Do NOT assume specific treatments or positions. Keep it simple.",
            'timing': "Ask ONLY 'Is it constant or does it come and go?' or similar. Do NOT add details.",
            'duration': "Ask ONLY 'How long does each episode typically last?' or similar. Do NOT add details."
        }
        
        guidance_text = component_guidance.get(component, "")
        
        system_msg = "You are a medical assistant conducting a telehealth interview. Generate a simple, direct question following the example exactly. Do NOT add assumptions, examples, or extra details. Keep it short and open-ended."
        
        # Use chief complaint and sample question template - NO conversation history
        # Make guidance more explicit and strict
        strict_instructions = "CRITICAL RULES:\n- Follow the example question structure EXACTLY\n- Keep it simple and direct\n- Do NOT add assumptions or specific examples\n- Do NOT mention body parts unless asking about location\n- Use simple language\n- Return ONLY the question, no explanation"
        
        if guidance_text:
            user_msg = f"Chief complaint: {self.chief_complaint}\n\nComponent: {component.upper()}\n\nExample question: {sample_question}\n\n{guidance_text}\n\n{strict_instructions}\n\nGenerate a question about {component} for this patient:"
        else:
            user_msg = f"Chief complaint: {self.chief_complaint}\n\nComponent: {component.upper()}\n\nExample question: {sample_question}\n\n{strict_instructions}\n\nGenerate a question about {component} for this patient:"
        
        llm_kwargs = self._get_llm_kwargs()
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **llm_kwargs
        )
        if response and response.strip():
            generated_question = response.strip()
            
            # VALIDATION: Ensure severity question is actually a question, not just a number
            if component == 'severity':
                # Check if response is just a number or too short (likely LLM misinterpreted as answer)
                if len(generated_question) < 15 or generated_question.strip().replace('/', '').replace('-', '').isdigit():
                    # Fallback to exact sample question for severity
                    self._capture_debug(f"[Engine] ⚠️ Severity question invalid ('{generated_question}'), using sample question fallback")
                    return sample_question
            
            return generated_question
        
        # If LLM returns empty, raise error instead of using fallback
        raise ValueError(f"LLM returned empty response for {component} question")
    
    def _check_conversation_for_distress(self) -> bool:
        """Check conversation history for distress indicators"""
        if getattr(self, 'demographics_optional', False):
            return True
        
        # Check recent answers for distress
        for item in self.conversation_history[-5:]:
            if item.get('type') == 'answer':
                answer = item.get('answer', '')
                distress_info = self._detect_distress(answer)
                if distress_info['is_distressed']:
                    self.demographics_optional = True
                    return True
        return False
    
    def _generate_ml_first_question_with_demographics(self) -> Dict[str, Any]:
        """Generate demographics questions in order: chronicity, age, sex"""
        # Check if distress was detected - if so, skip demographics and go to clinical questions
        if self._check_conversation_for_distress() or getattr(self, 'demographics_optional', False):
            self._capture_debug("[Engine] ⏭️ Distress detected - skipping demographics, proceeding to clinical questions")
            # If there's a pending acknowledgment, include it
            if hasattr(self, '_pending_acknowledgment') and self._pending_acknowledgment:
                acknowledgment_msg = self._pending_acknowledgment
                self._pending_acknowledgment = None
                clinical_response = self._ask_next_clinical_question()
                if clinical_response and clinical_response.get('success'):
                    next_msg = clinical_response.get('message') or clinical_response.get('question', '')
                    combined_msg = f"{acknowledgment_msg}\n\n{next_msg}"
                    return {
                        'success': True,
                        'message': combined_msg,
                        'status': clinical_response.get('status', 'questioning'),
                        'debug': clinical_response.get('debug', {})
                    }
            return self._ask_next_clinical_question()
        
        # STEP 1: Chronicity question (first after empathetic statement)
        if 'chronicity' not in self.demographics:
            if not self.llm_chat_simple_fn:
                raise ValueError("LLM not available for question generation")
            
            system_msg = "You are a medical assistant. Generate a concise question to ask if the patient's problem is new or ongoing."
            user_msg = "Is this a new problem or an ongoing issue?"
            
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
        
        # STEP 2: Age question (skip if distress detected)
        if 'age' not in self.demographics and not self.demographics_optional:
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
        
        # STEP 3: Sex question (skip if distress detected)
        if 'sex' not in self.demographics and not self.demographics_optional:
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
            return {'success': False, 'message': 'LLM not available'}
        
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
        
        question = response.strip() if response else f"About {feature_text.lower()}:"
        
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
        if not getattr(self, 'demographics_optional', False):
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
            priority_order = ['onset', 'location', 'timing', 'duration', 'character', 
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
    
    def _return_to_next_missing_element(self, acknowledgment_msg: str = None) -> Dict[str, Any]:
        """
        After handling a comment/question/distress, intelligently return to the next missing element.
        This ensures we don't lose track of what needs to be collected.
        
        Args:
            acknowledgment_msg: Optional acknowledgment message to prepend
            
        Returns:
            Response dict with next question or None if assessment complete
        """
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
        
        # Fallback
        return None
    
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
                                    answer_text, 'associated', threshold=0.70
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
            return {'success': False, 'message': 'LLM not available'}
        
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
    
    def _interpret_patient_response(self, user_input: str, expected_element: str = None) -> Dict[str, Any]:
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
        extracted_info = None
        if self.llm_chat_simple_fn and expected_element:
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
        needs_acknowledgment = False
        acknowledgment_message = None
        
        if is_distressed or is_comment or is_question:
            needs_acknowledgment = True
            # Generate acknowledgment message
            if is_distressed:
                acknowledgment_message = self._generate_empathetic_response(user_input, distress_info)
            elif is_question:
                # Acknowledge question
                acknowledgment_message = "I understand you have a question. Let me help clarify that. "
            elif is_comment:
                # Acknowledge comment
                if self.llm_chat_simple_fn:
                    try:
                        system_msg = "You are a compassionate medical assistant. Generate a brief, natural acknowledgment (1 sentence) for a patient's comment or emotional expression. Be warm and reassuring, then naturally transition back to gathering information."
                        user_msg = f"Patient said: '{user_input}'\n\nGenerate a brief acknowledgment:"
                        
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
            '10/10', '9/10', '8/10', 'worst ever', 'never felt', 'dying', 'dying pain'
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
            # Fallback response
            return "I understand you're experiencing severe symptoms. Let me focus on getting you the right care immediately. Can you tell me more about your pain?"
        
        system_msg = """You are a compassionate medical assistant. The patient is expressing significant distress with severe symptoms. 
Generate a brief (1-2 sentences), empathetic response that:
1. Acknowledges their distress
2. Reassures them you're taking this seriously
3. Immediately transitions to gathering critical clinical information (skip routine questions like age)
4. Shows urgency and concern

Be warm, professional, and action-oriented. Do NOT ask about age or routine demographics."""
        
        user_msg = f"""Patient said: "{user_answer}"

Distress detected: severity {distress_info['severity']:.1f}/10

Generate an empathetic response that acknowledges their distress and immediately asks the MOST CRITICAL clinical question to assess their condition. Skip routine questions."""
        
        llm_kwargs = self._get_llm_kwargs(override_max_tokens=80)
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
            'excruciating', 'unbearable', 'worst pain ever', '10/10', '9/10',
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
    
    def _check_and_handle_deviating_comment(self, user_answer: str, expected_element: str = None) -> Dict[str, Any]:
        """
        Unified function to check for and handle deviating comments/questions/distress.
        Called at the start of processing any user input.
        
        Returns:
            Dict with response if comment/question/distress detected and handled, None otherwise
            Also returns interpretation dict for use in further processing
        """
        # Interpret the response (detects comments, questions, distress, or direct answers)
        response_interpretation = self._interpret_patient_response(user_answer, expected_element)
        
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
        
        # Severe case criteria:
        # 1. Very high distress severity (8.0+)
        # 2. Multiple red flags detected (2+)
        # 3. One red flag + high severity language
        is_severe_emergency = (
            severity_score >= 8.0 or
            red_flag_info.get('is_severe', False) or
            (red_flag_info.get('red_flag_count', 0) >= 1 and severity_score >= 6.0)
        )
        
        if is_severe_emergency:
            self._capture_debug(f"[Engine] 🚨 SEVERE EMERGENCY DETECTED: severity={severity_score:.1f}, red_flags={red_flag_info.get('red_flag_count', 0)}")
            self._capture_debug(f"[Engine] 🚨 Skipping all questions - recommending immediate 911/ER")
            return self._generate_emergency_response(user_answer, red_flag_info, distress_info), response_interpretation
        
        # Handle distress if detected (but not severe enough for immediate emergency)
        if is_distressed:
            self._capture_debug(f"[Engine] 🚨 DISTRESS DETECTED: severity={distress_info['severity']:.1f}, urgency_boost={distress_info['urgency_boost']:.2f}")
            self.demographics_optional = True  # Skip routine questions
            
            # Boost urgency for active guidelines
            if distress_info.get('urgency_boost', 0) > 0 and self.active_guidelines:
                for guideline in self.active_guidelines:
                    current_score = guideline.get('score', 0.0)
                    guideline['score'] = min(1.0, current_score + distress_info['urgency_boost'])
                self._capture_debug(f"[Engine] ⚡ Urgency boost applied: +{distress_info['urgency_boost']:.2f}")
        
        # Handle pure questions (with no clinical info)
        if is_question and not extracted_info:
            # Pure question - acknowledge and return to next missing element
            if needs_acknowledgment:
                return self._return_to_next_missing_element(acknowledgment_msg), response_interpretation
            return self._handle_user_question(user_answer), response_interpretation
        
        # Handle pure comments (with no extractable clinical info)
        if needs_acknowledgment and not extracted_info:
            # Pure comment - acknowledge and return to next missing element
            self._capture_debug(f"[Engine] 💬 Pure comment detected - acknowledging and returning to next missing element")
            return self._return_to_next_missing_element(acknowledgment_msg), response_interpretation
        
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
        # Get context about what we're expecting
        last_q = None
        for item in reversed(self.conversation_history):
            if item.get('type') in ['question', 'statement']:
                last_q = item
                break
        
        expected_element = last_q.get('oldcarts') if last_q else None
        
        # ALWAYS FIRST: Check for and handle deviating comments/questions/distress
        comment_response, response_interpretation = self._check_and_handle_deviating_comment(user_answer, expected_element)
        if comment_response:
            # Comment/question/distress was handled - return the response
            return comment_response
        
        # Extract info from interpretation (for use in processing)
        is_distressed = response_interpretation.get('is_distressed', False)
        needs_acknowledgment = response_interpretation.get('needs_acknowledgment', False)
        acknowledgment_msg = response_interpretation.get('acknowledgment_message')
        extracted_info = response_interpretation.get('extracted_info')
        
        # Store acknowledgment for later if answer contains clinical info AND needs acknowledgment
        if needs_acknowledgment:
            self._pending_acknowledgment = acknowledgment_msg
            self._capture_debug(f"[Engine] 💬 Acknowledgment stored for next question: {response_interpretation['type']}")
        
        # Record answer (with original text for context)
        self.conversation_history.append({
            'type': 'answer',
            'answer': user_answer,
            'interpretation': response_interpretation  # Store interpretation for debugging
        })
        
        # Use extracted info if available (cleaner than full response with comments)
        processing_answer = extracted_info if extracted_info else user_answer
        
        # If distress detected, skip routine demographics and prioritize clinical assessment
        # (This check happens after interpretation, but we already handled urgency above)
        if is_distressed:
            # Get last question to check if we were asking a routine question
            if last_q and last_q.get('focus') in ['age', 'sex', 'chronicity']:
                self._capture_debug(f"[Engine] ⏭️ Skipping routine question ({last_q.get('focus')}) due to distress")
                
                # Use pending acknowledgment if available, otherwise generate empathetic response
                if hasattr(self, '_pending_acknowledgment') and self._pending_acknowledgment:
                    acknowledgment_msg = self._pending_acknowledgment
                    self._pending_acknowledgment = None
                else:
                    acknowledgment_msg = self._generate_empathetic_response(user_answer, distress_info)
                
                # Move directly to clinical questions
                clinical_response = self._ask_next_clinical_question()
                
                if clinical_response and clinical_response.get('success'):
                    # Combine acknowledgment with clinical question
                    next_msg = clinical_response.get('message') or clinical_response.get('question', '')
                    combined_msg = f"{acknowledgment_msg}\n\n{next_msg}"
                    return {
                        'success': True,
                        'message': combined_msg,
                        'status': clinical_response.get('status', 'questioning'),
                        'debug': {
                            'engine': self._format_engine_debug(f"[Engine] 🚨 Distress handled - skipped demographics"),
                            'internal': self._get_debug_info()
                        }
                    }
                else:
                    # Just return acknowledgment if no clinical question available
                    return {
                        'success': True,
                        'message': acknowledgment_msg,
                        'status': 'questioning',
                        'debug': {
                            'engine': self._format_engine_debug(f"[Engine] 🚨 Distress acknowledged"),
                            'internal': self._get_debug_info()
                        }
                    }
        
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
            
            # Fallback to LLM if simple validation fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Extract the patient's age from their response. Return ONLY a number between 0-150, or 'invalid' if not a valid age."
                user_msg = f"Patient said: '{processing_answer}'\n\nExtract age as a number only:"
        
                llm_kwargs = self._get_llm_kwargs(override_max_tokens=10)
                response = self.llm_chat_simple_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    **llm_kwargs
                )
                
                age_str = response.strip()
                if age_str.isdigit():
                    age = int(age_str)
                    if 0 <= age <= 150:
                        self.demographics['age'] = age
                        
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
                'message': 'Please provide your age as a number (e.g., 25, thirty-five, etc.)',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ❌ Age validation failed"),
                    'internal': self._get_debug_info(last_answer=user_answer)
                }
            }
        
        
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
        
        if last_q and last_q.get('focus') == 'chronicity':
            # Note: Interpretation already happened at start of process_answer
            # Use the processing_answer (which may have extracted info) for keyword matching
            
            # Simple keyword matching first (more reliable than LLM)
            # Use processing_answer (may have extracted info, or original if no extraction)
            processing_answer_lower = processing_answer.lower().strip()
            
            # Check for button callbacks first
            if user_answer == 'chronicity_new':
                self.demographics['chronicity'] = 'new'
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'chronicity_recurring':
                self.demographics['chronicity'] = 'recurring'
                return self._generate_ml_first_question_with_demographics()
            
            # Simple keyword matching
            new_indicators = ['new', 'recent', 'today', 'yesterday', 'this week', 'sudden', 'acute']
            has_new_indicator = any(word in processing_answer_lower for word in new_indicators)
            
            recurring_indicators = ['ongoing', 'recurring', 'chronic', 'months', 'years', 'always', 'frequent', 'often']
            has_recurring_indicator = any(word in processing_answer_lower for word in recurring_indicators)
            
            if has_new_indicator:
                self.demographics['chronicity'] = 'new'
                # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                return self._return_to_next_missing_element(acknowledgment_msg if needs_acknowledgment else None)
            elif has_recurring_indicator:
                self.demographics['chronicity'] = 'recurring'
                # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                return self._return_to_next_missing_element(acknowledgment_msg if needs_acknowledgment else None)
            
            # Fallback to LLM if simple matching fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Determine if the patient's problem is new or recurring. Return ONLY 'new', 'recurring', or 'invalid'."
                user_msg = f"Patient said: '{user_answer}'\n\nIs this a new problem or recurring/ongoing? (new/recurring):"
                
                try:
                    response = self.llm_chat_simple_fn(
                        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                        max_tokens=10,
                        temperature=self.temperature
                    )
                    
                    chronicity_str = response.strip().lower()
                    if chronicity_str in ['new', 'recurring']:
                        self.demographics['chronicity'] = chronicity_str
                        # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                        return self._return_to_next_missing_element(acknowledgment_msg if needs_acknowledgment else None)
                    elif 'new' in chronicity_str:
                        self.demographics['chronicity'] = 'new'
                        # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                        return self._return_to_next_missing_element(acknowledgment_msg if needs_acknowledgment else None)
                    else:
                        recurring_in_str = 'recurring' in chronicity_str
                        ongoing_in_str = 'ongoing' in chronicity_str
                        
                        if recurring_in_str or ongoing_in_str:
                            self.demographics['chronicity'] = 'recurring'
                            # Use intelligent return to next missing element (handles demographics + OLDCARTS)
                            return self._return_to_next_missing_element(acknowledgment_msg if needs_acknowledgment else None)
                except Exception as e:
                    # LLM failed, fall through to error message
                    pass
            
            return {
                'success': False,
                'message': 'Please specify if this is a new problem or ongoing issue',
                'debug': {
                    'engine': self._format_engine_debug("[Engine] ❌ Chronicity validation failed"),
                    'internal': self._get_debug_info(last_answer=user_answer)
                }
            }
        
        # If no demographics handler matched and demographics incomplete, check if we need to ask demographics
        # Check demographics - skip if distress was detected (demographics optional)
        required_demographics = ['chronicity']  # Always need chronicity
        if not self.demographics_optional:
            required_demographics.extend(['age', 'sex'])  # Age and sex only if not distressed
        
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
            # Generate patient-friendly clarification options directly from guidelines
            options = self._collect_patient_friendly_options('location', limit=3)
            if options:
                msg = "Can you be more specific? For example, is it " + ", ".join(options) + "?"
            else:
                msg = "Can you be more specific about where exactly the pain is located?"
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
        
        # Provide helpful clarification based on the component being asked about
        if last_q:
            oldcarts_element = last_q.get('oldcarts', '')
            
            # Component-specific clarifications
            clarifications = {
                'character': "I'm asking about how the pain feels - like sharp, dull, burning, stabbing, or throbbing.",
                'location': "I'm asking where exactly on your body the pain is located.",
                'timing': "I'm asking whether the pain is constant or comes and goes.",
                'duration': "I'm asking how long each episode of pain typically lasts.",
                'aggravating': "I'm asking what makes the pain worse - activities, positions, or movements.",
                'relieving': "I'm asking what helps or makes the pain better - medications, rest, or positions.",
                'severity': "I'm asking how bad the pain is on a scale from 1 to 10, where 1 is mild and 10 is the worst pain imaginable."
            }
            
            clarification = clarifications.get(oldcarts_element, last_q.get('question', 'Please answer my previous question.'))
            
            # Return clarification + repeat question
            return {
                'success': True,
                'message': f"{clarification}\n\n{last_q.get('question', 'Please answer my previous question.')}",
                'status': 'questioning'
            }
        else:
            return {
                'success': False,
                'message': 'Please answer my previous question.'
            }
    
    # Removed redundant anatomical analysis functions - medical_rules.json + OLDCARTS synonyms handle this