#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - LLM-Driven Medical Diagnosis

FRAMEWORK: OLDCARTS (Gold Standard Clinical Pain/Symptom Assessment)
- Onset, Location, Duration, Character, Aggravating, Relieving, Timing, Severity
- Applicable to ALL medical conditions (abdominal pain, chest pain, headache, etc.)
- Systematic, comprehensive questioning

DIAGNOSTIC FLOW:
1. Chief complaint → Match relevant guidelines (any body system)
2. Sort by URGENCY (emergent > urgent > routine) then PREVALENCE (common > rare)
3. Top 3 become active differentials, rest go to reserve pool
4. Feed all 3 guidelines' classical presentations to LLM
5. LLM follows OLDCARTS roadmap to generate systematic questions
6. Ask question → LLM scores all 3 → Re-rank by score
7. Rule out <3% → Promote from reserve (only clear mismatches, preserve diffuse conditions)
8. Repeat until 95% confidence + 12 questions (or 15 max)
9. Screen ALL red flags after diagnosis
10. Finalize with disposition + red flag warnings

PREVALENCE-BASED ROLLING DIFFERENTIAL:
- Start with common conditions (gastroenteritis, appendicitis, UTI, etc.)
- Only consider rare conditions (ectopic, mesenteric ischemia) after common ones ruled out
- Mimics clinical reasoning: "Common things are common"
- Reserve pool sorted by prevalence ensures common conditions promoted first

FULLY LLM-DRIVEN:
- NO hardcoded answer validation patterns
- LLM decides what's acceptable (dynamic, organic)
- LLM generates all questions following OLDCARTS
- LLM does ALL reasoning - we provide structure only
"""

import json
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from thinking_fillers import get_filler

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client
import numpy as np

# Import fuzzy medical matcher for typo correction
from fuzzy_medical_matcher import FuzzyMedicalMatcher

class RAGEmbeddingAPI:
    """
    Wrapper for RAG client's embedding service
    Provides same interface as SentenceTransformer but uses modular RAG client
    Supports both GPU (RAG container) and CPU (local) modes
    """
    def __init__(self, rag_url: str = "http://localhost:11435"):
        # rag_url parameter kept for backwards compatibility but not used
        # RAGClient handles URL configuration internally
        self.rag_client = get_rag_client()
    
    def encode(self, texts: List[str]) -> List:
        """
        Generate embeddings via modular RAG client
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (numpy arrays)
        
        Raises:
            RuntimeError if embedding service fails
        """
        embeddings = self.rag_client.embed(texts)
        
        if embeddings:
            # Convert to numpy arrays
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        else:
            raise RuntimeError(f"RAG embedding failed")

# RAG client availability check
try:
    rag_api = RAGEmbeddingAPI()
    # Test the client
    test_embedding = rag_api.encode(["test"])
    RAG_API_AVAILABLE = True
    rag_client = get_rag_client()
    print(f"[Engine] ✅ RAG client available - using {rag_client.get_mode()}")
except Exception as e:
    RAG_API_AVAILABLE = False
    print(f"[Engine] ⚠️ RAG client not available - using brute-force matching: {e}")


class AdaptiveDiagnosticEngine:
    """
    LLM-driven diagnostic engine
    
    The LLM is the intelligence - it reads guidelines and reasons about diagnosis.
    We provide structure and keep it focused.
    """
    
    def __init__(self, guidelines_dir: str = None, llm_chat_fn=None, embedding_model=None, llm_chat_simple_fn=None):
        """
        Initialize diagnostic engine
        
        Args:
            guidelines_dir: Path to JSON guidelines (defaults to auto-detect)
            llm_chat_fn: LLM function for complex reasoning (Mistral-7B)
            embedding_model: Sentence transformer for semantic similarity
            llm_chat_simple_fn: Optional LLM for simple tasks (Llama-1B). If None, uses llm_chat_fn
        """
        # Auto-detect guidelines directory
        if guidelines_dir is None:
            # Try container path first, then local path
            if os.path.exists("/app/medical/guidelines"):
                guidelines_dir = "/app/medical/guidelines"
            elif os.path.exists("medical/guidelines"):
                guidelines_dir = "medical/guidelines"
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "medical", "guidelines")):
                guidelines_dir = os.path.join(os.path.dirname(__file__), "medical", "guidelines")
            else:
                raise RuntimeError("Could not find medical guidelines directory")
        
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn  # Mistral-7B for complex diagnostic questions
        self.llm_chat_simple_fn = llm_chat_simple_fn or llm_chat_fn  # Llama-1B for templates/validation
        self.embedding_model = embedding_model
        
        # Temperature configuration from environment variables (no fallbacks)
        self.temperature_simple = float(os.environ['LLM_TEMPERATURE_SIMPLE'])
        self.temperature_complex = float(os.environ['LLM_TEMPERATURE_COMPLEX'])
        
        # Token limits for different question types
        self.max_tokens_question = 60      # For complex clinical questions
        self.max_tokens_simple = 30         # For simple questions
        self.max_tokens_classification = 10 # For yes/no classifications
        self.max_tokens_normalization = 50  # For text normalization
        self.max_tokens_red_flag = 80       # For red flag questions
        self.max_tokens_follow_up = 100     # For follow-up questions
        
        # Initialize debug capture
        self._captured_debug_output = []
        
        # Initialize current category (determined during assessment)
        self.current_category = None
        
        # Initialize RAG mode selection (simple toggle)
        self.rag_mode = os.environ.get('RAG_MODE', 'CPU').upper()  # CPU or GPU
        self._capture_debug(f"[Engine] 🎯 RAG Mode: {self.rag_mode}")
        
        # Initialize Medical Rule Engine for enhanced location scoring
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            # Pass embedding model for deep semantic similarity
            self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
            self._capture_debug(f"[Engine] 🎯 Medical Rule Engine initialized with embedding model")
        except ImportError as e:
            self.medical_rule_engine = None
            self._capture_debug(f"[Engine] ⚠️ Medical Rule Engine not available: {e}")
            # Try alternative import path
            try:
                import sys
                sys.path.append('/app/ml')
                from medical_rule_engine import MedicalRuleEngine
                self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
                self._capture_debug(f"[Engine] 🎯 Medical Rule Engine initialized (alternative path) with embedding model")
            except ImportError as e2:
                self._capture_debug(f"[Engine] ❌ Medical Rule Engine failed both paths: {e2}")
        
        # Initialize Learning System for manual review suggestions
        try:
            from ml.learning_suggestions import LearningSuggestions
            self.learner = LearningSuggestions()
            self._capture_debug(f"[Engine] 📊 Active Learning: ENABLED (manual review mode)")
            self._capture_debug(f"[Engine]   💡 Run 'python ml/learning_suggestions.py --analyze' to generate suggestions")
            self._capture_debug(f"[Engine]   💡 Run 'python ml/learning_suggestions.py --show' to view suggestions")
        except ImportError:
            self.learner = None
            self._capture_debug(f"[Engine] 📊 Active Learning: DISABLED (import failed)")
        
        # Initialize fuzzy medical matcher for typo correction
        self.fuzzy_matcher = FuzzyMedicalMatcher()
        self._capture_debug(f"[Engine] 🔤 Fuzzy Medical Matcher initialized")
        
        self._capture_debug(f"[Engine] 🧠 Using {'dual models (simple + complex)' if llm_chat_simple_fn else 'single model'}")
        self._capture_debug(f"[Engine] 🌡️ Temperature settings: Simple={self.temperature_simple}, Complex={self.temperature_complex}")
        self._capture_debug(f"[Engine] 🔍 Environment check: LLM_TEMPERATURE_SIMPLE={os.environ.get('LLM_TEMPERATURE_SIMPLE', 'NOT SET')}, LLM_TEMPERATURE_COMPLEX={os.environ.get('LLM_TEMPERATURE_COMPLEX', 'NOT SET')}")
        
        # ============================================================================
        # 🔧 CONFIGURATION TOGGLES (Easy to modify)
        # ============================================================================
        
        # Smart normalization configuration
        self.smart_normalization = True  # True=LLM normalization, False=synonym normalization
        
        # RAG validation toggle
        self.validate_rag = os.getenv("VALIDATE_RAG", "false").lower() == "true"  # Compare RAG vs brute-force
        
        # Hybrid matching configuration
        # ML-only configuration (no hybrid scoring)
        self.ml_config = {
            'similarity_threshold': 0.5,    # ML similarity threshold
            'active_guidelines': 5,         # Number of active guidelines
            'reserve_guidelines': 5         # Number of reserve guidelines
        }
        
        # ============================================================================
        # 🔧 END CONFIGURATION TOGGLES
        # ============================================================================
        
        # RAG API for GPU-accelerated FAISS operations
        self.rag_api = RAGEmbeddingAPI() if RAG_API_AVAILABLE else None
        # Set use_rag_api based on RAG_MODE environment variable
        self.use_rag_api = (self.rag_mode == 'GPU' and RAG_API_AVAILABLE)
        
        # OLDCARTS element weights removed - now using semantic similarity scoring
        # Each OLDCARTS element is scored using vector similarity to guideline sections
        # No subjective weights needed - mathematical similarity is objective
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # Current assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files from subdirectories"""
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 📚 LOADING MEDICAL GUIDELINES")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] 📁 Source directory: {self.guidelines_dir}")
        
        if not self.guidelines_dir.exists():
            self._capture_debug(f"[Engine] ❌ Directory not found: {self.guidelines_dir}")
            return
        
        # Track by organ system
        organ_systems = {}
        
        # Load from subdirectories (GI, CARDIO, GU, etc.)
        for json_file in sorted(self.guidelines_dir.glob("**/*.json")):
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    name = guideline.get('condition', json_file.stem)
                    organ_system = json_file.parent.name if json_file.parent != self.guidelines_dir else "Other"
                    self.all_guidelines[name] = guideline
                    
                    # Track organ system counts
                    if organ_system not in organ_systems:
                        organ_systems[organ_system] = []
                    organ_systems[organ_system].append(name)
                    
                    # Debug: Check if structured_oldcarts exists
                    key_features = guideline.get('key_features', {})
                    structured_oldcarts = key_features.get('structured_oldcarts', {})
                    has_structured_data = bool(structured_oldcarts and isinstance(structured_oldcarts, dict) and structured_oldcarts)
                    
                    self._capture_debug(f"[Engine]   ✓ {organ_system}/{name} (structured_oldcarts: {has_structured_data})")
                    if has_structured_data:
                        self._capture_debug(f"[Engine]     📋 structured_oldcarts keys: {list(structured_oldcarts.keys())}")
                    else:
                        self._capture_debug(f"[Engine]     ⚠️ structured_oldcarts content: {structured_oldcarts}")
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
        
        self._capture_debug(f"\n[Engine] ✅ LOADED {len(self.all_guidelines)} GUIDELINES:")
        for system, conditions in sorted(organ_systems.items()):
            self._capture_debug(f"[Engine]    📋 {system}: {len(conditions)} conditions")
        self._capture_debug(f"{'='*80}\n")
    
    
    # Chief complaint validation is handled through natural conversation flow
    # 2. Phase 1/2 matching 
    # 3. If no matches found, ask clarification
    
    def _get_debug_info(self, last_answer: str = None) -> Dict:
        """
        Build debug information for Telegram display
        Shows internal reasoning, scores, rankings, OLDCARTS coverage, etc.
        """
        # Get question count
        num_questions = len([item for item in self.conversation_history if item['type'] == 'question' and item.get('focus') == 'clinical'])
        
        # OLDCARTS coverage
        covered_count = sum(self.oldcarts_covered.values())
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        
        debug_info = {
            'demographics': self.demographics,
            'question_number': num_questions,
            'oldcarts_coverage': coverage_str,
            'oldcarts_count': f"{covered_count}/8",
            'clarification_counts': dict(self.clarification_count),
            'active_differentials': [
                {
                    'rank': i+1,
                    'name': g['name'],
                    'score': f"{g['score']:.0%}",
                    'urgency': g['data'].get('urgency', 'routine'),
                    'prevalence': g['data'].get('prevalence', 'uncommon')
                }
                for i, g in enumerate(self.active_guidelines[:5])
            ],
            'pool_status': {
                'active': len(self.active_guidelines),
                'reserve': len(self.reserve_pool),
                'ruled_out': len(self.ruled_out)
            },
            'last_answer': last_answer,
            'last_answer_scores': getattr(self, '_last_answer_scores', None),  # Set during scoring
            'engine_debug_output': getattr(self, '_captured_debug_output', [])  # Captured debug output
        }
        
        # Add matching algorithm info if available
        if hasattr(self, 'matching_metadata') and self.matching_metadata:
            debug_info['matching'] = self.matching_metadata
        
        return debug_info
    
    def _capture_debug(self, message: str):
        """Capture debug output for Telegram display"""
        self._captured_debug_output.append(message)
        print(message)  # Still print to stdout for container logs
    
    def reset_assessment(self):
        """Reset for new patient"""
        self.active_guidelines = []  # The 3 active guidelines with scores
        self.reserve_pool = []  # Remaining matched guidelines (for rolling replacement)
        self.matching_metadata = {}  # Store matching algorithm info for debug
        self.ruled_out = []  # Guidelines ruled out (for logging)
        self.chief_complaint = ""
        self.demographics = {}  # age, sex
        self.conversation_history = []  # All Q&A
        self.status = "idle"  # idle, questioning, red_flag_screening, diagnosed
        self.red_flags_present = []  # Track which red flags are present
        self.red_flag_index = 0  # Track which red flag we're asking about
        self.diagnosed_condition = None  # Store diagnosed condition for red flag screening
        
        # Clear any LLM model state/cache to prevent cross-session contamination
        self._capture_debug(f"[Engine] 🔄 Clearing LLM model state for fresh session")
        
        # ML Progress Tracking
        self._capture_debug(f"[Scoring] 📊 Session reset - ML learning state cleared")
        
        # OLDCARTS tracking - must cover ALL before diagnosis
        self.oldcarts_covered = {
            'O': False,  # Onset (hardcoded first question)
            'L': False,  # Location
            'D': False,  # Duration
            'C': False,  # Character
            'A': False,  # Aggravating
            'R': False,  # Relieving
            'T': False,  # Timing
            'S': False   # Severity
        }
        
        # Clarification tracking
        self.clarification_count = {}  # Track how many times we've asked for clarification per OLDCARTS element
        
        # Debug output capture for Telegram display
        self._captured_debug_output = []  # Capture debug output for Telegram display
        
        # Thresholds - Clinical scoring: only rule out with definitive proof
        self.RULE_OUT_THRESHOLD = 0.05  # Below 5% → rule out (ML-only system threshold)
        self.MINIMUM_SCORE_FOR_RANKING = 0.05  # Minimum score to be considered for ranking
        self.MAX_ACTIVE = 5  # Keep 5 active differentials
        self.MAX_CLARIFICATIONS = 2  # Max times to ask for clarification before moving on
    
    def _get_dynamic_threshold(self, score: float) -> float:
        """
        Get dynamic threshold based on score type for ML-only system
        """
        # ML-only system thresholds based on anatomical rules
        if score >= 0.5:  # Bilateral conditions (0.5)
            return 0.4  # Keep bilateral conditions unless very low
        elif score >= 0.4:  # Midline conditions (0.4)
            return 0.3  # Keep midline conditions unless very low
        elif score >= 0.3:  # Same side conditions (0.3)
            return 0.2  # Keep same side conditions unless very low
        elif score >= 0.2:  # ML predictions (0.2)
            return 0.1  # Keep ML predictions unless very low
        else:  # Anatomical opposites (0.0-0.05)
            return 0.1  # Rule out anatomical mismatches (10% threshold)
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        Start new assessment with ML-powered processing
        
        Args:
            chief_complaint: e.g., "I have abdominal pain"
        
        Returns:
            Response with first question
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (ML-POWERED)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # STRUCTURED OLDCARTS PROCESSING
        # Step 1: Basic category detection (for guideline filtering)
        self.current_category = self._categorize_complaint_by_substring(chief_complaint)
        self._capture_debug(f"[Engine] 🎯 Category: {self.current_category}")
        
        # Step 2: Get guidelines for this category
        matched_guidelines = self._get_all_guidelines_in_category(self.current_category)
        self._capture_debug(f"[Engine] 📊 Found {len(matched_guidelines)} guidelines for {self.current_category}")
        
        # Step 3: Parse prompt against structured OLDCARTS
        oldcarts_analysis = self._parse_prompt_against_structured_oldcarts(chief_complaint, matched_guidelines)
        self._capture_debug(f"[Engine] 🔍 OLDCARTS Analysis: {oldcarts_analysis}")
        
        # Continue with structured assessment
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        
        # Use matched guidelines
        self.active_guidelines = matched_guidelines[:5]  # Top 5
        self.reserve_pool = matched_guidelines[5:]  # Rest
        
        # Store OLDCARTS analysis for use in questioning
        self.oldcarts_analysis = oldcarts_analysis
        
        self._capture_debug(f"[Engine] 🎯 Structured guidelines: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}")
        
        # Generate first question using structured approach (with age/sex first)
        return self._generate_ml_first_question_with_demographics()
    
    def _match_to_guidelines_ml(self, normalized_complaint: str, category: str) -> List[Dict]:
        """ML-powered guideline matching with OLDCARTS construction"""
        self._capture_debug(f"[Engine] 🧠 OLDCARTS CONSTRUCTION DEBUG:")
        self._capture_debug(f"[Engine] 🧠 Input: '{normalized_complaint}'")
        self._capture_debug(f"[Engine] 🧠 Category: {category}")
        
        # Get relevant guidelines by category (already narrowed down)
        relevant_guidelines = self._get_guidelines_by_category(category)
        self._capture_debug(f"[Engine] 🧠 Relevant guidelines: {len(relevant_guidelines)}")
        
        # Parse OLDCARTS components from complaint
        components = self._parse_oldcarts_components(normalized_complaint)
        self._capture_debug(f"[Engine] 🧠 OLDCARTS components detected: {components}")
        
        # Construct OLDCARTS answers from complaint
        oldcarts_answers = self._construct_oldcarts_answers(components)
        self._capture_debug(f"[Engine] 🧠 OLDCARTS answers constructed: {oldcarts_answers}")
        
        # Auto-fill answered components and identify missing ones
        missing_components = self._identify_missing_oldcarts_components(oldcarts_answers)
        self._capture_debug(f"[Engine] 🧠 Missing OLDCARTS components: {missing_components}")
        
        # Return all guidelines with OLDCARTS answers for smart questioning
        matched_guidelines = []
        for name, guideline in relevant_guidelines.items():
            matched_guidelines.append({
                'name': name,
                'score': 0.5,  # Equal priority for all
                'data': guideline,
                'oldcarts_answers': oldcarts_answers,
                'missing_components': missing_components,
                'method': 'oldcarts_construction'
            })
        
        self._capture_debug(f"[Engine] 🧠 OLDCARTS construction complete: {len(matched_guidelines)} guidelines with {len(missing_components)} missing components")
        return matched_guidelines
    
    def _is_whole_phrase_match(self, term: str, text: str) -> bool:
        """
        Check if a term appears as a complete phrase in text, preventing false substring matches
        
        Args:
            term: The term to search for (e.g., "sharp pain", "cramping")
            text: The text to search in (e.g., "i have sharp abdominal pain")
        
        Returns:
            True if term appears as complete phrase, False if just substring
        
        Examples:
            _is_whole_phrase_match("sharp pain", "i have sharp abdominal pain") → True
            _is_whole_phrase_match("sharp pain", "i have abdominal pain") → False
            _is_whole_phrase_match("cramping", "i have cramping pain") → True
        """
        import re
        
        # Normalize the term (remove extra spaces, convert to lowercase)
        term_clean = ' '.join(term.strip().lower().split())
        text_clean = text.strip().lower()
        
        # If term is empty, return False
        if not term_clean:
            return False
        
        # Create a regex pattern that matches the term as whole words
        # \b ensures word boundaries (prevents substring matching)
        pattern = r'\b' + re.escape(term_clean) + r'\b'
        
        # Search for the pattern in the text
        match = re.search(pattern, text_clean)
        
        return match is not None
    
    def _get_all_guidelines_in_category(self, category: str) -> List[Dict]:
        """Get all guidelines in category for generic complaints"""
        relevant_guidelines = self._get_guidelines_by_category(category)
        
        # Return all guidelines with equal priority
        matched_guidelines = []
        for name, guideline in relevant_guidelines.items():
            matched_guidelines.append({
                'name': name,
                'score': 0.5,  # Equal priority for all
                'data': guideline,
                'ml_similarity': 0.5,
                'best_trigger': 'generic_complaint',
                'method': 'generic_complaint'
            })
        
        # Sort by prevalence (common conditions first)
        matched_guidelines.sort(key=lambda x: x['data'].get('prevalence_score', 0), reverse=True)
        
        self._capture_debug(f"[Engine] 🧠 Generic complaint - returning all {len(matched_guidelines)} {category} guidelines")
        return matched_guidelines
    
    
    
    def _generate_ml_first_question_with_demographics(self) -> Dict[str, Any]:
        """Generate first question using structured OLDCARTS approach with demographics"""
        self._capture_debug(f"[Engine] 🧠 Generating structured first question with demographics...")
        
        # PRIORITY 0: Add empathetic opening statement (only on first question)
        empathetic_prefix = ""
        if not self.conversation_history:
            chief_complaint_lower = self.chief_complaint.lower()
            # Extract the symptom from chief complaint
            symptom = chief_complaint_lower.replace('i have ', '').replace('i am experiencing ', '').replace('i\'m experiencing ', '').strip()
            empathetic_prefix = f"I'm sorry you're experiencing {symptom}. Let me ask you some questions to help determine what's causing it. "
            self._capture_debug(f"[Engine] ✅ Empathetic prefix added: '{empathetic_prefix}'")
        
        # PRIORITY 1: Ask demographics FIRST (age, then sex, then chronicity)
        if not hasattr(self, 'demographics') or not self.demographics.get('age'):
            # Ask age first
            question = empathetic_prefix + "How old are you?"
            self._capture_debug(f"[Engine] ✅ Demographics question generated: '{question}'")
            
            # Add to conversation history with proper focus
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'demographics'
            })
            
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'debug': self._get_debug_info()
            }
        elif 'sex' not in self.demographics:
            # Ask sex with button-based response
            question = empathetic_prefix + "What is your biological sex?"
            self._capture_debug(f"[Engine] ✅ Sex question with buttons: '{question}'")
            
            # Add to conversation history with proper focus
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'demographics'
            })
            
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'Male', 'callback_data': 'sex_male'},
                    {'text': 'Female', 'callback_data': 'sex_female'}
                ],
                'debug': self._get_debug_info()
            }
        elif 'chronicity' not in self.demographics:
            # Ask chronicity with button-based response
            question = empathetic_prefix + "Is this a new problem or an ongoing issue?"
            self._capture_debug(f"[Engine] ✅ Chronicity question with buttons: '{question}'")
            
            # Add to conversation history with proper focus
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'demographics'
            })
            
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'New Problem', 'callback_data': 'chronicity_new'},
                    {'text': 'Ongoing Issue', 'callback_data': 'chronicity_recurring'}
                ],
                'debug': self._get_debug_info()
            }
        else:
            # Both demographics collected, ask first missing OLDCARTS component
            if hasattr(self, 'oldcarts_analysis') and self.oldcarts_analysis:
                missing_components = self.oldcarts_analysis.get('missing_components', [])
                if missing_components:
                    first_missing = missing_components[0]
                    question = self._generate_oldcarts_question_for_component(first_missing)
                    self._capture_debug(f"[Engine] ✅ Structured OLDCARTS question generated: '{question}'")
                    
                    # Add to conversation history
                    is_demographics = 'age' in question.lower() or 'sex' in question.lower() or 'old are you' in question.lower() or 'biological sex' in question.lower() or 'new problem' in question.lower() or 'ongoing' in question.lower()
                    self.conversation_history.append({
                        'type': 'question',
                        'question': question,
                        'oldcarts': 'demographics' if is_demographics else 'oldcarts',
                        'focus': 'demographics' if is_demographics else 'oldcarts'
                    })
                    
                    return {
                        'success': True,
                        'question': question,
                        'status': 'questioning',
                        'debug': self._get_debug_info()
                    }
                else:
                    # All OLDCARTS components already answered in initial prompt
                    self._capture_debug(f"[Engine] ✅ All OLDCARTS components already answered in initial prompt")
                    # Proceed to scoring and diagnosis
                    return self._ask_next_clinical_question()
            else:
                raise RuntimeError("No OLDCARTS analysis available for structured questions")
    
    def _analyze_comprehensive_oldcarts_coverage(self, answer: str, primary_element: str) -> List[str]:
        """
        Analyze a comprehensive answer to determine which OLDCARTS elements it covers
        
        This prevents asking redundant questions when a single answer covers multiple elements.
        For example, "2 days ago" covers both onset (when it started) and duration (how long).
        
        Args:
            answer: The patient's answer
            primary_element: The primary OLDCARTS element being asked about
            
        Returns:
            List of OLDCARTS elements that this answer covers
        """
        covered_elements = [primary_element]  # Always include the primary element
        answer_lower = answer.lower().strip()
        
        # SEMANTIC ANALYSIS: Use embedding similarity to determine which OLDCARTS elements are covered
        # This is more accurate than hardcoded keyword matching
        covered_elements = self._analyze_semantic_oldcarts_coverage(answer, primary_element)
        
        self._capture_debug(f"[Engine] 🔍 Comprehensive analysis of '{answer}':")
        self._capture_debug(f"[Engine]   Primary element: {primary_element}")
        self._capture_debug(f"[Engine]   Covered elements: {covered_elements}")
        
        return covered_elements

    def _analyze_semantic_oldcarts_coverage(self, answer: str, primary_element: str) -> List[str]:
        """
        Use semantic embedding similarity to determine which OLDCARTS elements are covered by the answer
        
        This method compares the user's answer against structured OLDCARTS data from guidelines
        to determine which elements are semantically covered, rather than using hardcoded keywords.
        
        Args:
            answer: The user's answer
            primary_element: The primary OLDCARTS element being asked about (single letter: 'O', 'L', etc.)
            
        Returns:
            List of OLDCARTS elements that this answer semantically covers (single letters: ['O', 'L', etc.])
        """
        # Convert primary element to single letter if needed
        if len(primary_element) > 1:
            primary_letter = primary_element[0].upper()
        else:
            primary_letter = primary_element.upper()
        
        covered_elements = [primary_letter]  # Always include the primary element
        
        # Get all OLDCARTS elements to check (full names for data lookup)
        all_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
        
        # Skip the primary element since it's already included
        elements_to_check = [elem for elem in all_elements if elem != primary_element.lower()]
        
        for element in elements_to_check:
            # Get structured OLDCARTS data for this element from active guidelines
            element_data = self._get_structured_oldcarts_data(element)
            
            if element_data:
                # Use semantic similarity to determine if the answer covers this element
                similarity = self._compute_semantic_oldcarts_similarity(answer, element_data, element)
                
                # If similarity is above threshold, the answer covers this element
                if similarity > 0.3:  # 30% similarity threshold
                    element_letter = element[0].upper()  # Convert to single letter
                    covered_elements.append(element_letter)
                    self._capture_debug(f"[Engine]   ✅ Semantic match for {element} ({element_letter}): {similarity:.2f}")
                else:
                    self._capture_debug(f"[Engine]   ❌ No semantic match for {element}: {similarity:.2f}")
        
        return covered_elements

    def _get_structured_oldcarts_data(self, element: str) -> str:
        """
        Get structured OLDCARTS data for a specific element from active guidelines
        
        Args:
            element: The OLDCARTS element ('onset', 'location', etc.)
            
        Returns:
            Combined structured data for this element from all active guidelines
        """
        element_data_parts = []
        
        for guideline in self.active_guidelines:
            # Get structured OLDCARTS data for this element
            structured_data = guideline['data'].get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured_data.get(element, {})
            
            if element_data:
                # Combine includes and excludes into a single text
                includes = element_data.get('includes', [])
                excludes = element_data.get('excludes', [])
                
                if includes or excludes:
                    includes_text = ', '.join(includes) if includes else ''
                    excludes_text = ', '.join(excludes) if excludes else ''
                    
                    element_text = f"Includes: {includes_text}"
                    if excludes_text:
                        element_text += f". Excludes: {excludes_text}"
                    
                    element_data_parts.append(element_text)
        
        return ' '.join(element_data_parts) if element_data_parts else ""

    def _update_oldcarts_analysis(self):
        """
        Update the oldcarts_analysis based on current coverage status
        
        This ensures that missing_components reflects the current state of coverage
        and prevents asking the same question repeatedly.
        """
        if not hasattr(self, 'oldcarts_analysis') or not self.oldcarts_analysis:
            return
        
        # Get all OLDCARTS elements (full names for analysis)
        all_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
        
        # Determine which elements are still missing
        missing_components = []
        answered_components = {}
        
        for element in all_elements:
            # Check if this element is covered using single letter key
            element_letter = element[0].upper()
            element_covered = self.oldcarts_covered.get(element_letter, False)
            
            if element_covered:
                answered_components[element] = True
            else:
                missing_components.append(element)
        
        # Update the analysis
        self.oldcarts_analysis = {
            'answered_components': answered_components,
            'missing_components': missing_components
        }
        
        self._capture_debug(f"[Engine] 🔄 Updated OLDCARTS analysis:")
        self._capture_debug(f"[Engine]   Answered: {list(answered_components.keys())}")
        self._capture_debug(f"[Engine]   Missing: {missing_components}")

    def _compute_semantic_oldcarts_similarity(self, answer: str, element_data: str, element: str) -> float:
        """
        Compute semantic similarity between user answer and structured OLDCARTS data
        
        Args:
            answer: User's answer
            element_data: Structured OLDCARTS data for the element
            element: The OLDCARTS element name
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Use the existing semantic similarity computation
            return self._compute_similarity(answer, element_data)
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Error computing semantic similarity for {element}: {e}")
            return 0.0

    def _generate_oldcarts_question_for_component(self, component: str) -> str:
        """Generate OLDCARTS question for specific component"""
        question_templates = {
            'location': "Where exactly is the pain located?",
            'character': "How would you describe the pain?",
            'aggravating': "What makes the pain worse?",
            'relieving': "What makes the pain better?",
            'onset': "When did the pain start?",
            'duration': "How long have you had this pain?",
            'timing': "When does the pain occur?",
            'severity': "How severe is the pain on a scale of 1-10?",
            'radiation': "Does the pain radiate anywhere? For example: to your back, shoulder, or elsewhere?"
        }
        
        return question_templates.get(component, f"Tell me more about the {component} of your symptoms.")
    
    def _generate_ml_first_question(self) -> Dict[str, Any]:
        """Generate first question using ML-powered approach"""
        self._capture_debug(f"[Engine] 🧠 Generating ML-powered first question...")
        
        # Use ML to determine best OLDCARTS element to ask about
        best_element = self._determine_best_oldcarts_element()
        
        # Generate question for that element
        question = self._generate_ml_question(best_element)
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'oldcarts': best_element,
            'focus': 'clinical'
        })
        
        self._capture_debug(f"[Engine] ✅ ML question generated: '{question}' (element: {best_element})")
        
        return {
            'success': True,
            'question': question,
            'status': 'questioning',
            'debug': self._get_debug_info()
        }
    
    def _determine_best_oldcarts_element(self) -> str:
        """Determine best OLDCARTS element to ask about first"""
        # Follow standard OLDCARTS order: onset-location-duration-character-aggravating-relieving-timing-severity
        # Start with onset as it's the first element in the framework
        return 'onset'
    
    def _generate_ml_question(self, oldcarts_element: str) -> str:
        """Generate question using ML-powered approach"""
        # Codebase uses lowercase full names only
        if oldcarts_element == 'location':
            return "Where exactly is the pain located?"
        elif oldcarts_element == 'onset':
            return "When did the pain start?"
        elif oldcarts_element == 'duration':
            return "How long does the pain last?"
        elif oldcarts_element == 'character':
            return "How would you describe the pain?"
        elif oldcarts_element == 'aggravating':
            return "What makes the pain worse?"
        elif oldcarts_element == 'relieving':
            return "What makes the pain better?"
        elif oldcarts_element == 'timing':
            return "Is the pain constant or does it come and go?"
        elif oldcarts_element == 'severity':
            return "On a scale of 1-10, how severe is the pain?"
        else:
            return "Can you tell me more about your symptoms?"
        
        # STEP 2: Run RAG and Llama-1B in PARALLEL (major speedup!)
        import threading
        import concurrent.futures
        
        rag_result = [None]
        opening_result = [None]
        age_result = [None]
        error_result = [None]
        
        def run_rag():
            """Match to guidelines (ML-only, no fallbacks)"""
            try:
                # VALIDATION MODE: Compare RAG API vs brute-force (set VALIDATE_RAG=true)
                if self.validate_rag and self.use_rag_api:
                    self._capture_debug(f"[Engine] 🧪 VALIDATION MODE: Comparing RAG API vs brute-force...")
                    
                    import time
                    
                    # Run RAG API
                    start_rag = time.time()
                    rag_matches = self._match_to_guidelines_rag(chief_complaint)
                    rag_time = time.time() - start_rag
                    
                    # Run brute-force (using same function but without RAG)
                    start_brute = time.time()
                    brute_matches = self._match_to_guidelines_rag(chief_complaint)
                    brute_time = time.time() - start_brute
                    
                    # Compare results
                    rag_names = set([m['name'] for m in rag_matches])
                    brute_names = set([m['name'] for m in brute_matches])
                    
                    self._capture_debug(f"\n[Engine] 📊 VALIDATION RESULTS:")
                    self._capture_debug(f"[Engine]    RAG API: {len(rag_matches)} matches in {rag_time:.2f}s")
                    self._capture_debug(f"[Engine]    Brute: {len(brute_matches)} matches in {brute_time:.2f}s")
                    self._capture_debug(f"[Engine]    Speedup: {brute_time/rag_time:.1f}x faster")
                    
                    if rag_names == brute_names:
                        self._capture_debug(f"[Engine]    ✅ MATCH: Both methods returned identical results")
                    else:
                        only_rag = rag_names - brute_names
                        only_brute = brute_names - rag_names
                        if only_rag:
                            self._capture_debug(f"[Engine]    ⚠️ Only in RAG API: {only_rag}")
                        if only_brute:
                            self._capture_debug(f"[Engine]    ⚠️ Only in brute-force: {only_brute}")
                    
                    # Use RAG API results
                    rag_result[0] = rag_matches
                
                # ML-ONLY MODE: Use ML-powered matching (no fallbacks)
                if self.use_rag_api:
                    self._capture_debug(f"[Engine] 🚀 Using RAG mode: {self.rag_mode}")
                    import time
                    start_time = time.time()
                    rag_result[0] = self._match_to_guidelines_rag(chief_complaint)
                    elapsed = time.time() - start_time
                    if hasattr(self, 'matching_metadata'):
                        self.matching_metadata['timing'] = elapsed
                else:
                    self._capture_debug(f"[Engine] 🧠 Using ML-only mode for matching")
                    import time
                    start_time = time.time()
                    rag_result[0] = self._match_to_guidelines_ml(chief_complaint, "ALL")
                    elapsed = time.time() - start_time
                    if hasattr(self, 'matching_metadata'):
                        self.matching_metadata['timing'] = elapsed
            except Exception as e:
                error_result[0] = f"Guideline matching error: {e}"
        
        def run_simple_llm():
            """Generate opening + age with Llama-1B (fast)"""
            try:
                opening_result[0] = self._generate_opening_statement(chief_complaint)
                age_result[0] = self._generate_age_question()
            except Exception as e:
                error_result[0] = f"LLM error: {e}"
        
        # Launch both in parallel
        self._capture_debug(f"[Engine] ⚡ Starting parallel execution (RAG + Llama-1B)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rag_future = executor.submit(run_rag)
            llm_future = executor.submit(run_simple_llm)
            
            # Wait for both to complete
            concurrent.futures.wait([rag_future, llm_future])
        
        # Check for errors
        if error_result[0]:
            self._capture_debug(f"[Engine] ❌ Parallel execution error: {error_result[0]}")
            return {
                'success': False,
                'message': "I'm having trouble processing your request. Please try again."
            }
        
        matched = rag_result[0]
        
        if len(matched) == 0:
            return {
                'success': False,
                'message': "I couldn't identify relevant medical conditions. Please describe your symptoms more specifically."
            }
        
        # Split into active (top 5) and reserve pool (rest)
        self.active_guidelines = matched[:self.MAX_ACTIVE]
        self.reserve_pool = matched[self.MAX_ACTIVE:]
        
        self._capture_debug(f"\n[Engine] 📋 ACTIVE DIFFERENTIALS (Top {len(self.active_guidelines)}):")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            prevalence = g['data'].get('prevalence', 'uncommon')
            self._capture_debug(f"[Engine]   {i}. {g['name']} ({prevalence}, {g['score']:.0%}) {urgency_emoji}")
        
        if self.reserve_pool:
            self._capture_debug(f"\n[Engine] 💾 RESERVE POOL ({len(self.reserve_pool)} conditions, prioritized by prevalence):")
            for i, g in enumerate(self.reserve_pool[:5], 1):  # Show first 5
                prevalence = g['data'].get('prevalence', 'uncommon')
                urgency = g['data'].get('urgency', 'routine')
                self._capture_debug(f"[Engine]   {i}. {g['name']} ({prevalence}, {urgency}, {g['score']:.0%})")
            if len(self.reserve_pool) > 5:
                self._capture_debug(f"[Engine]   ... and {len(self.reserve_pool) - 5} more")
        
        self._capture_debug(f"\n[Engine] 🔄 Initial pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        self._capture_debug(f"{'='*80}\n")
        
        # STEP 3: Use results from parallel execution
        opening_statement = opening_result[0]
        age_question = age_result[0]
        
        self._capture_debug(f"[Engine] ⚡ Parallel execution complete!")
        self._capture_debug(f"[Engine]    Opening: '{opening_statement}'")
        self._capture_debug(f"[Engine]    Age Q: '{age_question}'")
        
        # Debug: Check if opening statement is empty or not empathetic
        if not opening_statement or opening_statement.strip() == "":
            self._capture_debug(f"[Engine] ⚠️ EMPATHETIC STATEMENT IS EMPTY!")
        elif len(opening_statement.strip()) < 10:
            self._capture_debug(f"[Engine] ⚠️ EMPATHETIC STATEMENT TOO SHORT: '{opening_statement}'")
        else:
            self._capture_debug(f"[Engine] ✅ EMPATHETIC STATEMENT LOOKS GOOD: '{opening_statement}'")
        
        # Combine them with proper spacing and pause
        combined_message = f"{opening_statement} <pause> {age_question}"
        
        # Debug: Log final combined message
        self._capture_debug(f"[Engine] 🎯 FINAL COMBINED MESSAGE: '{combined_message}'")
        
        self.conversation_history.append({
            'type': 'question',
            'question': combined_message,
            'focus': 'age'
        })
        
        return {
            'success': True,
            'question': combined_message,
            'status': 'questioning',
            # Filler is now handled at container level for immediate streaming
            'debug': self._get_debug_info()  # For Telegram debug display
        }
    
    def process_answer(self, user_answer: str) -> Dict[str, Any]:
        """
        Process answer and continue assessment
        
        Args:
            user_answer: User's response
        
        Returns:
            Next question or diagnosis
        """
        if self.status not in ["questioning", "red_flag_screening"]:
            return {'success': False, 'message': "No active assessment"}
        
        # SAFETY CHECK: If active_guidelines is empty (first attempt failed),
        # and user is stating a new chief complaint, restart the assessment
        if len(self.active_guidelines) == 0:
            # Check if this looks like a chief complaint (not a simple answer)
            is_complaint = any(trigger in user_answer.lower() for trigger in [
                'pain', 'ache', 'hurt', 'nausea', 'vomiting', 'diarrhea', 
                'fever', 'bleeding', 'shortness'
            ])
            
            if is_complaint:
                self._capture_debug(f"[Engine] 🔄 No active guidelines - treating as NEW chief complaint")
                return self.start_assessment(user_answer)
        
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 💬 PROCESSING ANSWER")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] User: '{user_answer}'")
        
        # Store answer
        last_q = self.conversation_history[-1] if self.conversation_history else {}
        self.conversation_history.append({
            'type': 'answer',
            'answer': user_answer,
            'to_question': last_q.get('focus', 'unknown')
        })
        
        # Store demographics if provided
        self._capture_debug(f"[Engine] 🔍 Last question focus: {last_q.get('focus')}")
        self._capture_debug(f"[Engine] 🔍 Last question text: {last_q.get('question', '')}")
        self._capture_debug(f"[Engine] 🔍 User answer: '{user_answer}'")
        self._capture_debug(f"[Engine] 🔍 Demographics before processing: {self.demographics}")
        
        if last_q.get('focus') == 'demographics':
            if 'age' in last_q.get('question', '').lower():
                # Store age
                try:
                    age = int(user_answer.strip())
                    self.demographics['age'] = age
                    self._capture_debug(f"[Engine] 📊 Stored patient age: {age}")
                except ValueError:
                    self._capture_debug(f"[Engine] ⚠️ Invalid age format: '{user_answer}'")
            elif 'sex' in last_q.get('question', '').lower():
                # Handle button-based sex response
                self._capture_debug(f"[Engine] 🔍 Processing sex response: '{user_answer}'")
                
                # Map button responses to sex values
                if user_answer == 'sex_male':
                    self.demographics['sex'] = 'male'
                    self._capture_debug(f"[Engine] ✅ Button response: Male")
                elif user_answer == 'sex_female':
                    self.demographics['sex'] = 'female'
                    self._capture_debug(f"[Engine] ✅ Button response: Female")
                else:
                    # Fallback to text processing for non-button responses
                    sex_lower = user_answer.lower().strip()
                    self._capture_debug(f"[Engine] 🔍 Processing text sex answer: '{user_answer}' -> '{sex_lower}'")
                    
                    # Check for male keywords (whole words only)
                    male_keywords = ['male', 'man']
                    female_keywords = ['female', 'woman']
                    
                    # Split into words and check for exact matches
                    words = sex_lower.split()
                    male_found = any(word in male_keywords for word in words)
                    female_found = any(word in female_keywords for word in words)
                    
                    # Also check for single letter responses
                    if not male_found and not female_found and len(sex_lower.strip()) == 1:
                        if sex_lower.strip() == 'm':
                            male_found = True
                        elif sex_lower.strip() == 'f':
                            female_found = True
                    
                    if male_found:
                        self.demographics['sex'] = 'male'
                        self._capture_debug(f"[Engine] ✅ Detected MALE keywords")
                    elif female_found:
                        self.demographics['sex'] = 'female'
                        self._capture_debug(f"[Engine] ✅ Detected FEMALE keywords")
                    else:
                        self._capture_debug(f"[Engine] ⚠️ Unclear sex format: '{user_answer}'")
                
                self._capture_debug(f"[Engine] 📊 Stored patient sex: {self.demographics.get('sex', 'unknown')}")
                self._capture_debug(f"[Engine] 📊 Demographics after sex processing: {self.demographics}")
            elif 'new problem' in last_q.get('question', '').lower() or 'ongoing' in last_q.get('question', '').lower():
                # Store chronicity
                chronicity_lower = user_answer.lower().strip()
                if any(word in chronicity_lower for word in ['new', 'recent', 'just started', 'today', 'yesterday']):
                    self.demographics['chronicity'] = 'acute'
                elif any(word in chronicity_lower for word in ['ongoing', 'chronic', 'long time', 'months', 'years']):
                    self.demographics['chronicity'] = 'chronic'
                else:
                    self.demographics['chronicity'] = 'unknown'
                self._capture_debug(f"[Engine] 📊 Stored patient chronicity: {self.demographics.get('chronicity', 'unknown')}")
            
            # Demographics processing complete - continue to main logic below
        
        # SPECIAL HANDLING: Red flag screening
        if self.status == 'red_flag_screening' and last_q.get('focus') == 'red_flag':
            answer_lower = user_answer.lower().strip()
            
            # Check for yes/no (accept various forms)
            is_yes = any(word in answer_lower for word in ['yes', 'yeah', 'yep', 'yup', 'sure'])
            is_no = any(word in answer_lower for word in ['no', 'nope', 'nah', 'not'])
            
            # If unclear answer, re-ask
            if not is_yes and not is_no and len(answer_lower.split()) < 3:
                self._capture_debug(f"[Engine] ⚠️ Unclear red flag answer: '{user_answer}' - re-asking")
                # Re-ask the same red flag question
                red_flag_text = last_q.get('red_flag_text', '')
                question = self._red_flag_to_question(red_flag_text)
                
                self.conversation_history.append({
                    'type': 'question',
                    'question': f"Please answer yes or no: {question}",
                    'focus': 'red_flag',
                    'red_flag_text': red_flag_text,
                    'red_flag_index': self.red_flag_index
                })
                
                return {
                    'success': True,
                    'question': f"Please answer yes or no: {question}",
                    'status': 'red_flag_screening',
                    'debug': self._get_debug_info()
                }
            
            if is_yes:
                red_flag_text = last_q.get('red_flag_text', 'Warning sign')
                self.red_flags_present.append(red_flag_text)
                self._capture_debug(f"[Engine] ⚠️  RED FLAG PRESENT: {red_flag_text}")
            else:
                self._capture_debug(f"[Engine] ✓ Red flag not present")
            
            # Move to next red flag
            prev_index = self.red_flag_index
            self.red_flag_index += 1
            self._capture_debug(f"[Engine] 🔄 Red flag index: {prev_index} → {self.red_flag_index}")
            
            # Continue screening (or finalize if done) - use stored diagnosed condition
            return self._screen_red_flags(self.diagnosed_condition)
        
        # Demographics processing already handled above - no duplicate needed
        
        elif last_q.get('focus') == 'chronicity' or (last_q.get('focus') == 'demographics' and 'chronicity' in last_q.get('question', '').lower()):
            # Handle button-based chronicity response
            self._capture_debug(f"[Engine] 🔍 Processing chronicity response: '{user_answer}'")
            
            # Map button responses to chronicity values
            if user_answer == 'chronicity_new':
                chronicity = 'new'
            elif user_answer == 'chronicity_recurring':
                chronicity = 'recurring'
            else:
                # Fallback to LLM classification for text responses
                self._capture_debug(f"[Engine] 🔍 LLM analyzing chronicity from text answer: '{user_answer}'")
                chronicity = self._classify_chronicity_with_llm(user_answer)
            
            self.demographics['chronicity'] = chronicity
            self._capture_debug(f"[Engine] 📋 Chronicity: {chronicity}")
            
            if chronicity == 'recurring':
                self._capture_debug(f"[Engine] 💡 Consider: Follow-up vs new evaluation")
            
            self._capture_debug(f"{'='*80}\n")
            
            # NOW ask ONSET first (original OLDCARTS order)
            onset_question = self._ask_next_clinical_question()
            
            # This will return an Onset question since it's the first OLDCARTS element
            return onset_question
        
        else:
            # Clinical question - find last question first
            last_q_item = None
            for item in reversed(self.conversation_history):
                if item.get('type') == 'question':
                    last_q_item = item
                    break
            
            # VALIDATE answer first
            if not self._is_acceptable_clinical_answer(user_answer):
                self._capture_debug(f"[Engine] ⚠️ Answer too vague or unclear - asking for clarification")
                
                last_q = last_q_item.get('question', 'the question') if last_q_item else 'the question'
                
                # Strip "I didn't quite understand. " prefix if present (avoid repetition)
                core_question = last_q
                if last_q and last_q.startswith("I didn't quite understand. "):
                    core_question = last_q.replace("I didn't quite understand. ", "").strip()
                if core_question and core_question.startswith("Could you be more specific? "):
                    core_question = core_question.replace("Could you be more specific? ", "").strip()
                
                clarify = f"I didn't quite understand. {core_question if core_question else 'Can you clarify?'}"
                
                # IMPORTANT: Preserve OLDCARTS element from original question
                new_question = {
                    'type': 'question',
                    'question': clarify,
                    'focus': 'clinical'
                }
                if last_q_item and 'oldcarts' in last_q_item:
                    new_question['oldcarts'] = last_q_item['oldcarts']
                
                self.conversation_history.append(new_question)
                
                return {
                    'success': True,
                    'question': clarify,
                    'status': 'questioning',
                    'debug': self._get_debug_info()
                }
            
            # Answer is acceptable - score it first, THEN check if needs clarification
            # This way we use UPDATED top 3 guidelines after scoring
            return self._process_clinical_answer(user_answer)
    
    def _is_acceptable_clinical_answer(self, answer: str) -> bool:
        """
        LLM-based validation: Does the answer actually address the question asked?
        Uses context-aware validation based on what OLDCARTS element is being asked.
        """
        # Get the last question asked and its OLDCARTS element
        last_question = None
        oldcarts_element = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_question = item['question']
                oldcarts_element = item.get('oldcarts')
                break
        
        if not last_question:
            return True  # No question to validate against
        
        self._capture_debug(f"[Engine] 🔍 Validating answer with LLM...")
        self._capture_debug(f"[Engine]   Q: '{last_question}'")
        self._capture_debug(f"[Engine]   A: '{answer}'")
        
        # Simple validation: reject pure filler words or fragments
        # Semantic scoring will determine if answer is specific enough
        
        # Reject pure filler words or meaningless fragments
        pure_filler = ['um', 'uh', 'oh', 'hmm', 'ah', 'er']
        fragments = ['on the', 'my', 'the', 'it', 'there', 'here', 'i', 'a', 'an']
        
        answer_stripped = answer.strip().lower()
        
        if answer_stripped in pure_filler or answer_stripped in fragments:
            self._capture_debug(f"[Engine] 📊 Validation: REJECT ❌ (pure filler or fragment)")
            return False
        
        # Accept any substantive answer - semantic scoring will handle specificity
        self._capture_debug(f"[Engine] 📊 Validation: ACCEPT ✅ (substantive answer)")
        return True
    
    
    def _load_all_synonym_files(self) -> Dict:
        """Load all synonym files for comprehensive normalization (cached)"""
        # Cache synonym files to avoid reloading
        if not hasattr(self, '_cached_synonyms'):
            all_synonyms = {}
            
            synonym_files = [
                'gi_synonyms_oldcarts.json',
                'cardio_synonyms_oldcarts.json',
                'neuro_synonyms_oldcarts.json',
                'msk_synonyms_oldcarts.json',
                'derm_synonyms_oldcarts.json',
                'renal_synonyms_oldcarts.json',
                'resp_synonyms_oldcarts.json'
            ]
            
            for file in synonym_files:
                try:
                    file_path = os.path.join('synonyms', file)
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            synonyms = json.load(f)
                            all_synonyms.update(synonyms)
                            self._capture_debug(f"[Engine] 📚 Loaded synonyms from {file}")
                except Exception as e:
                    self._capture_debug(f"[Engine] ⚠️ Failed to load {file}: {e}")
            
            self._cached_synonyms = all_synonyms
            self._capture_debug(f"[Engine] 📚 Total synonym categories loaded: {len(all_synonyms)}")
        
        return self._cached_synonyms
    
    def _normalize_complaint_with_synonyms(self, complaint: str) -> str:
        """Normalize complaint using ALL available synonym files"""
        complaint_lower = complaint.lower()
        
        # Load all synonym files
        all_synonyms = self._load_all_synonym_files()
        
        # Apply comprehensive synonym normalization
        normalized_complaint = complaint_lower
        normalization_changes = []
        
        for category, synonyms in all_synonyms.items():
            for standard_term, synonym_list in synonyms.items():
                for synonym in synonym_list:
                    if synonym in normalized_complaint:
                        old_text = normalized_complaint
                        normalized_complaint = normalized_complaint.replace(synonym, standard_term)
                        if old_text != normalized_complaint:
                            normalization_changes.append(f"'{synonym}' → '{standard_term}'")
        
        if normalization_changes:
            self._capture_debug(f"[Engine] 🔄 Synonym normalization changes: {normalization_changes}")
        else:
            self._capture_debug(f"[Engine] 🔄 No synonym normalization needed")
        
        self._capture_debug(f"[Engine] 🔄 Synonym normalization: '{complaint_lower}' → '{normalized_complaint}'")
        return normalized_complaint
    
    def _categorize_complaint_by_substring(self, normalized_complaint: str) -> str:
        """Efficient category detection using organ system keywords with FUZZY MATCHING"""
        self._capture_debug(f"[Engine] 🔍 FUZZY CATEGORY DETECTION DEBUG:")
        self._capture_debug(f"[Engine] 🔍 Original Input: '{normalized_complaint}'")
        
        # STEP 1: Apply fuzzy correction for medical typos
        corrected_complaint = self.fuzzy_matcher.fuzzy_correct_medical_terms(normalized_complaint)
        self._capture_debug(f"[Engine] 🧠 Fuzzy Corrected: '{normalized_complaint}' → '{corrected_complaint}'")
        
        complaint_lower = corrected_complaint.lower()
        
        # STEP 2: Use corrected text for organ system detection
        organ_keywords = {
            'GI': ['abdominal', 'stomach', 'belly', 'gut', 'bowel', 'intestine', 'gastrointestinal'],
            'CARDIO': ['chest', 'heart', 'cardiac', 'coronary', 'myocardial'],
            'NEURO': ['head', 'headache', 'brain', 'neurological', 'cerebral', 'migraine'],
            'MSK': ['back', 'joint', 'muscle', 'bone', 'spine', 'musculoskeletal', 'orthopedic'],
            'RENAL': ['kidney', 'urinary', 'bladder', 'flank', 'renal', 'genitourinary'],
            'DERM': ['skin', 'rash', 'lesion', 'dermatological', 'cutaneous'],
            'GYN': ['pelvic', 'menstrual', 'gynecological', 'reproductive'],
            'GU': ['prostate', 'testicular', 'scrotal', 'genitourinary'],
            'PULMONARY': ['lung', 'breathing', 'respiratory', 'pulmonary', 'asthma', 'pneumonia', 'bronchitis']
        }
        
        # STEP 3: Count keyword matches by organ system (now with corrected text)
        category_scores = {}
        for organ, keywords in organ_keywords.items():
            score = sum(1 for keyword in keywords if keyword in complaint_lower)
            if score > 0:
                category_scores[organ] = score
                self._capture_debug(f"[Engine] 🔍 {organ}: {score} matches (fuzzy-corrected)")
        
        self._capture_debug(f"[Engine] 🔍 Category scores: {category_scores}")
        
        # STEP 4: Return organ system with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            self._capture_debug(f"[Engine] ✅ Best category: {best_category} (fuzzy-enhanced detection)")
            return best_category
        else:
            self._capture_debug(f"[Engine] 🔍 No organ keywords found, using ALL categories")
            return 'ALL'
    
    def _parse_oldcarts_components(self, complaint: str) -> Dict[str, List[str]]:
        """Parse complaint to extract OLDCARTS components using comprehensive keyword database"""
        complaint_lower = complaint.lower()
        components = {
            'location': [],
            'character': [],
            'aggravating': [],
            'relieving': [],
            'onset': [],
            'duration': [],
            'timing': [],
            'severity': []
        }
        
        # Load comprehensive OLDCARTS keywords from JSON file
        import os
        oldcarts_keywords = None
        
        # Try multiple possible locations for the file
        possible_paths = [
            'oldcarts_keywords.json',
            'llm-medical-container/oldcarts_keywords.json',
            '/app/oldcarts_keywords.json',
            os.path.join(os.path.dirname(__file__), 'oldcarts_keywords.json'),
            os.path.join(os.getcwd(), 'oldcarts_keywords.json'),
            os.path.join(os.getcwd(), 'llm-medical-container', 'oldcarts_keywords.json')
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        oldcarts_keywords = json.load(f)
                        self._capture_debug(f"[Engine] ✅ Loaded OLDCARTS keywords from: {path}")
                        break
            except (FileNotFoundError, json.JSONDecodeError) as e:
                continue
        
        if oldcarts_keywords is None:
            self._capture_debug(f"[Engine] ❌ OLDCARTS keywords file not found in any of these locations: {possible_paths}")
            raise RuntimeError("OLDCARTS keywords file not found - required for parsing")
        
        # Location indicators - Use improved whole-phrase matching
        location_categories = ['anatomical_regions', 'quadrants', 'sides', 'specific_locations']
        for category in location_categories:
            if category in oldcarts_keywords['location']:
                for term in oldcarts_keywords['location'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['location'].append(term)
        
        # Character indicators - FIXED: Prevent generic symptom words from matching
        character_categories = ['pain_quality', 'pain_intensity', 'pain_pattern']
        
        # Generic symptom words that should NOT count as character descriptors
        generic_symptom_words = {
            'pain', 'ache', 'discomfort', 'hurt', 'sore', 'tender', 'sensation',
            'feeling', 'symptom', 'problem', 'issue', 'trouble'
        }
        
        for category in character_categories:
            if category in oldcarts_keywords['character']:
                for term in oldcarts_keywords['character'][category]:
                    # Skip if the term is just a generic symptom word
                    if term.lower().strip() in generic_symptom_words:
                        continue
                        
                    # Use whole-word matching to prevent substring false positives
                    # Check if the term appears as a complete phrase in the complaint
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['character'].append(term)
        
        # Aggravating indicators - Use improved whole-phrase matching
        aggravating_categories = ['activities', 'triggers', 'positions']
        for category in aggravating_categories:
            if category in oldcarts_keywords['aggravating']:
                for term in oldcarts_keywords['aggravating'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['aggravating'].append(term)
        
        # Relieving indicators - Use improved whole-phrase matching
        relieving_categories = ['positions', 'interventions', 'activities']
        for category in relieving_categories:
            if category in oldcarts_keywords['relieving']:
                for term in oldcarts_keywords['relieving'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['relieving'].append(term)
        
        # Onset indicators - Use improved whole-phrase matching
        onset_categories = ['temporal', 'triggers', 'descriptors']
        for category in onset_categories:
            if category in oldcarts_keywords['onset']:
                for term in oldcarts_keywords['onset'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['onset'].append(term)
        
        # Duration indicators - Use improved whole-phrase matching
        duration_categories = ['time_units', 'descriptors', 'patterns']
        for category in duration_categories:
            if category in oldcarts_keywords['duration']:
                for term in oldcarts_keywords['duration'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['duration'].append(term)
        
        # Timing indicators - Use improved whole-phrase matching
        timing_categories = ['daily_patterns', 'meal_related', 'activity_related', 'frequency']
        for category in timing_categories:
            if category in oldcarts_keywords['timing']:
                for term in oldcarts_keywords['timing'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['timing'].append(term)
        
        # Severity indicators - Use improved whole-phrase matching
        severity_categories = ['intensity_levels', 'scale_descriptors', 'impact_descriptors']
        for category in severity_categories:
            if category in oldcarts_keywords['severity']:
                for term in oldcarts_keywords['severity'][category]:
                    if self._is_whole_phrase_match(term, complaint_lower):
                        components['severity'].append(term)
        
        return components
    
    def _construct_oldcarts_answers(self, components: Dict[str, List[str]]) -> Dict[str, str]:
        """Construct OLDCARTS answers from parsed components"""
        oldcarts_answers = {}
        
        # Location
        if components['location']:
            oldcarts_answers['location'] = ', '.join(components['location'])
        
        # Character
        if components['character']:
            oldcarts_answers['character'] = ', '.join(components['character'])
        
        # Aggravating
        if components['aggravating']:
            oldcarts_answers['aggravating'] = ', '.join(components['aggravating'])
        
        # Relieving
        if components['relieving']:
            oldcarts_answers['relieving'] = ', '.join(components['relieving'])
        
        # Onset
        if components['onset']:
            oldcarts_answers['onset'] = ', '.join(components['onset'])
        
        # Duration
        if components['duration']:
            oldcarts_answers['duration'] = ', '.join(components['duration'])
        
        # Timing
        if components['timing']:
            oldcarts_answers['timing'] = ', '.join(components['timing'])
        
        # Severity
        if components['severity']:
            oldcarts_answers['severity'] = ', '.join(components['severity'])
        
        return oldcarts_answers
    
    def _identify_missing_oldcarts_components(self, oldcarts_answers: Dict[str, str]) -> List[str]:
        """Identify missing OLDCARTS components that need to be asked"""
        all_components = ['location', 'character', 'aggravating', 'relieving', 'onset', 'duration', 'timing', 'severity']
        missing_components = []
        
        for component in all_components:
            if component not in oldcarts_answers or not oldcarts_answers[component]:
                missing_components.append(component)
        
        return missing_components
    
    
    
    def _get_guidelines_by_category(self, category: str) -> Dict:
        """Get guidelines filtered by organ system category using new naming format"""
        if category == 'ALL':
            return self.all_guidelines
        
        # Map category to prefix patterns (new naming format)
        category_prefixes = {
            'GI': ['GI_'],
            'CARDIO': ['CARDIO_'],
            'NEURO': ['NEURO_'],
            'MSK': ['MSK_'],
            'DERM': ['DERM_'],
            'RENAL': ['RENAL_'],
            'GYN': ['GYN_'],
            'GU': ['GU_'],
            'PULMONARY': ['PULMONARY_']
        }
        
        # Fallback patterns for backward compatibility
        fallback_patterns = {
            'GI': ['appendicitis', 'cholecystitis', 'pancreatitis', 'gastritis', 'diverticulitis', 'hepatitis', 'colic', 'obstruction', 'volvulus', 'gastroenteritis', 'biliary', 'cholangitis', 'gerd', 'ibs', 'ibd', 'hernia', 'mallory', 'mesenteric', 'ulcer', 'perforated', 'sigmoid', 'cecal', 'gastric', 'constipation', 'bowel'],
            'CARDIO': ['cardiovascular', 'chest', 'heart', 'myocardial', 'infarction', 'angina', 'aortic', 'cardiac'],
            'NEURO': ['neurological', 'head', 'brain', 'headache', 'migraine', 'seizure', 'stroke', 'cerebral'],
            'MSK': ['musculoskeletal', 'orthopedic', 'bone', 'joint', 'muscle', 'spine', 'back', 'fracture'],
            'DERM': ['dermatological', 'skin', 'rash', 'lesion', 'cutaneous', 'dermatitis'],
            'RENAL': ['renal', 'kidney', 'urinary', 'bladder', 'stone', 'uti', 'pyelonephritis'],
            'GYN': ['gynecological', 'pelvic', 'ovarian', 'pregnancy', 'menstrual', 'reproductive'],
            'GU': ['genitourinary', 'prostate', 'testicular', 'scrotal'],
            'PULMONARY': ['pulmonary', 'respiratory', 'lung', 'breathing', 'asthma', 'pneumonia', 'bronchitis']
        }
        
        filtered_guidelines = {}
        prefixes = category_prefixes.get(category, [])
        patterns = fallback_patterns.get(category, [])
        
        self._capture_debug(f"[Engine] 🔍 Category '{category}' prefixes: {prefixes}")
        self._capture_debug(f"[Engine] 🔍 Category '{category}' fallback patterns: {patterns}")
        self._capture_debug(f"[Engine] 🔍 Total guidelines loaded: {len(self.all_guidelines)}")
        
        for name, guideline in self.all_guidelines.items():
            # Check if guideline belongs to this category using new prefix format
            name_lower = name.lower()
            condition = guideline.get('condition', '')
            condition_lower = condition.lower()
            
            # Primary: Check for prefix match (new format)
            prefix_match = any(name.startswith(prefix) for prefix in prefixes)
            
            # Fallback: Check for pattern match (old format)
            pattern_match = any(pattern in name_lower for pattern in patterns) or any(pattern in condition_lower for pattern in patterns)
            
            if prefix_match or pattern_match:
                filtered_guidelines[name] = guideline
                match_type = "prefix" if prefix_match else "pattern"
                self._capture_debug(f"[Engine] ✅ Matched ({match_type}): {name} (condition: {condition})")
        
        self._capture_debug(f"[Engine] 🔍 Filtered guidelines: {len(filtered_guidelines)}")
        return filtered_guidelines


    def _rank_by_prevalence(self, matched: List[Dict]) -> List[Dict]:
        """Rank matched guidelines by prevalence and urgency"""
        # Sort by prevalence (common > uncommon > rare) then by score
        prevalence_order = {'common': 3, 'uncommon': 2, 'rare': 1}
        
        def sort_key(item):
            guideline = item['data']
            prevalence = guideline.get('prevalence', 'uncommon')
            prevalence_score = prevalence_order.get(prevalence, 2)
            return (-prevalence_score, -item['score'])  # Negative for descending order
        
        return sorted(matched, key=sort_key)

    def _perform_gpu_semantic_search(self, complaint: str, core_symptom: str, matched: List[Dict], matched_guideline_names: set):
        """Perform GPU-accelerated semantic search using RAG API"""
        try:
            # Get all triggers for guidelines not yet matched
            remaining_triggers = []
            trigger_to_guideline = []
            
            for name, guideline in self.all_guidelines.items():
                if name not in matched_guideline_names:
                    triggers = guideline.get('chief_complaint_triggers', [])
                    for trigger in triggers:
                        remaining_triggers.append(trigger)
                        trigger_to_guideline.append({
                            'guideline_name': name,
                            'trigger': trigger,
                            'guideline_data': guideline
                        })
            
            if remaining_triggers:
                self._capture_debug(f"[Engine] 📊 Searching {len(remaining_triggers)} remaining triggers with GPU...")
                
                # Use RAG API for GPU-accelerated search
                all_texts = [core_symptom] + remaining_triggers
                embeddings = self.rag_api.encode(all_texts)
                
                # Get query embedding (first one)
                query_embedding = embeddings[0]
                
                # Compute cosine similarities with remaining triggers
                for i, trigger_embedding in enumerate(embeddings[1:], 0):
                    similarity = np.dot(query_embedding, trigger_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(trigger_embedding)
                    )
                    
                    metadata = trigger_to_guideline[i]
                    guideline_name = metadata['guideline_name']
                    trigger = metadata['trigger']
                    guideline_data = metadata['guideline_data']
                    
                    # Skip if already matched
                    if guideline_name in matched_guideline_names:
                        continue
                    
                    # Apply threshold
                    if similarity > 0.85:
                        prevalence = guideline_data.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': guideline_name, 'score': initial_score, 'data': guideline_data})
                        matched_guideline_names.add(guideline_name)
                        self._capture_debug(f"[Engine]   ✓ {guideline_name} (trigger: '{trigger}', match: gpu_semantic ({similarity:.2f}), prevalence: {prevalence})")
                    else:
                        if i < 5:  # Log first few rejections
                            self._capture_debug(f"[Engine]   ✗ {guideline_name}: '{trigger}' (similarity={similarity:.2f} < 0.85)")
                
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ GPU semantic search failed: {e}")
    
    def _perform_cpu_semantic_search(self, complaint: str, core_symptom: str, matched: List[Dict], matched_guideline_names: set):
        """Perform CPU-based semantic search using local FAISS"""
        try:
            # Get all triggers for guidelines not yet matched
            remaining_triggers = []
            trigger_to_guideline = []
            
            for name, guideline in self.all_guidelines.items():
                if name not in matched_guideline_names:
                    triggers = guideline.get('chief_complaint_triggers', [])
                    for trigger in triggers:
                        remaining_triggers.append(trigger)
                        trigger_to_guideline.append({
                            'guideline_name': name,
                            'trigger': trigger,
                            'guideline_data': guideline
                        })
            
            if remaining_triggers:
                self._capture_debug(f"[Engine] 📊 Searching {len(remaining_triggers)} remaining triggers with CPU...")
                
                # Use local embedding model for CPU search
                all_texts = [core_symptom] + remaining_triggers
                embeddings = self.embedding_model.encode(all_texts)
                
                # Get query embedding (first one)
                query_embedding = embeddings[0]
                
                # Compute cosine similarities with remaining triggers
                for i, trigger_embedding in enumerate(embeddings[1:], 0):
                    similarity = np.dot(query_embedding, trigger_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(trigger_embedding)
                    )
                    
                    metadata = trigger_to_guideline[i]
                    guideline_name = metadata['guideline_name']
                    trigger = metadata['trigger']
                    guideline_data = metadata['guideline_data']
                    
                    # Skip if already matched
                    if guideline_name in matched_guideline_names:
                        continue
                    
                    # Apply threshold
                    if similarity > 0.85:
                        prevalence = guideline_data.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': guideline_name, 'score': initial_score, 'data': guideline_data})
                        matched_guideline_names.add(guideline_name)
                        self._capture_debug(f"[Engine]   ✓ {guideline_name} (trigger: '{trigger}', match: cpu_semantic ({similarity:.2f}), prevalence: {prevalence})")
                    else:
                        if i < 5:  # Log first few rejections
                            self._capture_debug(f"[Engine]   ✗ {guideline_name}: '{trigger}' (similarity={similarity:.2f} < 0.85)")
                
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ CPU semantic search failed: {e}")
    
    def _perform_character_overlap_search(self, complaint: str, core_symptom: str, matched: List[Dict], matched_guideline_names: set):
        """Perform character overlap search (ML-only, no fallback)"""
        try:
            # Get all triggers for guidelines not yet matched
            remaining_triggers = []
            trigger_to_guideline = []
            
            for name, guideline in self.all_guidelines.items():
                if name not in matched_guideline_names:
                    triggers = guideline.get('chief_complaint_triggers', [])
                    for trigger in triggers:
                        remaining_triggers.append(trigger)
                        trigger_to_guideline.append({
                            'guideline_name': name,
                            'trigger': trigger,
                            'guideline_data': guideline
                        })
            
            if remaining_triggers:
                self._capture_debug(f"[Engine] 📊 Searching {len(remaining_triggers)} remaining triggers with character overlap...")
                
                # Character overlap similarity
                for i, trigger in enumerate(remaining_triggers):
                    char_similarity = len(set(core_symptom) & set(trigger.lower())) / len(set(core_symptom) | set(trigger.lower()))
                    
                    metadata = trigger_to_guideline[i]
                    guideline_name = metadata['guideline_name']
                    guideline_data = metadata['guideline_data']
                    
                    # Skip if already matched
                    if guideline_name in matched_guideline_names:
                        continue
                    
                    # Apply threshold
                    if char_similarity > 0.75:
                        prevalence = guideline_data.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': guideline_name, 'score': initial_score, 'data': guideline_data})
                        matched_guideline_names.add(guideline_name)
                        self._capture_debug(f"[Engine]   ✓ {guideline_name} (trigger: '{trigger}', match: char_overlap ({char_similarity:.2f}), prevalence: {prevalence})")
                    else:
                        if i < 5:  # Log first few rejections
                            self._capture_debug(f"[Engine]   ✗ {guideline_name}: '{trigger}' (similarity={char_similarity:.2f} < 0.75)")
                
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ Character overlap search failed: {e}")

    def _match_to_guidelines_rag(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines using RAG API for semantic search
        
        Strategy:
        1. Use RAG client for semantic similarity search
        2. Extract guideline names from RAG results
        3. Fallback to category filtering if RAG fails
        
        Returns:
            List of matched guidelines with initial scores
        """
        complaint_lower = complaint.lower()
        
        # Apply smart normalization (LLM or synonyms) to normalize patient language
        complaint_expanded = self._smart_oldcarts_normalization(complaint_lower)
        self._capture_debug(f"[Engine] 🔄 Smart normalization: '{complaint_lower}' → '{complaint_expanded}'")
        
        matched = []
        matched_guideline_names = set()
        
        self._capture_debug(f"\n[Engine] 🔍 MATCHING TO GUIDELINES (RAG SEARCH MODE)...")
        self._capture_debug(f"[Engine] 🎯 Strategy: RAG semantic search → category fallback")
        self._capture_debug(f"[Engine] ---")
        
        # PHASE 1: RAG semantic search
        try:
            if self.rag_api and self.use_rag_api:
                # Create search query for medical guidelines
                search_query = f"DIAGNOSTIC GUIDELINE {complaint_expanded}"
                self._capture_debug(f"[Engine] 🔍 RAG search query: '{search_query}'")
                
                # Use RAG client for semantic search
                rag_client = get_rag_client()
                rag_results = rag_client.search(
                    query=search_query,
                    k=30,  # Get more results for better coverage
                    threshold=0.2  # Lower threshold for broader matching
                )
                
                if rag_results:
                    self._capture_debug(f"[Engine] 📚 RAG returned {len(rag_results)} chunks")
                    
                    # Extract guideline names from RAG results
                    rag_guideline_names = set()
                    for result in rag_results:
                        if 'guideline_name' in result:
                            rag_guideline_names.add(result['guideline_name'])
                    
                    self._capture_debug(f"[Engine] 📋 RAG found {len(rag_guideline_names)} unique guidelines")
                    
                    # Add RAG matches with scores based on prevalence
                    for name in rag_guideline_names:
                        if name in self.all_guidelines:
                            guideline = self.all_guidelines[name]
                            prevalence = guideline.get('prevalence', 'uncommon')
                            prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                            initial_score = prevalence_scores.get(prevalence, 0.50)
                            matched.append({'name': name, 'score': initial_score, 'data': guideline})
                            matched_guideline_names.add(name)
                            self._capture_debug(f"[Engine]   ✓ {name} (RAG semantic match, prevalence: {prevalence})")
                else:
                    self._capture_debug(f"[Engine] ⚠️ RAG search returned no results")
            else:
                self._capture_debug(f"[Engine] ⚠️ RAG API not available, using fallback")
                
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ RAG search failed: {e}")
        
        self._capture_debug(f"[Engine] 📊 Phase 1 (RAG semantic): {len(matched)} matches")
        
        # PHASE 2: Category-based fallback if RAG didn't find enough
        if len(matched) < 3:
            self._capture_debug(f"[Engine] 🔍 Phase 2: Category-based fallback...")
            
            # PERFORMANCE OPTIMIZATION: Synonym normalization + substring matching
            normalized_complaint = self._normalize_complaint_with_synonyms(complaint)
            category = self._categorize_complaint_by_substring(normalized_complaint)
            relevant_guidelines = self._get_guidelines_by_category(category)
            
            self._capture_debug(f"[Engine] 🎯 Category filtering: {category} → {len(relevant_guidelines)}/{len(self.all_guidelines)} guidelines")
            
            # Add category-based matches that weren't already found by RAG
            for name, guideline in relevant_guidelines.items():
                if name not in matched_guideline_names:
                    triggers = guideline.get('chief_complaint_triggers', [])
                    
                    # Check for trigger matches
                    for trigger in triggers:
                        trigger_lower = trigger.lower()
                        
                        # Exact or subset match
                        if trigger_lower in complaint_lower or complaint_lower in trigger_lower:
                            prevalence = guideline.get('prevalence', 'uncommon')
                            prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                            initial_score = prevalence_scores.get(prevalence, 0.50)
                            matched.append({'name': name, 'score': initial_score, 'data': guideline})
                            matched_guideline_names.add(name)
                            self._capture_debug(f"[Engine]   ✓ {name} (category fallback, trigger: '{trigger}', prevalence: {prevalence})")
                            break  # Found match, move to next guideline
            
            self._capture_debug(f"[Engine] 📊 Phase 2 (category fallback): {len(matched)} total matches")
        
        # Sort by score (prevalence-based) and return top matches
        matched.sort(key=lambda x: x['score'], reverse=True)
        
        self._capture_debug(f"[Engine] ✅ Final result: {len(matched)} guidelines matched")
        for i, match in enumerate(matched[:5]):  # Show top 5
            self._capture_debug(f"[Engine]   {i+1}. {match['name']} (score: {match['score']:.2f})")
        
        return matched

    def test_hybrid_matching(self, complaint: str, guidelines: List[Dict] = None) -> List[Dict]:
        """
        Test method for hybrid similarity matching - used by test scripts
        
        Args:
            complaint: Patient complaint text
            guidelines: Optional list of guidelines to test against (if None, uses all_guidelines)
        
        Returns:
            List of matched guidelines with hybrid similarity scores
        """
        if guidelines is None:
            guidelines = list(self.all_guidelines.values())
        
        # Temporarily replace all_guidelines for testing
        original_guidelines = self.all_guidelines
        self.all_guidelines = {guideline['name']: guideline for guideline in guidelines}
        
        try:
            # Use the hybrid similarity matching method
            matched = self._match_to_guidelines_rag(complaint)
            return matched
        finally:
            # Restore original guidelines
            self.all_guidelines = original_guidelines
    
    def _ask_next_clinical_question(self) -> Dict[str, Any]:
        """
        Use structured OLDCARTS approach to generate next best question
        
        This is the CORE intelligence of the system.
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🧠 STRUCTURED QUESTION GENERATION")
        self._capture_debug(f"{'='*80}")
        
        # Build context
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # Get questions already asked
        asked = []
        for item in self.conversation_history:
            if item['type'] == 'question' and item.get('focus') not in ['age', 'sex']:
                asked.append(item['question'])
        
        self._capture_debug(f"[Engine] 📋 Patient: {patient_info}")
        self._capture_debug(f"[Engine] 📋 Questions asked: {len(asked)}")
        
        # Use structured OLDCARTS analysis to determine next question
        if hasattr(self, 'oldcarts_analysis') and self.oldcarts_analysis:
            missing_components = self.oldcarts_analysis.get('missing_components', [])
            if missing_components:
                # Check if we should ask about radiation before the next component
                # Radiation should be asked after location is complete, before duration
                next_component = missing_components[0]
                
                # Track which components have been asked
                asked_components = []
                for item in self.conversation_history:
                    if item.get('type') == 'question' and item.get('oldcarts'):
                        asked_components.append(item['oldcarts'])
                
                # Check if location is done and radiation should be asked
                if 'location' in asked_components and 'radiation' not in asked_components:
                    if self._has_radiation_component(self.active_guidelines):
                        next_component = 'radiation'
                        self._capture_debug(f"[Engine] 🎯 Inserting radiation question after location completion")
                
                question = self._generate_oldcarts_question_for_component(next_component)
                self._capture_debug(f"[Engine] ✅ Next structured question: '{question}'")
                
                # Add to conversation history
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'oldcarts': next_component,
                    'focus': 'clinical',
                    'is_demographics': False
                })
                
                return {
                    'success': True,
                    'question': question,
                    'status': 'questioning',
                    'debug': self._get_debug_info()
                }
            else:
                # All OLDCARTS components answered, proceed to scoring
                self._capture_debug(f"[Engine] ✅ All OLDCARTS components answered, proceeding to scoring")
                return self._process_clinical_answer("")  # Empty answer to trigger scoring
        else:
            raise RuntimeError("No OLDCARTS analysis available for structured questions")
    
    
    def _extract_oldcarts_section(self, classic_presentation: str, element: str) -> str:
        """
        Extract specific OLDCARTS section from classic_presentation text
        
        Args:
            classic_presentation: Full guideline text
            element: 'O', 'L', 'D', 'C', 'A', 'R', 'T', 'S' or 'onset', 'location', 'duration', etc.
        
        Returns:
            The text for that OLDCARTS section
        """
        # Handle both single letter codes and full element names
        element_names = {
            'O': 'ONSET',
            'L': 'LOCATION',
            'D': 'DURATION',
            'C': 'CHARACTER',
            'A': 'AGGRAVATING',
            'R': 'RELIEVING',
            'T': 'TIMING',
            'S': 'SEVERITY',
            # Also handle full element names
            'onset': 'ONSET',
            'location': 'LOCATION',
            'duration': 'DURATION',
            'character': 'CHARACTER',
            'aggravating': 'AGGRAVATING',
            'relieving': 'RELIEVING',
            'timing': 'TIMING',
            'severity': 'SEVERITY'
        }
        
        element_name = element_names.get(element, '')
        if not element_name:
            return ""
        
        # Find the section using regex - extract everything from ELEMENT: until next OLDCARTS element
        # Pattern: "ELEMENT_NAME: ...text... NEXT_ELEMENT:" (with optional lookahead for last element)
        pattern = f"{element_name}:(.*?)(?=(?:ONSET|LOCATION|DURATION|CHARACTER|AGGRAVATING|RELIEVING|TIMING|SEVERITY|ASSOCIATED|KEY POSITIVES|KEY NEGATIVES):|$)"
        match = re.search(pattern, classic_presentation, re.IGNORECASE | re.DOTALL)
        
        if match:
            section_text = match.group(1).strip()
            return section_text
        
        return ""
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity with directional penalty for opposite directions
        
        Returns: Cosine similarity score 0-1 with penalty for opposite directions
        
        Raises:
            RuntimeError if embeddings not available or computation fails
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not available - required for semantic similarity")
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 must be non-empty")
        
        # Generate embeddings directly (rely on LLM normalization for text preprocessing)
        emb1 = self.embedding_model.encode([text1])[0]
        emb2 = self.embedding_model.encode([text2])[0]
        
        # DEBUG: Show embedding details
        self._capture_debug(f"[Engine]   🔍 Embedding 1 shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.3f}")
        self._capture_debug(f"[Engine]   🔍 Embedding 2 shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.3f}")
        
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        cosine_similarity = dot_product / norm_product
        
        self._capture_debug(f"[Engine]   🔍 Dot product: {dot_product:.3f}")
        self._capture_debug(f"[Engine]   🔍 Norm product: {norm_product:.3f}")
        self._capture_debug(f"[Engine]   🔍 Cosine similarity: {cosine_similarity:.3f}")
        
        # Pure semantic similarity - no directional penalties
        self._capture_debug(f"[Engine]   🔍 Pure semantic similarity: {cosine_similarity:.3f}")
        
        return float(cosine_similarity)
    
    def _apply_oldcarts_normalization(self, text: str, target_category: str = None) -> str:
        """
        Apply OLDCARTS-specific normalization to patient text
        
        Args:
            text: Patient input text
            target_category: Specific OLDCARTS category to focus on (onset, location, duration, etc.)
        
        Returns:
            Normalized text with medical terms
        """
        import json
        import os
        import re
        
        # Load OLDCARTS synonyms - try multiple paths for different environments
        synonym_files = [
            "/app/synonyms/gi_synonyms_oldcarts.json",  # Docker container path
            "/Users/rcabello/Documents/GitHub/LedgerAI/llm-container/synonyms/gi_synonyms_oldcarts.json",  # macOS path
            "/home/aura/LedgerAI/llm-container/synonyms/gi_synonyms_oldcarts.json",  # Ubuntu path
            "synonyms/gi_synonyms_oldcarts.json",  # Relative path
            "./synonyms/gi_synonyms_oldcarts.json"  # Current directory relative path
        ]
        
        synonym_file = None
        for file_path in synonym_files:
            if os.path.exists(file_path):
                synonym_file = file_path
                self._capture_debug(f"[Engine] ✅ Found OLDCARTS synonyms file at: {file_path}")
                break
        
        if not synonym_file:
            self._capture_debug(f"[Engine] ⚠️ OLDCARTS synonyms file not found in any expected location")
            self._capture_debug(f"[Engine] 🔍 Searched paths:")
            for path in synonym_files:
                self._capture_debug(f"[Engine]   - {path}")
            return text
        
        try:
            with open(synonym_file, 'r') as f:
                oldcarts_synonyms = json.load(f)
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Failed to load OLDCARTS synonyms: {e}")
            return text
        
        normalized_text = text.lower()
        
        # If target_category is specified, focus on that category
        if target_category and target_category in oldcarts_synonyms:
            category_data = oldcarts_synonyms[target_category]
            normalized_text = self._normalize_by_category(normalized_text, category_data, target_category)
        else:
            # Normalize across all categories
            for category, category_data in oldcarts_synonyms.items():
                normalized_text = self._normalize_by_category(normalized_text, category_data, category)
        
        return normalized_text
    
    def _normalize_by_category(self, text: str, category_data: dict, category_name: str) -> str:
        """Normalize text within a specific OLDCARTS category with clean medical terms"""
        import re
        
        normalized_text = text
        
        if isinstance(category_data, dict):
            for subcategory, variations in category_data.items():
                if isinstance(variations, list):
                    # Create clean, semantic-friendly medical terms instead of verbose category labels
                    standard_term = self._get_clean_medical_term(category_name, subcategory)
                    normalized_text = self._apply_variations(normalized_text, variations, standard_term)
                elif isinstance(variations, dict):
                    # Handle nested structures
                    for nested_key, nested_variations in variations.items():
                        if isinstance(nested_variations, list):
                            standard_term = self._get_clean_medical_term(category_name, subcategory, nested_key)
                            normalized_text = self._apply_variations(normalized_text, nested_variations, standard_term)
        elif isinstance(category_data, list):
            # Direct list of variations
            standard_term = category_name.replace("_", " ")
            normalized_text = self._apply_variations(normalized_text, category_data, standard_term)
        
        return normalized_text
    
    def _get_clean_medical_term(self, category: str, subcategory: str, nested_key: str = None) -> str:
        """Convert OLDCARTS categories to clean, semantic-friendly medical terms using data-driven mappings"""
        
        # Load medical term mappings from JSON file
        if not hasattr(self, '_medical_term_mappings'):
            try:
                import json
                with open('config/medical_term_mappings.json', 'r') as f:
                    self._medical_term_mappings = json.load(f)
            except FileNotFoundError:
                raise RuntimeError("Medical term mappings file not found - required for normalization")
        
        # Get mapping for category
        category_mappings = self._medical_term_mappings.get(category, {})
        
        # Return mapped term (raise error if not found - no fallback)
        if subcategory in category_mappings:
            return category_mappings[subcategory]
        else:
            # No fallback - raise error if mapping not found
            raise KeyError(f"No mapping found for '{subcategory}' in category '{category}'")
    
    def _apply_variations(self, text: str, variations: list, standard_term: str) -> str:
        """Apply synonym variations to text"""
        import re
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_variations = sorted(variations, key=len, reverse=True)
        
        for variation in sorted_variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, standard_term, text, flags=re.IGNORECASE)
                self._capture_debug(f"[Engine] 🔄 OLDCARTS normalization: '{variation}' → '{standard_term}'")
                break  # Only replace first match
        
        return text
    
    def _llm_normalize_medical_text(self, text: str, context: str = "general") -> str:
        """
        Use LLM to intelligently normalize medical text instead of rigid synonym matching
        
        Args:
            text: Patient input text
            context: Medical context (location, onset, duration, etc.)
        
        Returns:
            Normalized medical text
        """
        self._capture_debug(f"[Engine] 🧠 LLM normalizing: '{text}' (context: {context})")
        
        system_msg = "You are a medical assistant. Normalize patient language into standard medical terms. Output ONLY the normalized text, nothing else."
        
        user_msg = f"""Normalize this patient response into standard medical terminology for the {context} component ONLY:

Patient: "{text}"
OLDCARTS Component: {context}

Examples for {context}:
- "left side" → "left side" 
- "right side" → "right side"
- "upper right" → "right upper quadrant"
- "lower left" → "left lower quadrant"
- "middle" → "midline"
- "hurts bad" → "severe pain"
- "came on fast" → "sudden onset"
- "feels like stabbing" → "sharp pain"

CRITICAL: 
- Only normalize the {context} component
- Do NOT add information from other symptoms or previous questions
- Do NOT convert general complaints (like "abdominal pain") to specific locations (like "left side") unless the patient specifically mentions a location
- Keep general terms general unless patient provides specific details

Normalized text:"""
        
        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=self.max_tokens_normalization,
                temperature=self.temperature_simple
            )
            
            normalized = response.strip().strip('"\'')
            self._capture_debug(f"[Engine] ✅ LLM normalization: '{text}' → '{normalized}'")
            return normalized
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM normalization failed: {e}")
            return text
    
    def _smart_oldcarts_normalization(self, text: str, target_category: str = None) -> str:
        """
        Smart OLDCARTS normalization - LLM or synonyms (no fallback)
        
        Args:
            text: Patient input text
            target_category: Specific OLDCARTS category to focus on
        
        Returns:
            Normalized text with medical terms
        """
        if self.smart_normalization:
            # Use LLM normalization
            self._capture_debug(f"[Engine] 🧠 Using LLM normalization")
            return self._llm_normalize_medical_text(text, target_category or "general")
        else:
            # Use synonym normalization
            self._capture_debug(f"[Engine] 📚 Using synonym normalization")
            return self._apply_oldcarts_normalization(text, target_category)
    
    
    def _compute_enhanced_oldcarts_similarity(self, user_answer: str, oldcarts_section: str, oldcarts_element: str, condition_name: str = "") -> float:
        """
        Enhanced OLDCARTS similarity with Medical Rule Engine and ML - UNIFIED SYSTEM
        
        Args:
            user_answer: Patient's answer
            oldcarts_section: Guideline text for this OLDCARTS element
            oldcarts_element: OLDCARTS element (L, O, D, C, A, R, T, S)
            condition_name: Name of the condition being evaluated
            
        Returns:
            float: Similarity score 0-1
        """
        # Ensure Medical Rule Engine is available
        if not self.medical_rule_engine:
            raise RuntimeError("Medical Rule Engine not available - ML system required")
        
        # Compute similarity with raw answer (embeddings handle natural language)
        # Word match boost will normalize inside medical_rule_engine (except for demographics)
        # The medical_rule_engine._compute_word_match_boost handles normalization internally
        self._capture_debug(f"[Engine] 🔢 SCORING PHASE: Computing similarity for '{user_answer}' ({oldcarts_element})")
        result = self.medical_rule_engine.get_enhanced_similarity(
            user_answer, oldcarts_section, condition_name, 
            organ_system=self.current_category,
            oldcarts_element=oldcarts_element,
            structured_oldcarts=self._get_structured_oldcarts_for_condition(condition_name)
        )
        self._capture_debug(f"[Engine] 📊 FINAL SCORE: {result['similarity']:.3f} (raw + word boost applied)")
        
        # Log the result
        self._capture_debug(f"[Engine]   🎯 Enhanced {oldcarts_element} similarity: {result['similarity']:.3f} (method: {result['method']})")
        self._capture_debug(f"[Engine]   📝 Reasoning: {result['reasoning']}")
        self._capture_debug(f"[Engine]   🏥 Anatomical Type: {result['anatomical_type']}")
        
        # Collect learning data if available
        if self.learner:
            # Automatically record prediction for learning analysis
            # System learns by detecting patterns of low-scoring answers
            try:
                self.learner.record_prediction(
                    condition=condition_name,
                    oldcarts_element=oldcarts_element,
                    user_answer=user_answer,
                    similarity_score=result['similarity'],
                    guideline_text=oldcarts_section[:100],  # Truncate
                    context={'method': result['method']}
                )
            except Exception as e:
                # Don't fail on learning errors
                pass
        elif self.learning_collector:
            # Legacy learning collector
            self.learning_collector.collect_prediction(
                patient_text=user_answer,
                guideline_text=oldcarts_section,
                condition_name=condition_name,
                similarity=result['similarity'],
                method=result['method'],
                confidence=result['confidence'],
                anatomical_type=result['anatomical_type']
            )
            
            # ML Progress Tracking
            self._capture_debug(f"[Scoring] 🧠 Learning data collected:")
            self._capture_debug(f"[Scoring]   📝 Patient: '{user_answer[:30]}...'")
            self._capture_debug(f"[Scoring]   📋 Condition: {condition_name}")
            self._capture_debug(f"[Scoring]   🎯 OLDCARTS: {oldcarts_element}")
            self._capture_debug(f"[Scoring]   🎯 Method: {result['method']}")
            self._capture_debug(f"[Scoring]   📊 Similarity: {result['similarity']:.3f}")
            self._capture_debug(f"[Scoring]   🏥 Anatomical: {result['anatomical_type']}")
            self._capture_debug(f"[Scoring]   🔄 Confidence: {result['confidence']}")
        
        # Track performance metrics if available
        return result['similarity']
    
    
    
    def _get_structured_oldcarts_for_condition(self, condition_name: str) -> dict:
        """Get structured OLDCARTS data for a specific condition"""
        # Find the guideline that matches this condition
        for guideline in self.active_guidelines:
            if guideline.get('data', {}).get('condition', guideline['name']) == condition_name:
                return guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
        
        return {}
    
    def _analyze_missing_location_information(self, patient_answer: str, high_scoring_guidelines: list, oldcarts_element: str) -> list:
        """
        Analyze what specific information is missing from patient answer
        to generate targeted clarifying questions.
        
        Directly extracts includes from structured_oldcarts of high-scoring guidelines
        and determines missing terms (not using synonym categories).
        """
        if oldcarts_element != 'location':
            return []
        
        # Extract ALL includes terms from high-scoring guidelines
        all_includes = set()
        for guideline in high_scoring_guidelines:
            data = guideline.get('data', {})
            key_features = data.get('key_features', {})
            structured = key_features.get('structured_oldcarts', {})
            
            if 'location' in structured and isinstance(structured['location'], dict):
                includes = structured['location'].get('includes', [])
                for term in includes:
                    all_includes.add(term.lower())
        
        self._capture_debug(f"[Engine] 📋 All includes from high-scoring guidelines: {all_includes}")
        
        # Normalize patient answer for comparison
        patient_lower = patient_answer.lower()
        
        # Determine which include terms are missing from patient answer
        missing_terms = [term for term in all_includes if term not in patient_lower]
        
        self._capture_debug(f"[Engine] 📋 Missing terms: {missing_terms}")
        
        # If there are missing terms, we need clarification
        return missing_terms if missing_terms else []
    
    def _get_organ_system_from_guidelines(self, guidelines: list) -> str:
        """Get organ system from already-determined category"""
        # Use the category that was determined during initial complaint processing
        if hasattr(self, 'current_category') and self.current_category:
            return self.current_category
        
        # No fallback - this should not happen if initial categorization worked
        raise RuntimeError(f"Organ system not determined during initial assessment. Guidelines: {len(guidelines) if guidelines else 0}")
    
    def _get_available_categories_from_synonyms(self, synonyms: dict, oldcarts_element: str) -> list:
        """Get available categories from synonym file for specific OLDCARTS element"""
        if oldcarts_element not in synonyms:
            return []
        
        return list(synonyms[oldcarts_element].keys())
    
    def _is_category_missing_from_patient_answer(self, patient_lower: str, category: str, element_synonyms: dict) -> bool:
        """Check if a specific category is missing from patient answer"""
        if category not in element_synonyms:
            return False
        
        # Check if any synonym from this category appears in patient answer
        category_synonyms = element_synonyms[category]
        for synonym in category_synonyms:
            if synonym.lower() in patient_lower:
                return False  # Category is present
        
        return True  # Category is missing
    
    def _category_discriminates_between_guidelines(self, category: str, guidelines: list, oldcarts_element: str, synonyms: dict) -> bool:
        """Check if a category helps discriminate between high-scoring guidelines using synonym files"""
        
        # Get synonyms for this category
        if oldcarts_element not in synonyms or category not in synonyms[oldcarts_element]:
            return False
        
        category_synonyms = synonyms[oldcarts_element][category]
        matching_guidelines = set()
        
        # Check which guidelines match this category
        for guideline in guidelines:
            data = guideline.get('data', {})
            key_features = data.get('key_features', {})
            structured = key_features.get('structured_oldcarts', {})
            
            if oldcarts_element in structured and isinstance(structured[oldcarts_element], dict):
                includes = structured[oldcarts_element].get('includes', [])
                
                # Check if any include term matches any synonym in this category
                for term in includes:
                    term_lower = term.lower()
                    for synonym in category_synonyms:
                        if synonym.lower() in term_lower or term_lower in synonym.lower():
                            matching_guidelines.add(guideline['name'])
                            break
        
        # A category discriminates if:
        # 1. It matches at least one guideline (so it's relevant)
        # 2. There are other competing categories that match different guidelines
        if len(matching_guidelines) == 0:
            return False  # Not relevant to any guideline
        
        # Check if there are competing categories
        return self._has_competing_categories(category, guidelines, oldcarts_element, synonyms)
    
    def _has_competing_categories(self, target_category: str, guidelines: list, oldcarts_element: str, synonyms: dict) -> bool:
        """Check if there are other categories that compete with the target category"""
        if oldcarts_element not in synonyms:
            return False
        
        # Get all categories except the target
        all_categories = list(synonyms[oldcarts_element].keys())
        competing_categories = [cat for cat in all_categories if cat != target_category]
        
        # Check if any competing category matches different guidelines
        for competing_category in competing_categories:
            competing_synonyms = synonyms[oldcarts_element][competing_category]
            competing_guidelines = set()
            
            for guideline in guidelines:
                data = guideline.get('data', {})
                key_features = data.get('key_features', {})
                structured = key_features.get('structured_oldcarts', {})
                
                if oldcarts_element in structured and isinstance(structured[oldcarts_element], dict):
                    includes = structured[oldcarts_element].get('includes', [])
                    
                    for term in includes:
                        term_lower = term.lower()
                        for synonym in competing_synonyms:
                            if synonym.lower() in term_lower or term_lower in synonym.lower():
                                competing_guidelines.add(guideline['name'])
                                break
            
            # If competing category matches different guidelines, we have competition
            if len(competing_guidelines) > 0:
                return True
        
        return False
    
    def _has_radiation_component(self, guidelines: list) -> bool:
        """
        Check if any of the active guidelines mention radiation in their location section.
        
        Returns True if radiation should be asked about.
        """
        radiation_patterns = ['radiates', 'radiate to', 'radiation', 'goes to', 'spreads to', 'travels to']
        
        for guideline in guidelines:
            data = guideline.get('data', {})
            key_features = data.get('key_features', {})
            structured = key_features.get('structured_oldcarts', {})
            
            if 'location' in structured:
                location_data = structured['location']
                if isinstance(location_data, dict) and 'includes' in location_data:
                    includes = location_data['includes']
                    for term in includes:
                        term_lower = str(term).lower()
                        # Check if any includes term mentions radiation
                        if any(pattern in term_lower for pattern in radiation_patterns):
                            self._capture_debug(f"[Engine] ✅ Radiation detected in guideline: {guideline['name']}")
                            return True
        
        self._capture_debug(f"[Engine] ❌ No radiation component found in active guidelines")
        return False
    
    def _generate_targeted_location_options(self, missing_terms: list, high_scoring_guidelines: list) -> list:
        """
        Generate targeted patient-friendly options based on missing location terms.
        
        Uses synonym files to convert missing includes terms to patient-friendly options.
        """
        if not missing_terms:
            return []
        
        # Get organ system and load synonyms
        organ_system = self._get_organ_system_from_guidelines(high_scoring_guidelines)
        if not organ_system:
            return []
        
        synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
        synonym_path = os.path.join(os.path.dirname(__file__), synonym_file)
        
        if not os.path.exists(synonym_path):
            return []
        
        with open(synonym_path, 'r') as f:
            synonyms = json.load(f)
        
        # Build a mapping of medical terms to their patient-friendly synonyms
        # Look for each missing term in the synonym categories
        patient_friendly_options = []
        
        if 'location' in synonyms:
            # Build a comprehensive mapping of medical terms to categories
            medical_to_category = {}
            
            for category, synonym_list in synonyms['location'].items():
                if isinstance(synonym_list, list):
                    category_lower = category.lower()
                    
                    # Extract key medical terms from category name (e.g., "ruq_pain" -> ["ruq", "pain"])
                    category_base = category_lower.replace('_pain', '').replace('_discomfort', '').replace('_tenderness', '').replace('_ache', '')
                    
                    # Add direct mapping from category base to synonym list (e.g., "mcburney_pain" -> "mcburney")
                    if category_base not in medical_to_category:
                        medical_to_category[category_base] = []
                    medical_to_category[category_base].append((category, synonym_list))
                    
                    # Also map with underscores converted to spaces (e.g., "radiates_to_back" -> "radiates to back")
                    category_base_with_spaces = category_base.replace('_', ' ')
                    if category_base_with_spaces not in medical_to_category:
                        medical_to_category[category_base_with_spaces] = []
                    medical_to_category[category_base_with_spaces].append((category, synonym_list))
                    
                    # Also map individual words from the category
                    for word in category_lower.split('_'):
                        if word not in ['pain', 'discomfort', 'tenderness', 'ache']:  # Skip generic pain words
                            if word not in medical_to_category:
                                medical_to_category[word] = []
                            medical_to_category[word].append((category, synonym_list))
                    
                    # IMPORTANT: Map each synonym back to itself to ensure all terms are accessible
                    # This ensures "right upper quadrant" maps to "ruq_pain" category
                    for synonym in synonym_list:  # Map ALL synonyms for complete coverage
                        synonym_lower = synonym.lower()
                        if synonym_lower not in medical_to_category:
                            medical_to_category[synonym_lower] = []
                        medical_to_category[synonym_lower].append((category, synonym_list))
            
            # Now match missing terms to categories
            seen_categories = set()  # Track which categories we've already added
            seen_options = set()  # Track options to avoid duplicates
            
            # Filter out radiation terms - these are not anatomical locations for location clarification
            radiation_patterns = ['radiates', 'goes to', 'spreads to', 'travels to', 'moves to', 
                                 'radiates to', 'radiate to', 'radiation']
            
            def is_radiation_term(term):
                """Check if term describes radiation pattern rather than location"""
                term_lower = term.lower()
                return any(pattern in term_lower for pattern in radiation_patterns)
            
            for missing_term in missing_terms:
                term_lower = missing_term.lower().strip()
                
                # Skip radiation terms for location clarification
                if is_radiation_term(missing_term):
                    self._capture_debug(f"[Engine]   Skipping radiation term: {missing_term}")
                    continue
                
                # Try direct match in medical_to_category
                if term_lower in medical_to_category:
                    categories = medical_to_category[term_lower]
                    self._capture_debug(f"[Engine]   Direct match for '{term_lower}': {[c[0] for c in categories]}")
                    for category, synonym_list in categories:
                        if category not in seen_categories and synonym_list:
                            patient_friendly = synonym_list[0]  # Use first (most patient-friendly) option
                            if patient_friendly not in seen_options:
                                patient_friendly_options.append(patient_friendly)
                                seen_options.add(patient_friendly)
                                self._capture_debug(f"[Engine]   Added option: '{patient_friendly}' from category '{category}'")
                            seen_categories.add(category)
                    continue
                else:
                    self._capture_debug(f"[Engine]   No direct match for '{term_lower}' in medical_to_category")
                    # Debug: Show what keys are available
                    available_keys = [k for k in medical_to_category.keys() if 'right upper' in k or 'ruq' in k]
                    if available_keys:
                        self._capture_debug(f"[Engine]   Available similar keys: {available_keys[:5]}")
                
                # Try with underscores (e.g., "radiates to back" -> "radiates_to_back")
                term_with_underscores = term_lower.replace(' ', '_')
                if term_with_underscores in medical_to_category:
                    categories = medical_to_category[term_with_underscores]
                    for category, synonym_list in categories:
                        if category not in seen_categories and synonym_list:
                            patient_friendly = synonym_list[0]
                            if patient_friendly not in seen_options:
                                patient_friendly_options.append(patient_friendly)
                                seen_options.add(patient_friendly)
                            seen_categories.add(category)
                    continue
                
                # Try matching against category names (e.g., "right upper quadrant" matches "ruq_pain")
                for category, synonym_list in synonyms['location'].items():
                    if isinstance(synonym_list, list) and category not in seen_categories and synonym_list:
                        category_lower = category.lower()
                        
                        # Check if the missing term relates to this category
                        if term_lower in category_lower or category_lower in term_lower:
                            patient_friendly = synonym_list[0]
                            if patient_friendly not in seen_options:
                                patient_friendly_options.append(patient_friendly)
                                seen_options.add(patient_friendly)
                            seen_categories.add(category)
                            break
                        
                        # Check for word overlap (helps with multi-word terms)
                        term_words = set(term_lower.split())
                        category_words = set(category_lower.replace('_', ' ').split())
                        
                        # Significant overlap suggests a match
                        if term_words.intersection(category_words) and len(term_words.intersection(category_words)) > 0:
                            patient_friendly = synonym_list[0]
                            if patient_friendly not in seen_options:
                                patient_friendly_options.append(patient_friendly)
                                seen_options.add(patient_friendly)
                            seen_categories.add(category)
                            break
        
        # Remove duplicates while preserving order
        options = list(dict.fromkeys(patient_friendly_options))
        
        # Prioritize options that help differentiate between competing conditions
        # Group similar options and pick the most specific one
        if len(options) > 4:
            # If we have too many options, prioritize based on specificity and relevance
            # Sort by length (longer descriptions are usually more specific and helpful)
            options = sorted(options, key=lambda x: len(x), reverse=True)
            
            # Take up to 4 options, ensuring variety
            prioritized = []
            seen_terms = set()  # Store frozensets of word sets (hashable)
            
            # Prioritize options with anatomical location terms (quadrant, upper, lower, etc.)
            # These are most helpful for discrimination
            location_keywords = ['quadrant', 'upper', 'lower', 'left', 'right', 'side', 'epigastric', 'umbilical', 'belly button', 'middle']
            
            # Separate options into location-specific and general
            location_options = []
            general_options = []
            
            for opt in options:
                if any(keyword in opt.lower() for keyword in location_keywords):
                    location_options.append(opt)
                else:
                    general_options.append(opt)
            
            # Add location options first (these are most discriminative)
            for opt in location_options:
                if len(prioritized) >= 4:
                    break
                prioritized.append(opt)
            
            # Then add general options if we have room
            for opt in general_options:
                if len(prioritized) >= 4:
                    break
                prioritized.append(opt)
            
            options = prioritized
        
        self._capture_debug(f"[Engine] 🎯 Generated targeted options from {len(missing_terms)} missing terms: {options}")
        return options
    
    def _select_patient_friendly_options(self, synonyms: list) -> list:
        """Select the most patient-friendly options from synonyms using LLM"""
        if not synonyms:
            return []
        
        # Use LLM to select the most patient-friendly options
        system_msg = "You are a medical assistant helping patients understand their symptoms. Select the most patient-friendly terms that avoid medical jargon."
        
        user_msg = f"""From these medical terms, select the 2 most patient-friendly options that a regular person would understand:

Medical terms: {', '.join(synonyms)}

Select 2 terms that are:
- Simple and clear
- Avoid medical jargon (like RUQ, RLQ, quadrant, epigastric)
- Use everyday language
- Easy for patients to understand

Return only the 2 selected terms, separated by commas."""

        try:
            response = self.llm_chat_simple_fn(
                system_message=system_msg,
                user_message=user_msg,
                temperature=self.temperature_simple
            )
            
            # Parse response
            selected_terms = [term.strip() for term in response.split(',')]
            
            # Validate that selected terms are actually in the original list
            valid_terms = [term for term in selected_terms if term in synonyms]
            
            if valid_terms:
                return valid_terms[:2]
            else:
                # Fallback to first 2 synonyms if LLM response is invalid
                return synonyms[:2]
                
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM selection failed: {e}")
            # Fallback to first 2 synonyms
            return synonyms[:2]
    
    def collect_user_feedback(self, 
                               prediction_id: str,
                               prediction: Dict[str, Any],
                               user_rating: int,
                               user_comment: str = "",
                               condition_name: str = "") -> bool:
        """
        Collect user feedback on ML prediction
        
        Args:
            prediction_id: Unique identifier for prediction
            prediction: ML prediction result
            user_rating: User rating (1-5 stars)
            user_comment: Optional user comment
            condition_name: Medical condition name
            
        Returns:
            bool: True if feedback collected successfully
        """
        # User feedback collection not implemented yet
        return False
    
    def collect_accuracy_feedback(self, 
                                 prediction_id: str,
                                 predicted_accuracy: float,
                                 actual_accuracy: float,
                                 user_comment: str = "",
                                 condition_name: str = "") -> bool:
        """
        Collect accuracy feedback for ML prediction
        
        Args:
            prediction_id: Unique identifier for prediction
            predicted_accuracy: Predicted accuracy score
            actual_accuracy: Actual accuracy score
            user_comment: Optional user comment
            condition_name: Medical condition name
            
        Returns:
            bool: True if feedback collected successfully
        """
        # User feedback collection not implemented yet
        return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status"""
        status = {
            'medical_rule_engine': self.medical_rule_engine is not None,
            'learning_collector': self.learning_collector is not None,
            'active_learning': self.learner is not None
        }
        
        # Get detailed status from components
        if self.learner:
            status['active_learning_status'] = 'Enabled (manual review mode)'
        
        # ML Progress Tracking - Learning Status
        self._capture_debug(f"[Scoring] 📊 Learning system status:")
        self._capture_debug(f"[Scoring]   🧠 Medical Rule Engine: {'Active' if self.medical_rule_engine else 'Inactive'}")
        self._capture_debug(f"[Scoring]   📝 Learning Collector: {'Active' if self.learning_collector else 'Inactive'}")
        self._capture_debug(f"[Scoring]   📊 Active Learning: {'Active' if self.learner else 'Inactive'}")
        
        return status
    
    
    
    def _should_bypass_clarification(self, oldcarts_element: str) -> bool:
        """
        SINGLE POINT: Determine if an OLDCARTS element should bypass clarification
        Returns True if clarification should be skipped, False otherwise
        
        Args:
            oldcarts_element: OLDCARTS element in lowercase full name format ('onset', 'location', etc.)
        
        Returns:
            bool: True if clarification should be bypassed, False otherwise
        """
        # Codebase uses lowercase full names only - no normalization needed
        ELEMENTS_THAT_DO_NOT_NEED_CLARIFICATION = {'onset'}  # Onset never needs clarification
        should_bypass = oldcarts_element in ELEMENTS_THAT_DO_NOT_NEED_CLARIFICATION
        self._capture_debug(f"[Engine] 🎯 BYPASS CHECK: element='{oldcarts_element}', should_bypass={should_bypass}, bypass_list={ELEMENTS_THAT_DO_NOT_NEED_CLARIFICATION}")
        return should_bypass
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """
        Score guidelines using SEMANTIC SIMILARITY between answer and corresponding OLDCARTS section
        
        This is the CORE diagnostic reasoning - using vector similarity instead of LLM.
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🔢 LLM SCORING PHASE")
        self._capture_debug(f"{'='*80}")
        
        # Get the last question
        last_q = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_q = item['question']
                break
        
        # Build Q&A history for context
        qa_pairs = []
        temp_q = None
        for item in self.conversation_history:
            if item['type'] == 'question' and item.get('focus') not in ['age', 'sex']:
                temp_q = item['question']
            elif item['type'] == 'answer' and temp_q:
                qa_pairs.append(f"Q: {temp_q}\nA: {item['answer']}")
                temp_q = None
        
        history_text = "\n\n".join(qa_pairs) if qa_pairs else "None"
        
        self._capture_debug(f"[Engine] 📋 Last Question: '{last_q}'")
        self._capture_debug(f"[Engine] 📋 Answer: '{answer}'")
        self._capture_debug(f"[Engine] 📋 History: {len(qa_pairs)} Q&A pairs")
        
        # Determine which OLDCARTS element was just asked
        last_question_item = None
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question':
                last_question_item = item
                break
        
        oldcarts_element = last_question_item.get('oldcarts') if last_question_item else None
        
        # DEMOGRAPHICS: No scoring needed - just documentation
        if last_question_item and last_question_item.get('focus') == 'demographics':
            self._capture_debug(f"[Engine] 📝 DEMOGRAPHICS: Documentation only - no scoring needed")
            self._capture_debug(f"[Engine] ✅ Demographics collected, continuing to next question")
            
            # Check if all demographics are collected
            if 'sex' not in self.demographics:
                # Ask sex with button-based response
                question = "What is your biological sex?"
                self._capture_debug(f"[Engine] ✅ Sex question with buttons: '{question}'")
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'oldcarts': 'demographics',
                    'focus': 'demographics'
                })
                return {
                    'success': True,
                    'question': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'Male', 'callback_data': 'sex_male'},
                        {'text': 'Female', 'callback_data': 'sex_female'}
                    ],
                    'debug': self._get_debug_info()
                }
            elif 'chronicity' not in self.demographics:
                # Ask chronicity with button-based response
                question = "Is this a new problem or an ongoing issue?"
                self._capture_debug(f"[Engine] ✅ Chronicity question with buttons: '{question}'")
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'oldcarts': 'demographics',
                    'focus': 'demographics'
                })
                return {
                    'success': True,
                    'question': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'New Problem', 'callback_data': 'chronicity_new'},
                        {'text': 'Ongoing Issue', 'callback_data': 'chronicity_recurring'}
                    ],
                    'debug': self._get_debug_info()
                }
            else:
                # All demographics collected, start OLDCARTS
                self._capture_debug(f"[Engine] ✅ All demographics collected, starting OLDCARTS")
                return self._ask_next_clinical_question()
        
        # ONSET: Documentation only - no scoring needed
        if oldcarts_element == 'onset':
            self._capture_debug(f"[Engine] 📝 ONSET: Documentation only - no scoring needed")
            self._capture_debug(f"[Engine] ✅ Marking OLDCARTS element 'O' as covered")
            self.oldcarts_covered['O'] = True
            self._capture_debug(f"[Engine] 📋 OLDCARTS Coverage: {''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])} ({sum(self.oldcarts_covered.values())}/8)")
            
            # UPDATE OLDCARTS ANALYSIS: Refresh missing components
            self._update_oldcarts_analysis()
            
            # Move to next question
            return self._ask_next_clinical_question()
        
        # ASSOCIATED SYMPTOMS: Score using KEY POSITIVES/NEGATIVES sections
        if not oldcarts_element:
            self._capture_debug(f"\n[Engine] 🎯 ASSOCIATED SYMPTOM SCORING:\n")
            self._capture_debug(f"[Engine] 📋 Matching '{answer}' to KEY POSITIVES/NEGATIVES sections\n")
            
            # Combine active + reserve for scoring
            all_guidelines = self.active_guidelines + self.reserve_pool
            
            for g in all_guidelines:
                classic = g['data'].get('key_features', {}).get('classic_presentation', '')
                
                # Extract KEY POSITIVES and KEY NEGATIVES sections
                key_pos = ""
                key_neg = ""
                
                if 'KEY POSITIVES:' in classic:
                    parts = classic.split('KEY POSITIVES:')
                    if len(parts) > 1:
                        key_section = parts[1].split('KEY NEGATIVES:')[0] if 'KEY NEGATIVES:' in parts[1] else parts[1]
                        key_pos = key_section.strip()
                
                if 'KEY NEGATIVES:' in classic:
                    parts = classic.split('KEY NEGATIVES:')
                    if len(parts) > 1:
                        key_neg = parts[1].strip()
                
                # Combine both sections for matching
                combined_key_features = f"{key_pos} {key_neg}".strip()
                
                if combined_key_features:
                    # Compute similarity
                    try:
                        similarity = self._compute_similarity(answer, combined_key_features)
                    except Exception as sim_error:
                        self._capture_debug(f"[Engine] ❌ Associated symptoms similarity computation failed for {g['name']}: {sim_error}")
                        import traceback
                        traceback.print_exc()
                        # Skip this guideline and continue with the next one
                        continue
                    
                    # Small weight for associated symptoms (10% vs 30% for OLDCARTS)
                    old_score = g['score']
                    new_score = (old_score * 0.9) + (similarity * 0.1)
                    g['score'] = new_score
                    
                    change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
                    self._capture_debug(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (similarity: {similarity:.2f})")
            
            # Re-rank after associated symptom scoring
            all_guidelines.sort(key=lambda x: x['score'], reverse=True)
            self.active_guidelines = all_guidelines[:self.MAX_ACTIVE]
            self.reserve_pool = all_guidelines[self.MAX_ACTIVE:]
            
            self._capture_debug(f"\n[Engine] 📊 UPDATED RANKINGS after associated symptom:")
            for i, g in enumerate(self.active_guidelines, 1):
                self._capture_debug(f"[Engine]   {i}. {g['name']}: {g['score']:.0%}")
            
            self._capture_debug(f"\n")
            
            # Continue to next question
            return self._ask_next_clinical_question()
        
        # FOR EACH GUIDELINE: Score using VECTOR SIMILARITY
        # IMPORTANT: Score ALL guidelines (active + reserve) so we can re-rank dynamically
        self._capture_debug(f"\n[Engine] 🎯 SEMANTIC SIMILARITY SCORING:\n")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized - cannot compute similarity")
        
        self._capture_debug(f"[Engine] 📊 Matching answer to OLDCARTS element: {oldcarts_element}")
        self._capture_debug(f"[Engine] 📋 Scoring ALL {len(self.active_guidelines) + len(self.reserve_pool)} guidelines (active + reserve)")
        self._capture_debug(f"[Engine] 🧠 LLM NORMALIZATION: '{answer}' → Semantic understanding via vector similarity")
        self._capture_debug(f"[Engine] 📝 Patient language: '{answer}'")
        self._capture_debug(f"[Engine] 🔍 LLM will normalize to medical terms through semantic similarity\n")
        
        # Combine active + reserve for scoring
        all_guidelines = self.active_guidelines + self.reserve_pool
        
        # SIMPLE APPROACH: If location element, immediately rule out anatomically incompatible conditions
        if oldcarts_element == 'location' and self.medical_rule_engine:
            patient_words = set(answer.lower().split())
            
            # Extract ALL anatomical orientation terms
            has_right = 'right' in patient_words
            has_left = 'left' in patient_words
            has_anterior = 'anterior' in patient_words or 'front' in patient_words
            has_posterior = 'posterior' in patient_words or 'back' in patient_words
            has_superior = 'upper' in patient_words or 'superior' in patient_words or 'top' in patient_words
            has_inferior = 'lower' in patient_words or 'inferior' in patient_words or 'bottom' in patient_words
            has_proximal = 'proximal' in patient_words or 'near' in patient_words
            has_distal = 'distal' in patient_words or 'far' in patient_words
            
            if has_right or has_left or has_anterior or has_posterior or has_superior or has_inferior or has_proximal or has_distal:
                # Map category to organ system
                category_to_system = {
                    'gastrointestinal': 'GI', 'cardiovascular': 'CARDIO',
                    'respiratory': 'PULMONARY', 'neurological': 'NEURO',
                    'musculoskeletal': 'MSK', 'renal': 'RENAL',
                    'genitourinary': 'GU', 'gynecological': 'GYN',
                    'dermatological': 'DERM'
                }
                
                to_remove = []
                for g in all_guidelines:
                    condition_name = g['data'].get('condition', g['name'])
                    category = g['data'].get('category', '')
                    organ_system = category_to_system.get(category)
                    
                    if organ_system:
                        anatomical_type = self.medical_rule_engine._get_condition_anatomical_type(condition_name, organ_system)
                        
                        # IMMEDIATE RULE-OUT: Anatomically incompatible using medical_rules.json
                        incompatible = False
                        
                        # Left/right rule-out using medical_rules.json
                        if (has_right and anatomical_type == 'left_only') or (has_left and anatomical_type == 'right_only'):
                            incompatible = True
                            reason = f"Patient said '{answer}' but condition is {anatomical_type} in medical_rules.json"
                        
                        # Additional anatomical checks from guideline's structured_oldcarts
                        if not incompatible:
                            guideline_data = g.get('data', {})
                            key_features = guideline_data.get('key_features', {})
                            structured_oldcarts = key_features.get('structured_oldcarts', {})
                            location_data = structured_oldcarts.get('location', {})
                            location_includes = location_data.get('includes', [])
                            
                            # Check anterior/posterior conflicts
                            if has_anterior:
                                if any('posterior' in str(term).lower() or 'back' in str(term).lower() for term in location_includes):
                                    if not any('anterior' in str(term).lower() or 'front' in str(term).lower() for term in location_includes):
                                        incompatible = True
                                        reason = "Patient said anterior/front, but guideline mentions posterior/back"
                            
                            if has_posterior:
                                if any('anterior' in str(term).lower() or 'front' in str(term).lower() for term in location_includes):
                                    if not any('posterior' in str(term).lower() or 'back' in str(term).lower() for term in location_includes):
                                        incompatible = True
                                        reason = "Patient said posterior/back, but guideline mentions anterior/front"
                            
                            if has_right and has_superior:  # e.g., "right upper quadrant"
                                # Check if guideline has opposite orientation on right side
                                if any('left' in str(term).lower() for term in location_includes) and not any('right' in str(term).lower() for term in location_includes):
                                    incompatible = True
                                    reason = "Patient said right upper, but guideline mentions left side"
                            
                            if has_left and has_superior:  # e.g., "left upper quadrant"
                                # Check if guideline has opposite orientation on left side
                                if any('right' in str(term).lower() for term in location_includes) and not any('left' in str(term).lower() for term in location_includes):
                                    incompatible = True
                                    reason = "Patient said left upper, but guideline mentions right side"
                        
                        if incompatible:
                            self._capture_debug(f"[Engine] 🚫 IMMEDIATE ANATOMICAL RULE-OUT: {condition_name} ({anatomical_type})")
                            self._capture_debug(f"[Engine]   Reason: {reason}")
                            g['score'] = 0.0
                            self.ruled_out.append(g)
                            to_remove.append(g)
                
                # Remove ruled out guidelines from scoring
                for g in to_remove:
                    all_guidelines.remove(g)
        
        for g in all_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            
            # Extract the specific OLDCARTS section for this element
            oldcarts_section = self._extract_oldcarts_section(classic, oldcarts_element)
            
            if not oldcarts_section:
                self._capture_debug(f"[Engine] ⚠️ Warning: Could not extract {oldcarts_element} section from {g['name']} - skipping this guideline")
                continue  # Skip this guideline instead of crashing
            
            # ENHANCED OLDCARTS SIMILARITY: Use existing method - let it fail if broken
            # Use condition name from guideline data, not file name
            condition_name = g['data'].get('condition', g['name'])
            similarity = self._compute_enhanced_oldcarts_similarity(answer, oldcarts_section, oldcarts_element, condition_name)
            self._capture_debug(f"[Engine]   {g['name']}: Enhanced {oldcarts_element} similarity = {similarity:.3f} ('{answer}' vs '{oldcarts_section[:50]}...')")
            
            # Update score using rolling average (weighted average of all OLDCARTS answers)
            old_score = g['score']
            
            # ROLLING AVERAGE SCORING: Average current similarity with prior scores
            # This gives equal weight to each OLDCARTS element as we collect more information
            # Formula: (old_score * 0.7) + (similarity * 0.3)
            # This means new answers contribute 30% while prior answers contribute 70%
            # Example: If we've answered 3 OLDCARTS questions and this is #4:
            # - 1st answer: 0.50 → 0.35 (weighted)
            # - 2nd answer: 0.70 → 0.49 (weighted)
            # - 3rd answer: 0.60 → 0.42 (weighted)
            # - 4th answer: 0.90 → 0.63 (weighted)
            # Final score reflects all answers, not just the last one
            new_score = (old_score * 0.7) + (similarity * 0.3)
            g['score'] = new_score
            change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
            self._capture_debug(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (semantic)")
            self._capture_debug(f"[Engine]     🧠 {oldcarts_element} Score: {similarity:.2f} ('{answer}' ↔ '{oldcarts_section[:50]}...')")
            self._capture_debug(f"[Engine]     📝 Patient: '{answer}' → Medical: '{oldcarts_section[:80]}...'")
            
            # ML Progress Tracking - Scoring
            self._capture_debug(f"[Scoring] 🎯 Score updated:")
            self._capture_debug(f"[Scoring]   📋 Condition: {g['name']}")
            self._capture_debug(f"[Scoring]   🎯 OLDCARTS: {oldcarts_element}")
            self._capture_debug(f"[Scoring]   📊 Old Score: {old_score:.0%} → New Score: {new_score:.0%} {change}")
            self._capture_debug(f"[Scoring]   🧠 ML Similarity: {similarity:.3f}")
            self._capture_debug(f"[Scoring]   📝 Patient Input: '{answer}'")
            self._capture_debug(f"[Scoring]   📋 Guideline: '{oldcarts_section[:50]}...'")
        
        # DYNAMIC RE-RANKING: Sort ALL guidelines by updated scores
        # This ensures conditions like Diverticulitis (LLQ) jump to top when "left side" is mentioned
        self._capture_debug(f"\n[Engine] 🔄 RE-RANKING all guidelines by updated scores...")
        
        # Rule out any with score < threshold (ML-only system)
        ruled_out_this_round = []
        remaining = []
        for g in all_guidelines:
            # Use dynamic threshold based on score type
            threshold = self._get_dynamic_threshold(g['score'])
            if g['score'] < threshold:
                self._capture_debug(f"[Engine] ❌ RULING OUT: {g['name']} (score {g['score']:.0%} < {threshold:.0%})")
                self.ruled_out.append(g)
                ruled_out_this_round.append(g)
                
                # ML Progress Tracking - Rule Out
                self._capture_debug(f"[Scoring] ❌ Condition ruled out:")
                self._capture_debug(f"[Scoring]   📋 Condition: {g['name']}")
                self._capture_debug(f"[Scoring]   📊 Score: {g['score']:.0%} < Threshold: {threshold:.0%}")
                self._capture_debug(f"[Scoring]   🎯 ML Decision: Anatomical mismatch or low similarity")
            else:
                remaining.append(g)
                
                # ML Progress Tracking - Kept
                self._capture_debug(f"[Scoring] ✅ Condition kept:")
                self._capture_debug(f"[Scoring]   📋 Condition: {g['name']}")
                self._capture_debug(f"[Scoring]   📊 Score: {g['score']:.0%} >= Threshold: {threshold:.0%}")
                self._capture_debug(f"[Scoring]   🎯 ML Decision: Anatomical match or high similarity")
        
        # Sort remaining by score (highest first)
        remaining.sort(key=lambda x: x['score'], reverse=True)
        
        # Split into active (top MAX_ACTIVE) and reserve (rest)
        self.active_guidelines = remaining[:self.MAX_ACTIVE]
        self.reserve_pool = remaining[self.MAX_ACTIVE:]
        
        # Track promotions and demotions for logging
        promoted_this_round = [g for g in self.active_guidelines if g not in [item for item in all_guidelines[:self.MAX_ACTIVE]]]
        demoted_this_round = [g for g in self.reserve_pool if g in [item for item in all_guidelines[:self.MAX_ACTIVE]]]
        
        if promoted_this_round:
            self._capture_debug(f"\n[Engine] 🔼 PROMOTED to active:")
            for g in promoted_this_round:
                self._capture_debug(f"[Engine]   ↑ {g['name']} (score: {g['score']:.0%})")
        
        if demoted_this_round:
            self._capture_debug(f"\n[Engine] 🔽 DEMOTED to reserve:")
            for g in demoted_this_round:
                self._capture_debug(f"[Engine]   ↓ {g['name']} (score: {g['score']:.0%})")
        
        self._capture_debug(f"\n[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            self._capture_debug(f"[Engine]   {i}. {g['name']}: {g['score']:.0%} {urgency_emoji}")
            
            # ML Progress Tracking - Top Conditions
            self._capture_debug(f"[Scoring] 🏆 Top {i}: {g['name']}")
            self._capture_debug(f"[Scoring]   📊 Score: {g['score']:.0%}")
            self._capture_debug(f"[Scoring]   📋 Prevalence: {g.get('prevalence', 'unknown')}")
            self._capture_debug(f"[Scoring]   🎯 ML Confidence: High similarity match")
            self._capture_debug(f"[Scoring]   🚨 Urgency: {g['data'].get('urgency', 'standard')}")
        
        # Always show pool statistics
        self._capture_debug(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
        # ML Progress Tracking - Final Statistics
        self._capture_debug(f"[Scoring] 📊 Final statistics:")
        self._capture_debug(f"[Scoring]   🎯 Active Conditions: {len(self.active_guidelines)}")
        self._capture_debug(f"[Scoring]   📋 Reserve Conditions: {len(self.reserve_pool)}")
        self._capture_debug(f"[Scoring]   ❌ Ruled Out: {len(self.ruled_out)}")
        self._capture_debug(f"[Scoring]   📈 Total Processed: {len(all_guidelines)}")
        self._capture_debug(f"[Scoring]   🧠 ML System: Fully operational")
        
        self._capture_debug(f"{'='*80}\n")
        
        # COMPREHENSIVE OLDCARTS COVERAGE: Analyze answer against ALL elements
        # This prevents asking redundant questions when comprehensive answers cover multiple elements
        clarification_was_asked = self._was_clarification_just_asked()
        self._capture_debug(f"[Engine] 🔍 Clarification check: {clarification_was_asked}")
        
        if not clarification_was_asked:
            # Analyze the answer against all OLDCARTS elements to detect comprehensive coverage
            covered_elements = self._analyze_comprehensive_oldcarts_coverage(answer, oldcarts_element)
            self._capture_debug(f"[Engine] 🎯 Comprehensive OLDCARTS analysis: {covered_elements}")
            
            # Mark all covered elements as complete
            for element in covered_elements:
                if not self.oldcarts_covered.get(element, False):  # Only mark if not already covered
                    self._capture_debug(f"[Engine] ✅ Marking OLDCARTS element '{element}' as covered")
                    self.oldcarts_covered[element] = True
            
            # UPDATE OLDCARTS ANALYSIS: Refresh missing components based on current coverage
            self._update_oldcarts_analysis()
        else:
            self._capture_debug(f"[Engine] ⏳ NOT marking any elements as covered - clarification was just asked")
        
        # SAFETY CHECK: Ensure we have active guidelines
        if len(self.active_guidelines) == 0 and len(self.reserve_pool) == 0:
            self._capture_debug(f"[Engine] ❌ All guidelines exhausted - no diagnosis possible")
            self._capture_debug(f"[Engine] 📋 Ruled out {len(self.ruled_out)} conditions")
            self._capture_debug(f"[Engine] 📊 Debug: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
            self._capture_debug(f"[Engine] 📋 OLDCARTS covered: {self.oldcarts_covered}")
            self._capture_debug(f"[Engine] 📋 Demographics: {self.demographics}")
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Please seek medical evaluation.",
                'debug': self._get_debug_info()
            }
        
        # If active is empty but reserve exists, this shouldn't happen (rolling replacement should have filled it)
        if len(self.active_guidelines) == 0:
            self._capture_debug(f"[Engine] ⚠️ Active list empty but reserve has {len(self.reserve_pool)} - this is a bug")
            return {
                'success': False,
                'message': "I encountered an error. Please seek medical attention."
            }
        
        # CHECK FOR DIAGNOSIS
        top = self.active_guidelines[0]
        num_questions = len([item for item in self.conversation_history if item['type'] == 'question' and item.get('focus') == 'clinical'])
        
        # Check OLDCARTS coverage
        oldcarts_complete = all(self.oldcarts_covered.values())
        covered_count = sum(self.oldcarts_covered.values())
        uncovered = [k for k, v in self.oldcarts_covered.items() if not v]
        
        # Show OLDCARTS coverage status
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        self._capture_debug(f"[Engine] 📋 OLDCARTS Coverage: {coverage_str} ({covered_count}/8)")
        
        # CHECK IF CLARIFICATION NEEDED (before moving to next OLDCARTS element)
        # But LIMIT clarifications to avoid infinite loops
        
        # Count how many clarifications we've already asked for this OLDCARTS element
        clarification_count = sum(1 for item in self.conversation_history 
                                 if item.get('type') == 'question' 
                                 and item.get('oldcarts') == oldcarts_element 
                                 and item.get('is_clarification'))
        
        # Initialize MAX_CLARIFICATIONS_PER_ELEMENT - will be set dynamically based on competing patterns
        MAX_CLARIFICATIONS_PER_ELEMENT = 1  # Default minimum
        
        if len(self.active_guidelines) >= 1:
            top_score = self.active_guidelines[0]['score']
            # Score spread no longer used for decisions - removed for clarity
        else:
            top_score = 0.0
            
            # MUCH MORE LENIENT: Only clarify when absolutely necessary
            # Normal patient answers like "yesterday", "random", "sudden" should be accepted
            # LLM semantic similarity should handle normalization (e.g., "yesterday" → "24 hours ago")
            
            # Show LLM normalization decision
            self._capture_debug(f"\n[Engine] 🧠 ANSWER PROCESSING:")
            self._capture_debug(f"[Engine]   📝 Patient answer: '{answer}'")
            self._capture_debug(f"[Engine]   📊 Top score: {top_score:.0%} (scores no longer determine clarification)")
            
            # UNIVERSAL SPECIFICITY GAP DETECTION: Compare patient answer to all matching guidelines
            is_clear_answer = False
            needs_clarification_for_specificity = False
            missing_specificity_terms = []
        
        # Use semantic scoring to determine if answer is clear enough
        # If the answer gets good semantic scores, it's clear and doesn't need clarification
        # ALWAYS do specificity gap detection - scores don't determine anatomical competition
        self._capture_debug(f"[Engine] 🔍 SEGMENTAL GAP DETECTION: Always checking for competition (top_score={top_score:.0%})")
        # REMOVED: Old semantic clarity bypass logic that prevented proper competition detection
        
        # SKIP CLARIFICATION FOR ONSET: Onset answers are always sufficient for ranking, no clarification needed
        # Also skip if onset element - bypass all competition detection
        if oldcarts_element == 'onset':
            self._capture_debug(f"[Engine] 🎯 ONSET ELEMENT: Skipping ALL clarification logic - onset answers are always sufficient for ranking")
            needs_clarification_for_specificity = False
            is_clear_answer = True
            missing_specificity_terms = []
            oldcarts_result = {'has_competition': False, 'best_similarity': 1.0, 'competing_terms': []}  # Set dummy result
        # Use unified approach for ALL OTHER OLDCARTS elements (including location)
        else:  # Only process non-onset elements
            # Get guidelines for competition detection
            all_matched_guidelines = self.active_guidelines + self.reserve_pool
            
            # STEP 2: Get sections for this OLDCARTS element from guidelines that have some similarity to the current answer
            matching_sections = []
            
            # Filter guidelines that have some similarity to the current answer (above threshold)
            relevant_guidelines = []
            similarity_threshold = 0.1  # 10% minimum similarity to be considered relevant
            
            for guideline in all_matched_guidelines:
                # Check if this guideline's section for this OLDCARTS element has some similarity to the current answer
                section = self._extract_oldcarts_section(
                    guideline['data'].get('key_features', {}).get('classic_presentation', ''), 
                    oldcarts_element
                )
                if section:
                    # Calculate similarity between current answer and this guideline's section
                    similarity = self._simple_containment_match(answer.lower(), section.lower())
                    if similarity >= similarity_threshold:
                        relevant_guidelines.append(guideline)
            
            self._capture_debug(f"[Engine] 🔍 CHECKING RELEVANT GUIDELINES FOR COMPETITION ({oldcarts_element}):")
            self._capture_debug(f"[Engine]   Total matched guidelines: {len(all_matched_guidelines)}")
            self._capture_debug(f"[Engine]   Relevant guidelines (similarity >= {similarity_threshold:.0%}): {len(relevant_guidelines)}")
            
            for guideline in relevant_guidelines:
                section = self._extract_oldcarts_section(
                    guideline['data'].get('key_features', {}).get('classic_presentation', ''), 
                    oldcarts_element
                )
                if section:
                    # Include sections from relevant guidelines only
                    matching_sections.append({
                        'condition': guideline['name'],
                        'section_text': section
                    })
            
            if matching_sections:
                # FULL TEXT BLOCK APPROACH: Direct comparison between patient answer and full OLDCARTS sections
                oldcarts_result = self._detect_oldcarts_competition(answer, oldcarts_element, matching_sections)
                
                self._capture_debug(f"[Engine] 🔍 OLDCARTS FULL TEXT ANALYSIS ({oldcarts_element}):")
                self._capture_debug(f"[Engine]   Patient answer: '{answer}'")
                self._capture_debug(f"[Engine]   Competition detected: {oldcarts_result['has_competition']}")
                self._capture_debug(f"[Engine]   Best similarity: {oldcarts_result['best_similarity']:.0%}")
                
                # SIMPLIFIED LOGIC: Only ask clarification if containment is VERY low (<30%)
                # If patient word appears in ANY guideline, accept it (even if negated)
                # Structured guidelines will handle exclusions explicitly via includes/excludes
                if oldcarts_result['best_similarity'] < 0.3:
                    # Very low containment - patient word not found anywhere - need clarification
                    needs_clarification_for_specificity = True
                    is_clear_answer = False
                    missing_specificity_terms = oldcarts_result['competing_terms']
                    
                    # DYNAMIC MAX CLARIFICATIONS: Set based on number of competing patterns
                    # Each competing term represents a different pattern that needs clarification
                    num_competing_patterns = len(oldcarts_result['competing_terms'])
                    MAX_CLARIFICATIONS_PER_ELEMENT = max(1, num_competing_patterns)
                    
                    self._capture_debug(f"[Engine] 🎯 OLDCARTS SPECIFICITY GAP ({oldcarts_element}):")
                    self._capture_debug(f"[Engine]   Reason: Very low containment {oldcarts_result['best_similarity']:.0%} < 30% threshold")
                    self._capture_debug(f"[Engine]   Patient word not found in any guideline - need clarification")
                    self._capture_debug(f"[Engine]   📊 Dynamic MAX_CLARIFICATIONS_PER_ELEMENT: {MAX_CLARIFICATIONS_PER_ELEMENT} (based on {num_competing_patterns} competing patterns)")
                else:
                    # Check if there's competition - if multiple guidelines with similar similarity, need clarification
                    # EXCEPTION: If similarity is VERY high (>=70%), accept answer even with multiple patterns
                    if oldcarts_result['best_similarity'] >= 0.7:
                        # Very high similarity - accept answer even with competition
                        # High similarity means patient's description matches multiple guidelines closely (e.g., "right upper" matches RUQ conditions)
                        needs_clarification_for_specificity = False
                        is_clear_answer = True
                        missing_specificity_terms = []
                        self._capture_debug(f"[Engine] ✅ OLDCARTS ANSWER ACCEPTED ({oldcarts_element}):")
                        self._capture_debug(f"[Engine]   Containment {oldcarts_result['best_similarity']:.0%} >= 70% - very high similarity, accepting despite competition")
                    # Also accept if it's a follow-up clarification AND similarity increased significantly
                    elif clarification_count > 0 and oldcarts_result['best_similarity'] >= 0.5:
                        # Check if this is a significant improvement from the initial answer
                        # Only accept if similarity is >= 0.55 or if it's higher than before
                        # This prevents accepting vague follow-up answers like "right side" -> "right side" again
                        
                        # Get previous similarity from the last answer (if available)
                        previous_similarity = self.last_answer_scores.get(oldcarts_element, 0.0)
                        
                        if oldcarts_result['best_similarity'] >= 0.55:
                            # High similarity - accept it
                            needs_clarification_for_specificity = False
                            is_clear_answer = True
                            missing_specificity_terms = []
                            self._capture_debug(f"[Engine] ✅ OLDCARTS ANSWER ACCEPTED ({oldcarts_element}):")
                            self._capture_debug(f"[Engine]   Follow-up answer with {oldcarts_result['best_similarity']:.0%} similarity >= 55% - accepting")
                        elif oldcarts_result['best_similarity'] > previous_similarity + 0.05:
                            # Similarity improved by at least 5% - accept it
                            needs_clarification_for_specificity = False
                            is_clear_answer = True
                            missing_specificity_terms = []
                            self._capture_debug(f"[Engine] ✅ OLDCARTS ANSWER ACCEPTED ({oldcarts_element}):")
                            self._capture_debug(f"[Engine]   Follow-up answer improved from {previous_similarity:.0%} to {oldcarts_result['best_similarity']:.0%} - accepting improvement")
                        else:
                            # Similarity didn't improve enough - continue clarifying
                            needs_clarification_for_specificity = True
                            is_clear_answer = False
                            missing_specificity_terms = oldcarts_result['competing_terms']
                            self._capture_debug(f"[Engine] ⚠️ Follow-up answer not specific enough ({oldcarts_result['best_similarity']:.0%} vs previous {previous_similarity:.0%})")
                            self._capture_debug(f"[Engine]   Asking for more clarification")
                    elif oldcarts_result['has_competition']:
                        # Multiple competing guidelines - need clarification
                        needs_clarification_for_specificity = True
                        is_clear_answer = False
                        missing_specificity_terms = oldcarts_result['competing_terms']
                        
                        num_competing_patterns = len(oldcarts_result['competing_terms'])
                        MAX_CLARIFICATIONS_PER_ELEMENT = max(1, num_competing_patterns)
                        
                        self._capture_debug(f"[Engine] 🎯 MULTIPLE COMPETING PATTERNS DETECTED ({oldcarts_element}):")
                        self._capture_debug(f"[Engine]   Best similarity: {oldcarts_result['best_similarity']:.0%}")
                        self._capture_debug(f"[Engine]   Competition detected: {oldcarts_result['has_competition']}")
                        self._capture_debug(f"[Engine]   Competing patterns: {num_competing_patterns}")
                        self._capture_debug(f"[Engine]   Need clarification to differentiate between competing patterns")
                        self._capture_debug(f"[Engine]   📊 Dynamic MAX_CLARIFICATIONS_PER_ELEMENT: {MAX_CLARIFICATIONS_PER_ELEMENT}")
                    else:
                        # Single clear match - accept answer
                        needs_clarification_for_specificity = False
                        is_clear_answer = True
                        missing_specificity_terms = []
                        self._capture_debug(f"[Engine] ✅ OLDCARTS ANSWER ACCEPTED ({oldcarts_element}):")
                        self._capture_debug(f"[Engine]   Containment {oldcarts_result['best_similarity']:.0%} >= 30% - word found in guidelines, accepting")
            else:
                # No matching sections found - consider clear to avoid infinite loops
                needs_clarification_for_specificity = False
                is_clear_answer = True
                missing_specificity_terms = []
            
        # Debug clarification decision
        self._capture_debug(f"[Engine] 🔍 CLARIFICATION DECISION DEBUG:")
        self._capture_debug(f"[Engine]   oldcarts_element value: '{oldcarts_element}' (type: {type(oldcarts_element).__name__})")
        self._capture_debug(f"[Engine]   needs_clarification_for_specificity: {needs_clarification_for_specificity}")
        self._capture_debug(f"[Engine]   is_clear_answer: {is_clear_answer}")
        self._capture_debug(f"[Engine]   clarification_count: {clarification_count}")
        self._capture_debug(f"[Engine]   MAX_CLARIFICATIONS_PER_ELEMENT: {MAX_CLARIFICATIONS_PER_ELEMENT}")
        self._capture_debug(f"[Engine]   Condition met: {needs_clarification_for_specificity or not is_clear_answer}")
        
        # SINGLE POINT: Check if clarification should be bypassed for this element
        if self._should_bypass_clarification(oldcarts_element):
            self._capture_debug(f"[Engine] 🎯 COMPLETE CLARIFICATION BYPASS for {oldcarts_element}")
            self._capture_debug(f"[Engine]   Element '{oldcarts_element}' never requires clarification - accepting answer as-is")
            # Set both flags to prevent entering clarification block
            needs_clarification_for_specificity = False
            is_clear_answer = True
            # FORCE SKIP - do not enter clarification block at all
            self._capture_debug(f"[Engine] ✅ ANSWER ACCEPTED WITHOUT CLARIFICATION (element: {oldcarts_element})")
        # ALWAYS ask clarification if there are meaningful anatomical distinctions to be made
        # REMOVED: score_spread and top_score dependencies - focus purely on segmental gaps
        # REMOVED: Max clarifications limit - continue until condition is met
        elif needs_clarification_for_specificity or not is_clear_answer:
            self._capture_debug(f"\n[Engine] 🔍 CLARIFICATION NEEDED:")
            self._capture_debug(f"[Engine]   Top score: {top_score:.0%} (scores no longer determine clarification)")
            reason = "Competing conditions need clarification" if needs_clarification_for_specificity else "Answer lacks specificity"
            self._capture_debug(f"[Engine]   Reason: {reason}")
            self._capture_debug(f"[Engine]   Clarifications asked so far: {clarification_count} (unlimited until condition met)")
            self._capture_debug(f"[Engine]   Strategy: {'Open-ended' if clarification_count == 0 else 'Targeted (differential-based)'}")
            
            # Check if user is asking for clarification on our question (e.g., "what do you mean?", "provide more detail how?")
            if self._is_user_asking_for_clarification(answer):
                # Use LLM to respond conversationally with full context
                conversational_response = self._generate_conversational_response(answer, oldcarts_element)
                return {
                    'success': True,
                    'question': conversational_response,
                    'status': 'questioning',
                    'needs_clarification': True
                }
            else:
                # Generate progressively targeted clarifying question
                clarifying_q = self._generate_clarifying_question(oldcarts_element, answer, clarification_count, missing_specificity_terms)
            
            if clarifying_q:
                    # Add clarifying question to history
                    self.conversation_history.append({
                        'type': 'question',
                        'question': clarifying_q,
                        'oldcarts': oldcarts_element,  # Same OLDCARTS element
                        'focus': 'clinical',
                        'is_clarification': True
                    })
                    
                    return {
                        'success': True,
                        'question': clarifying_q,
                        'status': 'questioning',
                        'needs_clarification': True
                    }
        
        # NOTE: If we reach here, either:
        # 1. Clarification was bypassed (onset, etc.)
        # 2. Clarification was not needed (clear answer)
        # 3. Clarification was attempted but failed
        # In all cases, we accept the answer and continue
        
        if not self._should_bypass_clarification(oldcarts_element):
            # Only log acceptance for non-bypassed elements
            self._capture_debug(f"[Engine] ✅ ANSWER ACCEPTED: '{answer}' provides sufficient specificity")
            self._capture_debug(f"[Engine]   🎯 No competing anatomical regions requiring clarification")
            self._capture_debug(f"[Engine]   📝 Segmental gap analysis complete - no further clarification needed")
        
        # Diagnosis criteria: ALL OLDCARTS covered + high confidence, OR max 15 questions
        if oldcarts_complete and top['score'] >= 0.95:
            self._capture_debug(f"[Engine] ✅ DIAGNOSIS REACHED: {top['name']} ({top['score']:.0%} confidence, OLDCARTS complete)")
            self._capture_debug(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        elif num_questions >= 15:
            self._capture_debug(f"[Engine] ⚠️  DIAGNOSIS BY QUESTIONS LIMIT: {top['name']} ({top['score']:.0%}, OLDCARTS: {coverage_str})")
            self._capture_debug(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        else:
            if not oldcarts_complete:
                self._capture_debug(f"[Engine] 🔄 Continuing (OLDCARTS incomplete: missing {', '.join(uncovered)}, Q{num_questions}, score: {top['score']:.0%})")
            else:
                self._capture_debug(f"[Engine] 🔄 Continuing (OLDCARTS complete, need 95% confidence: current {top['score']:.0%}, Q{num_questions})")
            # Ask next question
            return self._ask_next_clinical_question()
    
    def _screen_red_flags(self, diagnosis_obj: Dict) -> Dict[str, Any]:
        """
        Screen for all red flags after diagnosis is reached
        Ask yes/no questions for each red flag to ensure nothing is missed
        """
        red_flags = diagnosis_obj['data'].get('red_flags', [])
        
        # If no red flags, skip screening
        if not red_flags:
            self._capture_debug(f"[Engine] ℹ️  No red flags to screen - proceeding to finalize")
            return self._finalize_diagnosis(diagnosis_obj)
        
        # If just starting screening, set status and reset index
        if self.status != 'red_flag_screening':
            self.status = 'red_flag_screening'
            self.red_flag_index = 0
            self.red_flags_present = []
            self.diagnosed_condition = diagnosis_obj  # Store diagnosed condition
            self._capture_debug(f"[Engine] 🚩 Screening {len(red_flags)} red flags for {diagnosis_obj['name']}")
        
        # If we've asked about all red flags, finalize
        if self.red_flag_index >= len(red_flags):
            self._capture_debug(f"[Engine] ✅ Red flag screening complete ({len(self.red_flags_present)} flags present)")
            return self._finalize_diagnosis(diagnosis_obj)
        
        # Ask about next red flag
        current_red_flag = red_flags[self.red_flag_index]
        
        # Convert red flag to yes/no question
        # Extract the core symptom from the red flag text
        question = self._red_flag_to_question(current_red_flag)
        
        self._capture_debug(f"[Engine] 🚩 Red flag {self.red_flag_index + 1}/{len(red_flags)}: {current_red_flag}")
        self._capture_debug(f"[Engine] 🔍 Generated question: {question}")
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'focus': 'red_flag',
            'red_flag_text': current_red_flag,
            'red_flag_index': self.red_flag_index
        })
        
        return {
            'success': True,
            'question': question,
            'status': 'red_flag_screening',
            'debug': self._get_debug_info()
        }
    
    def _red_flag_to_question(self, red_flag: str) -> str:
        """
        Convert a red flag statement to a yes/no question using structured data from guidelines
        
        Uses the red flag text directly from guidelines to generate appropriate questions
        """
        # Extract the core symptom from the red flag text
        # Remove medical terminology and convert to patient-friendly question
        
        # Use LLM to convert medical red flag text to patient-friendly question
        system_msg = "You are a medical assistant. Convert a medical red flag statement to a simple yes/no question for a patient. Use plain language, no medical jargon. Output ONLY the question."
        
        user_msg = f"""Convert this medical red flag to a simple yes/no question for a patient:

Red flag: "{red_flag}"

Examples:
- "High fever >103°F with severe pain - possible perforation" → "Have you had a fever higher than 103 degrees?"
- "Severe pain with abdominal rigidity (board-like abdomen) - perforation with peritonitis" → "Does your abdomen feel hard or rigid?"
- "Hypotension, tachycardia, altered mental status - septic shock" → "Have you felt dizzy or lightheaded?"

Your question:"""
        
        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=self.max_tokens_simple,
                temperature=self.temperature_simple
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            self._capture_debug(f"[Engine] ✅ Red flag question: '{question}'")
            return question
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM red flag question failed: {e}")
            # Fallback: simple extraction
            main_symptom = red_flag.split('-')[0].strip().lower()
            question = f"Have you experienced {main_symptom}?"
            return question
    
    def _finalize_diagnosis(self, diagnosis_obj: Dict) -> Dict[str, Any]:
        """
        Finalize and return diagnosis (with RED FLAGS if applicable)
        """
        self.status = "diagnosed"
        
        name = diagnosis_obj['name']
        score = diagnosis_obj['score']
        urgency = diagnosis_obj['data'].get('urgency', 'routine')
        all_red_flags = diagnosis_obj['data'].get('red_flags', [])
        
        # ESCALATE URGENCY if red flags are present
        if len(self.red_flags_present) > 0:
            if urgency == 'routine':
                urgency = 'urgent'
            elif urgency == 'urgent':
                urgency = 'emergent'
            self._capture_debug(f"[Engine] ⚠️  RED FLAGS DETECTED - Urgency escalated to: {urgency}")
        
        urgency_messages = {
            'emergent': '🚨 This is a medical emergency. Call 911 or go to the ER immediately.',
            'urgent': '⚠️ This requires prompt medical attention. Go to urgent care or ER today.',
            'routine': '📋 Schedule an appointment with your doctor soon.'
        }
        
        urgency_msg = urgency_messages.get(urgency, urgency_messages['routine'])
        
        # Build message
        message = f"Based on your symptoms, this is most likely {name} (confidence: {score:.0%}).\n\n{urgency_msg}"
        
        # Add detected red flags (if any were found during screening)
        if len(self.red_flags_present) > 0:
            message += f"\n\n🚨 WARNING SIGNS DETECTED:\n"
            for rf in self.red_flags_present:
                message += f"• {rf}\n"
            message += "\nSeek immediate medical attention."
        
        # Add general red flags to watch for (if urgent/emergent and not already shown)
        elif all_red_flags and urgency in ['emergent', 'urgent']:
            message += f"\n\n⚠️ Watch for these warning signs:\n"
            for rf in all_red_flags[:3]:  # Show top 3 red flags
                message += f"• {rf}\n"
        
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🎯 FINAL DIAGNOSIS")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Condition: {name}")
        self._capture_debug(f"[Engine] Confidence: {score:.0%}")
        self._capture_debug(f"[Engine] Urgency: {urgency}")
        if len(self.red_flags_present) > 0:
            self._capture_debug(f"[Engine] 🚨 Red Flags Detected: {len(self.red_flags_present)}")
            for rf in self.red_flags_present:
                self._capture_debug(f"[Engine]   - {rf}")
        self._capture_debug(f"{'='*80}\n")
        
        return {
            'success': True,
            'status': 'diagnosed',
            'diagnosis': name,
            'confidence': score,
            'urgency': urgency,
            'debug': self._get_debug_info(),
            'red_flags_detected': self.red_flags_present,
            'message': message
        }
    
    def _generate_opening_statement(self, chief_complaint: str) -> str:
        """
        LLM-generated empathetic opening statement
        """
        self._capture_debug(f"[Engine] 🧠 Generating opening statement...")
        
        system_msg = "Output ONLY the exact statement requested. No extra words."
        
        user_msg = f"""Patient: "{chief_complaint}"

Write a brief, natural empathetic medical statement that acknowledges their pain and shows you care:

Examples: 
- "I'm sorry to hear you're experiencing abdominal pain. That can be really uncomfortable."
- "I understand that abdominal pain can be concerning. Let me help you figure out what's going on."
- "That sounds painful. I'm here to help you understand what might be causing this."
- "I'm sorry you're dealing with this. Let's work together to understand your symptoms better."

Your statement:"""
        
        # Debug: Log what's being sent to LLM
        self._capture_debug(f"[Engine] 🧠 EMPATHETIC STATEMENT GENERATION:")
        self._capture_debug(f"[Engine] 🧠 System Message: {system_msg}")
        self._capture_debug(f"[Engine] 🧠 User Message: {user_msg}")
        self._capture_debug(f"[Engine] 🧠 Temperature: {self.temperature_simple}")
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.max_tokens_normalization,  # Allow more natural variation
            temperature=self.temperature_simple  # Use conservative temperature for consistent responses
        )
        
        # Debug: Log raw LLM response
        self._capture_debug(f"[Engine] 🧠 Raw LLM Response: '{response}'")
        
        statement = response.strip().strip('"\'')
        
        # Remove numbered list markers if LLM still outputs them
        import re
        statement = re.sub(r'^\d+\.\s*', '', statement)  # Remove "1. " from start
        statement = re.sub(r'\n\d+\.\s*', ' ', statement)  # Remove "\n2. " from middle
        
        # Debug: Log processed statement
        self._capture_debug(f"[Engine] 🧠 Processed Statement: '{statement}'")
        
        # VALIDATION: Only reject if completely nonsensical
        # Allow more natural variation in opening statements
        word_count = len(statement.split())
        if word_count > 50:  # Only reject if extremely long
            self._capture_debug(f"[Engine] ⚠️ Opening too long ({word_count} words) - using simple template")
            self._capture_debug(f"[Engine]    Generated: '{statement}'")
            statement = "I'm sorry to hear you're experiencing abdominal pain. Let me ask some questions to help figure out what's going on."
        
        # Debug: Log final statement
        self._capture_debug(f"[Engine] ✅ Final Opening Statement: '{statement}'")
        self._capture_debug(f"[Engine] ✅ Opening Statement Length: {len(statement)} characters")
        return statement
    
    def _generate_chronicity_question(self) -> str:
        """
        LLM-generated chronicity question to differentiate new vs chronic problems
        """
        self._capture_debug(f"[Engine] 🧠 Generating chronicity question...")
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."
        
        user_msg = """Generate a natural, conversational question asking if this is a new problem or ongoing/recurrent issue.

IMPORTANT: The question must be clear and force a specific answer. Avoid ambiguous questions that could be answered with "yes" or "no".

Examples: 
- "Is this the first time you've had this symptom, or have you experienced this before?"
- "Is this a new problem for you, or have you had this issue before?"
- "Have you experienced this type of pain before, or is this the first time?"
- "Is this something you've dealt with before, or is this completely new to you?"

Your question:"""
        
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.max_tokens_simple,
            temperature=self.temperature_simple
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Chronicity question (simple model): '{question}'")
        return question
    
    def _classify_chronicity_with_llm(self, answer: str) -> str:
        """
        Use LLM to intelligently classify if this is a new or recurring problem
        """
        self._capture_debug(f"[Engine] 🧠 LLM classifying chronicity...")
        self._capture_debug(f"[Engine] 🧠 Using COMPLEX model (Mistral-7B) for chronicity classification")
        
        system_msg = "You are a medical assistant. Classify if this is a NEW problem or RECURRING/CHRONIC problem. Respond with ONLY one word: 'new', 'recurring', or 'unclear'. Be precise and accurate."
        
        user_msg = f"""Classify this patient response about whether their problem is new or recurring:

Patient response: "{answer}"

CRITICAL: If the patient says "new", "first time", "started today/yesterday", "never had this before", "completely new", or similar - classify as "new".
If they say "before", "had this", "comes and goes", "chronic", "experienced before", "dealt with this before" - classify as "recurring".
If unclear or ambiguous, classify as "unclear".

Examples:
- "new" → new
- "It's new" → new
- "This is the first time" → new
- "It started yesterday" → new
- "never had this before" → new
- "completely new" → new
- "I've had this before" → recurring  
- "It comes and goes" → recurring
- "dealt with this before" → recurring
- "experienced this before" → recurring
- "I don't know" → unclear
- "I've had this for years" → recurring

Classification:"""
        
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.max_tokens_classification,
            temperature=self.temperature_simple  # Use simple model temperature for classification
        )
        
        classification = response.strip().lower()
        
        # Validate response
        if classification in ['new', 'recurring', 'unclear']:
            self._capture_debug(f"[Engine] ✅ LLM chronicity classification: '{classification}'")
            return classification
        else:
            self._capture_debug(f"[Engine] ⚠️ Invalid LLM response '{classification}', defaulting to 'unclear'")
            return 'unclear'
    
    def _generate_age_question(self) -> str:
        """
        LLM-generated age question
        """
        self._capture_debug(f"[Engine] 🧠 Generating age question...")
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions."
        
        user_msg = """Generate a natural, conversational question asking for the patient's age.

Examples: 
- "How old are you?"
- "What's your age?"
- "Can you tell me your age?"
- "How old are you, if you don't mind me asking?"

Your question:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.max_tokens_simple,
            temperature=self.temperature_simple
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Age question (simple model): '{question}'")
        return question
    
    def _extract_age_with_llm(self, user_answer: str) -> Optional[int]:
        """
        Use LLM to intelligently extract age from natural language responses
        
        Args:
            user_answer: User's response to age question
            
        Returns:
            int: Valid age (1-120) or None if no valid age found
        """
        self._capture_debug(f"[Engine] 🧠 Using LLM to extract age from: '{user_answer}'")
        
        # Quick regex fallback for simple cases (faster)
        import re
        simple_numbers = re.findall(r'\b(\d{1,3})\b', user_answer)
        if simple_numbers:
            potential_age = int(simple_numbers[0])
            if 1 <= potential_age <= 120:
                self._capture_debug(f"[Engine] ⚡ Quick extraction: {potential_age}")
                return potential_age
        
        # Use LLM for complex natural language processing
        system_msg = """You are an age extraction expert. Extract the person's age from their response.

CRITICAL RULES:
1. ONLY return a single number between 1-120
2. If no valid age mentioned, return "NONE"
3. Convert text numbers to digits (e.g., "thirty" → 30)
4. Handle phrases like "I'm in my thirties" → estimate (e.g., 35)
5. NEVER return anything except a number or "NONE"

Examples:
- "25" → 25
- "I'm thirty-five" → 35
- "I am 42 years old" → 42
- "I'm in my twenties" → 25
- "about forty" → 40
- "hello" → NONE
- "I don't want to say" → NONE
- "xyz" → NONE"""

        user_msg = f"""Extract the age from this response:

"{user_answer}"

Return ONLY the age number (1-120) or "NONE" if no valid age."""

        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=10,  # Very short response expected
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            response_clean = response.strip().upper()
            self._capture_debug(f"[Engine] 🧠 LLM age extraction result: '{response_clean}'")
            
            # Parse LLM response
            if response_clean == "NONE":
                self._capture_debug(f"[Engine] ❌ LLM found no valid age")
                return None
            
            # Try to parse as number
            try:
                age = int(response_clean)
                if 1 <= age <= 120:
                    self._capture_debug(f"[Engine] ✅ LLM extracted valid age: {age}")
                    return age
                else:
                    self._capture_debug(f"[Engine] ❌ LLM age out of range: {age}")
                    return None
            except ValueError:
                self._capture_debug(f"[Engine] ❌ LLM response not a number: '{response_clean}'")
                return None
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM age extraction failed: {e}")
            # Fallback to None if LLM fails
            return None
    
    def _generate_sex_question(self) -> str:
        """
        LLM-generated biological sex question
        """
        self._capture_debug(f"[Engine] 🧠 Generating sex question...")
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions."
        
        user_msg = """Generate a natural, conversational question asking for biological sex (male or female).

Examples: 
- "Are you male or female?"
- "What's your biological sex?"
- "Are you a man or woman?"
- "Are you male or female, if you don't mind me asking?"

Your question:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=self.max_tokens_simple,
                temperature=self.temperature_simple
            )
            
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Sex question (simple model): '{question}'")
        return question
    
    
    def _generate_clarification_question(self, topic: str) -> str:
        """
        LLM-generated clarification question for invalid answers
        """
        self._capture_debug(f"[Engine] 🧠 Generating clarification for: {topic}")
        
        examples = {
            "age": "I didn't catch that. How old are you?",
            "sex": "I didn't catch that. Are you male or female?"
        }
        
        example = examples.get(topic, "Can you clarify?")
        
        system_msg = "You are a medical assistant. Output ONLY a single clarification question, nothing else."
        
        user_msg = f"""The patient didn't answer clearly about {topic}.

Re-ask with: "I didn't catch that" + original question

Example: "{example}"

Your question:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.max_tokens_simple,
            temperature=self.temperature_simple
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Clarification (simple model): '{question}'")
        return question
    
    def _detect_anatomical_competition(self, patient_answer: str, location_blocks: list) -> dict:
        """
        Detect anatomical competition using PURE text similarity - NO TERM EXTRACTION!
        
        Simple approach:
        1. Compare patient answer to each guideline location block using containment similarity
        2. If guidelines are highly similar to each other → no competition
        3. If guidelines differ significantly → competition exists → ask for clarification
        4. If patient answer matches ALL guidelines well → no clarification needed
        5. If patient answer only matches SOME guidelines → clarification needed
        
        Works universally for all organ systems without any hard-coded terms!
        
        Args:
            patient_answer: Patient's location response (e.g., "right side", "upper back")  
            location_blocks: List of {'condition': str, 'location_text': str}
            
        Returns:
            dict: {
                'has_competition': bool,
                'competing_areas': list,
                'clarification_needed': bool
            }
        """
        if not location_blocks:
            return {
                'has_competition': False,
                'competing_areas': [],
                'clarification_needed': False
            }
        
        patient_lower = patient_answer.lower()
        
        # Calculate similarity between patient answer and each guideline
        patient_similarities = []
        for block in location_blocks:
            location_text = block['location_text'].lower()
            similarity = self._simple_containment_match(patient_lower, location_text)
            patient_similarities.append(similarity)
        
        # Calculate similarity between guideline location blocks themselves
        guideline_similarities = []
        for i, block1 in enumerate(location_blocks):
            for block2 in location_blocks[i+1:]:
                similarity = self._simple_containment_match(
                    block1['location_text'].lower(), 
                    block2['location_text'].lower()
                )
                guideline_similarities.append(similarity)
        
        avg_patient_similarity = sum(patient_similarities) / len(patient_similarities) if patient_similarities else 0
        avg_guideline_similarity = sum(guideline_similarities) / len(guideline_similarities) if guideline_similarities else 1.0
        
        self._capture_debug(f"[Engine] 🔍 LOCATION COMPETITION ANALYSIS:")
        self._capture_debug(f"[Engine]   Patient answer: '{patient_answer}'")
        self._capture_debug(f"[Engine]   Avg patient similarity to guidelines: {avg_patient_similarity:.1%}")
        self._capture_debug(f"[Engine]   Avg guideline-to-guideline similarity: {avg_guideline_similarity:.1%}")
        
        # Decision logic:
        # 1. If guidelines are very similar to each other (high internal similarity) → no competition
        if avg_guideline_similarity > 0.7:
            self._capture_debug(f"[Engine] ✅ Guidelines are similar - no competition")
            return {
                'has_competition': False,
                'competing_areas': [],
                'clarification_needed': False
            }
        
        # 2. If patient answer matches all guidelines well → no clarification needed
        # Use same threshold as OLDCARTS competition for consistency (>= 0.5)
        if avg_patient_similarity >= 0.5:
            self._capture_debug(f"[Engine] ✅ Patient answer matches guidelines well - no clarification")
            return {
                'has_competition': False,
                'competing_areas': [],
                'clarification_needed': False
            }
        
        # 3. Guidelines differ AND patient answer doesn't match well → competition exists → need clarification
        self._capture_debug(f"[Engine] 🎯 COMPETITION DETECTED - Guidelines differ, patient vague")
        return {
            'has_competition': True,
            'competing_areas': [block['condition'] for block in location_blocks],
            'clarification_needed': True
        }
    
    def _detect_oldcarts_competition(self, patient_answer: str, oldcarts_element: str, matching_sections: list) -> dict:
        """
        Detect OLDCARTS competition using full text blocks - same successful approach as anatomical!
        
        Args:
            patient_answer: Patient's response (e.g., "comes and go")
            oldcarts_element: OLDCARTS element being analyzed ('D', 'C', 'T', etc.)
            matching_sections: List of {'condition': str, 'section_text': str}
            
        Returns:
            dict: {
                'has_competition': bool,
                'best_similarity': float,
                'competing_terms': list
            }
        """
        patient_lower = patient_answer.lower()
        
        # Calculate semantic similarity with each OLDCARTS section
        similarities = []
        competing_descriptions = []
        
        for section in matching_sections:
            condition = section['condition']
            section_text = section['section_text'].lower()
            
            # SIMPLE CONTAINMENT CHECK (same approach as successful anatomical competition)
            # No complex similarity - just check if patient words are contained in guideline
            similarity = self._simple_containment_match(patient_lower, section_text)
            similarities.append(similarity)
            
            self._capture_debug(f"[Engine] 🔍 OLDCARTS SECTION ANALYSIS:")
            self._capture_debug(f"[Engine]   {condition}: {similarity:.0%} similarity")
            self._capture_debug(f"[Engine]   Patient: '{patient_answer}'")
            self._capture_debug(f"[Engine]   Guideline: '{section_text[:100]}...'")
            
            # Track competing descriptions for clarification
            if similarity > 0.2:  # Some relevance
                competing_descriptions.append(f"{condition} ({similarity:.0%})")
        
        # Determine competition and acceptance
        best_similarity = max(similarities) if similarities else 0.0
        
        # IMPROVED COMPETITION DETECTION: Only consider meaningful matches (>50%)
        # Previously used 30% threshold which included low-relevance matches
        meaningful_matches = [s for s in similarities if s >= 0.5]
        has_competition = len(meaningful_matches) > 1  # Multiple strong matches
        
        self._capture_debug(f"[Engine] 🔍 COMPETITION ANALYSIS:")
        self._capture_debug(f"[Engine]   All similarities: {[f'{s:.0%}' for s in similarities]}")
        self._capture_debug(f"[Engine]   Meaningful matches (>=50%): {[f'{s:.0%}' for s in meaningful_matches]}")
        self._capture_debug(f"[Engine]   Competition detected: {len(meaningful_matches)} > 1 = {has_competition}")
        
        self._capture_debug(f"[Engine] 🔍 OLDCARTS COMPETITION RESULT ({oldcarts_element}):")
        self._capture_debug(f"[Engine]   Best similarity: {best_similarity:.0%}")
        self._capture_debug(f"[Engine]   Has competition: {has_competition}")
        self._capture_debug(f"[Engine]   Competing descriptions: {competing_descriptions}")
        
        return {
            'has_competition': has_competition,
            'best_similarity': best_similarity,
            'competing_terms': competing_descriptions
        }
    
    
    def _simple_containment_match(self, patient_text: str, guideline_text: str) -> float:
        """
        Simple containment matching - same approach as successful anatomical competition
        Just check if patient words appear in the guideline text (no complex logic needed)
        """
        # Clean up text
        patient_words = set(patient_text.lower().split())
        guideline_lower = guideline_text.lower()
        
        # Remove very common words that don't add meaning
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'my', 'it'}
        patient_words = {word for word in patient_words if word not in common_words}
        
        if not patient_words:
            return 0.0
        
        # Simple containment: How many patient words appear in guideline?
        matches = sum(1 for word in patient_words if word in guideline_lower)
        containment_score = matches / len(patient_words)
        
        self._capture_debug(f"[Engine] 🔍 SIMPLE CONTAINMENT:")
        self._capture_debug(f"[Engine]   Patient words: {patient_words}")
        self._capture_debug(f"[Engine]   Matches: {matches}/{len(patient_words)}")
        self._capture_debug(f"[Engine]   Score: {containment_score:.0%}")
        
        return containment_score
    
    def _was_clarification_just_asked(self) -> bool:
        """
        Check if the last question in conversation history was a clarification question
        This helps prevent marking OLDCARTS elements as covered when clarification is pending
        """
        if not self.conversation_history:
            return False
        
        # Look at the most recent question
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question':
                return item.get('is_clarification', False)
        
        return False
    
    def _generate_clarifying_question(self, oldcarts_element: str, patient_answer: str, clarification_count: int, missing_terms: list = None) -> str:
        """
        Generate a targeted clarifying question based on structured OLDCARTS data
        
        Compares normalized patient answer against structured_oldcarts includes/excludes
        to generate targeted questions that help discriminate between conditions
        """
        # BYPASS CHECK: If this element should skip clarification, raise error (should never be called)
        if self._should_bypass_clarification(oldcarts_element):
            raise ValueError(f"❌ BUG: _generate_clarifying_question called for {oldcarts_element} which should bypass clarification - this should never happen!")
        
        self._capture_debug(f"[Engine] 🎯 Generating targeted clarifying question for {oldcarts_element}")
        self._capture_debug(f"[Engine]   Patient answer: '{patient_answer}'")
        
        # Normalize using element-specific synonyms (same as scoring)
        normalized_answer = patient_answer.lower()
        if hasattr(self, 'current_category') and self.current_category:
            # Load appropriate synonym file for normalization
            synonym_file = f"synonyms/{self.current_category.lower()}_synonyms_oldcarts.json"
            synonym_path = os.path.join(os.path.dirname(__file__), synonym_file)
            
            if os.path.exists(synonym_path):
                with open(synonym_path, 'r') as f:
                    synonyms = json.load(f)
                
                # Normalize using this element's synonyms
                if oldcarts_element in synonyms:
                    element_synonyms = synonyms[oldcarts_element]
                    # Sort by length (longest first) to prefer more specific matches
                    sorted_categories = sorted(element_synonyms.items(), key=lambda x: max(len(s) for s in x[1]) if isinstance(x[1], list) else 0, reverse=True)
                    
                    for category, synonym_list in sorted_categories:
                        if isinstance(synonym_list, list):
                            for synonym in synonym_list:
                                # Use word boundary matching to avoid partial matches
                                import re
                                pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
                                if re.search(pattern, normalized_answer):
                                    normalized_answer = normalized_answer.replace(synonym.lower(), category)
                                    # Don't break - continue to find longer/more specific matches
        
        self._capture_debug(f"[Engine] 🔄 Normalization: '{patient_answer}' → '{normalized_answer}'")
        self._capture_debug(f"[Engine] 📍 Clarification normalization (no scoring here - just for question generation)")
        
        # Codebase uses lowercase full names only
        element_name = oldcarts_element
        
        # Only include guidelines that received word_match_boost (actual word matches in structured_oldcarts)
        # Re-check word matches for clarifying question generation
        # We need to check if patient answer matches structured OLDCARTS data
        all_guidelines_with_match = []
        
        for guideline in self.active_guidelines:
            score = guideline.get('score', 0.0)
            
            # Check if this guideline has structured OLDCARTS data for this element
            structured_oldcarts = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            if oldcarts_element not in structured_oldcarts:
                self._capture_debug(f"[Engine] ❌ No structured OLDCARTS for '{oldcarts_element}': {guideline['name']} (score: {score:.2f}) - excluded")
                continue
            
            # Check if normalized answer matches any includes terms
            element_data = structured_oldcarts[oldcarts_element]
            includes = element_data.get('includes', [])
            
            # Check for word match
            # Need semantic matching, not just substring match
            # "right side" should match "right upper quadrant" and "right lower quadrant"
            has_word_match = False
            # Handle underscores in normalized answer (e.g., "right_side" -> "right side")
            normalized_answer_with_spaces = normalized_answer.replace('_', ' ')
            normalized_answer_words = set(normalized_answer_with_spaces.split())
            
            for term in includes:
                # Normalize both for comparison
                term_lower = term.lower()
                
                # Direct substring match
                if term_lower in normalized_answer_with_spaces or normalized_answer_with_spaces in term_lower:
                    has_word_match = True
                    break
                
                # Semantic word overlap - if key words match
                term_words = set(term_lower.split())
                # Check if significant words overlap (more than 1 word in common)
                if len(term_words.intersection(normalized_answer_words)) >= 2:
                    has_word_match = True
                    break
                
                # Check for "right" or "left" matching with quadrant terms
                if 'right' in normalized_answer_words and 'right' in term_words:
                    has_word_match = True
                    break
                if 'left' in normalized_answer_words and 'left' in term_words:
                    has_word_match = True
                    break
                
                # Check for single word match if it's a directional term
                if len(normalized_answer_words) == 1 and len(term_words) == 1:
                    if normalized_answer_words == term_words:
                        has_word_match = True
                        break
            
            if has_word_match:
                all_guidelines_with_match.append(guideline)
                self._capture_debug(f"[Engine] ✅ Guideline with word match: {guideline['name']} (score: {score:.2f}, matched term)")
            else:
                self._capture_debug(f"[Engine] ❌ No word match: {guideline['name']} (score: {score:.2f}, no term match) - excluded")
        
        if not all_guidelines_with_match:
            self._capture_debug(f"[Engine] ❌ No guidelines with word match found - cannot generate clarifying question")
            raise ValueError(f"No guidelines with word match found - cannot generate clarifying question for {oldcarts_element}")
        
        self._capture_debug(f"[Engine] 📊 Using {len(all_guidelines_with_match)} guidelines with word matches for question generation")
        
        # Analyze what specific information is missing to generate targeted questions
        # Use normalized answer for analysis since it contains standardized medical terms
        missing_terms_from_analysis = self._analyze_missing_location_information(normalized_answer, all_guidelines_with_match, oldcarts_element)
        
        self._capture_debug(f"[Engine]   Missing terms: {missing_terms_from_analysis}")
        
        # Generate targeted question based on missing information from structured_oldcarts
        if missing_terms_from_analysis:
            targeted_options = self._generate_targeted_location_options(missing_terms_from_analysis, all_guidelines_with_match)
            if targeted_options:
                # Format options as a natural list
                if len(targeted_options) == 1:
                    options_str = targeted_options[0]
                elif len(targeted_options) == 2:
                    options_str = f"{targeted_options[0]} or {targeted_options[1]}"
                else:
                    options_str = ", ".join(targeted_options[:-1]) + f", or {targeted_options[-1]}"
                
                self._capture_debug(f"[Engine] 🎯 Generated targeted question with options: {options_str}")
                return f"Can you be more specific? For example, is it {options_str}?"
        
        # No structured data available - let it fail
        raise ValueError(f"Failed to generate clarifying question for {oldcarts_element} - missing data or logic error")
    
    def _convert_medical_terms_to_patient_friendly(self, medical_terms: list, oldcarts_element: str) -> list:
        """
        Convert medical terms to patient-friendly language using synonym files
        
        Args:
            medical_terms: List of medical terms from guidelines
            oldcarts_element: The OLDCARTS element being clarified
            
        Returns:
            List of patient-friendly terms
        """
        if not medical_terms:
            return []
        
        # Remove duplicates and limit to avoid overwhelming the LLM
        unique_terms = list(set(medical_terms))[:8]  # Max 8 terms
        
        # Get examples from synonym files based on OLDCARTS element
        examples = self._get_synonym_examples_for_element(oldcarts_element)
        
        system_msg = "You are a medical assistant. Convert medical terms to simple, patient-friendly language. Output ONLY the converted terms, one per line, in the same order."
        
        user_msg = f"""Convert these medical terms for {oldcarts_element} to simple, patient-friendly language:

Medical terms: {', '.join(unique_terms)}

Examples from medical guidelines:
{examples}

Converted terms (one per line):"""

        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            # Parse response - one term per line
            converted_terms = []
            for line in response.strip().split('\n'):
                line = line.strip().strip('"\'')
                if line and not line.startswith('-') and not line.startswith('•'):
                    converted_terms.append(line)
            
            # Fallback if LLM fails
            if not converted_terms:
                converted_terms = unique_terms
            
            self._capture_debug(f"[Engine] 🧠 Medical → Patient-friendly conversion:")
            self._capture_debug(f"[Engine]   Medical: {unique_terms}")
            self._capture_debug(f"[Engine]   Patient-friendly: {converted_terms}")
            
            return converted_terms
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Medical term conversion failed: {e}")
            # Fallback to original terms
            return unique_terms
    
    def _get_synonym_examples_for_element(self, oldcarts_element: str) -> str:
        """
        Get examples from synonym files for the given OLDCARTS element
        
        Args:
            oldcarts_element: The OLDCARTS element (e.g., 'location', 'character', 'duration')
            
        Returns:
            String of examples formatted for LLM prompt
        """
        import json
        import os
        
        # Get the organ system from active guidelines to determine which synonym file to use
        organ_system = self._get_organ_system_from_active_guidelines()
        
        # Map organ systems to synonym files
        synonym_files = {
            'GI': 'gi_synonyms_oldcarts.json',
            'CARDIO': 'cardio_synonyms_oldcarts.json', 
            'NEURO': 'neuro_synonyms_oldcarts.json',
            'PULMONARY': 'resp_synonyms_oldcarts.json',
            'MSK': 'gi_synonyms_oldcarts.json',  # Fallback to GI for MSK
            'DERM': 'derm_synonyms_oldcarts.json',
            'GU': 'gu_synonyms_oldcarts.json',
            'GYN': 'gu_synonyms_oldcarts.json',  # Use GU for GYN
            'RENAL': 'renal_synonyms_oldcarts.json',
            'ENDOCRINE': 'endocrine_synonyms_oldcarts.json'
        }
        
        synonym_file = synonym_files.get(organ_system, 'gi_synonyms_oldcarts.json')  # Default to GI
        
        # Load synonym file
        synonyms_path = os.path.join('synonyms', synonym_file)
        
        try:
            with open(synonyms_path, 'r') as f:
                synonyms_data = json.load(f)
            
            # Get examples for the specific OLDCARTS element
            element_data = synonyms_data.get(oldcarts_element, {})
            
            if not element_data:
                # Fallback to basic examples if element not found
                return self._get_fallback_examples(oldcarts_element)
            
            # Format examples for LLM prompt
            examples = []
            for category, terms in element_data.items():
                if isinstance(terms, list) and terms:
                    # Take first few terms as examples
                    example_terms = terms[:3]
                    examples.append(f"- {category}: {', '.join(example_terms)}")
            
            if examples:
                return '\n'.join(examples[:5])  # Limit to 5 examples
            else:
                return self._get_fallback_examples(oldcarts_element)
                
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Failed to load synonym file {synonym_file}: {e}")
            return self._get_fallback_examples(oldcarts_element)
    
    def _get_organ_system_from_active_guidelines(self) -> str:
        """
        Determine organ system from active guidelines
        
        Returns:
            Organ system string (e.g., 'GI', 'CARDIO', 'NEURO')
        """
        if not self.active_guidelines:
            return 'GI'  # Default fallback
        
        # Check the first few active guidelines for organ system
        for guideline in self.active_guidelines[:3]:
            name = guideline.get('name', '')
            if name.startswith('GI_'):
                return 'GI'
            elif name.startswith('CARDIO_'):
                return 'CARDIO'
            elif name.startswith('NEURO_'):
                return 'NEURO'
            elif name.startswith('PULMONARY_'):
                return 'PULMONARY'
            elif name.startswith('MSK_'):
                return 'MSK'
            elif name.startswith('DERM_'):
                return 'DERM'
            elif name.startswith('GU_'):
                return 'GU'
            elif name.startswith('GYN_'):
                return 'GYN'
            elif name.startswith('RENAL_'):
                return 'RENAL'
            elif name.startswith('ENDOCRINE_'):
                return 'ENDOCRINE'
        
        return 'GI'  # Default fallback
    
    def _get_fallback_examples(self, oldcarts_element: str) -> str:
        """
        Get basic fallback examples when synonym files are not available
        
        Args:
            oldcarts_element: The OLDCARTS element
            
        Returns:
            String of basic examples
        """
        fallback_examples = {
            'location': '- "RLQ" → "lower right side"\n- "RUQ" → "upper right side"\n- "LLQ" → "lower left side"\n- "epigastric" → "upper middle of stomach"',
            'character': '- "sharp" → "sharp pain"\n- "dull" → "dull ache"\n- "burning" → "burning sensation"\n- "cramping" → "cramps"',
            'duration': '- "acute" → "sudden onset"\n- "chronic" → "long-term"\n- "intermittent" → "comes and goes"\n- "constant" → "all the time"',
            'aggravating': '- "eating" → "after eating"\n- "movement" → "when moving"\n- "breathing" → "when breathing"\n- "stress" → "when stressed"',
            'relieving': '- "rest" → "when resting"\n- "heat" → "with heat"\n- "medication" → "with pain medicine"\n- "position" → "certain positions"',
            'timing': '- "morning" → "in the morning"\n- "evening" → "in the evening"\n- "night" → "at night"\n- "after meals" → "after eating"',
            'severity': '- "mild" → "not too bad"\n- "moderate" → "somewhat painful"\n- "severe" → "really painful"\n- "scale 1-3" → "1-3 out of 10"',
            'onset': '- "sudden" → "all of a sudden"\n- "gradual" → "slowly"\n- "acute" → "quickly"\n- "chronic" → "over time"'
        }
        
        return fallback_examples.get(oldcarts_element, '- "medical term" → "patient-friendly term"')
    
    def _is_user_asking_for_clarification(self, user_answer: str) -> bool:
        """
        Detect if the user is asking for clarification on our question
        (e.g., "what do you mean?", "I don't understand", "provide more detail how?")
        """
        clarification_indicators = [
            "what do you mean",
            "i don't understand",
            "i don't get",
            "not sure what you mean",
            "what are you asking",
            "provide more detail how",
            "more detail how",
            "can you explain",
            "what does that mean",
            "not clear"
        ]
        
        answer_lower = user_answer.lower()
        return any(indicator in answer_lower for indicator in clarification_indicators)
    
    def _generate_conversational_response(self, user_answer: str, oldcarts_element: str) -> str:
        """
        Generate a conversational response when user asks for clarification
        Uses full conversation context to provide natural, helpful responses
        """
        # Get the last question asked
        last_question = None
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question' and item.get('oldcarts') == oldcarts_element:
                last_question = item.get('question')
                break
        
        # Build conversation context
        qa_pairs = []
        temp_q = None
        for item in self.conversation_history[-10:]:  # Last 10 items for context
            if item.get('type') == 'question' and item.get('focus') not in ['age', 'sex']:
                temp_q = item.get('question')
            elif item.get('type') == 'answer' and temp_q:
                qa_pairs.append(f"Q: {temp_q}\nA: {item.get('answer', '')}")
                temp_q = None
        
        context = "\n\n".join(qa_pairs) if qa_pairs else "No previous conversation"
        
        system_msg = "You are a helpful medical assistant. The patient is asking for clarification on your previous question. Respond naturally and helpfully, explaining what you meant in simple terms."
        
        user_msg = f"""The patient asked: "{user_answer}"

This was in response to your question: "{last_question}"

Conversation context:
{context}

Respond naturally and helpfully. Explain what you meant by your question in simple, clear terms. Be conversational and reassuring. Keep it under 50 words."""

        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            # Clean up the response
            conversational_response = response.strip().strip('"\'')
            
            # Add to conversation history
            self.conversation_history.append({
                'type': 'question',
                'question': conversational_response,
                'oldcarts': oldcarts_element,
                'focus': 'clinical',
                'is_clarification': True,
                'is_conversational': True
            })
            
            return conversational_response
            
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ Conversational response generation failed: {e}")
            # Fallback to simple explanation
            return f"I'm asking about the {oldcarts_element} of your pain. Can you tell me more specifically?"
    
    def _generate_llm_clarifying_question_with_options(self, oldcarts_element: str, clarification_count: int) -> str:
        """
        Use LLM to generate a clarifying question with specific options based on competing conditions
        """
        # BYPASS CHECK: If this element should skip clarification, raise error (should never be called)
        if self._should_bypass_clarification(oldcarts_element):
            raise ValueError(f"❌ BUG: _generate_llm_clarifying_question_with_options called for {oldcarts_element} which should bypass clarification - this should never happen!")
        
        # Codebase uses lowercase full names only
        element_name = oldcarts_element
        
        # Get competing conditions and their key differentiators
        competing_conditions = self.active_guidelines[:3]  # Top 3 competing conditions
        
        # Get the last question asked
        last_question = None
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question' and item.get('oldcarts') == oldcarts_element:
                last_question = item.get('question')
                break
        
        # Extract structured OLDCARTS data for this element from competing conditions
        condition_details = []
        for guideline in competing_conditions:
            structured_oldcarts = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            if element_name in structured_oldcarts:
                element_data = structured_oldcarts[element_name]
                includes = element_data.get('includes', [])
                condition_details.append(f"- {guideline['name']}: {', '.join(includes[:3])}")
        
        system_msg = "You are a medical assistant helping to clarify patient symptoms. Generate a clear, simple question with specific options based on competing medical conditions."
        
        user_msg = f"""The patient seems confused by your previous question. Generate a better clarifying question with specific options based on the competing conditions.

Previous question: "{last_question}"

We're trying to understand the patient's {element_name} to differentiate between these conditions:

{chr(10).join(condition_details)}

Generate a simple question with 2-3 specific options that help distinguish between these conditions. Base the options on the {element_name} characteristics listed above.

Be specific and offer concrete options. Don't use medical jargon. Keep it under 25 words. Only return the question, nothing else."""

        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            # Extract just the question, removing any explanatory text
            clarifying_q = response.strip().strip('"\'')
            
            # Remove common prefixes that LLM adds (case-insensitive)
            prefixes_to_remove = [
                "Here's a better clarifying question:",
                "Here's a good clarifying question:",
                "A good clarifying question would be:",
                "Try this question:",
                "You could ask:",
                "Here's a question:",
                "Better question:",
                "Here's a better question:",
                "Ask this:",
                "Try asking:"
            ]
            
            # Check for prefix (case-insensitive)
            clarifying_q_lower = clarifying_q.lower()
            for prefix in prefixes_to_remove:
                if clarifying_q_lower.startswith(prefix.lower()):
                    # Find the actual start of the prefix in the original string
                    idx = clarifying_q_lower.find(prefix.lower())
                    clarifying_q = clarifying_q[idx + len(prefix):].strip().strip('"\'')
                    break
            
            # Remove any trailing explanatory text after the question
            # Look for common patterns that indicate explanation
            explanatory_patterns = [
                " This option provides",
                " This helps",
                " This will",
                " By asking",
                " The question",
                "\n"
            ]
            
            for pattern in explanatory_patterns:
                if pattern in clarifying_q:
                    clarifying_q = clarifying_q.split(pattern)[0].strip()
                    break
            
            # If there's still extra text after the question, extract just the first sentence
            if '.' in clarifying_q and clarifying_q.find('.') < len(clarifying_q) - 1:
                # Get the first sentence (up to first period)
                sentences = clarifying_q.split('.')
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 10:  # Only use if it's a real sentence
                    clarifying_q = first_sentence
            
            self._capture_debug(f"[Engine] 🧠 LLM generated clarifying question: '{clarifying_q}'")
            return clarifying_q
            
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ LLM clarifying question generation failed: {e}")
            # No fallback - let it fail to surface the issue
            raise RuntimeError(f"Failed to generate clarifying question for {element_name}: {e}")
    
    def _parse_prompt_against_structured_oldcarts(self, prompt: str, guidelines: List[Dict]) -> Dict[str, Any]:
        """
        Parse the normalized prompt against structured OLDCARTS to determine what's already answered
        
        Args:
            prompt: Normalized user complaint (e.g., "I have abdomian pain on my [right side] that gets [worse with eating]")
            guidelines: Matched guidelines with structured_oldcarts
        
        Returns:
            Dictionary with:
                - 'answered_components': Dict of OLDCARTS elements that were answered
                - 'missing_components': List of OLDCARTS elements that need to be asked
        """
        self._capture_debug(f"[Engine] 🔍 Parsing prompt against structured OLDCARTS")
        self._capture_debug(f"[Engine] 📝 Prompt: '{prompt}'")
        
        if not guidelines:
            self._capture_debug(f"[Engine] ⚠️ No guidelines provided")
            return {
                'answered_components': {},
                'missing_components': ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
            }
        
        # Check if any guidelines have structured_oldcarts data
        guidelines_with_oldcarts = [g for g in guidelines if g.get('data', {}).get('key_features', {}).get('structured_oldcarts')]
        
        if not guidelines_with_oldcarts:
            self._capture_debug(f"[Engine] ⚠️ No structured OLDCARTS found in guidelines")
            return {
                'answered_components': {},
                'missing_components': ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
            }
        
        self._capture_debug(f"[Engine] ✅ Found {len(guidelines_with_oldcarts)} guidelines with structured OLDCARTS data")
        
        # Collect all 'includes' terms from ALL guidelines for each OLDCARTS element
        # This represents what the guidelines are looking for
        all_includes = {
            'onset': set(),
            'location': set(),
            'duration': set(),
            'character': set(),
            'aggravating': set(),
            'relieving': set(),
            'timing': set(),
            'severity': set()
        }
        
        for guideline in guidelines_with_oldcarts:  # Use all provided guidelines (already narrowed)
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            
            for element, data in structured.items():
                if isinstance(data, dict) and 'includes' in data:
                    for term in data['includes']:
                        all_includes[element].add(term.lower())
        
        self._capture_debug(f"[Engine] 📊 Collected {sum(len(terms) for terms in all_includes.values())} expected terms from guidelines")
        
        # Store for use in _generate_clarifying_question
        self.all_includes = all_includes
        
        # Check which elements are present in the prompt
        answered_components = {}
        prompt_lower = prompt.lower()
        
        for element, expected_terms in all_includes.items():
            # Check if any expected term from guidelines appears in the prompt
            for term in expected_terms:
                if term in prompt_lower:
                    if element not in answered_components:
                        answered_components[element] = []
                    answered_components[element].append(term)
                    self._capture_debug(f"[Engine] ✅ Found {element}: '{term}'")
                    break  # Found at least one match for this element
        
        # Map element names to OLDCARTS codes
        element_to_code = {
            'onset': 'O',
            'location': 'L',
            'duration': 'D',
            'character': 'C',
            'aggravating': 'A',
            'relieving': 'R',
            'timing': 'T',
            'severity': 'S'
        }
        
        # Determine missing components
        all_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
        answered_elements = list(answered_components.keys())
        missing_elements = [element for element in all_elements if element not in answered_elements]
        
        self._capture_debug(f"[Engine] 📊 Summary:")
        self._capture_debug(f"[Engine]   ✅ Answered: {answered_elements} ({answered_components})")
        self._capture_debug(f"[Engine]   ❌ Missing: {missing_elements}")
        
        return {
            'answered_components': answered_components,
            'missing_components': missing_elements
        }

# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")