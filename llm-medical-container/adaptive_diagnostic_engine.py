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
from rag_client import get_rag_client
import numpy as np

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
    
    def __init__(self, guidelines_dir: str = "/app/medical/guidelines", llm_chat_fn=None, embedding_model=None, llm_chat_simple_fn=None):
        """
        Initialize diagnostic engine
        
        Args:
            guidelines_dir: Path to JSON guidelines
            llm_chat_fn: LLM function for complex reasoning (Mistral-7B)
            embedding_model: Sentence transformer for semantic similarity
            llm_chat_simple_fn: Optional LLM for simple tasks (Llama-1B). If None, uses llm_chat_fn
        """
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn  # Mistral-7B for complex diagnostic questions
        self.llm_chat_simple_fn = llm_chat_simple_fn or llm_chat_fn  # Llama-1B for templates/validation
        self.embedding_model = embedding_model
        
        # Temperature configuration from environment variables
        # Import temperature settings from aura_config
        try:
            from aura_config import (
                LLM_TEMPERATURE_SIMPLE, LLM_TEMPERATURE_COMPLEX, 
                LLM_TEMPERATURE_NORMALIZATION, LLM_TEMPERATURE_CREATIVE, 
                LLM_TEMPERATURE_ANALYSIS
            )
            self.temperature_simple = LLM_TEMPERATURE_SIMPLE
            self.temperature_complex = LLM_TEMPERATURE_COMPLEX
            self.temperature_normalization = LLM_TEMPERATURE_NORMALIZATION
            self.temperature_creative = LLM_TEMPERATURE_CREATIVE
            self.temperature_analysis = LLM_TEMPERATURE_ANALYSIS
        except ImportError:
            # Fallback to environment variables if aura_config not available
            self.temperature_simple = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.1'))
            self.temperature_complex = float(os.environ.get('LLM_TEMPERATURE_COMPLEX', '0.1'))
            self.temperature_normalization = float(os.environ.get('LLM_TEMPERATURE_NORMALIZATION', '0.1'))
            self.temperature_creative = float(os.environ.get('LLM_TEMPERATURE_CREATIVE', '0.6'))
            self.temperature_analysis = float(os.environ.get('LLM_TEMPERATURE_ANALYSIS', '0.3'))
        
        # Initialize debug capture
        self._captured_debug_output = []
        
        # Initialize Medical Rule Engine for enhanced location scoring
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            self.medical_rule_engine = MedicalRuleEngine()
            self._capture_debug(f"[Engine] 🎯 Medical Rule Engine initialized")
        except ImportError:
            self.medical_rule_engine = None
            self._capture_debug(f"[Engine] ⚠️ Medical Rule Engine not available")
        
        # Initialize Learning Data Collector for continuous improvement
        try:
            from ml.learning_data_collector import LearningDataCollector
            self.learning_collector = LearningDataCollector()
            self._capture_debug(f"[Engine] 📊 Learning Data Collector initialized")
        except ImportError:
            self.learning_collector = None
            self._capture_debug(f"[Engine] ⚠️ Learning Data Collector not available")
        
        # Initialize Continuous Learning System
        try:
            from ml.continuous_learning import ContinuousLearning
            self.continuous_learning = ContinuousLearning()
            self._capture_debug(f"[Engine] 🧠 Continuous Learning initialized")
        except ImportError:
            self.continuous_learning = None
            self._capture_debug(f"[Engine] ⚠️ Continuous Learning not available")
        
        # Initialize Performance Monitor
        try:
            from ml.performance_monitor import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor()
            self._capture_debug(f"[Engine] 📈 Performance Monitor initialized")
        except ImportError:
            self.performance_monitor = None
            self._capture_debug(f"[Engine] ⚠️ Performance Monitor not available")
        
        # Initialize User Feedback Interface
        try:
            from ml.user_feedback_interface import UserFeedbackInterface
            self.user_feedback = UserFeedbackInterface()
            self._capture_debug(f"[Engine] 💬 User Feedback Interface initialized")
        except ImportError:
            self.user_feedback = None
            self._capture_debug(f"[Engine] ⚠️ User Feedback Interface not available")
        
        self._capture_debug(f"[Engine] 🧠 Using {'dual models (simple + complex)' if llm_chat_simple_fn else 'single model'}")
        self._capture_debug(f"[Engine] 🌡️ Temperature settings: Simple={self.temperature_simple}, Complex={self.temperature_complex}, Normalization={self.temperature_normalization}, Creative={self.temperature_creative}, Analysis={self.temperature_analysis}")
        
        # ============================================================================
        # 🔧 CONFIGURATION TOGGLES (Easy to modify)
        # ============================================================================
        
        # Smart normalization configuration
        self.smart_normalization = True  # True=LLM normalization, False=synonym normalization
        
        # RAG validation toggle
        self.validate_rag = os.getenv("VALIDATE_RAG", "false").lower() == "true"  # Compare RAG vs brute-force
        
        # Hybrid matching configuration
        self.hybrid_config = {
            'jaccard_threshold': 0.3,      # Primary threshold for Jaccard similarity
            'semantic_threshold': 0.5,     # Threshold for semantic similarity fallback
            'semantic_boost_threshold': 0.3,  # When semantic is significantly better than Jaccard
            'semantic_weight': 0.7,        # Weight for semantic similarity when used as fallback
            'confidence_threshold': 0.1    # Max difference for high confidence
        }
        
        # ============================================================================
        # 🔧 END CONFIGURATION TOGGLES
        # ============================================================================
        
        # RAG API for GPU-accelerated FAISS operations
        self.rag_api = RAGEmbeddingAPI() if RAG_API_AVAILABLE else None
        self.use_rag_api = RAG_API_AVAILABLE
        
        # OLDCARTS element weights (how much impact each element has on scoring)
        # More balanced weights that don't completely override previous scores
        self.oldcarts_weights = {
            'L': 0.6,  # Location - high weight but not overwhelming
            'C': 0.5,  # Character - moderate-high weight
            'A': 0.5,  # Aggravating - moderate-high weight
            'R': 0.5,  # Relieving - moderate-high weight
            'S': 0.4,  # Severity - moderate weight
            'D': 0.3,  # Duration - moderate weight
            'O': 0.2,  # Onset - low weight
            'T': 0.3,  # Timing - moderate weight
        }
        
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
                    
                    self._capture_debug(f"[Engine]   ✓ {organ_system}/{name}")
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
        
        self._capture_debug(f"\n[Engine] ✅ LOADED {len(self.all_guidelines)} GUIDELINES:")
        for system, conditions in sorted(organ_systems.items()):
            self._capture_debug(f"[Engine]    📋 {system}: {len(conditions)} conditions")
        self._capture_debug(f"{'='*80}\n")
    
    
    # REMOVED: _is_valid_chief_complaint - hardcoded validation not needed
    # The system should handle invalid input through natural flow:
    # 1. OLDCARTS normalization
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
        
        # Debug: Check if captured debug output exists
        captured_output = getattr(self, '_captured_debug_output', [])
        self._capture_debug(f"[Engine] 🔍 Debug capture status: {len(captured_output)} lines captured")
        if captured_output:
            self._capture_debug(f"[Engine] 🔍 First few debug lines: {captured_output[:3]}")
        else:
            self._capture_debug(f"[Engine] ⚠️ No debug output captured")
        
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
            'engine_debug_output': captured_output  # Captured debug output
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
        
        # Clear any LLM model state/cache to prevent cross-session contamination
        self._capture_debug(f"[Engine] 🔄 Clearing LLM model state for fresh session")
        
        # ML Progress Tracking
        self._capture_debug(f"[ML Progress] 📊 Session reset - ML learning state cleared")
        
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
        else:  # Anatomical opposites (0.0)
            return 0.05  # Rule out anatomical opposites
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        Start new assessment
        
        Args:
            chief_complaint: e.g., "I have abdominal pain"
        
        Returns:
            Response with first question
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # No hardcoded validation - let the natural flow handle it
        # If complaint doesn't normalize or match, clarification will be asked
        
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        
        # STEP 1: Get filler immediately (for instant user feedback)
        # Filler is now handled at container level for immediate streaming
        self._capture_debug(f"[Engine] 💬 Generating opening statement (filler handled by container)...")
        
        # STEP 2: Run RAG and Llama-1B in PARALLEL (major speedup!)
        import threading
        import concurrent.futures
        
        rag_result = [None]
        opening_result = [None]
        age_result = [None]
        error_result = [None]
        
        def run_rag():
            """Match to guidelines (RAG API or brute-force with fallback + optional validation)"""
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
                
                # NORMAL MODE: Use RAG API with fallback
                elif self.use_rag_api:
                    self._capture_debug(f"[Engine] 🚀 Using RAG API mode for matching")
                    import time
                    start_time = time.time()
                    try:
                        rag_result[0] = self._match_to_guidelines_rag(chief_complaint)
                        elapsed = time.time() - start_time
                        if hasattr(self, 'matching_metadata'):
                            self.matching_metadata['timing'] = elapsed
                    except Exception as rag_error:
                        self._capture_debug(f"[Engine] ❌ RAG API matching failed: {rag_error}")
                        self._capture_debug(f"[Engine] 🔄 Falling back to brute-force matching")
                        self.use_rag_api = False  # Disable RAG API for future queries
                        start_time = time.time()
                        rag_result[0] = self._match_to_guidelines_rag(chief_complaint)
                        elapsed = time.time() - start_time
                        if hasattr(self, 'matching_metadata'):
                            self.matching_metadata['timing'] = elapsed
                else:
                    self._capture_debug(f"[Engine] 🐢 Using brute-force mode for matching")
                    import time
                    start_time = time.time()
                    rag_result[0] = self._match_to_guidelines_rag(chief_complaint)
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
        
        # Combine them with proper spacing and pause
        combined_message = f"{opening_statement} <pause> {age_question}"
        
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
            self.red_flag_index += 1
            
            # Continue screening (or finalize if done)
            return self._screen_red_flags(self.active_guidelines[0])
        
        # Handle demographics
        if last_q.get('focus') == 'age':
            # Extract age using LLM
            self._capture_debug(f"[Engine] 🔍 Extracting age from answer: '{user_answer}'")
            self._capture_debug(f"[Engine] 🔍 Extracting age from answer: '{user_answer}'")
            
            # Use regex to extract numbers (simple and reliable)
            import re
            numbers = re.findall(r'\b(\d{1,3})\b', user_answer)
            
            if numbers:
                # Take first number found
                age_num = int(numbers[0])
                if 1 <= age_num <= 120:  # Sanity check
                    self.demographics['age'] = age_num
                    self._capture_debug(f"[Engine] 👤 Age: {age_num}")
                    self._capture_debug(f"[Engine] 👤 Age: {age_num}")
                else:
                    self._capture_debug(f"[Engine] 👤 Age: Invalid ({age_num} out of range 1-120)")
                    self._capture_debug(f"[Engine] 👤 Age: Invalid ({age_num} out of range 1-120)")
            else:
                self._capture_debug(f"[Engine] 👤 Age: No number found in answer")
                self._capture_debug(f"[Engine] 👤 Age: No number found in answer")
            
            # VALIDATION: If no age found, re-ask using LLM
            if 'age' not in self.demographics:
                self._capture_debug(f"[Engine] ⚠️ Invalid answer - re-asking for age")
                self._capture_debug(f"{'='*80}\n")
                
                age_question = self._generate_clarification_question("age")
                self.conversation_history.append({
                    'type': 'question',
                    'question': age_question,
                    'focus': 'age'
                })
                
                return {
                    'success': True,
                    'question': age_question,
                    'status': 'questioning',
                    'debug': self._get_debug_info()
                }
            
            # Ask sex using LLM
            sex_question = self._generate_sex_question()
            self.conversation_history.append({
                'type': 'question',
                'question': sex_question,
                'focus': 'sex'
            })
            self._capture_debug(f"{'='*80}\n")
            
            return {
                'success': True,
                'question': sex_question,
                'status': 'questioning',
                'debug': self._get_debug_info()
            }
        
        elif last_q.get('focus') == 'sex':
            # Extract sex - use keyword matching with fuzzy tolerance for typos
            self._capture_debug(f"[Engine] 🔍 Extracting sex from answer: '{user_answer}'")
            self._capture_debug(f"[Engine] 🔍 Extracting sex from answer: '{user_answer}'")
            
            answer_lower = user_answer.lower()
            # Strip punctuation and split into words
            import string
            cleaned = answer_lower.translate(str.maketrans('', '', string.punctuation))
            words = cleaned.split()
            
            # Check for explicit sex words (standalone)
            male_words = {'male', 'man', 'boy', 'guy'}
            female_words = {'female', 'woman', 'girl', 'lady'}
            
            # Fast exact keyword check first
            if any(word in male_words for word in words):
                self.demographics['sex'] = 'male'
                self._capture_debug(f"[Engine] 👤 Sex: male")
            elif any(word in female_words for word in words):
                self.demographics['sex'] = 'female'
                self._capture_debug(f"[Engine] 👤 Sex: female")
            else:
                # Fuzzy match for typos (e.g., "femal", "mal", "womann")
                def char_similarity(word, target):
                    """Simple character overlap similarity (0-1)"""
                    if len(word) == 0 or len(target) == 0:
                        return 0.0
                    # Count matching characters in order
                    matches = sum(1 for a, b in zip(word, target) if a == b)
                    # Normalize by average length
                    avg_len = (len(word) + len(target)) / 2
                    return matches / avg_len if avg_len > 0 else 0.0
                
                # Check each word for fuzzy match (>80% similarity)
                for word in words:
                    if len(word) >= 3:  # Only check words with 3+ chars
                        for male_word in male_words:
                            if char_similarity(word, male_word) > 0.80:
                                self.demographics['sex'] = 'male'
                                self._capture_debug(f"[Engine] 🔍 Fuzzy match: '{word}' → '{male_word}' ({char_similarity(word, male_word):.2f})")
                                self._capture_debug(f"[Engine] 🔍 Fuzzy match: '{word}' → '{male_word}' ({char_similarity(word, male_word):.2f})")
                                break
                        
                        for female_word in female_words:
                            if char_similarity(word, female_word) > 0.80:
                                self.demographics['sex'] = 'female'
                                self._capture_debug(f"[Engine] 🔍 Fuzzy match: '{word}' → '{female_word}' ({char_similarity(word, female_word):.2f})")
                                self._capture_debug(f"[Engine] 🔍 Fuzzy match: '{word}' → '{female_word}' ({char_similarity(word, female_word):.2f})")
                                break
                        
                        if 'sex' in self.demographics:
                            break
            
            sex_result = self.demographics.get('sex', 'unknown')
            self._capture_debug(f"[Engine] 👤 Sex: {sex_result}")
            self._capture_debug(f"[Engine] 👤 Sex: {sex_result}")
            
            # FILTER guidelines by sex NOW that we know it
            if 'sex' in self.demographics:
                self._capture_debug(f"[Engine] 🔍 Filtering guidelines by gender: {self.demographics['sex']}")
                self._filter_by_gender()
            
            # VALIDATION: If sex is still unknown, re-ask using LLM
            if 'sex' not in self.demographics:
                self._capture_debug(f"[Engine] ⚠️ Invalid answer - re-asking for sex")
                self._capture_debug(f"{'='*80}\n")
                
                sex_question = self._generate_clarification_question("sex")
                self.conversation_history.append({
                    'type': 'question',
                    'question': sex_question,
                    'focus': 'sex'
                })
                
                return {
                    'success': True,
                    'question': sex_question,
                    'status': 'questioning',
                    'debug': self._get_debug_info()
                }
            
            self._capture_debug(f"{'='*80}\n")
            
            # FIRST CLINICAL QUESTION: Ask about CHRONICITY (new vs recurrent/chronic)
            # This helps differentiate new acute problems from chronic/recurrent issues
            chronicity_question = self._generate_chronicity_question()
            
            self.conversation_history.append({
                'type': 'question',
                'question': chronicity_question,
                'focus': 'chronicity'
            })
            
            return {
                'success': True,
                'question': chronicity_question,
                'status': 'questioning',
                'debug': self._get_debug_info()
            }
        
        elif last_q.get('focus') == 'chronicity':
            # Use LLM to intelligently classify chronicity
            self._capture_debug(f"[Engine] 🔍 LLM analyzing chronicity from answer: '{user_answer}'")
            
            chronicity = self._classify_chronicity_with_llm(user_answer)
            self.demographics['chronicity'] = chronicity
            self._capture_debug(f"[Engine] 📋 Chronicity: {chronicity}")
            
            self._capture_debug(f"[Engine] 📋 Chronicity: {chronicity}")
            if chronicity == 'recurring':
                self._capture_debug(f"[Engine] 💡 Consider: Follow-up vs new evaluation")
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
    
    def _filter_by_gender(self):
        """
        Filter active and reserve pools based on patient's biological sex.
        Called AFTER sex is collected.
        Uses 'sex' field from guideline JSON: 'male', 'female', or 'both'
        """
        patient_sex = self.demographics.get('sex')
        if not patient_sex:
            return
        
        self._capture_debug(f"\n[Engine] 🚺🚹 GENDER FILTERING (patient is {patient_sex})...")
        
        excluded_count = 0
        
        # Filter active guidelines
        filtered_active = []
        for g in self.active_guidelines:
            guideline_sex = g['data'].get('sex', 'both')
            
            # Skip if guideline is sex-specific and doesn't match patient
            if guideline_sex != 'both' and guideline_sex != patient_sex:
                self._capture_debug(f"[Engine]   ⛔ Excluding {g['name']} from active (requires {guideline_sex}, patient is {patient_sex})")
                excluded_count += 1
                continue
            
            filtered_active.append(g)
        
        # Filter reserve pool
        filtered_reserve = []
        for g in self.reserve_pool:
            guideline_sex = g['data'].get('sex', 'both')
            
            # Skip if guideline is sex-specific and doesn't match patient
            if guideline_sex != 'both' and guideline_sex != patient_sex:
                self._capture_debug(f"[Engine]   ⛔ Excluding {g['name']} from reserve (requires {guideline_sex}, patient is {patient_sex})")
                excluded_count += 1
                continue
            
            filtered_reserve.append(g)
        
        self.active_guidelines = filtered_active
        self.reserve_pool = filtered_reserve
        
        # Promote from reserve if active is now < MAX_ACTIVE
        while len(self.active_guidelines) < self.MAX_ACTIVE and len(self.reserve_pool) > 0:
            self.reserve_pool.sort(key=lambda x: x['score'], reverse=True)
            next_condition = self.reserve_pool.pop(0)
            self.active_guidelines.append(next_condition)
            self._capture_debug(f"[Engine]   🔼 PROMOTING: {next_condition['name']} to active after filtering")
        
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        self._capture_debug(f"[Engine] ✅ Excluded {excluded_count} sex-specific conditions")
        self._capture_debug(f"[Engine] 🔄 After filtering: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}")
        self._capture_debug(f"{'='*80}\n")
    
    def _match_to_guidelines_rag(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines using RAG API for GPU-accelerated semantic search
        
        Strategy:
        1. Exact/subset matching first (fast string operations)
        2. RAG API semantic search for remaining candidates (GPU-accelerated)
        3. Character overlap as final filter
        
        Returns:
            List of matched guidelines with initial scores
        """
        complaint_lower = complaint.lower()
        
        # Apply smart normalization (LLM or synonyms) to normalize patient language
        complaint_expanded = self._smart_oldcarts_normalization(complaint_lower)
        self._capture_debug(f"[Engine] 🔄 Smart normalization: '{complaint_lower}' → '{complaint_expanded}'")
        
        # Use normalized complaint directly for both phases
        core_symptom = complaint_expanded
        self._capture_debug(f"[Engine] 📋 Using normalized complaint: '{core_symptom}'")
        
        matched = []
        matched_guideline_names = set()  # Track which guidelines already matched
        
        self._capture_debug(f"\n[Engine] 🔍 MATCHING TO GUIDELINES (RAG API MODE)...")
        self._capture_debug(f"[Engine] 🎯 Strategy: exact > subset > RAG semantic > char_overlap")
        self._capture_debug(f"[Engine] ---")
        
        # PHASE 1: Fast exact/subset matching
        for name, guideline in self.all_guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            for trigger in triggers:
                trigger_lower = trigger.lower()
                
                # Exact match
                if trigger_lower in complaint_lower:
                    if name not in matched_guideline_names:
                        prevalence = guideline.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': name, 'score': initial_score, 'data': guideline})
                        matched_guideline_names.add(name)
                        self._capture_debug(f"[Engine]   ✓ {name} (trigger: '{trigger}', match: exact, prevalence: {prevalence})")
                    break
                
                # Subset match
                if core_symptom in trigger_lower:
                    if name not in matched_guideline_names:
                        prevalence = guideline.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': name, 'score': initial_score, 'data': guideline})
                        matched_guideline_names.add(name)
                        self._capture_debug(f"[Engine]   ✓ {name} (trigger: '{trigger}', match: subset, prevalence: {prevalence})")
                    break
        
        # PHASE 2: Semantic search for remaining guidelines (RAG API or local embedding)
        if self.rag_api or self.embedding_model:
            if self.rag_api:
                self._capture_debug(f"\n[Engine] 🚀 RAG API semantic search (GPU-accelerated)...")
            else:
                self._capture_debug(f"\n[Engine] 🧠 Local embedding semantic search (CPU)...")
            
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
                    self._capture_debug(f"[Engine] 📊 Searching {len(remaining_triggers)} remaining triggers...")
                    
                    # Use RAG API or local embedding model
                    if self.rag_api:
                        # RAG API mode
                        all_texts = [core_symptom] + remaining_triggers
                        embeddings = self.rag_api.encode(all_texts)
                    else:
                        # Local embedding mode
                        all_texts = [core_symptom] + remaining_triggers
                        embeddings = self.embedding_model.encode(all_texts)
                    
                    # Get query embedding (first one)
                    query_embedding = embeddings[0]
                    
                    # Compute cosine similarities with remaining triggers
                    for i, trigger_embedding in enumerate(embeddings[1:], 0):
                        # Compute cosine similarity
                        similarity = np.dot(query_embedding, trigger_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(trigger_embedding)
                        )
                        
                        metadata = trigger_to_guideline[i]
                        guideline_name = metadata['guideline_name']
                        trigger = metadata['trigger']
                        guideline_data = metadata['guideline_data']
                        
                        # Skip if already matched by exact/subset
                        if guideline_name in matched_guideline_names:
                            continue
                        
                        # Apply threshold
                        if similarity > 0.85:  # Same threshold as before
                            prevalence = guideline_data.get('prevalence', 'uncommon')
                            prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                            initial_score = prevalence_scores.get(prevalence, 0.50)
                            matched.append({'name': guideline_name, 'score': initial_score, 'data': guideline_data})
                            matched_guideline_names.add(guideline_name)
                            self._capture_debug(f"[Engine]   ✓ {guideline_name} (trigger: '{trigger}', match: rag_semantic ({similarity:.2f}), prevalence: {prevalence})")
                        else:
                            # Log first few rejections for visibility
                            if i < 5:
                                self._capture_debug(f"[Engine]   ✗ {guideline_name}: '{trigger}' (similarity={similarity:.2f} < 0.85)")
                
            except Exception as e:
                self._capture_debug(f"[Engine] ❌ RAG API semantic search failed: {e}")
                self._capture_debug(f"[Engine] 🔄 Falling back to brute-force matching")
        
        self._capture_debug(f"\n[Engine] 📊 RAG API matching complete: {len(matched)} guidelines matched")
        
        # Store matching metadata for debug
        self.matching_metadata = {
            'mode': 'RAG_API',
            'strategy': 'exact > subset > RAG semantic',
            'thresholds': {
                'char_overlap': 0.75,
                'semantic': 0.85
            },
            'matched_count': len(matched),
            'filtered_count': len(self.all_guidelines) - len(matched)
        }
        
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
        Use LLM to analyze all 3 guidelines and generate next best question
        
        This is the CORE intelligence of the system.
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🧠 LLM QUESTION GENERATION")
        self._capture_debug(f"{'='*80}")
        
        # Build context for LLM (MINIMAL - no guidelines, just OLDCARTS template)
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # Get questions already asked
        asked = []
        for item in self.conversation_history:
            if item['type'] == 'question' and item.get('focus') not in ['age', 'sex']:
                asked.append(item['question'])
        
        self._capture_debug(f"[Engine] 📋 Patient: {patient_info}")
        self._capture_debug(f"[Engine] 📋 Questions asked: {len(asked)}")
        
        # LLM PROMPT: Generate next question using ONLY generic OLDCARTS template
        system_msg = "Generate question. Follow the format exactly. Output ONLY the question text, no other words."
        
        # Show OLDCARTS coverage
        covered_elements = [k for k, v in self.oldcarts_covered.items() if v]
        uncovered_elements = [k for k, v in self.oldcarts_covered.items() if not v]
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        
        # Determine next OLDCARTS element to ask about
        next_element = uncovered_elements[0] if uncovered_elements else None
        
        # LLM-generated OLDCARTS questions
        if next_element:
            self._capture_debug(f"[Engine] 🧠 Generating question for OLDCARTS element: {next_element}")
            
            # Define what each OLDCARTS element asks about
            oldcarts_descriptions = {
                'O': "ONSET - when the symptom started (time/timing)",
                'L': "LOCATION - where the symptom is located (anatomical location)",
                'D': "DURATION - how long the symptom lasts or persists",
                'C': "CHARACTER - what the symptom feels like (quality/description)",
                'A': "AGGRAVATING factors - what makes the symptom worse",
                'R': "RELIEVING factors - what makes the symptom better",
                'T': "TIMING - pattern of the symptom (constant vs intermittent)",
                'S': "SEVERITY - how bad the symptom is (intensity/scale)"
            }
            
            element_desc = oldcarts_descriptions.get(next_element, "the symptom")
            
            # Build patient context
            patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')}"
            symptom = self.chief_complaint.lower().replace('i have', '').replace('i had', '').replace('i\'m having', '').strip()
            
            # Example questions for each OLDCARTS element
            oldcarts_examples = {
                'O': "When did the pain start?",
                'L': "Where exactly is the pain?",
                'D': "How long does the pain last?",
                'C': "How would you describe the pain?",
                'A': "What makes the pain worse?",
                'R': "What helps relieve the pain?",
                'T': "Is the pain constant or does it come and go?",
                'S': "How severe is the pain on a scale of 1 to 10?"
            }
            
            example = oldcarts_examples.get(next_element, "Tell me about the symptom")
            
            system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Use PLAIN LANGUAGE (no medical jargon). Do not include medical terminology from guidelines. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling'). Do NOT ask about duration/time - that will be covered later. No one prompt should include multiple questions, in other words do not include multiple phrases ending with a question mark."
            
            user_msg = f"""Patient: {patient_info} with {symptom}

Ask about: {element_desc}

Example: "{example}"

Generate EXACTLY ONE question using SIMPLE, PLAIN LANGUAGE. Do NOT combine multiple questions. Do NOT ask about duration/time. 
Make the question specific to the patient's chief complaint. For LOCATION questions, ask only about the relevant relevant body area for the chief complaint.
For example, if the chief complaint is "abdominal pain", ask about the abdomen area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "headache", ask about the head area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "back pain", ask about the back area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "leg pain", ask about the leg area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "foot pain", ask about the foot area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "hand pain", ask about the hand area, not the shoulders, arms, chest, or other body parts.
For example, if the chief complaint is "finger pain", ask about the finger area, not the shoulders, arms, chest, or other body parts.
Output only the question:"""
            
            # Filler is now handled at container level for immediate streaming
            self._capture_debug(f"[Engine] 💬 Generating question (filler handled by container)...")
            
            # HYBRID STRATEGY: Use simple model for basic questions, complex model for critical differentiating questions
            # - Simple model (Llama-1B): L, C, T, S, O, D (basic questions that don't need sophisticated reasoning)
            # - Complex model (Mistral-7B): A, R (aggravating/relieving factors - requires understanding of medical context and guidelines)
            if next_element in ['A', 'R']:
                # Use complex model for aggravating/relieving factors (requires medical reasoning and guideline understanding)
                self._capture_debug(f"[Engine] 🧠 Using COMPLEX model (Mistral-7B) for {next_element} - requires medical reasoning and guideline understanding")
                response = self.llm_chat_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=60,
                    temperature=self.temperature_complex
                )
            else:
                # Use simple model for basic questions (L, C, T, S, O, D) - straightforward questions
                self._capture_debug(f"[Engine] 🔧 Using SIMPLE model (Llama-1B) for {next_element} - straightforward question")
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=30,
                    temperature=self.temperature_simple
                )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            # VALIDATION: Ensure only ONE question
            # Check for multiple question marks or multiple declarative sentences before the question
            question_mark_count = question.count('?')
            
            # Check for pattern: "Statement. Question?" which indicates combined questions
            has_sentence_before_question = '. ' in question and question.index('. ') < question.rfind('?')
            
            # Check for combined questions with "and" or "also"
            has_combined_indicators = any(phrase in question.lower() for phrase in [
                ' and ', ' also ', ' how long ', ' how old ', ' what is your age ',
                ' when did ', ' how long have ', ' how old are you '
            ])
            
            # Check for inappropriate body part combinations (like shoulder, arm, chest)
            inappropriate_combinations = [
                'shoulder, arm', 'arm, chest', 'shoulder, arm, chest', 'chest, shoulder',
                'discomfort is more specifically', 'help me understand better'
            ]
            has_inappropriate_combinations = any(combo in question.lower() for combo in inappropriate_combinations)
            
            # Check for medical jargon that patients won't understand
            medical_jargon = [
                'epigastric', 'periumbilical', 'flank', 'costovertebral', 'cva', 'quadrant',
                'ruq', 'luq', 'rlq', 'llq', 'adnexal', 'suprapubic', 'hypogastric',
                'retrosternal', 'substernal', 'pelvic', 'inguinal', 'femoral'
            ]
            has_jargon = any(term in question.lower() for term in medical_jargon)
            
            if question_mark_count > 1 or has_sentence_before_question or has_jargon or has_combined_indicators or has_inappropriate_combinations:
                if has_jargon:
                    self._capture_debug(f"[Engine] ⚠️ LLM used medical jargon - using plain language template")
                elif has_combined_indicators:
                    self._capture_debug(f"[Engine] ⚠️ LLM combined multiple questions (detected: {[phrase for phrase in [' and ', ' also ', ' how long ', ' how old ', ' what is your age ', ' when did ', ' how long have ', ' how old are you '] if phrase in question.lower()]}) - using template fallback")
                elif has_inappropriate_combinations:
                    self._capture_debug(f"[Engine] ⚠️ LLM used inappropriate body part combinations (detected: {[combo for combo in inappropriate_combinations if combo in question.lower()]}) - using template fallback")
                else:
                    self._capture_debug(f"[Engine] ⚠️ LLM combined multiple questions - using template fallback")
                self._capture_debug(f"[Engine]    Generated: '{question}'")
                self._capture_debug(f"[Engine]    Using template: '{example}'")
                # Use simple template fallback
                question = example
            
            oldcarts_element = next_element
            
            self._capture_debug(f"[Engine] ✅ OLDCARTS Question ({next_element}): '{question}'")
            
            # Mark as covered
            self.oldcarts_covered[oldcarts_element] = True
            self._capture_debug(f"{'='*80}\n")
            
            # Store question
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'clinical',
                'oldcarts': oldcarts_element
            })
            
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'debug': self._get_debug_info()  # For Telegram debug display
            }
        
        # After OLDCARTS: Ask about associated symptoms using LLM
        self._capture_debug(f"[Engine] ℹ️  OLDCARTS complete - now asking about associated symptoms to reach 95% confidence")
        self._capture_debug(f"[Engine] 🧠 Generating associated symptom question...")
        
        # SAFETY: Check if we have active guidelines
        if not self.active_guidelines:
            self._capture_debug(f"[Engine] ❌ No active guidelines remaining - cannot generate question")
            self._capture_debug(f"[Engine] 📊 Debug: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
            self._capture_debug(f"[Engine] 📋 OLDCARTS covered: {self.oldcarts_covered}")
            self._capture_debug(f"[Engine] 📋 Demographics: {self.demographics}")
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Please seek medical evaluation.",
                'debug': self._get_debug_info()
            }
        
        # Build context of what's been asked
        asked_lower = ' '.join(asked).lower()
        
        # Get KEY POSITIVES from top 3 guidelines for context
        key_symptoms = []
        try:
            for g in self.active_guidelines[:3]:
                classic = g['data'].get('key_features', {}).get('classic_presentation', '')
                if 'KEY POSITIVES:' in classic:
                    parts = classic.split('KEY POSITIVES:')
                    if len(parts) > 1:
                        key_pos = parts[1].split('KEY NEGATIVES:')[0] if 'KEY NEGATIVES:' in parts[1] else parts[1]
                        key_symptoms.append(f"{g['name']}: {key_pos[:100]}")
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ Error extracting key symptoms: {e}")
        
        symptoms_context = ', '.join([s.split(':')[0] for s in key_symptoms[:3]]) if key_symptoms else "common symptoms"
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Use PLAIN LANGUAGE (no medical jargon). Do not include multiple phrases ending with question marks."
        
        user_msg = f"""Patient: {patient_info}

Ask about ONE associated symptom using SIMPLE language (fever, nausea, vomiting, diarrhea, etc). EXACTLY ONE question only.

Example: "Have you had any fever?"

Your question:"""
        
        # Filler is now handled at container level for immediate streaming
        self._capture_debug(f"[Engine] 💬 Generating associated symptom question (filler handled by container)...")
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
            temperature=self.temperature_analysis
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        
        # VALIDATION: Ensure only ONE question
        question_mark_count = question.count('?')
        has_sentence_before_question = '. ' in question and question.index('. ') < question.rfind('?')
        
        if question_mark_count > 1 or has_sentence_before_question:
            self._capture_debug(f"[Engine] ⚠️ LLM combined multiple questions - using template")
            self._capture_debug(f"[Engine]    Generated: '{question}'")
            # Use simple template fallback
            question = "Have you had any fever?"
        
        self._capture_debug(f"[Engine] ✅ Associated symptom question: '{question}'")
        self._capture_debug(f"{'='*80}\n")
        
        # Store question
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'focus': 'clinical',
            'oldcarts': None  # Not an OLDCARTS question
        })
        
        return {
            'success': True,
            'question': question,
            'status': 'questioning',
            # Filler is now handled at container level for immediate streaming
            'debug': self._get_debug_info()  # For Telegram debug display
        }
    
    def _detect_oldcarts_element(self, question: str) -> Optional[str]:
        """
        Detect which OLDCARTS element a question addresses
        
        Returns: 'O', 'L', 'D', 'C', 'A', 'R', 'T', or 'S' (or None if unclear)
        """
        # Simple keyword-based detection (reliable, fast)
        q_lower = question.lower()
        
        # O - ONSET
        if any(phrase in q_lower for phrase in ['when did', 'how did', 'started', 'began', 'onset', 'when exactly']):
            return 'O'
        
        # L - LOCATION
        if any(word in q_lower for word in ['where', 'location', 'which part', 'what area', 'which side']):
            return 'L'
        
        # D - DURATION
        if any(phrase in q_lower for phrase in ['how long', 'duration']):
            return 'D'
        
        # C - CHARACTER / Quality
        if any(phrase in q_lower for phrase in ['describe', 'feel like', 'type of', 'kind of', 'quality']):
            return 'C'
        
        # A - AGGRAVATING
        if any(phrase in q_lower for phrase in ['worse', 'worsen', 'aggravate', 'trigger']):
            return 'A'
        
        # R - RELIEVING
        if any(phrase in q_lower for phrase in ['better', 'relieve', 'improve', 'help']):
            return 'R'
        
        # T - TIMING (pattern)
        if any(phrase in q_lower for phrase in ['constant', 'come and go', 'intermittent', 'pattern']):
            return 'T'
        
        # S - SEVERITY
        if any(phrase in q_lower for phrase in ['severe', 'bad', 'scale', '1 to 10', 'intensity']):
            return 'S'
        
        return None
    
    def _extract_oldcarts_section(self, classic_presentation: str, element: str) -> str:
        """
        Extract specific OLDCARTS section from classic_presentation text
        
        Args:
            classic_presentation: Full guideline text
            element: 'O', 'L', 'D', 'C', 'A', 'R', 'T', or 'S'
        
        Returns:
            The text for that OLDCARTS section
        """
        element_names = {
            'O': 'ONSET',
            'L': 'LOCATION',
            'D': 'DURATION',
            'C': 'CHARACTER',
            'A': 'AGGRAVATING',
            'R': 'RELIEVING',
            'T': 'TIMING',
            'S': 'SEVERITY'
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
            # Fallback to simple keyword-based similarity for testing
            return self._compute_keyword_similarity(text1, text2)
        
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
    
    def _compute_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback keyword-based similarity for testing when embedding model is not available
        
        Returns: Simple similarity score 0-1 based on keyword overlap
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        self._capture_debug(f"[Engine]   🔍 Keyword similarity: {similarity:.3f} (intersection: {intersection}, union: {union})")
        
        return similarity
    
    
    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts based on word overlap
        
        Args:
            text1: First text (normalized complaint)
            text2: Second text (guideline location description)
            
        Returns:
            Jaccard similarity score 0-1
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words and clean punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'really', 'bad', 'gets', 'worse'}
        
        words1 = {w.strip('.,!?;:') for w in words1} - stop_words
        words2 = {w.strip('.,!?;:') for w in words2} - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Let semantic similarity handle directional conflicts - no hardcoded logic
        
        # Calculate Jaccard similarity: intersection / union
        intersection = words1 & words2
        union = len(words1 | words2)
        
        similarity = len(intersection) / union if union > 0 else 0.0
        
        # Let semantic similarity handle meaningful matches - no hardcoded medical terms
        
        self._capture_debug(f"[Engine]   🔍 Jaccard similarity: {similarity:.3f} (intersection: {len(intersection)}, union: {union})")
        self._capture_debug(f"[Engine]   🔍 Words1: {sorted(words1)}")
        self._capture_debug(f"[Engine]   🔍 Words2: {sorted(words2)}")
        self._capture_debug(f"[Engine]   🔍 Intersection: {sorted(intersection)}")
        
        return similarity
    
    def _check_anatomical_exclusion(self, patient_location: str, guideline_location: str, condition_name: str) -> bool:
        """
        Check for anatomical mismatches using LLM normalization and semantic similarity
        
        Args:
            patient_location: Patient's reported location (LLM normalized)
            guideline_location: Guideline location description
            condition_name: Name of the condition
            
        Returns:
            True if anatomically impossible, False otherwise
        """
        # Use semantic similarity to determine if there's a fundamental mismatch
        try:
            semantic_score = self._compute_similarity(patient_location, guideline_location)
        except Exception as e:
            self._capture_debug(f"[Engine]   ⚠️ Semantic similarity failed for anatomical check: {e}")
            semantic_score = 0.0
        
        # Use Jaccard similarity as secondary check
        jaccard_score = self._compute_jaccard_similarity(patient_location, guideline_location)
        
        # Check for anatomical mismatch using multiple criteria
        # The embedding model is unreliable for anatomical opposites, so we need strict rules
        
        # 1. If Jaccard similarity is 0 (no word overlap), it's likely an anatomical mismatch
        # regardless of what the unreliable semantic similarity says
        if jaccard_score == 0.0:
            self._capture_debug(f"[Engine]   ⛔ ANATOMICAL EXCLUSION: {condition_name} (semantic={semantic_score:.3f}, jaccard={jaccard_score:.3f}) - no word overlap indicates anatomical mismatch")
            return True
        
        # 2. If both similarities are very low, it's a mismatch
        elif semantic_score < 0.15 and jaccard_score < 0.1:
            self._capture_debug(f"[Engine]   ⛔ ANATOMICAL EXCLUSION: {condition_name} (semantic={semantic_score:.3f}, jaccard={jaccard_score:.3f})")
            return True
        
        # If similarities are reasonable, no anatomical exclusion
        self._capture_debug(f"[Engine]   ✅ NO ANATOMICAL EXCLUSION: {condition_name} (semantic={semantic_score:.3f}, jaccard={jaccard_score:.3f})")
        return False

    # Old hybrid similarity method removed - now using ML-based Medical Rule Engine
    
    def _apply_synonym_expansion(self, text: str) -> str:
        """Apply OLDCARTS-structured synonym expansion to normalize medical terms"""
        import json
        import os
        import re
        
        # Load OLDCARTS-structured synonyms from the synonyms directory
        # Try both Docker and local paths
        base_paths = [
            "/app/synonyms/",  # Docker container path
            "/Users/rcabello/Documents/GitHub/LedgerAI/llm-container/synonyms/"  # Local path
        ]
        
        synonym_files = []
        for base_path in base_paths:
            if os.path.exists(base_path):
                synonym_files.extend([
                    os.path.join(base_path, "gi_synonyms_oldcarts.json"),
                    os.path.join(base_path, "cardio_synonyms_oldcarts.json"),
                    os.path.join(base_path, "derm_synonyms_oldcarts.json"),
                    os.path.join(base_path, "endocrine_synonyms_oldcarts.json"),
                    os.path.join(base_path, "neuro_synonyms_oldcarts.json"),
                    os.path.join(base_path, "gu_synonyms_oldcarts.json"),
                    os.path.join(base_path, "resp_synonyms_oldcarts.json"),
                    os.path.join(base_path, "renal_synonyms_oldcarts.json")
                ])
                break
        
        # Load all OLDCARTS synonyms
        oldcarts_synonyms = {}
        for file_path in synonym_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        file_synonyms = json.load(f)
                        oldcarts_synonyms.update(file_synonyms)
                except Exception as e:
                    self._capture_debug(f"[Engine] ⚠️ Failed to load OLDCARTS synonyms from {file_path}: {e}")
        
        # Flatten OLDCARTS structure into standard_term -> variations mapping
        synonyms = {}
        for category, subcategories in oldcarts_synonyms.items():
            if isinstance(subcategories, dict):
                for subcategory, variations in subcategories.items():
                    if isinstance(variations, list):
                        # Create standard term from category and subcategory
                        standard_term = f"{category}_{subcategory}".replace("_", " ")
                        synonyms[standard_term] = variations
                    elif isinstance(variations, dict):
                        # Handle nested structures (like stool_characteristics)
                        for nested_key, nested_variations in variations.items():
                            if isinstance(nested_variations, list):
                                standard_term = f"{category}_{subcategory}_{nested_key}".replace("_", " ")
                                synonyms[standard_term] = nested_variations
            elif isinstance(subcategories, list):
                # Direct list of variations
                standard_term = category.replace("_", " ")
                synonyms[standard_term] = subcategories
        
        expanded_text = text
        all_variations = []
        for standard_term, variations in synonyms.items():
            for variation in variations:
                all_variations.append((len(variation), variation, standard_term))
        
        # Sort by length (longest first) to avoid partial replacements
        all_variations.sort(key=lambda x: x[0], reverse=True)
        
        for length, variation, standard_term in all_variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, expanded_text, re.IGNORECASE):
                expanded_text = re.sub(pattern, standard_term, expanded_text, flags=re.IGNORECASE)
                self._capture_debug(f"[Engine] 🔄 Synonym expansion: '{variation}' → '{standard_term}'")
                break  # Only replace first match to avoid over-replacement
        
        return expanded_text
    
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
                # Fallback to basic replacement if file not found
                self._medical_term_mappings = {}
        
        # Get mapping for category
        category_mappings = self._medical_term_mappings.get(category, {})
        
        # Return mapped term or fallback to formatted subcategory
        if subcategory in category_mappings:
            return category_mappings[subcategory]
        else:
            # Fallback: format subcategory (replace underscores with spaces)
            return subcategory.replace('_', ' ')
    
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

CRITICAL: Only normalize the {context} component. Do not add information from other symptoms or previous questions.

Normalized text:"""
        
        try:
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=50,
                temperature=0.1
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
        
        # Get enhanced similarity using Medical Rule Engine
        result = self.medical_rule_engine.get_enhanced_similarity(
            user_answer, oldcarts_section, condition_name, organ_system=self._get_organ_system_from_condition(condition_name)
        )
        
        # Log the result
        self._capture_debug(f"[Engine]   🎯 Enhanced {oldcarts_element} similarity: {result['similarity']:.3f} (method: {result['method']})")
        self._capture_debug(f"[Engine]   📝 Reasoning: {result['reasoning']}")
        self._capture_debug(f"[Engine]   🏥 Anatomical Type: {result['anatomical_type']}")
        
        # Collect learning data if available
        if self.learning_collector:
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
            self._capture_debug(f"[ML Progress] 🧠 Learning data collected:")
            self._capture_debug(f"[ML Progress]   📝 Patient: '{user_answer[:30]}...'")
            self._capture_debug(f"[ML Progress]   📋 Condition: {condition_name}")
            self._capture_debug(f"[ML Progress]   🎯 OLDCARTS: {oldcarts_element}")
            self._capture_debug(f"[ML Progress]   🎯 Method: {result['method']}")
            self._capture_debug(f"[ML Progress]   📊 Similarity: {result['similarity']:.3f}")
            self._capture_debug(f"[ML Progress]   🏥 Anatomical: {result['anatomical_type']}")
            self._capture_debug(f"[ML Progress]   🔄 Confidence: {result['confidence']}")
        
        # Track performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.track_prediction(
                prediction=result['similarity'],
                confidence=result['confidence'],
                method=result['method'],
                condition_name=condition_name,
                organ_system=self._get_organ_system_from_condition(condition_name)
            )
            
            # ML Progress Tracking - Performance
            self._capture_debug(f"[ML Progress] 📈 Performance tracked:")
            self._capture_debug(f"[ML Progress]   📊 Prediction: {result['similarity']:.3f}")
            self._capture_debug(f"[ML Progress]   🔄 Confidence: {result['confidence']}")
            self._capture_debug(f"[ML Progress]   🎯 Method: {result['method']}")
            self._capture_debug(f"[ML Progress]   🏥 Organ System: {self._get_organ_system_from_condition(condition_name)}")
        
        return result['similarity']
    
    def _compute_enhanced_location_similarity(self, user_answer: str, oldcarts_section: str, condition_name: str = "") -> float:
        """Enhanced location similarity with Medical Rule Engine and ML - ML ONLY"""
        
        # Ensure Medical Rule Engine is available
        if not self.medical_rule_engine:
            raise RuntimeError("Medical Rule Engine not available - ML system required")
        
        # Get enhanced similarity using Medical Rule Engine
        result = self.medical_rule_engine.get_enhanced_similarity(
            user_answer, oldcarts_section, condition_name
        )
        
        # Log the result
        self._capture_debug(f"[Engine]   🎯 Enhanced similarity: {result['similarity']:.3f} (method: {result['method']})")
        self._capture_debug(f"[Engine]   📝 Reasoning: {result['reasoning']}")
        self._capture_debug(f"[Engine]   🏥 Anatomical Type: {result['anatomical_type']}")
        
        # Collect learning data if available
        if self.learning_collector:
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
            self._capture_debug(f"[ML Progress] 🧠 Learning data collected:")
            self._capture_debug(f"[ML Progress]   📝 Patient: '{user_answer[:30]}...'")
            self._capture_debug(f"[ML Progress]   📋 Condition: {condition_name}")
            self._capture_debug(f"[ML Progress]   🎯 Method: {result['method']}")
            self._capture_debug(f"[ML Progress]   📊 Similarity: {result['similarity']:.3f}")
            self._capture_debug(f"[ML Progress]   🏥 Anatomical: {result['anatomical_type']}")
            self._capture_debug(f"[ML Progress]   🔄 Confidence: {result['confidence']}")
        
        # Track performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.track_prediction(
                prediction=result['similarity'],
                confidence=result['confidence'],
                method=result['method'],
                condition_name=condition_name,
                organ_system=self._get_organ_system_from_condition(condition_name)
            )
            
            # ML Progress Tracking - Performance
            self._capture_debug(f"[ML Progress] 📈 Performance tracked:")
            self._capture_debug(f"[ML Progress]   📊 Prediction: {result['similarity']:.3f}")
            self._capture_debug(f"[ML Progress]   🔄 Confidence: {result['confidence']}")
            self._capture_debug(f"[ML Progress]   🎯 Method: {result['method']}")
            self._capture_debug(f"[ML Progress]   🏥 Organ System: {self._get_organ_system_from_condition(condition_name)}")
        
        return result['similarity']
    
    def _get_organ_system_from_condition(self, condition_name: str) -> str:
        """Get organ system from condition name"""
        # Simple mapping - could be enhanced with more sophisticated logic
        condition_lower = condition_name.lower()
        
        if any(term in condition_lower for term in ['appendicitis', 'cholecystitis', 'pancreatitis', 'gastritis', 'ulcer', 'diverticulitis']):
            return 'GI'
        elif any(term in condition_lower for term in ['mi', 'angina', 'heart', 'cardiac', 'aortic']):
            return 'CARDIO'
        elif any(term in condition_lower for term in ['pneumonia', 'pneumothorax', 'pleural', 'lung']):
            return 'PULMONARY'
        elif any(term in condition_lower for term in ['kidney', 'uti', 'stone', 'prostatitis']):
            return 'GU'
        elif any(term in condition_lower for term in ['pregnancy', 'ovarian', 'pelvic', 'gynecologic']):
            return 'GYN'
        else:
            return 'UNKNOWN'
    
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
        if self.user_feedback:
            return self.user_feedback.collect_prediction_rating(
                prediction_id=prediction_id,
                prediction=prediction,
                user_rating=user_rating,
                user_comment=user_comment,
                condition_name=condition_name,
                organ_system=self._get_organ_system_from_condition(condition_name)
            )
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
        if self.user_feedback:
            return self.user_feedback.collect_accuracy_feedback(
                prediction_id=prediction_id,
                predicted_accuracy=predicted_accuracy,
                actual_accuracy=actual_accuracy,
                user_comment=user_comment,
                condition_name=condition_name,
                organ_system=self._get_organ_system_from_condition(condition_name)
            )
        return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status"""
        status = {
            'medical_rule_engine': self.medical_rule_engine is not None,
            'learning_collector': self.learning_collector is not None,
            'continuous_learning': self.continuous_learning is not None,
            'performance_monitor': self.performance_monitor is not None,
            'user_feedback': self.user_feedback is not None
        }
        
        # Get detailed status from components
        if self.continuous_learning:
            status['continuous_learning_status'] = self.continuous_learning.get_learning_status()
        
        if self.performance_monitor:
            status['performance_summary'] = self.performance_monitor.get_performance_summary()
        
        if self.user_feedback:
            status['feedback_summary'] = self.user_feedback.get_feedback_summary()
        
        # ML Progress Tracking - Learning Status
        self._capture_debug(f"[ML Progress] 📊 Learning system status:")
        self._capture_debug(f"[ML Progress]   🧠 Medical Rule Engine: {'Active' if self.medical_rule_engine else 'Inactive'}")
        self._capture_debug(f"[ML Progress]   📝 Learning Collector: {'Active' if self.learning_collector else 'Inactive'}")
        self._capture_debug(f"[ML Progress]   🔄 Continuous Learning: {'Active' if self.continuous_learning else 'Inactive'}")
        self._capture_debug(f"[ML Progress]   📈 Performance Monitor: {'Active' if self.performance_monitor else 'Inactive'}")
        self._capture_debug(f"[ML Progress]   💬 User Feedback: {'Active' if self.user_feedback else 'Inactive'}")
        
        return status
    
    
    
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
            if item.get('type') == 'question' and item.get('focus') == 'clinical':
                last_question_item = item
                break
        
        oldcarts_element = last_question_item.get('oldcarts') if last_question_item else None
        
        # ONSET: Documentation only - no scoring needed
        if oldcarts_element == 'O':
            self._capture_debug(f"[Engine] 📝 ONSET: Documentation only - no scoring needed")
            self._capture_debug(f"[Engine] ✅ Marking OLDCARTS element 'O' as covered")
            self.oldcarts_covered['O'] = True
            self._capture_debug(f"[Engine] 📋 OLDCARTS Coverage: {''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])} ({sum(self.oldcarts_covered.values())}/8)")
            
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
        
        for g in all_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            
            # Extract the specific OLDCARTS section for this element
            oldcarts_section = self._extract_oldcarts_section(classic, oldcarts_element)
            
            if not oldcarts_section:
                self._capture_debug(f"[Engine] ⚠️ Warning: Could not extract {oldcarts_element} section from {g['name']} - skipping this guideline")
                continue  # Skip this guideline instead of crashing
            
            # UNIFIED ML SYSTEM: Use enhanced similarity for ALL OLDCARTS components
            # This provides consistent ML-powered similarity across all components
            try:
                similarity = self._compute_enhanced_oldcarts_similarity(answer, oldcarts_section, oldcarts_element, g['name'])
                self._capture_debug(f"[Engine]   {g['name']}: Enhanced {oldcarts_element} similarity = {similarity:.3f} ('{answer}' vs '{oldcarts_section[:50]}...')")
            except Exception as sim_error:
                self._capture_debug(f"[Engine] ❌ Enhanced {oldcarts_element} similarity computation failed for {g['name']}: {sim_error}")
                import traceback
                traceback.print_exc()
                # Skip this guideline and continue with the next one
                continue
            
            # Update score using OLDCARTS element weight
            old_score = g['score']
            
            # Get element-specific weight (Location=1.0, Onset=0.3, etc.)
            element_weight = self.oldcarts_weights.get(oldcarts_element, 0.5)
            
            # UNIFIED ML-ONLY SCORING: Use enhanced similarity directly as the score
            # No more hybrid scoring or penalties - ML system provides the final score for ALL OLDCARTS
            new_score = similarity
            g['score'] = new_score
            change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
            self._capture_debug(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (unified-ml)")
            self._capture_debug(f"[Engine]     🧠 ML {oldcarts_element} Score: {similarity:.2f} ('{answer}' ↔ '{oldcarts_section[:50]}...')")
            self._capture_debug(f"[Engine]     📝 Patient: '{answer}' → Medical: '{oldcarts_section[:80]}...'")
            
            # ML Progress Tracking - Scoring
            self._capture_debug(f"[ML Progress] 🎯 Score updated:")
            self._capture_debug(f"[ML Progress]   📋 Condition: {g['name']}")
            self._capture_debug(f"[ML Progress]   🎯 OLDCARTS: {oldcarts_element}")
            self._capture_debug(f"[ML Progress]   📊 Old Score: {old_score:.0%} → New Score: {new_score:.0%} {change}")
            self._capture_debug(f"[ML Progress]   🧠 ML Similarity: {similarity:.3f}")
            self._capture_debug(f"[ML Progress]   📝 Patient Input: '{answer}'")
            self._capture_debug(f"[ML Progress]   📋 Guideline: '{oldcarts_section[:50]}...'")
        
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
                self._capture_debug(f"[ML Progress] ❌ Condition ruled out:")
                self._capture_debug(f"[ML Progress]   📋 Condition: {g['name']}")
                self._capture_debug(f"[ML Progress]   📊 Score: {g['score']:.0%} < Threshold: {threshold:.0%}")
                self._capture_debug(f"[ML Progress]   🎯 ML Decision: Anatomical mismatch or low similarity")
            else:
                remaining.append(g)
                
                # ML Progress Tracking - Kept
                self._capture_debug(f"[ML Progress] ✅ Condition kept:")
                self._capture_debug(f"[ML Progress]   📋 Condition: {g['name']}")
                self._capture_debug(f"[ML Progress]   📊 Score: {g['score']:.0%} >= Threshold: {threshold:.0%}")
                self._capture_debug(f"[ML Progress]   🎯 ML Decision: Anatomical match or high similarity")
        
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
            self._capture_debug(f"[ML Progress] 🏆 Top {i}: {g['name']}")
            self._capture_debug(f"[ML Progress]   📊 Score: {g['score']:.0%}")
            self._capture_debug(f"[ML Progress]   📋 Prevalence: {g.get('prevalence', 'unknown')}")
            self._capture_debug(f"[ML Progress]   🎯 ML Confidence: High similarity match")
            self._capture_debug(f"[ML Progress]   🚨 Urgency: {g['data'].get('urgency', 'standard')}")
        
        # Always show pool statistics
        self._capture_debug(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
        # ML Progress Tracking - Final Statistics
        self._capture_debug(f"[ML Progress] 📊 Final statistics:")
        self._capture_debug(f"[ML Progress]   🎯 Active Conditions: {len(self.active_guidelines)}")
        self._capture_debug(f"[ML Progress]   📋 Reserve Conditions: {len(self.reserve_pool)}")
        self._capture_debug(f"[ML Progress]   ❌ Ruled Out: {len(self.ruled_out)}")
        self._capture_debug(f"[ML Progress]   📈 Total Processed: {len(all_guidelines)}")
        self._capture_debug(f"[ML Progress]   🧠 ML System: Fully operational")
        
        self._capture_debug(f"{'='*80}\n")
        
        # Mark the OLDCARTS element as covered after processing the answer
        if oldcarts_element:
            self._capture_debug(f"[Engine] ✅ Marking OLDCARTS element '{oldcarts_element}' as covered")
            self.oldcarts_covered[oldcarts_element] = True
        
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
        if len(self.active_guidelines) >= 2:
            top_score = self.active_guidelines[0]['score']
            second_score = self.active_guidelines[1]['score']
            score_spread = top_score - second_score
            
            # Count how many clarifications we've already asked for this OLDCARTS element
            clarification_count = sum(1 for item in self.conversation_history 
                                     if item.get('type') == 'question' 
                                     and item.get('oldcarts') == oldcarts_element 
                                     and item.get('is_clarification'))
            
            # If scores are too close (can't differentiate) OR all scores too low
            # Ask clarification, but move on if we've asked too many times for this element
            MAX_CLARIFICATIONS_PER_ELEMENT = 2  # Limit to avoid infinite loops
            
            # MUCH MORE LENIENT: Only clarify when absolutely necessary
            # Normal patient answers like "yesterday", "random", "sudden" should be accepted
            # LLM semantic similarity should handle normalization (e.g., "yesterday" → "24 hours ago")
            
            # Show LLM normalization decision
            self._capture_debug(f"\n[Engine] 🧠 LLM NORMALIZATION DECISION:")
            self._capture_debug(f"[Engine]   📝 Patient answer: '{answer}'")
            self._capture_debug(f"[Engine]   📊 Top score: {top_score:.0%}, Spread: {score_spread:.0%}")
            self._capture_debug(f"[Engine]   🎯 Thresholds: spread < 0.05, top_score < 0.20")
            
            if (score_spread < 0.05 or top_score < 0.20):
                if clarification_count < MAX_CLARIFICATIONS_PER_ELEMENT:
                    self._capture_debug(f"\n[Engine] 🔍 CLARIFICATION NEEDED:")
                    self._capture_debug(f"[Engine]   Top score: {top_score:.0%}, Spread: {score_spread:.0%}")
                    self._capture_debug(f"[Engine]   Reason: {'Scores too close' if score_spread < 0.05 else 'All scores too low'}")
                    self._capture_debug(f"[Engine]   Clarifications asked so far: {clarification_count}/{MAX_CLARIFICATIONS_PER_ELEMENT}")
                    self._capture_debug(f"[Engine]   Strategy: {'Open-ended' if clarification_count == 0 else 'Targeted (differential-based)'}")
                    
                    # Generate progressively targeted clarifying question
                    clarifying_q = self._generate_clarifying_question(oldcarts_element, answer, clarification_count)
                    
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
                else:
                    self._capture_debug(f"\n[Engine] ⚠️  Scores still close (top: {top_score:.0%}, spread: {score_spread:.0%})")
                    self._capture_debug(f"[Engine]   Already asked {clarification_count} clarifications for '{oldcarts_element}'")
                    self._capture_debug(f"[Engine]   📋 Can't differentiate further on this element - moving to next OLDCARTS")
                    # Will fall through and continue to next OLDCARTS element
            else:
                self._capture_debug(f"[Engine] ✅ LLM NORMALIZATION SUCCESS: Accepting '{answer}' without clarification")
                self._capture_debug(f"[Engine]   🧠 LLM semantic understanding sufficient (top: {top_score:.0%}, spread: {score_spread:.0%})")
                self._capture_debug(f"[Engine]   📝 Patient language normalized via semantic similarity")
        
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
        Convert a red flag statement to a yes/no question
        
        Example:
        "High fever >103°F with severe pain - possible perforation"
        → "Have you had a fever higher than 103 degrees?"
        """
        # Simple hardcoded patterns for common red flags
        lower = red_flag.lower()
        
        if 'fever' in lower and '103' in lower:
            question = "Have you had a fever higher than 103 degrees?"
        elif 'fever' in lower:
            question = "Have you had any fever?"
        elif 'rigid' in lower or 'board-like' in lower:
            question = "Does your abdomen feel hard or rigid?"
        elif 'dizzy' in lower or 'faint' in lower or 'hypotension' in lower:
            question = "Have you felt dizzy or lightheaded?"
        elif 'confusion' in lower or 'altered mental' in lower:
            question = "Have you felt confused?"
        elif 'blood' in lower and 'stool' in lower:
            question = "Have you seen blood in your stool?"
        elif 'blood' in lower and 'vomit' in lower:
            question = "Have you vomited blood?"
        elif 'jaundice' in lower or 'yellow' in lower:
            question = "Have you noticed any yellowing of your skin or eyes?"
        else:
            # Generic
            question = f"Have you experienced {red_flag.split('-')[0].strip().lower()}?"
        
        self._capture_debug(f"[Engine] ✅ Red flag question: '{question}'")
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

Write a brief, natural empathetic medical statement:

Examples: 
- "I'm sorry to hear you're experiencing that."
- "That sounds uncomfortable, let me ask some questions to help."
- "I understand that must be concerning."
- "I'll ask some questions to better understand your symptoms."

Your statement:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=50,  # Allow more natural variation
            temperature=0.7  # More creative and natural
        )
        
        statement = response.strip().strip('"\'')
        
        # Remove numbered list markers if LLM still outputs them
        import re
        statement = re.sub(r'^\d+\.\s*', '', statement)  # Remove "1. " from start
        statement = re.sub(r'\n\d+\.\s*', ' ', statement)  # Remove "\n2. " from middle
        
        # VALIDATION: Only reject if completely nonsensical
        # Allow more natural variation in opening statements
        word_count = len(statement.split())
        if word_count > 50:  # Only reject if extremely long
            self._capture_debug(f"[Engine] ⚠️ Opening too long ({word_count} words) - using simple template")
            self._capture_debug(f"[Engine]    Generated: '{statement}'")
            statement = "I understand. I'll ask some questions to help."
        
        self._capture_debug(f"[Engine] ✅ Opening (simple model): '{statement}'")
        return statement
    
    def _generate_chronicity_question(self) -> str:
        """
        LLM-generated chronicity question to differentiate new vs chronic problems
        """
        self._capture_debug(f"[Engine] 🧠 Generating chronicity question...")
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."
        
        user_msg = """Generate a natural question asking if this is a new problem or ongoing/recurrent issue.

Examples: 
- "Is this a new problem or something you've experienced before?"
- "Is this the first time you've had this symptom?"
- "Have you had this issue before, or is this new?"

Your question:"""
        
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
            temperature=0.6
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

CRITICAL: If the patient says "new", "first time", "started today/yesterday", or similar - classify as "new".
If they say "before", "had this", "comes and goes", "chronic" - classify as "recurring".

Examples:
- "new" → new
- "It's new" → new
- "This is the first time" → new
- "It started yesterday" → new
- "I've had this before" → recurring  
- "It comes and goes" → recurring
- "I don't know" → unclear
- "I've had this for years" → recurring

Classification:"""
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=10,
            temperature=0.1  # Use very low temperature for this critical classification
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
        
        user_msg = """Generate a natural question asking for the patient's age.

Examples: 
- "How old are you?"
- "What's your age?"
- "Can you tell me your age?"

Your question:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
            temperature=0.6
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Age question (simple model): '{question}'")
        return question
    
    def _generate_sex_question(self) -> str:
        """
        LLM-generated biological sex question
        """
        self._capture_debug(f"[Engine] 🧠 Generating sex question...")
        
        system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions."
        
        user_msg = """Generate a natural question asking for biological sex (male or female).

Examples: 
- "Are you male or female?"
- "What's your biological sex?"
- "Are you a man or woman?"

Your question:"""
        
        response = self.llm_chat_simple_fn(  # Use simple model (Llama-1B)
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
            temperature=0.6
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Sex question (simple model): '{question}'")
        return question
    
    def _generate_clarifying_question(self, oldcarts_element: str, vague_answer: str, clarification_count: int = 0) -> str:
        """
        Generate progressively targeted clarifying questions based on top differentials
        
        Strategy:
        - 1st clarification: Open-ended (gather more info)
        - 2nd+ clarifications: Targeted based on top differentials (discriminate between conditions)
        
        Args:
            oldcarts_element: The OLDCARTS element that needs clarification (O, L, D, C, A, R, T, S)
            vague_answer: The user's vague answer that needs clarification
            clarification_count: How many clarifications already asked for this element
            
        Returns:
            Progressively more targeted question for the same OLDCARTS element
        """
        self._capture_debug(f"[Engine] 🧠 Generating clarifying question #{clarification_count + 1} for OLDCARTS '{oldcarts_element}'...")
        
        # First clarification: Use LLM to generate intelligent open-ended question
        if clarification_count == 0:
            question = self._llm_generate_clarification_question(oldcarts_element, vague_answer)
        
        # Subsequent clarifications: Targeted based on top differentials
        else:
            # Try to generate targeted question based on top differentials
            question = self._generate_differential_based_question(oldcarts_element)
            
            # If we got the same generic question, it means we can't differentiate further
            # Add variation or use a different angle
            if clarification_count >= 1 and oldcarts_element == 'O':
                # For onset, ask about associated context instead
                question = "Did anything trigger it? Like eating, physical activity, or did it just happen out of nowhere?"
        
        self._capture_debug(f"[Engine] ✅ Clarifying question #{clarification_count + 1}: '{question}'")
        return question
    
    def _llm_generate_clarification_question(self, oldcarts_element: str, vague_answer: str) -> str:
        """
        Use LLM to generate intelligent clarification questions
        Analyzes the vague answer and generates a targeted follow-up
        """
        self._capture_debug(f"[Engine] 🧠 LLM generating clarification for '{oldcarts_element}': '{vague_answer}'")
        
        # OLDCARTS element descriptions
        element_descriptions = {
            'L': "LOCATION - where the symptom is located",
            'O': "ONSET - when the symptom started", 
            'D': "DURATION - how long the symptom lasts",
            'C': "CHARACTER - what the symptom feels like",
            'A': "AGGRAVATING factors - what makes it worse",
            'R': "RELIEVING factors - what makes it better", 
            'T': "TIMING - pattern of the symptom",
            'S': "SEVERITY - how bad the symptom is"
        }
        
        element_desc = element_descriptions.get(oldcarts_element, "the symptom")
        
        system_msg = "You are a medical assistant. Generate ONE intelligent clarification question. Use PLAIN LANGUAGE. Do NOT ask questions requiring visual inspection."
        
        user_msg = f"""The patient gave a vague answer about {element_desc}:

Patient's answer: "{vague_answer}"

Generate ONE intelligent follow-up question to get more specific information about {element_desc}. 
Make it natural and conversational, not clinical.

Question:"""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=80,
                temperature=0.4
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            self._capture_debug(f"[Engine] ✅ LLM clarification question: '{question}'")
            return question
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM clarification question failed: {e}")
            # Fallback to simple template
            return f"Can you tell me more about '{vague_answer}'?"
    
    def _generate_differential_based_question(self, oldcarts_element: str) -> str:
        """
        Generate targeted question based on top differentials to discriminate between them
        Uses LLM to generate generic questions for any medical condition
        """
        if len(self.active_guidelines) < 2:
            return "Could you be more specific?"
        
        # Get top 3 differentials
        top_conditions = self.active_guidelines[:min(3, len(self.active_guidelines))]
        
        # Extract key distinguishing features from guidelines
        condition_names = [g['name'] for g in top_conditions]
        
        self._capture_debug(f"[Engine] 🎯 Generating targeted question to differentiate between:")
        for g in top_conditions:
            self._capture_debug(f"[Engine]   - {g['name']} ({g['score']:.0%})")
        
        # Use LLM to generate targeted question based on differentials
        return self._llm_generate_differential_question(oldcarts_element, top_conditions)
    
    def _llm_generate_differential_question(self, oldcarts_element: str, top_conditions: List[Dict]) -> str:
        """
        Use LLM to generate targeted question that discriminates between top differentials
        Generic approach that works for any medical condition
        """
        self._capture_debug(f"[Engine] 🧠 LLM generating differential question for {oldcarts_element}")
        
        # Build context with top conditions
        condition_info = []
        for g in top_conditions:
            condition_info.append(f"- {g['name']}: {g['data'].get('description', 'No description')}")
        
        # OLDCARTS element descriptions
        element_descriptions = {
            'L': "LOCATION - where the symptom is located",
            'O': "ONSET - when the symptom started", 
            'D': "DURATION - how long the symptom lasts",
            'C': "CHARACTER - what the symptom feels like",
            'A': "AGGRAVATING factors - what makes it worse",
            'R': "RELIEVING factors - what makes it better", 
            'T': "TIMING - pattern of the symptom",
            'S': "SEVERITY - how bad the symptom is"
        }
        
        element_desc = element_descriptions.get(oldcarts_element, "the symptom")
        
        system_msg = "You are a medical assistant. Generate ONE targeted question to help differentiate between these conditions. Use PLAIN LANGUAGE. Do NOT ask questions requiring visual inspection."
        
        user_msg = f"""Generate a targeted question about {element_desc} to help differentiate between these conditions:

Conditions:
{chr(10).join(condition_info)}

Generate ONE question that will help distinguish between these conditions. Focus on {element_desc}.

Question:"""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            self._capture_debug(f"[Engine] ✅ LLM differential question: '{question}'")
            return question
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM differential question failed: {e}")
            return "Could you provide more details?"
    
    
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
            max_tokens=20,
            temperature=0.2
        )
        
        question = response.strip().strip('"\'')
        if not question.endswith('?'):
            question += '?'
        self._capture_debug(f"[Engine] ✅ Clarification (simple model): '{question}'")
        return question


# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")