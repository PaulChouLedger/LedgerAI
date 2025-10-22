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
7. Rule out <30% → Promote from reserve (prioritize COMMON conditions)
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
        
        self._capture_debug(f"[Engine] 🧠 Using {'dual models (simple + complex)' if llm_chat_simple_fn else 'single model'}")
        
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
        self.oldcarts_weights = {
            'L': 1.0,  # Location - full weight (definitive - left vs right rules out immediately)
            'C': 0.9,  # Character - high weight (sharp vs dull matters)
            'A': 0.8,  # Aggravating - high weight (post-meal, exertion diagnostic)
            'R': 0.8,  # Relieving - high weight (what helps is important)
            'S': 0.7,  # Severity - moderate weight (subjective but useful)
            'D': 0.5,  # Duration - supportive only (varies widely)
            'O': 0.3,  # Onset - weakest (sudden vs gradual overlaps for many conditions)
            'T': 0.4,  # Timing - weak to moderate (constant vs intermittent)
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
        
        # Thresholds
        self.RULE_OUT_THRESHOLD = 0.30  # Below 30% → rule out and replace
        self.MAX_ACTIVE = 5  # Keep 5 active differentials
        self.MAX_CLARIFICATIONS = 2  # Max times to ask for clarification before moving on
    
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
            
            system_msg = "You are a medical assistant. Output ONLY ONE question. Use PLAIN LANGUAGE (no medical jargon). Never combine multiple questions. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."
            
            user_msg = f"""Patient: {patient_info} with {symptom}

Ask about: {element_desc}

Example: "{example}"

Generate EXACTLY ONE similar question using SIMPLE, PLAIN LANGUAGE that anyone can understand (open-ended, NOT yes/no). Do NOT combine multiple questions:"""
            
            # Filler is now handled at container level for immediate streaming
            self._capture_debug(f"[Engine] 💬 Generating question (filler handled by container)...")
            
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=40,
                temperature=0.2
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            # VALIDATION: Ensure only ONE question
            # Check for multiple question marks or multiple declarative sentences before the question
            question_mark_count = question.count('?')
            
            # Check for pattern: "Statement. Question?" which indicates combined questions
            has_sentence_before_question = '. ' in question and question.index('. ') < question.rfind('?')
            
            # Check for medical jargon that patients won't understand
            medical_jargon = [
                'epigastric', 'periumbilical', 'flank', 'costovertebral', 'cva', 'quadrant',
                'ruq', 'luq', 'rlq', 'llq', 'adnexal', 'suprapubic', 'hypogastric',
                'retrosternal', 'substernal', 'pelvic', 'inguinal', 'femoral'
            ]
            has_jargon = any(term in question.lower() for term in medical_jargon)
            
            if question_mark_count > 1 or has_sentence_before_question or has_jargon:
                if has_jargon:
                    self._capture_debug(f"[Engine] ⚠️ LLM used medical jargon - using plain language template")
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
        
        system_msg = "You are a medical assistant. Output ONLY ONE question. Use PLAIN LANGUAGE (no medical jargon). Never combine multiple questions."
        
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
            temperature=0.3
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
        
        # Generate embeddings
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
    
    def _has_anatomical_mismatch(self, complaint: str, guideline_location: str) -> bool:
        """
        Check for anatomical location mismatches (e.g., RUQ vs RLQ)
        
        Args:
            complaint: Normalized patient complaint
            guideline_location: Guideline location description
            
        Returns:
            True if there's an anatomical mismatch that should rule out the condition
        """
        complaint_lower = complaint.lower()
        guideline_lower = guideline_location.lower()
        
        # Define anatomical location mappings
        anatomical_locations = {
            'ruq': ['right upper quadrant', 'ruq', 'upper right'],
            'luq': ['left upper quadrant', 'luq', 'upper left'],
            'rlq': ['right lower quadrant', 'rlq', 'lower right'],
            'llq': ['left lower quadrant', 'llq', 'lower left'],
            'epigastric': ['epigastric', 'upper mid', 'upper middle'],
            'periumbilical': ['periumbilical', 'around belly button', 'around navel'],
            'flank': ['flank', 'side']
        }
        
        # Extract locations from complaint and guideline
        complaint_locations = set()
        guideline_locations = set()
        
        for location_type, variations in anatomical_locations.items():
            for variation in variations:
                if variation in complaint_lower:
                    complaint_locations.add(location_type)
                if variation in guideline_lower:
                    guideline_locations.add(location_type)
        
        # Check for direct conflicts
        if complaint_locations and guideline_locations:
            # If complaint mentions specific quadrant and guideline mentions different quadrant
            if complaint_locations.isdisjoint(guideline_locations):
                # Check for specific quadrant conflicts
                quadrant_conflicts = [
                    ('ruq', 'rlq'), ('ruq', 'llq'), ('ruq', 'luq'),
                    ('luq', 'rlq'), ('luq', 'llq'), ('luq', 'ruq'),
                    ('rlq', 'ruq'), ('rlq', 'llq'), ('rlq', 'luq'),
                    ('llq', 'ruq'), ('llq', 'rlq'), ('llq', 'luq')
                ]
                
                for loc1, loc2 in quadrant_conflicts:
                    if loc1 in complaint_locations and loc2 in guideline_locations:
                        self._capture_debug(f"[Engine]   🚫 ANATOMICAL CONFLICT: {loc1.upper()} vs {loc2.upper()}")
                        return True
        
        return False
    
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
        
        # Check for directional conflicts (left vs right)
        # Only rule out if guideline has OPPOSITE direction and NO mention of user's direction
        has_left_user = 'left' in words1
        has_right_user = 'right' in words1
        has_left_guideline = 'left' in words2
        has_right_guideline = 'right' in words2
        
        # Rule out ONLY if clear directional conflict (opposite direction with no mention of user's side)
        if has_left_user and has_right_guideline and not has_left_guideline:
            # User says "left" but guideline only mentions "right" (never "left")
            self._capture_debug(f"[Engine]   ⛔ Directional conflict: user='left', guideline has 'right' only (no 'left' mentioned)")
            return 0.0
        
        if has_right_user and has_left_guideline and not has_right_guideline:
            # User says "right" but guideline only mentions "left" (never "right")
            self._capture_debug(f"[Engine]   ⛔ Directional conflict: user='right', guideline has 'left' only (no 'right' mentioned)")
            return 0.0
        
        # If guideline mentions BOTH left and right (e.g., "RUQ, may radiate to left")
        # → Do NOT rule out, let Jaccard score naturally handle it
        
        # Calculate Jaccard similarity: intersection / union
        intersection = words1 & words2
        union = len(words1 | words2)
        
        similarity = len(intersection) / union if union > 0 else 0.0
        
        # Check if single-word match is meaningful
        # Directional words (left, right, upper, lower) are meaningful even alone
        directional_terms = {'left', 'right', 'upper', 'lower', 'epigastric', 'periumbilical', 'ruq', 'luq', 'rlq', 'llq', 'flank', 'groin', 'chest', 'substernal'}
        meaningful_matches = intersection & directional_terms
        
        if len(intersection) == 1 and not meaningful_matches:
            # Only 1 weak word match (e.g., just "pain", "abdomen") - penalize moderately
            similarity = similarity * 0.5  # Reduce by 50%
            self._capture_debug(f"[Engine]   ⚠️  Weak single-word match (reduced by 50%)")
        # If match is directional (left, right, etc.), keep full score - it's meaningful!
        
        self._capture_debug(f"[Engine]   🔍 Jaccard similarity: {similarity:.3f} (intersection: {len(intersection)}, union: {union})")
        self._capture_debug(f"[Engine]   🔍 Words1: {sorted(words1)}")
        self._capture_debug(f"[Engine]   🔍 Words2: {sorted(words2)}")
        self._capture_debug(f"[Engine]   🔍 Intersection: {sorted(intersection)}")
        
        return similarity
    
    def _compute_hybrid_similarity(self, complaint: str, guideline_location: str) -> Dict[str, float]:
        """
        Compute hybrid similarity using both Jaccard and semantic similarity
        
        Args:
            complaint: Normalized patient complaint
            guideline_location: Guideline location description
            
        Returns:
            Dictionary with similarity scores and confidence metrics
        """
        # Primary: Jaccard similarity (fast, reliable for normalized terms)
        jaccard_score = self._compute_jaccard_similarity(complaint, guideline_location)
        
        # Secondary: Semantic similarity (handles edge cases)
        semantic_score = 0.0
        if self.embedding_model:
            try:
                semantic_score = self._compute_similarity(complaint, guideline_location)
            except Exception as e:
                self._capture_debug(f"[Engine]   ⚠️ Semantic similarity failed: {e}")
                semantic_score = 0.0
        
        # Determine final score and confidence
        final_score = jaccard_score
        confidence = "high"
        method_used = "jaccard"
        
        # Hybrid logic with emphasis on Jaccard (70% weight)
        if jaccard_score > self.hybrid_config['jaccard_threshold']:
            # High Jaccard confidence - use it as primary
            final_score = jaccard_score
            confidence = "high"
            method_used = "jaccard"
            
        elif jaccard_score == 0.0:
            # Zero Jaccard = clear keyword mismatch (e.g., "left side" vs "right side")
            # Give very low weight to semantic to emphasize the mismatch
            final_score = semantic_score * 0.2  # Only 20% of semantic (very low)
            confidence = "low"
            method_used = "jaccard_mismatch"
            
        elif semantic_score > jaccard_score + self.hybrid_config['semantic_boost_threshold']:
            # Semantic significantly better - use hybrid with Jaccard emphasis
            final_score = (0.7 * jaccard_score) + (0.3 * semantic_score)
            confidence = "medium"
            method_used = "hybrid_semantic_boost"
            
        elif semantic_score > self.hybrid_config['semantic_threshold']:
            # Semantic above threshold - use hybrid
            final_score = (0.7 * jaccard_score) + (0.3 * semantic_score)
            confidence = "medium"
            method_used = "hybrid"
            
        else:
            # Use Jaccard as primary
            final_score = jaccard_score
            confidence = "low" if jaccard_score < 0.2 else "medium"
            method_used = "jaccard"
        
        # Check if both methods agree (high confidence)
        if abs(jaccard_score - semantic_score) < self.hybrid_config['confidence_threshold']:
            confidence = "high"
        
        return {
            'final_score': final_score,
            'jaccard_score': jaccard_score,
            'semantic_score': semantic_score,
            'confidence': confidence,
            'method_used': method_used
        }
    
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
        """Convert OLDCARTS categories to clean, semantic-friendly medical terms"""
        
        if category == "location":
            if subcategory == "abdominal_pain":
                return "abdominal pain"
            elif subcategory == "ruq_pain":
                return "right upper quadrant"
            elif subcategory == "luq_pain":
                return "left upper quadrant"
            elif subcategory == "rlq_pain":
                return "right lower quadrant"
            elif subcategory == "llq_pain":
                return "left lower quadrant"
            elif subcategory == "epigastric_pain":
                return "epigastric"
            elif subcategory == "periumbilical_pain":
                return "periumbilical"
            elif subcategory == "flank_pain":
                return "flank"
            else:
                return subcategory.replace('_', ' ')
        
        elif category == "character":
            if subcategory == "sharp":
                return "sharp"
            elif subcategory == "dull":
                return "dull"
            elif subcategory == "cramping":
                return "cramping"
            elif subcategory == "burning":
                return "burning"
            elif subcategory == "stabbing":
                return "sharp"
            else:
                return subcategory.replace('_', ' ')
        
        elif category == "onset":
            if subcategory == "sudden":
                return "sudden onset"
            elif subcategory == "gradual":
                return "gradual onset"
            else:
                return f"{subcategory.replace('_', ' ')} onset"
        
        elif category == "aggravating_factors":
            if subcategory == "eating":
                return "postprandial"
            elif subcategory == "movement":
                return "worse with movement"
            else:
                return f"worsened by {subcategory.replace('_', ' ')}"
        
        elif category == "relieving_factors":
            return f"relieved by {subcategory.replace('_', ' ')}"
        
        elif category == "associated_symptoms":
            if subcategory == "nausea":
                return "nausea"
            elif subcategory == "vomiting":
                return "vomiting"
            elif subcategory == "fever":
                return "fever"
            else:
                return subcategory.replace('_', ' ')
        
        elif category == "severity":
            if subcategory == "severe":
                return "severe"
            elif subcategory == "mild":
                return "mild"
            elif subcategory == "moderate":
                return "moderate"
            else:
                return subcategory.replace('_', ' ')
        
        elif category == "timing":
            if subcategory == "constant":
                return "constant"
            elif subcategory == "intermittent":
                return "intermittent"
            elif subcategory == "colicky":
                return "colicky"
            else:
                return subcategory.replace('_', ' ')
        
        elif category == "radiation":
            if subcategory == "back":
                return "radiates to back"
            elif subcategory == "shoulder":
                return "radiates to shoulder"
            else:
                return f"radiates to {subcategory.replace('_', ' ')}"
        
        else:
            # Default: create clean term from subcategory
            if nested_key:
                return f"{subcategory.replace('_', ' ')} {nested_key.replace('_', ' ')}"
            else:
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
        
        user_msg = f"""Normalize this patient response into standard medical terminology:

Patient: "{text}"
Context: {context}

Examples:
- "left side" → "left side of abdomen" 
- "hurts bad" → "severe pain"
- "came on fast" → "sudden onset"
- "feels like stabbing" → "sharp pain"
- "under my ribs" → "upper abdomen"

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
    
    def _compute_enhanced_location_similarity(self, user_answer: str, oldcarts_section: str) -> float:
        """Hybrid similarity (Jaccard + Semantic) for location matching with smart normalization"""
        # Apply smart normalization (LLM or synonyms) focusing on location
        user_answer_expanded = self._smart_oldcarts_normalization(user_answer.lower(), target_category="location")
        self._capture_debug(f"[Engine] 🔄 Smart location normalization: '{user_answer}' → '{user_answer_expanded}'")
        
        # Use hybrid similarity (Jaccard + Semantic) with emphasis on Jaccard
        hybrid_result = self._compute_hybrid_similarity(user_answer_expanded, oldcarts_section)
        
        # DEBUG: Show hybrid similarity calculation
        self._capture_debug(f"[Engine]   🎯 Hybrid similarity: Jaccard={hybrid_result['jaccard_score']:.3f}, Semantic={hybrid_result['semantic_score']:.3f}")
        self._capture_debug(f"[Engine]   📊 Final score: {hybrid_result['final_score']:.3f} (method: {hybrid_result['method_used']}, confidence: {hybrid_result['confidence']})")
        self._capture_debug(f"[Engine]   📝 '{user_answer_expanded}' vs '{oldcarts_section[:80]}...'")
        
        return hybrid_result['final_score']
    
    
    
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
            
            # KEYWORD FILTER: For location questions, skip opposite-sided conditions
            # This is faster and more accurate than semantic similarity for directional terms
            if oldcarts_element == 'L':
                answer_lower = answer.lower()
                section_upper = oldcarts_section.upper()
                
                # Use enhanced location similarity with multi-stage filtering
                # This will handle "left lower belly pain towards my pelvis" vs "LEFT LOWER QUADRANT (LLQ)"
                try:
                    similarity = self._compute_enhanced_location_similarity(answer, oldcarts_section)
                    self._capture_debug(f"[Engine]   {g['name']}: Enhanced location similarity = {similarity:.3f} ('{answer}' vs '{oldcarts_section[:50]}...')")
                except Exception as sim_error:
                    self._capture_debug(f"[Engine] ❌ Enhanced similarity computation failed for {g['name']}: {sim_error}")
                    import traceback
                    traceback.print_exc()
                    # Skip this guideline and continue with the next one
                    continue
            else:
                # Compute semantic similarity normally for non-location questions
                try:
                    similarity = self._compute_similarity(answer, oldcarts_section)
                except Exception as sim_error:
                    self._capture_debug(f"[Engine] ❌ Similarity computation failed for {g['name']}: {sim_error}")
                    import traceback
                    traceback.print_exc()
                    # Skip this guideline and continue with the next one
                    continue
            
            # Update score using OLDCARTS element weight
            old_score = g['score']
            
            # Get element-specific weight (Location=1.0, Onset=0.3, etc.)
            element_weight = self.oldcarts_weights.get(oldcarts_element, 0.5)
            
            if similarity == 0.0:
                # Hard mismatch (e.g., left vs right) - apply full weight of element
                # Location mismatch (weight=1.0) rules out completely
                # Onset mismatch (weight=0.3) has minimal impact
                new_score = old_score * (1 - element_weight)
                g['score'] = new_score
                change = "❌"
                self._capture_debug(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (keyword mismatch, weight={element_weight:.1f})")
            else:
                # Normal weighted average using element-specific weight
                # Location (weight=1.0): new_score = similarity (replaces old score)
                # Onset (weight=0.3): new_score = 70% old + 30% similarity (mostly preserves old score)
                new_score = (old_score * (1 - element_weight)) + (similarity * element_weight)
                g['score'] = new_score
                change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
                self._capture_debug(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (weight={element_weight:.1f})")
                self._capture_debug(f"[Engine]     🧠 LLM Semantic Match: {similarity:.2f} ('{answer}' ↔ '{oldcarts_section[:50]}...')")
                self._capture_debug(f"[Engine]     📝 Patient: '{answer}' → Medical: '{oldcarts_section[:80]}...'")
        
        # DYNAMIC RE-RANKING: Sort ALL guidelines by updated scores
        # This ensures conditions like Diverticulitis (LLQ) jump to top when "left side" is mentioned
        self._capture_debug(f"\n[Engine] 🔄 RE-RANKING all guidelines by updated scores...")
        
        # Rule out any with score < threshold
        ruled_out_this_round = []
        remaining = []
        for g in all_guidelines:
            if g['score'] < self.RULE_OUT_THRESHOLD:
                self._capture_debug(f"[Engine] ❌ RULING OUT: {g['name']} (score {g['score']:.0%} < {self.RULE_OUT_THRESHOLD:.0%})")
                self.ruled_out.append(g)
                ruled_out_this_round.append(g)
            else:
                remaining.append(g)
        
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
        
        # Always show pool statistics
        self._capture_debug(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
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
        
        system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."
        
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
        
        system_msg = "You are a medical assistant. Classify if this is a NEW problem or RECURRING/CHRONIC problem. Respond with only: 'new', 'recurring', or 'unclear'"
        
        user_msg = f"""Classify this patient response about whether their problem is new or recurring:

Patient response: "{answer}"

Examples:
- "It's new" → new
- "I've had this before" → recurring  
- "This is the first time" → new
- "It comes and goes" → recurring
- "I don't know" → unclear
- "It started yesterday" → new
- "I've had this for years" → recurring

Classification:"""
        
        response = self.llm_chat_simple_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=10,
            temperature=0.1
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
        
        system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else."
        
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
        
        system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else."
        
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