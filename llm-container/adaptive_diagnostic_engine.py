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

# Try to import FAISS (CPU version for guideline matching)
try:
    import faiss
    FAISS_AVAILABLE = True
    print("[Engine] ✅ FAISS-CPU available - will use for fast semantic matching")
except ImportError:
    FAISS_AVAILABLE = False
    print("[Engine] ⚠️ FAISS not available - using brute-force matching (slower for 500+ guidelines)")


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
        
        print(f"[Engine] 🧠 Using {'dual models (simple + complex)' if llm_chat_simple_fn else 'single model'}")
        
        # FAISS index for fast semantic matching
        self.faiss_index = None
        self.trigger_metadata = []  # Maps FAISS index positions to (guideline_name, trigger) tuples
        self.use_faiss = False  # Will be enabled after successful index build
        self.validate_faiss = os.getenv("VALIDATE_FAISS", "false").lower() == "true"  # Compare FAISS vs brute-force
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # Build FAISS index after guidelines are loaded
        if FAISS_AVAILABLE and self.embedding_model:
            self._build_faiss_index()
        
        # Current assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files from subdirectories"""
        print(f"\n{'='*80}")
        print(f"[Engine] 📚 LOADING MEDICAL GUIDELINES")
        print(f"{'='*80}")
        print(f"[Engine] 📁 Source directory: {self.guidelines_dir}")
        
        if not self.guidelines_dir.exists():
            print(f"[Engine] ❌ Directory not found: {self.guidelines_dir}")
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
                    
                    print(f"[Engine]   ✓ {organ_system}/{name}")
            except Exception as e:
                print(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
        
        print(f"\n[Engine] ✅ LOADED {len(self.all_guidelines)} GUIDELINES:")
        for system, conditions in sorted(organ_systems.items()):
            print(f"[Engine]    📋 {system}: {len(conditions)} conditions")
        print(f"{'='*80}\n")
    
    def _build_faiss_index(self):
        """
        Build FAISS index from all guideline triggers for fast semantic search
        
        This is a ONE-TIME startup cost that enables fast querying:
        - Brute-force: O(n) comparisons per query
        - FAISS: O(log n) comparisons per query
        """
        try:
            print(f"\n{'='*80}")
            print(f"[Engine] 🏗️  BUILDING FAISS INDEX FOR FAST SEMANTIC MATCHING")
            print(f"{'='*80}")
            
            import time
            start_time = time.time()
            
            # Extract all triggers from all guidelines
            all_triggers = []
            trigger_to_guideline = []
            
            for guideline_name, guideline in self.all_guidelines.items():
                triggers = guideline.get('chief_complaint_triggers', [])
                for trigger in triggers:
                    all_triggers.append(trigger)
                    trigger_to_guideline.append({
                        'guideline_name': guideline_name,
                        'trigger': trigger,
                        'guideline_data': guideline
                    })
            
            print(f"[Engine] 📋 Extracted {len(all_triggers)} triggers from {len(self.all_guidelines)} guidelines")
            
            if len(all_triggers) == 0:
                print("[Engine] ⚠️ No triggers found - FAISS index not built")
                return
            
            # Generate embeddings for all triggers
            print(f"[Engine] 🧠 Generating embeddings for {len(all_triggers)} triggers...")
            embeddings = self.embedding_model.encode(all_triggers)
            
            # Convert to numpy array with float32 (FAISS requirement)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_np.shape[1]
            
            print(f"[Engine] 📐 Embedding dimension: {dimension}")
            print(f"[Engine] 📊 Total vectors: {len(embeddings_np)}")
            
            # Create FAISS index (CPU version - L2 distance, then convert to cosine)
            # Use IndexFlatIP (Inner Product) for cosine similarity
            # First normalize vectors, then IP = cosine similarity
            faiss.normalize_L2(embeddings_np)
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings_np)
            
            # Store metadata
            self.trigger_metadata = trigger_to_guideline
            
            build_time = time.time() - start_time
            
            print(f"[Engine] ✅ FAISS index built successfully!")
            print(f"[Engine]    ⏱️  Build time: {build_time:.2f}s")
            print(f"[Engine]    📊 Index size: {self.faiss_index.ntotal} vectors")
            print(f"[Engine]    🎯 Ready for fast semantic search")
            print(f"{'='*80}\n")
            
            # Enable FAISS mode
            self.use_faiss = True
            print(f"[Engine] 🚀 FAISS mode ENABLED (brute-force available as fallback)")
            
        except Exception as e:
            print(f"[Engine] ❌ FAISS index build failed: {e}")
            print(f"[Engine] 🔄 Falling back to brute-force matching")
            import traceback
            traceback.print_exc()
            self.faiss_index = None
            self.trigger_metadata = []
            self.use_faiss = False
    
    def _is_valid_chief_complaint(self, complaint: str) -> bool:
        """
        Validate that chief complaint is coherent (not garbled transcription)
        
        Checks:
        1. Contains common medical/symptom words
        2. Not complete gibberish (e.g., "domino pain", "word salad")
        3. Has reasonable length
        
        Returns:
            True if valid, False if nonsensical
        """
        complaint_lower = complaint.lower().strip()
        
        # Too short or too long
        if len(complaint_lower) < 5 or len(complaint_lower) > 200:
            return False
        
        # Common medical/symptom words that should appear in valid complaints
        medical_keywords = [
            'pain', 'hurt', 'ache', 'sore', 'burn', 'itch', 'bleed', 'swell',
            'fever', 'cough', 'nausea', 'vomit', 'dizzy', 'tired', 'weak', 'short of breath',
            'chest', 'abdomen', 'stomach', 'head', 'back', 'leg', 'arm', 'throat',
            'sick', 'ill', 'problem', 'issue', 'concern', 'discomfort', 'symptom'
        ]
        
        # Common filler words that indicate sentence structure
        common_words = [
            'i', 'have', 'had', 'my', 'the', 'a', 'an', 'is', 'am', 'feeling',
            'experiencing', 'been', 'getting'
        ]
        
        # Extract words from complaint
        words = complaint_lower.split()
        
        # Check if contains at least one medical keyword
        has_medical_term = any(keyword in complaint_lower for keyword in medical_keywords)
        
        if not has_medical_term:
            # No medical terms - likely garbled
            print(f"[Engine] 🔍 Validation: No medical keywords found in '{complaint}'")
            return False
        
        # Check for obvious gibberish patterns
        gibberish_patterns = [
            'domino pain',  # Known bad transcription
            'diamond pain',
            'domain pain',
            'dummy pain'
        ]
        
        for pattern in gibberish_patterns:
            if pattern in complaint_lower:
                print(f"[Engine] 🔍 Validation: Detected gibberish pattern '{pattern}'")
                return False
        
        # If we got here, it looks valid
        print(f"[Engine] ✅ Validation: Chief complaint appears valid")
        return True
    
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
            'last_answer_scores': getattr(self, '_last_answer_scores', None)  # Set during scoring
        }
        
        # Add matching algorithm info if available
        if hasattr(self, 'matching_metadata') and self.matching_metadata:
            debug_info['matching'] = self.matching_metadata
        
        return debug_info
    
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
        print(f"\n{'='*80}")
        print(f"[Engine] 🚀 NEW ASSESSMENT")
        print(f"{'='*80}")
        print(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # VALIDATION: Check if chief complaint is coherent
        if not self._is_valid_chief_complaint(chief_complaint):
            print(f"[Engine] ❌ Invalid chief complaint (garbled/nonsensical)")
            print(f"{'='*80}\n")
            return {
                'success': False,
                'message': "I didn't quite understand that. Can you describe your symptoms again?"
            }
        
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        
        # STEP 1: Get filler immediately (for instant user feedback)
        filler = get_filler('opening', use_audio=True)
        print(f"[Engine] 💬 Filler (for immediate response): [{filler['id']}] '{filler['text']}'")
        if 'audio_path' in filler:
            print(f"[Engine]    🎵 Audio: {filler['audio_path']}")
        
        # STEP 2: Run RAG and Llama-1B in PARALLEL (major speedup!)
        import threading
        import concurrent.futures
        
        rag_result = [None]
        opening_result = [None]
        age_result = [None]
        error_result = [None]
        
        def run_rag():
            """Match to guidelines (FAISS or brute-force with fallback + optional validation)"""
            try:
                # VALIDATION MODE: Compare FAISS vs brute-force (set VALIDATE_FAISS=true)
                if self.validate_faiss and self.use_faiss:
                    print(f"[Engine] 🧪 VALIDATION MODE: Comparing FAISS vs brute-force...")
                    
                    import time
                    
                    # Run FAISS
                    start_faiss = time.time()
                    faiss_matches = self._match_to_guidelines_faiss(chief_complaint)
                    faiss_time = time.time() - start_faiss
                    
                    # Run brute-force
                    start_brute = time.time()
                    brute_matches = self._match_to_guidelines(chief_complaint)
                    brute_time = time.time() - start_brute
                    
                    # Compare results
                    faiss_names = set([m['name'] for m in faiss_matches])
                    brute_names = set([m['name'] for m in brute_matches])
                    
                    print(f"\n[Engine] 📊 VALIDATION RESULTS:")
                    print(f"[Engine]    FAISS: {len(faiss_matches)} matches in {faiss_time:.2f}s")
                    print(f"[Engine]    Brute: {len(brute_matches)} matches in {brute_time:.2f}s")
                    print(f"[Engine]    Speedup: {brute_time/faiss_time:.1f}x faster")
                    
                    if faiss_names == brute_names:
                        print(f"[Engine]    ✅ MATCH: Both methods returned identical results")
                    else:
                        only_faiss = faiss_names - brute_names
                        only_brute = brute_names - faiss_names
                        if only_faiss:
                            print(f"[Engine]    ⚠️ Only in FAISS: {only_faiss}")
                        if only_brute:
                            print(f"[Engine]    ⚠️ Only in brute-force: {only_brute}")
                    
                    # Use FAISS results
                    rag_result[0] = faiss_matches
                
                # NORMAL MODE: Use FAISS with fallback
                elif self.use_faiss:
                    print(f"[Engine] 🚀 Using FAISS mode for matching")
                    import time
                    start_time = time.time()
                    try:
                        rag_result[0] = self._match_to_guidelines_faiss(chief_complaint)
                        elapsed = time.time() - start_time
                        if hasattr(self, 'matching_metadata'):
                            self.matching_metadata['timing'] = elapsed
                    except Exception as faiss_error:
                        print(f"[Engine] ❌ FAISS matching failed: {faiss_error}")
                        print(f"[Engine] 🔄 Falling back to brute-force matching")
                        self.use_faiss = False  # Disable FAISS for future queries
                        start_time = time.time()
                        rag_result[0] = self._match_to_guidelines(chief_complaint)
                        elapsed = time.time() - start_time
                        if hasattr(self, 'matching_metadata'):
                            self.matching_metadata['timing'] = elapsed
                else:
                    print(f"[Engine] 🐢 Using brute-force mode for matching")
                    import time
                    start_time = time.time()
                    rag_result[0] = self._match_to_guidelines(chief_complaint)
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
        print(f"[Engine] ⚡ Starting parallel execution (RAG + Llama-1B)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rag_future = executor.submit(run_rag)
            llm_future = executor.submit(run_simple_llm)
            
            # Wait for both to complete
            concurrent.futures.wait([rag_future, llm_future])
        
        # Check for errors
        if error_result[0]:
            print(f"[Engine] ❌ Parallel execution error: {error_result[0]}")
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
        
        print(f"\n[Engine] 📋 ACTIVE DIFFERENTIALS (Top {len(self.active_guidelines)}):")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            prevalence = g['data'].get('prevalence', 'uncommon')
            print(f"[Engine]   {i}. {g['name']} ({prevalence}, {g['score']:.0%}) {urgency_emoji}")
        
        if self.reserve_pool:
            print(f"\n[Engine] 💾 RESERVE POOL ({len(self.reserve_pool)} conditions, prioritized by prevalence):")
            for i, g in enumerate(self.reserve_pool[:5], 1):  # Show first 5
                prevalence = g['data'].get('prevalence', 'uncommon')
                urgency = g['data'].get('urgency', 'routine')
                print(f"[Engine]   {i}. {g['name']} ({prevalence}, {urgency}, {g['score']:.0%})")
            if len(self.reserve_pool) > 5:
                print(f"[Engine]   ... and {len(self.reserve_pool) - 5} more")
        
        print(f"\n[Engine] 🔄 Initial pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        print(f"{'='*80}\n")
        
        # STEP 3: Use results from parallel execution
        opening_statement = opening_result[0]
        age_question = age_result[0]
        
        print(f"[Engine] ⚡ Parallel execution complete!")
        print(f"[Engine]    Opening: '{opening_statement}'")
        print(f"[Engine]    Age Q: '{age_question}'")
        
        # Combine them with proper spacing
        combined_message = f"{opening_statement} {age_question}"
        
        self.conversation_history.append({
            'type': 'question',
            'question': combined_message,
            'focus': 'age'
        })
        
        return {
            'success': True,
            'question': combined_message,
            'status': 'questioning',
            'filler': filler,  # Play/send this immediately while waiting for main response
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
                print(f"[Engine] 🔄 No active guidelines - treating as NEW chief complaint")
                return self.start_assessment(user_answer)
        
        print(f"\n{'='*80}")
        print(f"[Engine] 💬 PROCESSING ANSWER")
        print(f"{'='*80}")
        print(f"[Engine] User: '{user_answer}'")
        
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
                print(f"[Engine] ⚠️ Unclear red flag answer: '{user_answer}' - re-asking")
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
                    'status': 'red_flag_screening'
                }
            
            if is_yes:
                red_flag_text = last_q.get('red_flag_text', 'Warning sign')
                self.red_flags_present.append(red_flag_text)
                print(f"[Engine] ⚠️  RED FLAG PRESENT: {red_flag_text}")
            else:
                print(f"[Engine] ✓ Red flag not present")
            
            # Move to next red flag
            self.red_flag_index += 1
            
            # Continue screening (or finalize if done)
            return self._screen_red_flags(self.active_guidelines[0])
        
        # Handle demographics
        if last_q.get('focus') == 'age':
            # Extract age using LLM
            print(f"[Engine] 🔍 Extracting age from answer: '{user_answer}'")
            
            # Use regex to extract numbers (simple and reliable)
            import re
            numbers = re.findall(r'\b(\d{1,3})\b', user_answer)
            
            if numbers:
                # Take first number found
                age_num = int(numbers[0])
                if 1 <= age_num <= 120:  # Sanity check
                    self.demographics['age'] = age_num
                    print(f"[Engine] 👤 Age: {age_num}")
                else:
                    print(f"[Engine] 👤 Age: Invalid ({age_num} out of range 1-120)")
            else:
                print(f"[Engine] 👤 Age: No number found in answer")
            
            # VALIDATION: If no age found, re-ask using LLM
            if 'age' not in self.demographics:
                print(f"[Engine] ⚠️ Invalid answer - re-asking for age")
                print(f"{'='*80}\n")
                
                age_question = self._generate_clarification_question("age")
                self.conversation_history.append({
                    'type': 'question',
                    'question': age_question,
                    'focus': 'age'
                })
                
                return {
                    'success': True,
                    'question': age_question,
                    'status': 'questioning'
                }
            
            # Ask sex using LLM
            sex_question = self._generate_sex_question()
            self.conversation_history.append({
                'type': 'question',
                'question': sex_question,
                'focus': 'sex'
            })
            print(f"{'='*80}\n")
            
            return {
                'success': True,
                'question': sex_question,
                'status': 'questioning'
            }
        
        elif last_q.get('focus') == 'sex':
            # Extract sex - use keyword matching with fuzzy tolerance for typos
            print(f"[Engine] 🔍 Extracting sex from answer: '{user_answer}'")
            
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
            elif any(word in female_words for word in words):
                self.demographics['sex'] = 'female'
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
                                print(f"[Engine] 🔍 Fuzzy match: '{word}' → '{male_word}' ({char_similarity(word, male_word):.2f})")
                                break
                        
                        for female_word in female_words:
                            if char_similarity(word, female_word) > 0.80:
                                self.demographics['sex'] = 'female'
                                print(f"[Engine] 🔍 Fuzzy match: '{word}' → '{female_word}' ({char_similarity(word, female_word):.2f})")
                                break
                        
                        if 'sex' in self.demographics:
                            break
            
            print(f"[Engine] 👤 Sex: {self.demographics.get('sex', 'unknown')}")
            
            # FILTER guidelines by sex NOW that we know it
            if 'sex' in self.demographics:
                self._filter_by_gender()
            
            # VALIDATION: If sex is still unknown, re-ask using LLM
            if 'sex' not in self.demographics:
                print(f"[Engine] ⚠️ Invalid answer - re-asking for sex")
                print(f"{'='*80}\n")
                
                sex_question = self._generate_clarification_question("sex")
                self.conversation_history.append({
                    'type': 'question',
                    'question': sex_question,
                    'focus': 'sex'
                })
                
                return {
                    'success': True,
                    'question': sex_question,
                    'status': 'questioning'
                }
            
            print(f"{'='*80}\n")
            
            # FIRST CLINICAL QUESTION: Always ask about CHRONICITY (when started)
            # This is the most important differentiator (acute vs chronic)
            timing_question = "When did the pain start?"
            
            # Mark ONSET as covered
            self.oldcarts_covered['O'] = True
            
            print(f"[Engine] 💬 First question: ONSET (O in OLDCARTS)")
            
            self.conversation_history.append({
                'type': 'question',
                'question': timing_question,
                'focus': 'clinical',
                'oldcarts': 'O'
            })
            
            return {
                'success': True,
                'question': timing_question,
                'status': 'questioning'
            }
        
        else:
            # Clinical question - find last question first
            last_q_item = None
            for item in reversed(self.conversation_history):
                if item.get('type') == 'question':
                    last_q_item = item
                    break
            
            # VALIDATE answer first
            if not self._is_acceptable_clinical_answer(user_answer):
                print(f"[Engine] ⚠️ Answer too vague or unclear - asking for clarification")
                
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
                    'status': 'questioning'
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
        
        print(f"[Engine] 🔍 Validating answer with LLM...")
        print(f"[Engine]   Q: '{last_question}'")
        print(f"[Engine]   A: '{answer}'")
        
        # Simple validation: reject pure filler words or fragments
        # Semantic scoring will determine if answer is specific enough
        
        # Reject pure filler words or meaningless fragments
        pure_filler = ['um', 'uh', 'oh', 'hmm', 'ah', 'er']
        fragments = ['on the', 'my', 'the', 'it', 'there', 'here', 'i', 'a', 'an']
        
        answer_stripped = answer.strip().lower()
        
        if answer_stripped in pure_filler or answer_stripped in fragments:
            print(f"[Engine] 📊 Validation: REJECT ❌ (pure filler or fragment)")
            return False
        
        # Accept any substantive answer - semantic scoring will handle specificity
        print(f"[Engine] 📊 Validation: ACCEPT ✅ (substantive answer)")
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
        
        print(f"\n[Engine] 🚺🚹 GENDER FILTERING (patient is {patient_sex})...")
        
        excluded_count = 0
        
        # Filter active guidelines
        filtered_active = []
        for g in self.active_guidelines:
            guideline_sex = g['data'].get('sex', 'both')
            
            # Skip if guideline is sex-specific and doesn't match patient
            if guideline_sex != 'both' and guideline_sex != patient_sex:
                print(f"[Engine]   ⛔ Excluding {g['name']} from active (requires {guideline_sex}, patient is {patient_sex})")
                excluded_count += 1
                continue
            
            filtered_active.append(g)
        
        # Filter reserve pool
        filtered_reserve = []
        for g in self.reserve_pool:
            guideline_sex = g['data'].get('sex', 'both')
            
            # Skip if guideline is sex-specific and doesn't match patient
            if guideline_sex != 'both' and guideline_sex != patient_sex:
                print(f"[Engine]   ⛔ Excluding {g['name']} from reserve (requires {guideline_sex}, patient is {patient_sex})")
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
            print(f"[Engine]   🔼 PROMOTING: {next_condition['name']} to active after filtering")
        
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"[Engine] ✅ Excluded {excluded_count} sex-specific conditions")
        print(f"[Engine] 🔄 After filtering: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}")
        print(f"{'='*80}\n")
    
    def _match_to_guidelines_faiss(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines using FAISS for fast semantic search
        
        Strategy:
        1. Exact/subset matching first (fast string operations)
        2. FAISS semantic search for remaining candidates (single query)
        3. Character overlap as final filter
        
        Returns:
            List of matched guidelines with initial scores
        """
        complaint_lower = complaint.lower()
        
        # Extract core symptom
        filler_words = ['i', 'have', 'my', 'the', 'a', 'an', 'is', 'am', 'feel', 'feeling']
        symptom_words = [w for w in complaint_lower.split() if w not in filler_words]
        core_symptom = ' '.join(symptom_words)
        
        matched = []
        matched_guideline_names = set()  # Track which guidelines already matched
        
        print(f"\n[Engine] 🔍 MATCHING TO GUIDELINES (FAISS MODE)...")
        print(f"[Engine] 📋 Core symptom extracted: '{core_symptom}'")
        print(f"[Engine] 🎯 Strategy: exact > subset > FAISS semantic > char_overlap")
        print(f"[Engine] ---")
        
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
                        print(f"[Engine]   ✓ {name} (trigger: '{trigger}', match: exact, prevalence: {prevalence})")
                    break
                
                # Subset match
                if core_symptom in trigger_lower:
                    if name not in matched_guideline_names:
                        prevalence = guideline.get('prevalence', 'uncommon')
                        prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                        initial_score = prevalence_scores.get(prevalence, 0.50)
                        matched.append({'name': name, 'score': initial_score, 'data': guideline})
                        matched_guideline_names.add(name)
                        print(f"[Engine]   ✓ {name} (trigger: '{trigger}', match: subset, prevalence: {prevalence})")
                    break
        
        # PHASE 2: FAISS semantic search for remaining guidelines
        if self.faiss_index and self.faiss_index.ntotal > 0:
            print(f"\n[Engine] 🚀 FAISS semantic search (checking {self.faiss_index.ntotal} triggers)...")
            
            # Generate query embedding and normalize
            query_embedding = self.embedding_model.encode([core_symptom])
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding_np)
            
            # Search for top K most similar triggers
            k = min(100, self.faiss_index.ntotal)  # Get top 100 candidates
            distances, indices = self.faiss_index.search(query_embedding_np, k)
            
            print(f"[Engine] 📊 FAISS returned {len(indices[0])} candidates")
            
            # Process FAISS results (distances are cosine similarities after normalization)
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS padding for not enough results
                    break
                
                metadata = self.trigger_metadata[idx]
                guideline_name = metadata['guideline_name']
                trigger = metadata['trigger']
                guideline_data = metadata['guideline_data']
                
                # Skip if already matched by exact/subset
                if guideline_name in matched_guideline_names:
                    continue
                
                similarity = float(distance)  # Cosine similarity (0-1)
                
                # Apply threshold
                if similarity > 0.75:  # Same threshold as brute-force
                    prevalence = guideline_data.get('prevalence', 'uncommon')
                    prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                    initial_score = prevalence_scores.get(prevalence, 0.50)
                    matched.append({'name': guideline_name, 'score': initial_score, 'data': guideline_data})
                    matched_guideline_names.add(guideline_name)
                    print(f"[Engine]   ✓ {guideline_name} (trigger: '{trigger}', match: faiss_semantic ({similarity:.2f}), prevalence: {prevalence})")
                else:
                    # Log first few rejections for visibility
                    if i < 5:
                        print(f"[Engine]   ✗ {guideline_name}: '{trigger}' (similarity={similarity:.2f} < 0.75)")
        
        print(f"\n[Engine] 📊 FAISS matching complete: {len(matched)} guidelines matched")
        
        # Store matching metadata for debug
        self.matching_metadata = {
            'mode': 'FAISS',
            'strategy': 'exact > subset > FAISS semantic',
            'thresholds': {
                'char_overlap': 0.75,
                'semantic': 0.75
            },
            'matched_count': len(matched),
            'filtered_count': len(self.all_guidelines) - len(matched)
        }
        
        return matched
    
    def _match_to_guidelines(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines
        
        Returns:
            List of matched guidelines with initial scores, sorted by relevance
        """
        complaint_lower = complaint.lower()
        
        # Extract core symptom by removing common filler words
        # This allows "I have abdominal pain" to match "lower abdominal pain"
        filler_words = ['i', 'have', 'my', 'the', 'a', 'an', 'is', 'am', 'feel', 'feeling']
        symptom_words = [w for w in complaint_lower.split() if w not in filler_words]
        core_symptom = ' '.join(symptom_words)
        
        matched = []
        
        print(f"\n[Engine] 🔍 MATCHING TO GUIDELINES...")
        print(f"[Engine] 📋 Core symptom extracted: '{core_symptom}'")
        print(f"[Engine] 🎯 Matching strategy:")
        print(f"[Engine]    1. Exact match (trigger in complaint)")
        print(f"[Engine]    2. Subset match (symptom in trigger)")
        print(f"[Engine]    3. Character overlap (Jaccard > 0.75)")
        print(f"[Engine]    4. Semantic similarity (cosine > 0.75)")
        print(f"[Engine] ---")
        
        # Thresholds
        CHAR_OVERLAP_THRESHOLD = 0.75  # Increased from 0.65
        SEMANTIC_THRESHOLD = 0.75  # Increased to reject wrong locations and generic descriptions
        
        # Helper function for character overlap
        def char_overlap(str1: str, str2: str) -> float:
            """Calculate character-level overlap between two strings (Jaccard similarity)"""
            set1 = set(str1.lower().replace(' ', ''))
            set2 = set(str2.lower().replace(' ', ''))
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        rejected_guidelines = []  # Track filtered guidelines
        
        for name, guideline in self.all_guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            print(f"\n[Engine] 🔍 Evaluating: {name}")
            print(f"[Engine]    Triggers: {triggers}")
            
            # Check if any trigger matches using HYBRID approach
            matched_trigger = None
            match_type = None
            
            for trigger in triggers:
                trigger_lower = trigger.lower()
                
                # FAST PATH: Exact keyword matching
                # 1. Exact: trigger in complaint (e.g., "chest pain" in "I have chest pain")
                if trigger_lower in complaint_lower:
                    print(f"[Engine]    ✅ EXACT MATCH: '{trigger_lower}' found in '{complaint_lower}'")
                    matched_trigger = trigger
                    match_type = "exact"
                    break
                else:
                    print(f"[Engine]    ❌ Exact: '{trigger_lower}' not in '{complaint_lower}'")
                
                # 2. Subset: core symptom in trigger (e.g., "abdominal pain" in "lower abdominal pain")
                if core_symptom in trigger_lower:
                    print(f"[Engine]    ✅ SUBSET MATCH: '{core_symptom}' found in '{trigger_lower}'")
                    matched_trigger = trigger
                    match_type = "subset"
                    break
                else:
                    print(f"[Engine]    ❌ Subset: '{core_symptom}' not in '{trigger_lower}'")
            
            # Track best scores for this guideline (for final rejection logging)
            best_overlap = 0.0
            best_trigger_for_overlap = None
            best_semantic = 0.0
            best_trigger_for_semantic = None
            
            # CHARACTER OVERLAP: Check if trigger and symptom share significant characters
            # This filters "chest pain" (~0.3 overlap) from "abdominal pain"
            if not matched_trigger:
                print(f"[Engine]    🔤 Checking character overlap (Jaccard similarity)...")
                for trigger in triggers:
                    overlap = char_overlap(core_symptom, trigger)
                    print(f"[Engine]       '{core_symptom}' vs '{trigger}' = {overlap:.2f} (need >{CHAR_OVERLAP_THRESHOLD})")
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_trigger_for_overlap = trigger
                    if overlap > CHAR_OVERLAP_THRESHOLD:
                        print(f"[Engine]    ✅ CHAR OVERLAP MATCH: {overlap:.2f} > {CHAR_OVERLAP_THRESHOLD}")
                        matched_trigger = trigger
                        match_type = f"char_overlap ({overlap:.2f})"
                        break
                
                if not matched_trigger and best_overlap > 0.0:
                    print(f"[Engine]    ❌ Best overlap: {best_overlap:.2f} < {CHAR_OVERLAP_THRESHOLD} (rejected)")
            
            # SEMANTIC PATH: Use embeddings for fuzzy/synonym matching
            # Handles typos ("abdomnal pain"), synonyms ("belly ache" = "abdominal pain")
            # STRICT threshold to avoid false matches across body regions
            if not matched_trigger and self.embedding_model:
                print(f"[Engine]    🧠 Checking semantic similarity (embeddings)...")
                for trigger in triggers:
                    try:
                        similarity = self._compute_similarity(core_symptom, trigger)
                        print(f"[Engine]       '{core_symptom}' vs '{trigger}' = {similarity:.2f} (need >{SEMANTIC_THRESHOLD})")
                        if similarity > best_semantic:
                            best_semantic = similarity
                            best_trigger_for_semantic = trigger
                    except Exception as sim_error:
                        print(f"[Engine] ❌ Initial similarity computation failed for trigger '{trigger}': {sim_error}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next trigger
                        continue
                    if similarity > SEMANTIC_THRESHOLD:
                        print(f"[Engine]    ✅ SEMANTIC MATCH: {similarity:.2f} > {SEMANTIC_THRESHOLD}")
                        matched_trigger = trigger
                        match_type = f"semantic ({similarity:.2f})"
                        break
                
                if not matched_trigger and best_semantic > 0.0:
                    print(f"[Engine]    ❌ Best semantic: {best_semantic:.2f} < {SEMANTIC_THRESHOLD} (rejected)")
            
            if matched_trigger:
                # Initial score based on PREVALENCE from guideline JSON
                prevalence = guideline.get('prevalence', 'uncommon')
                
                prevalence_scores = {
                    'common': 0.60,    # Frequent conditions
                    'uncommon': 0.50,  # Moderate frequency
                    'rare': 0.40       # Low frequency but important
                }
                
                initial_score = prevalence_scores.get(prevalence, 0.50)
                
                matched.append({
                    'name': name,
                    'score': initial_score,
                    'data': guideline
                })
                print(f"[Engine]    ══════════════════════════════════════")
                print(f"[Engine]    ✅ ACCEPTED: {name}")
                print(f"[Engine]       Match: {match_type}")
                print(f"[Engine]       Trigger: '{matched_trigger}'")
                print(f"[Engine]       Prevalence: {prevalence}")
                print(f"[Engine]       Initial score: {initial_score:.0%}")
            else:
                # Guideline rejected - log ONCE with best score info
                print(f"[Engine]    ══════════════════════════════════════")
                print(f"[Engine]    ❌ REJECTED: {name} (no match found)")
                
                # Determine primary rejection reason (semantic takes priority as it's the final check)
                if best_semantic > 0.0:
                    rejected_guidelines.append({
                        'name': name,
                        'reason': 'semantic_low',
                        'trigger': best_trigger_for_semantic,
                        'score': best_semantic,
                        'threshold': SEMANTIC_THRESHOLD,
                        'char_overlap': best_overlap  # Include char overlap for reference
                    })
                elif best_overlap > 0.0:
                    rejected_guidelines.append({
                        'name': name,
                        'reason': 'char_overlap_low',
                        'trigger': best_trigger_for_overlap,
                        'score': best_overlap,
                        'threshold': CHAR_OVERLAP_THRESHOLD,
                        'semantic': None
                    })
        
        # Print filtered guidelines summary
        if rejected_guidelines:
            print(f"\n[Engine] 🚫 FILTERED OUT ({len(rejected_guidelines)} guidelines):")
            # Group by reason
            char_filtered = [g for g in rejected_guidelines if g['reason'] == 'char_overlap_low']
            sem_filtered = [g for g in rejected_guidelines if g['reason'] == 'semantic_low']
            
            if char_filtered:
                print(f"[Engine] 📊 Character overlap < {CHAR_OVERLAP_THRESHOLD}:")
                for g in sorted(char_filtered, key=lambda x: -x['score'])[:5]:  # Show top 5
                    print(f"[Engine]    ✗ {g['name']}: '{g['trigger']}' (overlap={g['score']:.2f}, need >{g['threshold']:.2f})")
            
            if sem_filtered:
                print(f"[Engine] 📊 Semantic similarity < {SEMANTIC_THRESHOLD}:")
                for g in sorted(sem_filtered, key=lambda x: -x['score'])[:5]:  # Show top 5
                    print(f"[Engine]    ✗ {g['name']}: '{g['trigger']}' (similarity={g['score']:.2f}, need >{g['threshold']:.2f})")
        
        # PREVALENCE-FIRST sorting with urgency boost
        # NOTE: Gender filtering happens AFTER sex is collected (see _filter_by_gender)
        # Goal: Common urgent (appendicitis) BEFORE rare emergent (ectopic)
        # But emergent conditions get a boost to stay competitive
        
        urgency_boost = {
            'emergent': 0.15,  # +15% boost
            'urgent': 0.00,    # No change
            'routine': -0.05   # -5% penalty
        }
        
        # Apply urgency boost to scores (prevalence + urgency adjustment)
        for m in matched:
            urgency = m['data'].get('urgency', 'routine')
            m['combined_score'] = m['score'] + urgency_boost.get(urgency, 0)
        
        # Sort by combined score (prevalence + urgency boost)
        matched.sort(key=lambda x: -x['combined_score'])
        
        print(f"\n[Engine] 📊 SORTED BY COMBINED SCORE (prevalence + urgency boost):")
        print(f"[Engine] 📋 Total matched: {len(matched)} conditions")
        for i, m in enumerate(matched, 1):  # Show ALL
            urgency = m['data'].get('urgency', 'routine')
            prevalence = m['data'].get('prevalence', 'uncommon')
            print(f"[Engine]   {i}. {m['name']} ({prevalence}, {urgency}, combined: {m['combined_score']:.0%})")
        
        # Store matching metadata for debug
        self.matching_metadata = {
            'mode': 'brute-force',
            'strategy': 'exact > subset > char_overlap(>0.75) > semantic(>0.75)',
            'thresholds': {
                'char_overlap': CHAR_OVERLAP_THRESHOLD,
                'semantic': SEMANTIC_THRESHOLD
            },
            'matched_count': len(matched),
            'filtered_count': len(rejected_guidelines)
        }
        
        return matched
    
    def _ask_next_clinical_question(self) -> Dict[str, Any]:
        """
        Use LLM to analyze all 3 guidelines and generate next best question
        
        This is the CORE intelligence of the system.
        """
        print(f"\n{'='*80}")
        print(f"[Engine] 🧠 LLM QUESTION GENERATION")
        print(f"{'='*80}")
        
        # Build context for LLM (MINIMAL - no guidelines, just OLDCARTS template)
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # Get questions already asked
        asked = []
        for item in self.conversation_history:
            if item['type'] == 'question' and item.get('focus') not in ['age', 'sex']:
                asked.append(item['question'])
        
        print(f"[Engine] 📋 Patient: {patient_info}")
        print(f"[Engine] 📋 Questions asked: {len(asked)}")
        
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
            print(f"[Engine] 🧠 Generating question for OLDCARTS element: {next_element}")
            
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
            
            system_msg = "You are a medical assistant. Output ONLY ONE question. Use PLAIN LANGUAGE (no medical jargon). Never combine multiple questions."
            
            user_msg = f"""Patient: {patient_info} with {symptom}

Ask about: {element_desc}

Example: "{example}"

Generate EXACTLY ONE similar question using SIMPLE, PLAIN LANGUAGE that anyone can understand (open-ended, NOT yes/no). Do NOT combine multiple questions:"""
            
            # Get thinking filler before LLM call
            filler = get_filler('question_generation', use_audio=True)
            print(f"[Engine] 💬 Filler: [{filler['id']}] '{filler['text']}'")
            
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
                    print(f"[Engine] ⚠️ LLM used medical jargon - using plain language template")
                else:
                    print(f"[Engine] ⚠️ LLM combined multiple questions - using template fallback")
                print(f"[Engine]    Generated: '{question}'")
                print(f"[Engine]    Using template: '{example}'")
                # Use simple template fallback
                question = example
            
            oldcarts_element = next_element
            
            print(f"[Engine] ✅ OLDCARTS Question ({next_element}): '{question}'")
            
            # Mark as covered
            self.oldcarts_covered[oldcarts_element] = True
            print(f"{'='*80}\n")
            
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
                'filler': filler,  # Play/send this immediately while waiting
                'debug': self._get_debug_info()  # For Telegram debug display
            }
        
        # After OLDCARTS: Ask about associated symptoms using LLM
        print(f"[Engine] ℹ️  OLDCARTS complete - now asking about associated symptoms to reach 95% confidence")
        print(f"[Engine] 🧠 Generating associated symptom question...")
        
        # SAFETY: Check if we have active guidelines
        if not self.active_guidelines:
            print(f"[Engine] ❌ No active guidelines remaining - cannot generate question")
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Please seek medical evaluation."
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
            print(f"[Engine] ⚠️ Error extracting key symptoms: {e}")
        
        symptoms_context = ', '.join([s.split(':')[0] for s in key_symptoms[:3]]) if key_symptoms else "common symptoms"
        
        system_msg = "You are a medical assistant. Output ONLY ONE question. Use PLAIN LANGUAGE (no medical jargon). Never combine multiple questions."
        
        user_msg = f"""Patient: {patient_info}

Ask about ONE associated symptom using SIMPLE language (fever, nausea, vomiting, diarrhea, etc). EXACTLY ONE question only.

Example: "Have you had any fever?"

Your question:"""
        
        # Get thinking filler before LLM call
        filler = get_filler('question_generation', use_audio=True)
        print(f"[Engine] 💬 Filler: [{filler['id']}] '{filler['text']}'")
        
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
            print(f"[Engine] ⚠️ LLM combined multiple questions - using template")
            print(f"[Engine]    Generated: '{question}'")
            # Use simple template fallback
            question = "Have you had any fever?"
        
        print(f"[Engine] ✅ Associated symptom question: '{question}'")
        print(f"{'='*80}\n")
        
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
            'filler': filler,  # Play/send this immediately while waiting
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
        Compute semantic similarity between two texts using embeddings
        
        Returns: Similarity score 0-1
        
        Raises:
            RuntimeError if embeddings not available or computation fails
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 must be non-empty")
        
        # Generate embeddings
        emb1 = self.embedding_model.encode([text1])[0]
        emb2 = self.embedding_model.encode([text2])[0]
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def _compute_enhanced_location_similarity(self, user_answer: str, oldcarts_section: str) -> float:
        """Semantic similarity + keyword similarity boost for location matching"""
        # Step 1: Get base semantic similarity
        semantic_similarity = self._compute_similarity(user_answer, oldcarts_section)
        
        # Step 2: Calculate keyword similarity boost
        keyword_boost = self._compute_keyword_similarity_boost(user_answer, oldcarts_section)
        
        # Step 3: Combine semantic + keyword boost
        enhanced_similarity = min(1.0, semantic_similarity + keyword_boost)
        
        if keyword_boost > 0:
            print(f"[Engine]   📈 Enhanced similarity: {semantic_similarity:.3f} + {keyword_boost:.3f} = {enhanced_similarity:.3f}")
        
        return enhanced_similarity
    
    def _compute_keyword_similarity_boost(self, user_answer: str, oldcarts_section: str) -> float:
        """Calculate weighted keyword similarity boost using fuzzy matching"""
        user_words = user_answer.lower().split()
        guideline_words = oldcarts_section.lower().split()
        
        if not user_words or not guideline_words:
            return 0.0
        
        # Calculate weighted matches
        total_weight = 0.0
        matched_words = []
        
        for user_word in user_words:
            best_match_weight = 0.0
            best_match_word = None
            
            for guideline_word in guideline_words:
                # Calculate fuzzy match score (0.0 to 1.0)
                fuzzy_score = self._fuzzy_match(user_word, guideline_word)
                
                if fuzzy_score > best_match_weight:
                    best_match_weight = fuzzy_score
                    best_match_word = guideline_word
            
            # Only count matches above threshold
            if best_match_weight > 0.6:  # 60% similarity threshold
                total_weight += best_match_weight
                matched_words.append(f"{user_word}≈{best_match_word}({best_match_weight:.2f})")
        
        # Calculate weighted ratio
        weighted_ratio = total_weight / len(user_words)
        
        # Convert to boost (0.0 to 0.4 range) - increased for complex descriptions
        keyword_boost = weighted_ratio * 0.4
        
        if keyword_boost > 0.02:  # Only log significant boosts
            print(f"[Engine]   🔑 Weighted keyword boost: {total_weight:.2f}/{len(user_words)} words = {weighted_ratio:.3f} → +{keyword_boost:.3f}")
            print(f"[Engine]   🔑 Matched words: {matched_words}")
        
        return keyword_boost
    
    def _fuzzy_match(self, word1: str, word2: str) -> float:
        """Calculate fuzzy match score between two words (0.0 to 1.0)"""
        # Exact match
        if word1 == word2:
            return 1.0
        
        # Substring match (one word contains the other)
        if word1 in word2 or word2 in word1:
            return 0.8
        
        # Character overlap similarity (Jaccard)
        set1 = set(word1)
        set2 = set(word2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        jaccard_score = len(intersection) / len(union)
        
        # Boost for similar length words
        length_ratio = min(len(word1), len(word2)) / max(len(word1), len(word2))
        length_boost = length_ratio * 0.2
        
        return min(1.0, jaccard_score + length_boost)
    
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """
        Score guidelines using SEMANTIC SIMILARITY between answer and corresponding OLDCARTS section
        
        This is the CORE diagnostic reasoning - using vector similarity instead of LLM.
        """
        print(f"\n{'='*80}")
        print(f"[Engine] 🔢 LLM SCORING PHASE")
        print(f"{'='*80}")
        
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
        
        print(f"[Engine] 📋 Last Question: '{last_q}'")
        print(f"[Engine] 📋 Answer: '{answer}'")
        print(f"[Engine] 📋 History: {len(qa_pairs)} Q&A pairs")
        
        # Determine which OLDCARTS element was just asked
        last_question_item = None
        for item in reversed(self.conversation_history):
            if item.get('type') == 'question' and item.get('focus') == 'clinical':
                last_question_item = item
                break
        
        oldcarts_element = last_question_item.get('oldcarts') if last_question_item else None
        
        # ASSOCIATED SYMPTOMS: Score using KEY POSITIVES/NEGATIVES sections
        if not oldcarts_element:
            print(f"\n[Engine] 🎯 ASSOCIATED SYMPTOM SCORING:\n")
            print(f"[Engine] 📋 Matching '{answer}' to KEY POSITIVES/NEGATIVES sections\n")
            
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
                        print(f"[Engine] ❌ Associated symptoms similarity computation failed for {g['name']}: {sim_error}")
                        import traceback
                        traceback.print_exc()
                        # Skip this guideline and continue with the next one
                        continue
                    
                    # Small weight for associated symptoms (10% vs 30% for OLDCARTS)
                    old_score = g['score']
                    new_score = (old_score * 0.9) + (similarity * 0.1)
                    g['score'] = new_score
                    
                    change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
                    print(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (similarity: {similarity:.2f})")
            
            # Re-rank after associated symptom scoring
            all_guidelines.sort(key=lambda x: x['score'], reverse=True)
            self.active_guidelines = all_guidelines[:self.MAX_ACTIVE]
            self.reserve_pool = all_guidelines[self.MAX_ACTIVE:]
            
            print(f"\n[Engine] 📊 UPDATED RANKINGS after associated symptom:")
            for i, g in enumerate(self.active_guidelines, 1):
                print(f"[Engine]   {i}. {g['name']}: {g['score']:.0%}")
            
            print(f"\n")
            
            # Continue to next question
            return self._ask_next_clinical_question()
        
        # FOR EACH GUIDELINE: Score using VECTOR SIMILARITY
        # IMPORTANT: Score ALL guidelines (active + reserve) so we can re-rank dynamically
        print(f"\n[Engine] 🎯 SEMANTIC SIMILARITY SCORING:\n")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized - cannot compute similarity")
        
        print(f"[Engine] 📊 Matching answer to OLDCARTS element: {oldcarts_element}")
        print(f"[Engine] 📋 Scoring ALL {len(self.active_guidelines) + len(self.reserve_pool)} guidelines (active + reserve)\n")
        
        # Combine active + reserve for scoring
        all_guidelines = self.active_guidelines + self.reserve_pool
        
        for g in all_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            
            # Extract the specific OLDCARTS section for this element
            oldcarts_section = self._extract_oldcarts_section(classic, oldcarts_element)
            
            if not oldcarts_section:
                print(f"[Engine] ⚠️ Warning: Could not extract {oldcarts_element} section from {g['name']} - skipping this guideline")
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
                    print(f"[Engine]   {g['name']}: Enhanced location similarity = {similarity:.3f} ('{answer}' vs '{oldcarts_section[:50]}...')")
                except Exception as sim_error:
                    print(f"[Engine] ❌ Enhanced similarity computation failed for {g['name']}: {sim_error}")
                    import traceback
                    traceback.print_exc()
                    # Skip this guideline and continue with the next one
                    continue
            else:
                # Compute semantic similarity normally for non-location questions
                try:
                    similarity = self._compute_similarity(answer, oldcarts_section)
                except Exception as sim_error:
                    print(f"[Engine] ❌ Similarity computation failed for {g['name']}: {sim_error}")
                    import traceback
                    traceback.print_exc()
                    # Skip this guideline and continue with the next one
                    continue
            
            # Update score
            old_score = g['score']
            if similarity == 0.0:
                # Hard mismatch (e.g., left vs right) - rule out immediately
                new_score = 0.0
                g['score'] = new_score
                change = "❌"
                print(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change} (keyword mismatch)")
            else:
                # Normal weighted average (70% old score + 30% new similarity)
                # This prevents wild swings while incorporating new information
                new_score = (old_score * 0.7) + (similarity * 0.3)
                g['score'] = new_score
                change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
                print(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change}")
                print(f"[Engine]     Similarity: {similarity:.2f} to {oldcarts_element} section")
                print(f"[Engine]     Section: {oldcarts_section[:80]}...")
        
        # DYNAMIC RE-RANKING: Sort ALL guidelines by updated scores
        # This ensures conditions like Diverticulitis (LLQ) jump to top when "left side" is mentioned
        print(f"\n[Engine] 🔄 RE-RANKING all guidelines by updated scores...")
        
        # Rule out any with score < threshold
        ruled_out_this_round = []
        remaining = []
        for g in all_guidelines:
            if g['score'] < self.RULE_OUT_THRESHOLD:
                print(f"[Engine] ❌ RULING OUT: {g['name']} (score {g['score']:.0%} < {self.RULE_OUT_THRESHOLD:.0%})")
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
            print(f"\n[Engine] 🔼 PROMOTED to active:")
            for g in promoted_this_round:
                print(f"[Engine]   ↑ {g['name']} (score: {g['score']:.0%})")
        
        if demoted_this_round:
            print(f"\n[Engine] 🔽 DEMOTED to reserve:")
            for g in demoted_this_round:
                print(f"[Engine]   ↓ {g['name']} (score: {g['score']:.0%})")
        
        print(f"\n[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            print(f"[Engine]   {i}. {g['name']}: {g['score']:.0%} {urgency_emoji}")
        
        # Always show pool statistics
        print(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
        print(f"{'='*80}\n")
        
        # LOCATION CLARIFICATION: Use guidelines to determine if more detail needed
        # Extract actual location descriptions from top active guidelines and compare
        if oldcarts_element == 'L' and len(self.active_guidelines) >= 2:
            try:
                print(f"[Engine] 🔍 Checking if location answer differentiates top diagnoses...")
                print(f"[Engine] 📊 Active guidelines count: {len(self.active_guidelines)}")
                
                # Get L sections from top 5 active guidelines
                location_texts = []
                for i, g in enumerate(self.active_guidelines[:5]):
                    print(f"[Engine] 📋 Processing guideline {i+1}: {g['name']}")
                    classic = g['data'].get('key_features', {}).get('classic_presentation', '')
                    location_section = self._extract_oldcarts_section(classic, 'L')
                    print(f"[Engine] 📍 Location section: '{location_section[:50]}...' (length: {len(location_section)})")
                    if location_section:
                        location_texts.append(f"{g['name']}: {location_section}")
                    else:
                        print(f"[Engine] ⚠️ No location section found for {g['name']}")
            except Exception as loc_error:
                print(f"[Engine] ❌ Location clarification check failed: {loc_error}")
                import traceback
                traceback.print_exc()
                # Mark location as covered and move on
                self.oldcarts_covered['L'] = True
                location_texts = []  # Empty list to skip clarification
            
            print(f"[Engine] 📊 Location texts collected: {len(location_texts)}")
            if len(location_texts) >= 2:
                try:
                    print(f"[Engine] 🔍 Computing similarity between location sections...")
                    # Compute similarity between top guidelines' location sections
                    # If they're very different, patient answer may not differentiate
                    loc_embeddings = []
                    for i, loc_text in enumerate(location_texts[:3]):
                        # Just the location description, not the guideline name
                        loc_desc = loc_text.split(':', 1)[1].strip() if ':' in loc_text else loc_text
                        print(f"[Engine] 📍 Embedding {i+1}: '{loc_desc[:30]}...'")
                        try:
                            emb = self.embedding_model.encode([loc_desc])[0]
                            loc_embeddings.append(emb)
                            print(f"[Engine] ✅ Embedding {i+1} successful (shape: {emb.shape})")
                        except Exception as emb_error:
                            print(f"[Engine] ❌ Embedding {i+1} failed: {emb_error}")
                            import traceback
                            traceback.print_exc()
                            # Skip this embedding and continue
                            continue
                    
                    # Compute pairwise similarities between guideline locations
                    import numpy as np
                    similarities = []
                    for i in range(len(loc_embeddings)):
                        for j in range(i+1, len(loc_embeddings)):
                            sim = np.dot(loc_embeddings[i], loc_embeddings[j]) / (
                                np.linalg.norm(loc_embeddings[i]) * np.linalg.norm(loc_embeddings[j])
                            )
                            similarities.append((sim + 1) / 2)  # Normalize to [0, 1]
                    
                    avg_location_similarity = np.mean(similarities) if similarities else 1.0
                    
                    print(f"[Engine] 📊 Guideline location similarity: {avg_location_similarity:.2f}")
                except Exception as sim_error:
                    print(f"[Engine] ❌ Failed to compute location similarity: {sim_error}")
                    import traceback
                    traceback.print_exc()
                    # Default to high similarity (no clarification needed)
                    avg_location_similarity = 1.0
                    print(f"[Engine] 🔄 Defaulting to high similarity - skipping clarification")
                
                # If top guidelines describe DIFFERENT locations (low similarity), need clarification
                # But limit clarifications to prevent endless loops
                
                # Check how many times we've asked for location clarification
                location_clarifications = self.clarification_count.get('L', 0)
                
                print(f"[Engine] 📊 Clarification tracker: L={location_clarifications}/{self.MAX_CLARIFICATIONS}, Covered={self.oldcarts_covered.get('L', False)}")
                print(f"[Engine] 📊 Avg location similarity: {avg_location_similarity:.2f} (need >0.85 for specificity)")
                
                # SAFEGUARD: If already asked 2+ clarifications, FORCE move on (prevent infinite loop)
                if location_clarifications >= self.MAX_CLARIFICATIONS:
                    print(f"[Engine] ⚠️ Max location clarifications ALREADY reached ({location_clarifications}/{self.MAX_CLARIFICATIONS}) - forcing Location as covered")
                    self.oldcarts_covered['L'] = True
                elif avg_location_similarity < 0.85 and location_clarifications < self.MAX_CLARIFICATIONS:
                    print(f"[Engine] ⚠️ Top guidelines have diverse locations - need more specific answer (clarification #{location_clarifications + 1}/{self.MAX_CLARIFICATIONS})")
                    
                    # SAFETY: Find the last question item
                    last_q_item = None
                    for item in reversed(self.conversation_history):
                        if item.get('type') == 'question':
                            last_q_item = item
                            break
                    
                    if not last_q_item:
                        print(f"[Engine] ❌ No last question item - cannot generate clarification")
                        self.oldcarts_covered['L'] = True
                    else:
                        try:
                            # Collect ALL location-related Q&A pairs so far for full context
                            location_history = []
                            temp_q = None
                            for item in self.conversation_history:
                                if item.get('type') == 'question' and item.get('oldcarts') == 'L':
                                    temp_q = item.get('question', '')
                                elif item.get('type') == 'answer' and temp_q:
                                    a_text = item.get('answer', '')
                                    if temp_q and a_text:
                                        location_history.append(f"Q: {temp_q}\nA: {a_text}")
                                    temp_q = None  # Reset
                            
                            history_text = '\n'.join(location_history) if location_history else "None"
                            
                            # Use programmatic template-based approach for reliable questions
                            # This ensures we only ask questions that match guideline structure
                            
                            # Get thinking filler
                            filler = get_filler('location_clarification', use_audio=True)
                            print(f"[Engine] 💬 Filler: [{filler['id']}] '{filler['text']}'")
                            
                            # Use LLM to generate clarification based on guideline LOCATION sections
                            # This is more elegant than hardcoded logic
                            
                            # Collect LOCATION sections from top guidelines
                            location_sections = []
                            for g in self.active_guidelines[:3]:
                                try:
                                    location_section = self._extract_oldcarts_section(g['data'], 'LOCATION')
                                    if location_section:
                                        location_sections.append(f"{g['name']}: {location_section}")
                                except:
                                    continue
                            
                            if location_sections:
                                # Use LLM to generate appropriate clarification question
                                clarify_system = "You are a medical assistant. Output ONLY ONE question. Never combine multiple questions."
                                
                                clarify_user = f"""Patient's current answer: "{answer}"

Guideline LOCATION sections for reference:
{chr(10).join(location_sections)}

The patient's answer is still too vague. Ask EXACTLY ONE simple follow-up question to narrow down the location based on what the guidelines actually describe.

Use PLAIN LANGUAGE only (no medical jargon).

Your question:"""
                                
                                clarify_response = self.llm_chat_simple_fn(
                                    [
                                        {"role": "system", "content": clarify_system},
                                        {"role": "user", "content": clarify_user}
                                    ],
                                    max_tokens=40,
                                    temperature=0.3
                                )
                                
                                clarify_location = clarify_response.strip().strip('"\'')
                                if not clarify_location.endswith('?'):
                                    clarify_location += '?'
                                
                                # Simple validation - reject if too complex
                                if clarify_location.count('?') > 1 or len(clarify_location.split()) > 15:
                                    clarify_location = "Can you be more specific about the exact location?"
                            else:
                                clarify_location = "Can you be more specific about the exact location?"
                            
                            print(f"[Engine] 📍 Using template-based clarification: '{clarify_location}'")
                            
                            print(f"[Engine] 💬 Clarification: '{clarify_location}'")
                            print(f"{'='*80}\n")
                            
                            # Increment clarification counter
                            self.clarification_count['L'] = location_clarifications + 1
                            
                            # Preserve OLDCARTS element (keep as 'L' so we can ask again)
                            self.conversation_history.append({
                                'type': 'question',
                                'question': clarify_location,
                                'focus': 'clinical',
                                'oldcarts': 'L'  # Keep as location
                            })
                            
                            return {
                                'success': True,
                                'question': clarify_location,
                                'status': 'questioning',
                                'filler': filler,  # Play/send this immediately while waiting
                                'debug': self._get_debug_info(last_answer=answer)  # For Telegram debug display
                            }
                        
                        except Exception as clarify_error:
                            print(f"[Engine] ❌ Failed to generate location clarification: {clarify_error}")
                            import traceback
                            traceback.print_exc()
                            # Force mark as covered and move on
                            self.oldcarts_covered['L'] = True
                
                elif avg_location_similarity < 0.85:
                    # Hit max clarifications - force move on
                    print(f"[Engine] ⚠️ Max location clarifications reached ({location_clarifications}/{self.MAX_CLARIFICATIONS}) - accepting answer and moving on")
                    self.oldcarts_covered['L'] = True  # Force mark as covered
                else:
                    print(f"[Engine] ✅ Location answer has sufficient anatomical detail")
                    self.oldcarts_covered['L'] = True  # Mark as covered - we have enough detail!
            else:
                # No location texts or only one guideline - can't compare
                # Just mark as covered
                print(f"[Engine] ℹ️  Not enough guidelines to compare locations - accepting answer")
                self.oldcarts_covered['L'] = True
        
        # SAFETY CHECK: Ensure we have active guidelines
        if len(self.active_guidelines) == 0 and len(self.reserve_pool) == 0:
            print(f"[Engine] ❌ All guidelines exhausted - no diagnosis possible")
            print(f"[Engine] 📋 Ruled out {len(self.ruled_out)} conditions")
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Please seek medical evaluation."
            }
        
        # If active is empty but reserve exists, this shouldn't happen (rolling replacement should have filled it)
        if len(self.active_guidelines) == 0:
            print(f"[Engine] ⚠️ Active list empty but reserve has {len(self.reserve_pool)} - this is a bug")
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
        print(f"[Engine] 📋 OLDCARTS Coverage: {coverage_str} ({covered_count}/8)")
        
        # Diagnosis criteria: ALL OLDCARTS covered + high confidence, OR max 15 questions
        if oldcarts_complete and top['score'] >= 0.95:
            print(f"[Engine] ✅ DIAGNOSIS REACHED: {top['name']} ({top['score']:.0%} confidence, OLDCARTS complete)")
            print(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        elif num_questions >= 15:
            print(f"[Engine] ⚠️  DIAGNOSIS BY QUESTIONS LIMIT: {top['name']} ({top['score']:.0%}, OLDCARTS: {coverage_str})")
            print(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        else:
            if not oldcarts_complete:
                print(f"[Engine] 🔄 Continuing (OLDCARTS incomplete: missing {', '.join(uncovered)}, Q{num_questions}, score: {top['score']:.0%})")
            else:
                print(f"[Engine] 🔄 Continuing (OLDCARTS complete, need 95% confidence: current {top['score']:.0%}, Q{num_questions})")
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
            print(f"[Engine] ℹ️  No red flags to screen - proceeding to finalize")
            return self._finalize_diagnosis(diagnosis_obj)
        
        # If just starting screening, set status and reset index
        if self.status != 'red_flag_screening':
            self.status = 'red_flag_screening'
            self.red_flag_index = 0
            self.red_flags_present = []
            print(f"[Engine] 🚩 Screening {len(red_flags)} red flags for {diagnosis_obj['name']}")
        
        # If we've asked about all red flags, finalize
        if self.red_flag_index >= len(red_flags):
            print(f"[Engine] ✅ Red flag screening complete ({len(self.red_flags_present)} flags present)")
            return self._finalize_diagnosis(diagnosis_obj)
        
        # Ask about next red flag
        current_red_flag = red_flags[self.red_flag_index]
        
        # Convert red flag to yes/no question
        # Extract the core symptom from the red flag text
        question = self._red_flag_to_question(current_red_flag)
        
        print(f"[Engine] 🚩 Red flag {self.red_flag_index + 1}/{len(red_flags)}: {current_red_flag}")
        
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
            'status': 'red_flag_screening'
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
            question = "Have your eyes or skin turned yellow?"
        else:
            # Generic
            question = f"Have you experienced {red_flag.split('-')[0].strip().lower()}?"
        
        print(f"[Engine] ✅ Red flag question: '{question}'")
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
            print(f"[Engine] ⚠️  RED FLAGS DETECTED - Urgency escalated to: {urgency}")
        
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
        
        print(f"\n{'='*80}")
        print(f"[Engine] 🎯 FINAL DIAGNOSIS")
        print(f"{'='*80}")
        print(f"[Engine] Condition: {name}")
        print(f"[Engine] Confidence: {score:.0%}")
        print(f"[Engine] Urgency: {urgency}")
        if len(self.red_flags_present) > 0:
            print(f"[Engine] 🚨 Red Flags Detected: {len(self.red_flags_present)}")
            for rf in self.red_flags_present:
                print(f"[Engine]   - {rf}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'status': 'diagnosed',
            'diagnosis': name,
            'confidence': score,
            'urgency': urgency,
            'red_flags_detected': self.red_flags_present,
            'message': message
        }
    
    def _generate_opening_statement(self, chief_complaint: str) -> str:
        """
        LLM-generated empathetic opening statement
        """
        print(f"[Engine] 🧠 Generating opening statement...")
        
        system_msg = "Output ONLY the exact statement requested. No extra words."
        
        user_msg = f"""Patient: "{chief_complaint}"

Write a brief, natural empathetic statement to show you care:

Examples: 
- "I'm sorry to hear you're experiencing that."
- "That sounds uncomfortable, I'm here to help."
- "I understand that must be concerning."

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
            print(f"[Engine] ⚠️ Opening too long ({word_count} words) - using simple template")
            print(f"[Engine]    Generated: '{statement}'")
            statement = "I understand. I'll ask some questions to help."
        
        print(f"[Engine] ✅ Opening (simple model): '{statement}'")
        return statement
    
    def _generate_age_question(self) -> str:
        """
        LLM-generated age question
        """
        print(f"[Engine] 🧠 Generating age question...")
        
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
        print(f"[Engine] ✅ Age question (simple model): '{question}'")
        return question
    
    def _generate_sex_question(self) -> str:
        """
        LLM-generated biological sex question
        """
        print(f"[Engine] 🧠 Generating sex question...")
        
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
        print(f"[Engine] ✅ Sex question (simple model): '{question}'")
        return question
    
    def _generate_clarification_question(self, topic: str) -> str:
        """
        LLM-generated clarification question for invalid answers
        """
        print(f"[Engine] 🧠 Generating clarification for: {topic}")
        
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
        print(f"[Engine] ✅ Clarification (simple model): '{question}'")
        return question


# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")
