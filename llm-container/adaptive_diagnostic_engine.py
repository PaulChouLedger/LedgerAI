#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - Intelligent Medical Diagnosis System

This system uses:
1. JSON guidelines for chief complaint matching and scoring criteria
2. RAG for rich clinical content (questions, reasoning, differentials)
3. LLM for natural language understanding and question generation
4. Multi-guideline simultaneous scoring (not rigid decision tree)

Key differences from triage:
- Evaluates ALL guidelines simultaneously with weighted scoring
- Extracts multiple features from natural language answers
- Adapts question order based on information gain
- Can backtrack and update scores
- Natural conversation, not multiple choice
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests


class AdaptiveDiagnosticEngine:
    """
    Adaptive diagnostic engine that mimics clinical reasoning
    
    Hybrid Architecture:
    1. Rolling Top-5 Differential List (scalability)
    2. LLM + RAG Question Generation (intelligence)
    
    Flow:
    1. User states chief complaint
    2. Match to ALL relevant guidelines (JSON triggers + synonyms)
    3. Maintain top 5 active, rest in reserve pool
    4. For each question: Retrieve RAG content for top 3 differentials
    5. LLM reads clinical guidelines and generates intelligent question
    6. User answers → extract features, score all active guidelines
    7. Rule out low-scoring, promote from reserve (rolling update)
    8. Repeat until diagnosis clear
    9. Provide education using RAG content
    """
    
    def __init__(self, guidelines_dir: str = "/app/medical/guidelines", llm_chat_fn=None):
        """
        Initialize adaptive diagnostic engine
        
        Args:
            guidelines_dir: Path to directory containing JSON guidelines
            llm_chat_fn: LLM chat function for question generation
        """
        self.guidelines_dir = Path(guidelines_dir)
        self.guidelines = {}
        self.synonyms = {}
        self.llm_chat_fn = llm_chat_fn  # For intelligent question generation
        
        # Load all JSON guidelines
        self._load_guidelines()
        
        # Load medical synonyms for better matching
        self._load_synonyms()
        
        # Active assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files"""
        print(f"[Adaptive] 📚 Loading guidelines from {self.guidelines_dir}")
        
        if not self.guidelines_dir.exists():
            print(f"[Adaptive] ⚠️ Guidelines directory not found: {self.guidelines_dir}")
            return
        
        for json_file in self.guidelines_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    
                    # Use condition name as key
                    name = guideline.get('condition', guideline.get('guideline_name'))
                    if name:
                        self.guidelines[name] = guideline
                        print(f"[Adaptive]   ✓ Loaded: {name}")
            
            except Exception as e:
                print(f"[Adaptive] ⚠️ Failed to load {json_file.name}: {e}")
        
        print(f"[Adaptive] ✅ Loaded {len(self.guidelines)} guidelines")
    
    def _load_synonyms(self):
        """Load medical synonym dictionaries"""
        synonyms_dir = Path("/app/synonyms")
        
        if not synonyms_dir.exists():
            print(f"[Adaptive] ⚠️ Synonyms directory not found")
            return
        
        for syn_file in synonyms_dir.glob("*_synonyms.json"):
            try:
                with open(syn_file, 'r') as f:
                    syn_data = json.load(f)
                    self.synonyms.update(syn_data)
            except Exception as e:
                print(f"[Adaptive] ⚠️ Failed to load synonyms from {syn_file.name}: {e}")
        
        print(f"[Adaptive] ✅ Loaded {len(self.synonyms)} synonym mappings")
    
    def reset_assessment(self):
        """Reset state for new assessment"""
        self.active_guidelines = []  # Top 5 most likely diagnoses
        self.reserve_pool = []       # Remaining matched guidelines (sorted by score)
        self.ruled_out = []          # Guidelines ruled out (for reference)
        self.answered_features = {}  # Dict of extracted clinical features
        self.raw_answers = []        # Raw user responses for recap
        self.questions_asked = []    # History of questions asked
        self.demographics = {}       # Patient demographics (age, sex)
        self.chief_complaint = ""    # Original complaint
        self.status = "idle"         # idle, questioning, diagnosed
        self.diagnosis = None        # Final diagnosis when reached
        
        # Configuration
        self.MAX_ACTIVE = 5          # Keep top 5 differentials active
        self.RULE_OUT_THRESHOLD = 0.3  # Score below this → ruled out
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        Start new diagnostic assessment
        
        Args:
            chief_complaint: User's initial complaint (e.g., "I have abdominal pain")
        
        Returns:
            Response dict with first question
        """
        print(f"\n[Adaptive] 🔄 Starting new assessment for: '{chief_complaint}'")
        
        self.reset_assessment()
        self.status = "questioning"
        self.chief_complaint = chief_complaint  # Store for later use
        
        # Extract and normalize chief complaint
        normalized = self._normalize_text(chief_complaint)
        
        # Match to guidelines
        matched = self._match_chief_complaint(normalized)
        
        if not matched:
            print(f"[Adaptive] ❌ No guidelines matched for: '{chief_complaint}'")
            return {
                'success': False,
                'message': "I couldn't identify a specific medical condition from that description. Could you provide more details about your symptoms?"
            }
        
        print(f"[Adaptive] 🎯 Matched {len(matched)} guidelines:")
        for name, score in matched:
            print(f"[Adaptive]    - {name} (initial: {score:.2f})")
        
        # Sort matched guidelines by score
        matched_sorted = sorted(matched, key=lambda x: x[1], reverse=True)
        
        # Split into active (top 5) and reserve pool
        all_matched = [
            {
                'name': name,
                'score': initial_score,
                'guideline_data': self.guidelines[name],
                'rag_content': None
            }
            for name, initial_score in matched_sorted
        ]
        
        self.active_guidelines = all_matched[:self.MAX_ACTIVE]
        self.reserve_pool = all_matched[self.MAX_ACTIVE:]
        
        print(f"[Adaptive] 📊 Active differentials (top {len(self.active_guidelines)}):")
        for i, g in enumerate(self.active_guidelines, 1):
            print(f"[Adaptive]    {i}. {g['name']}: {g['score']:.3f}")
        
        if self.reserve_pool:
            print(f"[Adaptive] 💾 Reserve pool: {len(self.reserve_pool)} additional guidelines")
        
        # Extract symptom from chief complaint (e.g., "abdominal pain")
        symptom = self._extract_symptom_from_complaint(chief_complaint)
        
        # Generate natural, varied empathy + age question using LLM
        opening_question = self._generate_opening_message(symptom)
        
        self.questions_asked.append({
            'focus': 'demographics_age',
            'question': opening_question,
            'value': 'critical'
        })
        
        return {
            'success': True,
            'question': opening_question,
            'status': 'questioning',
            'differentials': [
                {'name': g['name'], 'score': g['score']} 
                for g in self.active_guidelines[:3]
            ]
        }
    
    def process_answer(self, user_answer: str) -> Dict[str, Any]:
        """
        Process user's answer and continue assessment
        
        Args:
            user_answer: User's natural language response
        
        Returns:
            Response dict with next question or diagnosis
        """
        # Check if this looks like a NEW chief complaint (restart assessment)
        if self._is_new_chief_complaint(user_answer):
            print(f"[Adaptive] 🔄 Detected new chief complaint - restarting assessment")
            return self.start_assessment(user_answer)
        
        if self.status != "questioning":
            return {
                'success': False,
                'message': "No active assessment"
            }
        
        print(f"\n[Adaptive] 💬 Processing answer: '{user_answer}'")
        
        # Validate answer is meaningful (not garbage transcription)
        if not self._is_valid_medical_response(user_answer):
            print(f"[Adaptive] ⚠️ Invalid/unclear answer - asking for clarification")
            last_q = self.questions_asked[-1] if self.questions_asked else None
            
            if last_q:
                return {
                    'success': True,
                    'question': f"I didn't quite catch that. {last_q['question']}",
                    'status': 'questioning'
                }
            else:
                return {
                    'success': False,
                    'message': "I didn't understand. Can you repeat that?"
                }
        
        # Context-aware validation: Check if answer actually addresses the question
        last_q = self.questions_asked[-1] if self.questions_asked else None
        if last_q and not self._answer_addresses_question(last_q, user_answer):
            print(f"[Adaptive] ⚠️ Answer doesn't address the question - re-asking")
            return {
                'success': True,
                'question': f"Could you be more specific? {last_q['question']}",
                'status': 'questioning'
            }
        
        # Extract ALL clinical features from the answer FIRST
        extracted = self._extract_features_from_text(user_answer)
        
        print(f"[Adaptive] 📝 Extracted features: {list(extracted.keys())}")
        
        # Handle demographics questions first
        last_q = self.questions_asked[-1] if self.questions_asked else None
        
        if last_q and last_q.get('focus') == 'demographics_age':
            # Extract age
            age_match = re.search(r'\d+', user_answer)
            if age_match:
                self.demographics['age'] = int(age_match.group())
                print(f"[Adaptive] 👤 Age: {self.demographics['age']}")
            
            # Ask for sex next (LLM-generated for variety)
            sex_question = self._generate_sex_question()
            
            self.questions_asked.append({
                'focus': 'demographics_sex',
                'question': sex_question,
                'value': 'critical'
            })
            
            return {
                'success': True,
                'question': sex_question,
                'status': 'questioning'
            }
        
        elif last_q and last_q.get('focus') == 'demographics_sex':
            # Extract sex
            text_lower = user_answer.lower()
            if 'female' in text_lower or 'woman' in text_lower or 'girl' in text_lower:
                self.demographics['sex'] = 'female'
            elif 'male' in text_lower or 'man' in text_lower or 'boy' in text_lower:
                self.demographics['sex'] = 'male'
            
            print(f"[Adaptive] 👤 Sex: {self.demographics.get('sex', 'unknown')}")
            
            # Now ask LOCATION (first clinical question for GI/abdominal complaints)
            if 'abdominal' in self.chief_complaint.lower() or 'stomach' in self.chief_complaint.lower() or 'belly' in self.chief_complaint.lower():
                location_q = "Where in your abdomen is the pain located?"
                
                self.questions_asked.append({
                    'focus': 'pain_location',
                    'question': location_q,
                    'value': 'critical'
                })
                
                return {
                    'success': True,
                    'question': location_q,
                    'status': 'questioning'
                }
        
        # Check if we just asked location and need clarification
        elif last_q and last_q.get('focus') == 'pain_location':
            text_lower = user_answer.lower()
            
            # Check if answer is ambiguous (needs clarification)
            ambiguous_right = 'right' in text_lower and not any(specific in text_lower for specific in ['upper right', 'lower right', 'ruq', 'rlq'])
            ambiguous_left = 'left' in text_lower and not any(specific in text_lower for specific in ['upper left', 'lower left', 'luq', 'llq'])
            
            if ambiguous_right:
                print(f"[Adaptive] ⚠️ Ambiguous location: 'right side' - clarifying")
                self.questions_asked.append({
                    'focus': 'pain_location_clarify',
                    'question': 'Is it in the upper right (below your ribs) or lower right side of your abdomen?',
                    'value': 'critical'
                })
                
                return {
                    'success': True,
                    'question': 'Is it in the upper right (below your ribs) or lower right side of your abdomen?',
                    'status': 'questioning'
                }
            
            elif ambiguous_left:
                print(f"[Adaptive] ⚠️ Ambiguous location: 'left side' - clarifying")
                self.questions_asked.append({
                    'focus': 'pain_location_clarify',
                    'question': 'Is it in the upper left or lower left side of your abdomen?',
                    'value': 'critical'
                })
                
                return {
                    'success': True,
                    'question': 'Is it in the upper left or lower left side of your abdomen?',
                    'status': 'questioning'
                }
        
        # Store raw answer + normalized matches for recap
        answer_record = {
            'question': last_q['question'] if last_q else 'chief_complaint',
            'question_focus': last_q.get('focus', 'unknown') if last_q else 'chief_complaint',
            'raw_answer': user_answer,
            'normalized_matches': [],
            'timestamp': len(self.raw_answers)
        }
        
        # Add normalized matches if any
        if 'positive_findings' in extracted:
            for finding in extracted['positive_findings']:
                answer_record['normalized_matches'].append({
                    'matched_to': finding['normalized_response'],
                    'fuzzy': finding.get('fuzzy_match', False),
                    'similarity': finding.get('similarity', 1.0)
                })
        
        self.raw_answers.append(answer_record)
        
        # Update answered features (merge, don't replace)
        if extracted:  # Only update if we extracted something
            self.answered_features.update(extracted)
            
            # Re-score ALL active guidelines
            self._score_all_guidelines()
        else:
            # No features extracted, but answer was valid
            # This might be a vague answer - continue asking
            print(f"[Adaptive] 💡 No specific features extracted, but answer was valid - continuing")
        
        # ROLLING UPDATE: Remove ruled-out guidelines, pull from reserve
        self._update_differential_list()
        
        # Sort by score
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        # Print current top candidates
        print(f"[Adaptive] 📊 Current active differentials:")
        for i, g in enumerate(self.active_guidelines, 1):
            print(f"[Adaptive]    {i}. {g['name']}: {g['score']:.3f}")
        
        if self.ruled_out:
            print(f"[Adaptive] ❌ Ruled out: {len(self.ruled_out)} conditions")
        if self.reserve_pool:
            print(f"[Adaptive] 💾 Reserve pool: {len(self.reserve_pool)} remaining")
        
        # Check if diagnosis reached
        # IMPORTANT: Require minimum questions to avoid premature diagnosis
        MIN_QUESTIONS_FOR_DIAGNOSIS = 4  # Must ask at least 4 questions
        questions_answered = len(self.raw_answers)
        
        if len(self.active_guidelines) > 0:
            top = self.active_guidelines[0]
            
            print(f"[Adaptive] 🔍 Diagnosis check: score={top['score']:.3f}, questions={questions_answered}/{MIN_QUESTIONS_FOR_DIAGNOSIS}")
            
            # High confidence diagnosis (AND minimum questions met)
            if top['score'] > 0.90 and questions_answered >= MIN_QUESTIONS_FOR_DIAGNOSIS:
                print(f"[Adaptive] ✅ High confidence threshold met - finalizing diagnosis")
                return self._finalize_diagnosis(top)
            
            # All critical questions answered with good score
            if questions_answered >= MIN_QUESTIONS_FOR_DIAGNOSIS and top['score'] > 0.80:
                print(f"[Adaptive] ✅ Minimum questions met with good score - finalizing diagnosis")
                return self._finalize_diagnosis(top)
            
            # Safety valve: if answered 6+ questions, finalize even with lower score
            if questions_answered >= 6:
                print(f"[Adaptive] ✅ Asked {questions_answered} questions - finalizing with available data")
                return self._finalize_diagnosis(top)
            
            # Otherwise, continue asking
            if questions_answered < MIN_QUESTIONS_FOR_DIAGNOSIS:
                print(f"[Adaptive] 🔄 Need more questions ({questions_answered}/{MIN_QUESTIONS_FOR_DIAGNOSIS}) - continuing")
        
        # Check if we've exhausted all guidelines
        if len(self.active_guidelines) == 0 and len(self.reserve_pool) == 0:
            print(f"[Adaptive] ⚠️ All guidelines ruled out - no diagnosis possible")
            self.reset_assessment()
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Please seek medical attention for a proper evaluation."
            }
        
        # Ask next discriminating question
        return self._ask_next_question()
    
    def _match_chief_complaint(self, normalized_text: str) -> List[Tuple[str, float]]:
        """
        Match chief complaint to guidelines using triggers and synonyms
        
        Returns:
            List of (guideline_name, initial_score) tuples
        """
        matched = []
        
        print(f"[Adaptive] 🔍 Matching '{normalized_text}' against {len(self.guidelines)} guidelines")
        
        for name, guideline in self.guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            print(f"[Adaptive]   Checking {name}: triggers={triggers}")
            
            # Check each trigger
            for trigger in triggers:
                trigger_normalized = self._normalize_text(trigger)
                
                # Direct match
                if trigger_normalized in normalized_text:
                    print(f"[Adaptive]     ✅ Direct match: '{trigger_normalized}' in text")
                    matched.append((name, 0.5))  # Base score for match
                    break
                
                # Synonym match
                for word in trigger_normalized.split():
                    if word in self.synonyms:
                        for synonym in self.synonyms[word]:
                            if synonym in normalized_text:
                                print(f"[Adaptive]     ✅ Synonym match: '{synonym}' → '{word}'")
                                matched.append((name, 0.4))  # Slightly lower for synonym
                                break
        
        # Remove duplicates, keep highest score
        unique_matched = {}
        for name, score in matched:
            if name not in unique_matched or score > unique_matched[name]:
                unique_matched[name] = score
        
        print(f"[Adaptive] 🎯 Matched {len(unique_matched)} guidelines")
        
        return list(unique_matched.items())
    
    def _update_differential_list(self):
        """
        Update the active differential list using rolling replacement strategy
        
        Key algorithm:
        1. Rule out guidelines with score < threshold
        2. Move ruled-out to ruled_out list
        3. Pull from reserve pool to maintain MAX_ACTIVE (5) guidelines
        4. This ensures we always consider top candidates without overwhelming
        """
        # Identify guidelines to rule out
        to_remove = []
        for guideline_obj in self.active_guidelines:
            if guideline_obj['score'] < self.RULE_OUT_THRESHOLD:
                to_remove.append(guideline_obj)
                print(f"[Adaptive] ❌ Ruling out: {guideline_obj['name']} (score: {guideline_obj['score']:.3f} < {self.RULE_OUT_THRESHOLD})")
        
        # Move to ruled_out
        for guideline_obj in to_remove:
            self.active_guidelines.remove(guideline_obj)
            self.ruled_out.append(guideline_obj)
        
        # Pull from reserve to maintain MAX_ACTIVE
        while len(self.active_guidelines) < self.MAX_ACTIVE and len(self.reserve_pool) > 0:
            # Get next from reserve (already sorted)
            next_guideline = self.reserve_pool.pop(0)
            self.active_guidelines.append(next_guideline)
            print(f"[Adaptive] 🔄 Promoting from reserve: {next_guideline['name']} (score: {next_guideline['score']:.3f})")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()
    
    def _extract_symptom_from_complaint(self, complaint: str) -> str:
        """Extract the symptom from chief complaint for empathy message"""
        complaint_lower = complaint.lower()
        
        if 'abdominal pain' in complaint_lower or 'stomach pain' in complaint_lower or 'belly pain' in complaint_lower:
            return "abdominal pain"
        elif 'chest pain' in complaint_lower:
            return "chest pain"
        elif 'headache' in complaint_lower or 'head pain' in complaint_lower:
            return "headache"
        elif 'pain' in complaint_lower:
            return "pain"
        else:
            return "these symptoms"
    
    def _is_new_chief_complaint(self, text: str) -> bool:
        """
        Detect if user is stating a NEW chief complaint vs answering a question
        
        Only returns True if it's clearly a different complaint, NOT just
        providing more detail about the current complaint.
        """
        text_lower = text.lower().strip()
        
        # If we already have a chief complaint, check if this is DIFFERENT
        if self.chief_complaint:
            current_symptom = self._extract_symptom_from_complaint(self.chief_complaint)
            new_symptom = self._extract_symptom_from_complaint(text)
            
            # Same symptom type = not a new complaint
            if current_symptom == new_symptom:
                return False
        
        # Only trigger on clear new chief complaints with symptom
        new_complaint_patterns = [
            ('i have', ['pain', 'ache', 'hurt', 'discomfort', 'burning', 'pressure', 'fever', 'cough', 'shortness']),
            ('i am having', ['pain', 'ache', 'difficulty', 'trouble']),
            ('i feel', ['pain', 'dizzy', 'weak', 'sick']),
        ]
        
        for pattern, symptoms in new_complaint_patterns:
            if text_lower.startswith(pattern):
                if any(symptom in text_lower for symptom in symptoms):
                    return True
        
        return False
    
    def _answer_addresses_question(self, question_obj: Dict, answer: str) -> bool:
        """
        LLM-based validation: Does the answer actually address what was asked?
        
        Pure LLM approach - let the AI use internal logic to evaluate responses.
        No hardcoded patterns.
        
        Returns False if answer is non-responsive to the question.
        """
        question = question_obj.get('question', '')
        
        # Use LLM to evaluate if answer addresses the question
        validation_prompt = f"""EVALUATE THIS PATIENT RESPONSE:

THE QUESTION ASKED:
{question}

THE PATIENT'S ANSWER:
{answer}

IMPORTANT: The answer text is SEPARATE from the question text. Do NOT confuse clarifications in the question with the patient's answer.

Does the patient's answer provide the information requested?

Think step-by-step:
Step 1: What information does the question request? (Ignore empathy/preamble)
Step 2: Looking ONLY at the patient's answer text, does it provide that information?
Step 3: Decision: YES or NO

Examples:

Q: "How old are you?" 
A: "35"
Step 1: Question requests age
Step 2: Answer provides age (35)
Decision: YES

Q: "Can you tell me your sex?"
A: "female"
Step 1: Question requests biological sex for medical purposes
Step 2: Answer provides biological sex (female)
Decision: YES

Q: "Are you male or female?"
A: "male"
Step 1: Question requests biological sex
Step 2: Answer provides biological sex (male)
Decision: YES

Q: "Is it upper right or lower right?" 
A: "on the upper"
Step 1: Question asks to choose between upper vs lower
Step 2: Answer says "upper" (specifies which)
Decision: YES

Q: "Is it upper right or lower right?" 
A: "right side"
Step 1: Question asks to choose between upper vs lower
Step 2: Answer says "right side" but doesn't specify upper or lower
Decision: NO

Q: "When did it start?" 
A: "yesterday"
Step 1: Question asks for timing
Step 2: Answer provides timing (yesterday)
Decision: YES

Now evaluate. Show your reasoning, then end with "Decision: YES" or "Decision: NO":"""
        
        try:
            # Debug: Show the validation prompt being sent
            print(f"\n[Adaptive] 🧠 VALIDATION PROMPT:")
            print(f"[Adaptive]    Q: '{question}'")
            print(f"[Adaptive]    A: '{answer}'")
            
            # Call LLM for validation
            response = self.llm_chat_fn(
                [{"role": "user", "content": validation_prompt}],
                max_tokens=100,  # Allow space for reasoning
                temperature=0.0  # Zero temp for most consistent judgments
            )
            
            response_text = response.strip()
            
            # Debug: Show LLM's full reasoning
            print(f"[Adaptive] 🧠 LLM REASONING:")
            for line in response_text.split('\n'):
                print(f"[Adaptive]    {line}")
            
            # Parse decision from response
            response_upper = response_text.upper()
            
            # Look for "Decision: YES" or "Decision: NO" in the response
            if 'DECISION:' in response_upper or 'DECISION =' in response_upper:
                # Extract the line with the decision
                decision_line = [line for line in response_text.split('\n') if 'decision' in line.lower()]
                if decision_line:
                    decision_text = decision_line[-1].upper()  # Take last occurrence
                    
                    if 'YES' in decision_text:
                        print(f"[Adaptive] ✅ Answer validation: ACCEPTED")
                        return True
                    elif 'NO' in decision_text:
                        print(f"[Adaptive] ❌ Answer validation: REJECTED")
                        return False
            
            # Fallback: Look for YES/NO anywhere in response
            if 'YES' in response_upper and 'NO' not in response_upper:
                print(f"[Adaptive] ✅ Answer validation: ACCEPTED (fallback parsing)")
                return True
            elif 'NO' in response_upper and 'YES' not in response_upper:
                print(f"[Adaptive] ❌ Answer validation: REJECTED (fallback parsing)")
                return False
            else:
                # Unclear response - be permissive
                print(f"[Adaptive] ⚠️ Unclear LLM validation response - accepting answer")
                return True
                
        except Exception as e:
            print(f"[Adaptive] ⚠️ Answer validation failed (LLM error): {e}")
            import traceback
            traceback.print_exc()
            # On error, be permissive (assume answer is acceptable)
            return True
    
    def _is_valid_medical_response(self, text: str) -> bool:
        """
        Validate that response is meaningful (not garbage transcription or too short)
        
        Returns False for:
        - Very short fragments ("on the", "time.", "go.")
        - Gibberish ("good else to go")
        - Empty responses
        
        Returns True for:
        - Numbers (age, duration, etc.)
        - Demographic answers (male/female)
        - Valid medical responses
        """
        text = text.strip()
        
        # Too short (but allow single-char if it's a number or letter)
        if len(text) < 2:
            return False
        
        # Check if this is a number (age, duration, etc.) - ALWAYS VALID
        if re.search(r'\d+', text):
            return True
        
        # Check if this is a demographic answer (male/female/man/woman) - ALWAYS VALID
        demographic_answers = ['male', 'female', 'man', 'woman', 'boy', 'girl', 'm', 'f']
        if text.lower() in demographic_answers:
            return True
        
        # Only punctuation or filler words
        filler_patterns = [
            r'^(on the|the|a|an|to|for|with)[\s\.,]*$',
            r'^[\.,;:!?]+$',
            r'^(uh|um|er|ah)[\s\.,]*$'
        ]
        
        for pattern in filler_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        # Has at least one real word (2+ chars, reduced from 3)
        words = text.split()
        real_words = [w for w in words if len(w.strip('.,!?')) >= 2]
        
        if len(real_words) == 0:
            return False
        
        return True
    
    def _extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract clinical features using HYBRID approach:
        1. Universal features (onset time, severity) - simple patterns
        2. Guideline-specific features - from JSON expected_positive_responses
        
        Args:
            text: User's natural language text
        
        Returns:
            Dict with 'universal_features' and 'positive_findings'/'negative_findings'
        """
        features = {}
        text_lower = text.lower()
        
        # ==== UNIVERSAL FEATURES (not guideline-specific) ====
        
        # Onset timing - simple time marker detection
        if 'hour' in text_lower:
            features['onset_timing'] = 'acute_hours'
        elif 'day' in text_lower:
            features['onset_timing'] = 'acute_days'
        elif 'week' in text_lower:
            features['onset_timing'] = 'subacute'
        elif any(marker in text_lower for marker in ['month', 'year', 'chronic', 'always', 'long time']):
            features['onset_timing'] = 'chronic'
        
        # Special cases for "today"/"yesterday" without "day" in them
        elif 'today' in text_lower or 'tonight' in text_lower or 'this morning' in text_lower:
            features['onset_timing'] = 'acute_hours'
        elif 'yesterday' in text_lower or 'last night' in text_lower:
            features['onset_timing'] = 'acute_days'
        
        # Severity (universal)
        severity_match = re.search(r'(\d+)\s*(?:out of|/)\s*10', text_lower)
        if severity_match:
            features['severity_score'] = int(severity_match.group(1))
        elif any(word in text_lower for word in ['severe', 'terrible', 'unbearable', 'worst']):
            features['severity_score'] = 9
        elif any(word in text_lower for word in ['moderate', 'medium', 'okay']):
            features['severity_score'] = 5
        elif any(word in text_lower for word in ['mild', 'slight', 'minor']):
            features['severity_score'] = 3
        
        # ==== GUIDELINE-SPECIFIC FEATURES (from JSON) ====
        
        # For each active guideline, check all diagnostic questions
        for guideline_obj in self.active_guidelines:
            guideline = guideline_obj['guideline_data']
            guideline_name = guideline_obj['name']
            
            diagnostic_questions = guideline.get('diagnostic_questions', [])
            
            for question in diagnostic_questions:
                question_focus = question.get('question_focus', '')
                expected_positive = question.get('expected_positive_responses', [])
                negative_responses = question.get('negative_responses', [])
                diagnostic_value = question.get('diagnostic_value', 'moderate')
                
                # Check for negative responses first (rule out)
                for negative in negative_responses:
                    if negative.lower() in text_lower:
                        # Track negative findings
                        if 'negative_findings' not in features:
                            features['negative_findings'] = []
                        features['negative_findings'].append({
                            'guideline': guideline_name,
                            'question': question_focus,
                            'response': negative
                        })
                
                # Check for positive responses (with fuzzy matching for misspellings)
                for positive in expected_positive:
                    positive_lower = positive.lower()
                    
                    # Direct substring match
                    if positive_lower in text_lower:
                        # Track positive findings
                        if 'positive_findings' not in features:
                            features['positive_findings'] = []
                        features['positive_findings'].append({
                            'guideline': guideline_name,
                            'question': question_focus,
                            'response': positive,
                            'normalized_response': positive,  # What it matched to
                            'value': diagnostic_value
                        })
                        break
                    
                    # Fuzzy match for misspellings/mispronunciations
                    # Use SequenceMatcher for character-level similarity
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, positive_lower, text_lower).ratio()
                    
                    # Also check if key words from expected response are in text
                    positive_words = set(positive_lower.split())
                    text_words = set(text_lower.split())
                    word_overlap = len(positive_words & text_words) / len(positive_words) if positive_words else 0
                    
                    # Match if high similarity OR high word overlap
                    if similarity > 0.7 or word_overlap > 0.6:
                        if 'positive_findings' not in features:
                            features['positive_findings'] = []
                        features['positive_findings'].append({
                            'guideline': guideline_name,
                            'question': question_focus,
                            'response': positive,
                            'normalized_response': positive,  # Normalized to expected
                            'value': diagnostic_value,
                            'fuzzy_match': True,
                            'similarity': similarity
                        })
                        print(f"[Adaptive]    🔍 Fuzzy match: '{text_lower[:50]}' → '{positive}' (similarity: {similarity:.2f})")
                        break
        
        # Debug: Show what was extracted
        if len(features) > 0:
            print(f"[Adaptive] 🔍 Feature extraction:")
            if 'onset_timing' in features:
                print(f"[Adaptive]    ⏰ Onset: {features['onset_timing']}")
            if 'severity_score' in features:
                print(f"[Adaptive]    📊 Severity: {features['severity_score']}/10")
            if 'positive_findings' in features:
                for finding in features['positive_findings']:
                    print(f"[Adaptive]    ✅ {finding['guideline']}: {finding['question']} ({finding['value']})")
            if 'negative_findings' in features:
                for finding in features['negative_findings']:
                    print(f"[Adaptive]    ❌ {finding['guideline']}: {finding['question']} (rule out)")
        else:
            print(f"[Adaptive] ⚠️ No features matched: '{text}'")
        
        return features
    
    def _score_all_guidelines(self):
        """
        Score ALL active guidelines based on guideline-driven feature matches
        
        Uses the diagnostic_questions from each guideline's JSON to score matches.
        No hardcoded logic - 100% data-driven!
        """
        print(f"[Adaptive] 🔢 Scoring {len(self.active_guidelines)} guidelines")
        
        # Weight mapping for diagnostic_value
        diagnostic_weights = {
            'critical': 0.30,
            'high': 0.20,
            'moderate': 0.10,
            'low': 0.05
        }
        
        for guideline_obj in self.active_guidelines:
            guideline = guideline_obj['guideline_data']  # FIX: Extract guideline data
            guideline_name = guideline_obj['name']
            initial_score = guideline_obj['score']
            score = initial_score
            
            print(f"[Adaptive]   Scoring {guideline_name} (initial: {initial_score:.3f})")
            
            # Score universal features (apply to all guidelines)
            if 'onset_timing' in self.answered_features:
                onset = self.answered_features['onset_timing']
                urgency = guideline.get('urgency', '').lower()
                
                # Acute onset matches urgent/emergent conditions
                if onset in ['acute_hours', 'acute_days'] and urgency in ['urgent', 'emergent', 'emergency']:
                    score += 0.10
                    print(f"[Adaptive]     ✅ Onset timing matches urgency: +0.10")
                # Chronic onset doesn't match urgent conditions
                elif onset == 'chronic' and urgency in ['urgent', 'emergent', 'emergency']:
                    score -= 0.20
                    print(f"[Adaptive]     ❌ Chronic onset rules out urgent condition: -0.20")
            
            if 'severity_score' in self.answered_features:
                severity = self.answered_features['severity_score']
                urgency = guideline.get('urgency', '').lower()
                
                # High severity matches urgent conditions
                if severity >= 7 and urgency in ['urgent', 'emergent', 'emergency']:
                    score += 0.05
                    print(f"[Adaptive]     ✅ High severity ({severity}/10): +0.05")
            
            # Count positive findings for this guideline (from JSON)
            positive_count = 0
            if 'positive_findings' in self.answered_features:
                for finding in self.answered_features['positive_findings']:
                    if finding['guideline'] == guideline_name:
                        weight = diagnostic_weights.get(finding['value'], 0.10)
                        score += weight
                        positive_count += 1
                        print(f"[Adaptive]     ✅ {finding['question']}: +{weight:.2f} ({finding['value']})")
            
            # Penalize negative findings
            negative_count = 0
            if 'negative_findings' in self.answered_features:
                for finding in self.answered_features['negative_findings']:
                    if finding['guideline'] == guideline_name:
                        score -= 0.15  # Penalty for negative finding
                        negative_count += 1
                        print(f"[Adaptive]     ❌ {finding['question']}: -0.15 (rule out)")
            
            # Update score
            guideline_obj['score'] = max(0.0, min(score, 1.0))  # Clamp between 0-1
            
            print(f"[Adaptive]     Final: {guideline_obj['score']:.3f}")
    
    def _ask_next_question(self) -> Dict[str, Any]:
        """
        Select and ask the most discriminating next question
        
        Strategy: Find the question that appears in MULTIPLE top guidelines
        with high diagnostic value - this will best differentiate the differential.
        
        100% data-driven from guideline JSON!
        """
        if not self.active_guidelines:
            return {
                'success': False,
                'message': "I need more information to make a diagnosis."
            }
        
        # Collect all questions from ALL active guidelines
        all_questions = {}  # question_focus → list of (guideline_name, diagnostic_value)
        
        for guideline_obj in self.active_guidelines[:3]:  # Top 3 differentials
            guideline = guideline_obj['guideline_data']
            guideline_name = guideline_obj['name']
            diagnostic_questions = guideline.get('diagnostic_questions', [])
            
            for q in diagnostic_questions:
                focus = q.get('question_focus', '')
                value = q.get('diagnostic_value', 'moderate')
                
                if focus not in all_questions:
                    all_questions[focus] = []
                all_questions[focus].append({
                    'guideline': guideline_name,
                    'value': value,
                    'question_data': q
                })
        
        # Track which question_focus areas we've already asked about
        asked_focuses = set()
        for q in self.questions_asked:
            asked_focuses.add(q.get('focus', ''))
        
        print(f"[Adaptive] 📋 Already asked: {asked_focuses}")
        
        # Score each potential question by discriminating power
        question_scores = {}
        
        for focus, guidelines_asking in all_questions.items():
            if focus in asked_focuses:
                continue  # Skip already asked
            
            # Calculate discriminating power:
            # - How many top differentials have this question? (breadth)
            # - What's the diagnostic value? (importance)
            # - Is it 'critical' for top diagnosis? (priority)
            
            num_guidelines = len(guidelines_asking)
            avg_value_weight = sum([
                {'critical': 0.30, 'high': 0.20, 'moderate': 0.10, 'low': 0.05}.get(g['value'], 0.10)
                for g in guidelines_asking
            ]) / num_guidelines
            
            # Bonus if top guideline considers it critical/high
            top_guideline_bonus = 0
            for g in guidelines_asking:
                if g['guideline'] == self.active_guidelines[0]['name']:
                    if g['value'] == 'critical':
                        top_guideline_bonus = 0.5
                    elif g['value'] == 'high':
                        top_guideline_bonus = 0.3
            
            # Combined score
            discriminating_score = (num_guidelines * 0.3) + avg_value_weight + top_guideline_bonus
            question_scores[focus] = {
                'score': discriminating_score,
                'data': guidelines_asking[0]['question_data']  # Use first guideline's question
            }
        
        # Use LLM to generate intelligent question (if available)
        if self.llm_chat_fn:
            question_text = self._generate_llm_question()
        else:
            # Fallback: Use template-based generation
            if question_scores:
                best_focus = max(question_scores, key=lambda f: question_scores[f]['score'])
                best_question_data = question_scores[best_focus]['data']
                question_text = self._generate_question_from_focus(best_focus, best_question_data)
                
                print(f"[Adaptive] ✅ Selected: '{best_focus}' (score: {question_scores[best_focus]['score']:.2f})")
            else:
                question_text = "Can you tell me more about your symptoms?"
        
        # Track the question
        self.questions_asked.append({
            'focus': 'llm_generated' if self.llm_chat_fn else best_focus,
            'question': question_text,
            'value': 'high'
        })
        
        return {
            'success': True,
            'question': question_text,
            'status': 'questioning',
            'differentials': [
                {'name': g['name'], 'score': g['score']} 
                for g in self.active_guidelines[:3]
            ]
        }
        
        # All questions asked - try to finalize
        if len(self.active_guidelines) > 0:
            return self._finalize_diagnosis(self.active_guidelines[0])
        
        return {
            'success': False,
            'message': "I need more information to make a diagnosis."
        }
    
    def _generate_llm_question(self) -> str:
        """
        Use LLM + RAG to generate intelligent, conversational diagnostic question
        
        This is the KEY to making the system feel natural and adaptive.
        The LLM reads full clinical guidelines and reasons about what to ask.
        """
        print(f"[Adaptive] 🤖 Generating LLM-driven question...")
        
        # Get top 3 differentials for context
        top_3 = self.active_guidelines[:3]
        
        # Build clinical context from JSON key_features (structured, focused)
        guidelines_content = []
        for guideline_obj in top_3:
            guideline_name = guideline_obj['name']
            guideline_data = guideline_obj['guideline_data']
            
            # Get key_features from JSON
            key_features = guideline_data.get('key_features', {})
            classic_presentation = key_features.get('classic_presentation', '')
            
            # Get urgency
            urgency = guideline_data.get('urgency', 'routine')
            
            # Build focused clinical summary
            clinical_summary = f"""
{guideline_name} (Score: {guideline_obj['score']:.0%}, Urgency: {urgency}):
Classic Presentation: {classic_presentation}
"""
            
            guidelines_content.append({
                'name': guideline_name,
                'score': guideline_obj['score'],
                'content': clinical_summary
            })
            
            print(f"[Adaptive]   📚 Using key_features for {guideline_name}")
        
        # Build patient summary
        patient_summary = []
        for answer_obj in self.raw_answers:
            patient_summary.append(f"- {answer_obj['raw_answer']}")
        
        patient_info = "\n".join(patient_summary) if patient_summary else "No information yet"
        
        # Build differentials list
        differentials_list = "\n".join([
            f"{i}. {g['name']} (confidence: {g['score']:.0%})"
            for i, g in enumerate(top_3, 1)
        ])
        
        # Build clinical guidelines context from key_features
        guidelines_text = ""
        for g_content in guidelines_content:
            guidelines_text += g_content['content']
        
        # Build list of what we already know
        already_know = []
        if self.demographics.get('age'):
            already_know.append(f"Age: {self.demographics['age']}")
        if self.demographics.get('sex'):
            already_know.append(f"Sex: {self.demographics['sex']}")
        
        for answer in self.raw_answers:
            focus = answer.get('question_focus', '')
            if focus not in ['demographics_age', 'demographics_sex']:
                already_know.append(f"{focus}: {answer['raw_answer']}")
        
        already_assessed = "\n".join(already_know) if already_know else "None yet"
        
        # Prompt for LLM - FOCUSED on classic presentations
        prompt = f"""You are a physician conducting a medical interview about the patient's SYMPTOMS.

PATIENT: {self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')}
CHIEF COMPLAINT: {self.chief_complaint}

TOP 3 POSSIBLE DIAGNOSES:
{guidelines_text}

INFORMATION ALREADY GATHERED:
{already_assessed}

YOUR TASK:
Ask the SINGLE MOST IMPORTANT medical question about the patient's SYMPTOMS to help distinguish between these diagnoses.

CRITICAL REQUIREMENTS:
- Ask about a SYMPTOM or MEDICAL SIGN (fever, nausea, vomiting, pain quality, etc.)
- Ask ONLY ONE question (never combine multiple symptoms)
- Use simple, conversational language
- Be specific and direct

GOOD EXAMPLES (medical symptoms):
"Have you had any fever?"
"Have you vomited?"
"When did the pain start?"
"Is the pain sharp or dull?"
"Have you noticed any yellowing of your skin or eyes?"

BAD EXAMPLES (avoid these):
"What is the temperature like today?" (asking about weather, not fever!)
"Describe pain patterns and exacerbation sites" (too technical)
"Any fever, chills, or nausea?" (combining multiple symptoms)

OUTPUT ONLY THE SYMPTOM QUESTION (no number, no preamble):"""
        
        # Call LLM
        try:
            # Debug: Show abbreviated prompt context
            print(f"\n[Adaptive] 🧠 QUESTION GENERATION PROMPT:")
            print(f"[Adaptive]    Patient: {self.demographics.get('age', '?')} yo {self.demographics.get('sex', '?')}")
            print(f"[Adaptive]    Chief complaint: {self.chief_complaint}")
            print(f"[Adaptive]    Top differentials: {', '.join([g['name'] for g in top_3])}")
            
            response = self.llm_chat_fn(
                [{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            # Extract question from response
            question = response.strip()
            
            # Debug: Show raw LLM output before cleaning
            print(f"[Adaptive] 🧠 LLM RAW OUTPUT:")
            print(f"[Adaptive]    '{question}'")
            
            # IMMEDIATE garbage detection before any processing
            from collections import Counter
            if len(question) > 10:
                char_counts = Counter(c for c in question if c.isalnum())
                if char_counts:
                    most_common_char, count = char_counts.most_common(1)[0]
                    total_alnum = len([c for c in question if c.isalnum()])
                    ratio = count / total_alnum if total_alnum > 0 else 0
                    
                    if ratio > 0.4:  # More than 40% is same character = garbage
                        print(f"[Adaptive] ⚠️ GARBAGE DETECTED: char='{most_common_char}', ratio={ratio:.2f}, output='{question[:100]}'")
                        print(f"[Adaptive] 🔄 Using simple fallback question")
                        return "Can you tell me more about your symptoms?"
            
            # Clean up any meta-text, numbers, prefixes
            question = re.sub(r'^\d+[\.)]\s*', '', question)  # Remove "3. " or "3) "
            question = re.sub(r'^(Question|Q\d+|Next question):\s*', '', question, flags=re.IGNORECASE)
            question = question.split('\n')[0]  # Take first line only
            question = question.strip()
            
            # Strip quotes if LLM added them
            question = question.strip('"\'')
            
            # Debug: Show cleaned question
            print(f"[Adaptive] 🧠 LLM CLEANED OUTPUT:")
            print(f"[Adaptive]    '{question}'")
            
            # Ensure ends with ?
            if not question.endswith('?'):
                question += '?'
            
            # Final validation - reject if still too complex
            if len(question.split()) > 20 or '?' in question[:-1]:  # Multiple questions
                print(f"[Adaptive] ⚠️ LLM question too complex, using fallback")
                question = "Can you tell me more about your symptoms?"
            
            print(f"[Adaptive] ✅ FINAL QUESTION: '{question}'")
            
            return question
        
        except Exception as e:
            print(f"[Adaptive] ❌ LLM question generation failed: {e}")
            # Fallback to template
            return "Can you tell me more about your symptoms?"
    
    def _generate_opening_message(self, symptom: str) -> str:
        """
        Generate varied, natural opening empathy message + age question
        
        Uses LLM to avoid repetitive phrasing like:
        "I'm sorry you're experiencing X. Let me ask you some questions..."
        
        Returns empathy + age question combined (e.g., "I understand you're having stomach pain. 
        To help you better, can you tell me your age?")
        """
        print(f"\n[Adaptive] 💬 Generating opening message for: {symptom}")
        
        prompt = f"""You are a compassionate physician starting a medical interview.

The patient just said they have: {symptom}

Generate a natural opening that:
1. Shows empathy (varied phrasing, not always "I'm sorry")
2. Smoothly asks for their age

EXAMPLES:
"I understand you're having {symptom}. To help you better, can you tell me your age?"
"That sounds uncomfortable. Let me ask some questions to figure out what's going on. First, how old are you?"
"I'm here to help with your {symptom}. To start, what's your age?"

Use similar empathetic, conversational tone. Keep it SHORT (1-2 sentences max).

OUTPUT ONLY THE COMBINED MESSAGE (no preamble):"""
        
        try:
            response = self.llm_chat_fn(
                [{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.8  # Higher temp for variety
            )
            
            question = response.strip()
            
            # Debug: Show LLM output
            print(f"[Adaptive] 🧠 LLM OPENING RAW: '{question}'")
            
            # Strip quotes if LLM added them
            question = question.strip('"\'')
            
            # Ensure ends with ?
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Adaptive] ✅ FINAL OPENING: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Adaptive] ❌ Opening generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            return f"I understand you're having {symptom}. To help you, can you tell me your age?"
    
    def _generate_sex_question(self) -> str:
        """
        Generate varied, natural way to ask about patient's sex
        
        Avoids repetitive "Are you male or female?"
        """
        print(f"\n[Adaptive] 💬 Generating sex question")
        
        prompt = """You are a physician conducting a medical interview.

You need to ask about the patient's biological sex (for medical diagnostic purposes).

Generate a natural, respectful way to ask this. Vary the phrasing.

EXAMPLES:
"Are you male or female?"
"What's your biological sex?"
"Are you a man or a woman?"
"Can you tell me your sex?"

Use similar conversational tone. Keep it SHORT (one simple question).

OUTPUT ONLY THE QUESTION (no preamble):"""
        
        try:
            response = self.llm_chat_fn(
                [{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.8  # Higher temp for variety
            )
            
            question = response.strip()
            
            # Debug: Show LLM output
            print(f"[Adaptive] 🧠 LLM SEX QUESTION RAW: '{question}'")
            
            # Strip quotes if LLM added them
            question = question.strip('"\'')
            
            # Ensure ends with ?
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Adaptive] ✅ FINAL SEX QUESTION: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Adaptive] ❌ Sex question generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            return "Are you male or female?"
    
    def _generate_question_from_focus(self, focus: str, question_data: Dict) -> str:
        """
        Generate natural question from focus area
        
        Uses simple templates based on focus keywords.
        Could be enhanced with LLM in future.
        """
        focus_lower = focus.lower()
        
        # Use context as the question if available
        context = question_data.get('context', '')
        
        # Simple template mapping
        if 'onset' in focus_lower:
            return "When did this pain start?"
        elif 'location' in focus_lower or 'where' in focus_lower:
            return "Where exactly do you feel the pain?"
        elif 'migration' in focus_lower or 'move' in focus_lower:
            return "Did the pain start in one place and move to another?"
        elif 'quality' in focus_lower or 'character' in focus_lower:
            return "How would you describe the pain?"
        elif 'severity' in focus_lower:
            return "On a scale of 1-10, how severe is the pain?"
        elif 'appetite' in focus_lower or 'nausea' in focus_lower:
            return "Have you had any nausea or loss of appetite?"
        elif 'fever' in focus_lower:
            return "Have you had any fever?"
        elif 'bowel' in focus_lower:
            return "Have you had any changes in your bowel movements?"
        elif 'movement' in focus_lower:
            return "Does movement make the pain worse?"
        else:
            # Fallback: use the focus as-is
            return f"Can you tell me about {focus}?"
    
    def _finalize_diagnosis(self, diagnosis_obj: Dict) -> Dict[str, Any]:
        """
        Finalize diagnosis and provide education using RAG
        
        Args:
            diagnosis_obj: The guideline object with highest score
        
        Returns:
            Response with diagnosis and education
        """
        self.status = "diagnosed"
        self.diagnosis = diagnosis_obj
        
        guideline_name = diagnosis_obj['name']
        score = diagnosis_obj['score']
        urgency = diagnosis_obj['guideline_data'].get('urgency', 'routine')
        
        print(f"[Adaptive] ✅ DIAGNOSIS: {guideline_name} (confidence: {score:.2%})")
        
        # Retrieve education content from RAG
        education = self._get_education_from_rag(guideline_name, urgency)
        
        # Build diagnosis response
        response = {
            'success': True,
            'status': 'diagnosed',
            'diagnosis': guideline_name,
            'confidence': score,
            'urgency': urgency,
            'education': education,
            'message': self._format_diagnosis_message(guideline_name, urgency, education)
        }
        
        return response
    
    def _get_education_from_rag(self, guideline_name: str, urgency: str) -> Dict[str, str]:
        """Retrieve education content from RAG"""
        try:
            # Use metadata-based retrieval to get ALL chunks from this guideline
            response = requests.get(
                f"http://localhost:11435/rag/guideline/{guideline_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                chunks = data.get('results', [])
                
                # Extract relevant sections
                education = {
                    'description': '',
                    'urgency_info': '',
                    'red_flags': '',
                    'typical_treatment': ''
                }
                
                # Combine all chunk text
                all_text = '\n'.join([chunk.get('text', '') for chunk in chunks])
                
                # Extract sections (simple approach for now)
                if '🚨' in all_text or 'RED FLAG' in all_text.upper():
                    # Extract red flags section
                    red_flag_match = re.search(r'(RED FLAG.*?(?=\n\n|\Z))', all_text, re.DOTALL | re.IGNORECASE)
                    if red_flag_match:
                        education['red_flags'] = red_flag_match.group(1)
                
                return education
        
        except Exception as e:
            print(f"[Adaptive] ⚠️ Failed to retrieve education from RAG: {e}")
        
        return {}
    
    def _format_diagnosis_message(self, diagnosis: str, urgency: str, education: Dict) -> str:
        """Format diagnosis message for user with clinical recap"""
        
        urgency_messages = {
            'emergency': '🚨 This is a medical emergency. Call 911 immediately.',
            'urgent': '⚠️ This requires prompt medical attention. Go to the emergency room or urgent care today.',
            'semi_urgent': '⏰ This should be evaluated by a doctor within 24-48 hours.',
            'routine': '📋 Schedule an appointment with your primary care doctor.'
        }
        
        urgency_msg = urgency_messages.get(urgency, urgency_messages['routine'])
        
        # Build clinical recap from raw answers
        # Deduplicate: keep only the LAST answer for each question_focus
        focus_to_answer = {}
        for answer_obj in self.raw_answers:
            focus = answer_obj.get('question_focus', 'unknown')
            focus_to_answer[focus] = answer_obj  # Overwrites previous answer for same focus
        
        # Build recap from deduplicated answers
        recap_parts = []
        for focus, answer_obj in focus_to_answer.items():
            raw = answer_obj['raw_answer']
            
            # Skip non-informative answers
            if raw.lower() in ['yes', 'no', 'yes, sir', 'no, sir']:
                continue
            
            # Use focus area to determine how to phrase it
            if 'onset' in focus or 'when' in focus or 'timeline' in focus:
                recap_parts.append(f"pain started {raw}")
            elif 'location' in focus or 'where' in focus:
                recap_parts.append(f"located {raw}")
            elif 'migration' in focus or 'move' in focus:
                recap_parts.append(f"{raw}")
            elif 'quality' in focus or 'character' in focus:
                recap_parts.append(f"pain described as {raw}")
            elif 'severity' in focus:
                recap_parts.append(f"severity {raw}")
            elif 'appetite' in focus or 'nausea' in focus:
                recap_parts.append(f"with nausea/loss of appetite")
            elif 'fever' in focus:
                recap_parts.append(f"with fever")
            else:
                # Generic - just include the answer if substantive
                if len(raw.split()) > 2:  # More than 2 words
                    recap_parts.append(raw)
        
        recap = ", ".join(recap_parts) if recap_parts else "your symptoms"
        
        # Build message with recap
        message = f"Based on your symptoms - {recap} - this is likely {diagnosis}.\n\n{urgency_msg}"
        
        if education.get('red_flags'):
            message += f"\n\n{education['red_flags']}"
        
        return message


# Test function
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    
    # Test assessment
    response = engine.start_assessment("I have abdominal pain")
    print(f"\nResponse: {response}")

