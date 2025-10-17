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


class AdaptiveDiagnosticEngine:
    """
    LLM-driven diagnostic engine
    
    The LLM is the intelligence - it reads guidelines and reasons about diagnosis.
    We provide structure and keep it focused.
    """
    
    def __init__(self, guidelines_dir: str = "/app/medical/guidelines", llm_chat_fn=None, embedding_model=None):
        """
        Initialize diagnostic engine
        
        Args:
            guidelines_dir: Path to JSON guidelines
            llm_chat_fn: LLM function for reasoning
            embedding_model: Sentence transformer for semantic similarity
        """
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn
        self.embedding_model = embedding_model
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # Current assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files from subdirectories"""
        print(f"\n{'='*80}")
        print(f"[Engine] 📚 LOADING MEDICAL GUIDELINES")
        print(f"{'='*80}")
        
        if not self.guidelines_dir.exists():
            print(f"[Engine] ❌ Directory not found: {self.guidelines_dir}")
            return
        
        # Load from subdirectories (GI, GU, GYN, etc.)
        for json_file in sorted(self.guidelines_dir.glob("**/*.json")):
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    name = guideline.get('condition', json_file.stem)
                    organ_system = json_file.parent.name if json_file.parent != self.guidelines_dir else "Other"
                    self.all_guidelines[name] = guideline
                    print(f"[Engine]   ✓ {organ_system}/{name}")
            except Exception as e:
                print(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
        
        print(f"[Engine] ✅ Loaded {len(self.all_guidelines)} guidelines")
        print(f"{'='*80}\n")
    
    def reset_assessment(self):
        """Reset for new patient"""
        self.active_guidelines = []  # The 3 active guidelines with scores
        self.reserve_pool = []  # Remaining matched guidelines (for rolling replacement)
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
        
        # Thresholds
        self.RULE_OUT_THRESHOLD = 0.30  # Below 30% → rule out and replace
        self.MAX_ACTIVE = 3  # Keep 3 active differentials
    
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
        
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        
        # STEP 1: Match to 5 guidelines based on chief complaint triggers
        matched = self._match_to_guidelines(chief_complaint)
        
        if len(matched) == 0:
            return {
                'success': False,
                'message': "I couldn't identify relevant medical conditions. Please describe your symptoms more specifically."
            }
        
        # Split into active (top 3) and reserve pool (rest)
        # Active = highest urgency + prevalence
        # Reserve = sorted by prevalence (common first, rare last)
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
        print(f"{'='*80}\n")
        
        # STEP 2: Generate empathetic opening statement (separate from questions)
        opening_statement = self._generate_opening_statement(chief_complaint)
        
        # STEP 3: Generate age question
        age_question = self._generate_age_question()
        
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
            'status': 'questioning'
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
            
            extract_system = "Extract age number from this answer. Output ONLY the number, nothing else."
            extract_user = f"Question: How old are you?\nAnswer: {user_answer}\n\nExtracted age:"
            
            age_response = self.llm_chat_fn(
                [
                    {"role": "system", "content": extract_system},
                    {"role": "user", "content": extract_user}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            age_result = age_response.strip()
            try:
                self.demographics['age'] = int(age_result)
                print(f"[Engine] 👤 Age: {self.demographics['age']}")
            except ValueError:
                print(f"[Engine] 👤 Age: Could not extract from '{age_result}'")
            
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
            # Extract sex using LLM
            print(f"[Engine] 🔍 Extracting sex from answer: '{user_answer}'")
            
            extract_system = f"Extract biological sex from this answer. Output ONLY 'male' or 'female'."
            extract_user = f"Question: Are you male or female?\nAnswer: {user_answer}\n\nExtracted sex:"
            
            sex_response = self.llm_chat_fn(
                [
                    {"role": "system", "content": extract_system},
                    {"role": "user", "content": extract_user}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            sex_result = sex_response.strip().lower()
            if 'female' in sex_result:
                self.demographics['sex'] = 'female'
            elif 'male' in sex_result:
                self.demographics['sex'] = 'male'
            
            print(f"[Engine] 👤 Sex: {self.demographics.get('sex', 'unknown')}")
            
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
            # Clinical question - VALIDATE answer first
            if not self._is_acceptable_clinical_answer(user_answer):
                print(f"[Engine] ⚠️ Answer too vague or unclear - asking for clarification")
                
                # Find last question and extract CORE question (strip clarification prefix)
                last_q_item = None
                for item in reversed(self.conversation_history):
                    if item.get('type') == 'question':
                        last_q_item = item
                        break
                
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
            
            # Answer is acceptable - proceed with scoring
            return self._process_clinical_answer(user_answer)
    
    def _is_acceptable_clinical_answer(self, answer: str) -> bool:
        """
        LLM-based validation: Does the answer actually address the question asked?
        
        Uses LLM to determine if answer is meaningful/responsive.
        """
        # Get the last question asked
        last_question = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_question = item['question']
                break
        
        if not last_question:
            return True  # No question to validate against
        
        print(f"[Engine] 🔍 Validating answer with LLM...")
        print(f"[Engine]   Q: '{last_question}'")
        print(f"[Engine]   A: '{answer}'")
        
        # Use LLM to validate (fully dynamic - direct analysis)
        system_msg = f"""Does this answer address the question?

Question: "{last_question}"
Answer: "{answer}"

Accept the answer unless it's ONLY filler words (um, uh, oh, hmm) or completely unrelated. Any attempt to answer the question is valid - including yes, no, single words, or short phrases.

Output 'yes' to accept or 'no' to reject."""

        user_msg = "Valid?"
        
        print(f"[Engine] 🧠 VALIDATION PROMPT (FULL):")
        print(f"[Engine]   === SYSTEM ===")
        print(f"{system_msg}")
        print(f"[Engine]   === USER ===")
        print(f"{user_msg}")
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=50,  # Increased to capture potential reasoning
            temperature=0.0
        )
        
        result = response.strip().lower()
        is_valid = 'yes' in result or 'valid' in result
        
        print(f"[Engine] 🤖 RAW LLM RESPONSE: '{response}'")
        print(f"[Engine] 📊 PARSED: '{result}' → {'ACCEPT ✅' if is_valid else 'REJECT ❌'}")
        
        return is_valid
    
    def _match_to_guidelines(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines
        
        Returns:
            List of matched guidelines with initial scores, sorted by relevance
        """
        complaint_lower = complaint.lower()
        matched = []
        
        print(f"\n[Engine] 🔍 MATCHING TO GUIDELINES...")
        
        for name, guideline in self.all_guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            # Check if any trigger matches
            for trigger in triggers:
                if trigger.lower() in complaint_lower:
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
                    print(f"[Engine]   ✓ {name} (trigger: '{trigger}', prevalence: {prevalence}, initial: {initial_score:.0%})")
                    break
        
        # PREVALENCE-FIRST sorting with urgency boost
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
        for i, m in enumerate(matched[:10], 1):  # Show top 10
            urgency = m['data'].get('urgency', 'routine')
            prevalence = m['data'].get('prevalence', 'uncommon')
            print(f"[Engine]   {i}. {m['name']} ({prevalence}, {urgency}, combined: {m['combined_score']:.0%})")
        if len(matched) > 10:
            print(f"[Engine]   ... and {len(matched) - 10} more")
        
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
        
        # Flexible prompts for each OLDCARTS element - natural variation
        if next_element:
            element_prompts = {
                'L': {
                    'system': "Ask where the pain is located. Be conversational and natural.",
                    'user': "Generate question about pain location:"
                },
                'D': {
                    'system': "Ask how long the pain lasts. Be conversational and natural.",
                    'user': "Generate question about pain duration:"
                },
                'C': {
                    'system': "Ask what the pain feels like (sharp, dull, etc). Be conversational and natural.",
                    'user': "Generate question about pain quality:"
                },
                'A': {
                    'system': "Ask what makes the pain worse. Be conversational and natural.",
                    'user': "Generate question about what worsens pain:"
                },
                'R': {
                    'system': "Ask what helps relieve the pain. Be conversational and natural.",
                    'user': "Generate question about what relieves pain:"
                },
                'T': {
                    'system': "Ask about pain timing/pattern. Be conversational and natural.",
                    'user': "Generate question about pain pattern:"
                },
                'S': {
                    'system': "Ask about pain severity. Be conversational and natural.",
                    'user': "Generate question about pain severity:"
                }
            }
            
            prompt_set = element_prompts.get(next_element, {'system': "Ask about symptom.", 'user': "Question:"})
            system_msg = prompt_set['system']
            user_msg = prompt_set['user']
        else:
            system_msg = "Ask about associated symptoms (fever, nausea, vomiting). Be conversational."
            user_msg = "Generate question about associated symptoms:"

        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=20,
                temperature=0.3  # Moderate - natural variation
            )
            
            question = response.strip().strip('"\'')
            
            # AGGRESSIVE CLEANING: Remove any meta-commentary
            # If LLM outputs "I think..." or "Let me...", extract just the question
            question_lines = question.split('\n')
            for line in question_lines:
                line = line.strip()
                # Skip meta lines
                if any(line.startswith(prefix) for prefix in ['I think', 'Let me', 'Here', 'I would', 'The question']):
                    continue
                # If line has a question mark, use it
                if '?' in line:
                    question = line
                    break
            
            # Clean quotes and ensure ends with ?
            question = question.strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            # No hardcoded multi-part detection - trust the LLM output
            
            # Use LLM to check if this is a repeat question
            is_repeat = False
            if len(asked) > 0:
                print(f"[Engine] 🔍 Checking if question is repeat...")
                
                repeat_check_system = f"Is this new question essentially the same as any previously asked question?\n\nNew: {question}\nPreviously asked: {', '.join(asked[-5:])}\n\nOutput 'yes' if repeat, 'no' if different."
                repeat_check_user = "Repeat?"
                
                repeat_response = self.llm_chat_fn(
                    [
                        {"role": "system", "content": repeat_check_system},
                        {"role": "user", "content": repeat_check_user}
                    ],
                    max_tokens=3,
                    temperature=0.0
                )
                
                if 'yes' in repeat_response.lower():
                    print(f"[Engine] ⚠️ Question is repeat - generating alternative...")
                    is_repeat = True
            
            # If repeat, ask LLM to generate a different follow-up question
            if is_repeat:
                print(f"[Engine] 🔄 Generating alternative follow-up question...")
                
                alt_system_msg = f"Patient has {self.chief_complaint}. Generate a different follow-up question about associated symptoms (fever, nausea, vomiting, etc)."
                alt_user_msg = f"Already asked: {', '.join(asked[-3:])}. Generate different question:"
                
                alt_response = self.llm_chat_fn(
                    [
                        {"role": "system", "content": alt_system_msg},
                        {"role": "user", "content": alt_user_msg}
                    ],
                    max_tokens=20,
                    temperature=0.4
                )
                question = alt_response.strip().strip('"\'')
                if not question.endswith('?'):
                    question += '?'
            
            # Detect which OLDCARTS element this question addresses
            oldcarts_element = self._detect_oldcarts_element(question)
            
            print(f"[Engine] ✅ Generated Question: '{question}'")
            if oldcarts_element:
                print(f"[Engine] 📋 OLDCARTS Element: {oldcarts_element}")
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
                'status': 'questioning'
            }
        
        except Exception as e:
            print(f"[Engine] ❌ Question generation failed: {e}")
            raise RuntimeError(f"LLM question generation failed: {e}")
    
    def _detect_oldcarts_element(self, question: str) -> Optional[str]:
        """
        Detect which OLDCARTS element a question addresses
        
        Returns: 'O', 'L', 'D', 'C', 'A', 'R', 'T', or 'S' (or None if unclear)
        """
        # Use LLM to classify question into OLDCARTS category
        print(f"[Engine] 🔍 Detecting OLDCARTS element for: '{question}'")
        
        system_msg = f"""Classify this question into ONE OLDCARTS category:

Question: "{question}"

Categories:
O = Onset/timing (when did it start?)
L = Location (where is pain?)
D = Duration (how long does it last?)
C = Character (what does it feel like? sharp/dull/etc)
A = Aggravating factors (what makes it worse?)
R = Relieving factors (what helps?)
T = Timing pattern (constant or intermittent?)
S = Severity (how bad is it?)

Output ONLY the single letter (O/L/D/C/A/R/T/S)."""
        
        user_msg = "Letter:"
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=2,
                temperature=0.0
            )
            
            result = response.strip().upper()
            
            # Extract just the letter
            if len(result) == 1 and result in 'OLDCARTS':
                print(f"[Engine] ✅ Detected OLDCARTS: {result}")
                return result
            else:
                print(f"[Engine] ⚠️ Could not classify - response: '{result}'")
                return None
        
        except Exception as e:
            print(f"[Engine] ❌ OLDCARTS detection failed: {e}")
            raise RuntimeError(f"OLDCARTS detection failed: {e}")
    
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
        
        # Find the section using regex
        # Pattern: "ELEMENT_NAME: ...text... NEXT_ELEMENT:"
        pattern = f"{element_name}:([^.]*(?:\\.[^A-Z:][^.]*)*)"
        match = re.search(pattern, classic_presentation, re.IGNORECASE)
        
        if match:
            section_text = match.group(1).strip()
            # Clean up - stop at next OLDCARTS element or ASSOCIATED/KEY
            for stop_word in ['ONSET:', 'LOCATION:', 'DURATION:', 'CHARACTER:', 'AGGRAVATING:', 'RELIEVING:', 'TIMING:', 'SEVERITY:', 'ASSOCIATED', 'KEY POSITIVES', 'KEY NEGATIVES']:
                if stop_word in section_text:
                    section_text = section_text.split(stop_word)[0].strip()
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
        
        # FOR EACH GUIDELINE: Score using VECTOR SIMILARITY
        print(f"\n[Engine] 🎯 SEMANTIC SIMILARITY SCORING:\n")
        
        if not oldcarts_element:
            raise RuntimeError(f"Question has no OLDCARTS element assigned - cannot score")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized - cannot compute similarity")
        
        print(f"[Engine] 📊 Matching answer to OLDCARTS element: {oldcarts_element}")
        
        for g in self.active_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            
            # Extract the specific OLDCARTS section for this element
            oldcarts_section = self._extract_oldcarts_section(classic, oldcarts_element)
            
            if not oldcarts_section:
                raise RuntimeError(f"Could not extract {oldcarts_element} section from {g['name']}")
            
            # Compute semantic similarity between answer and guideline's OLDCARTS section
            similarity = self._compute_similarity(answer, oldcarts_section)
            
            # Update score with weighted average (70% old score + 30% new similarity)
            # This prevents wild swings while incorporating new information
            old_score = g['score']
            new_score = (old_score * 0.7) + (similarity * 0.3)
            g['score'] = new_score
            
            change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
            print(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change}")
            print(f"[Engine]     Similarity: {similarity:.2f} to {oldcarts_element} section")
            print(f"[Engine]     Section: {oldcarts_section[:80]}...")
        
        # ROLLING REPLACEMENT: Rule out low-scoring guidelines and promote from reserve
        ruled_out_this_round = []
        for g in list(self.active_guidelines):  # Use list() to avoid modification during iteration
            if g['score'] < self.RULE_OUT_THRESHOLD:
                print(f"[Engine] ❌ RULING OUT: {g['name']} (score {g['score']:.0%} < {self.RULE_OUT_THRESHOLD:.0%})")
                self.active_guidelines.remove(g)
                self.ruled_out.append(g)
                ruled_out_this_round.append(g)
        
        # Promote from reserve to maintain MAX_ACTIVE
        # PRIORITIZE BY PREVALENCE: common conditions before rare ones
        promoted_this_round = []
        while len(self.active_guidelines) < self.MAX_ACTIVE and len(self.reserve_pool) > 0:
            # Sort reserve pool by prevalence (higher score = more common)
            # This ensures common conditions are considered before rare ones
            self.reserve_pool.sort(key=lambda x: x['score'], reverse=True)
            
            # Take the highest-prevalence condition
            next_condition = self.reserve_pool.pop(0)
            prevalence = next_condition['data'].get('prevalence', 'uncommon')
            self.active_guidelines.append(next_condition)
            promoted_this_round.append(next_condition)
            print(f"[Engine] 🔼 PROMOTING: {next_condition['name']} ({prevalence}, score: {next_condition['score']:.0%}) from reserve")
        
        # RE-RANK by score
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            print(f"[Engine]   {i}. {g['name']}: {g['score']:.0%} {urgency_emoji}")
        
        if ruled_out_this_round or promoted_this_round:
            print(f"\n[Engine] 🔄 Pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
        
        print(f"{'='*80}\n")
        
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
        # Use LLM to convert red flag to question
        print(f"[Engine] 🧠 Converting red flag to question...")
        print(f"[Engine]   Red flag: {red_flag}")
        
        system_msg = f"Convert this warning sign into a simple yes/no question for the patient:\n\n'{red_flag}'\n\nMake it conversational and patient-friendly."
        
        user_msg = "Generate yes/no question:"
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=25,
                temperature=0.3
            )
            
            question = response.strip().strip('"\'')
            
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Engine] ✅ Red flag question: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Engine] ❌ Red flag conversion failed: {e}")
            raise RuntimeError(f"Red flag conversion failed: {e}")
    
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
        print(f"[Engine] 🧠 Generating LLM opening statement...")
        
        system_msg = f"Patient says: '{chief_complaint}'. Acknowledge their concern briefly and say you'll ask questions to help."
        
        user_msg = "Generate empathetic response (1-2 sentences):"
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=35,
                temperature=0.4
            )
            
            statement = response.strip().strip('"\'')
            
            if not statement.endswith('.') and not statement.endswith('!'):
                statement += '.'
            
            print(f"[Engine] ✅ Opening: '{statement}'")
            return statement
        
        except Exception as e:
            print(f"[Engine] ❌ Opening generation failed: {e}")
            raise RuntimeError(f"Opening generation failed: {e}")
    
    def _generate_age_question(self) -> str:
        """
        LLM-generated age question (conversational)
        """
        print(f"[Engine] 🧠 Generating LLM age question...")
        
        system_msg = "Ask for patient's age conversationally."
        
        user_msg = "Generate question asking for age:"
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=12,
                temperature=0.3
            )
            
            question = response.strip().strip('"\'')
            
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Engine] ✅ Age question: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Engine] ❌ Age generation failed: {e}")
            raise RuntimeError(f"Age generation failed: {e}")
    
    def _generate_sex_question(self) -> str:
        """
        LLM-generated sex question (conversational)
        """
        print(f"[Engine] 🧠 Generating LLM sex question...")
        
        system_msg = "Ask for patient's biological sex conversationally."
        
        user_msg = "Generate question asking for sex (male/female):"
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=15,
                temperature=0.3
            )
            
            question = response.strip().strip('"\'')
            
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Engine] ✅ Sex question: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Engine] ❌ Sex generation failed: {e}")
            raise RuntimeError(f"Sex generation failed: {e}")
    
    def _generate_clarification_question(self, topic: str) -> str:
        """
        LLM-generated clarification when answer was unclear
        """
        print(f"[Engine] 🧠 Generating clarification for: {topic}")
        
        # ULTRA-SIMPLE: Just use hardcoded templates (LLM keeps deviating)
        if topic == "age":
            question = "I didn't catch that. How old are you?"
            print(f"[Engine] 💬 Using template (age)")
        elif topic == "sex":
            question = "I didn't catch that. Are you male or female?"
            print(f"[Engine] 💬 Using template (sex)")
        else:
            # For other topics, use LLM
            system_msg = f"You are a medical assistant. Politely re-ask about {topic}."
            user_msg = f"Re-ask about {topic}:"
            
            try:
                response = self.llm_chat_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=20,
                    temperature=0.5
                )
                
                question = response.strip().strip('"\'')
            except Exception as e:
                print(f"[Engine] ❌ Clarification generation failed: {e}")
                raise RuntimeError(f"Clarification generation failed: {e}")
        
        # Ensure ends with ?
        if not question.endswith('?'):
            question += '?'
        
        print(f"[Engine] ✅ Clarification: '{question}'")
        return question


# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")
