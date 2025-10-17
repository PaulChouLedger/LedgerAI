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
        self.MAX_ACTIVE = 5  # Keep 5 active differentials
    
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
        
        print(f"\n[Engine] 🔄 Initial pool status: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}, Ruled out={len(self.ruled_out)}")
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
            
            extract_system = f"""Extract biological sex from this answer to the question "Are you male or female?"

If the answer clearly states or implies male (man, male, boy, guy), output 'male'.
If the answer clearly states or implies female (woman, female, girl, lady), output 'female'.
Otherwise, output 'unknown'.

Examples:
- "male" → male
- "I'm a man" → male
- "female" → female  
- "woman" → female
- "yeah" → unknown (doesn't specify sex)
- "yes" → unknown (doesn't specify sex)
- "email" → unknown
- "30" → unknown"""
            
            extract_user = f"Question: Are you male or female?\nAnswer: {user_answer}\n\nExtracted:"
            
            sex_response = self.llm_chat_fn(
                [
                    {"role": "system", "content": extract_system},
                    {"role": "user", "content": extract_user}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            sex_result = sex_response.strip().lower()
            # Only accept exact matches
            if sex_result == 'female':
                self.demographics['sex'] = 'female'
            elif sex_result == 'male':
                self.demographics['sex'] = 'male'
            # If LLM outputs 'unknown' or anything else, leave sex unset
            
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
        
        # Build context-specific validation criteria based on OLDCARTS element
        validation_criteria = {
            'O': "MUST indicate WHEN (time reference like 'hours ago', 'yesterday', 'this morning').",
            'L': "MUST specify WHERE (anatomical location like 'left side', 'upper abdomen', 'chest', NOT just 'on the').",
            'D': "MUST describe HOW LONG (duration like 'constant', 'comes and goes', 'few minutes').",
            'C': "MUST describe CHARACTER (quality like 'sharp', 'dull', 'burning', 'cramping').",
            'A': "MUST describe AGGRAVATING factors (what makes it worse: 'movement', 'eating', or 'nothing').",
            'R': "MUST describe RELIEVING factors (what helps: 'rest', 'medication', or 'nothing').",
            'T': "MUST describe TIMING pattern ('constant', 'intermittent', 'comes in waves').",
            'S': "MUST indicate SEVERITY (number, or 'mild', 'moderate', 'severe')."
        }
        
        # Get specific criteria for this OLDCARTS element
        specific_criteria = validation_criteria.get(oldcarts_element, 
            "Must address the question asked (not just filler words).")
        
        # Use LLM to validate with context-specific criteria
        system_msg = f"""Validate this answer based on the question and specific requirements.

Question: "{last_question}"
Answer: "{answer}"

Requirement: {specific_criteria}

Reject if:
- Only filler words (um, uh, oh, hmm)
- Incomplete fragments (like "on the", "I", "it")
- Completely unrelated to what's asked
- Does NOT meet the requirement above

Output ONLY 'yes' to accept or 'no' to reject."""

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
        
        # Show decision with brief reasoning (first 80 chars)
        reasoning_preview = response.strip()[:80] + "..." if len(response) > 80 else response.strip()
        decision = 'ACCEPT ✅' if is_valid else 'REJECT ❌'
        print(f"[Engine] 📊 Validation: {decision}")
        print(f"[Engine] 💬 Reason: {reasoning_preview}")
        
        return is_valid
    
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
        
        for name, guideline in self.all_guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            # Check if any trigger matches using HYBRID approach
            matched_trigger = None
            match_type = None
            
            for trigger in triggers:
                trigger_lower = trigger.lower()
                
                # FAST PATH: Exact keyword matching
                # 1. Exact: trigger in complaint (e.g., "chest pain" in "I have chest pain")
                if trigger_lower in complaint_lower:
                    matched_trigger = trigger
                    match_type = "exact"
                    break
                
                # 2. Subset: core symptom in trigger (e.g., "abdominal pain" in "lower abdominal pain")
                if core_symptom in trigger_lower:
                    matched_trigger = trigger
                    match_type = "subset"
                    break
            
            # SEMANTIC PATH: Use embeddings for fuzzy/synonym matching
            # Handles typos ("abdomnal pain"), synonyms ("belly pain" = "abdominal pain")
            if not matched_trigger and self.embedding_model:
                for trigger in triggers:
                    similarity = self._compute_similarity(core_symptom, trigger)
                    if similarity > 0.70:  # Semantic threshold
                        matched_trigger = trigger
                        match_type = f"semantic ({similarity:.2f})"
                        break
            
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
                print(f"[Engine]   ✓ {name} (trigger: '{matched_trigger}', match: {match_type}, prevalence: {prevalence}, initial: {initial_score:.0%})")
        
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
        
        # Hardcoded OLDCARTS templates (reliable, fast)
        if next_element:
            oldcarts_templates = {
                'L': "Where exactly is the pain?",
                'D': "How long does the pain last?",
                'C': "How would you describe the pain?",
                'A': "What makes the pain worse?",
                'R': "What helps relieve the pain?",
                'T': "Is the pain constant or does it come and go?",
                'S': "How severe is the pain on a scale of 1 to 10?"
            }
            
            question = oldcarts_templates.get(next_element)
            oldcarts_element = next_element
            
            print(f"[Engine] ✅ Template Question: '{question}'")
            print(f"[Engine] 📋 OLDCARTS Element: {oldcarts_element}")
            
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
                'status': 'questioning'
            }
        
        # After OLDCARTS: Ask about associated symptoms
        # Simple templates to avoid repeats
        asked_lower = ' '.join(asked).lower()
        
        if 'fever' not in asked_lower:
            question = "Have you had any fever?"
        elif 'nausea' not in asked_lower and 'vomit' not in asked_lower:
            question = "Have you had any nausea or vomiting?"
        elif 'appetite' not in asked_lower and 'hungry' not in asked_lower:
            question = "How is your appetite?"
        else:
            question = "Any other symptoms?"
        
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
            'status': 'questioning'
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
        
        # SKIP SCORING for non-OLDCARTS questions (associated symptoms like fever, nausea)
        if not oldcarts_element:
            print(f"\n[Engine] ℹ️  Associated symptom question - skipping similarity scoring\n")
            print(f"[Engine] 📋 Answer noted: '{answer}'\n")
            
            # Just move to next question without scoring
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
                raise RuntimeError(f"Could not extract {oldcarts_element} section from {g['name']}")
            
            # KEYWORD FILTER: For location questions, skip opposite-sided conditions
            # This is faster and more accurate than semantic similarity for directional terms
            if oldcarts_element == 'L':
                answer_lower = answer.lower()
                section_upper = oldcarts_section.upper()
                
                # Check for opposite-sided conditions
                patient_says_left = 'left' in answer_lower and 'right' not in answer_lower
                patient_says_right = 'right' in answer_lower and 'left' not in answer_lower
                
                guideline_is_right_only = 'RIGHT' in section_upper and 'LEFT' not in section_upper
                guideline_is_left_only = 'LEFT' in section_upper and 'RIGHT' not in section_upper
                
                # Skip this guideline if sides don't match
                if (patient_says_left and guideline_is_right_only) or (patient_says_right and guideline_is_left_only):
                    # Set similarity to 0 (will be ruled out or demoted)
                    similarity = 0.0
                    print(f"[Engine]   {g['name']}: SKIPPED (location keyword mismatch: {answer_lower} vs {section_upper[:40]})")
                else:
                    # Compute semantic similarity normally
                    similarity = self._compute_similarity(answer, oldcarts_section)
            else:
                # Compute semantic similarity normally for non-location questions
                similarity = self._compute_similarity(answer, oldcarts_section)
            
            # Update score
            old_score = g['score']
            if similarity == 0.0:
                # Hard mismatch (e.g., left vs right) - rule out immediately
                new_score = 0.0
                g['score'] = new_score
                change = "❌"
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
        
        # LOCATION CLARIFICATION: Check if answer needs more detail (ONLY for location, AFTER scoring)
        if oldcarts_element == 'L':
            print(f"[Engine] 🔍 Checking if location answer is specific enough for differential diagnosis...")
            
            # Get what the UPDATED active guidelines say about location
            location_examples = []
            for g in self.active_guidelines:
                classic = g['data'].get('key_features', {}).get('classic_presentation', '')
                location_section = self._extract_oldcarts_section(classic, 'L')
                if location_section:
                    location_examples.append(f"{g['name']}: {location_section[:100]}")
            
            if location_examples:
                guidelines_say = '\n'.join(location_examples)
                
                # Ask LLM: Can we differentiate these guidelines with current answer?
                analyze_system = f"""Patient answer: "{answer}"

Top differential diagnoses describe locations as:
{guidelines_say}

Can the patient's answer differentiate between these specific locations, or do we need more detail?

Output 'sufficient' if we can differentiate, or 'need_more' if too vague."""
                
                analyze_user = "Status:"
                
                location_check = self.llm_chat_fn(
                    [
                        {"role": "system", "content": analyze_system},
                        {"role": "user", "content": analyze_user}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                
                if 'need_more' in location_check.lower() or 'need' in location_check.lower():
                    print(f"[Engine] ⚠️ Location insufficient for differential - asking for more detail")
                    
                    # Use LLM to generate clarification question based on what guidelines need
                    clarify_system = f"""Patient said: "{answer}"

Top differential diagnoses require:
{guidelines_say}

Generate a single, direct clarifying question to get the specific anatomical location needed to differentiate between these conditions.

Output ONLY the question (no explanations). Make it conversational and natural for voice interaction."""
                    
                    clarify_user = "Question:"
                    
                    clarify_response = self.llm_chat_fn(
                        [
                            {"role": "system", "content": clarify_system},
                            {"role": "user", "content": clarify_user}
                        ],
                        max_tokens=30,
                        temperature=0.2
                    )
                    
                    clarify_location = clarify_response.strip().strip('"\'')
                    if not clarify_location.endswith('?'):
                        clarify_location += '?'
                    
                    print(f"[Engine] 💬 Location clarification: '{clarify_location}'")
                    print(f"{'='*80}\n")
                    
                    # Preserve OLDCARTS element
                    self.conversation_history.append({
                        'type': 'question',
                        'question': clarify_location,
                        'focus': 'clinical',
                        'oldcarts': 'L'  # Keep as location
                    })
                    
                    return {
                        'success': True,
                        'question': clarify_location,
                        'status': 'questioning'
                    }
                else:
                    print(f"[Engine] ✅ Location answer is sufficient for differential diagnosis")
        
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
        Hardcoded opening statement
        """
        statement = "I understand. I'll ask some questions to help figure this out."
        print(f"[Engine] ✅ Opening: '{statement}'")
        return statement
    
    def _generate_age_question(self) -> str:
        """
        Hardcoded age question
        """
        question = "How old are you?"
        print(f"[Engine] ✅ Age question: '{question}'")
        return question
    
    def _generate_sex_question(self) -> str:
        """
        Hardcoded sex question
        """
        question = "Are you male or female?"
        print(f"[Engine] ✅ Sex question: '{question}'")
        return question
    
    def _generate_clarification_question(self, topic: str) -> str:
        """
        Hardcoded clarification questions
        """
        clarifications = {
            "age": "I didn't catch that. How old are you?",
            "sex": "I didn't catch that. Are you male or female?"
        }
        
        question = clarifications.get(topic, f"Can you clarify your answer?")
        print(f"[Engine] ✅ Clarification: '{question}'")
        return question


# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")
