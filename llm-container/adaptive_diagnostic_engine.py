#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - LLM-Driven Medical Diagnosis

SIMPLIFIED APPROACH:
1. Chief complaint → Match relevant guidelines
2. Sort by URGENCY (emergent > urgent > routine) then PREVALENCE (common > rare)
3. Top 3 become active differentials, rest go to reserve pool
4. Feed all 3 guidelines' classical presentations to LLM
5. LLM analyzes and develops question roadmap
6. Ask question → LLM scores all 3 → Re-rank by score
7. Rule out <30% → Promote from reserve (prioritize COMMON conditions)
8. Repeat until diagnosis clear

PREVALENCE-BASED ROLLING DIFFERENTIAL:
- Start with common conditions (appendicitis, cholecystitis, UTI, etc.)
- Only consider rare conditions (ectopic, mesenteric ischemia) after common ones ruled out
- Mimics clinical reasoning: "Common things are common"
- Reserve pool sorted by prevalence ensures common conditions promoted first

NO complex feature extraction, NO pattern matching
LLM does ALL the reasoning - we just provide structure
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class AdaptiveDiagnosticEngine:
    """
    LLM-driven diagnostic engine
    
    The LLM is the intelligence - it reads guidelines and reasons about diagnosis.
    We provide structure and keep it focused.
    """
    
    def __init__(self, guidelines_dir: str = "/app/medical/guidelines", llm_chat_fn=None):
        """
        Initialize diagnostic engine
        
        Args:
            guidelines_dir: Path to JSON guidelines
            llm_chat_fn: LLM function for reasoning
        """
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # Current assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files"""
        print(f"\n{'='*80}")
        print(f"[Engine] 📚 LOADING MEDICAL GUIDELINES")
        print(f"{'='*80}")
        
        if not self.guidelines_dir.exists():
            print(f"[Engine] ❌ Directory not found: {self.guidelines_dir}")
            return
        
        for json_file in sorted(self.guidelines_dir.glob("*.json")):
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    name = guideline.get('condition', json_file.stem)
                    self.all_guidelines[name] = guideline
                    print(f"[Engine]   ✓ {name}")
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
        
        # STEP 2: Use LLM to generate empathetic opening + age question
        opening_question = self._generate_opening_question(chief_complaint)
        
        self.conversation_history.append({
            'type': 'question',
            'question': opening_question,
            'focus': 'age'
        })
        
        return {
            'success': True,
            'question': opening_question,
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
            answer_lower = user_answer.lower()
            is_yes = any(word in answer_lower for word in ['yes', 'yeah', 'yep', 'yup', 'sure'])
            
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
            # Extract age
            age_match = re.search(r'\d+', user_answer)
            if age_match:
                self.demographics['age'] = int(age_match.group())
                print(f"[Engine] 👤 Age: {self.demographics['age']}")
            else:
                print(f"[Engine] 👤 Age: Not found in answer")
            
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
            # Extract sex
            text_lower = user_answer.lower()
            if 'female' in text_lower or 'woman' in text_lower:
                self.demographics['sex'] = 'female'
            elif 'male' in text_lower or 'man' in text_lower:
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
            
            print(f"[Engine] 💬 First question: CHRONICITY (when started)")
            
            self.conversation_history.append({
                'type': 'question',
                'question': timing_question,
                'focus': 'clinical'
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
                last_q = None
                for item in reversed(self.conversation_history):
                    if item.get('type') == 'question':
                        last_q = item.get('question', 'the question')
                        break
                
                # Strip "I didn't quite understand. " prefix if present (avoid repetition)
                core_question = last_q
                if last_q and last_q.startswith("I didn't quite understand. "):
                    core_question = last_q.replace("I didn't quite understand. ", "").strip()
                if core_question and core_question.startswith("Could you be more specific? "):
                    core_question = core_question.replace("Could you be more specific? ", "").strip()
                
                clarify = f"I didn't quite understand. {core_question if core_question else 'Can you clarify?'}"
                
                self.conversation_history.append({
                    'type': 'question',
                    'question': clarify,
                    'focus': 'clinical'
                })
                
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
        
        print(f"[Engine] 🔍 Validating answer...")
        print(f"[Engine]   Q: '{last_question}'")
        print(f"[Engine]   A: '{answer}'")
        
        # PROGRAMMATIC VALIDATION for simple cases (avoid LLM for obvious answers)
        answer_lower = answer.lower().strip()
        
        # Accept common yes/no variations
        yes_no_words = ['yes', 'no', 'yeah', 'nope', 'nah', 'yep', 'sure', 'maybe', 'sometimes', 'not really']
        if any(word in answer_lower for word in yes_no_words):
            print(f"[Engine]   ✓ Programmatic validation: ACCEPT (yes/no response)")
            return True
        
        # Accept time-related words
        time_words = ['today', 'yesterday', 'hour', 'day', 'week', 'month', 'year', 'ago', 'morning', 'night']
        if any(word in answer_lower for word in time_words):
            print(f"[Engine]   ✓ Programmatic validation: ACCEPT (time reference)")
            return True
        
        # Accept location words
        location_words = ['right', 'left', 'upper', 'lower', 'middle', 'center', 'side', 'top', 'bottom']
        if any(word in answer_lower for word in location_words):
            print(f"[Engine]   ✓ Programmatic validation: ACCEPT (location)")
            return True
        
        # Accept numbers (age, severity, etc.)
        if any(char.isdigit() for char in answer):
            print(f"[Engine]   ✓ Programmatic validation: ACCEPT (contains number)")
            return True
        
        # Reject single vague words
        vague_single_words = ['oh', 'um', 'uh', 'well', 'so', 'now', 'then', 'just']
        if answer_lower in vague_single_words:
            print(f"[Engine]   ✗ Programmatic validation: REJECT (vague single word)")
            return False
        
        # For everything else, use LLM validation
        print(f"[Engine]   → Using LLM validation...")
        
        # Use LLM to validate
        system_msg = "You are a medical validator. Does the answer provide requested information? Output ONLY 'yes' or 'no'."
        
        user_msg = f"""Q: {last_question}
A: {answer}

Valid answers include:
- Yes/no responses
- Time words (today, yesterday, hour, day, week, ago)
- Location words (right, left, upper, lower, side)
- Symptom words (fever, nausea, vomiting, sharp, dull)

Invalid answers:
- Single vague words (now, oh, so, up, well)
- Non-responsive words

Is "{answer}" a valid response?

Output (yes/no):"""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=3,
                temperature=0.0
            )
            
            result = response.strip().lower()
            is_valid = 'yes' in result or 'valid' in result
            
            print(f"[Engine]   LLM validation: '{result}' → {'ACCEPT' if is_valid else 'REJECT'}")
            
            return is_valid
        
        except Exception as e:
            print(f"[Engine] ⚠️ Validation failed: {e} - accepting by default")
            return True  # On error, be permissive
    
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
        
        # Build context for LLM
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # Get ONLY classical presentations (minimal context for LLM)
        # Full guideline with red flags sent ONLY at final diagnosis
        guidelines_context = []
        for i, g in enumerate(self.active_guidelines, 1):
            classic = g['data'].get('key_features', {}).get('classic_presentation', 'N/A')
            urgency = g['data'].get('urgency', 'routine')
            
            guidelines_context.append(f"""
Guideline {i}: {g['name']} (Current Score: {g['score']:.0%}, Urgency: {urgency})
Classic Presentation: {classic}
""")
        
        guidelines_text = "\n".join(guidelines_context)
        
        # Get questions already asked
        asked = []
        for item in self.conversation_history:
            if item['type'] == 'question' and item.get('focus') not in ['age', 'sex']:
                asked.append(item['question'])
        
        asked_text = "\n".join([f"- {q}" for q in asked]) if asked else "None yet"
        
        print(f"[Engine] 📋 Patient: {patient_info}")
        print(f"[Engine] 📋 Guidelines in context: {len(self.active_guidelines)}")
        print(f"[Engine] 📋 Questions asked: {len(asked)}")
        
        # LLM PROMPT: Generate next question
        # Read classical presentations and extract KEY DISCRIMINATING FEATURE
        system_msg = "You are a diagnostic AI. Ask ONE simple question about ONE symptom only. Never combine questions with 'and'. Output ONLY the question."
        
        # Extract KEY features from each guideline (CAPS words indicate important differentiators)
        key_features = []
        for g in self.active_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            # Look for CAPITALIZED keywords (these are the discriminators)
            caps_words = re.findall(r'\b[A-Z]{3,}[A-Z\s]*', classic)
            if caps_words:
                key_features.append(f"{g['name']}: {', '.join(caps_words[:3])}")
        
        features_text = '\n'.join(key_features) if key_features else "Location, Timing, Quality"
        
        print(f"[Engine] 📋 Extracted key features:")
        print(f"{features_text}")
        
        # Build list of already asked questions to avoid repeats
        asked_list = "\n".join([f"- {q}" for q in asked]) if asked else "None"
        
        user_msg = f"""Patient: {patient_info}

Key features to ask about:
{features_text}

Already asked:
{asked_list}

Generate ONE specific medical question based on the key features above.
Ask about ONE thing only (location/migration/timing/quality/fever/nausea/vomiting/triggers).
DO NOT combine questions with "and".

Examples:
- "Where exactly is the pain located?"
- "Did the pain migrate from one place to another?"
- "When did the pain start?"
- "Have you had any fever?"
- "Have you had nausea or vomiting?"
- "Does eating make the pain worse?"
- "Is the pain constant or does it come and go?"

Question:"""

        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=25,  # Just the question
                temperature=0.3
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
            
            # VALIDATION: Reject multi-part questions (containing "and")
            if ' and ' in question.lower() or ' or ' in question.lower():
                # Extract just the first part before "and/or"
                if ' and ' in question.lower():
                    question = question.split(' and ')[0] + '?'
                    print(f"[Engine] ⚠️ Multi-part question detected - using first part only")
                elif ' or ' in question.lower() and question.count('?') > 1:
                    question = question.split('?')[0] + '?'
                    print(f"[Engine] ⚠️ Multi-part question detected - using first part only")
            
            # VALIDATION: Check if we already asked this (or very similar)
            is_repeat = False
            for prev_q in asked:
                # Simple similarity check - if >60% of words overlap, it's a repeat
                q_words = set(question.lower().split())
                prev_words = set(prev_q.lower().split())
                overlap = len(q_words & prev_words) / len(q_words) if q_words else 0
                
                if overlap > 0.6:
                    print(f"[Engine] ⚠️ Question too similar to already asked: '{prev_q}'")
                    print(f"[Engine] ⚠️ Asking about different symptom...")
                    is_repeat = True
                    break
            
            # If repeat, ask about a different symptom from the list
            if is_repeat:
                # Check what we haven't asked about yet
                asked_lower = ' '.join(asked).lower()
                if 'fever' not in asked_lower:
                    question = "Have you had any fever?"
                elif 'nausea' not in asked_lower and 'vomit' not in asked_lower:
                    question = "Have you had any nausea or vomiting?"
                elif 'eating' not in asked_lower and 'food' not in asked_lower:
                    question = "Does eating make the pain worse?"
                elif 'move' not in asked_lower and 'movement' not in asked_lower:
                    question = "Does movement make the pain worse?"
                else:
                    question = "How would you describe the pain?"
            
            print(f"[Engine] ✅ Generated Question: '{question}'")
            print(f"{'='*80}\n")
            
            # Store question
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'clinical'
            })
            
            return {
                'success': True,
                'question': question,
                'status': 'questioning'
            }
        
        except Exception as e:
            print(f"[Engine] ❌ Question generation failed: {e}")
            raise RuntimeError(f"LLM question generation failed: {e}")
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """
        Use LLM to score all 5 guidelines based on the answer
        
        This is the CORE diagnostic reasoning.
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
        
        # Build patient summary for scoring
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # FOR EACH GUIDELINE: Ask LLM to score it
        print(f"\n[Engine] 🎯 SCORING EACH GUIDELINE:\n")
        
        for g in self.active_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', 'N/A')
            
            # Scoring prompt - ULTRA-STRICT: System + user roles, number only
            system_msg = "You are a diagnostic scoring AI. Output ONLY integers 0-100. No explanations. Lower scores for 'no' answers to key features."
            
            user_msg = f"""{g['name']}:
{classic}

Patient: {patient_info}
Question: {last_q}
Answer: {answer}

If answer is "no" to a key feature, give LOW score.
If answer matches classic presentation, give HIGH score.

Score 0-100:"""

            try:
                response = self.llm_chat_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=3,  # Just need "50" or "75"
                    temperature=0.0
                )
                
                # Extract score
                score_text = response.strip()
                score_match = re.search(r'\d+', score_text)
                
                if score_match:
                    new_score = int(score_match.group()) / 100.0  # Convert to 0-1
                    old_score = g['score']
                    g['score'] = new_score
                    
                    change = "↑" if new_score > old_score else "↓" if new_score < old_score else "="
                    print(f"[Engine]   {g['name']}: {old_score:.0%} → {new_score:.0%} {change}")
                    print(f"[Engine]     LLM returned: '{score_text}'")
                else:
                    print(f"[Engine]   {g['name']}: ⚠️ Could not parse score from LLM response: '{score_text}'")
            
            except Exception as e:
                print(f"[Engine] ⚠️ Scoring failed for {g['name']}: {e}")
        
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
        
        # Diagnosis criteria: High confidence OR asked enough questions
        if top['score'] >= 0.90 and num_questions >= 7:
            print(f"[Engine] ✅ DIAGNOSIS REACHED: {top['name']} ({top['score']:.0%} confidence)")
            print(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        elif num_questions >= 12:
            print(f"[Engine] ✅ DIAGNOSIS BY QUESTIONS LIMIT: {top['name']} ({top['score']:.0%} confidence)")
            print(f"[Engine] 🚩 Starting RED FLAG screening...")
            return self._screen_red_flags(top)
        else:
            print(f"[Engine] 🔄 Continuing (Q{num_questions}, top score: {top['score']:.0%})")
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
        # Simple conversion: extract key symptom and make it a question
        # These are meant to be simple, direct yes/no questions
        
        lower = red_flag.lower()
        
        # Common patterns
        if 'fever' in lower and '103' in lower:
            return "Have you had a fever higher than 103 degrees?"
        elif 'fever' in lower:
            return "Have you had any fever or chills?"
        elif 'pain that then improves' in lower or 'sudden' in lower and 'better' in lower:
            return "Did the pain suddenly get much better after being severe?"
        elif 'rigid' in lower or 'board-like' in lower:
            return "Is your abdomen very hard or rigid when you press on it?"
        elif 'hypotension' in lower or 'dizzy' in lower or 'faint' in lower:
            return "Have you felt dizzy, lightheaded, or like you might faint?"
        elif 'tachycardia' in lower or 'heart' in lower:
            return "Is your heart racing or beating very fast?"
        elif 'altered mental' in lower or 'confusion' in lower:
            return "Have you felt confused or had trouble thinking clearly?"
        elif 'blood' in lower and ('stool' in lower or 'diarrhea' in lower):
            return "Have you seen any blood in your stool or diarrhea?"
        elif 'blood' in lower and 'vomit' in lower:
            return "Have you vomited any blood?"
        elif 'jaundice' in lower or 'yellow' in lower:
            return "Have your eyes or skin turned yellow?"
        elif 'unable to pass gas' in lower or 'no bowel movement' in lower:
            return "Have you been unable to pass gas or have a bowel movement?"
        else:
            # Generic fallback: Ask if they're experiencing the symptom
            return f"Are you experiencing: {red_flag.split('-')[0].strip()}?"
    
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
    
    def _generate_opening_question(self, chief_complaint: str) -> str:
        """
        LLM-generated opening: Show empathy and ask for age
        """
        print(f"[Engine] 🧠 Generating LLM opening question...")
        
        system_msg = "You are a medical assistant. Your task: show brief empathy, then ask 'How old are you?'"
        
        user_msg = f"""Patient says: "{chief_complaint}"

Output format: "[Empathy]. How old are you?"
Example: "I'm sorry to hear that. How old are you?"

Your response:"""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=25,
                temperature=0.6
            )
            
            question = response.strip().strip('"\'')
            
            # Ensure it ends with the age question
            if 'how old' not in question.lower():
                # Force it if LLM didn't follow instructions
                question = f"I understand. How old are you?"
            
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Engine] ✅ Opening: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Engine] ❌ Opening generation failed: {e}")
            raise RuntimeError(f"Opening question generation failed: {e}")
    
    def _generate_sex_question(self) -> str:
        """
        LLM-generated sex question
        """
        print(f"[Engine] 🧠 Generating LLM sex question...")
        
        system_msg = "You are a medical assistant. Ask for biological sex (male/female). Output ONE question only."
        
        user_msg = """Your question:"""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=15,
                temperature=0.4
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
            print(f"[Engine] ✅ Sex question: '{question}'")
            return question
        
        except Exception as e:
            print(f"[Engine] ❌ Sex question generation failed: {e}")
            raise RuntimeError(f"Sex question generation failed: {e}")
    
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
