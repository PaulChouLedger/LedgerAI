#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - LLM-Driven Medical Diagnosis

SIMPLIFIED APPROACH:
1. Chief complaint → Match 5 most relevant guidelines
2. Feed all 5 guidelines' classical presentations to LLM
3. LLM analyzes and develops question roadmap
4. Ask question → LLM scores all 5 → Re-rank by score
5. Repeat until diagnosis clear

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
        self.active_guidelines = []  # The 5 chosen guidelines with scores
        self.chief_complaint = ""
        self.demographics = {}  # age, sex
        self.conversation_history = []  # All Q&A
        self.status = "idle"  # idle, questioning, diagnosed
    
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
        
        # Take top 5 (or fewer if less than 5 matched)
        self.active_guidelines = matched[:5]
        
        print(f"\n[Engine] 📋 SELECTED 5 GUIDELINES:")
        for i, g in enumerate(self.active_guidelines, 1):
            print(f"[Engine]   {i}. {g['name']} (initial score: {g['score']:.2f})")
        print(f"{'='*80}\n")
        
        # STEP 2: Ask demographics first (age/sex) - simple templates
        age_question = "How old are you?"
        
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
    
    def process_answer(self, user_answer: str) -> Dict[str, Any]:
        """
        Process answer and continue assessment
        
        Args:
            user_answer: User's response
        
        Returns:
            Next question or diagnosis
        """
        if self.status != "questioning":
            return {'success': False, 'message': "No active assessment"}
        
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
        
        # Handle demographics
        if last_q.get('focus') == 'age':
            # Extract age
            age_match = re.search(r'\d+', user_answer)
            if age_match:
                self.demographics['age'] = int(age_match.group())
                print(f"[Engine] 👤 Age: {self.demographics['age']}")
            
            # Ask sex
            sex_question = "Are you male or female?"
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
            print(f"{'='*80}\n")
            
            # NOW: Feed guidelines to LLM and get first clinical question
            return self._ask_next_clinical_question()
        
        else:
            # Clinical question - use LLM to score and ask next
            return self._process_clinical_answer(user_answer)
    
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
                    matched.append({
                        'name': name,
                        'score': 0.5,  # Initial score (neutral)
                        'data': guideline
                    })
                    print(f"[Engine]   ✓ {name} (trigger: '{trigger}')")
                    break
        
        # Sort by name for now (can add better scoring later)
        matched.sort(key=lambda x: x['name'])
        
        return matched
    
    def _ask_next_clinical_question(self) -> Dict[str, Any]:
        """
        Use LLM to analyze all 5 guidelines and generate next best question
        
        This is the CORE intelligence of the system.
        """
        print(f"\n{'='*80}")
        print(f"[Engine] 🧠 LLM QUESTION GENERATION")
        print(f"{'='*80}")
        
        # Build context for LLM
        patient_info = f"{self.demographics.get('age', '?')} year old {self.demographics.get('sex', '?')} with {self.chief_complaint}"
        
        # Get classical presentations from all 5 guidelines
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
        prompt = f"""You are a diagnostic AI analyzing these 5 possible conditions:

{guidelines_text}

Patient: {patient_info}

Questions already asked:
{asked_text}

Your task: Generate ONE key question that will best differentiate between these 5 conditions.
- Focus on symptoms from the classical presentations
- Ask about ONE specific symptom at a time
- Make it conversational and clear
- Prioritize the most discriminating questions first

Output ONLY the question (no explanation):"""

        try:
            response = self.llm_chat_fn(
                [{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            question = response.strip().strip('"\'')
            if not question.endswith('?'):
                question += '?'
            
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
        
        # FOR EACH GUIDELINE: Ask LLM to score it
        print(f"\n[Engine] 🎯 SCORING EACH GUIDELINE:\n")
        
        for g in self.active_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', 'N/A')
            
            # Scoring prompt
            scoring_prompt = f"""You are scoring how well a patient matches this condition:

Condition: {g['name']}
Classic Presentation: {classic}

Patient History:
{history_text}

Latest Q&A:
Q: {last_q}
A: {answer}

Based on ALL the information, rate how likely this patient has {g['name']}.

Output ONLY a score from 0-100 (integer only, no explanation):"""

            try:
                response = self.llm_chat_fn(
                    [{"role": "user", "content": scoring_prompt}],
                    max_tokens=5,
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
                else:
                    print(f"[Engine]   {g['name']}: Could not parse score from '{score_text}'")
            
            except Exception as e:
                print(f"[Engine] ⚠️ Scoring failed for {g['name']}: {e}")
        
        # RE-RANK by score
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n[Engine] 📊 UPDATED RANKINGS:")
        for i, g in enumerate(self.active_guidelines, 1):
            urgency_emoji = "🚨" if g['data'].get('urgency') == 'emergent' else "⚠️" if g['data'].get('urgency') == 'urgent' else "📋"
            print(f"[Engine]   {i}. {g['name']}: {g['score']:.0%} {urgency_emoji}")
        
        print(f"{'='*80}\n")
        
        # CHECK FOR DIAGNOSIS
        top = self.active_guidelines[0]
        num_questions = len([item for item in self.conversation_history if item['type'] == 'question' and item.get('focus') == 'clinical'])
        
        # Diagnosis criteria: High confidence OR asked enough questions
        if top['score'] >= 0.80 and num_questions >= 3:
            print(f"[Engine] ✅ DIAGNOSIS REACHED: {top['name']} ({top['score']:.0%} confidence)")
            return self._finalize_diagnosis(top)
        elif num_questions >= 8:
            print(f"[Engine] ✅ DIAGNOSIS BY QUESTIONS LIMIT: {top['name']} ({top['score']:.0%} confidence)")
            return self._finalize_diagnosis(top)
        else:
            print(f"[Engine] 🔄 Continuing (Q{num_questions}, top score: {top['score']:.0%})")
            # Ask next question
            return self._ask_next_clinical_question()
    
    def _finalize_diagnosis(self, diagnosis_obj: Dict) -> Dict[str, Any]:
        """
        Finalize and return diagnosis
        """
        self.status = "diagnosed"
        
        name = diagnosis_obj['name']
        score = diagnosis_obj['score']
        urgency = diagnosis_obj['data'].get('urgency', 'routine')
        
        urgency_messages = {
            'emergent': '🚨 This is a medical emergency. Call 911 or go to the ER immediately.',
            'urgent': '⚠️ This requires prompt medical attention. Go to urgent care or ER today.',
            'routine': '📋 Schedule an appointment with your doctor soon.'
        }
        
        urgency_msg = urgency_messages.get(urgency, urgency_messages['routine'])
        
        message = f"Based on your symptoms, this is most likely {name} (confidence: {score:.0%}).\n\n{urgency_msg}"
        
        print(f"\n{'='*80}")
        print(f"[Engine] 🎯 FINAL DIAGNOSIS")
        print(f"{'='*80}")
        print(f"[Engine] Condition: {name}")
        print(f"[Engine] Confidence: {score:.0%}")
        print(f"[Engine] Urgency: {urgency}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'status': 'diagnosed',
            'diagnosis': name,
            'confidence': score,
            'urgency': urgency,
            'message': message
        }


# Test
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    print(f"\nEngine initialized with {len(engine.all_guidelines)} guidelines")
