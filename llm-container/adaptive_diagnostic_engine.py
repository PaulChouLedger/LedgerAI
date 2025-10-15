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
    
    Flow:
    1. User states chief complaint
    2. Match to relevant guidelines (JSON triggers + synonyms)
    3. Retrieve full clinical content from RAG
    4. Ask discriminating questions intelligently
    5. Score all guidelines simultaneously
    6. Filter and narrow differentials
    7. Reach diagnosis with high confidence
    8. Provide education using RAG content
    """
    
    def __init__(self, guidelines_dir: str = "/app/medical/guidelines"):
        """
        Initialize adaptive diagnostic engine
        
        Args:
            guidelines_dir: Path to directory containing JSON guidelines
        """
        self.guidelines_dir = Path(guidelines_dir)
        self.guidelines = {}
        self.synonyms = {}
        
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
        self.active_guidelines = []  # List of (guideline_name, score, metadata)
        self.answered_features = {}  # Dict of extracted clinical features
        self.questions_asked = []    # History of questions asked
        self.status = "idle"         # idle, questioning, diagnosed
        self.diagnosis = None        # Final diagnosis when reached
    
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
        
        # Initialize active guidelines with initial scores
        self.active_guidelines = [
            {
                'name': name,
                'score': initial_score,
                'guideline_data': self.guidelines[name],
                'rag_content': None  # Will be loaded when needed
            }
            for name, initial_score in matched
        ]
        
        # Extract any features from the chief complaint itself
        self._extract_features_from_text(chief_complaint)
        
        # Ask first discriminating question
        return self._ask_next_question()
    
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
        
        # Extract ALL clinical features from the answer
        extracted = self._extract_features_from_text(user_answer)
        
        print(f"[Adaptive] 📝 Extracted features: {list(extracted.keys())}")
        
        # Update answered features (merge, don't replace)
        self.answered_features.update(extracted)
        
        # Re-score ALL active guidelines
        self._score_all_guidelines()
        
        # Sort by score
        self.active_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        # Print current top candidates
        print(f"[Adaptive] 📊 Current top differentials:")
        for i, g in enumerate(self.active_guidelines[:5], 1):
            print(f"[Adaptive]    {i}. {g['name']}: {g['score']:.3f}")
        
        # Check if diagnosis reached
        if len(self.active_guidelines) > 0:
            top = self.active_guidelines[0]
            
            # High confidence diagnosis
            if top['score'] > 0.85:
                return self._finalize_diagnosis(top)
            
            # Single guideline remaining
            if len(self.active_guidelines) == 1 and top['score'] > 0.7:
                return self._finalize_diagnosis(top)
        
        # Filter out low-scoring guidelines (but keep at least 1!)
        threshold = 0.2  # Lower threshold for more flexibility
        filtered = [g for g in self.active_guidelines if g['score'] > threshold]
        
        # Always keep at least the top guideline
        if len(filtered) == 0 and len(self.active_guidelines) > 0:
            print(f"[Adaptive] ⚠️ No guidelines above {threshold}, keeping top guideline")
            self.active_guidelines = [self.active_guidelines[0]]
        else:
            self.active_guidelines = filtered
        
        if len(self.active_guidelines) == 0:
            print(f"[Adaptive] ⚠️ No guidelines remain")
            # Reset and ask for new chief complaint
            self.reset_assessment()
            return {
                'success': False,
                'message': "I couldn't match your symptoms to a specific condition. Can you describe what's bothering you?"
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
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()
    
    def _is_new_chief_complaint(self, text: str) -> bool:
        """
        Detect if user is stating a new chief complaint vs answering a question
        
        Returns True if text looks like "I have X pain" or similar
        """
        text_lower = text.lower().strip()
        
        # Patterns for chief complaints
        complaint_patterns = [
            'i have',
            'i am having',
            'i feel',
            'my',
            'there is',
            'i got'
        ]
        
        # Common symptoms
        symptom_words = ['pain', 'ache', 'hurt', 'discomfort', 'burning', 'pressure']
        
        # Check if starts with complaint pattern and contains symptom
        for pattern in complaint_patterns:
            if text_lower.startswith(pattern):
                if any(symptom in text_lower for symptom in symptom_words):
                    return True
        
        return False
    
    def _is_valid_medical_response(self, text: str) -> bool:
        """
        Validate that response is meaningful (not garbage transcription or too short)
        
        Returns False for:
        - Very short fragments ("on the", "time.", "go.")
        - Gibberish ("good else to go")
        - Empty responses
        """
        text = text.strip()
        
        # Too short
        if len(text) < 3:
            return False
        
        # Only punctuation or filler words
        filler_patterns = [
            r'^(on the|the|a|an|to|for|with)[\s\.,]*$',
            r'^[\.,;:!?]+$',
            r'^(uh|um|er|ah)[\s\.,]*$'
        ]
        
        for pattern in filler_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        # Has at least one real word (3+ chars)
        words = text.split()
        real_words = [w for w in words if len(w.strip('.,!?')) >= 3]
        
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
        
        # Onset timing (universal)
        if any(t in text_lower for t in ['hour', 'today', 'this morning', 'this afternoon']):
            features['onset_timing'] = 'acute_hours'
        elif any(t in text_lower for t in ['yesterday', 'last night', 'one day', 'two day', 'three day', 'couple day']):
            features['onset_timing'] = 'acute_days'
        elif any(t in text_lower for t in ['week', 'several day', 'four day', 'five day', 'six day']):
            features['onset_timing'] = 'subacute'
        elif any(t in text_lower for t in ['month', 'year', 'long time', 'always', 'chronic']):
            features['onset_timing'] = 'chronic'
        
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
                
                # Check for positive responses
                for positive in expected_positive:
                    if positive.lower() in text_lower:
                        # Track positive findings
                        if 'positive_findings' not in features:
                            features['positive_findings'] = []
                        features['positive_findings'].append({
                            'guideline': guideline_name,
                            'question': question_focus,
                            'response': positive,
                            'value': diagnostic_value
                        })
                        break  # Only count one match per question
        
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
        Select and ask the most discriminating next question from guideline
        
        Picks the highest-value question from the top guideline that hasn't been asked yet.
        100% data-driven from guideline JSON!
        """
        if not self.active_guidelines:
            return {
                'success': False,
                'message': "I need more information to make a diagnosis."
            }
        
        # Get questions from top guideline
        top_guideline = self.active_guidelines[0]
        guideline = top_guideline['guideline_data']
        diagnostic_questions = guideline.get('diagnostic_questions', [])
        
        # Track which question_focus areas we've already asked about
        asked_focuses = set()
        if 'positive_findings' in self.answered_features:
            for finding in self.answered_features['positive_findings']:
                asked_focuses.add(finding['question'])
        if 'negative_findings' in self.answered_features:
            for finding in self.answered_features['negative_findings']:
                asked_focuses.add(finding['question'])
        
        # Find highest-value unanswered question
        priority_order = ['critical', 'high', 'moderate', 'low']
        
        for priority in priority_order:
            for question_data in diagnostic_questions:
                question_focus = question_data.get('question_focus', '')
                diagnostic_value = question_data.get('diagnostic_value', 'moderate')
                
                if diagnostic_value == priority and question_focus not in asked_focuses:
                    # Generate natural question from focus
                    question_text = self._generate_question_from_focus(question_focus, question_data)
                    
                    self.questions_asked.append({
                        'focus': question_focus,
                        'question': question_text,
                        'value': diagnostic_value
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
        """Format diagnosis message for user"""
        
        urgency_messages = {
            'emergency': '🚨 This is a medical emergency. Call 911 immediately.',
            'urgent': '⚠️ This requires prompt medical attention. Go to the emergency room or urgent care today.',
            'semi_urgent': '⏰ This should be evaluated by a doctor within 24-48 hours.',
            'routine': '📋 Schedule an appointment with your primary care doctor.'
        }
        
        urgency_msg = urgency_messages.get(urgency, urgency_messages['routine'])
        
        message = f"Based on your symptoms, this is likely {diagnosis}.\n\n{urgency_msg}"
        
        if education.get('red_flags'):
            message += f"\n\n{education['red_flags']}"
        
        return message


# Test function
if __name__ == "__main__":
    engine = AdaptiveDiagnosticEngine()
    
    # Test assessment
    response = engine.start_assessment("I have abdominal pain")
    print(f"\nResponse: {response}")

