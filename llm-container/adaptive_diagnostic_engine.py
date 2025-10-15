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
        if self.status != "questioning":
            return {
                'success': False,
                'message': "No active assessment"
            }
        
        print(f"\n[Adaptive] 💬 Processing answer: '{user_answer}'")
        
        # Extract ALL clinical features from the answer
        extracted = self._extract_features_from_text(user_answer)
        
        print(f"[Adaptive] 📝 Extracted features: {list(extracted.keys())}")
        
        # Update answered features
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
        
        # Filter out low-scoring guidelines
        self.active_guidelines = [
            g for g in self.active_guidelines 
            if g['score'] > 0.3  # Keep only reasonable candidates
        ]
        
        if len(self.active_guidelines) == 0:
            print(f"[Adaptive] ⚠️ No guidelines remain above threshold")
            return {
                'success': False,
                'message': "I need more information. Can you describe your symptoms in more detail?"
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
        
        for name, guideline in self.guidelines.items():
            triggers = guideline.get('chief_complaint_triggers', [])
            
            # Check each trigger
            for trigger in triggers:
                trigger_normalized = self._normalize_text(trigger)
                
                # Direct match
                if trigger_normalized in normalized_text:
                    matched.append((name, 0.5))  # Base score for match
                    break
                
                # Synonym match
                for word in trigger_normalized.split():
                    if word in self.synonyms:
                        for synonym in self.synonyms[word]:
                            if synonym in normalized_text:
                                matched.append((name, 0.4))  # Slightly lower for synonym
                                break
        
        # Remove duplicates, keep highest score
        unique_matched = {}
        for name, score in matched:
            if name not in unique_matched or score > unique_matched[name]:
                unique_matched[name] = score
        
        return list(unique_matched.items())
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()
    
    def _extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract clinical features from natural language using LLM
        
        This is key to being non-rigid - we extract MULTIPLE pieces
        of information from a single answer.
        
        Args:
            text: User's natural language text
        
        Returns:
            Dict of extracted features
        """
        # TODO: Use LLM to extract features intelligently
        # For now, use pattern matching
        
        features = {}
        text_lower = text.lower()
        
        # Onset patterns
        onset_patterns = {
            'acute': ['today', 'this morning', 'few hours', 'suddenly', 'yesterday'],
            'subacute': ['few days', 'couple days', 'last week'],
            'chronic': ['weeks', 'months', 'long time', 'always']
        }
        
        for category, patterns in onset_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    features['onset'] = category
                    break
        
        # Location patterns
        location_patterns = {
            'RLQ': ['right lower', 'lower right', 'right side of stomach', 'right abdomen'],
            'RUQ': ['right upper', 'upper right'],
            'epigastric': ['upper stomach', 'upper abdomen', 'stomach area'],
            'LLQ': ['left lower', 'lower left'],
            'LUQ': ['left upper', 'upper left']
        }
        
        for location, patterns in location_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    features['location'] = location
                    break
        
        # Migration pattern
        if any(word in text_lower for word in ['moved', 'migrated', 'started', 'shifted']):
            if 'belly button' in text_lower or 'umbilicus' in text_lower or 'navel' in text_lower:
                if 'right' in text_lower:
                    features['migration_pattern'] = 'periumbilical_to_RLQ'
        
        # Quality
        quality_patterns = {
            'sharp': ['sharp', 'stabbing', 'knife-like'],
            'dull': ['dull', 'aching'],
            'burning': ['burning', 'hot'],
            'cramping': ['cramping', 'crampy', 'cramp'],
            'pressure': ['pressure', 'heavy', 'elephant', 'crushing']
        }
        
        for quality, patterns in quality_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    features['quality'] = quality
                    break
        
        # Associated symptoms
        if any(word in text_lower for word in ['fever', 'hot', 'temperature']):
            features['fever'] = True
        
        if any(word in text_lower for word in ['nausea', 'nauseous', 'sick', 'queasy']):
            features['nausea'] = True
        
        if any(word in text_lower for word in ['vomit', 'vomiting', 'threw up', 'throw up']):
            features['vomiting'] = True
        
        return features
    
    def _score_all_guidelines(self):
        """
        Score ALL active guidelines against ALL answered features
        
        This is the key to simultaneous evaluation - every guideline
        is scored against every piece of information we have.
        """
        for guideline_obj in self.active_guidelines:
            guideline = guideline_obj['guideline_data']
            score = guideline_obj['score']  # Start with initial match score
            
            # Score based on answered features
            # TODO: Use more sophisticated scoring from JSON criteria
            
            # Location scoring (high weight)
            if 'location' in self.answered_features:
                user_location = self.answered_features['location']
                
                # Check if guideline specifies typical location
                key_features = guideline.get('key_features', {})
                classic_pres = key_features.get('classic_presentation', '').lower()
                
                if 'RLQ' in user_location or 'right lower' in user_location:
                    if 'rlq' in classic_pres or 'right lower quadrant' in classic_pres:
                        score += 0.35  # High weight for location match
            
            # Migration pattern (very specific)
            if 'migration_pattern' in self.answered_features:
                if self.answered_features['migration_pattern'] == 'periumbilical_to_RLQ':
                    if 'periumbilical' in str(guideline).lower() and 'migrat' in str(guideline).lower():
                        score += 0.30  # Very specific for appendicitis
            
            # Onset
            if 'onset' in self.answered_features:
                user_onset = self.answered_features['onset']
                urgency = guideline.get('urgency', '').lower()
                
                if user_onset == 'acute' and urgency in ['urgent', 'emergent', 'emergency']:
                    score += 0.10
            
            # Quality
            if 'quality' in self.answered_features:
                # TODO: Check against expected responses in diagnostic_questions
                score += 0.05
            
            # Associated symptoms
            if self.answered_features.get('fever'):
                score += 0.05
            
            if self.answered_features.get('nausea') or self.answered_features.get('vomiting'):
                score += 0.05
            
            # Update score
            guideline_obj['score'] = min(score, 1.0)  # Cap at 1.0
    
    def _ask_next_question(self) -> Dict[str, Any]:
        """
        Select and ask the most discriminating next question
        
        This uses information theory - which question will
        best narrow the differential diagnosis?
        """
        # Determine what we still need to know
        critical_features = ['onset', 'location', 'quality', 'migration_pattern']
        
        # Find first missing critical feature
        for feature in critical_features:
            if feature not in self.answered_features:
                question = self._generate_question_for_feature(feature)
                
                self.questions_asked.append({
                    'feature': feature,
                    'question': question
                })
                
                return {
                    'success': True,
                    'question': question,
                    'status': 'questioning',
                    'differentials': [
                        {'name': g['name'], 'score': g['score']} 
                        for g in self.active_guidelines[:3]
                    ]
                }
        
        # If we've asked all critical questions, ask about associated symptoms
        if 'fever' not in self.answered_features:
            return {
                'success': True,
                'question': "Have you had any fever?",
                'status': 'questioning'
            }
        
        # Fallback - try to finalize with current info
        if len(self.active_guidelines) > 0:
            return self._finalize_diagnosis(self.active_guidelines[0])
        
        return {
            'success': False,
            'message': "I need more information to make a diagnosis."
        }
    
    def _generate_question_for_feature(self, feature: str) -> str:
        """Generate natural question for a specific feature"""
        
        question_templates = {
            'onset': "When did this pain start?",
            'location': "Where exactly do you feel the pain?",
            'quality': "How would you describe the pain?",
            'migration_pattern': "Did the pain start in one place and move to another?",
            'severity': "On a scale of 1-10, how severe is the pain?",
            'radiation': "Does the pain spread or radiate anywhere?"
        }
        
        return question_templates.get(feature, "Can you tell me more about your symptoms?")
    
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

