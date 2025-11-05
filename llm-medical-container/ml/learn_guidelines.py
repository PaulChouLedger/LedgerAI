#!/usr/bin/env python3
"""
Machine Learning system for learning new guideline terms from patient responses.
Tracks responses that don't match existing guideline terms and suggests additions.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class GuidelineLearner:
    """Learn new guideline terms from patient responses that don't match existing terms"""
    
    def __init__(self, learning_dir: str = None):
        self.learning_dir = Path(learning_dir) if learning_dir else Path(__file__).parent.parent / 'data' / 'learning'
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Track unmatched responses
        self.unmatched_responses_file = self.learning_dir / 'guideline_unmatched.jsonl'
        self.suggestions_file = self.learning_dir / 'guideline_suggestions.json'
        
        # Load existing suggestions
        self.suggestions = self._load_suggestions()
    
    def record_unmatched_response(self, user_input: str, oldcarts_element: str, 
                                 organ_system: str, condition: str, 
                                 matched_confidence: float = 0.0, context: Dict = None):
        """
        Record a patient response that didn't match any existing guideline terms
        
        Args:
            user_input: What the patient said
            oldcarts_element: Which OLDCARTS element was being asked
            organ_system: Organ system (GI, CARDIO, etc.)
            condition: Condition name from guideline
            matched_confidence: Confidence of best match (if any)
            context: Additional context
        """
        # Only record if confidence is low (indicating poor match)
        if matched_confidence > 0.6:
            return  # Good match, no need to record
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input.lower().strip(),
            'oldcarts_element': oldcarts_element,
            'organ_system': organ_system.upper(),
            'condition': condition,
            'matched_confidence': matched_confidence,
            'context': context or {}
        }
        
        # Append to unmatched responses
        with open(self.unmatched_responses_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction) + '\n')
    
    def analyze_unmatched_patterns(self, min_occurrences: int = 3, 
                                   max_confidence: float = 0.5) -> Dict:
        """
        Analyze unmatched responses to find new term candidates
        
        Args:
            min_occurrences: Minimum times a pattern must occur
            max_confidence: Maximum confidence to consider (lower = less matched)
            
        Returns:
            Dictionary of suggested new terms organized by condition and element
        """
        if not self.unmatched_responses_file.exists():
            return {}
        
        # Group by (organ_system, condition, oldcarts_element)
        pattern_groups = defaultdict(list)
        
        with open(self.unmatched_responses_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    if interaction.get('matched_confidence', 1.0) <= max_confidence:
                        key = (
                            interaction['organ_system'],
                            interaction['condition'],
                            interaction['oldcarts_element']
                        )
                        pattern_groups[key].append(interaction['user_input'])
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Find patterns
        suggestions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for (organ_system, condition, element), user_inputs in pattern_groups.items():
            if len(user_inputs) < min_occurrences:
                continue
            
            # Count frequency
            input_counts = defaultdict(int)
            for user_input in user_inputs:
                input_counts[user_input] += 1
            
            # Check against existing guideline terms
            guideline_file = self._get_guideline_file(organ_system, condition)
            existing_terms = self._load_existing_terms(guideline_file, element)
            
            # Suggest terms that appear frequently but aren't in guidelines
            for user_input, count in input_counts.items():
                if count >= min_occurrences:
                    normalized = self._normalize_input(user_input)
                    
                    # Check if it's already a term
                    if not self._matches_existing_term(normalized, existing_terms):
                        suggestions[organ_system][condition][element].append({
                            'term': normalized,
                            'occurrences': count,
                            'confidence': count / len(user_inputs),
                            'suggested_medical': self._suggest_medical_term(normalized, element)
                        })
        
        return suggestions
    
    def _normalize_input(self, user_input: str) -> str:
        """Normalize user input for term storage"""
        normalized = user_input.lower().strip()
        normalized = ' '.join(normalized.split())  # Remove extra whitespace
        return normalized
    
    def _matches_existing_term(self, normalized: str, existing_terms: List[Dict]) -> bool:
        """Check if normalized input matches any existing guideline term"""
        for term_obj in existing_terms:
            if isinstance(term_obj, dict):
                medical = term_obj.get('medical', '').lower()
                patient_friendly = term_obj.get('patient_friendly', '').lower()
                
                if normalized in medical or normalized in patient_friendly:
                    return True
                if medical in normalized or patient_friendly in normalized:
                    return True
        
        return False
    
    def _suggest_medical_term(self, patient_term: str, element: str) -> str:
        """
        Suggest a medical term for a patient-friendly term
        For now, returns the patient term capitalized - can be enhanced with ML
        """
        # Simple capitalization for now
        # Could be enhanced with medical term extraction/NER
        words = patient_term.split()
        if words:
            # Capitalize first word
            words[0] = words[0].capitalize()
        return ' '.join(words)
    
    def _get_guideline_file(self, organ_system: str, condition: str) -> Path:
        """Get path to guideline file"""
        # Normalize condition name to filename
        filename = condition.replace(' ', '_').replace('/', '_')
        filename = f"{organ_system}_{filename}.json"
        
        guideline_dir = Path(__file__).parent.parent / 'medical' / 'guidelines' / organ_system
        return guideline_dir / filename
    
    def _load_existing_terms(self, guideline_file: Path, element: str) -> List[Dict]:
        """Load existing terms from guideline for an element"""
        if not guideline_file.exists():
            return []
        
        try:
            with open(guideline_file, 'r', encoding='utf-8') as f:
                guideline = json.load(f)
            
            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
            if element in structured:
                element_data = structured[element]
                if isinstance(element_data, dict):
                    return element_data.get('includes', [])
        
        except (json.JSONDecodeError, KeyError):
            pass
        
        return []
    
    def generate_suggestions(self, min_occurrences: int = 3, max_confidence: float = 0.5):
        """Generate and save guideline suggestions"""
        suggestions = self.analyze_unmatched_patterns(min_occurrences, max_confidence)
        
        # Merge with existing suggestions
        for organ_system, conditions in suggestions.items():
            if organ_system not in self.suggestions:
                self.suggestions[organ_system] = {}
            
            for condition, elements in conditions.items():
                if condition not in self.suggestions[organ_system]:
                    self.suggestions[organ_system][condition] = {}
                
                for element, new_terms in elements.items():
                    if element not in self.suggestions[organ_system][condition]:
                        self.suggestions[organ_system][condition][element] = []
                    
                    # Add new suggestions
                    existing_terms = {t['term'] for t in self.suggestions[organ_system][condition][element]}
                    for new_term in new_terms:
                        if new_term['term'] not in existing_terms:
                            self.suggestions[organ_system][condition][element].append(new_term)
        
        # Save suggestions
        self._save_suggestions()
        
        return suggestions
    
    def _load_suggestions(self) -> Dict:
        """Load existing suggestions"""
        if self.suggestions_file.exists():
            try:
                with open(self.suggestions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return {}
    
    def _save_suggestions(self):
        """Save suggestions to file"""
        with open(self.suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(self.suggestions, f, indent=2, ensure_ascii=False)
    
    def apply_suggestion(self, organ_system: str, condition: str, element: str, 
                        term: str, medical_term: str = None, patient_friendly: str = None):
        """
        Apply a suggested term to the guideline file
        
        Args:
            organ_system: Organ system (GI, CARDIO, etc.)
            condition: Condition name
            element: OLDCARTS element
            term: The patient term (used as patient_friendly if not provided)
            medical_term: Medical term (auto-generated if not provided)
            patient_friendly: Patient-friendly term (uses term if not provided)
        """
        guideline_file = self._get_guideline_file(organ_system, condition)
        
        if not guideline_file.exists():
            print(f"⚠️  Guideline file not found: {guideline_file}")
            return False
        
        try:
            # Load guideline
            with open(guideline_file, 'r', encoding='utf-8') as f:
                guideline = json.load(f)
            
            # Get or create structured_oldcarts
            if 'key_features' not in guideline:
                guideline['key_features'] = {}
            if 'structured_oldcarts' not in guideline['key_features']:
                guideline['key_features']['structured_oldcarts'] = {}
            if element not in guideline['key_features']['structured_oldcarts']:
                guideline['key_features']['structured_oldcarts'][element] = {
                    'includes': [],
                    'excludes': []
                }
            
            # Prepare new term object
            new_term = {
                'medical': medical_term or self._suggest_medical_term(term, element),
                'patient_friendly': patient_friendly or term
            }
            
            # Check if term already exists
            includes = guideline['key_features']['structured_oldcarts'][element]['includes']
            for existing_term in includes:
                if isinstance(existing_term, dict):
                    if (existing_term.get('medical', '').lower() == new_term['medical'].lower() or
                        existing_term.get('patient_friendly', '').lower() == new_term['patient_friendly'].lower()):
                        print(f"⚠️  Term already exists in guideline")
                        return False
            
            # Add new term
            includes.append(new_term)
            
            # Save updated guideline
            with open(guideline_file, 'w', encoding='utf-8') as f:
                json.dump(guideline, f, indent=2, ensure_ascii=False)
            
            # Remove from suggestions
            if (organ_system in self.suggestions and 
                condition in self.suggestions[organ_system] and
                element in self.suggestions[organ_system][condition]):
                self.suggestions[organ_system][condition][element] = [
                    t for t in self.suggestions[organ_system][condition][element]
                    if t['term'] != term
                ]
                if not self.suggestions[organ_system][condition][element]:
                    del self.suggestions[organ_system][condition][element]
            
            self._save_suggestions()
            
            print(f"✅ Added term to {organ_system}.{condition}.{element}: {new_term}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying suggestion: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_suggestions_summary(self) -> Dict:
        """Get summary of pending suggestions"""
        summary = {
            'total_organ_systems': len(self.suggestions),
            'total_conditions': 0,
            'total_suggestions': 0,
            'by_organ_system': {}
        }
        
        for organ_system, conditions in self.suggestions.items():
            condition_count = len(conditions)
            total = sum(
                len(terms) 
                for condition in conditions.values()
                for terms in condition.values()
            )
            summary['total_conditions'] += condition_count
            summary['total_suggestions'] += total
            summary['by_organ_system'][organ_system] = {
                'conditions': condition_count,
                'suggestions': total
            }
        
        return summary

