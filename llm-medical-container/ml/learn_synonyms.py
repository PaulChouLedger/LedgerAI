#!/usr/bin/env python3
"""
Machine Learning system for learning new synonym expansions from user interactions.
Tracks patient responses and suggests new synonyms when patterns are detected.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re

class SynonymLearner:
    """Learn new synonyms from patient responses and medical term matching"""
    
    def __init__(self, learning_dir: str = None):
        self.learning_dir = Path(learning_dir) if learning_dir else Path(__file__).parent.parent / 'data' / 'learning'
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Track interactions: user_input -> matched_term mappings
        self.interaction_history_file = self.learning_dir / 'synonym_interactions.jsonl'
        self.suggestions_file = self.learning_dir / 'synonym_suggestions.json'
        
        # Load existing suggestions
        self.suggestions = self._load_suggestions()
    
    def record_interaction(self, user_input: str, matched_term: str, oldcarts_element: str, 
                          organ_system: str, confidence: float, context: Dict = None):
        """
        Record a user interaction where a term was matched
        
        Args:
            user_input: What the patient actually said
            matched_term: The medical term that was matched
            oldcarts_element: Which OLDCARTS element (onset, location, etc.)
            organ_system: Organ system (GI, CARDIO, etc.)
            confidence: Matching confidence score
            context: Additional context (chief complaint, condition, etc.)
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input.lower().strip(),
            'matched_term': matched_term,
            'oldcarts_element': oldcarts_element,
            'organ_system': organ_system.upper(),
            'confidence': confidence,
            'context': context or {}
        }
        
        # Append to interaction history
        with open(self.interaction_history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction) + '\n')
    
    def analyze_patterns(self, min_occurrences: int = 3, min_confidence: float = 0.7) -> Dict:
        """
        Analyze interaction patterns to find new synonym candidates
        
        Args:
            min_occurrences: Minimum times a pattern must occur to suggest
            min_confidence: Minimum confidence score to consider
            
        Returns:
            Dictionary of suggested synonyms organized by element and term
        """
        if not self.interaction_history_file.exists():
            return {}
        
        # Group interactions by (organ_system, oldcarts_element, matched_term)
        pattern_groups = defaultdict(list)
        
        with open(self.interaction_history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    if interaction.get('confidence', 0) >= min_confidence:
                        key = (
                            interaction['organ_system'],
                            interaction['oldcarts_element'],
                            interaction['matched_term']
                        )
                        pattern_groups[key].append(interaction['user_input'])
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Find patterns that aren't already in synonyms
        suggestions = defaultdict(lambda: defaultdict(list))
        
        for (organ_system, element, matched_term), user_inputs in pattern_groups.items():
            if len(user_inputs) < min_occurrences:
                continue
            
            # Count frequency of each unique user input
            input_counts = defaultdict(int)
            for user_input in user_inputs:
                input_counts[user_input] += 1
            
            # Check if these inputs are already in synonyms
            synonym_file = self._get_synonym_file(organ_system)
            existing_synonyms = self._load_existing_synonyms(synonym_file, element, matched_term)
            
            # Suggest new synonyms that appear frequently but aren't in existing list
            for user_input, count in input_counts.items():
                if count >= min_occurrences:
                    # Normalize the input
                    normalized = self._normalize_input(user_input)
                    
                    # Check if it's already a synonym
                    if normalized not in existing_synonyms:
                        # Check if it's semantically distinct from existing synonyms
                        if self._is_semantically_distinct(normalized, existing_synonyms):
                            suggestions[organ_system][f"{element}.{matched_term}"].append({
                                'synonym': normalized,
                                'occurrences': count,
                                'confidence': count / len(user_inputs)
                            })
        
        return suggestions
    
    def _normalize_input(self, user_input: str) -> str:
        """Normalize user input for synonym storage"""
        # Lowercase and strip
        normalized = user_input.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common filler words at start/end
        filler_words = ['i', 'i\'m', 'i\'ve', 'it', 'it\'s', 'the', 'a', 'an', 'my', 'me']
        words = normalized.split()
        while words and words[0] in filler_words:
            words.pop(0)
        while words and words[-1] in filler_words:
            words.pop()
        
        return ' '.join(words) if words else normalized
    
    def _is_semantically_distinct(self, new_input: str, existing_synonyms: List[str]) -> bool:
        """
        Check if new_input is semantically distinct from existing synonyms
        Uses simple heuristics - semantic matching will handle actual similarity
        """
        new_lower = new_input.lower()
        
        for existing in existing_synonyms:
            existing_lower = existing.lower()
            
            # Exact match
            if new_lower == existing_lower:
                return False
            
            # One contains the other (likely redundant)
            if new_lower in existing_lower or existing_lower in new_lower:
                # If they're very similar length, likely redundant
                if abs(len(new_lower) - len(existing_lower)) < 5:
                    return False
        
        return True
    
    def _get_synonym_file(self, organ_system: str) -> Path:
        """Get path to synonym file for organ system"""
        return Path(__file__).parent.parent / 'medical' / 'synonyms' / f"{organ_system.lower()}_synonyms_oldcarts.json"
    
    def _load_existing_synonyms(self, synonym_file: Path, element: str, matched_term: str) -> List[str]:
        """Load existing synonyms for a term"""
        if not synonym_file.exists():
            return []
        
        try:
            with open(synonym_file, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            if element in synonyms_data:
                # Get synonyms for this term (handle underscore keys)
                term_key = matched_term.replace(' ', '_').lower()
                if term_key in synonyms_data[element]:
                    return [s.lower() for s in synonyms_data[element][term_key]]
            
        except (json.JSONDecodeError, KeyError):
            pass
        
        return []
    
    def generate_suggestions(self, min_occurrences: int = 3, min_confidence: float = 0.7):
        """Generate and save synonym suggestions"""
        suggestions = self.analyze_patterns(min_occurrences, min_confidence)
        
        # Merge with existing suggestions
        for organ_system, term_suggestions in suggestions.items():
            if organ_system not in self.suggestions:
                self.suggestions[organ_system] = {}
            
            for term_key, new_syns in term_suggestions.items():
                if term_key not in self.suggestions[organ_system]:
                    self.suggestions[organ_system][term_key] = []
                
                # Add new suggestions (avoid duplicates)
                existing_synonyms = {s['synonym'] for s in self.suggestions[organ_system][term_key]}
                for new_syn in new_syns:
                    if new_syn['synonym'] not in existing_synonyms:
                        self.suggestions[organ_system][term_key].append(new_syn)
        
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
    
    def apply_suggestion(self, organ_system: str, term_key: str, synonym: str):
        """
        Apply a suggested synonym to the actual synonym file
        
        Args:
            organ_system: Organ system (GI, CARDIO, etc.)
            term_key: Format "element.term" (e.g., "associated.nausea")
            synonym: The synonym to add
        """
        element, matched_term = term_key.split('.', 1)
        synonym_file = self._get_synonym_file(organ_system)
        
        if not synonym_file.exists():
            print(f"⚠️  Synonym file not found: {synonym_file}")
            return False
        
        try:
            # Load existing synonyms
            with open(synonym_file, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            # Get or create element
            if element not in synonyms_data:
                synonyms_data[element] = {}
            
            # Normalize term key
            term_key_normalized = matched_term.replace(' ', '_').lower()
            
            # Get or create term list
            if term_key_normalized not in synonyms_data[element]:
                synonyms_data[element][term_key_normalized] = []
            
            # Add synonym if not already present
            if synonym.lower() not in [s.lower() for s in synonyms_data[element][term_key_normalized]]:
                synonyms_data[element][term_key_normalized].append(synonym)
                synonyms_data[element][term_key_normalized].sort()  # Keep sorted
            
            # Save updated synonyms
            with open(synonym_file, 'w', encoding='utf-8') as f:
                json.dump(synonyms_data, f, indent=2, ensure_ascii=False)
            
            # Remove from suggestions
            if organ_system in self.suggestions and term_key in self.suggestions[organ_system]:
                self.suggestions[organ_system][term_key] = [
                    s for s in self.suggestions[organ_system][term_key]
                    if s['synonym'] != synonym
                ]
                if not self.suggestions[organ_system][term_key]:
                    del self.suggestions[organ_system][term_key]
            
            self._save_suggestions()
            
            print(f"✅ Added synonym '{synonym}' to {organ_system}.{term_key}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying suggestion: {e}")
            return False
    
    def get_suggestions_summary(self) -> Dict:
        """Get summary of pending suggestions"""
        summary = {
            'total_organ_systems': len(self.suggestions),
            'total_suggestions': 0,
            'by_organ_system': {}
        }
        
        for organ_system, term_suggestions in self.suggestions.items():
            total = sum(len(syns) for syns in term_suggestions.values())
            summary['total_suggestions'] += total
            summary['by_organ_system'][organ_system] = {
                'terms': len(term_suggestions),
                'synonyms': total
            }
        
        return summary

