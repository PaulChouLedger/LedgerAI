#!/usr/bin/env python3
"""
Interaction Learning System
Learn from user interactions to refine structured_oldcarts and synonym files

This enables:
1. User corrections → Update includes/excludes in structured_oldcarts
2. New patient terminology → Add to synonym files
3. Missing patterns → Improve guidelines over time
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

class InteractionLearning:
    """
    Learn from user interactions to improve guidelines and synonyms
    """
    
    def __init__(self, learning_dir: str = "ml/learning_data"):
        self.learning_dir = learning_dir
        os.makedirs(learning_dir, exist_ok=True)
        
        # Track corrections, new terms, and patterns
        self.corrections_file = os.path.join(learning_dir, "corrections.jsonl")
        self.synonym_expansions_file = os.path.join(learning_dir, "synonym_expansions.jsonl")
        self.pattern_detections_file = os.path.join(learning_dir, "pattern_detections.jsonl")
    
    def record_correction(self, condition: str, oldcarts_element: str, user_answer: str, 
                         expected_term: str, actual_result: str, context: Dict):
        """
        Record when user's answer was misinterpreted or scored incorrectly
        
        Args:
            condition: Medical condition name
            oldcarts_element: OLDCARTS element (location, onset, etc.)
            user_answer: What patient said
            expected_term: What system should have matched to
            actual_result: What system actually did
            context: Additional context (scores, guidelines used, etc.)
        """
        correction = {
            'timestamp': datetime.now().isoformat(),
            'condition': condition,
            'oldcarts_element': oldcarts_element,
            'user_answer': user_answer,
            'expected_term': expected_term,
            'actual_result': actual_result,
            'context': context
        }
        
        # Append to corrections file
        with open(self.corrections_file, 'a') as f:
            f.write(json.dumps(correction) + '\n')
        
        print(f"[Learning] 📝 Recorded correction: '{user_answer}' should map to '{expected_term}'")
    
    def record_synonym_expansion(self, organ_system: str, oldcarts_element: str, 
                                 category: str, new_synonym: str, context: Dict):
        """
        Record new patient terminology that should be added to synonyms
        
        Args:
            organ_system: Organ system (GI, CARDIO, etc.)
            oldcarts_element: OLDCARTS element
            category: Synonym category (ruq_pain, sudden, etc.)
            new_synonym: New patient term to add
            context: When this term was used
        """
        expansion = {
            'timestamp': datetime.now().isoformat(),
            'organ_system': organ_system,
            'oldcarts_element': oldcarts_element,
            'category': category,
            'new_synonym': new_synonym,
            'context': context
        }
        
        with open(self.synonym_expansions_file, 'a') as f:
            f.write(json.dumps(expansion) + '\n')
        
        print(f"[Learning] 📝 Recorded synonym expansion: '{new_synonym}' → '{category}'")
    
    def record_pattern_detection(self, condition: str, oldcarts_element: str, 
                                pattern: str, frequency: int, patient_variations: List[str]):
        """
        Record patterns in how patients describe symptoms
        
        Example: Many patients say "hurts when I breathe" for pleuritic pain
        → Should add "hurts when breathing" to aggravating includes
        
        Args:
            condition: Medical condition
            oldcarts_element: OLDCARTS element
            pattern: Detected pattern (e.g., "triggered by breathing")
            frequency: How often seen
            patient_variations: List of patient descriptions
        """
        detection = {
            'timestamp': datetime.now().isoformat(),
            'condition': condition,
            'oldcarts_element': oldcarts_element,
            'pattern': pattern,
            'frequency': frequency,
            'patient_variations': patient_variations
        }
        
        with open(self.pattern_detections_file, 'a') as f:
            f.write(json.dumps(detection) + '\n')
        
        print(f"[Learning] 📝 Recorded pattern: '{pattern}' (frequency: {frequency})")
    
    def analyze_corrections(self, min_occurrences: int = 3) -> Dict[str, Any]:
        """
        Analyze corrections to identify systematic issues
        
        Returns suggestions for improving structured_oldcarts
        """
        if not os.path.exists(self.corrections_file):
            return {}
        
        corrections_by_condition = defaultdict(list)
        corrections_by_element = defaultdict(list)
        
        # Load all corrections
        with open(self.corrections_file, 'r') as f:
            for line in f:
                correction = json.loads(line)
                corrections_by_condition[correction['condition']].append(correction)
                corrections_by_element[correction['oldcarts_element']].append(correction)
        
        suggestions = {}
        
        # Find conditions with multiple corrections
        for condition, corrections in corrections_by_condition.items():
            if len(corrections) >= min_occurrences:
                # Analyze what's going wrong
                user_answers = [c['user_answer'] for c in corrections]
                expected_terms = [c['expected_term'] for c in corrections]
                
                suggestions[condition] = {
                    'frequency': len(corrections),
                    'common_user_answers': user_answers,
                    'expected_mappings': expected_terms,
                    'suggestion': 'Consider adding these terms to includes/excludes'
                }
        
        return suggestions
    
    def generate_updates(self, llm_fn=None) -> Dict[str, Any]:
        """
        Generate updates for structured_oldcarts and synonym files
        
        Uses learning data to propose improvements
        """
        updates = {
            'structured_oldcarts_updates': {},
            'synonym_updates': {},
            'confidence': {}
        }
        
        # Analyze patterns
        suggestions = self.analyze_corrections()
        
        # For each condition with issues, propose updates
        for condition, data in suggestions.items():
            # Option 1: Use LLM to generate updates
            if llm_fn:
                proposed_updates = self._llm_generate_updates(condition, data, llm_fn)
                updates['structured_oldcarts_updates'][condition] = proposed_updates
            else:
                # Option 2: Simple frequency-based suggestions
                updates['structured_oldcarts_updates'][condition] = {
                    'add_to_includes': self._extract_common_terms(data['common_user_answers']),
                    'confidence': data['frequency'] / 10  # Simple confidence metric
                }
        
        # Analyze synonym expansions
        if os.path.exists(self.synonym_expansions_file):
            with open(self.synonym_expansions_file, 'r') as f:
                expansions = [json.loads(line) for line in f]
            
            # Group by category
            expansions_by_category = defaultdict(list)
            for exp in expansions:
                key = f"{exp['organ_system']}:{exp['oldcarts_element']}:{exp['category']}"
                expansions_by_category[key].append(exp['new_synonym'])
            
            # Suggest additions
            for key, synonyms in expansions_by_category.items():
                if len(synonyms) >= 3:  # At least 3 occurrences
                    updates['synonym_updates'][key] = {
                        'add': list(set(synonyms)),
                        'confidence': len(synonyms) / 10
                    }
        
        return updates
    
    def _llm_generate_updates(self, condition: str, data: Dict, llm_fn) -> Dict:
        """
        Use LLM to generate proposed updates for structured_oldcarts
        """
        prompt = f"""Given this learning data from patient interactions:

Condition: {condition}
Frequency: {data['frequency']} incorrect mappings
Common patient answers: {data['common_user_answers']}
Expected mappings: {data['expected_mappings']}

Propose updates to structured_oldcarts includes/excludes to fix these issues.
Return JSON with suggested additions/changes."""
        
        try:
            response = llm_fn(prompt)
            # Parse and return updates
            return json.loads(response)
        except Exception as e:
            print(f"⚠️ LLM update generation failed: {e}")
            return {}
    
    def _extract_common_terms(self, user_answers: List[str]) -> List[str]:
        """Extract common terms from user answers"""
        # Simple frequency analysis
        word_counts = defaultdict(int)
        for answer in user_answers:
            words = answer.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += 1
        
        # Return top terms
        return [term for term, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]]


def apply_learning_updates(updates: Dict[str, Any], guidelines_dir: str, synonyms_dir: str, dry_run: bool = True):
    """
    Apply generated updates to actual files
    
    Args:
        updates: Updates from generate_updates()
        guidelines_dir: Directory containing guideline JSON files
        synonyms_dir: Directory containing synonym JSON files
        dry_run: If True, show what would be done
    """
    # Apply structured_oldcarts updates
    for condition, changes in updates.get('structured_oldcarts_updates', {}).items():
        if dry_run:
            print(f"📋 Would update: {condition}")
            print(f"   Add to includes: {changes.get('add_to_includes', [])}")
        else:
            # TODO: Actually update the guideline file
            pass
    
    # Apply synonym updates
    for key, changes in updates.get('synonym_updates', {}).items():
        if dry_run:
            print(f"📋 Would update synonyms: {key}")
            print(f"   Add: {changes.get('add', [])}")
        else:
            # TODO: Actually update the synonym file
            pass


if __name__ == '__main__':
    # Example usage
    learner = InteractionLearning()
    
    # Record some example corrections
    learner.record_correction(
        condition="Acute Appendicitis",
        oldcarts_element="location",
        user_answer="hurts near my hip bone",
        expected_term="right lower quadrant",
        actual_result="matched to hip instead of RLQ",
        context={}
    )
    
    learner.record_synonym_expansion(
        organ_system="GI",
        oldcarts_element="location",
        category="rlq_pain",
        new_synonym="hurts near my hip bone",
        context={}
    )
    
    # Analyze and generate updates
    print("\n🔍 Analyzing learning data...")
    updates = learner.generate_updates()
    
    print("\n📊 Proposed Updates:")
    print(json.dumps(updates, indent=2))
    
    # Apply updates (dry run)
    print("\n🎯 Applying updates...")
    apply_learning_updates(updates, "medical/guidelines", "synonyms", dry_run=True)

