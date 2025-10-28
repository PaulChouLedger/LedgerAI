#!/usr/bin/env python3
"""
Learning Suggestion System
Records user interactions and generates suggestions for manual review

Usage:
  python learning_suggestions.py --analyze
  python learning_suggestions.py --show-suggestions
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

class LearningSuggestions:
    """
    Track user interactions and generate update suggestions for manual review
    """
    
    def __init__(self, learning_dir: str = "ml/learning_data"):
        self.learning_dir = learning_dir
        os.makedirs(learning_dir, exist_ok=True)
        
        self.corrections_file = os.path.join(learning_dir, "corrections.jsonl")
        self.synonym_expansions_file = os.path.join(learning_dir, "synonym_expansions.jsonl")
        self.suggestions_file = os.path.join(learning_dir, "suggestions.json")
    
    def record_prediction(self, condition: str, oldcarts_element: str, user_answer: str, 
                          similarity_score: float, guideline_text: str, context: Dict):
        """
        Record prediction automatically during scoring
        
        This creates a baseline to detect patterns of low-scoring answers
        that might indicate missing synonyms or includes terms
        """
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'condition': condition,
            'oldcarts_element': oldcarts_element,
            'user_answer': user_answer,
            'similarity_score': similarity_score,
            'guideline_text': guideline_text[:100],  # Truncate for storage
            'context': context
        }
        
        with open(self.corrections_file, 'a') as f:
            f.write(json.dumps(prediction) + '\n')
    
    def record_correction(self, condition: str, oldcarts_element: str, user_answer: str, 
                         expected_term: str, actual_result: str, context: Dict):
        """Record correction (called when user provides explicit feedback)"""
        correction = {
            'timestamp': datetime.now().isoformat(),
            'type': 'explicit_correction',
            'condition': condition,
            'oldcarts_element': oldcarts_element,
            'user_answer': user_answer,
            'expected_term': expected_term,
            'actual_result': actual_result,
            'context': context
        }
        
        with open(self.corrections_file, 'a') as f:
            f.write(json.dumps(correction) + '\n')
    
    def record_synonym_expansion(self, organ_system: str, oldcarts_element: str, 
                                 category: str, new_synonym: str, context: Dict):
        """Record new synonym term"""
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
    
    def analyze_and_suggest(self, min_occurrences: int = 5, low_score_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Analyze learning data and generate suggestions for user review
        
        Detects patterns from both automatic predictions and explicit corrections
        
        Args:
            min_occurrences: Minimum times a pattern must appear
            low_score_threshold: Similarity scores below this are considered "low"
        
        Returns:
            Dictionary with suggestions for manual review
        """
        suggestions = {
            'timestamp': datetime.now().isoformat(),
            'structured_oldcarts_updates': {},
            'synonym_updates': {},
            'summary': {}
        }
        
        # Analyze predictions and corrections
        if os.path.exists(self.corrections_file):
            patterns_by_condition = defaultdict(lambda: defaultdict(list))
            low_scores_by_condition = defaultdict(lambda: defaultdict(list))
            
            with open(self.corrections_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    key = f"{record['condition']}:{record['oldcarts_element']}"
                    
                    # Categorize by type
                    if record.get('type') == 'explicit_correction':
                        patterns_by_condition[key].append(record)
                    elif 'similarity_score' in record:
                        # Automatic prediction - check for low scores
                        if record['similarity_score'] < low_score_threshold:
                            low_scores_by_condition[key].append(record)
            
            # Generate suggestions from low-scoring patterns (automatic detection)
            all_patterns = dict(patterns_by_condition)
            all_patterns.update(low_scores_by_condition)
            
            for key, records in all_patterns.items():
                if len(records) >= min_occurrences:
                    condition, element = key.split(':')
                    
                    # Extract common user answers
                    user_answers = [r['user_answer'] for r in records]
                    avg_score = sum(r.get('similarity_score', 0) for r in records if 'similarity_score' in r) / len([r for r in records if 'similarity_score' in r]) if any('similarity_score' in r for r in records) else 0
                    
                    # Determine reason
                    is_low_score = any('similarity_score' in r for r in records)
                    reason = f"Low similarity scores ({len(low_scores_by_condition[key])} occurrences)" if is_low_score else f"Explicit corrections ({len(patterns_by_condition[key])} occurrences)"
                    
                    # Suggest adding to includes
                    suggestions['structured_oldcarts_updates'][f"{condition}_{element}"] = {
                        'condition': condition,
                        'element': element,
                        'frequency': len(records),
                        'avg_similarity_score': avg_score,
                        'suggested_adds_to_includes': self._extract_key_terms(user_answers),
                        'reason': reason,
                        'confidence': min(0.95, len(records) / 20),
                        'examples': user_answers[:5],
                        'detection_method': 'automatic' if is_low_score else 'explicit'
                    }
        
        # Analyze synonym expansions
        if os.path.exists(self.synonym_expansions_file):
            expansions_by_category = defaultdict(list)
            
            with open(self.synonym_expansions_file, 'r') as f:
                for line in f:
                    expansion = json.loads(line)
                    key = f"{expansion['organ_system']}:{expansion['oldcarts_element']}:{expansion['category']}"
                    expansions_by_category[key].append(expansion)
            
            for key, expansions in expansions_by_category.items():
                if len(expansions) >= min_occurrences:
                    new_synonyms = list(set([e['new_synonym'] for e in expansions]))
                    
                    suggestions['synonym_updates'][key] = {
                        'key': key,
                        'frequency': len(expansions),
                        'suggested_synonyms': new_synonyms,
                        'reason': f"Patients used these terms {len(expansions)} times",
                        'confidence': min(0.95, len(expansions) / 15),
                        'examples': [e['context'] for e in expansions[:5]]
                    }
        
        # Generate summary
        suggestions['summary'] = {
            'total_corrections': self._count_records(self.corrections_file),
            'total_synonym_expansions': self._count_records(self.synonym_expansions_file),
            'structured_updates_count': len(suggestions['structured_oldcarts_updates']),
            'synonym_updates_count': len(suggestions['synonym_updates']),
            'highest_confidence': max(
                [s.get('confidence', 0) for s in suggestions['structured_oldcarts_updates'].values()] +
                [s.get('confidence', 0) for s in suggestions['synonym_updates'].values()] +
                [0]
            )
        }
        
        # Save suggestions
        with open(self.suggestions_file, 'w') as f:
            json.dump(suggestions, f, indent=2)
        
        return suggestions
    
    def _extract_key_terms(self, user_answers: List[str]) -> List[str]:
        """Extract meaningful terms from user answers"""
        # Simple approach: return unique answers (for now)
        # Could be enhanced with NLP to extract key phrases
        unique_answers = list(set(user_answers))
        return unique_answers[:10]  # Limit to top 10
    
    def _count_records(self, filepath: str) -> int:
        """Count records in a JSONL file"""
        if not os.path.exists(filepath):
            return 0
        
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def show_suggestions(self):
        """Display suggestions in a user-friendly format"""
        if not os.path.exists(self.suggestions_file):
            print("❌ No suggestions found. Run --analyze first.")
            return
        
        with open(self.suggestions_file, 'r') as f:
            suggestions = json.load(f)
        
        print("\n" + "="*80)
        print("LEARNING SUGGESTIONS")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   Total Corrections: {suggestions['summary']['total_corrections']}")
        print(f"   Total Synonym Expansions: {suggestions['summary']['total_synonym_expansions']}")
        print(f"   Structured Updates Suggested: {suggestions['summary']['structured_updates_count']}")
        print(f"   Synonym Updates Suggested: {suggestions['summary']['synonym_updates_count']}")
        print(f"   Highest Confidence: {suggestions['summary']['highest_confidence']:.2%}")
        
        # Show structured_oldcarts suggestions
        if suggestions['structured_oldcarts_updates']:
            print(f"\n📋 STRUCTURED OLDCARTS UPDATES:")
            print("-"*80)
            
            for key, suggestion in suggestions['structured_oldcarts_updates'].items():
                print(f"\n🎯 {suggestion['condition']} ({suggestion['element']})")
                print(f"   Frequency: {suggestion['frequency']} occurrences")
                print(f"   Confidence: {suggestion['confidence']:.2%}")
                print(f"   Reason: {suggestion['reason']}")
                print(f"   Suggested Adds to Includes:")
                for term in suggestion['suggested_adds_to_includes']:
                    print(f"      • {term}")
                print(f"   Examples:")
                for example in suggestion['examples']:
                    print(f"      • {example}")
        
        # Show synonym suggestions
        if suggestions['synonym_updates']:
            print(f"\n📝 SYNONYM UPDATES:")
            print("-"*80)
            
            for key, suggestion in suggestions['synonym_updates'].items():
                print(f"\n🎯 {suggestion['key']}")
                print(f"   Frequency: {suggestion['frequency']} occurrences")
                print(f"   Confidence: {suggestion['confidence']:.2%}")
                print(f"   Reason: {suggestion['reason']}")
                print(f"   Suggested Synonyms to Add:")
                for term in suggestion['suggested_synonyms']:
                    print(f"      • {term}")
        
        print("\n" + "="*80)
        print("💡 To apply these changes, use: python scripts/apply_suggestions.py")
        print("="*80 + "\n")
    
    def export_for_manual_review(self, output_file: str = "suggestions_for_review.json"):
        """Export suggestions in a format suitable for manual review"""
        if not os.path.exists(self.suggestions_file):
            print("❌ No suggestions found. Run --analyze first.")
            return
        
        with open(self.suggestions_file, 'r') as f:
            suggestions = json.load(f)
        
        # Format for easy manual review
        review_format = {
            'generated_at': suggestions['timestamp'],
            'summary': suggestions['summary'],
            'recommendations': []
        }
        
        # Add structured_oldcarts recommendations
        for key, suggestion in suggestions['structured_oldcarts_updates'].items():
            review_format['recommendations'].append({
                'type': 'structured_oldcarts',
                'condition': suggestion['condition'],
                'element': suggestion['element'],
                'action': 'Add to includes',
                'items': suggestion['suggested_adds_to_includes'],
                'confidence': suggestion['confidence'],
                'frequency': suggestion['frequency'],
                'examples': suggestion['examples']
            })
        
        # Add synonym recommendations
        for key, suggestion in suggestions['synonym_updates'].items():
            review_format['recommendations'].append({
                'type': 'synonym',
                'category': suggestion['key'],
                'action': 'Add to synonyms',
                'items': suggestion['suggested_synonyms'],
                'confidence': suggestion['confidence'],
                'frequency': suggestion['frequency']
            })
        
        # Save for review
        with open(output_file, 'w') as f:
            json.dump(review_format, f, indent=2)
        
        print(f"✅ Export saved to {output_file}")
        print(f"   {len(review_format['recommendations'])} recommendations ready for manual review")


def main():
    parser = argparse.ArgumentParser(description="Learning Suggestions System")
    parser.add_argument('--analyze', action='store_true', help='Analyze learning data and generate suggestions')
    parser.add_argument('--show', action='store_true', help='Show existing suggestions')
    parser.add_argument('--export', action='store_true', help='Export suggestions for manual review')
    parser.add_argument('--output', default='suggestions_for_review.json', help='Output file for export')
    parser.add_argument('--min-occurrences', type=int, default=5, help='Minimum occurrences to suggest update')
    
    args = parser.parse_args()
    
    learner = LearningSuggestions()
    
    if args.analyze:
        print("🔍 Analyzing learning data...")
        suggestions = learner.analyze_and_suggest(min_occurrences=args.min_occurrences)
        print(f"\n✅ Analysis complete!")
        print(f"   Generated {len(suggestions['structured_oldcarts_updates'])} structured updates")
        print(f"   Generated {len(suggestions['synonym_updates'])} synonym updates")
        print("\nRun with --show to view suggestions")
    
    if args.show:
        learner.show_suggestions()
    
    if args.export:
        learner.export_for_manual_review(args.output)
    
    if not any([args.analyze, args.show, args.export]):
        parser.print_help()


if __name__ == '__main__':
    main()

