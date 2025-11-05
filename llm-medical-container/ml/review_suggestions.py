#!/usr/bin/env python3
"""
Review and apply ML-generated suggestions for synonyms and guidelines.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.learn_synonyms import SynonymLearner
from ml.learn_guidelines import GuidelineLearner

def review_synonym_suggestions():
    """Review and interactively apply synonym suggestions"""
    learner = SynonymLearner()
    
    # Generate suggestions
    print("🔍 Analyzing interaction patterns...")
    suggestions = learner.generate_suggestions(min_occurrences=3, min_confidence=0.7)
    
    summary = learner.get_suggestions_summary()
    print(f"\n📊 Synonym Suggestions Summary:")
    print(f"   Organ systems: {summary['total_organ_systems']}")
    print(f"   Total suggestions: {summary['total_suggestions']}")
    
    if summary['total_suggestions'] == 0:
        print("\n✅ No new suggestions at this time")
        return
    
    print("\n" + "="*60)
    print("SYNONYM SUGGESTIONS")
    print("="*60)
    
    for organ_system, term_suggestions in suggestions.items():
        print(f"\n📁 {organ_system.upper()}:")
        for term_key, syns in term_suggestions.items():
            print(f"\n  {term_key}:")
            for syn in syns:
                print(f"    - '{syn['synonym']}' (occurrences: {syn['occurrences']}, confidence: {syn['confidence']:.2f})")
    
    # Interactive application
    print("\n" + "="*60)
    print("Apply suggestions? (y/n/a for all/q to quit)")
    response = input().strip().lower()
    
    if response == 'q':
        return
    
    if response == 'a':
        # Apply all
        applied = 0
        for organ_system, term_suggestions in suggestions.items():
            for term_key, syns in term_suggestions.items():
                for syn in syns:
                    if learner.apply_suggestion(organ_system, term_key, syn['synonym']):
                        applied += 1
        print(f"\n✅ Applied {applied} suggestions")
    elif response == 'y':
        # Interactive selection
        for organ_system, term_suggestions in suggestions.items():
            for term_key, syns in term_suggestions.items():
                print(f"\n{organ_system}.{term_key}:")
                for i, syn in enumerate(syns, 1):
                    print(f"  [{i}] {syn['synonym']} (occ: {syn['occurrences']})")
                print("  [a] Apply all  [s] Skip")
                choice = input("  Choice: ").strip().lower()
                
                if choice == 'a':
                    for syn in syns:
                        learner.apply_suggestion(organ_system, term_key, syn['synonym'])
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(syns):
                        learner.apply_suggestion(organ_system, term_key, syns[idx]['synonym'])

def review_guideline_suggestions():
    """Review and interactively apply guideline suggestions"""
    learner = GuidelineLearner()
    
    # Generate suggestions
    print("🔍 Analyzing unmatched responses...")
    suggestions = learner.generate_suggestions(min_occurrences=3, max_confidence=0.5)
    
    summary = learner.get_suggestions_summary()
    print(f"\n📊 Guideline Suggestions Summary:")
    print(f"   Organ systems: {summary['total_organ_systems']}")
    print(f"   Conditions: {summary['total_conditions']}")
    print(f"   Total suggestions: {summary['total_suggestions']}")
    
    if summary['total_suggestions'] == 0:
        print("\n✅ No new suggestions at this time")
        return
    
    print("\n" + "="*60)
    print("GUIDELINE SUGGESTIONS")
    print("="*60)
    
    for organ_system, conditions in suggestions.items():
        print(f"\n📁 {organ_system.upper()}:")
        for condition, elements in conditions.items():
            print(f"\n  {condition}:")
            for element, terms in elements.items():
                print(f"    {element}:")
                for term in terms:
                    print(f"      - '{term['term']}' -> medical: '{term['suggested_medical']}' (occ: {term['occurrences']})")
    
    # Interactive application
    print("\n" + "="*60)
    print("Apply suggestions? (y/n/a for all/q to quit)")
    response = input().strip().lower()
    
    if response == 'q':
        return
    
    if response == 'a':
        # Apply all
        applied = 0
        for organ_system, conditions in suggestions.items():
            for condition, elements in conditions.items():
                for element, terms in elements.items():
                    for term in terms:
                        if learner.apply_suggestion(
                            organ_system, condition, element,
                            term['term'], term['suggested_medical']
                        ):
                            applied += 1
        print(f"\n✅ Applied {applied} suggestions")
    elif response == 'y':
        # Interactive selection
        for organ_system, conditions in suggestions.items():
            for condition, elements in conditions.items():
                for element, terms in elements.items():
                    print(f"\n{organ_system}.{condition}.{element}:")
                    for i, term in enumerate(terms, 1):
                        print(f"  [{i}] '{term['term']}' -> '{term['suggested_medical']}' (occ: {term['occurrences']})")
                    print("  [a] Apply all  [s] Skip")
                    choice = input("  Choice: ").strip().lower()
                    
                    if choice == 'a':
                        for term in terms:
                            learner.apply_suggestion(
                                organ_system, condition, element,
                                term['term'], term['suggested_medical']
                            )
                    elif choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(terms):
                            term = terms[idx]
                            learner.apply_suggestion(
                                organ_system, condition, element,
                                term['term'], term['suggested_medical']
                            )

if __name__ == "__main__":
    print("="*60)
    print("ML LEARNING SUGGESTION REVIEWER")
    print("="*60)
    print("\n[1] Review Synonym Suggestions")
    print("[2] Review Guideline Suggestions")
    print("[3] Review Both")
    print("[q] Quit")
    
    choice = input("\nChoice: ").strip().lower()
    
    if choice == '1':
        review_synonym_suggestions()
    elif choice == '2':
        review_guideline_suggestions()
    elif choice == '3':
        review_synonym_suggestions()
        print("\n" + "="*60)
        review_guideline_suggestions()
    elif choice == 'q':
        pass
    else:
        print("Invalid choice")

