#!/usr/bin/env python3
"""
Apply Learning Suggestions
Manually review and apply learning suggestions

Usage:
  python apply_suggestions.py --review
  python apply_suggestions.py --apply-all --condition "Acute Appendicitis"
  python apply_suggestions.py --apply-specific 0
"""

import json
import argparse
import os
import sys
from pathlib import Path

def load_suggestions(suggestions_file: str = "ml/learning_data/suggestions_for_review.json"):
    """Load suggestions for review"""
    if not os.path.exists(suggestions_file):
        print(f"❌ Suggestions file not found: {suggestions_file}")
        return None
    
    with open(suggestions_file, 'r') as f:
        return json.load(f)

def show_review_interface(suggestions):
    """Display suggestions with option to apply"""
    print("\n" + "="*80)
    print("LEARNING SUGGESTIONS - REVIEW & APPLY")
    print("="*80)
    
    recommendations = suggestions['recommendations']
    
    if not recommendations:
        print("\n✅ No recommendations to review")
        return
    
    for idx, rec in enumerate(recommendations):
        print(f"\n{'#'*80}")
        print(f"RECOMMENDATION #{idx+1}")
        print(f"{'#'*80}")
        
        print(f"\n📋 Type: {rec['type']}")
        if rec['type'] == 'structured_oldcarts':
            print(f"   Condition: {rec['condition']}")
            print(f"   Element: {rec['element']}")
        else:
            print(f"   Category: {rec['category']}")
        
        print(f"   Action: {rec['action']}")
        print(f"   Confidence: {rec['confidence']:.0%}")
        print(f"   Frequency: {rec['frequency']} occurrences")
        
        print(f"\n   Items to {rec['action'].lower()}:")
        for item in rec['items']:
            print(f"      • {item}")
        
        if 'examples' in rec:
            print(f"\n   Examples from patient interactions:")
            for example in rec['examples']:
                print(f"      • {example}")
        
        print(f"\n   Apply this change? (y/n/skip): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            apply_recommendation(rec)
            print(f"   ✅ Applied!")
        elif response == 'skip':
            print(f"   ⏭️  Skipped")
        else:
            print(f"   ❌ Declined")

def apply_recommendation(recommendation):
    """Apply a specific recommendation"""
    if recommendation['type'] == 'structured_oldcarts':
        apply_structured_update(recommendation)
    elif recommendation['type'] == 'synonym':
        apply_synonym_update(recommendation)

def apply_structured_update(rec):
    """Apply structured_oldcarts update"""
    condition = rec['condition']
    element = rec['element']
    items = rec['items']
    
    # Find guideline file
    guideline_file = find_guideline_file(condition)
    if not guideline_file:
        print(f"   ⚠️  Could not find guideline file for {condition}")
        return
    
    # Load guideline
    with open(guideline_file, 'r') as f:
        guideline = json.load(f)
    
    # Update structured_oldcarts
    if 'key_features' not in guideline:
        guideline['key_features'] = {}
    
    if 'structured_oldcarts' not in guideline['key_features']:
        guideline['key_features']['structured_oldcarts'] = {}
    
    if element not in guideline['key_features']['structured_oldcarts']:
        guideline['key_features']['structured_oldcarts'][element] = {'includes': [], 'excludes': []}
    
    # Add items to includes
    current_includes = guideline['key_features']['structured_oldcarts'][element].get('includes', [])
    new_items = [item for item in items if item not in current_includes]
    current_includes.extend(new_items)
    
    guideline['key_features']['structured_oldcarts'][element]['includes'] = current_includes
    
    # Save updated guideline
    with open(guideline_file, 'w') as f:
        json.dump(guideline, f, indent=2)
    
    print(f"   📝 Updated: {guideline_file}")
    print(f"      Added {len(new_items)} items to {element} includes")

def apply_synonym_update(rec):
    """Apply synonym update"""
    category_parts = rec['category'].split(':')
    if len(category_parts) != 3:
        print(f"   ⚠️  Invalid category format: {rec['category']}")
        return
    
    organ_system, oldcarts_element, category = category_parts
    
    # Find synonym file
    synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
    if not os.path.exists(synonym_file):
        print(f"   ⚠️  Synonym file not found: {synonym_file}")
        return
    
    # Load synonyms
    with open(synonym_file, 'r') as f:
        synonyms = json.load(f)
    
    # Update synonyms
    if oldcarts_element not in synonyms:
        synonyms[oldcarts_element] = {}
    
    if category not in synonyms[oldcarts_element]:
        synonyms[oldcarts_element][category] = []
    
    # Add new synonyms
    current_synonyms = synonyms[oldcarts_element][category]
    new_synonyms = [item for item in rec['items'] if item not in current_synonyms]
    current_synonyms.extend(new_synonyms)
    
    synonyms[oldcarts_element][category] = current_synonyms
    
    # Save updated synonyms
    with open(synonym_file, 'w') as f:
        json.dump(synonyms, f, indent=2)
    
    print(f"   📝 Updated: {synonym_file}")
    print(f"      Added {len(new_synonyms)} synonyms to {category}")

def find_guideline_file(condition: str) -> str:
    """Find guideline file for a condition"""
    # Search all guideline directories
    for guidelines_dir in Path("medical/guidelines").iterdir():
        if guidelines_dir.is_dir():
            guideline_file = guidelines_dir / f"{condition.replace(' ', '_')}.json"
            if guideline_file.exists():
                return str(guideline_file)
            
            # Try with different naming conventions
            for file in guidelines_dir.glob(f"*{condition.replace(' ', '_')}*"):
                return str(file)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Apply learning suggestions")
    parser.add_argument('--review', action='store_true', help='Review suggestions interactively')
    parser.add_argument('--apply-all', action='store_true', help='Apply all suggestions')
    parser.add_argument('--apply-specific', type=int, help='Apply specific recommendation by index')
    parser.add_argument('--condition', help='Filter by condition name')
    
    args = parser.parse_args()
    
    suggestions = load_suggestions()
    if not suggestions:
        return
    
    if args.review:
        show_review_interface(suggestions)
    elif args.apply_all:
        # TODO: Implement apply all with condition filter
        print("Not yet implemented")
    elif args.apply_specific is not None:
        # TODO: Implement apply specific
        print("Not yet implemented")
    else:
        print("💡 Use --review to interactively review and apply suggestions")
        print("   Use --apply-all to apply all suggestions")
        print("   Use --apply-specific N to apply recommendation #N")


if __name__ == '__main__':
    main()

