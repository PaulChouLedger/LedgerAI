#!/usr/bin/env python3
"""
Simple test script to test OLDCARTS normalization without the full adaptive diagnostic engine
"""

import sys
import os
import json
import re

def test_oldcarts_normalization_simple():
    """Test OLDCARTS normalization using the synonym files directly"""
    print("🧪 Simple OLDCARTS Normalization Test")
    print("=" * 50)
    
    # Find the synonyms file
    synonym_file = None
    possible_paths = [
        'llm-container/synonyms/gi_synonyms_oldcarts.json',
        './llm-container/synonyms/gi_synonyms_oldcarts.json',
        'synonyms/gi_synonyms_oldcarts.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            synonym_file = path
            print(f"✅ Found synonyms file: {path}")
            break
    
    if not synonym_file:
        print("❌ Could not find gi_synonyms_oldcarts.json file")
        print("   Current directory:", os.getcwd())
        print("   Available files:", os.listdir('.'))
        return False
    
    # Load the synonyms
    try:
        with open(synonym_file, 'r') as f:
            oldcarts_synonyms = json.load(f)
        print(f"✅ Successfully loaded OLDCARTS synonyms")
        print(f"   Found {len(oldcarts_synonyms)} main categories")
    except Exception as e:
        print(f"❌ Failed to load synonyms: {e}")
        return False
    
    # Test normalization function
    def normalize_text(text):
        """Simple normalization function"""
        normalized_text = text.lower()
        
        # Flatten OLDCARTS structure into standard_term -> variations mapping
        synonyms = {}
        for category, subcategories in oldcarts_synonyms.items():
            if isinstance(subcategories, dict):
                for subcategory, variations in subcategories.items():
                    if isinstance(variations, list):
                        # Create standard term from category and subcategory
                        standard_term = f"{category}_{subcategory}".replace("_", " ")
                        synonyms[standard_term] = variations
                    elif isinstance(variations, dict):
                        # Handle nested structures
                        for nested_key, nested_variations in variations.items():
                            if isinstance(nested_variations, list):
                                standard_term = f"{category}_{subcategory}_{nested_key}".replace("_", " ")
                                synonyms[standard_term] = nested_variations
            elif isinstance(subcategories, list):
                # Direct list of variations
                standard_term = category.replace("_", " ")
                synonyms[standard_term] = subcategories
        
        # Apply synonym replacements
        all_variations = []
        for standard_term, variations in synonyms.items():
            for variation in variations:
                all_variations.append((len(variation), variation, standard_term))
        
        # Sort by length (longest first) to avoid partial replacements
        all_variations.sort(key=lambda x: x[0], reverse=True)
        
        # Debug: Show some of the loaded synonyms
        print(f"   📊 Loaded {len(all_variations)} synonym variations")
        print(f"   📋 Sample synonyms:")
        for i, (length, variation, standard_term) in enumerate(all_variations[:10]):
            print(f"      {i+1}. '{variation}' → '{standard_term}'")
        if len(all_variations) > 10:
            print(f"      ... and {len(all_variations) - 10} more")
        
        for length, variation, standard_term in all_variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                normalized_text = re.sub(pattern, standard_term, normalized_text, flags=re.IGNORECASE)
                print(f"   🔄 '{variation}' → '{standard_term}'")
                # Don't break - continue to find other matches
        
        return normalized_text
    
    # Test specific terms first
    print(f"\n🔍 Testing specific terms...")
    
    # Flatten OLDCARTS structure into standard_term -> variations mapping
    synonyms = {}
    for category, subcategories in oldcarts_synonyms.items():
        if isinstance(subcategories, dict):
            for subcategory, variations in subcategories.items():
                if isinstance(variations, list):
                    # Create standard term from category and subcategory
                    standard_term = f"{category}_{subcategory}".replace("_", " ")
                    synonyms[standard_term] = variations
                elif isinstance(variations, dict):
                    # Handle nested structures
                    for nested_key, nested_variations in variations.items():
                        if isinstance(nested_variations, list):
                            standard_term = f"{category}_{subcategory}_{nested_key}".replace("_", " ")
                            synonyms[standard_term] = nested_variations
        elif isinstance(subcategories, list):
            # Direct list of variations
            standard_term = category.replace("_", " ")
            synonyms[standard_term] = subcategories
    
    test_terms = ["tummy", "belly ache", "queasy", "want to throw up", "upper right"]
    for term in test_terms:
        found = False
        for standard_term, variations in synonyms.items():
            if term in variations:
                print(f"   ✅ Found '{term}' → '{standard_term}'")
                found = True
                break
        if not found:
            print(f"   ❌ '{term}' not found in synonyms")
    
    # Test cases
    test_cases = [
        "my tummy hurts really bad in the upper right",
        "I have sharp stabbing pain that started suddenly after eating",
        "my belly ache gets worse when I move and goes to my back",
        "I feel queasy and want to throw up, it's really painful",
        "pain in my left lower belly that stays in one spot"
    ]
    
    print(f"\n🔍 Testing OLDCARTS normalization...")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{test_input}'")
        normalized = normalize_text(test_input)
        print(f"Normalized: '{normalized}'")
        
        if normalized != test_input.lower():
            print("✅ Normalization applied")
        else:
            print("⚠️  No normalization applied")
    
    print(f"\n" + "=" * 50)
    print("✅ Simple OLDCARTS normalization test completed!")
    return True

if __name__ == "__main__":
    test_oldcarts_normalization_simple()
