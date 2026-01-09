#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore Original Working RAG Examples
Removes all examples we added and keeps only the original ~140 working examples

The original working dataset had ~135-140 examples. We've been adding examples
to fix issues, but this may have introduced wrong patterns. If the original was
working, we should restore it and only add conversational examples for CoT toggle.
"""

import json
import hashlib
from datetime import datetime

def get_example_hash(example):
    """Get a hash of an example for comparison"""
    return hashlib.md5(json.dumps(example, sort_keys=True).encode()).hexdigest()

def get_example_age(example):
    """Check if example looks like it was added recently (by checking for new patterns)"""
    assistant = next((m['content'] for m in example.get('messages', []) if m.get('role') == 'assistant'), '')
    user = next((m['content'] for m in example.get('messages', []) if m.get('role') == 'user'), '')
    
    # Check for patterns that suggest these are new examples we added
    # New examples might have:
    # - TechFlow Systems, CloudScale Technologies, DataFlow Systems, InnovateAI Solutions, QuantumTech (our new companies)
    # - Explicit "complete scan" patterns
    # - Anti-hallucination patterns
    new_patterns = [
        'TechFlow Systems',
        'CloudScale Technologies', 
        'DataFlow Systems',
        'InnovateAI Solutions',
        'QuantumTech',
        'TechFlow Innovations',
        'CloudScale Innovations',
        'DataFlow Analytics',
        'InnovateAI Solutions'
    ]
    
    # Also check for explicit LedgerAI examples we added for anti-hallucination
    is_ledger_ai_exact = 'LedgerAI' in user and 'co-founders' in user.lower() and len(user) > 2000
    
    # Check if it matches our new example patterns
    has_new_pattern = any(pattern in user for pattern in new_patterns)
    
    return has_new_pattern or is_ledger_ai_exact

if __name__ == "__main__":
    print("=" * 80)
    print("Restore Original Working RAG Examples")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will remove examples we added (~22 examples)")
    print("   and keep only the original ~140 working examples")
    print()
    print("The original working dataset had ~135-140 examples.")
    print("We added ~22 examples that may have incorrect reasoning patterns.")
    print()
    print("If the original was working, we should restore it.")
    print()
    
    # Load current dataset
    try:
        with open("rag_cot_training_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples")
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found!")
        exit(1)
    
    # Identify new examples (ones we added)
    original_examples = []
    new_examples = []
    
    for example in data:
        if get_example_age(example):
            new_examples.append(example)
        else:
            original_examples.append(example)
    
    print(f"\n📊 Analysis:")
    print(f"   Original examples (likely working): {len(original_examples)}")
    print(f"   New examples we added: {len(new_examples)}")
    print()
    
    # Check if original count matches expected
    if len(original_examples) < 135:
        print(f"⚠️  Warning: Only {len(original_examples)} original examples found")
        print(f"   Expected ~135-140. We may have misidentified some examples.")
        print()
        print("   Options:")
        print("   1. Keep all examples if original count is unclear")
        print("   2. Manually verify which examples are original")
        print()
        
        response = input("Proceed with restoration? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            exit(0)
    
    # Create backup first
    backup_file = f"rag_cot_training_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Created backup: {backup_file}")
    print()
    
    # Save restored dataset (original examples only)
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Restored original working examples")
    print(f"   Removed {len(new_examples)} new examples")
    print(f"   Kept {len(original_examples)} original examples")
    print(f"   Saved to: {output_file}")
    print()
    print("📋 Next steps:")
    print("   1. Regenerate merged dataset: python3 merge_cot_toggle_dataset.py")
    print("   2. Only conversational examples will be added for CoT toggle")
    print("   3. Original working RAG examples will be preserved exactly")
    print()
    print("=" * 80)
