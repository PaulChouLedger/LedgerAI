#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Training Examples Have Complete FINAL ANSWERs
Checks that all [KEEP] items from reasoning appear in FINAL ANSWER
"""

import json
import re

def extract_keep_items(reasoning_text):
    """Extract all items marked [KEEP] from reasoning"""
    keep_items = []
    # Pattern to match: - Item: [name] ... Action: [KEEP]
    pattern = r'- Item:\s*([^\n-]+?)(?:\s*-\s*[^\n-]+?)*\s*Action:\s*\[KEEP\]'
    matches = re.findall(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        # Extract just the name (first part before any dashes or colons)
        name = match.split('-')[0].split(':')[0].strip()
        # Clean up common prefixes
        name = re.sub(r'^(Item|Role|Evidence):\s*', '', name, flags=re.IGNORECASE).strip()
        if name:
            keep_items.append(name)
    return keep_items

def extract_final_answer_items(final_answer_text):
    """Extract items mentioned in FINAL ANSWER"""
    # Simple extraction - look for names (capitalized words)
    # This is a heuristic - may need refinement
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', final_answer_text)
    return names

if __name__ == "__main__":
    print("=" * 80)
    print("Verifying Complete FINAL ANSWERs in Training Dataset")
    print("=" * 80)
    print()
    
    # Load dataset
    try:
        with open("rag_cot_training_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples")
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found!")
        exit(1)
    
    # Check CoT examples
    cot_examples = []
    for example in data:
        messages = example.get("messages", [])
        assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
        if assistant_msg and "REASONING:" in assistant_msg.get("content", ""):
            cot_examples.append(example)
    
    print(f"📊 Found {len(cot_examples)} CoT examples")
    print()
    
    # Verify each example
    incomplete_count = 0
    for idx, example in enumerate(cot_examples):
        messages = example.get("messages", [])
        assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
        if not assistant_msg:
            continue
        
        content = assistant_msg.get("content", "")
        
        # Extract reasoning and final answer
        if "FINAL ANSWER:" in content:
            reasoning_text = content.split("FINAL ANSWER:")[0]
            final_answer_text = content.split("FINAL ANSWER:")[-1]
        elif "Final Answer:" in content:
            reasoning_text = content.split("Final Answer:")[0]
            final_answer_text = content.split("Final Answer:")[-1]
        else:
            continue
        
        # Extract KEEP items from reasoning
        keep_items = extract_keep_items(reasoning_text)
        
        # Check if all KEEP items appear in FINAL ANSWER
        missing_items = []
        for item in keep_items:
            # Check if item name appears in final answer (case-insensitive, partial match)
            item_lower = item.lower()
            final_lower = final_answer_text.lower()
            # Check for full name or first+last name
            name_parts = item_lower.split()
            if len(name_parts) >= 2:
                # Check if both first and last name appear
                if not (name_parts[0] in final_lower and name_parts[-1] in final_lower):
                    missing_items.append(item)
            else:
                # Single name - check if it appears
                if item_lower not in final_lower:
                    missing_items.append(item)
        
        if missing_items:
            incomplete_count += 1
            if incomplete_count <= 5:  # Show first 5 issues
                print(f"⚠️  Example {idx+1}: Missing items in FINAL ANSWER")
                print(f"   KEEP items: {keep_items}")
                print(f"   Missing: {missing_items}")
                print(f"   FINAL ANSWER: {final_answer_text[:150]}...")
                print()
    
    print(f"📊 Verification Results:")
    print(f"   Total CoT examples: {len(cot_examples)}")
    print(f"   Examples with incomplete FINAL ANSWER: {incomplete_count}")
    print(f"   Examples with complete FINAL ANSWER: {len(cot_examples) - incomplete_count}")
    print()
    
    if incomplete_count == 0:
        print("✅ All examples have complete FINAL ANSWERs!")
    else:
        print(f"⚠️  {incomplete_count} examples need fixing")
    
    print("=" * 80)
