#!/usr/bin/env python3
"""
Fix DISCARD violations in training dataset.

The issue: Some training examples have items marked [DISCARD] in REASONING
but those items still appear in FINAL ANSWER. This teaches the model to
violate the DISCARD rule.

Solution: Manually review and fix each violation to ensure DISCARD items
are properly excluded from FINAL ANSWER.
"""

import json
import re

def extract_discard_items(reasoning_section):
    """Extract items marked [DISCARD] from reasoning section."""
    discard_items = []
    lines = reasoning_section.split('\n')
    current_item = None
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('- Item:'):
            item_name = line_stripped.replace('- Item:', '').strip()
            current_item = item_name
        elif current_item and '[DISCARD]' in line_stripped:
            discard_items.append(current_item)
            current_item = None
        elif current_item and '[KEEP]' in line_stripped:
            current_item = None
        elif 'End of scan' in line_stripped:
            current_item = None
    
    return discard_items

def item_in_text(item, text):
    """Check if an item (or its key parts) appears in text."""
    item_words = [w for w in item.split() if len(w) > 3]
    if not item_words:
        return False
    
    # Check if key parts appear
    for word in item_words:
        if word.lower() in text.lower():
            # More precise: check if it's in a similar context
            # For now, simple word match is sufficient
            return True
    return False

def fix_discard_violation(example, index):
    """Fix a single example with DISCARD violation."""
    assistant_msg = example['messages'][2]['content']
    
    # Split into reasoning and FINAL ANSWER
    if "FINAL ANSWER:" in assistant_msg:
        reasoning_section = assistant_msg.split("FINAL ANSWER:")[0]
        final_answer_section = assistant_msg.split("FINAL ANSWER:")[-1]
        final_marker = "FINAL ANSWER:"
    elif "Final Answer:" in assistant_msg:
        reasoning_section = assistant_msg.split("Final Answer:")[0]
        final_answer_section = assistant_msg.split("Final Answer:")[-1]
        final_marker = "Final Answer:"
    else:
        return False, "No FINAL ANSWER marker found"
    
    # Extract DISCARD items
    discard_items = extract_discard_items(reasoning_section)
    if not discard_items:
        return False, "No DISCARD items found"
    
    # Check for violations
    violating_items = []
    for discard_item in discard_items:
        if item_in_text(discard_item, final_answer_section):
            violating_items.append(discard_item)
    
    if not violating_items:
        return False, "No violations found"
    
    # Strategy: For each violation, we need to carefully remove it from FINAL ANSWER
    # But we need to preserve the structure and readability
    
    # For now, log the violations so we can manually review
    print(f"\nExample {index}: {len(violating_items)} DISCARD violation(s)")
    print(f"  DISCARD items: {violating_items[:3]}")
    print(f"  FINAL ANSWER preview: {final_answer_section[:150]}...")
    
    # Try to fix automatically by removing violating items
    cleaned_final = final_answer_section
    
    for violating_item in violating_items:
        # Remove the item - be smart about it
        # For person names: Remove full name
        # For dates: Remove date
        # For amounts: Remove amount
        # For phrases: Remove phrase
        
        # Create a pattern to match the item
        # Escape special regex chars
        item_pattern = re.escape(violating_item)
        
        # Try exact match first (case-insensitive)
        cleaned_final = re.sub(item_pattern, '', cleaned_final, flags=re.IGNORECASE)
        
        # Also try removing if it's part of a list (e.g., "John Doe, Jane Smith" -> "Jane Smith")
        # Remove trailing commas and "and" before the item
        cleaned_final = re.sub(r',\s*' + item_pattern, '', cleaned_final, flags=re.IGNORECASE)
        cleaned_final = re.sub(r'\s+and\s+' + item_pattern, '', cleaned_final, flags=re.IGNORECASE)
        cleaned_final = re.sub(item_pattern + r'\s*,', '', cleaned_final, flags=re.IGNORECASE)
        cleaned_final = re.sub(item_pattern + r'\s+and\s+', '', cleaned_final, flags=re.IGNORECASE)
    
    # Clean up
    cleaned_final = re.sub(r'\s+', ' ', cleaned_final).strip()
    cleaned_final = re.sub(r'\s*,\s*,', ',', cleaned_final)
    cleaned_final = re.sub(r'\s*,\s*and\s*,', ', and', cleaned_final)
    cleaned_final = re.sub(r'^\s*,?\s*', '', cleaned_final)
    cleaned_final = re.sub(r'\s*,?\s*\.\s*$', '.', cleaned_final)
    
    # Verify fix
    still_violating = any(item_in_text(item, cleaned_final) for item in violating_items)
    
    if still_violating:
        return False, f"Still violating after fix: {violating_items[:2]}"
    
    # Reconstruct
    if final_marker == "FINAL ANSWER:":
        new_assistant_msg = reasoning_section.rstrip() + "\nFINAL ANSWER:\n" + cleaned_final
    else:
        new_assistant_msg = reasoning_section.rstrip() + "\nFinal Answer:\n" + cleaned_final
    
    example['messages'][2]['content'] = new_assistant_msg
    return True, f"Fixed: Removed {len(violating_items)} violating item(s)"

# Main
if __name__ == "__main__":
    print("=" * 100)
    print("FIXING DISCARD VIOLATIONS IN TRAINING DATASET")
    print("=" * 100)
    print()
    
    # Load dataset
    with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    violations_found = 0
    violations_fixed = 0
    violations_manual_review = []
    
    for i, example in enumerate(dataset):
        if '[DISCARD]' not in example['messages'][2]['content']:
            continue
        
        success, message = fix_discard_violation(example, i)
        
        if "violation" in message.lower() or success:
            violations_found += 1
            
            if success:
                violations_fixed += 1
                print(f"  ✅ {message}")
            else:
                violations_manual_review.append(i)
                print(f"  ⚠️  {message} - needs manual review")
    
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total examples: {len(dataset)}")
    print(f"Violations found: {violations_found}")
    print(f"Violations fixed: {violations_fixed}")
    print(f"Violations needing manual review: {len(violations_manual_review)}")
    
    if violations_manual_review:
        print(f"\n⚠️  Examples needing manual review: {violations_manual_review[:10]}")
    
    if violations_fixed > 0:
        # Save fixed dataset
        with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved fixed dataset: {violations_fixed} examples fixed")
    else:
        print(f"\n⚠️  No automatic fixes applied. All violations need manual review.")
