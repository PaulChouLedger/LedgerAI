#!/usr/bin/env python3
"""
Identify DISCARD violations in training dataset.

This script finds examples where items marked [DISCARD] in REASONING
still appear in FINAL ANSWER. These violations need to be fixed manually.

Usage:
    python identify_discard_violations.py
    # Review the output and fix violations in rag_cot_training_dataset.json
"""

import json
import re
from typing import List, Dict, Tuple

def extract_discard_items(reasoning_section: str) -> List[str]:
    """Extract all items marked [DISCARD] from reasoning section."""
    discard_items = []
    lines = reasoning_section.split('\n')
    current_item = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for Item line
        if line_stripped.startswith('- Item:'):
            item_name = line_stripped.replace('- Item:', '').strip()
            current_item = item_name
        elif line_stripped.startswith('Item:'):
            item_name = line_stripped.replace('Item:', '').strip()
            current_item = item_name
        elif current_item and '[DISCARD]' in line_stripped:
            # Found DISCARD action for current item
            discard_items.append(current_item)
            current_item = None
        elif current_item and '[KEEP]' in line_stripped:
            # Found KEEP action, don't add to discard list
            current_item = None
        elif 'End of scan' in line_stripped or '- End of scan' in line_stripped:
            current_item = None
    
    return discard_items

def extract_keep_items(reasoning_section: str) -> List[str]:
    """Extract all items marked [KEEP] from reasoning section."""
    keep_items = []
    lines = reasoning_section.split('\n')
    current_item = None
    
    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('- Item:') or line_stripped.startswith('Item:'):
            item_name = line_stripped.replace('- Item:', '').replace('Item:', '').strip()
            current_item = item_name
        elif current_item and '[KEEP]' in line_stripped:
            keep_items.append(current_item)
            current_item = None
        elif current_item and '[DISCARD]' in line_stripped:
            current_item = None
        elif 'End of scan' in line_stripped or '- End of scan' in line_stripped:
            current_item = None
    
    return keep_items

def item_appears_in_text(item: str, text: str) -> bool:
    """Check if an item (or its significant parts) appears in text."""
    # Extract significant words (length > 3)
    item_words = [w for w in item.split() if len(w) > 3]
    
    if not item_words:
        # For short items (like dates, single words), check exact match
        item_lower = item.lower().strip()
        text_lower = text.lower()
        # Check for exact word boundaries
        pattern = r'\b' + re.escape(item_lower) + r'\b'
        return bool(re.search(pattern, text_lower))
    
    # For multi-word items, check if key parts appear
    for word in item_words:
        word_lower = word.lower()
        text_lower = text.lower()
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        if re.search(pattern, text_lower):
            return True
    
    return False

def find_discard_violations(dataset: List[Dict]) -> List[Dict]:
    """Find all examples with DISCARD violations."""
    violations = []
    
    for i, example in enumerate(dataset):
        assistant_msg = example['messages'][2]['content']
        
        if '[DISCARD]' not in assistant_msg:
            continue
        
        # Split into reasoning and FINAL ANSWER
        reasoning_section = ""
        final_answer_section = ""
        final_marker = None
        
        if "FINAL ANSWER:" in assistant_msg:
            parts = assistant_msg.split("FINAL ANSWER:", 1)
            reasoning_section = parts[0].strip()
            final_answer_section = parts[1].strip() if len(parts) > 1 else ""
            final_marker = "FINAL ANSWER:"
        elif "Final Answer:" in assistant_msg:
            parts = assistant_msg.split("Final Answer:", 1)
            reasoning_section = parts[0].strip()
            final_answer_section = parts[1].strip() if len(parts) > 1 else ""
            final_marker = "Final Answer:"
        else:
            continue
        
        if not final_answer_section:
            continue
        
        # Extract DISCARD and KEEP items
        discard_items = extract_discard_items(reasoning_section)
        keep_items = extract_keep_items(reasoning_section)
        
        if not discard_items:
            continue
        
        # Check if any DISCARD items appear in FINAL ANSWER
        violating_items = []
        for discard_item in discard_items:
            if item_appears_in_text(discard_item, final_answer_section):
                violating_items.append(discard_item)
        
        if violating_items:
            # Get query for context
            user_msg = example['messages'][1]['content']
            query = ""
            if "Question:" in user_msg:
                query = user_msg.split("Question:")[-1].strip()
            
            violations.append({
                'index': i,
                'query': query,
                'discard_items': discard_items,
                'keep_items': keep_items,
                'violating_items': violating_items,
                'reasoning_preview': reasoning_section[:500],
                'final_answer': final_answer_section,
                'final_answer_preview': final_answer_section[:300]
            })
    
    return violations

def print_violation_report(violations: List[Dict], output_file: str = None):
    """Print a detailed report of violations."""
    lines = []
    
    lines.append("=" * 100)
    lines.append("DISCARD VIOLATIONS REPORT")
    lines.append("=" * 100)
    lines.append(f"\nFound {len(violations)} examples with DISCARD violations")
    lines.append(f"These need to be fixed manually in rag_cot_training_dataset.json")
    lines.append("\n")
    
    for v in violations:
        lines.append("=" * 100)
        lines.append(f"VIOLATION #{v['index']} - Example Index: {v['index']}")
        lines.append("=" * 100)
        lines.append(f"\n📋 Query: {v['query']}")
        lines.append(f"\n📊 Summary:")
        lines.append(f"   • Total [DISCARD] items in REASONING: {len(v['discard_items'])}")
        lines.append(f"   • Total [KEEP] items in REASONING: {len(v['keep_items'])}")
        lines.append(f"   • DISCARD items that appear in FINAL ANSWER: {len(v['violating_items'])}")
        lines.append(f"\n❌ DISCARD Items (should NOT be in FINAL ANSWER):")
        for item in v['discard_items']:
            is_violating = item in v['violating_items']
            marker = "  ❌ VIOLATION" if is_violating else "  ✅ OK"
            lines.append(f"   {marker}: {item}")
        if v['keep_items']:
            lines.append(f"\n✅ KEEP Items (should be in FINAL ANSWER):")
            for item in v['keep_items']:
                lines.append(f"   • {item}")
        lines.append(f"\n📝 REASONING Section (first 500 chars):")
        lines.append("-" * 100)
        lines.append(v['reasoning_preview'])
        lines.append("-" * 100)
        lines.append(f"\n❌ FINAL ANSWER (contains DISCARD items):")
        lines.append("-" * 100)
        lines.append(v['final_answer'])
        lines.append("-" * 100)
        
        # Highlight violating items in FINAL ANSWER
        if v['violating_items']:
            lines.append(f"\n🔍 Violating Items Found in FINAL ANSWER:")
            final_lower = v['final_answer'].lower()
            for viol_item in v['violating_items']:
                # Find where it appears
                item_words = [w for w in viol_item.split() if len(w) > 3]
                for word in item_words:
                    if word.lower() in final_lower:
                        # Find context around the word
                        idx = final_lower.find(word.lower())
                        start = max(0, idx - 50)
                        end = min(len(v['final_answer']), idx + len(word) + 50)
                        context = v['final_answer'][start:end]
                        lines.append(f"   • '{viol_item}' appears near: ...{context}...")
        
        lines.append(f"\n💡 FIX REQUIRED:")
        lines.append(f"   Remove the following items from FINAL ANSWER:")
        for viol_item in v['violating_items']:
            lines.append(f"   - {viol_item}")
        lines.append(f"\n   Ensure FINAL ANSWER only contains [KEEP] items:")
        if v['keep_items']:
            for keep_item in v['keep_items']:
                lines.append(f"   + {keep_item}")
        else:
            lines.append(f"   + (No [KEEP] items - FINAL ANSWER should be minimal/empty)")
        lines.append("\n")
    
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"\nTotal violations found: {len(violations)}")
    lines.append(f"\nExample indices to fix: {[v['index'] for v in violations]}")
    lines.append(f"\nTo fix:")
    lines.append(f"1. Open rag_cot_training_dataset.json")
    lines.append(f"2. For each example index listed above, find the example")
    lines.append(f"3. Remove [DISCARD] items from FINAL ANSWER")
    lines.append(f"4. Ensure FINAL ANSWER only contains [KEEP] items")
    lines.append(f"5. If all items are [DISCARD], FINAL ANSWER should be minimal")
    lines.append("\n")
    
    # Print to console
    report_text = "\n".join(lines)
    print(report_text)
    
    # Save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✅ Report saved to: {output_file}")
    
    return report_text

def main():
    """Main function."""
    print("Loading dataset...")
    try:
        with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in dataset: {e}")
        return
    
    print(f"✅ Loaded {len(dataset)} examples")
    print("Analyzing for DISCARD violations...")
    print()
    
    violations = find_discard_violations(dataset)
    
    if not violations:
        print("✅ No DISCARD violations found!")
        return
    
    # Generate report
    print_violation_report(violations, output_file='DISCARD_VIOLATIONS_REPORT.txt')
    
    # Also create a simple index list for quick reference
    indices = [v['index'] for v in violations]
    print(f"\n📋 Quick Reference - Violation Indices:")
    print(f"   {indices}")
    print(f"\n   Total: {len(indices)} examples need fixing")

if __name__ == "__main__":
    main()
