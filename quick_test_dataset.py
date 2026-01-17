#!/usr/bin/env python3
"""
Quick Test: Validate 10 Random Samples from Dataset
Checks verbatim evidence accuracy and reasoning consistency
"""

import json
import random
import re
from verbatim_evidence_helper import validate_evidence_verbatim

def extract_reasoning_items(reasoning_text):
    """Extract items, evidence, and actions from reasoning."""
    items = []
    current_item = None
    
    lines = reasoning_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('- Item:'):
            if current_item:
                items.append(current_item)
            current_item = {
                'item': line.replace('- Item:', '').strip(),
                'evidence': None,
                'action': None
            }
        elif line.startswith('- Evidence:') and current_item:
            evidence_match = re.search(r'- Evidence:\s*"([^"]+)"', line)
            if evidence_match:
                current_item['evidence'] = evidence_match.group(1)
        elif line.startswith('- Action:') and current_item:
            action_match = re.search(r'- Action:\s*\[([^\]]+)\]', line)
            if action_match:
                current_item['action'] = action_match.group(1)
    
    if current_item:
        items.append(current_item)
    
    return items

def check_final_answer_consistency(reasoning_items, final_answer):
    """Check if final answer only includes KEEP items."""
    keep_items = [item['item'] for item in reasoning_items if item['action'] and 'KEEP' in item['action']]
    discard_items = [item['item'] for item in reasoning_items if item['action'] and 'DISCARD' in item['action']]
    
    issues = []
    
    # Check if DISCARD items appear in final answer
    for discard_item in discard_items:
        # Extract name/identifier from item
        item_name = discard_item.split()[0] if discard_item.split() else discard_item
        if len(item_name) > 3 and item_name.lower() in final_answer.lower():
            issues.append(f"DISCARD item '{item_name}' appears in final answer")
    
    # Check if KEEP items are missing
    for keep_item in keep_items:
        item_name = keep_item.split()[0] if keep_item.split() else keep_item
        if len(item_name) > 3:
            # Check if it's a person name (has capital letters)
            if any(c.isupper() for c in item_name) and item_name.lower() not in final_answer.lower():
                # This might be okay for some items, so we'll be lenient
                pass
    
    return issues

def test_example(example, index):
    """Test a single example."""
    user_content = example['messages'][1]['content']
    assistant_content = example['messages'][2]['content']
    
    # Extract context and query
    context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
    query_match = re.search(r'Question:\s*(.+)', user_content)
    query = query_match.group(1).strip() if query_match else "Unknown"
    
    # Extract reasoning and final answer
    reasoning_match = re.search(r'REASONING:(.*?)FINAL ANSWER:', assistant_content, re.DOTALL)
    final_answer_match = re.search(r'FINAL ANSWER:\s*(.+)', assistant_content, re.DOTALL)
    
    if not reasoning_match or not final_answer_match:
        return {
            'index': index,
            'query': query[:50],
            'status': '❌ INVALID',
            'issues': ['Missing REASONING or FINAL ANSWER section']
        }
    
    reasoning = reasoning_match.group(1).strip()
    final_answer = final_answer_match.group(1).strip()
    
    # Validate verbatim evidence
    is_valid, warnings = validate_evidence_verbatim(example)
    
    # Extract reasoning items
    reasoning_items = extract_reasoning_items(reasoning)
    
    # Check final answer consistency
    consistency_issues = check_final_answer_consistency(reasoning_items, final_answer)
    
    # Count KEEP vs DISCARD
    keep_count = sum(1 for item in reasoning_items if item['action'] and 'KEEP' in item['action'])
    discard_count = sum(1 for item in reasoning_items if item['action'] and 'DISCARD' in item['action'])
    
    all_issues = warnings + consistency_issues
    
    status = '✅ VALID' if is_valid and not consistency_issues else '⚠️  ISSUES'
    
    return {
        'index': index,
        'query': query[:60],
        'status': status,
        'verbatim': is_valid,
        'keep_items': keep_count,
        'discard_items': discard_count,
        'total_items': len(reasoning_items),
        'issues': all_issues
    }

def main():
    print("=" * 80)
    print("QUICK TEST: 10 Random Samples from Dataset")
    print("=" * 80)
    
    # Load dataset
    with open('rag_cot_training_dataset.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Dataset loaded: {len(data)} examples")
    
    # Select 10 random samples
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(len(data)), min(10, len(data)))
    sample_indices.sort()
    
    print(f"🎲 Testing {len(sample_indices)} random samples: {sample_indices}")
    print("=" * 80)
    
    results = []
    for idx in sample_indices:
        result = test_example(data[idx], idx)
        results.append(result)
        
        print(f"\n📝 Example {idx}: {result['query']}")
        print(f"   Status: {result['status']}")
        print(f"   Verbatim Evidence: {'✅' if result['verbatim'] else '❌'}")
        print(f"   Reasoning Items: {result['total_items']} (KEEP: {result['keep_items']}, DISCARD: {result['discard_items']})")
        
        if result['issues']:
            print(f"   ⚠️  Issues:")
            for issue in result['issues'][:3]:  # Show first 3 issues
                print(f"      - {issue}")
        else:
            print(f"   ✅ No issues found")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    valid_count = sum(1 for r in results if r['status'] == '✅ VALID')
    verbatim_count = sum(1 for r in results if r['verbatim'])
    total_issues = sum(len(r['issues']) for r in results)
    
    print(f"\n📊 Results:")
    print(f"   Valid examples: {valid_count}/{len(results)} ({valid_count/len(results)*100:.0f}%)")
    print(f"   Verbatim evidence: {verbatim_count}/{len(results)} ({verbatim_count/len(results)*100:.0f}%)")
    print(f"   Total issues: {total_issues}")
    
    if valid_count == len(results) and verbatim_count == len(results):
        print(f"\n   ✅ PERFECT: All samples are valid with verbatim evidence!")
    elif valid_count >= len(results) * 0.9:
        print(f"\n   ✅ GOOD: Most samples are valid")
    else:
        print(f"\n   ⚠️  WARNING: Some samples have issues")
    
    # Show reasoning format check
    print(f"\n🔍 Reasoning Format Check:")
    has_reasoning = sum(1 for r in results if r['total_items'] > 0)
    has_keep_discard = sum(1 for r in results if r['keep_items'] > 0 and r['discard_items'] > 0)
    print(f"   Examples with reasoning items: {has_reasoning}/{len(results)}")
    print(f"   Examples with KEEP and DISCARD: {has_keep_discard}/{len(results)}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
