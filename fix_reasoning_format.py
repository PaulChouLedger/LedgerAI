#!/usr/bin/env python3
"""
Fix examples with incorrect reasoning format - missing Evidence: line
Converts from:
  - Item: X | Role: Y | Action: Z
  OR
  - Item: X
  - Role: Y
  - Action: Z

To:
  - Item: X
  - Evidence: "[quote from context]"
  - Action: Z
"""

import json
import re

# Load dataset
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print("=" * 80)
print("FIXING REASONING FORMAT")
print("=" * 80)
print()

fixed_count = 0

for i, example in enumerate(dataset):
    assistant_msg = example['messages'][2]['content']
    user_msg = example['messages'][1]['content']
    
    # Check if this example needs fixing
    needs_fix = False
    
    # Pattern 1: Item: X | Role: Y | Action: Z
    if re.search(r'Item:.*\|.*Role:.*\|.*Action:', assistant_msg):
        needs_fix = True
        pattern_type = "Item|Role|Action"
    # Pattern 2: Has Item: and Role: but no Evidence:
    elif 'Item:' in assistant_msg and 'Role:' in assistant_msg and 'Evidence:' not in assistant_msg:
        needs_fix = True
        pattern_type = "Item+Role (no Evidence)"
    
    if not needs_fix:
        continue
    
    print(f"Fixing example {i} (pattern: {pattern_type})...")
    
    # Extract context from user message
    context_match = re.search(r'Knowledge context: (.*?)\n---', user_msg, re.DOTALL)
    if not context_match:
        context_match = re.search(r'Knowledge context: (.*?)(?:\n---|$)', user_msg, re.DOTALL)
    context = context_match.group(1) if context_match else ""
    
    # Split reasoning into lines
    lines = assistant_msg.split('\n')
    new_lines = []
    i_line = 0
    
    while i_line < len(lines):
        line = lines[i_line]
        
        # Check if this is a line with the wrong format
        if re.search(r'Item:.*\|.*Role:.*\|.*Action:', line):
            # Extract Item, Role, Action
            match = re.search(r'Item: ([^|]+) \| Role: "([^"]+)" \| Action: (\[KEEP\]|\[DISCARD\].*?)(?:\.|$)', line)
            if match:
                item = match.group(1).strip()
                role_quote = match.group(2)
                action = match.group(3).strip()
                
                # Find evidence in context
                # Try to find the person/item in context
                evidence = f'"{role_quote}"'
                if item in context:
                    # Try to find a better quote
                    item_pattern = re.escape(item)
                    evidence_match = re.search(f'{item_pattern}.*?{re.escape(role_quote)}', context, re.IGNORECASE)
                    if evidence_match:
                        evidence = f'"{evidence_match.group(0)}"'
                
                new_lines.append(f"- Item: {item}")
                new_lines.append(f"- Evidence: {evidence}")
                new_lines.append(f"- Action: {action}")
                i_line += 1
                continue
        
        # Check if this is Item: followed by Role: (separate lines)
        if line.strip().startswith('- Item:') and i_line + 1 < len(lines):
            next_line = lines[i_line + 1]
            if next_line.strip().startswith('- Role:') and 'Evidence:' not in assistant_msg:
                item_match = re.search(r'Item: (.+)', line)
                role_match = re.search(r'Role: (.+)', next_line)
                
                if item_match and role_match:
                    item = item_match.group(1).strip()
                    role = role_match.group(1).strip()
                    
                    # Find evidence in context
                    evidence = f'"{role}"'
                    if item in context:
                        item_pattern = re.escape(item)
                        evidence_match = re.search(f'{item_pattern}.*?{re.escape(role)}', context, re.IGNORECASE)
                        if evidence_match:
                            evidence = f'"{evidence_match.group(0)}"'
                    
                    new_lines.append(f"- Item: {item}")
                    new_lines.append(f"- Evidence: {evidence}")
                    # Skip the Role line and get Action from next line
                    i_line += 2
                    if i_line < len(lines) and lines[i_line].strip().startswith('- Action:'):
                        new_lines.append(lines[i_line].strip())
                        i_line += 1
                    continue
        
        # Keep original line
        new_lines.append(line)
        i_line += 1
    
    # Reconstruct assistant message
    new_assistant_msg = '\n'.join(new_lines)
    
    # Update the example
    dataset[i]['messages'][2]['content'] = new_assistant_msg
    fixed_count += 1
    
    if fixed_count <= 2:
        print(f"  Before: {assistant_msg[:200]}...")
        print(f"  After:  {new_assistant_msg[:200]}...")
        print()

print(f"✅ Fixed {fixed_count} examples")
print()

# Save updated dataset
print("Saving updated dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("✅ Dataset saved!")
print()
print("Verifying fixes...")

# Verify all examples now have proper format
all_good = True
for i, example in enumerate(dataset):
    assistant_msg = example['messages'][2]['content']
    
    if 'REASONING:' in assistant_msg:
        # Check for wrong format
        if re.search(r'Item:.*\|.*Role:.*\|.*Action:', assistant_msg):
            print(f"❌ Example {i} still has wrong format")
            all_good = False
        elif 'Item:' in assistant_msg and 'Evidence:' not in assistant_msg and 'Role:' in assistant_msg:
            print(f"❌ Example {i} still missing Evidence:")
            all_good = False

if all_good:
    print("✅ All examples now have proper reasoning format!")
else:
    print("⚠️  Some examples still need fixing")
