#!/usr/bin/env python3
"""
Simplify training dataset with step-by-step array-based approach.

This breaks down complex conditional logic into simple steps:
1. Build arrays (KEEP_ARRAY and DISCARD_ARRAY) based on verbatim evidence
2. Analyze arrays to build FINAL ANSWER (use only KEEP_ARRAY)

This reduces cognitive load and makes it easier for small models to learn.
"""

import json
import re

# New simplified system prompt with array-based approach
# IMPROVED: Emphasizes EXTRACTION first, arrays second
NEW_SYSTEM_PROMPT = """You are a data extraction bot. Extract items from context based on the query.

STEP 1: SCAN AND EXTRACT ALL items, then build arrays:
   - FIRST: Scan ALL chunks completely from start to finish to FIND ALL matching items.
   - CRITICAL: Continue scanning until you find ALL items. Do NOT stop after finding some items.
   - Initialize arrays at start: [KEEP_ARRAY] = [], [DISCARD_ARRAY] = []
   - For EACH item you FIND in the context:
     * Extract the item: Item: [name or thing]
     * Extract verbatim evidence: Evidence: "[exact quote from context]"
     * Decide: Action: [KEEP] or [DISCARD]
     * THEN add to array: → Add to [KEEP_ARRAY] if [KEEP], or [DISCARD_ARRAY] if [DISCARD]
   - Before writing "End of scan.", verify you found ALL matching items (e.g., if query asks for co-founders, check you found all co-founders, not just some).
   - Write "End of scan." after scanning all chunks completely.
   - List the arrays: [KEEP_ARRAY]: [...] and [DISCARD_ARRAY]: [...]

STEP 2: Build FINAL ANSWER:
   - Use ONLY items from [KEEP_ARRAY]
   - NEVER use items from [DISCARD_ARRAY]
   - Include ALL items from [KEEP_ARRAY] in FINAL ANSWER
   - If [KEEP_ARRAY] is empty, state "No [query items] found in the context."

CRITICAL RULES - DO NOT VIOLATE:

RULE 1 - COMPLETE SCANNING AND EXTRACTION (MOST IMPORTANT):
You MUST scan EVERY chunk from start to finish.
You MUST FIND and EXTRACT ALL items from the context.
Do NOT stop after finding one match - continue scanning until you find ALL items.
You must scan ALL chunks completely and extract ALL relevant items.
Before ending scan, verify: Did you find ALL matching items?
If the query asks for multiple items (e.g., "co-founders", "benefits", "locations"), extract ALL of them, not just some.

RULE 2 - ARRAY SEPARATION (MOST IMPORTANT):
Items in [DISCARD_ARRAY] are FORBIDDEN in FINAL ANSWER.
FINAL ANSWER uses ONLY [KEEP_ARRAY].
NEVER include items from [DISCARD_ARRAY] in FINAL ANSWER.

RULE 3 - ARRAY COMPLETENESS:
All [KEEP] items must be in [KEEP_ARRAY].
All [DISCARD] items must be in [DISCARD_ARRAY].
FINAL ANSWER must include ALL items from [KEEP_ARRAY].

RULE 4 - QUERY MATCHING:
Read the query word by word to understand what is being asked.
Extract ONLY items that match what the query asks for.
If the query asks for X, extract only X. Do NOT extract Y if query asks for X.
Opposites or different categories should be marked [DISCARD] and added to [DISCARD_ARRAY].

RULE 5 - MULTIPLE ATTRIBUTES:
Some items can have multiple attributes or roles.
Read the ENTIRE description completely before deciding.
If the item has the attribute that matches the query, mark [KEEP] and add to [KEEP_ARRAY].
If the item does NOT have the attribute that matches the query, mark [DISCARD] and add to [DISCARD_ARRAY]."""

def convert_reasoning_to_array_format(reasoning_text):
    """Convert existing reasoning format to array-based format."""
    lines = reasoning_text.split('\n')
    keep_items = []
    discard_items = []
    current_item = None
    current_evidence = None
    current_action = None
    
    new_reasoning = []
    new_reasoning.append("REASONING:")
    new_reasoning.append("")
    new_reasoning.append("[KEEP_ARRAY] = []")
    new_reasoning.append("[DISCARD_ARRAY] = []")
    new_reasoning.append("")
    
    for line in lines:
        line_stripped = line.strip()
        
        # Extract Item
        if '- Item:' in line or 'Item:' in line:
            item_match = re.search(r'[-\s]*Item:\s*([^\n-]+?)(?:\s*[-]|\s*Evidence|\s*$)', line, re.IGNORECASE)
            if item_match:
                current_item = item_match.group(1).strip()
                new_reasoning.append(f"- Item: {current_item}")
        
        # Extract Evidence
        elif 'Evidence:' in line or '- Evidence:' in line:
            evidence_match = re.search(r'[-\s]*Evidence:\s*"([^"]+)"', line, re.IGNORECASE)
            if evidence_match:
                current_evidence = evidence_match.group(1)
                new_reasoning.append(f"  Evidence: \"{current_evidence}\"")
            else:
                # Try without quotes
                evidence_match = re.search(r'[-\s]*Evidence:\s*(.+)', line, re.IGNORECASE)
                if evidence_match:
                    current_evidence = evidence_match.group(1).strip()
                    new_reasoning.append(f"  Evidence: \"{current_evidence}\"")
        
        # Extract Action
        elif '[KEEP]' in line or '[DISCARD]' in line:
            if '[KEEP]' in line:
                current_action = 'KEEP'
                new_reasoning.append(f"  Action: [KEEP]")
                if current_item:
                    keep_items.append(current_item)
                    new_reasoning.append(f"  → Added to [KEEP_ARRAY]")
            elif '[DISCARD]' in line:
                current_action = 'DISCARD'
                new_reasoning.append(f"  Action: [DISCARD]")
                if current_item:
                    discard_items.append(current_item)
                    new_reasoning.append(f"  → Added to [DISCARD_ARRAY]")
            new_reasoning.append("")
            current_item = None
            current_evidence = None
        
        # End of scan
        elif 'End of scan' in line or 'End scan' in line:
            new_reasoning.append("- End of scan.")
            new_reasoning.append("")
            break
    
    # Add array summaries
    new_reasoning.append(f"[KEEP_ARRAY]: {keep_items}")
    new_reasoning.append(f"[DISCARD_ARRAY]: {discard_items}")
    new_reasoning.append("")
    
    return '\n'.join(new_reasoning), keep_items, discard_items

def update_dataset_with_arrays(dataset_path, output_path):
    """Update dataset with array-based format."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    
    for i, example in enumerate(data):
        messages = example.get('messages', [])
        
        # Update system prompt
        for msg in messages:
            if msg.get('role') == 'system':
                msg['content'] = NEW_SYSTEM_PROMPT
                updated_count += 1
                break
        
        # Update assistant message with array format
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                
                # Split into reasoning and final answer
                if 'FINAL ANSWER' in content or 'Final Answer' in content:
                    # Find FINAL ANSWER marker
                    final_answer_marker = 'FINAL ANSWER' if 'FINAL ANSWER' in content else 'Final Answer'
                    parts = content.split(final_answer_marker, 1)
                    reasoning_section = parts[0].strip()
                    final_answer_section = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Convert reasoning to array format
                    new_reasoning, keep_items, discard_items = convert_reasoning_to_array_format(reasoning_section)
                    
                    # Update final answer to reference arrays
                    if keep_items or discard_items:
                        # Check if final answer already mentions arrays
                        if '[KEEP_ARRAY]' not in final_answer_section:
                            # Prepend array reference if there are KEEP items
                            if keep_items:
                                final_answer_section = f"Using items from [KEEP_ARRAY] only:\n{final_answer_section}"
                            elif not keep_items and discard_items:
                                # No KEEP items, only DISCARD
                                final_answer_section = f"No items found in [KEEP_ARRAY]. {final_answer_section}"
                    
                    # Reconstruct content
                    msg['content'] = f"{new_reasoning}\n\nFINAL ANSWER:\n{final_answer_section}"
                else:
                    # No FINAL ANSWER marker - try to convert anyway
                    new_reasoning, keep_items, discard_items = convert_reasoning_to_array_format(content)
                    if keep_items or discard_items:
                        final_answer_section = f"Using items from [KEEP_ARRAY] only:\n[Generate answer based on KEEP_ARRAY items]"
                        msg['content'] = f"{new_reasoning}\n\nFINAL ANSWER:\n{final_answer_section}"
                    break
    
    # Save updated dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updated_count} examples with array-based format")
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    dataset_path = "rag_cot_training_dataset.json"
    output_path = "rag_cot_training_dataset.json"  # Overwrite original
    
    print("=" * 80)
    print("SIMPLIFYING DATASET WITH ARRAY-BASED APPROACH")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Update system prompt with array-based instructions")
    print("  2. Convert reasoning to show [KEEP_ARRAY] and [DISCARD_ARRAY]")
    print("  3. Update FINAL ANSWER to reference arrays")
    print()
    
    update_dataset_with_arrays(dataset_path, output_path)
    
    print()
    print("✅ Dataset updated successfully!")
    print("   Ready for training with simplified array-based approach")
