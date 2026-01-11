#!/usr/bin/env python3
"""
Simplify the overly complex system prompt in rag_cot_training_dataset.json
Reduce from 942 words to ~130 words while keeping essential instructions
"""

import json

# Simplified system prompt (7.5x shorter)
SIMPLIFIED_PROMPT = """You are a precise data extraction bot. Extract items from context based on the query.

PROCESS:
1. Start with REASONING:
2. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
3. End scan with: - End of scan.
4. Provide FINAL ANSWER: based ONLY on [KEEP] items.

RULES:
- Scan ALL chunks completely - do NOT stop after finding some matches.
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must include ALL items marked [KEEP].
- Read the query carefully - extract ONLY items that match the query intent.
- Items can have multiple attributes - check ALL attributes to see if ANY match the query."""

print("=" * 80)
print("SIMPLIFYING SYSTEM PROMPT")
print("=" * 80)
print()

# Load dataset
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"📊 Loaded {len(dataset)} examples")
print()

# Check current prompt
current_prompt = dataset[0]['messages'][0]['content']
print(f"Current prompt: {len(current_prompt)} chars, {len(current_prompt.split())} words")
print(f"Simplified prompt: {len(SIMPLIFIED_PROMPT)} chars, {len(SIMPLIFIED_PROMPT.split())} words")
print(f"Reduction: {len(current_prompt)/len(SIMPLIFIED_PROMPT):.1f}x shorter")
print()

# Update all examples
print("Updating all examples...")
updated_count = 0
for i, example in enumerate(dataset):
    example['messages'][0]['content'] = SIMPLIFIED_PROMPT
    updated_count += 1
    if (i + 1) % 50 == 0:
        print(f"  Updated {i + 1}/{len(dataset)} examples...")

print(f"✅ Updated {updated_count} examples")
print()

# Save updated dataset
print("Saving updated dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("✅ Dataset saved with simplified system prompt!")
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  ✅ System prompt reduced: {len(current_prompt)} → {len(SIMPLIFIED_PROMPT)} chars")
print(f"  ✅ Word count reduced: {len(current_prompt.split())} → {len(SIMPLIFIED_PROMPT.split())} words")
print(f"  ✅ Examples updated: {updated_count}")
print()
print("💡 The model will now focus on learning reasoning patterns from examples")
print("   rather than memorizing the prompt structure!")
print()
print("Query location: In user message after 'Question:'")
print("   Format: 'Knowledge context: ... ---\\nQuestion: ...'")
