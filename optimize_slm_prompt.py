#!/usr/bin/env python3
"""
Optimize system prompt for Small Language Model (SLM) comprehension.
SLMs (1.5B params) need: explicit instructions, concrete language, repetition, action verbs.
"""

import json

# SLM-optimized prompt - optimal for 1.5B parameter models
SLM_OPTIMIZED_PROMPT = """You are a data extraction bot. Extract items from context based on the query.

STEP 1: Start with REASONING:
STEP 2: For EACH item found, write:
   - Item: [name or thing]
   - Evidence: "[exact quote]"
   - Action: [KEEP] or [DISCARD]
STEP 3: Write "End of scan."
STEP 4: Write FINAL ANSWER using ONLY [KEEP] items.

CRITICAL RULES - DO NOT VIOLATE:

RULE 1 - COMPLETE SCANNING:
Scan EVERY chunk from start to finish.
Do NOT stop after finding one match.
You must scan ALL chunks completely.

RULE 2 - DISCARD ITEMS (MOST IMPORTANT):
If you write [DISCARD] for an item, that item MUST NOT appear in FINAL ANSWER.
[DISCARD] items are FORBIDDEN in FINAL ANSWER.
Never write a [DISCARD] item in FINAL ANSWER.

RULE 3 - KEEP ITEMS:
If you write [KEEP] for an item, that item MUST appear in FINAL ANSWER.
Count how many [KEEP] items you have.
Include ALL [KEEP] items in FINAL ANSWER.

RULE 4 - QUERY MATCHING:
Read the query word by word.
"benefits" means benefits only, NOT drawbacks.
"products" means products only, NOT services.
"co-founders" means co-founders only, NOT other roles.
If query says "benefits", mark drawbacks as [DISCARD].

RULE 5 - MULTIPLE ATTRIBUTES:
Some items have multiple attributes (e.g., "CEO and Co-Founder").
Read the ENTIRE description.
If the query asks for "co-founder" and the item says "CEO and Co-Founder", mark [KEEP].
If the query asks for "CFO" and the item says "CEO and Co-Founder", mark [DISCARD]."""

print("=" * 80)
print("SLM-OPTIMIZED SYSTEM PROMPT")
print("=" * 80)
print()
print("Key optimizations for SLM (1.5B params):")
print()
print("1. ✅ STEP-BY-STEP structure (numbered steps)")
print("2. ✅ NUMBERED RULES (RULE 1, RULE 2) - easier for SLMs to parse")
print("3. ✅ DISCARD rule repeated 3 times with 'FORBIDDEN' - critical emphasis")
print("4. ✅ Concrete examples instead of abstract 'query intent'")
print("5. ✅ Active imperative verbs ('Do NOT stop', 'You must scan')")
print("6. ✅ Explicit negative examples ('NOT drawbacks', 'NOT services')")
print("7. ✅ Simple sentence structure (subject-verb-object)")
print("8. ✅ Explicit counting instruction ('Count how many [KEEP] items')")
print("9. ✅ 'MOST IMPORTANT' label on DISCARD rule")
print("10. ✅ Short sentences (average 6-10 words)")
print()
print("Word count:", len(SLM_OPTIMIZED_PROMPT.split()))
print("Character count:", len(SLM_OPTIMIZED_PROMPT))
print()

# Load dataset
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"📊 Loaded {len(dataset)} examples")
print()

# Update all examples
print("Updating all examples with SLM-optimized prompt...")
updated_count = 0
for i, example in enumerate(dataset):
    example['messages'][0]['content'] = SLM_OPTIMIZED_PROMPT
    updated_count += 1
    if (i + 1) % 50 == 0:
        print(f"  Updated {i + 1}/{len(dataset)} examples...")

print(f"✅ Updated {updated_count} examples")
print()

# Save updated dataset
print("Saving updated dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("✅ Dataset saved with SLM-optimized system prompt!")
print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print()
print("The prompt is now optimized for Small Language Model comprehension:")
print("  - Explicit, numbered rules (easier parsing)")
print("  - DISCARD rule emphasized 3x with 'FORBIDDEN'")
print("  - Concrete examples instead of abstract concepts")
print("  - Active imperative verbs (clearer instructions)")
print("  - Simple sentence structure (better understanding)")
print()
print("This should help the 1.5B parameter model learn:")
print("  ✅ DISCARD enforcement (critical issue)")
print("  ✅ Complete scanning (early stopping issue)")
print("  ✅ Query intent matching (benefits vs drawbacks)")
print("  ✅ Multiple item extraction (stopping after first match)")
