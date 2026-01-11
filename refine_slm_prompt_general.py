#!/usr/bin/env python3
"""
Refine SLM-optimized prompt to use generalizable patterns instead of specific examples.
This prevents tunnel vision - model learns the principle, not just specific patterns.
"""

import json

# Refined prompt - generalizable patterns (no specific examples that could cause tunnel vision)
SLM_GENERALIZED_PROMPT = """You are a data extraction bot. Extract items from context based on the query.

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
Read the query word by word to understand what is being asked.
Extract ONLY items that match what the query asks for.
If the query asks for X, extract only X. Do NOT extract Y if query asks for X.
Opposites or different categories should be marked [DISCARD].

RULE 5 - MULTIPLE ATTRIBUTES:
Some items can have multiple attributes or roles.
Read the ENTIRE description completely before deciding.
If the item has the attribute that matches the query, mark [KEEP].
If the item does NOT have the attribute that matches the query, mark [DISCARD]."""

print("=" * 80)
print("REFINING SLM PROMPT: REMOVING SPECIFIC EXAMPLES")
print("=" * 80)
print()

print("Key Changes:")
print("  1. RULE 4: Removed specific examples (benefits/drawbacks, products/services)")
print("     → Now uses general principle: 'If query asks for X, extract only X'")
print()
print("  2. RULE 5: Removed specific examples (co-founder/CFO)")
print("     → Now uses general principle: 'If item has attribute that matches query, mark [KEEP]'")
print()
print("Why this is better:")
print("  ✅ Prevents tunnel vision - works for ANY query type")
print("  ✅ Model learns the PRINCIPLE, not specific patterns")
print("  ✅ Generalizes to: revenue, technologies, locations, dates, etc.")
print("  ✅ Training dataset already has diverse examples")
print()

# Load dataset
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"📊 Loaded {len(dataset)} examples")
print()

# Update all examples
print("Updating all examples with generalized prompt...")
updated_count = 0
for i, example in enumerate(dataset):
    example['messages'][0]['content'] = SLM_GENERALIZED_PROMPT
    updated_count += 1
    if (i + 1) % 50 == 0:
        print(f"  Updated {i + 1}/{len(dataset)} examples...")

print(f"✅ Updated {updated_count} examples")
print()

# Save updated dataset
print("Saving updated dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("✅ Dataset saved with generalized SLM-optimized prompt!")
print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print()
print("The prompt now:")
print("  ✅ Uses generalizable patterns (works for all query types)")
print("  ✅ Avoids tunnel vision (no specific examples in prompt)")
print("  ✅ Still SLM-optimized (numbered rules, repetition, concrete language)")
print("  ✅ DISCARD rule still emphasized 3x (critical rule)")
print("  ✅ Training dataset provides specific examples (diverse coverage)")
print()
print("This balances:")
print("  - SLM comprehension (explicit, numbered, repeated)")
print("  - Generalization (principle-based, not example-based)")
print("  - Avoiding tunnel vision (works for ANY query type)")
