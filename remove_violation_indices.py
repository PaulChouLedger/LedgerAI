#!/usr/bin/env python3
"""
Remove DISCARD violation examples from training dataset.

Keeps the important real-world examples (indices 0-5) and removes
the 28 examples with DISCARD violations.
"""

import json

# Indices to remove (violations found)
violation_indices = [63, 68, 82, 83, 85, 86, 87, 91, 106, 108, 109, 111, 112, 115, 116, 117, 119, 121, 124, 125, 126, 129, 132, 135, 141, 163, 166, 170]

# Important examples to keep (real-world examples)
keep_indices = [0, 1, 2, 3, 4, 5]

print("=" * 100)
print("REMOVING DISCARD VIOLATION EXAMPLES")
print("=" * 100)
print()

# Load dataset
print("Loading dataset...")
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

original_count = len(dataset)
print(f"✅ Loaded {original_count} examples")
print()

# Filter out violation indices (but keep 0-5)
print("Removing violation examples...")
cleaned_dataset = []

for i, example in enumerate(dataset):
    if i in violation_indices:
        print(f"  ❌ Removing example {i} (violation)")
    elif i in keep_indices:
        print(f"  ✅ Keeping example {i} (important real-world example)")
        cleaned_dataset.append(example)
    else:
        # Keep all other examples
        cleaned_dataset.append(example)

new_count = len(cleaned_dataset)
removed_count = original_count - new_count

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Original examples: {original_count}")
print(f"Removed examples: {removed_count}")
print(f"Remaining examples: {new_count}")
print(f"Important examples kept (0-5): ✅")
print()

# Save cleaned dataset
print("Saving cleaned dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Saved cleaned dataset: {new_count} examples")
print()
print("⚠️  Note: The dataset has been updated. Make sure to:")
print("   1. Verify examples 0-5 are still present")
print("   2. Retrain the model with the cleaned dataset")
print("   3. Test the new model")
