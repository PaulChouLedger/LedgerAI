#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge RAG+CoT Dataset with Conversational Dataset
Creates a mixed dataset for training conditional CoT behavior
"""

import json
import random
import sys

def load_dataset(filepath):
    """Load a JSON dataset file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)

def merge_datasets(rag_cot_dataset, conversational_dataset, ratio=0.5, max_total_examples=None):
    """
    Merge RAG+CoT dataset with conversational dataset
    
    Args:
        rag_cot_dataset: List of RAG+CoT examples (with CoT system prompt)
        conversational_dataset: List of conversational examples (without CoT)
        ratio: Ratio of RAG+CoT examples (0.5 = 50/50, 0.6 = 60/40 RAG/Conversational)
        max_total_examples: Maximum total examples (None = use all, set to limit dataset size)
    
    Returns:
        Merged and shuffled dataset
    """
    print(f"📊 Dataset Statistics:")
    print(f"   RAG+CoT examples: {len(rag_cot_dataset)}")
    print(f"   Conversational examples: {len(conversational_dataset)}")
    print()
    
    # Calculate target sizes
    if max_total_examples:
        # Limit total size while maintaining ratio
        target_rag = int(max_total_examples * ratio)
        target_conv = max_total_examples - target_rag
        # Don't exceed available examples
        target_rag = min(target_rag, len(rag_cot_dataset))
        target_conv = min(target_conv, len(conversational_dataset))
    else:
        total_examples = len(rag_cot_dataset) + len(conversational_dataset)
        target_rag = int(total_examples * ratio)
        target_conv = total_examples - target_rag
    
    # Adjust if we don't have enough examples
    if len(rag_cot_dataset) < target_rag:
        print(f"⚠️  Warning: Only {len(rag_cot_dataset)} RAG+CoT examples available, using all of them")
        target_rag = len(rag_cot_dataset)
        target_conv = len(conversational_dataset)
    elif len(conversational_dataset) < target_conv:
        print(f"⚠️  Warning: Only {len(conversational_dataset)} conversational examples available, using all of them")
        target_conv = len(conversational_dataset)
        target_rag = len(rag_cot_dataset)
    
    # Sample datasets to match target sizes
    # Prioritize complex examples (those with longer contexts or multiple chunks)
    if len(rag_cot_dataset) > target_rag:
        # Sort by context length (longer = more complex, prioritize these)
        def get_context_length(example):
            messages = example.get("messages", [])
            user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
            if user_msg:
                content = user_msg.get("content", "")
                # Count chunks (separated by ---)
                chunks = content.split("---")
                return len(content), len(chunks)
            return 0, 0
        
        # Sort by complexity (length, then number of chunks)
        sorted_rag = sorted(rag_cot_dataset, key=get_context_length, reverse=True)
        # Take top complex examples first, then random sample from rest
        complex_count = min(10, target_rag // 2)  # Take top 10 complex examples
        complex_examples = sorted_rag[:complex_count]
        remaining = sorted_rag[complex_count:]
        remaining_needed = target_rag - complex_count
        if remaining_needed > 0:
            remaining_samples = random.sample(remaining, min(remaining_needed, len(remaining)))
            rag_cot_samples = complex_examples + remaining_samples
        else:
            rag_cot_samples = complex_examples[:target_rag]
        print(f"   Prioritized {complex_count} complex examples (longer contexts, multiple chunks)")
    else:
        rag_cot_samples = rag_cot_dataset
    
    if len(conversational_dataset) > target_conv:
        conv_samples = random.sample(conversational_dataset, target_conv)
    else:
        conv_samples = conversational_dataset
    
    # Merge
    merged = rag_cot_samples + conv_samples
    
    # Shuffle to mix RAG and conversational examples
    random.shuffle(merged)
    
    print(f"✅ Merged Dataset:")
    print(f"   Total examples: {len(merged)}")
    print(f"   RAG+CoT examples: {len(rag_cot_samples)} ({len(rag_cot_samples)/len(merged)*100:.1f}%)")
    print(f"   Conversational examples: {len(conv_samples)} ({len(conv_samples)/len(merged)*100:.1f}%)")
    print()
    
    return merged

def verify_dataset(dataset):
    """Verify dataset structure and count CoT vs non-CoT examples"""
    cot_count = 0
    non_cot_count = 0
    
    for example in dataset:
        messages = example.get("messages", [])
        if not messages:
            continue
        
        # Check system prompt for CoT instructions
        system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)
        if system_msg:
            content = system_msg.get("content", "")
            if "REASONING:" in content or "Start with REASONING" in content:
                cot_count += 1
            else:
                non_cot_count += 1
    
    print(f"🔍 Dataset Verification:")
    print(f"   Examples with CoT system prompt: {cot_count}")
    print(f"   Examples without CoT (conversational): {non_cot_count}")
    print(f"   Total: {cot_count + non_cot_count}")
    print()
    
    return cot_count, non_cot_count

if __name__ == "__main__":
    print("=" * 80)
    print("Merge RAG+CoT Dataset with Conversational Dataset")
    print("=" * 80)
    print()
    
    # File paths
    rag_cot_file = "rag_cot_training_dataset.json"
    conversational_file = "conversational_dataset.json"
    output_file = "rag_cot_toggle_training_dataset.json"
    
    # Load datasets
    print("📂 Loading datasets...")
    rag_cot_dataset = load_dataset(rag_cot_file)
    conversational_dataset = load_dataset(conversational_file)
    print()
    
    # Merge datasets (50/50 ratio, limit to ~280 examples for faster training)
    # Updated: 140 RAG examples (135 original + 5 new complex examples)
    # Target: ~280 total (140 RAG + 140 conversational) for balanced 50/50 split
    # This keeps training time reasonable (~1-2 hours instead of 4-5 hours)
    # Training time scales with dataset size: 135 examples = ~30 min, 280 = ~1-1.5 hours, 2135 = ~5 hours
    max_total = 280  # 2x the RAG dataset size for balanced training (includes new complex examples)
    print(f"📊 Limiting dataset to {max_total} examples (140 RAG + 140 conversational)")
    print(f"   Includes 5 new complex multi-chunk examples to improve extraction accuracy")
    print(f"   This reduces training time from ~5 hours to ~1-1.5 hours")
    print()
    merged_dataset = merge_datasets(rag_cot_dataset, conversational_dataset, ratio=0.5, max_total_examples=max_total)
    
    # Verify
    cot_count, non_cot_count = verify_dataset(merged_dataset)
    
    # Save merged dataset
    print(f"💾 Saving merged dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(merged_dataset)} examples to {output_file}")
    print()
    
    # Show samples
    print("📋 Sample Examples:")
    print()
    print("1. RAG+CoT Example:")
    rag_example = next((ex for ex in merged_dataset if "REASONING:" in str(ex)), None)
    if rag_example:
        print(json.dumps(rag_example, indent=2)[:500] + "...")
    print()
    
    print("2. Conversational Example:")
    conv_example = next((ex for ex in merged_dataset if "REASONING:" not in str(ex)), None)
    if conv_example:
        print(json.dumps(conv_example, indent=2)[:500] + "...")
    print()
    
    print("=" * 80)
    print("✅ Dataset merge complete!")
    print(f"   Ready for training with: {output_file}")
    print("=" * 80)
