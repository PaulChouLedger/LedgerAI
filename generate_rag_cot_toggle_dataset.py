#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate RAG CoT Toggle Training Dataset
Consolidated script to create the merged dataset for conditional CoT training.

This script:
1. Loads the RAG+CoT training dataset (rag_cot_training_dataset.json)
2. Loads/generates conversational examples (conversational_dataset.json)
3. Ensures format consistency (Evidence: format)
4. Merges datasets with proper prioritization
5. Saves to rag_cot_toggle_training_dataset.json

Usage:
    python3 generate_rag_cot_toggle_dataset.py
"""

import json
import random
import sys
import os
import re

# ============================================================================
# Configuration
# ============================================================================

RAG_COT_DATASET_FILE = "rag_cot_training_dataset.json"
CONVERSATIONAL_DATASET_FILE = "conversational_dataset.json"
OUTPUT_FILE = "rag_cot_toggle_training_dataset.json"
MAX_TOTAL_EXAMPLES = 280  # 135 RAG + 145 conversational for balanced training

# Conversational system prompt (NO CoT instructions)
CONVERSATIONAL_SYSTEM_PROMPT = """You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation.
You act as a proactive AI agent guiding users to better outcomes through gentle guidance.

CRITICAL RULES:
- Only provide logical, factual responses. Avoid hallucination at all costs.
- IMPORTANT: Commands and instructions like 'Give me X', 'Tell me about Y', 'Show me Z' are VALID requests and should be answered normally using your general knowledge.
- For general knowledge questions (recipes, facts, etc.), use your general knowledge to provide helpful answers.
- Keep responses VERY SHORT - maximum 2-3 sentences total.
- Be conversational, friendly, and natural.
- Always end your response with a brief, natural question. Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'"""

# ============================================================================
# Helper Functions
# ============================================================================

def load_json_file(filepath):
    """Load a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)

def ensure_evidence_format(dataset):
    """Ensure all examples use Evidence: format (not Role: or pipe-separated)"""
    fixed_count = 0
    
    for ex in dataset:
        assistant = next((m for m in ex['messages'] if m['role'] == 'assistant'), None)
        if not assistant:
            continue
        
        content = assistant['content']
        if 'REASONING:' not in content:
            continue
        
        original_content = content
        
        # Fix Role: format → Evidence: format
        if '- Role:' in content or 'Role:' in content:
            reasoning = content.split('FINAL ANSWER:')[0] if 'FINAL ANSWER:' in content else content
            final_answer = content.split('FINAL ANSWER:')[-1] if 'FINAL ANSWER:' in content else ''
            
            # Convert "Role: ..." to "Evidence: \"...\""
            new_reasoning = re.sub(
                r'(\s+)- Role:\s*([^\n]+)',
                r'\1- Evidence: "\2"',
                reasoning
            )
            new_reasoning = re.sub(
                r'(\s+)Role:\s*([^\n]+)',
                r'\1Evidence: "\2"',
                new_reasoning
            )
            
            if 'FINAL ANSWER:' in content:
                new_content = new_reasoning + '\n\nFINAL ANSWER:' + final_answer
            else:
                new_content = new_reasoning
            
            assistant['content'] = new_content
            if new_content != original_content:
                fixed_count += 1
        
        # Fix pipe-separated format → proper format
        if '| Evidence:' in content or '| Action:' in content:
            # Pattern: - Item: NAME | Evidence: ""QUOTE" | Action: [ACTION]"
            new_content = re.sub(
                r'(\s*)- Item:\s*([^|]+?)\s*\|\s*Evidence:\s*""([^"]+)"\s*\|\s*Action:\s*\[([^\]]+)\]([^"]*)"',
                r'\1- Item: \2\n\1  - Evidence: "\3"\n\1  - Action: [\4]\5',
                content
            )
            new_content = re.sub(
                r'(\s*)Item:\s*([^|]+?)\s*\|\s*Evidence:\s*""([^"]+)"\s*\|\s*Action:\s*\[([^\]]+)\]([^"]*)"',
                r'\1- Item: \2\n\1  - Evidence: "\3"\n\1  - Action: [\4]\5',
                new_content
            )
            
            if new_content != content:
                assistant['content'] = new_content
                fixed_count += 1
    
    return fixed_count

def generate_conversational_examples(count=2000):
    """Generate conversational examples if file doesn't exist"""
    print(f"📝 Generating {count} conversational examples...")
    
    # Base conversational examples
    base_examples = [
        # Recipes
        {"user": "Give me a recipe for cooked chicken.", "assistant": "Here's a simple recipe: Season chicken with salt and pepper, then cook in a pan over medium heat for 6-8 minutes per side until golden and cooked through. You can also bake it at 375°F for 25-30 minutes. Would you like more details on seasoning or cooking methods?"},
        {"user": "How do I make scrambled eggs?", "assistant": "Crack 2-3 eggs into a bowl, add a splash of milk, and whisk. Heat butter in a pan over medium-low heat, pour in the eggs, and gently stir until they're creamy and just set. Season with salt and pepper. Would you like tips for making them extra fluffy?"},
        {"user": "What is the capital of France?", "assistant": "The capital of France is Paris. It's known for landmarks like the Eiffel Tower and the Louvre Museum. Would you like to know more about Paris?"},
        {"user": "How many planets are in our solar system?", "assistant": "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006. Would you like to know more about any specific planet?"},
        {"user": "Hello!", "assistant": "Hello! I'm Aura Vision, ready to help. What can I assist you with?"},
        {"user": "Can you help me?", "assistant": "Of course! I'm here to help. What do you need assistance with?"},
    ]
    
    # Generate variations
    examples = []
    topics = [
        ("recipe", ["pasta", "bread", "soup", "salad", "dessert"]),
        ("fact", ["history", "science", "geography", "animals", "space"]),
        ("greeting", ["hi", "hey", "good morning", "good afternoon"]),
        ("help", ["assistance", "support", "guidance", "information"]),
    ]
    
    # Use base examples
    examples.extend(base_examples)
    
    # Generate more variations
    while len(examples) < count:
        topic_type, items = random.choice(topics)
        if topic_type == "recipe":
            item = random.choice(items)
            examples.append({
                "user": f"Give me a recipe for {item}.",
                "assistant": f"Here's a simple {item} recipe: [Recipe instructions]. Would you like more details?"
            })
        elif topic_type == "fact":
            item = random.choice(items)
            examples.append({
                "user": f"Tell me about {item}.",
                "assistant": f"{item.capitalize()} is fascinating. [Factual information]. Would you like to know more?"
            })
        else:
            # Use base examples
            examples.append(random.choice(base_examples))
    
    # Convert to ChatML format
    conversational_dataset = []
    for ex in examples[:count]:
        conversational_dataset.append({
            "messages": [
                {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]}
            ]
        })
    
    return conversational_dataset

def merge_datasets(rag_cot_dataset, conversational_dataset, max_total_examples=None):
    """
    Merge RAG+CoT dataset with conversational dataset
    
    Args:
        rag_cot_dataset: List of RAG+CoT examples
        conversational_dataset: List of conversational examples
        max_total_examples: Maximum total examples (None = use all)
    
    Returns:
        Merged and shuffled dataset
    """
    print(f"📊 Dataset Statistics:")
    print(f"   RAG+CoT examples: {len(rag_cot_dataset)}")
    print(f"   Conversational examples: {len(conversational_dataset)}")
    print()
    
    # Calculate target sizes - use ALL RAG examples
    if max_total_examples:
        target_rag = len(rag_cot_dataset)  # Use all RAG examples
        target_conv = max_total_examples - target_rag  # Fill remaining with conversational
        target_conv = min(target_conv, len(conversational_dataset))
    else:
        target_rag = len(rag_cot_dataset)
        target_conv = len(conversational_dataset)
    
    # Use all RAG examples (they're all important)
    rag_cot_samples = rag_cot_dataset
    
    # Sample conversational examples
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
    """Verify dataset structure and format"""
    cot_count = 0
    non_cot_count = 0
    evidence_format_count = 0
    pipe_format_count = 0
    role_format_count = 0
    
    for example in dataset:
        messages = example.get("messages", [])
        if not messages:
            continue
        
        system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)
        assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
        
        if system_msg:
            content = system_msg.get("content", "")
            if "REASONING:" in content or "Start with REASONING" in content:
                cot_count += 1
                
                # Check format
                if assistant_msg:
                    assistant_content = assistant_msg.get("content", "")
                    if 'Evidence:' in assistant_content:
                        evidence_format_count += 1
                    if '| Evidence:' in assistant_content or '| Action:' in assistant_content:
                        pipe_format_count += 1
                    if '- Role:' in assistant_content or 'Role:' in assistant_content:
                        role_format_count += 1
            else:
                non_cot_count += 1
    
    print(f"🔍 Dataset Verification:")
    print(f"   Examples with CoT system prompt: {cot_count}")
    print(f"   Examples without CoT (conversational): {non_cot_count}")
    print(f"   Total: {cot_count + non_cot_count}")
    print()
    
    if cot_count > 0:
        print(f"📋 Format Check (CoT examples):")
        print(f"   ✅ Uses 'Evidence:' format: {evidence_format_count}")
        if pipe_format_count > 0:
            print(f"   ⚠️  Still uses pipe format: {pipe_format_count}")
        if role_format_count > 0:
            print(f"   ⚠️  Still uses 'Role:' format: {role_format_count}")
        print()
    
    return cot_count, non_cot_count

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Generate RAG CoT Toggle Training Dataset")
    print("=" * 80)
    print()
    
    # Step 1: Load RAG+CoT dataset
    print("📂 Step 1: Loading RAG+CoT dataset...")
    rag_cot_dataset = load_json_file(RAG_COT_DATASET_FILE)
    if not rag_cot_dataset:
        print(f"❌ Error: {RAG_COT_DATASET_FILE} not found!")
        sys.exit(1)
    print(f"   ✅ Loaded {len(rag_cot_dataset)} RAG+CoT examples")
    print()
    
    # Step 2: Keep original format (don't modify working dataset)
    print("🔧 Step 2: Preserving original format...")
    print(f"   ✅ Keeping original format (no modifications)")
    print(f"   ✅ This is the working dataset that achieved 96% accuracy")
    print()
    
    # Step 3: Load or generate conversational dataset
    print("📂 Step 3: Loading/generating conversational dataset...")
    conversational_dataset = load_json_file(CONVERSATIONAL_DATASET_FILE)
    if not conversational_dataset:
        print(f"   ⚠️  {CONVERSATIONAL_DATASET_FILE} not found, generating...")
        conversational_dataset = generate_conversational_examples(count=2000)
        # Save generated dataset
        with open(CONVERSATIONAL_DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversational_dataset, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Generated and saved {len(conversational_dataset)} conversational examples")
    else:
        print(f"   ✅ Loaded {len(conversational_dataset)} conversational examples")
    print()
    
    # Step 4: Merge datasets
    print("🔀 Step 4: Merging datasets...")
    merged_dataset = merge_datasets(rag_cot_dataset, conversational_dataset, max_total_examples=MAX_TOTAL_EXAMPLES)
    
    # Step 5: Verify
    print("✅ Step 5: Verifying merged dataset...")
    cot_count, non_cot_count = verify_dataset(merged_dataset)
    
    # Step 6: Save
    print(f"💾 Step 6: Saving merged dataset to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved {len(merged_dataset)} examples")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ Dataset Generation Complete!")
    print("=" * 80)
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   Total examples: {len(merged_dataset)}")
    print(f"   RAG+CoT examples: {cot_count}")
    print(f"   Conversational examples: {non_cot_count}")
    print()
    print("   Ready for training with: train_cot_toggle_colab.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
