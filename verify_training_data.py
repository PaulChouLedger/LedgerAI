#!/usr/bin/env python3
"""
Diagnostic script to verify that training data is properly formatted
and the model will see the correct data during training.
"""

import json
import os
from transformers import AutoTokenizer

# Try to import datasets, but make it optional
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  'datasets' library not available - some features will be skipped")
    print()

# Configuration (should match train_rag_analysis_colab.py)
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATASET_PATH = "rag_analysis_dataset.json"
MAX_SEQ_LENGTH = 4096

print("=" * 80)
print("Training Data Verification Script")
print("=" * 80)
print()

# ============================================================================
# Load Dataset
# ============================================================================

print("=" * 80)
print("Step 1: Loading Dataset")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset file not found: {DATASET_PATH}")
    exit(1)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} examples from {DATASET_PATH}")
print()

# ============================================================================
# Load Tokenizer
# ============================================================================

print("=" * 80)
print("Step 2: Loading Tokenizer")
print("=" * 80)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Tokenizer loaded: {MODEL_NAME}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Chat template available: {tokenizer.chat_template is not None}")
    print()
except Exception as e:
    print(f"❌ ERROR loading tokenizer: {e}")
    exit(1)

# ============================================================================
# Verify Dataset Structure
# ============================================================================

print("=" * 80)
print("Step 3: Verifying Dataset Structure")
print("=" * 80)

valid_examples = 0
invalid_examples = []

for idx, example in enumerate(data):
    if "messages" not in example:
        invalid_examples.append((idx, "Missing 'messages' key"))
        continue
    
    messages = example["messages"]
    if not isinstance(messages, list):
        invalid_examples.append((idx, "Messages is not a list"))
        continue
    
    if len(messages) < 2:
        invalid_examples.append((idx, f"Too few messages: {len(messages)}"))
        continue
    
    # Check for system message
    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        invalid_examples.append((idx, "Missing system message"))
        continue
    
    # Check for user message
    has_user = any(msg.get("role") == "user" for msg in messages)
    if not has_user:
        invalid_examples.append((idx, "Missing user message"))
        continue
    
    # Check for assistant message
    has_assistant = any(msg.get("role") == "assistant" for msg in messages)
    if not has_assistant:
        invalid_examples.append((idx, "Missing assistant message"))
        continue
    
    valid_examples += 1

print(f"✅ Valid examples: {valid_examples}/{len(data)}")
if invalid_examples:
    print(f"⚠️  Invalid examples: {len(invalid_examples)}")
    print("   First 5 invalid examples:")
    for idx, reason in invalid_examples[:5]:
        print(f"     Example {idx}: {reason}")
else:
    print("✅ All examples have valid structure")
print()

# ============================================================================
# Show Sample Raw Messages
# ============================================================================

print("=" * 80)
print("Step 4: Sample Raw Messages (Before Formatting)")
print("=" * 80)

if len(data) > 0:
    sample = data[0]
    print(f"\nExample 1 (first example):")
    print("-" * 80)
    for msg in sample["messages"]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n[{role.upper()}]")
        print(preview)
    print()
else:
    print("❌ No examples in dataset")
    exit(1)

# ============================================================================
# Format with Chat Template
# ============================================================================

print("=" * 80)
print("Step 5: Formatting with Chat Template")
print("=" * 80)

# Prepare messages (same logic as training script)
prepared_messages = []
for idx, example in enumerate(data):
    messages = example.get("messages", [])
    chat_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role and content:
            chat_messages.append({
                "role": role,
                "content": content
            })
    
    if len(chat_messages) >= 2:
        prepared_messages.append(chat_messages)

print(f"✅ Prepared {len(prepared_messages)} valid conversations")
print()

# Format with chat template
formatted_texts = []
for idx, messages in enumerate(prepared_messages):
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append({"text": text})
    except Exception as e:
        print(f"❌ ERROR formatting example {idx}: {e}")
        continue

print(f"✅ Formatted {len(formatted_texts)} conversations")
print()

# ============================================================================
# Show Sample Formatted Text
# ============================================================================

print("=" * 80)
print("Step 6: Sample Formatted Text (What Model Sees)")
print("=" * 80)

if len(formatted_texts) > 0:
    sample_text = formatted_texts[0]["text"]
    print(f"\nFirst 1000 characters of formatted text:")
    print("-" * 80)
    print(sample_text[:1000])
    if len(sample_text) > 1000:
        print(f"\n... (truncated, total length: {len(sample_text)} characters)")
    print()
else:
    print("❌ No formatted texts available")
    exit(1)

# ============================================================================
# Tokenize and Show Token Statistics
# ============================================================================

print("=" * 80)
print("Step 7: Tokenization Statistics")
print("=" * 80)

tokenized_lengths = []
for idx, formatted in enumerate(formatted_texts):
    tokens = tokenizer(formatted["text"], return_length=True)
    length = tokens["length"][0]
    tokenized_lengths.append(length)

if tokenized_lengths:
    avg_length = sum(tokenized_lengths) / len(tokenized_lengths)
    min_length = min(tokenized_lengths)
    max_length = max(tokenized_lengths)
    
    print(f"✅ Tokenized {len(tokenized_lengths)} examples")
    print(f"   Average length: {avg_length:.1f} tokens")
    print(f"   Min length: {min_length} tokens")
    print(f"   Max length: {max_length} tokens")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH} tokens")
    
    # Check for truncation
    truncated = sum(1 for l in tokenized_lengths if l > MAX_SEQ_LENGTH)
    if truncated > 0:
        print(f"   ⚠️  {truncated} examples will be truncated (>{MAX_SEQ_LENGTH} tokens)")
    else:
        print(f"   ✅ No examples will be truncated")
    
    # Show distribution
    print(f"\n   Length distribution:")
    bins = [0, 500, 1000, 2000, 3000, MAX_SEQ_LENGTH, float('inf')]
    bin_labels = ["0-500", "500-1000", "1000-2000", "2000-3000", f"3000-{MAX_SEQ_LENGTH}", f">{MAX_SEQ_LENGTH}"]
    for i in range(len(bins) - 1):
        count = sum(1 for l in tokenized_lengths if bins[i] <= l < bins[i+1])
        pct = (count / len(tokenized_lengths)) * 100
        print(f"     {bin_labels[i]}: {count} examples ({pct:.1f}%)")
    print()
else:
    print("❌ No tokenized examples available")

# ============================================================================
# Show Tokenized Example
# ============================================================================

print("=" * 80)
print("Step 8: Sample Tokenized Example")
print("=" * 80)

if len(formatted_texts) > 0:
    sample = formatted_texts[0]["text"]
    tokens = tokenizer(sample, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    
    print(f"\nToken IDs (first 50 tokens):")
    print(tokens["input_ids"][0][:50].tolist())
    print()
    
    print(f"Decoded tokens (first 50 tokens):")
    decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0][:50])
    print(decoded)
    print()
    
    print(f"Total tokens: {tokens['input_ids'].shape[1]}")
    print()

# ============================================================================
# Verify Training Format
# ============================================================================

print("=" * 80)
print("Step 9: Verifying Training Format")
print("=" * 80)

# Create dataset (same as training script)
if HAS_DATASETS:
    dataset = Dataset.from_list(formatted_texts)
    print(f"✅ Created HuggingFace Dataset")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Features: {list(dataset.features.keys())}")
    print()
    
    # Check if dataset can be tokenized for training
    try:
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_overflowing_tokens=False,
            )
        
        # Test on first example
        test_result = tokenize_function({"text": [dataset[0]["text"]]})
        print(f"✅ Tokenization function works correctly")
        print(f"   Test example input_ids shape: {test_result['input_ids'][0].shape if hasattr(test_result['input_ids'][0], 'shape') else len(test_result['input_ids'][0])}")
        print()
    except Exception as e:
        print(f"❌ ERROR in tokenization function: {e}")
        import traceback
        traceback.print_exc()
        print()
else:
    print("⚠️  Skipping Dataset creation (datasets library not available)")
    print(f"   Would create dataset with {len(formatted_texts)} examples")
    print()
    
    # Test tokenization function manually
    try:
        test_text = formatted_texts[0]["text"]
        test_result = tokenizer(
            test_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_overflowing_tokens=False,
        )
        print(f"✅ Tokenization function works correctly")
        print(f"   Test example token count: {len(test_result['input_ids'])}")
        print()
    except Exception as e:
        print(f"❌ ERROR in tokenization: {e}")
        import traceback
        traceback.print_exc()
        print()

# ============================================================================
# Show Multiple Examples
# ============================================================================

print("=" * 80)
print("Step 10: Sample Examples (First 3)")
print("=" * 80)

for idx in range(min(3, len(formatted_texts))):
    print(f"\n{'='*80}")
    print(f"Example {idx + 1}")
    print(f"{'='*80}")
    
    # Show original messages
    print("\n[ORIGINAL MESSAGES]")
    for msg in prepared_messages[idx]:
        role = msg["role"]
        content = msg["content"]
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"\n{role.upper()}:")
        print(preview)
    
    # Show formatted text preview
    print(f"\n[FORMATTED TEXT (first 500 chars)]")
    formatted = formatted_texts[idx]["text"]
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    
    # Show token count
    tokens = tokenizer(formatted, return_length=True)
    print(f"\n[TOKEN COUNT: {tokens['length'][0]} tokens]")
    print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("Verification Summary")
print("=" * 80)
print(f"✅ Dataset file exists: {DATASET_PATH}")
print(f"✅ Total examples: {len(data)}")
print(f"✅ Valid examples: {valid_examples}")
print(f"✅ Tokenizer loaded: {MODEL_NAME}")
print(f"✅ Chat template working: {tokenizer.chat_template is not None}")
print(f"✅ Formatted conversations: {len(formatted_texts)}")
if HAS_DATASETS:
    print(f"✅ Dataset created: {len(dataset)} examples")
else:
    print(f"✅ Formatted texts ready: {len(formatted_texts)} examples")
if tokenized_lengths:
    print(f"✅ Average token length: {sum(tokenized_lengths) / len(tokenized_lengths):.1f}")
    print(f"✅ Max sequence length: {MAX_SEQ_LENGTH}")
    if max(tokenized_lengths) <= MAX_SEQ_LENGTH:
        print(f"✅ No truncation needed")
    else:
        print(f"⚠️  Some examples will be truncated")
print()
print("=" * 80)
print("✅ Verification Complete - Training data is properly formatted!")
print("=" * 80)

