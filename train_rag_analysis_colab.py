#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chunk Analysis Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Trains model on general RAG analysis patterns (not entity-specific)

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning)
- LoRA Rank: 128 (reduced from 256 to prevent overfitting)
- Strategy: 1.5B provides better base reasoning + LoRA adaptation for RAG analysis patterns
- Epochs: 5 (reduced from 25 to prevent overfitting)
- Regularization: weight_decay=0.15 (evaluation disabled - crashes system)

To use in Colab:
1. Upload rag_analysis_dataset_v2.json (or rag_analysis_dataset.json) to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

The model will learn general RAG analysis skills:
- Read entire RAG chunks completely (6-8 sentences each)
- Analyze and understand meaning in chunks (not just keywords)
- Extract relevant information to query (any type: entities, facts, concepts, relationships)
- Ignore irrelevant information (even in HIGH relevance chunks)
- Use scoring to determine if information directly answers query
- Handle various query types (factual, analytical, comparison, relationship, list)
"""

import json
import os
import shutil
import torch
from datasets import Dataset

# Disable caching to ensure fresh loads
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Workaround for Unsloth 2025.12.3 bug: is_unsupported_gemma not defined
# Patch it before importing SFTTrainer
try:
    import unsloth.trainer
    # Set the missing variable
    setattr(unsloth.trainer, 'is_unsupported_gemma', False)
except Exception as e:
    print(f"⚠️  Warning: Could not patch Unsloth trainer ({e}), continuing anyway...")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# Install Dependencies (Colab)
# ============================================================================
# Uncomment if running in Colab:
# !pip install unsloth trl peft accelerate bitsandbytes datasets

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Qwen 2.5 1.5B - better instruction following and reasoning

# Dataset path (prefer v2, fallback to v1)
if os.path.exists("rag_analysis_dataset_v2.json"):
    DATASET_PATH = "rag_analysis_dataset_v2.json"
    print("✅ Using RAG analysis dataset v2")
elif os.path.exists("rag_analysis_dataset.json"):
    DATASET_PATH = "rag_analysis_dataset.json"
    print("✅ Using RAG analysis dataset v1")
else:
    raise FileNotFoundError("Neither rag_analysis_dataset_v2.json nor rag_analysis_dataset.json found. Please run generate_rag_dataset_v2.py first.")

MAX_SEQ_LENGTH = 8192  # Increased to accommodate full chunk examples and provide headroom for longer chunks
OUTPUT_DIR = "outputs_rag_analysis"
GGUF_OUTPUT_DIR = "gguf_model_rag_analysis"

# System prompt for RAG analysis (fallback - dataset should have its own)
# NOTE: Dataset examples include this system prompt, but this is kept as fallback
# Updated to match new 7-step core principles format
SYSTEM_PROMPT = """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements (role, entity, attribute, relationship, etc.)
- Understand the scope and context of what needs to be extracted

STEP 2: READ EACH CHUNK COMPLETELY
- Read the entire chunk from start to finish
- Do not stop at keywords - read for full context and meaning
- Understand the complete context before making extraction decisions

STEP 3: ANALYZE CHUNK MEANING
- Understand the semantic meaning, not just surface-level keywords
- Identify entities, relationships, attributes, and concepts mentioned
- Recognize how information relates to the query

STEP 4: EVALUATE RELEVANCE
- Determine if information directly answers or addresses the query
- Apply query-specific filtering (match role, entity, attribute, etc. as requested)
- CRITICAL: For role queries, match the EXACT role (e.g., "co-founders" ≠ "CEO" ≠ "CTO" - extract ONLY the exact role requested)
- CRITICAL: For company queries, extract information ONLY about the company that matches the query. Use the company name EXACTLY as it appears in the chunks (RAG handles fuzzy matching at retrieval - if chunk says "TechCorp", extract "TechCorp" even if query said "Tech Corp"). Do NOT extract information about other companies
- Ignore information that is similar but does NOT answer the query
- Use relevance scores to guide prioritization (HIGH ≥0.70, MEDIUM 0.50-0.69, LOW <0.50)

STEP 5: EXTRACT MATCHING INFORMATION
- Extract only information that passes the relevance evaluation
- Apply exact matching - use information exactly as it appears in chunks
- Track all matching items across all chunks

STEP 6: VERIFY COMPLETENESS
- Ensure you have read ALL chunks completely
- Verify you extracted ALL matching items (do not stop after first match)
- Confirm extraction is complete before finalizing response

STEP 7: SYNTHESIZE RESPONSE
- Combine information from all chunks into coherent answer
- Format naturally and directly address the query
- CRITICAL: If after reading ALL chunks completely you find NO information that matches the query (wrong role, wrong company, or missing entirely), you MUST respond with exactly: "I don't have that information in the provided documents"
- DO NOT infer, guess, or make up information - if it's not explicitly in the chunks, say "I don't have that information in the provided documents"

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding.

ESSENTIAL GUIDELINES:
- NEVER hallucinate - only use information that appears in the provided chunks
- NEVER make up names, entities, or information - if information doesn't exist, say "I don't have that information in the provided documents"
- CRITICAL: If you cannot find the EXACT information requested in ANY chunk, you MUST respond with "I don't have that information in the provided documents" - DO NOT guess, infer, or make up information
- Use EXACT information from chunks - never substitute or modify names, terms, or entities
- Apply query-specific filtering during Step 4 (evaluate relevance) - match what the query specifically asks for
- Extract ALL matching items - complete Step 6 (verify completeness) ensures nothing is missed
- Relevance scores guide prioritization but do not override the evaluation steps

QUERY TYPE HANDLING (applied during Step 4 - Evaluate Relevance):
- Role/entity queries: Filter by the SPECIFIC role mentioned (e.g., "co-founders" means ONLY co-founders, NOT CEOs, CTOs, or other roles). If the query asks for "co-founders", extract ONLY people explicitly labeled as co-founders, NOT other roles even if they are at the same company
- Company-specific queries: Extract information ONLY about the company that matches the query. If query asks about "TechCorp", extract information ONLY about the matching company in chunks (RAG handles fuzzy matching like "Tech Corp" → "TechCorp" at retrieval level). Do NOT extract information about other companies mentioned in the same chunk
- Comparison queries: Extract information comparing the entities mentioned
- Relationship queries: Extract connection information between entities
- Analytical queries: Extract reasoning, causation, or explanation
- Process queries: Extract step-by-step information
- List queries: Extract ALL items that match the query criteria - read ALL chunks completely before responding

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""

# ============================================================================
# GPU Check
# ============================================================================

print("=" * 80)
print("GPU Configuration Check")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected. Training will be very slow on CPU.")
print("=" * 80)
print()

# ============================================================================
# Load and Prepare Dataset
# ============================================================================

print("=" * 80)
print("Loading RAG Analysis Dataset")
print("=" * 80)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} conversations from {DATASET_PATH}")

# Analyze dataset
query_types = {}
for conv in data:
    qtype = conv.get("query_type", "unknown")
    query_types[qtype] = query_types.get(qtype, 0) + 1

print()
print("Dataset breakdown:")
for qtype, count in query_types.items():
    print(f"  - {qtype}: {count} examples")
print()

# Prepare message structures
prepared_messages = []

for idx, conversation in enumerate(data):
    messages = conversation.get("messages", [])

    if not messages:
        continue

    # Build messages list with system prompt
    chat_messages = []

    # Check if dataset already has a system prompt
    has_system = any(msg.get("role") == "system" for msg in messages)

    # Use system prompt from dataset if present, otherwise use fallback
    if not has_system:
        chat_messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

    # Add all conversation messages (preserve system prompt from dataset if present)
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()

        if role and content:
            chat_messages.append({
                "role": role,
                "content": content
            })

    if len(chat_messages) < 2:  # Need at least system + one user/assistant message
        continue

    prepared_messages.append(chat_messages)

    # Print progress every 50 items, or on the last item
    if (idx + 1) % 50 == 0 or (idx + 1) == len(data):
        print(f"  Processed {idx + 1}/{len(data)} conversations...")

print(f"✅ Prepared {len(prepared_messages)} valid conversations")
print()

# ============================================================================
# Load Model and Tokenizer
# ============================================================================

print("=" * 80)
print("Loading Model and Tokenizer")
print("=" * 80)

# Disable model caching to ensure fresh load
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto detection
    load_in_4bit=True,
)

# Qwen 2.5 uses its own chat template (automatically set by tokenizer)
print(f"✅ Model loaded: {MODEL_NAME}")
print(f"✅ Chat template: Qwen 2.5 format (auto-configured by tokenizer)")
print()

# ============================================================================
# Format Dataset with Chat Template
# ============================================================================

print("=" * 80)
print("Formatting Dataset with Chat Template")
print("=" * 80)

# Format all conversations using the tokenizer's chat template
formatted_texts = []
for idx, messages in enumerate(prepared_messages):
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    formatted_texts.append({"text": text})

    # Print progress every 50 items, or on the last item
    if (idx + 1) % 50 == 0 or (idx + 1) == len(prepared_messages):
        print(f"  Formatted {idx + 1}/{len(prepared_messages)} conversations...")

print(f"✅ Formatted {len(formatted_texts)} conversations")
print()

# Create HuggingFace Dataset with already-formatted text
dataset = Dataset.from_list(formatted_texts)

print("=" * 80)
print("Dataset Statistics")
print("=" * 80)
print(f"Total conversations: {len(dataset)}")
if len(dataset) > 0:
    sample_text = dataset[0]['text']
    print(f"Sample text length: {len(sample_text)} characters")
    
    # Show sample formatted text (first 800 chars) so user can verify formatting
    print(f"\n📋 Sample formatted text (first 800 characters):")
    print("-" * 80)
    print(sample_text[:800])
    if len(sample_text) > 800:
        print(f"... (truncated, total: {len(sample_text)} chars)")
    print("-" * 80)
    
    # Show tokenization stats
    sample_tokens = tokenizer(sample_text, return_length=True)
    print(f"\n📊 Tokenization stats:")
    print(f"   Sample token count: {sample_tokens['length'][0]} tokens")
    
    # Check a few more examples for token counts
    token_counts = []
    for i in range(min(10, len(dataset))):
        tokens = tokenizer(dataset[i]['text'], return_length=True)
        token_counts.append(tokens['length'][0])
    
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        print(f"   Average (first 10): {avg_tokens:.1f} tokens")
        print(f"   Min: {min_tokens} tokens, Max: {max_tokens} tokens")
        print(f"   Max sequence length: {MAX_SEQ_LENGTH} tokens")
        if max_tokens > MAX_SEQ_LENGTH:
            print(f"   ⚠️  Some examples will be truncated")
        else:
            print(f"   ✅ No truncation needed")
print("=" * 80)
print()

# ============================================================================
# Add LoRA Adapters
# ============================================================================

print("=" * 80)
print("Configuring LoRA Adapters")
print("=" * 80)

# LoRA Configuration
# r=16: ~11M trainable (0.85%) - Extremely minimal capacity
# r=32: ~22M trainable (1.7%) - Very minimal capacity (CURRENT)
# r=64: ~45M trainable (3.4%) - Still too much capacity (loss decreased too fast: 1.99→0.0076 in 80 steps)
# r=128: ~90M trainable (6.80%) - Too much capacity (loss decreased too fast)
# r=256: ~180M trainable (13.6%) - Higher capacity, causes overfitting
#
# Using r=32 with Qwen2.5-1.5B - loss still decreasing too fast (1.99→0.0076 in 80 steps) with r=64.

LORA_RANK = 16  # Increased from 8 - more capacity for complex patterns (role filtering, cross-company, multi-chunk extraction)
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank (32)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,  # Unsloth optimized: 0.0 is fastest. Using weight_decay=0.3 for regularization instead
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)

print("✅ LoRA adapters configured")
print()

# ============================================================================
# Setup Training
# ============================================================================

print("=" * 80)
print("Configuring Training")
print("=" * 80)

# Training arguments optimized for Unsloth and RAG analysis
# Reduced epochs and added regularization to prevent overfitting (loss was hitting 0.0)
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=500,  # Reduced from 1000 - faster warmup, more time in actual training
    num_train_epochs=5,  # Increased from 4 - with 6,250 examples, more epochs help model learn all patterns
    learning_rate=3e-6,  # Reduced from 4e-6 - slower, more stable learning for better generalization
    max_grad_norm=1.0,  # CRITICAL: Gradient clipping to prevent explosions (gradient norm spiked to 41.08)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,  # More frequent logging for better monitoring
    optim="adamw_8bit",
    weight_decay=0.3,  # Reduced from 0.35 - less aggressive regularization, allows model to learn more patterns while still preventing overfitting
    lr_scheduler_type="cosine",  # Cosine scheduler for smoother learning
    seed=3407,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    eval_strategy="no",  # Disabled - evaluation crashes system
    save_total_limit=3,  # Keep 3 best checkpoints
    dataloader_pin_memory=False,
    report_to="none",  # Disable Weights & Biases logging
    max_steps=-1,  # Use epochs instead
    save_safetensors=True,
)

# Use full dataset for training (no validation split - evaluation disabled)
train_dataset = dataset

print(f"   - Train examples: {len(train_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("✅ Training configured")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs} (increased from 4 - with 6,250 examples, more epochs help model learn all patterns)")
print(f"   - Learning rate: {training_args.learning_rate} (reduced from 4e-6 - slower, more stable learning for better generalization)")
print(f"   - Gradient clipping: max_norm={training_args.max_grad_norm} (prevents training instability)")
# Calculate approximate trainable parameters
approx_params = {
    8: (5.5, 0.35),
    16: (11, 0.85),
    32: (22, 1.7),
    64: (45, 3.4),
    128: (90, 6.8),
    256: (180, 13.6),
    512: (360, 27.2),
}
params_m, params_pct = approx_params.get(LORA_RANK, (5.5, 0.35))
print(f"   - LoRA rank: {LORA_RANK} (~{params_m}M trainable parameters, ~{params_pct}% of model) (increased from 8 - more capacity for complex patterns)")
print(f"   - LoRA alpha: {LORA_ALPHA} (2x rank for optimal scaling)")
print(f"   - Warmup steps: {training_args.warmup_steps} (reduced from 1000 - faster warmup, more training time)")
print(f"   - Scheduler: {training_args.lr_scheduler_type}")
print(f"   - Logging steps: {training_args.logging_steps} (more frequent monitoring)")
print()
print("📊 Training Goals:")
print("   ✅ Read entire RAG chunks completely (6-8 sentences each)")
print("   ✅ Analyze and understand meaning in chunks (not just keywords)")
print("   ✅ Extract relevant information to query (any type: entities, facts, concepts, etc.)")
print("   ✅ Ignore irrelevant information (even in HIGH relevance chunks)")
print("   ✅ Use scoring to determine if information directly answers query")
print("   ✅ Handle various query types (factual, analytical, comparison, relationship)")
print()

# ============================================================================
# Train Model
# ============================================================================

print("=" * 80)
print("Starting Training")
print("=" * 80)
print()

trainer_stats = trainer.train()

print()
print("=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"Training stats: {trainer_stats}")
print()

# ============================================================================
# Save Model
# ============================================================================

print("=" * 80)
print("Saving Model")
print("=" * 80)

# Save in HuggingFace format
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")

# Save in GGUF format for deployment
print(f"Converting to GGUF format in {GGUF_OUTPUT_DIR}...")

# Clean up any existing GGUF directory to avoid conflicts (like medical bot does implicitly)
if os.path.exists(GGUF_OUTPUT_DIR):
    import shutil
    print(f"Cleaning up existing {GGUF_OUTPUT_DIR} directory...")
    shutil.rmtree(GGUF_OUTPUT_DIR)
    print(f"✅ Cleaned up existing directory")

try:
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method="q4_k_m"  # Q4_K_M quantization for good balance
    )
except (RuntimeError, Exception) as e:
    print("\n" + "="*80)
    print("⚠️  GGUF Conversion Failed")
    print("="*80)
    print(f"Error: {e}")
    print("\n✅ The HuggingFace model has been saved successfully and can be used directly!")
    print(f"   Location: {OUTPUT_DIR}/")
    print("\n💡 You can use the HuggingFace format with transformers/unsloth - it works perfectly!")
    print("\n🔧 If you need GGUF format, try these troubleshooting steps:")
    print("   1. Check available disk space: !df -h")
    print("   2. Check available memory: !free -h")
    print("   3. The merged model is in: gguf_model_rag_analysis/")
    print("   4. Run manual conversion script to see actual error:")
    print("      !python manual_gguf_conversion.py")
    print("   5. Or try converting from outputs_rag_analysis/ directly")
    print("\n📝 Note: HuggingFace format works with:")
    print("   - Unsloth FastLanguageModel.from_pretrained()")
    print("   - transformers AutoModelForCausalLM.from_pretrained()")
    print("   - Both work for inference!")
    print("="*80)
    # Don't raise - allow training to complete successfully
    # The HuggingFace model is what matters most

# Rename GGUF file to append "-rag-analysis" to filename (if conversion succeeded)
if os.path.exists(GGUF_OUTPUT_DIR):
    gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
    if gguf_files:
        # Find the Q4_K_M file (the quantized one we want)
        q4_file = [f for f in gguf_files if "Q4_K_M" in f]
        if q4_file:
            original_file = os.path.join(GGUF_OUTPUT_DIR, q4_file[0])
            # Extract base name and add "-rag-analysis" before .gguf extension
            base_name = os.path.splitext(q4_file[0])[0]
            new_filename = f"{base_name}-rag-analysis.gguf"
            new_file = os.path.join(GGUF_OUTPUT_DIR, new_filename)
            if not os.path.exists(new_file):  # Only rename if new file doesn't exist
                shutil.move(original_file, new_file)
                print(f"✅ GGUF model renamed to: {new_filename}")
            else:
                print(f"✅ GGUF model already exists as: {new_filename}")
        else:
            # Use first GGUF file if no Q4_K_M found
            original_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
            base_name = os.path.splitext(gguf_files[0])[0]
            new_filename = f"{base_name}-rag-analysis.gguf"
            new_file = os.path.join(GGUF_OUTPUT_DIR, new_filename)
            if not os.path.exists(new_file):
                shutil.move(original_file, new_file)
                print(f"✅ GGUF model renamed to: {new_filename}")
            else:
                print(f"✅ GGUF model already exists as: {new_filename}")
    else:
        print(f"✅ GGUF model saved to {GGUF_OUTPUT_DIR} (no .gguf files found yet)")

print()
print("=" * 80)
print("🎉 Fine-tuning Complete!")
print("=" * 80)
print(f"Your RAG analysis model is ready:")
print(f"  - HuggingFace format: {OUTPUT_DIR}/")
print(f"  - GGUF format: {GGUF_OUTPUT_DIR}/")
print()
print("The model has been trained to:")
print("  ✅ Read entire RAG chunks completely (6-8 sentences each)")
print("  ✅ Analyze and understand meaning in chunks (not just keywords)")
print("  ✅ Extract relevant information to query (any type: entities, facts, concepts, relationships)")
print("  ✅ Ignore irrelevant information (even in HIGH relevance chunks)")
print("  ✅ Use scoring to determine if information directly answers query")
print("  ✅ Handle various query types (factual, analytical, comparison, relationship, list)")
print("  ✅ Return only the final answer (no internal reasoning steps)")
print()

# For Colab: Download the GGUF file
try:
    from google.colab import files
    
    if os.path.exists(GGUF_OUTPUT_DIR):
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
        if gguf_files:
            print(f"\n📦 Found {len(gguf_files)} GGUF file(s) in {GGUF_OUTPUT_DIR}/")
            for f in gguf_files:
                file_path = os.path.join(GGUF_OUTPUT_DIR, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                print(f"   - {f} ({file_size:.1f} MB)")
            
            # Prefer the -rag-analysis file, then Q4_K_M, then any GGUF file
            rag_file = [f for f in gguf_files if "-rag-analysis" in f]
            q4_file = [f for f in gguf_files if "Q4_K_M" in f]
            
            if rag_file:
                gguf_file = os.path.join(GGUF_OUTPUT_DIR, rag_file[0])
                print(f"\n📥 Downloading: {rag_file[0]}")
            elif q4_file:
                gguf_file = os.path.join(GGUF_OUTPUT_DIR, q4_file[0])
                print(f"\n📥 Downloading: {q4_file[0]} (Q4_K_M quantized)")
            else:
                gguf_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
                print(f"\n📥 Downloading: {gguf_files[0]}")
            
            files.download(gguf_file)
            print(f"✅ Download initiated")
        else:
            print(f"\n⚠️  No GGUF files found in {GGUF_OUTPUT_DIR}/")
            print(f"   Directory exists: {os.path.exists(GGUF_OUTPUT_DIR)}")
            if os.path.exists(GGUF_OUTPUT_DIR):
                all_files = os.listdir(GGUF_OUTPUT_DIR)
                print(f"   Files in directory: {all_files[:10]}...")  # Show first 10 files
    else:
        print(f"\n⚠️  GGUF output directory does not exist: {GGUF_OUTPUT_DIR}")
except ImportError:
    print("\n⚠️  Not running in Colab - skipping file download")
except Exception as e:
    print(f"\n⚠️  Error downloading GGUF file: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)

