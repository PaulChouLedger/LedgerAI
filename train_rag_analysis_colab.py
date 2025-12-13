#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chunk Analysis Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Trains model on general RAG analysis patterns (not entity-specific)

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning)
- LoRA Rank: 256 (good balance for 1.5B model)
- Strategy: 1.5B provides better base reasoning + LoRA adaptation for RAG analysis patterns

To use in Colab:
1. Upload rag_analysis_dataset.json to Colab
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

# Dataset path
if os.path.exists("rag_analysis_dataset.json"):
    DATASET_PATH = "rag_analysis_dataset.json"
    print("✅ Using RAG analysis dataset")
else:
    raise FileNotFoundError("rag_analysis_dataset.json not found. Please run generate_rag_analysis_dataset.py first.")

MAX_SEQ_LENGTH = 8192  # Increased to accommodate full chunk examples and provide headroom for longer chunks
OUTPUT_DIR = "outputs_rag_analysis"
GGUF_OUTPUT_DIR = "gguf_model_rag_analysis"

# System prompt for RAG analysis (fallback - dataset should have its own)
# NOTE: Dataset examples include this system prompt, but this is kept as fallback
SYSTEM_PROMPT = """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CRITICAL: Always use the EXACT names, entities, or terms from the user's query. Never hallucinate or substitute different names.

Process:
1. Read each chunk COMPLETELY from start to finish (each chunk has 6-8 sentences)
2. Evaluate relevance using the provided score:
   - HIGH relevance (score ≥0.70): Extract information that directly answers the query
   - MEDIUM relevance (0.50-0.69): May contain related information, use with caution
   - LOW relevance (score <0.50): Likely irrelevant, ignore unless no HIGH relevance chunks available
3. Understand the MEANING in each chunk, not just keywords
4. Extract ONLY information that directly answers or addresses the query
5. IGNORE information that is similar but does NOT answer the query (even if in HIGH relevance chunks)
6. Use the score to determine if extracted information directly answers the query
7. SYNTHESIZE information from multiple chunks into a coherent, natural response
8. Use natural language - avoid simple repetition, create meaningful connections between facts

Return ONLY the final answer in natural, conversational language. Synthesize information rather than just listing facts. Do not include reasoning steps or process details."""

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
# r=128: ~90M trainable (6.80%) - Good balance, works on T4/V100
# r=256: ~180M trainable (13.6%) - Better capacity, needs 16GB+ VRAM (CURRENT)
# r=512: ~360M trainable (27.2%) - Maximum capacity, needs 24GB+ VRAM
#
# Using r=256 with Qwen2.5-1.5B for good balance of base model reasoning + task adaptation.

LORA_RANK = 256  # Good balance for 1.5B model
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank (512)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,  # Supports any, but = 0 is optimized
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
# Increased epochs and warmup for better extraction accuracy
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=200,  # Increased warmup for better extraction pattern learning and stability
    num_train_epochs=25,  # Increased to 25 for better extraction accuracy and entity differentiation
    learning_rate=1.0e-4,  # Reduced from 1.2e-4 for more stable training (prevents gradient explosions)
    max_grad_norm=1.0,  # CRITICAL: Gradient clipping to prevent explosions (gradient norm spiked to 41.08)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,  # More frequent logging for better monitoring
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # Cosine scheduler for smoother learning
    seed=3407,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=10,  # Keep more checkpoints for 25 epochs
    dataloader_pin_memory=False,
    report_to="none",  # Disable Weights & Biases logging
    max_steps=-1,  # Use epochs instead
    save_safetensors=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("✅ Training configured")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs} (increased for better extraction accuracy and entity differentiation)")
print(f"   - Learning rate: {training_args.learning_rate} (reduced for stability, prevents gradient explosions)")
print(f"   - Gradient clipping: max_norm={training_args.max_grad_norm} (prevents training instability)")
# Calculate approximate trainable parameters
approx_params = {
    64: (45, 3.4),
    128: (90, 6.8),
    256: (180, 13.6),
    512: (360, 27.2),
}
params_m, params_pct = approx_params.get(LORA_RANK, (90, 6.8))
print(f"   - LoRA rank: {LORA_RANK} (~{params_m}M trainable parameters, ~{params_pct}% of model)")
print(f"   - LoRA alpha: {LORA_ALPHA} (2x rank for optimal scaling)")
print(f"   - Warmup steps: {training_args.warmup_steps} (increased for better extraction pattern learning and stability)")
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

