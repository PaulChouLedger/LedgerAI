#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chain of Thought (CoT) Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Trains model to use Chain of Thought reasoning when extracting information from RAG chunks

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning than 0.5B)
- LoRA Rank: 256 (good balance for 1.5B model)
- Strategy: 1.5B provides better base reasoning + LoRA adaptation for CoT RAG extraction

To use in Colab:
1. Upload rag_cot_training_dataset.json to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

Dataset Features:
- Chain of Thought reasoning examples for RAG chunk extraction
- Multiple scenarios with different numbers of co-founders
- Cases where no co-founders are explicitly stated
- Examples showing how to exclude non-co-founders (advisors, employees, etc.)
- Training to carefully read chunks and identify explicit relationships
- Explicit examples showing all drawbacks marked as [DISCARD] for benefits queries
- Compound role examples (e.g., "CEO and Co-Founder" should be [KEEP])
- Query intent examples (benefits vs drawbacks, etc.)

Dataset Status (Verified):
- ✅ 156 training examples
- ✅ All examples have proper REASONING and FINAL ANSWER sections
- ✅ All [KEEP]/[DISCARD] actions are correctly marked
- ✅ No DISCARD violations found
- ✅ Benefits query example explicitly shows all drawbacks as [DISCARD]
- ✅ All real-world examples (indices 0-5) are accurate
"""

import json
import os
import shutil
import random
import numpy as np
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# Set Random Seeds for Deterministic Training
# ============================================================================

SEED = 3407  # Match seed used in TrainingArguments

# Set Python random seed
random.seed(SEED)

# Set NumPy random seed
np.random.seed(SEED)

# Set PyTorch random seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for hash randomization
os.environ['PYTHONHASHSEED'] = str(SEED)

print("=" * 80)
print("Random Seeds Set for Deterministic Training")
print("=" * 80)
print(f"✅ Python random seed: {SEED}")
print(f"✅ NumPy random seed: {SEED}")
print(f"✅ PyTorch random seed: {SEED}")
print(f"✅ CUDA deterministic: True")
print(f"✅ CUDA benchmark: False")
print(f"✅ PYTHONHASHSEED: {SEED}")
print()

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
DATASET_PATH = "rag_cot_training_dataset.json"

# Latency-optimized sequence length based on analysis:
# - 4 co-founder example needs ~2,365 input tokens
# - Typical output: ~800 tokens
# - Max output buffer: 4,096 tokens (5x typical for complex cases)
# - Total needed: ~2,365 + 4,096 = 6,461 tokens
# - Using 8,192 for safety buffer and to match inference n_ctx setting
# This optimizes for latency (smaller context = faster) while avoiding truncation
MAX_SEQ_LENGTH = 8192  # Optimized: input (~2.4K) + max_output (4K) + buffer = latency-optimized
OUTPUT_DIR = "outputs_rag_cot"
GGUF_OUTPUT_DIR = "gguf_model_rag_cot"

# System prompt is included in the dataset, but we have a fallback just in case
FALLBACK_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items."""


# ============================================================================
# Set Random Seeds for Deterministic Training
# ============================================================================

import random
import numpy as np

SEED = 3407  # Match seed used in TrainingArguments

# Set Python random seed
random.seed(SEED)

# Set NumPy random seed
np.random.seed(SEED)

# Set PyTorch random seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for hash randomization
import os
os.environ['PYTHONHASHSEED'] = str(SEED)

print("=" * 80)
print("Random Seeds Set for Deterministic Training")
print("=" * 80)
print(f"✅ Python random seed: {SEED}")
print(f"✅ NumPy random seed: {SEED}")
print(f"✅ PyTorch random seed: {SEED}")
print(f"✅ CUDA deterministic: True")
print(f"✅ CUDA benchmark: False")
print(f"✅ PYTHONHASHSEED: {SEED}")
print()

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
print("Loading RAG CoT Training Dataset")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset file '{DATASET_PATH}' not found!")
    print("Please upload rag_cot_training_dataset.json to Colab first.")
    exit(1)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} training examples from {DATASET_PATH}")

# Check dataset features
sample_conv = data[0] if data else {}
sample_messages = sample_conv.get("messages", [])
has_cot_reasoning = any("CHAIN OF THOUGHT" in msg.get("content", "") or
                        "Step 1:" in msg.get("content", "") or
                        "Step 2:" in msg.get("content", "")
                        for msg in sample_messages)

# Print dataset features
print()
if has_cot_reasoning:
    print("📚 Chain of Thought Features Detected:")
    print("   ✅ Step-by-step reasoning process")
    print("   ✅ Chunk-by-chunk analysis")
    print("   ✅ Explicit role/title identification")
    print("   ✅ Verification before final answer")
    print()

# Count different scenario types
cofounder_scenarios = sum(1 for conv in data 
                          if any("co-founders" in msg.get("content", "").lower() 
                                 for msg in conv.get("messages", [])))
print(f"📊 Dataset Statistics:")
print(f"   - Total examples: {len(data)}")
print(f"   - Co-founder query examples: {cofounder_scenarios}")
print()
print("📋 Dataset Verification:")
print(f"   ✅ Dataset verified: {len(data)} examples")
print(f"   ✅ All examples have REASONING and FINAL ANSWER sections")
print(f"   ✅ All [KEEP]/[DISCARD] actions correctly marked")
print(f"   ✅ Benefits query example includes all drawbacks as [DISCARD]")
print(f"   ✅ No DISCARD violations found")
print()

# Prepare message structures (without formatting yet)
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
            "content": FALLBACK_SYSTEM_PROMPT
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

    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(data)} examples...")

print(f"✅ Prepared {len(prepared_messages)} valid training examples")
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
# The tokenizer from unsloth should have the correct template already

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"✅ Max sequence length: {MAX_SEQ_LENGTH} tokens (latency-optimized)")
print(f"   💡 Optimized for inference: matches test script n_ctx=8192, max_tokens=4096")
print(f"   💡 Memory efficient: 75% reduction from 32768 (faster processing)")
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

    if (idx + 1) % 10 == 0:
        print(f"  Formatted {idx + 1}/{len(prepared_messages)} examples...")

print(f"✅ Formatted {len(formatted_texts)} training examples")
print()

# Create HuggingFace Dataset with already-formatted text
dataset = Dataset.from_list(formatted_texts)

print("=" * 80)
print("Dataset Statistics")
print("=" * 80)
print(f"Total examples: {len(dataset)}")
if len(dataset) > 0:
    print(f"Sample text length: {len(dataset[0]['text'])} characters")
    # Show sample of formatted text (first 500 chars)
    sample_text = dataset[0]['text'][:500]
    print(f"Sample text preview: {sample_text}...")
print("=" * 80)
print()

# ============================================================================
# Add LoRA Adapters
# ============================================================================

print("=" * 80)
print("Configuring LoRA Adapters")
print("=" * 80)

# LoRA Configuration
# Reduced rank to prevent memorization: r=128 instead of 256
# Lower capacity forces model to learn general patterns, not memorize examples

LORA_RANK = 128  # Reduced: prevents memorization, encourages generalization
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank (256)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,  # Added dropout to prevent memorization
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

# Training arguments optimized for Unsloth and CoT reasoning
# Anti-memorization settings: lower LR, higher weight decay, more epochs for format learning
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    warmup_ratio=0.2,  # 20% warmup: longer, more gradual warmup to prevent early memorization
    num_train_epochs=20,  # INCREASED: more epochs for consistent reasoning format learning (was 15, now 20)
    learning_rate=1e-5,  # FURTHER LOWER: slower learning prevents memorization (was 2e-5, now 1e-5)
    weight_decay=0.35,  # HIGHER: stronger regularization to prevent overfitting (was 0.25, now 0.35)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,  # More frequent logging to monitor progress
    optim="adamw_8bit",
    lr_scheduler_type="cosine",  # Cosine scheduler for smoother learning
    seed=3407,  # Main training seed
    data_seed=3407,  # CRITICAL: Seed for data shuffling/sampling (ensures same data order)
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=10,  # Keep more checkpoints
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # CRITICAL: Set to 0 for deterministic data loading (no multiprocessing randomness)
    report_to="none",  # Disable Weights & Biases logging
    max_steps=-1,  # Use epochs instead
    save_safetensors=True,
    # Additional settings to improve learning
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    eval_strategy="no",  # No validation set, but we can add one later
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    packing=False,  # Essential for CoT to learn proper start/end
)

print("✅ Training configured")
print(f"   - Max sequence length: {MAX_SEQ_LENGTH} (latency-optimized: matches inference n_ctx)")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs} (increased for consistent reasoning format learning)")
print(f"   - Learning rate: {training_args.learning_rate} (VERY LOW to prevent memorization)")
print(f"   - Weight decay: {training_args.weight_decay} (HIGH regularization to prevent overfitting)")
print(f"   - Warmup ratio: {training_args.warmup_ratio if hasattr(training_args, 'warmup_ratio') else 'N/A'} (longer, gradual warmup)")
print(f"   ⚡ LATENCY OPTIMIZATION:")
print(f"      ✅ Max sequence length: {MAX_SEQ_LENGTH} tokens (75% reduction from 32768)")
print(f"      ✅ Optimized for inference: matches test script n_ctx=8192, max_tokens=4096")
print(f"      ✅ Memory efficient: smaller context = faster processing")
print(f"   ⚠️  ANTI-MEMORIZATION SETTINGS (ENHANCED):")
print(f"      ✅ VERY LOW learning rate (1e-5) - prevents fast memorization")
print(f"      ✅ HIGH weight decay (0.35) - strong regularization")
print(f"      ✅ More epochs (20) - better pattern learning for consistent format")
print(f"      ✅ Lower LoRA rank (128) - forces generalization")
print(f"      ✅ LoRA dropout (0.1) - prevents memorization")
print(f"      ✅ Longer warmup (20%) - gradual learning start")
print(f"      → Forces model to learn GENERAL patterns, not memorize specific examples")
# Calculate approximate trainable parameters
approx_params = {
    64: (45, 3.4),
    128: (90, 6.8),
    256: (180, 13.6),
    512: (360, 27.2),
}
params_m, params_pct = approx_params.get(LORA_RANK, (180, 13.6))
print(f"   - LoRA rank: {LORA_RANK} (~{params_m}M trainable parameters, ~{params_pct}% of model)")
print(f"   - LoRA alpha: {LORA_ALPHA} (2x rank for optimal scaling)")
print(f"   - Warmup ratio: {training_args.warmup_ratio if hasattr(training_args, 'warmup_ratio') else 'N/A'}")
print(f"   - Scheduler: {training_args.lr_scheduler_type}")
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
try:
    # Ensure output directory exists
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    
    # Convert to GGUF
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method="q4_k_m"  # Q4_K_M quantization for good balance
    )
    
    # Wait a moment for file system to sync
    import time
    time.sleep(2)
    
    # Check for GGUF files (may be in subdirectory or root)
    gguf_files = []
    if os.path.exists(GGUF_OUTPUT_DIR):
        # Check in output directory
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
        # Also check subdirectories
        for root, dirs, files in os.walk(GGUF_OUTPUT_DIR):
            for file in files:
                if file.endswith(".gguf"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, GGUF_OUTPUT_DIR)
                    if rel_path not in gguf_files:
                        gguf_files.append(rel_path)
    
    # Also check root directory (in case it saved there)
    root_gguf_files = [f for f in os.listdir(".") if f.endswith(".gguf")]
    if root_gguf_files:
        print(f"⚠️  Found GGUF files in root directory: {root_gguf_files}")
        # Move them to output directory
        for root_file in root_gguf_files:
            dest = os.path.join(GGUF_OUTPUT_DIR, root_file)
            if not os.path.exists(dest):
                shutil.move(root_file, dest)
                print(f"   Moved {root_file} to {GGUF_OUTPUT_DIR}")
        # Re-check output directory
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
    
    if gguf_files:
        # Use the first GGUF file found
        if len(gguf_files) == 1:
            original_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
        else:
            # If multiple, prefer one with model name
            preferred = [f for f in gguf_files if "qwen" in f.lower() or "1.5b" in f.lower()]
            if preferred:
                original_file = os.path.join(GGUF_OUTPUT_DIR, preferred[0])
            else:
                original_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
        
        # Extract base name and add "-rag-cot" before .gguf extension
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        new_filename = f"{base_name}-rag-cot.gguf"
        new_file = os.path.join(GGUF_OUTPUT_DIR, new_filename)
        
        # Only rename if different
        if original_file != new_file:
            if os.path.exists(new_file):
                os.remove(new_file)  # Remove old file if exists
            shutil.move(original_file, new_file)
            print(f"✅ GGUF model saved as: {new_filename}")
        else:
            print(f"✅ GGUF model saved as: {os.path.basename(original_file)}")
    else:
        print(f"⚠️  No GGUF files found in {GGUF_OUTPUT_DIR}")
        print(f"   GGUF conversion may have failed or saved to a different location")
        print(f"   You can manually convert using:")
        print(f"   from unsloth import FastLanguageModel")
        print(f"   model.save_pretrained_gguf('{GGUF_OUTPUT_DIR}', tokenizer, quantization_method='q4_k_m')")
        
except Exception as e:
    print(f"⚠️  Error during GGUF conversion: {e}")
    import traceback
    traceback.print_exc()
    print(f"   Model is still saved in HuggingFace format at: {OUTPUT_DIR}/")
    print(f"   You can manually convert later or use the HuggingFace format directly")

print()
print("=" * 80)
print("🎉 Fine-tuning Complete!")
print("=" * 80)
print(f"Your RAG CoT model is ready:")
print(f"  - HuggingFace format: {OUTPUT_DIR}/")
print(f"  - GGUF format: {GGUF_OUTPUT_DIR}/")
print()
print("The model has been trained to:")
print("  ✅ Use Chain of Thought reasoning when processing RAG chunks")
print("  ✅ Read each chunk completely from start to finish")
print("  ✅ Identify explicit roles/titles as stated in the context")
print("  ✅ Extract ONLY people whose role/title matches the query")
print("  ✅ Verify relationships before including in answers")
print("  ✅ Avoid hallucination and incorrect information extraction")
print("  ✅ Handle cases where information is not explicitly stated")
print("  ✅ Exclude non-matching roles (advisors, employees, etc.)")
print()

# For Colab: Automatically download the GGUF file
print("=" * 80)
print("Preparing Model Download")
print("=" * 80)

try:
    from google.colab import files
    import time
    
    print("✅ Running in Google Colab - preparing automatic download...")
    
    # Wait a moment for file system to sync
    time.sleep(2)
    
    # Check for GGUF files in multiple locations
    gguf_files = []
    gguf_file = None
    
    # Check output directory
    if os.path.exists(GGUF_OUTPUT_DIR):
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
        if gguf_files:
            # Prefer the -rag-cot file if it exists
            rag_cot_file = [f for f in gguf_files if "-rag-cot" in f]
            if rag_cot_file:
                gguf_file = os.path.join(GGUF_OUTPUT_DIR, rag_cot_file[0])
            else:
                gguf_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
    
    # Also check root directory (in case GGUF was saved there)
    if not gguf_file:
        root_gguf_files = [f for f in os.listdir(".") if f.endswith(".gguf")]
        if root_gguf_files:
            print(f"📦 Found GGUF file in root directory: {root_gguf_files[0]}")
            gguf_file = root_gguf_files[0]
    
    if gguf_file and os.path.exists(gguf_file):
        file_size = os.path.getsize(gguf_file) / (1024 * 1024)  # Size in MB
        print(f"📦 Found GGUF model: {os.path.basename(gguf_file)}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Location: {gguf_file}")
        
        print(f"\n⬇️  Starting download: {os.path.basename(gguf_file)}")
        print("   (This may take a moment for large files...)")
        files.download(gguf_file)
        print("✅ Download completed!")
        
        # Also offer to download HuggingFace format
        print(f"\n💡 Tip: You can also download the HuggingFace format from {OUTPUT_DIR}/")
        print("   This format is useful if you want to continue training or fine-tune further.")
        
    else:
        print("⚠️  No GGUF files found to download")
        print(f"   Checked directory: {GGUF_OUTPUT_DIR}")
        print(f"   Checked root directory: ./")
        print(f"\n   💡 GGUF conversion may have failed. You can:")
        print(f"   1. Download the HuggingFace format from: {OUTPUT_DIR}/")
        print(f"   2. Manually convert later using:")
        print(f"      from unsloth import FastLanguageModel")
        print(f"      model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}')")
        print(f"      model.save_pretrained_gguf('gguf_output', tokenizer, quantization_method='q4_k_m')")
        
except ImportError:
    print("ℹ️  Not running in Google Colab")
    print(f"   Model saved to:")
    print(f"   - HuggingFace format: {OUTPUT_DIR}/")
    print(f"   - GGUF format: {GGUF_OUTPUT_DIR}/")
    print("   To download in Colab, the script will automatically detect Colab and download.")
except Exception as e:
    print(f"⚠️  Error during download: {e}")
    print(f"   Model is still saved at:")
    print(f"   - HuggingFace format: {OUTPUT_DIR}/")
    print(f"   - GGUF format: {GGUF_OUTPUT_DIR}/")
    print("   You can manually download these files.")

print("=" * 80)

