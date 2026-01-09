#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT Toggle Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Trains model to conditionally use Chain of Thought reasoning:
- WITH CoT when RAG context is provided (CoT system prompt)
- WITHOUT CoT for conversational queries (conversational system prompt)

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning)
- LoRA Rank: 128 (prevents memorization, encourages generalization)
- Strategy: Model learns to conditionally use CoT based on system prompt

To use in Colab:
1. Upload rag_cot_toggle_training_dataset.json to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

Dataset Features:
- Mixed dataset: 50% RAG+CoT examples, 50% conversational examples
- Model learns to use CoT only when CoT system prompt is present
- Conversational examples train model to respond naturally without CoT
"""

import json
import os
import shutil
import torch
from datasets import Dataset
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

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Qwen 2.5 1.5B

# Dataset path (merged dataset with CoT toggle examples)
DATASET_PATH = "rag_cot_toggle_training_dataset.json"

MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "outputs_cot_toggle"
GGUF_OUTPUT_DIR = "gguf_model_cot_toggle"

# System prompts are included in the dataset, but we have fallbacks just in case
COT_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- Read entire descriptions/chunks completely - titles may appear later in the text."""

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
# GPU Check
# ============================================================================

print("=" * 80)
print("GPU Configuration Check")
print("=" * 80)

if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  CUDA not available - training will be slow on CPU")
print()

# ============================================================================
# Load Dataset
# ============================================================================

print("=" * 80)
print("Loading Dataset")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset file '{DATASET_PATH}' not found!")
    print(f"   Please run merge_cot_toggle_dataset.py first to create the merged dataset.")
    exit(1)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} examples from {DATASET_PATH}")

# Count CoT vs conversational examples
cot_count = 0
conv_count = 0
for example in data:
    messages = example.get("messages", [])
    system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)
    if system_msg:
        content = system_msg.get("content", "")
        if "REASONING:" in content or "Start with REASONING" in content:
            cot_count += 1
        else:
            conv_count += 1

print(f"   RAG+CoT examples: {cot_count}")
print(f"   Conversational examples: {conv_count}")
print()

# ============================================================================
# Prepare Messages
# ============================================================================

print("=" * 80)
print("Preparing Message Structures")
print("=" * 80)

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
        # Try to determine which system prompt to use based on user message
        user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        if user_msg and "Knowledge context:" in user_msg.get("content", ""):
            chat_messages.append({
                "role": "system",
                "content": COT_SYSTEM_PROMPT
            })
        else:
            chat_messages.append({
                "role": "system",
                "content": CONVERSATIONAL_SYSTEM_PROMPT
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

    if (idx + 1) % 100 == 0:
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

    if (idx + 1) % 100 == 0:
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

LORA_RANK = 128  # Match original: prevents memorization, encourages generalization
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank (256)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,  # Match original: added dropout to prevent memorization
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

print(f"✅ LoRA configured: rank={LORA_RANK}, alpha={LORA_ALPHA}")
print()

# ============================================================================
# Training Configuration
# ============================================================================

print("=" * 80)
print("Training Configuration")
print("=" * 80)

# Training arguments optimized for Unsloth and CoT reasoning
# Anti-memorization settings: lower LR, higher weight decay, more epochs (but slower learning)
# MATCHING ORIGINAL train_rag_cot_colab.py settings to prevent rapid loss drop
TRAINING_ARGS = TrainingArguments(
    per_device_train_batch_size=1,  # Match original: smaller batch size
    gradient_accumulation_steps=8,  # Match original: effective batch size = 8
    warmup_steps=50,  # Match original: shorter warmup to prevent early memorization
    num_train_epochs=10,  # Reduced from 15: prevent overfitting/memorization with larger dataset
    learning_rate=2e-5,  # Match original: LOWER learning rate prevents memorization (was 2e-4, too high!)
    weight_decay=0.25,  # Match original: HIGHER weight decay for stronger regularization (was 0.01, too low!)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,  # Match original: more frequent logging
    optim="adamw_8bit",
    lr_scheduler_type="cosine",  # Match original: cosine scheduler for smoother learning
    seed=3407,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=10,  # Match original: keep more checkpoints
    dataloader_pin_memory=False,  # Match original
    report_to="none",  # Disable wandb/tensorboard
    max_steps=-1,  # Match original: use epochs instead of max_steps
    save_safetensors=True,  # Match original
    gradient_checkpointing=True,  # Match original: enable gradient checkpointing
    eval_strategy="no",  # No validation set
)

print(f"✅ Training arguments configured")
print(f"   Epochs: {TRAINING_ARGS.num_train_epochs}")
print(f"   Learning rate: {TRAINING_ARGS.learning_rate}")
print(f"   Batch size: {TRAINING_ARGS.per_device_train_batch_size}")
print(f"   Gradient accumulation: {TRAINING_ARGS.gradient_accumulation_steps}")
print()

# ============================================================================
# Train Model
# ============================================================================

print("=" * 80)
print("Starting Training")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    packing=False,  # Don't pack sequences
)

# Train
trainer.train()

print()
print("=" * 80)
print("Training Complete!")
print("=" * 80)
print()

# ============================================================================
# Save Model
# ============================================================================

print("=" * 80)
print("Saving Model")
print("=" * 80)

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ LoRA adapters saved to: {OUTPUT_DIR}")
print()

# Merge and save full model
print("Merging LoRA adapters with base model...")
FastLanguageModel.for_inference(model)  # Enable inference mode
model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")
print(f"✅ Merged model saved to: {OUTPUT_DIR}_merged")
print()

# ============================================================================
# Convert to GGUF (Optional - for llama.cpp)
# ============================================================================

print("=" * 80)
print("Converting to GGUF Format")
print("=" * 80)

try:
    from unsloth import is_bfloat16_supported
    FastLanguageModel.for_inference(model)  # Enable inference mode
    
    # Convert to GGUF
    model.save_pretrained_gguf(GGUF_OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"✅ GGUF model saved to: {GGUF_OUTPUT_DIR}")
    print(f"   Model file: {GGUF_OUTPUT_DIR}/model-q4_k_m.gguf")
except Exception as e:
    print(f"⚠️  GGUF conversion failed: {e}")
    print("   You can convert manually using llama.cpp later")
print()

print("=" * 80)
print("✅ Training Pipeline Complete!")
print("=" * 80)
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📁 Merged model: {OUTPUT_DIR}_merged")
if os.path.exists(GGUF_OUTPUT_DIR):
    print(f"📁 GGUF model: {GGUF_OUTPUT_DIR}")
print()
print("Next steps:")
print("1. Test the model with test_rag_cot_model_colab.py (update to use new model)")
print("2. Deploy the GGUF model to llm-container")
print("3. Verify CoT toggle behavior (CoT with RAG, no CoT for conversational)")
print("=" * 80)
