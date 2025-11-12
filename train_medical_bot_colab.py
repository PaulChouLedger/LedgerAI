#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Bot Fine-Tuning Script for Llama-3.2 (Google Colab Version)
Properly formats medical conversations using Llama-3.2 chat template

To use in Colab:
1. Upload medical_sft_dataset.json to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# ============================================================================
# Install Dependencies (Colab)
# ============================================================================
# Uncomment if running in Colab:
# !pip install unsloth trl peft accelerate bitsandbytes datasets

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"  # Use Instruct version for chat
DATASET_PATH = "medical_sft_dataset.json"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"
GGUF_OUTPUT_DIR = "gguf_model"

# Medical system prompt to guide the model's behavior
MEDICAL_SYSTEM_PROMPT = """You are a professional medical assistant designed to help with patient assessment and documentation. Your role is to:

1. Conduct thorough medical history taking following structured frameworks (e.g., SOCRATES, OLD CARTS)
2. Ask appropriate follow-up questions to gather complete clinical information
3. Use professional medical terminology while remaining empathetic
4. Document information clearly and systematically
5. Recognize when to escalate urgent medical concerns
6. Maintain patient privacy and confidentiality

Always be professional, empathetic, and thorough in your medical assessments."""

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
print("Loading Medical Dataset")
print("=" * 80)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} conversations from {DATASET_PATH}")
print()

# Prepare message structures (without formatting yet)
prepared_messages = []

for idx, conversation in enumerate(data):
    messages = conversation.get("messages", [])
    
    if not messages:
        continue
    
    # Build messages list with system prompt
    chat_messages = []
    
    # Add system prompt at the beginning if not already present
    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        chat_messages.append({
            "role": "system",
            "content": MEDICAL_SYSTEM_PROMPT
        })
    
    # Add all conversation messages
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        
        if role and content:
            # Skip system messages if we already added our own
            if role == "system" and not has_system:
                continue
            chat_messages.append({
                "role": role,
                "content": content
            })
    
    if len(chat_messages) < 2:  # Need at least system + one user/assistant message
        continue
    
    prepared_messages.append(chat_messages)
    
    if (idx + 1) % 100 == 0:
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

# Set chat template for Llama-3.2
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"✅ Chat template configured for Llama-3.2")
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
    print(f"Sample text length: {len(dataset[0]['text'])} characters")
print("=" * 80)
print()

# ============================================================================
# Add LoRA Adapters
# ============================================================================

print("=" * 80)
print("Configuring LoRA Adapters")
print("=" * 80)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher = more capacity, more memory
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
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

# Training arguments optimized for Unsloth and medical conversations
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=50,  # Increased for better convergence
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=25,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_pin_memory=False,
    report_to="none",  # Disable Weights & Biases logging
    # Medical-specific optimizations
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
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Learning rate: {training_args.learning_rate}")
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
model.save_pretrained_gguf(
    GGUF_OUTPUT_DIR,
    tokenizer,
    quantization_method="q4_k_m"  # Q4_K_M quantization for good balance
)
print(f"✅ GGUF model saved to {GGUF_OUTPUT_DIR}")

print()
print("=" * 80)
print("🎉 Fine-tuning Complete!")
print("=" * 80)
print(f"Your medical bot model is ready:")
print(f"  - HuggingFace format: {OUTPUT_DIR}/")
print(f"  - GGUF format: {GGUF_OUTPUT_DIR}/")
print()

# For Colab: Download the GGUF file
from google.colab import files
import os

gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
if gguf_files:
    gguf_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
    print(f"Downloading: {gguf_file}")
    files.download(gguf_file)
else:
    print("⚠️  No GGUF files found to download")

print("=" * 80)

