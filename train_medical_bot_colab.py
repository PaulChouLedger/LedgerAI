#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Bot Fine-Tuning Script for Llama-3.2 (Google Colab Version)
Properly formats medical conversations using Llama-3.2 chat template
Trains model to think like a doctor with clinical reasoning

To use in Colab:
1. Upload medical_sft_dataset_high_quality.json (or other dataset) to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

Dataset Priority:
- medical_sft_dataset_high_quality.json (highest priority - includes clinical reasoning and associated symptoms)
- medical_sft_dataset_differential_reasoning.json
- medical_sft_dataset_with_reasoning.json
- medical_sft_dataset_complete.json
- medical_sft_dataset_enriched.json
- medical_sft_dataset.json (fallback)

Note: The high-quality dataset includes:
- Clinical reasoning after each OLD CARTS answer (comparative thinking, rule-in/rule-out logic)
- Progressive narrowing of differential diagnosis with probability rankings
- Associated symptoms with reasoning
- Final diagnostic reasoning with ranked differential
- System prompt: "Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability"
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
# Priority: enhanced > high quality > differential reasoning > end-of-conversation reasoning > complete > enriched > original
if os.path.exists("medical_sft_dataset_enhanced.json"):
    DATASET_PATH = "medical_sft_dataset_enhanced.json"
elif os.path.exists("medical_sft_dataset_high_quality.json"):
    DATASET_PATH = "medical_sft_dataset_high_quality.json"
elif os.path.exists("medical_sft_dataset_differential_reasoning.json"):
    DATASET_PATH = "medical_sft_dataset_differential_reasoning.json"
elif os.path.exists("medical_sft_dataset_with_reasoning.json"):
    DATASET_PATH = "medical_sft_dataset_with_reasoning.json"
elif os.path.exists("medical_sft_dataset_complete.json"):
    DATASET_PATH = "medical_sft_dataset_complete.json"
elif os.path.exists("medical_sft_dataset_enriched.json"):
    DATASET_PATH = "medical_sft_dataset_enriched.json"
else:
    DATASET_PATH = "medical_sft_dataset.json"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"
GGUF_OUTPUT_DIR = "gguf_model"

# Fallback system prompt (used only if dataset doesn't include system prompts)
# The high-quality dataset includes its own system prompt that encourages clinical reasoning
FALLBACK_SYSTEM_PROMPT = """You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.

IMPORTANT RULES:
- ONLY ask medical questions when the patient mentions a symptom, pain, or medical concern
- If the patient is just greeting you or having casual conversation, respond naturally and wait for them to mention a medical issue
- NEVER make up or assume symptoms the patient hasn't mentioned
- Always ask questions, never make statements about patient information
- NEVER ask redundant questions about information already provided

CRITICAL SEQUENCE - You MUST follow this EXACT order for EVERY conversation. DO NOT skip any step:

STEP 1: Show empathy and acknowledge their concern (REQUIRED - do this FIRST when patient mentions a symptom)
STEP 2: Ask if this is new or an ongoing problem (REQUIRED - do this SECOND, BEFORE age)
STEP 3: Ask their age (REQUIRED - do this THIRD, AFTER chronicity)
STEP 4: Ask their biological sex (REQUIRED - do this FOURTH, AFTER age)
STEP 5: THEN and ONLY THEN ask about the symptom using OLD CARTS - one question at a time

CRITICAL: After collecting demographics (age, biological sex), you MUST ONLY ask OLD CARTS questions. 
DO NOT ask about age, biological sex, or demographics again during HPI.
DO NOT ask questions like "how old is your [symptom]?" - this makes no sense.

OLD CARTS QUESTION FORMATS (use these exact patterns):
- Onset (O): "When did [symptom] start?" or "When did it start?"
- Location (L): "Where exactly is the [symptom] located?"
- Duration (D): "How long has the [symptom] been present?"
- Character (C): "What does the [symptom] feel like? For example, is it sharp, heavy, burning, or pressure?"
- Aggravating (A): "What makes the [symptom] worse?"
- Alleviating (A): "What makes the [symptom] better?"
- Radiation (R): "Does the [symptom] spread to other areas?"
- Timing (T): "Is the [symptom] constant or does it come and go?"
- Severity (S): "On a scale from 1 to 10, how severe is the [symptom]?"

When asking OLD CARTS questions, ask about: when it started, where it is, how long it's been present, what it feels like, what makes it worse, what makes it better, if it spreads, if it's constant or comes and goes, and how severe it is.

After each OLD CARTS answer, provide clinical reasoning showing:
- How the answer affects the differential diagnosis
- Comparative thinking (e.g., "more concerning for X than Y")
- Rule-in/rule-out logic
- Updated probability rankings
- Progressive narrowing of the differential

Be natural and conversational. Ask only one question at a time."""

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

# Check if dataset includes clinical reasoning
sample_conv = data[0] if data else {}
sample_messages = sample_conv.get("messages", [])
has_reasoning = any("CLINICAL REASONING" in msg.get("content", "") or 
                    "more concerning" in msg.get("content", "").lower() or
                    "probability" in msg.get("content", "").lower()
                    for msg in sample_messages)

if has_reasoning:
    print("ℹ️  Dataset includes clinical reasoning:")
    print("   - Clinical reasoning after each OLD CARTS answer")
    print("   - Comparative thinking (more concerning for X than Y)")
    print("   - Rule-in/rule-out logic with probability rankings")
    print("   - Progressive narrowing of differential diagnosis")
    print("   - Associated symptoms with reasoning")
    print("   - Final diagnostic reasoning with ranked differential")
else:
    print("ℹ️  Dataset format detected (may not include clinical reasoning)")
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

# LoRA Configuration
# r=128: ~90M trainable (6.80%) - Good balance, works on T4/V100
# r=256: ~180M trainable (13.6%) - Better capacity, needs 16GB+ VRAM
# r=512: ~360M trainable (27.2%) - Maximum capacity, needs 24GB+ VRAM
# 
# Recommendation: Start with r=128. If model underfits or you have extra VRAM, try r=256.
# See LORA_CONFIGURATION_GUIDE.md for details.

LORA_RANK = 128  # Change to 256 or 512 to train more parameters
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank

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

# Training arguments optimized for Unsloth and medical conversations
# Enhanced for clinical reasoning and OLD CARTS framework adherence
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=75,  # Increased warmup for better clinical reasoning learning
    num_train_epochs=10,  # Increased to 10 for better element identification and condition matching
    learning_rate=1.5e-4,  # Slightly lower for more stable learning
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=25,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # Cosine scheduler for smoother learning
    seed=3407,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=5,  # Keep more checkpoints for 10 epochs
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
print(f"   - Epochs: {training_args.num_train_epochs} (optimized for clinical reasoning and OLD CARTS adherence)")
print(f"   - Learning rate: {training_args.learning_rate}")
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
print(f"   - To train more parameters, change LORA_RANK to 256 or 512 (see LORA_CONFIGURATION_GUIDE.md)")
print(f"   - Warmup steps: {training_args.warmup_steps}")
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
print(f"Your medical bot model is ready with clinical reasoning capabilities:")
print(f"  - HuggingFace format: {OUTPUT_DIR}/")
print(f"  - GGUF format: {GGUF_OUTPUT_DIR}/")
print()
print("The model has been trained to:")
print("  ✅ Follow OLD CARTS sequence (empathy → chronicity → age → sex → OLD CARTS)")
print("  ✅ Provide clinical reasoning after each answer")
print("  ✅ Use comparative thinking (more concerning for X than Y)")
print("  ✅ Build ranked differential diagnoses with probability updates")
print("  ✅ Progressively narrow differential as more information is gathered")
print("  ✅ Include associated symptoms with reasoning")
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

