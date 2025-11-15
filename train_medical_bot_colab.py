#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Bot Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Properly formats medical conversations using Qwen 2.5 chat template
Trains model to think like a doctor with clinical reasoning

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning than 0.5B)
- LoRA Rank: 256 (good balance for 1.5B model)
- Strategy: 1.5B provides better base reasoning + LoRA adaptation for task patterns

To use in Colab:
1. Upload medical_sft_dataset_enhanced.json (or other dataset) to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

Dataset Priority (automatically selects best available):
1. medical_sft_dataset_enhanced.json (LATEST & MOST ADVANCED - highest priority)
   - Negative examples (what NOT to ask)
   - Improved OLD CARTS question formats
   - Better instruction following examples
   - Clinical reasoning and associated symptoms
2. medical_sft_dataset_high_quality.json (includes clinical reasoning)
3. medical_sft_dataset_differential_reasoning.json
4. medical_sft_dataset_with_reasoning.json
5. medical_sft_dataset_complete.json
6. medical_sft_dataset_enriched.json
7. medical_sft_dataset.json (fallback)

Note: The ENHANCED dataset (recommended) includes:
- Negative examples showing what NOT to ask (prevents repetitive/nonsensical questions)
- Improved OLD CARTS question formats (prevents awkward phrasing)
- Better instruction following examples
- Clinical reasoning after each OLD CARTS answer (comparative thinking, rule-in/rule-out logic)
- Progressive narrowing of differential diagnosis with probability rankings
- Associated symptoms with reasoning
- Final diagnostic reasoning with ranked differential
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

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Qwen 2.5 1.5B - better instruction following and reasoning

# Dataset Priority (highest to lowest):
# 1. medical_sft_dataset_enhanced.json - LATEST & MOST ADVANCED (includes negative examples, better OLD CARTS formats)
# 2. medical_sft_dataset_high_quality.json - Includes clinical reasoning
# 3. medical_sft_dataset_differential_reasoning.json - Includes differential reasoning
# 4. medical_sft_dataset_with_reasoning.json - Includes basic reasoning
# 5. medical_sft_dataset_complete.json - Complete conversations
# 6. medical_sft_dataset_enriched.json - Enriched version
# 7. medical_sft_dataset.json - Original fallback

if os.path.exists("medical_sft_dataset_enhanced.json"):
    DATASET_PATH = "medical_sft_dataset_enhanced.json"
    print("✅ Using ENHANCED dataset (latest, most advanced)")
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
FALLBACK_SYSTEM_PROMPT = """You are a medical professional with extensive medical knowledge. You can:
1. Answer general medical information questions (e.g., "what is diabetes?")
2. Conduct clinical assessments when patients mention symptoms (using OLD CARTS framework)

CRITICAL: Distinguish between:
- **Informational questions** (e.g., "what is diabetes?", "explain heart attack") → Provide clear, helpful medical information
- **Symptom complaints** (e.g., "I have chest pain") → Conduct systematic OLD CARTS assessment

When conducting a clinical assessment, your job is to STRUCTURE the conversation and REASON SYSTEMATICALLY to arrive at a diagnosis.

CRITICAL INSIGHT: You already know medical facts. What you need is STRUCTURE:
1. Ask questions SYSTEMATICALLY using OLD CARTS framework (one element at a time)
2. After EACH answer, reason methodically: How does this affect the differential?
3. Build diagnosis STEP-BY-STEP: Rule IN conditions that match, Rule OUT conditions that don't
4. Progressively narrow: Each collected element should refine your differential
5. Use your medical knowledge, but apply it in a STRUCTURED, SYSTEMATIC way

SYSTEMATIC REASONING PROCESS:
- Step 1: Collect Onset (O) → Reason: How does onset pattern affect differential?
- Step 2: Collect Location (L) → Reason: How does location narrow the differential?
- Step 3: Collect Character (C) → Reason: How does character further refine diagnosis?
- Continue systematically through all OLD CARTS elements
- Build diagnosis progressively: Each answer should update your differential rankings

Use your medical knowledge to understand:
- Anatomical relationships (e.g., "right upper quadrant pain" → liver, gallbladder, biliary system)
- Medical terminology (e.g., "epigastric" → stomach/pancreas, "pleuritic" → pleural/pulmonary)
- Clinical patterns (e.g., "fatty meal trigger" → gallbladder, "worse with breathing" → pulmonary)
But apply this knowledge SYSTEMATICALLY through structured questioning and step-by-step reasoning.

IMPORTANT RULES:
- Answer general medical questions naturally and informatively
- ONLY ask medical assessment questions when the patient mentions a symptom, pain, or medical concern
- If the patient is just greeting you or having casual conversation, respond naturally and wait for them to mention a medical issue
- NEVER make up or assume symptoms the patient hasn't mentioned
- Always ask questions during assessments, never make statements about patient information
- NEVER ask redundant questions about information already provided
- USE YOUR MEDICAL KNOWLEDGE: When patients describe symptoms, use your understanding of anatomy, medical terminology, and clinical patterns to make connections and guide your reasoning

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
if "enhanced" in DATASET_PATH.lower():
    print("   📚 This is the ENHANCED dataset with:")
    print("      - Negative examples (what NOT to ask)")
    print("      - Improved OLD CARTS question formats")
    print("      - Better instruction following examples")

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

# Qwen 2.5 uses its own chat template (automatically set by tokenizer)
# The tokenizer from unsloth should have the correct template already
# If needed, we can verify it's using Qwen format: <|im_start|>system\n...<|im_end|>\n

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
# r=256: ~180M trainable (13.6%) - Better capacity, needs 16GB+ VRAM (CURRENT)
# r=512: ~360M trainable (27.2%) - Maximum capacity, needs 24GB+ VRAM
# 
# Using r=256 with Qwen2.5-1.5B for good balance of base model reasoning + task adaptation.
# 1.5B provides better instruction following and reasoning than 0.5B.
# See BASE_MODEL_SIZE_VS_LORA_RANK.md and UPGRADE_TO_1.5B_RECOMMENDATION.md for details.

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

# Rename GGUF file to append "-medical" to filename
gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
if gguf_files:
    original_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
    # Extract base name and add "-medical" before .gguf extension
    base_name = os.path.splitext(gguf_files[0])[0]
    new_filename = f"{base_name}-medical.gguf"
    new_file = os.path.join(GGUF_OUTPUT_DIR, new_filename)
    shutil.move(original_file, new_file)
    print(f"✅ GGUF model saved as: {new_filename}")
else:
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

gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
if gguf_files:
    # Prefer the -medical file if it exists, otherwise use any GGUF file
    medical_file = [f for f in gguf_files if "-medical" in f]
    if medical_file:
        gguf_file = os.path.join(GGUF_OUTPUT_DIR, medical_file[0])
    else:
        gguf_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
    print(f"Downloading: {gguf_file}")
    files.download(gguf_file)
else:
    print("⚠️  No GGUF files found to download")

print("=" * 80)

