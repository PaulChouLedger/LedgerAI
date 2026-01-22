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
1. Upload medical_sft_dataset_complex.json (or other dataset) to Colab
2. Run: !pip install unsloth trl peft accelerate bitsandbytes datasets
3. Run this script

Dataset Priority (automatically selects best available):
1. medical_sft_dataset_complex.json (LATEST & MOST ADVANCED - highest priority)
   - Complex cross-organ system differentiation (GERD vs Cardiac, RUQ vs RLQ, etc.)
   - Clarification questions for ambiguous answers
   - Progressive scoring/ranking with rolling differential diagnosis
   - Associated symptom questions based on top 3 conditions
   - Context-aware OLD CARTS questions with examples from top conditions
   - Equal American/British English slang coverage
   - Generalizable methodology for any medical condition
2. medical_sft_dataset_enhanced_smart_intelligent.json (Enhanced with smart features)
3. medical_sft_dataset_enhanced_smart.json (Enhanced with smart features)
4. medical_sft_dataset_enhanced.json (Enhanced - includes negative examples, better OLD CARTS formats)
5. medical_sft_dataset_high_quality.json (includes clinical reasoning)
6. medical_sft_dataset_differential_reasoning.json
7. medical_sft_dataset_with_reasoning.json
8. medical_sft_dataset_complete.json
9. medical_sft_dataset_enriched.json
10. medical_sft_dataset.json (fallback)

Note: The COMPLEX dataset (recommended) includes:
- Complex cross-organ system differentiation scenarios
- Clarification questions when patient answers are ambiguous
- Progressive scoring: condition rankings update after each OLD CARTS answer
- Context-aware OLD CARTS questions with 1-2 example answers from top 3 conditions
- Associated symptom questions (after OLD CARTS) to differentiate top 3 conditions
- Clinical reasoning after each answer with updated probability rankings
- Generalizable methodology: trained process works for any medical condition
"""

import json
import os
import shutil
import torch
import subprocess
import sys
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

#
# Model selection
# ---------------
# Recommended default for Colab A100 (especially 80GB): 14B gives a large reasoning lift
# over 1.5B while still being easy to fine-tune with LoRA.
#
# Note: This script uses Unsloth 4-bit loading + LoRA (QLoRA-style).
# If you switch to a different base model, prefer an Unsloth *-bnb-4bit* checkpoint.
MODEL_NAME = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"

# Dataset Priority (highest to lowest):
# 1. medical_sft_dataset_complex.json - LATEST & MOST ADVANCED
#    - Complex cross-organ system differentiation (GERD vs Cardiac, RUQ vs RLQ, etc.)
#    - Clarification questions for ambiguous answers
#    - Progressive scoring/ranking with rolling differential diagnosis
#    - Associated symptom questions based on top 3 conditions
#    - Context-aware OLD CARTS questions with examples from top conditions
#    - Equal American/British English slang coverage
#    - LLM knowledge leveraged for any condition (generalizable methodology)
# 2. medical_sft_dataset_enhanced_smart_intelligent.json - Enhanced with smart features
#    - Smart OLD CARTS question selection (skip irrelevant elements)
#    - British slang variations for UK market
#    - Intelligent follow-up questions based on diagnosis
#    - Clinical reasoning and skip tags
# 3. medical_sft_dataset_enhanced_smart.json - Enhanced with smart features
#    - Smart OLD CARTS question selection
#    - British slang variations
# 4. medical_sft_dataset_enhanced.json - Enhanced (includes negative examples, better OLD CARTS formats)
# 5. medical_sft_dataset_high_quality.json - Includes clinical reasoning
# 6. medical_sft_dataset_differential_reasoning.json - Includes differential reasoning
# 7. medical_sft_dataset_with_reasoning.json - Includes basic reasoning
# 8. medical_sft_dataset_complete.json - Complete conversations
# 9. medical_sft_dataset_enriched.json - Enriched version
# 10. medical_sft_dataset.json - Original fallback

if os.path.exists("medical_sft_dataset_complex.json"):
    DATASET_PATH = "medical_sft_dataset_complex.json"
    print("✅ Using COMPLEX dataset (latest, most advanced)")
    print("   📚 Features:")
    print("      - Complex cross-organ system differentiation")
    print("      - Clarification questions for ambiguous answers")
    print("      - Progressive scoring/ranking with rolling differential")
    print("      - Associated symptom questions (top 3 conditions)")
    print("      - Context-aware OLD CARTS questions with examples")
    print("      - Equal American/British English coverage")
    print("      - Generalizable methodology for any condition")
elif os.path.exists("medical_sft_dataset_enhanced_smart_intelligent.json"):
    DATASET_PATH = "medical_sft_dataset_enhanced_smart_intelligent.json"
    print("✅ Using ENHANCED SMART INTELLIGENT dataset")
    print("   📚 Features:")
    print("      - Smart OLD CARTS question selection (skip irrelevant elements)")
    print("      - British slang variations for UK market")
    print("      - Intelligent follow-up questions based on diagnosis")
    print("      - Clinical reasoning and skip tags")
elif os.path.exists("medical_sft_dataset_enhanced_smart.json"):
    DATASET_PATH = "medical_sft_dataset_enhanced_smart.json"
    print("✅ Using ENHANCED SMART dataset")
    print("   📚 Features:")
    print("      - Smart OLD CARTS question selection")
    print("      - British slang variations")
elif os.path.exists("medical_sft_dataset_enhanced.json"):
    DATASET_PATH = "medical_sft_dataset_enhanced.json"
    print("✅ Using ENHANCED dataset (includes negative examples, better OLD CARTS formats)")
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

# Check for smart features
sample_conv = data[0] if data else {}
has_smart_features = sample_conv.get("smart_features", False)
has_intelligent_followups = sample_conv.get("has_intelligent_followups", False)
has_relevant_oldcarts = "relevant_oldcarts" in sample_conv
has_differential_diagnosis = "differential_diagnosis" in sample_conv

# Check for variants (British/American)
has_variants = any(conv.get("variant") in ["american", "british"] for conv in data[:10])

# Check if dataset includes clinical reasoning
sample_messages = sample_conv.get("messages", [])
has_reasoning = any("CLINICAL REASONING" in msg.get("content", "") or
                    "more concerning" in msg.get("content", "").lower() or
                    "probability" in msg.get("content", "").lower()
                    for msg in sample_messages)

# Check for progressive scoring/ranking
has_progressive_scoring = any("CURRENT DIFFERENTIAL DIAGNOSIS" in msg.get("content", "") or
                              "ranked by probability" in msg.get("content", "").lower()
                              for msg in sample_messages)

# Check for associated symptoms
has_associated_symptoms = any("Associated Symptom Assessment" in msg.get("content", "") or
                              "Associated Symptom" in msg.get("content", "")
                              for msg in sample_messages)

# Check for clarification questions
has_clarification = any("upper abdomen" in msg.get("content", "").lower() and
                        "lower abdomen" in msg.get("content", "").lower() and
                        msg.get("role") == "assistant"
                        for msg in sample_messages)

# Check for skip tags
has_skip_tags = any(msg.get("metadata", {}).get("skip") for conv in data[:10]
                    for msg in conv.get("messages", []))

# Check for context-aware questions with examples
has_examples_in_questions = any("For example" in msg.get("content", "") and
                                "?" in msg.get("content", "") and
                                msg.get("role") == "assistant"
                                for msg in sample_messages)

# Print dataset features
print()
if has_smart_features:
    print("📚 Smart Features Detected:")
    print("   ✅ Smart OLD CARTS question selection")
    if has_relevant_oldcarts:
        print("   ✅ Relevance metadata for each conversation")
    if has_skip_tags:
        print("   ✅ Skip tags for irrelevant questions")
    if has_variants:
        american_count = sum(1 for c in data if c.get("variant") == "american")
        british_count = sum(1 for c in data if c.get("variant") == "british")
        print(f"   ✅ British slang variations ({british_count} British, {american_count} American variants)")
    if has_intelligent_followups:
        followup_count = sum(1 for c in data if c.get("has_intelligent_followups"))
        print(f"   ✅ Intelligent follow-up questions ({followup_count} conversations)")
        print("      - Diagnosis-specific questions")
        print("      - Medication, risk factor, and lifestyle questions")
        print("      - Clinical reasoning for follow-ups")
    print()

if has_progressive_scoring:
    print("📊 Progressive Scoring Features:")
    print("   ✅ Rolling differential diagnosis with probability rankings")
    print("   ✅ Condition scores update after each OLD CARTS answer")
    print("   ✅ Rankings shown after each element assessment")
    print()

if has_associated_symptoms:
    print("🔍 Associated Symptom Features:")
    print("   ✅ Associated symptom questions based on top 3 conditions")
    print("   ✅ Questions designed to differentiate between likely diagnoses")
    print("   ✅ Clinical reasoning with updated rankings after each symptom")
    print()

if has_clarification:
    print("❓ Clarification Question Features:")
    print("   ✅ Clarifying questions for ambiguous answers")
    print("   ✅ Location clarification (RUQ vs RLQ, etc.)")
    print("   ✅ Answer sufficiency assessment")
    print()

if has_examples_in_questions:
    print("💡 Context-Aware Question Features:")
    print("   ✅ Example answers in OLD CARTS questions")
    print("   ✅ Examples based on top 3 ranking conditions")
    print("   ✅ Guides users toward diagnostically useful answers")
    print()

if has_differential_diagnosis:
    print("🎯 Differential Diagnosis Features:")
    print("   ✅ Explicit differential diagnosis lists per conversation")
    print("   ✅ Cross-organ system differentiation")
    print()

if has_reasoning:
    print("ℹ️  Clinical Reasoning Features:")
    print("   - Clinical reasoning after each OLD CARTS answer")
    print("   - Comparative thinking (more concerning for X than Y)")
    print("   - Rule-in/rule-out logic with probability rankings")
    print("   - Progressive narrowing of differential diagnosis")
    if has_associated_symptoms:
        print("   - Associated symptoms with reasoning")
    print("   - Final diagnostic reasoning with ranked differential")
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

# LoRA rank guidance:
# - 1.5B: r=256 is reasonable.
# - 7B/14B: start with r=128 (better generalization on small datasets; plenty of capacity).
# - 32B+: start with r=64–128 unless you have a large dataset.
LORA_RANK = 128
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
    # A100 80GB can handle more, but keep defaults conservative and stable.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=75,  # Increased warmup for better clinical reasoning learning
    # With the current dataset (often ~100 conversations), 10 epochs can overfit quickly on 14B+.
    # Start with 3 epochs; increase only if you expand the dataset meaningfully.
    num_train_epochs=3,
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

# Pre-install llama.cpp to avoid Unsloth's broken build process
print("Pre-installing llama.cpp (workaround for Unsloth build issues)...")
try:
    if not os.path.exists("llama.cpp"):
        print("   Cloning llama.cpp repository...")
        subprocess.check_call([
            "git", "clone", "--depth", "1", 
            "https://github.com/ggerganov/llama.cpp.git"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ llama.cpp cloned")
    else:
        print("   ✅ llama.cpp already exists")
    
    # Build llama.cpp with correct CMake options (without deprecated LLAMA_CURL)
    llama_cpp_build = os.path.join("llama.cpp", "build")
    if not os.path.exists(os.path.join(llama_cpp_build, "bin", "quantize")):
        print("   Building llama.cpp (this may take a few minutes)...")
        current_dir = os.getcwd()
        os.chdir("llama.cpp")
        try:
            # Build with CMake (correct options, no deprecated LLAMA_CURL)
            subprocess.check_call([
                "cmake", "-B", "build", 
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=OFF"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([
                "cmake", "--build", "build", "--config", "Release", "-j"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chdir(current_dir)
            print("   ✅ llama.cpp built successfully")
        except Exception as build_err:
            os.chdir(current_dir)
            print(f"   ⚠️  llama.cpp build failed: {build_err}")
            print("   Will let Unsloth try its own build (may fail)")
    else:
        print("   ✅ llama.cpp already built")
        
except Exception as e:
    print(f"   ⚠️  Pre-installation failed: {e}")
    print("   Will let Unsloth try its own build (may fail)")

# Convert to GGUF using Unsloth
# If llama.cpp is pre-built, Unsloth will use it instead of trying to build
try:
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
        
except RuntimeError as e:
    error_msg = str(e)
    if "llama.cpp" in error_msg or "FAILED building" in error_msg or "CMake failed" in error_msg:
        print(f"\n   ⚠️  GGUF conversion failed: llama.cpp build issue")
        print(f"   This is a known issue with Unsloth's automatic llama.cpp installation")
        print(f"   The model is saved in HuggingFace format and can be converted later")
        print(f"   Model saved at: {OUTPUT_DIR}/")
    else:
        print(f"\n   ❌ GGUF conversion failed: {e}")
        print(f"   Model is still saved in HuggingFace format at: {OUTPUT_DIR}/")
except Exception as e:
    print(f"\n   ❌ Unexpected error during GGUF conversion: {e}")
    print(f"   Model is still saved in HuggingFace format at: {OUTPUT_DIR}/")

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
if "complex" in DATASET_PATH.lower():
    print("  ✅ Handle complex cross-organ system differentiation")
    print("  ✅ Ask clarification questions for ambiguous answers")
    print("  ✅ Provide context-aware OLD CARTS questions with examples from top conditions")
    print("  ✅ Ask associated symptom questions based on top 3 ranking conditions")
    print("  ✅ Use progressive scoring with rolling rankings")
    print("  ✅ Apply generalizable methodology to any medical condition")
elif "smart" in DATASET_PATH.lower():
    print("  ✅ Skip irrelevant OLD CARTS questions (smart question selection)")
    if "intelligent" in DATASET_PATH.lower():
        print("  ✅ Ask intelligent follow-up questions based on diagnosis")
        print("  ✅ Leverage medical knowledge (medications, risk factors, etc.)")
if any(conv.get("variant") in ["american", "british"] for conv in data[:10]):
    print("  ✅ Handle both American and British English")
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

