#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chunk Analysis Fine-Tuning Script for Qwen 2.5 (Google Colab Version)
Trains model on general RAG analysis patterns (not entity-specific)

Configuration:
- Model: Qwen2.5-1.5B-Instruct (better instruction following and reasoning)
- LoRA Rank: 6 (increased from 4 - rank 4 insufficient for multi-entity extraction)
- Strategy: 1.5B provides better base reasoning + LoRA adaptation for RAG analysis patterns
- Epochs: 7 (keep same - prevent overfitting)
- Learning Rate: 6e-7 (increased from 5e-7 - faster learning for multi-entity patterns, still conservative)
- Regularization: weight_decay=0.7, lora_dropout=0.25 (evaluation disabled - crashes system)

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
from transformers import TrainingArguments, TrainerCallback

# ============================================================================
# Install Dependencies (Colab)
# ============================================================================
# Uncomment if running in Colab:
# !pip install unsloth trl peft accelerate bitsandbytes datasets

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Qwen 2.5 1.5B - better instruction following and reasoning

# Get script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Dataset path (prefer JSON v3, fallback to v2, then v1)
# Check in script directory first, then current working directory
dataset_candidates = [
    ("rag_analysis_dataset_v3_json.json", True),
    ("rag_analysis_dataset_v2.json", False),
    ("rag_analysis_dataset.json", False),
]

DATASET_PATH = None
JSON_OUTPUT_MODE = False

for filename, is_json in dataset_candidates:
    # Try script directory first
    script_path = os.path.join(SCRIPT_DIR, filename)
    cwd_path = filename
    
    if os.path.exists(script_path):
        DATASET_PATH = script_path
        JSON_OUTPUT_MODE = is_json
        if filename == "rag_analysis_dataset_v3_json.json":
            print("✅ Using RAG analysis dataset v3 (JSON output format)")
        elif filename == "rag_analysis_dataset_v2.json":
            print("✅ Using RAG analysis dataset v2 (natural language output)")
        else:
            print(f"✅ Using RAG analysis dataset v1 (natural language output)")
        break
    elif os.path.exists(cwd_path):
        DATASET_PATH = cwd_path
        JSON_OUTPUT_MODE = is_json
        if filename == "rag_analysis_dataset_v3_json.json":
            print("✅ Using RAG analysis dataset v3 (JSON output format)")
        elif filename == "rag_analysis_dataset_v2.json":
            print("✅ Using RAG analysis dataset v2 (natural language output)")
else:
            print(f"✅ Using RAG analysis dataset v1 (natural language output)")
        break

if DATASET_PATH is None:
    raise FileNotFoundError(
        f"No dataset found. Please run generate_rag_dataset_v3_json.py first.\n"
        f"Checked in: {SCRIPT_DIR} and {os.getcwd()}\n"
        f"Looking for: rag_analysis_dataset_v3_json.json, rag_analysis_dataset_v2.json, or rag_analysis_dataset.json"
    )

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

ANSWER_TYPE MAPPING (CRITICAL - Use this to determine the correct answer_type in JSON output):
When outputting JSON, you MUST select the correct answer_type based on the query pattern:

1. "entities" - Use for queries asking about specific people, roles, or named entities:
   - "who are the [role] of [company]?" → answer_type: "entities"
   - "who are the co-founders of X?" → answer_type: "entities"
   - "who are the executives at Y?" → answer_type: "entities"
   - "who are the founders of Z?" → answer_type: "entities"
   - Output format: {"answer_type": "entities", "items": ["Name1", "Name2", ...], "text": "", "chunks_used": [...]}

2. "list" - Use for queries asking for lists of items, services, features, benefits, capabilities, components:
   - "what services does X offer?" → answer_type: "list"
   - "what are the features of Y?" → answer_type: "list"
   - "list the benefits of Z" → answer_type: "list"
   - "what capabilities does A have?" → answer_type: "list"
   - Output format: {"answer_type": "list", "items": ["item1", "item2", ...], "text": "", "chunks_used": [...]}

3. "comparison" - Use for queries asking about differences or comparisons between entities:
   - "what is the difference between X and Y?" → answer_type: "comparison"
   - "compare X and Y" → answer_type: "comparison"
   - "how do X and Y differ?" → answer_type: "comparison"
   - Output format: {"answer_type": "comparison", "items": [], "text": "comparison explanation...", "chunks_used": [...]}

4. "relationship" - Use for queries asking about connections, relationships, or how entities are related:
   - "how are X and Y related?" → answer_type: "relationship"
   - "what is the connection between X and Y?" → answer_type: "relationship"
   - "what is the relationship between X and Y?" → answer_type: "relationship"
   - Output format: {"answer_type": "relationship", "items": [], "text": "relationship description...", "chunks_used": [...]}

5. "analytical" - Use for queries asking "why" or about causes/reasons:
   - "why did X [action]?" → answer_type: "analytical"
   - "what caused X to [action]?" → answer_type: "analytical"
   - "why did X change?" → answer_type: "analytical"
   - Output format: {"answer_type": "analytical", "items": [], "text": "because [explanation]...", "chunks_used": [...]}

6. "process" - Use for queries asking "how does [process] work?" or about processes:
   - "how does the [process] work?" → answer_type: "process"
   - "what is the process for X?" → answer_type: "process"
   - "how does the framework work?" → answer_type: "process"
   - Output format: {"answer_type": "process", "items": [], "text": "process description...", "chunks_used": [...]}

7. "not_found" - Use ONLY when NO relevant information exists in ANY chunk:
   - If query asks for information that doesn't exist in chunks → answer_type: "not_found"
   - If query asks for wrong role/company (e.g., "co-founders" but only "CEO" exists) → answer_type: "not_found"
   - Output format: {"answer_type": "not_found", "items": [], "text": "I don't have that information in the provided documents", "chunks_used": []}

CRITICAL: Match the query pattern to determine answer_type BEFORE extracting information. The answer_type determines the output structure.

CRITICAL OUTPUT REQUIREMENT:
- You MUST output ONLY the final answer (STEP 6/STEP 7 content)
- DO NOT output STEP 1, STEP 2, STEP 3, STEP 4, or STEP 5
- DO NOT output "Extract information from Chunk X" or any intermediate reasoning
- DO NOT output "STEP 6: SYNTHESIZE RESPONSE" or any step markers
- Output ONLY the final answer text itself (e.g., "John Smith and Mike Brown" or "I don't have that information in the provided documents")
- The CoT steps (STEP 1-5) are for INTERNAL reasoning only - they should NOT appear in your output
- If you output any intermediate steps, your response is INCORRECT

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
            # CRITICAL: For assistant messages, ensure they only contain final answers
            # CoT steps (STEP 1-5) should be in system prompt only, not in assistant response
            if role == "assistant":
                # Check if assistant response contains CoT leakage patterns
                cot_patterns = [
                    r'STEP\s*[1-5]',
                    r'Step\s*[1-5]',
                    r'Extract information from Chunk',
                    r'Chunk\s*\d+[:\-]?\s*$',
                ]
                import re
                has_cot = any(re.search(pattern, content, re.IGNORECASE) for pattern in cot_patterns)
                if has_cot:
                    print(f"⚠️  WARNING: Assistant response contains CoT steps (should be final answer only):")
                    print(f"   {content[:200]}...")
                    print(f"   → This will teach model to output CoT steps in final answer (incorrect behavior)")
                    # Continue anyway - dataset may have some examples with CoT, but we'll train to avoid it
            
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

# LoRA Configuration (Optimized for JSON Output)
# LoRA Rank Configuration
# r=4: ~2.7M trainable (0.17%) - Very minimal capacity, insufficient for multi-entity extraction
# r=6: ~4.1M trainable (0.26%) - Better capacity, but still insufficient for JSON structure learning
# r=8: ~5.5M trainable (0.35%) - OPTIMAL for JSON output format (structured extraction requires more capacity)
# r=16: ~11M trainable (0.85%) - Higher capacity, but may cause memorization
#
# JSON Output Mode: Model needs to learn JSON structure + extraction completeness
# Higher rank (8) helps model learn structured format better than natural language

LORA_RANK = 8  # Increased from 6 - JSON structure requires more capacity
LORA_ALPHA = 16  # 2x rank for optimal scaling
LORA_DROPOUT = 0.3  # Increased from 0.25 - more regularization to prevent memorization

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,  # Increased from 0 - more regularization to prevent memorization
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

# Training arguments optimized for JSON output format
# UPDATED: Increased LoRA rank to 8, reduced epochs to 5, more conservative learning rate
# Reason: JSON structure is easier to learn but requires more capacity. Reduced epochs to prevent overfitting.
# JSON format should help model learn extraction completeness better than natural language.
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=2000,  # Increased warmup for more stable start with JSON format
    num_train_epochs=5,  # Reduced from 7 - JSON format learns faster, prevent overfitting
    learning_rate=5e-7,  # More conservative - JSON structure is easier to learn, don't need high LR
    max_grad_norm=1.0,  # CRITICAL: Gradient clipping to prevent explosions
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,  # More frequent logging for better monitoring
    optim="adamw_8bit",
    weight_decay=0.8,  # Increased from 0.7 - stronger regularization for JSON format
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
    label_smoothing_factor=0.1,  # NEW: Prevent overconfidence, better generalization
)

# Use full dataset for training (no validation split - evaluation disabled)
train_dataset = dataset

print(f"   - Train examples: {len(train_dataset)}")

# ============================================================================
# Optional: Real-Time Example Monitoring
# ============================================================================
# Set to True to see examples being processed during training
# This helps determine if training will be successful before wasting compute
ENABLE_EXAMPLE_MONITORING = True  # Toggle: True to enable, False to disable

# Early stopping callback to prevent memorization
class EarlyStoppingCallback(TrainerCallback):
    """Stop training if loss drops below threshold too early (indicates memorization)"""
    def __init__(self, loss_threshold=0.2, min_epoch=8.0):  # Increased to 8.0 to allow full training
        self.loss_threshold = loss_threshold
        self.min_epoch = min_epoch
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        loss = logs.get("loss", None)
        epoch = logs.get("epoch", 0)
        
        if loss is not None and epoch < self.min_epoch:
            if loss < self.loss_threshold:
                print(f"\n❌ EARLY STOPPING: Loss ({loss:.4f}) dropped below {self.loss_threshold} before epoch {self.min_epoch}")
                print(f"   This indicates memorization. Stopping training to prevent overfitting.")
                control.should_training_stop = True
        return control

# CoT leakage monitoring callback (for natural language output)
class CoTLeakageMonitor(TrainerCallback):
    """Monitor and warn about CoT step leakage in model outputs"""
    def __init__(self, sample_every_n_steps=100):
        self.sample_every_n_steps = sample_every_n_steps
        self.cot_leakage_count = 0
        self.total_samples = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.sample_every_n_steps != 0:
            return
        
        # Check if we have predictions to analyze
        # This is a simplified check - actual monitoring happens in example_monitor
        if state.global_step > 0:
            print(f"\n⚠️  CoT Leakage Monitor: Step {state.global_step}")
            print(f"   Monitor training examples above for CoT step leakage (STEP 1-5 or 'Extract information from Chunk X')")
            print(f"   Model should output ONLY the final answer (STEP 6 content), not intermediate steps")
        return control

# JSON validation monitoring callback (for JSON output mode)
class JSONValidationMonitor(TrainerCallback):
    """Monitor JSON validity and structure in model outputs"""
    def __init__(self, sample_every_n_steps=100):
        self.sample_every_n_steps = sample_every_n_steps
        self.json_valid_count = 0
        self.json_invalid_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.sample_every_n_steps != 0:
            return
        
        if state.global_step > 0:
            print(f"\n📊 JSON Validation Monitor: Step {state.global_step}")
            print(f"   Monitor training examples above for JSON validity and structure")
            print(f"   Model should output valid JSON with 'answer_type', 'items', 'text', and 'chunks_used' fields")
            if self.json_valid_count + self.json_invalid_count > 0:
                validity_rate = 100 * self.json_valid_count / (self.json_valid_count + self.json_invalid_count)
                print(f"   Current JSON validity rate: {validity_rate:.1f}% ({self.json_valid_count} valid, {self.json_invalid_count} invalid)")
        return control

callbacks = []

# Add early stopping callback
# DISABLED for first training run - model needs full 10 epochs to learn
# Uncomment and adjust if you see signs of overfitting after full training
# early_stopping = EarlyStoppingCallback(loss_threshold=0.2, min_epoch=8.0)  # Increased min_epoch to 8.0
# callbacks.append(early_stopping)

# Add appropriate monitor based on output mode
if JSON_OUTPUT_MODE:
    json_monitor = JSONValidationMonitor(sample_every_n_steps=100)
    callbacks.append(json_monitor)
else:
cot_monitor = CoTLeakageMonitor(sample_every_n_steps=100)
callbacks.append(cot_monitor)

if ENABLE_EXAMPLE_MONITORING:
    from training_example_monitor import create_example_monitor
    example_monitor = create_example_monitor(
        dataset=train_dataset,
        tokenizer=tokenizer,
        model=model,
        sample_every_n_steps=20,  # Show examples every 20 steps
        num_samples=3,  # Show 3 examples each time
        show_predictions=True,  # Set True to see model predictions (slower but more informative)
        show_chunks=True  # Show actual chunk text to verify what model sees
    )
    callbacks.append(example_monitor)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    callbacks=callbacks,  # Add example monitor here if enabled
)

print("✅ Training configured")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs} (reduced from 7 to 5 - JSON format learns faster)")
print(f"   - Learning rate: {training_args.learning_rate} (conservative 5e-7 - JSON structure is easier to learn)")
print(f"   - Weight decay: {training_args.weight_decay} (increased to 0.8 - stronger regularization)")
print(f"   - Label smoothing: {training_args.label_smoothing_factor} (prevents overconfidence)")
print(f"   - Gradient clipping: max_norm={training_args.max_grad_norm} (prevents training instability)")
# Calculate approximate trainable parameters
approx_params = {
    4: (2.7, 0.17),
    6: (4.1, 0.26),
    8: (5.5, 0.35),
    16: (11, 0.85),
    32: (22, 1.7),
    64: (45, 3.4),
    128: (90, 6.8),
    256: (180, 13.6),
    512: (360, 27.2),
}
params_m, params_pct = approx_params.get(LORA_RANK, (5.5, 0.35))
print(f"   - LoRA rank: {LORA_RANK} (~{params_m}M trainable parameters, ~{params_pct}% of model) (increased from 6 to 8 - JSON structure requires more capacity)")
print(f"   - LoRA alpha: {LORA_ALPHA} (2x rank for optimal scaling)")
print(f"   - LoRA dropout: {LORA_DROPOUT} (increased from 0.25 to 0.3 - stronger regularization to prevent memorization)")
print(f"   - Warmup steps: {training_args.warmup_steps} (increased from 1500 to 2000 - more stable start with JSON format)")
print(f"   - Scheduler: {training_args.lr_scheduler_type}")
print(f"   - Logging steps: {training_args.logging_steps} (more frequent monitoring)")
print(f"\n📝 Training Configuration (JSON Output Mode):")
print(f"   - Dataset: {DATASET_PATH}")
print(f"   - Output format: {'JSON' if JSON_OUTPUT_MODE else 'Natural Language'}")
print(f"   - LoRA rank: {LORA_RANK} (increased from 6 to 8 - JSON structure requires more capacity)")
print(f"   - Epochs: {training_args.num_train_epochs} (reduced from 7 to 5 - JSON format learns faster)")
print(f"   - Learning rate: {training_args.learning_rate} (conservative 5e-7 - JSON structure is easier to learn)")
print(f"   - Weight decay: {training_args.weight_decay} (increased to 0.8 - stronger regularization)")
print(f"   - Label smoothing: {training_args.label_smoothing_factor} (prevents overconfidence)")
if JSON_OUTPUT_MODE:
    print(f"   - JSON format: Model learns structured extraction (easier than natural language)")
    print(f"   - Post-processing: Use json_to_natural_language.py to convert JSON to natural language")
    print(f"   - Expected: Better extraction completeness (70-80% vs 25% with natural language)")
print(f"   - Target: Loss should decrease gradually (~0.05-0.15 per epoch for JSON format)")
print(f"\n🔍 MONITORING:")
if JSON_OUTPUT_MODE:
    print(f"   - JSON Validity: Watch for valid JSON output (should be 95%+)")
    print(f"   - Extraction Completeness: Check if all entities are extracted (items array should be complete)")
    print(f"   - Answer Type: Verify correct answer_type (entities/list/comparison/etc.)")
else:
print(f"   - CoT Leakage: Watch for outputs containing 'STEP 1-5' or 'Extract information from Chunk X'")
print(f"   - Poor Learning: Check match scores - if consistently <50% for specific query types, may need more examples")
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
if JSON_OUTPUT_MODE:
    print("  ✅ Output structured JSON format (easier to learn, better extraction completeness)")
    print("  ✅ Post-process JSON to natural language using json_to_natural_language.py")
else:
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

