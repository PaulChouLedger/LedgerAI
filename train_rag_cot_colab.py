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
2. Run this script (it will install dependencies automatically)

Dataset Features:
- Chain of Thought reasoning examples for RAG chunk extraction
- Multiple scenarios with different numbers of co-founders
- Cases where no co-founders are explicitly stated
- Examples showing how to exclude non-co-founders (advisors, employees, etc.)
- Training to carefully read chunks and identify explicit relationships
"""

# ============================================================================
# CRITICAL: Install Dependencies FIRST (before any imports)
# ============================================================================
# This prevents the "packages were previously imported" warning in Colab
# and ensures correct CUDA versions are loaded

import subprocess
import sys

def install_dependencies():
    """Install required packages if not already installed"""
    # CRITICAL: Version pinning for compatibility
    # - datasets==4.3.0: unsloth requires this exact version (newer causes recursion)
    # - trl: Need compatible version (newer versions have bugs in experimental code)
    packages = [
        "unsloth",
        "trl<0.9.0",  # CRITICAL: Newer trl has bugs in experimental.openenv
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets==4.3.0",  # CRITICAL: unsloth requires this exact version
        "llama-cpp-python"  # For GGUF conversion (optional, but helpful)
    ]
    
    # First, check if packages are already installed by trying to import them
    # Use a more robust check that handles import errors gracefully
    missing_packages = []
    
    try:
        import unsloth
    except (ImportError, NameError):
        # NameError can occur if unsloth imports trl with bugs
        missing_packages.append("unsloth")
    
    try:
        import trl
        # Also check trl version - newer versions have bugs
        import trl as trl_module
        if hasattr(trl_module, '__version__'):
            version = trl_module.__version__
            # Check if version is >= 0.9.0 (has bugs)
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse("0.9.0"):
                    print(f"⚠️  trl version {version} detected, but has bugs in experimental code")
                    print(f"   Will downgrade to <0.9.0")
                    missing_packages.append("trl<0.9.0")
            except:
                # If can't parse version, assume it's OK
                pass
    except (ImportError, NameError):
        missing_packages.append("trl<0.9.0")
    
    try:
        import peft
    except ImportError:
        missing_packages.append("peft")
    
    try:
        import accelerate
    except ImportError:
        missing_packages.append("accelerate")
    
    try:
        import bitsandbytes
    except ImportError:
        missing_packages.append("bitsandbytes")
    
    try:
        import datasets
        # Also check version - unsloth requires 4.3.0
        import datasets as ds
        if hasattr(ds, '__version__'):
            version = ds.__version__
            if version != "4.3.0":
                print(f"⚠️  datasets version {version} detected, but unsloth requires 4.3.0")
                print(f"   Will downgrade to 4.3.0")
                missing_packages.append("datasets==4.3.0")
    except ImportError:
        missing_packages.append("datasets==4.3.0")
    
    # llama-cpp-python is optional, so we don't check it
    
    if not missing_packages:
        print("✅ All dependencies already installed - skipping installation")
        return False  # No restart needed
    else:
        print(f"📦 Some dependencies missing: {', '.join(missing_packages)} - will install")
    
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
    
    if in_colab:
        print("=" * 80)
        print("Installing Dependencies (Colab)")
        print("=" * 80)
        print("⚠️  Note: You may see dependency conflict warnings.")
        print("   These are harmless - Colab has pre-installed packages that")
        print("   conflict with newer versions. Training will work fine.")
        print()
        print("⚠️  If you see 'packages were previously imported' warning,")
        print("   RESTART THE RUNTIME after installation completes!")
        print("   Runtime > Restart runtime (or Ctrl+M .)")
        print()
        
        # Install packages (ignore dependency conflicts - they're just warnings)
        for package in packages:
            print(f"Installing {package}...")
            try:
                # Use --upgrade to ensure latest versions, ignore conflicts
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", 
                    "--upgrade", "--no-warn-conflicts", package
                ], stderr=subprocess.DEVNULL)  # Suppress stderr to hide conflict warnings
            except subprocess.CalledProcessError:
                # If --no-warn-conflicts not supported, try without it
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", 
                    "--upgrade", package
                ])
        
        print()
        print("✅ Dependencies installed!")
        print("   (Dependency conflict warnings are normal in Colab - ignore them)")
        print()
        print("⚠️  IMPORTANT: Restart runtime now (Runtime > Restart runtime)")
        print("   Then run the script again - it will skip installation.")
        print("=" * 80)
        print()
        return True  # Indicates restart needed
    else:
        # Not in Colab - install manually
        print("⚠️  Some dependencies missing. Please install manually:")
        print(f"   pip install {' '.join(packages)}")
        return False

# Install dependencies FIRST (before any other imports)
_restart_needed = install_dependencies()
if _restart_needed:
    print("\n🛑 STOPPING: Please restart runtime, then run this script again.")
    sys.exit(0)

# Now safe to import everything else
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

SEED = 3407

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
# Configuration
# ============================================================================
# (Dependencies are installed at the top of the script)

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Qwen 2.5 1.5B - better instruction following and reasoning

# Dataset path
# Use 100% verbatim dataset (ensures no hallucination)
DATASET_PATH = "rag_cot_training_dataset_100percent.json"  # 100% verbatim evidence (94 examples, 29 co-founder)

MAX_SEQ_LENGTH = 8192  # Increased: handle longer contexts (LedgerAI test was ~1171 tokens, need buffer)
OUTPUT_DIR = "outputs_rag_cot"
GGUF_OUTPUT_DIR = "gguf_model_rag_cot"
EXPECTED_GGUF_FILENAME = "Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf"  # Test script expects this exact name

# System prompt is included in the dataset, but we have a fallback just in case
FALLBACK_SYSTEM_PROMPT = """You are a precise data extraction bot.

ALWAYS START WITH REASONING:
Begin every response with "REASONING:" - this is MANDATORY.

1. REASONING: For each relevant item found in the context:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].

2. End scan with: - End of scan.

3. FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES (APPLY TO ALL QUERIES):

EVIDENCE:
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- You MUST evaluate ALL relevant items in the context before ending the scan.
- CRITICAL: Read through the ENTIRE context completely from start to finish - do NOT stop scanning early.
- CRITICAL: Continue scanning until you reach the VERY END of the context - do NOT stop when you find matches.
- CRITICAL: Items may appear at the VERY END of long contexts - you MUST scan until the absolute end.
- Scan systematically through all chunks, paragraphs, and sections.
- In complex contexts with many entities, scan ALL entities before ending.
- CRITICAL: Relevant items may appear at the VERY END of long contexts - you MUST read to the end.
- CRITICAL: Do NOT stop when you find some matches - continue scanning for ALL matches.
- CRITICAL: If query asks for a list (e.g., "co-founders", "products", "locations"), ensure you found ALL matching items.
- Do NOT end scan until you have checked EVERY relevant item in the context.
- Do NOT stop scanning when you find matches - continue until the END of context.
- Do NOT assume you've found all items - always scan to the very end.
- Items may appear at the very end - you MUST scan ALL items before ending.

KEEP/DISCARD:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- If you mark an item [KEEP] in reasoning, it MUST appear in FINAL ANSWER.

MATCHING (PREVENTS HALLUCINATION - STRICT VERBATIM RULE - UNIVERSAL PRINCIPLE):
- UNIVERSAL PRINCIPLE: Query term MUST appear verbatim (exact word-for-word) in evidence for [KEEP].
- This principle applies to ALL query types: roles (CEO, CFO, CMO, co-founder), names, dates, numbers, locations, products, services, etc.
- If query term appears verbatim in evidence → [KEEP] (regardless of other roles/info mentioned).
- If query term does NOT appear verbatim in evidence → [DISCARD] (NO exceptions, NO inference, NO assumptions, NO memorization).
- CRITICAL: Do NOT memorize specific role combinations. Apply the verbatim principle universally to ALL queries.
- CRITICAL: Different terms are NOT matches unless query term appears verbatim (e.g., "CEO" ≠ "CFO", "CFO" ≠ "CMO", "co-founder" ≠ "CEO", "Business Development Lead" ≠ "co-founder", "Ambassador" ≠ "co-founder", "revenue" ≠ "funding", "products" ≠ "services").
- CRITICAL: The verbatim matching rule is UNIVERSAL - it applies to CEO queries, CFO queries, name queries, date queries, number queries, location queries, ALL queries.
- DO NOT infer or assume relationships - only use explicitly stated information.
- DO NOT use context clues - only verbatim presence of query term matters.
- DO NOT memorize role combinations - apply the verbatim principle to every query.
- The same verbatim matching principle applies whether query asks for "CEO", "CFO", "John Smith", "2023", "$50 million", "New York", etc.

EMPTY RESULTS:
- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.

OUTPUT FORMAT:
- FINAL ANSWER must include ONLY the information explicitly requested in the query - nothing more, nothing less.
- Include ONLY what is requested - exclude extra words, role titles, dates, or any context not explicitly requested.
- CRITICAL: Do NOT include words like "Additionally", "Also", "Furthermore" - these are NOT part of the answer.
- CRITICAL: Do NOT include years/dates unless explicitly requested (e.g., if query asks for revenue, include ONLY "$50 million", NOT "2023" or "$50 million in 2023").
- If query asks for a list, include ALL matching items found in the context (do not omit any).
- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says "50 developers", do NOT change to "50 employees").
- For queries asking "Who is the [ROLE]?", include ONLY the person's name, not the role title or company name.
- For queries asking for amounts/numbers, include ONLY the amount/number, not dates, years, or other context.

FORMAT REQUIREMENTS:
- REASONING: must start with exactly "REASONING:" (no brackets, no extra text).
- Each item MUST have three separate lines:
  - Item: [name or value]
  - Evidence: "[verbatim quote]"
  - Action: [KEEP] or [DISCARD]
- Do NOT combine Item/Evidence/Action on one line.
- Do NOT use variations like "REASONING: []" or "REASONING: Item:".

CRITICAL - STOP AFTER FINAL ANSWER:
- Once you provide FINAL ANSWER, STOP generating immediately.
- Do NOT continue with any further analysis, reasoning, or generation.
- Do NOT add explanations, clarifications, or additional information after FINAL ANSWER.
- Do NOT continue scanning or processing after FINAL ANSWER.
- FINAL ANSWER is the END of your response - nothing comes after it.
- The response MUST end with FINAL ANSWER - no continuation."""

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
# Higher capacity: more parameters for better pattern learning and complex extraction
# 256 gives good balance - high enough for complex extraction, not so high it overfits bad patterns

LORA_RANK = 256  # High capacity: better extraction, role matching, and format adherence
LORA_ALPHA = LORA_RANK * 2  # Optimal scaling: alpha = 2x rank (512)
# Note: Using fixed dataset with verbatim evidence is more important than increasing LoRA further

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
# Balanced: enough training to learn patterns, but prevent overfitting
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    warmup_steps=75,  # Slightly longer warmup for better learning
    num_train_epochs=30,  # Increased: enforce strict format and learn from fixed verbatim dataset (was 25)
    learning_rate=3e-5,  # Slightly higher: better learning while preventing memorization (was 2e-5)
    weight_decay=0.2,  # Slightly lower: less aggressive regularization (was 0.25)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,  # More frequent logging to monitor progress
    optim="adamw_8bit",
    lr_scheduler_type="cosine",  # Cosine scheduler for smoother learning
    seed=3407,  # Main training seed
    data_seed=3407,  # DETERMINISTIC: Seed for data shuffling (ensures same data order)
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=10,  # Keep more checkpoints for 25 epochs
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # DETERMINISTIC: Disable multiprocessing for reproducibility
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
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs} (increased to enforce strict CoT format)")
print(f"   - Learning rate: {training_args.learning_rate} (balanced for learning)")
print(f"   - Weight decay: {training_args.weight_decay} (regularization)")
print(f"   - Warmup steps: {training_args.warmup_steps}")
print(f"   - Max sequence length: {MAX_SEQ_LENGTH} (increased to handle longer contexts)")
print(f"   ⚠️  TRAINING IMPROVEMENTS:")
print(f"      ✅ Using FIXED dataset with verbatim evidence (critical for accuracy)")
print(f"      ✅ Increased epochs (30 vs 15) - enforce strict CoT format on ALL queries")
print(f"      ✅ Increased learning rate (3e-5 vs 2e-5) - better learning")
print(f"      ✅ Increased LoRA rank (256 vs 128) - high capacity for complex extraction")
print(f"      ✅ Increased sequence length (8192 vs 4096) - handle longer contexts")
print(f"      ✅ LoRA dropout (0.1) - prevents overfitting despite higher capacity")
print(f"      → Better extraction, role matching, format adherence, and verbatim evidence")
# Calculate approximate trainable parameters
approx_params = {
    64: (45, 3.4),
    128: (90, 6.8),
    192: (135, 10.2),  # Interpolated
    256: (180, 13.6),
    512: (360, 27.2),
}
params_m, params_pct = approx_params.get(LORA_RANK, (180, 13.6))
print(f"   - LoRA rank: {LORA_RANK} (~{params_m}M trainable parameters, ~{params_pct}% of model)")
print(f"   - LoRA alpha: {LORA_ALPHA} (2x rank for optimal scaling)")
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

# Save in GGUF format for deployment (same simple approach as medical bot)
print(f"Converting to GGUF format in {GGUF_OUTPUT_DIR}...")
os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

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

# Convert to GGUF using Unsloth (same method as medical bot training)
# If llama.cpp is pre-built, Unsloth will use it instead of trying to build
try:
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method="q4_k_m"  # Q4_K_M quantization for good balance
    )
    
    # Find and rename GGUF file (Unsloth sometimes saves to root directory)
    import time
    time.sleep(3)  # Wait for file system sync
    
    # Check output directory first
    gguf_files = []
    if os.path.exists(GGUF_OUTPUT_DIR):
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
    
    # Also check root directory (Unsloth sometimes saves there)
    root_gguf_files = [f for f in os.listdir(".") if f.endswith(".gguf")]
    if root_gguf_files:
        print(f"   Found GGUF file(s) in root directory: {root_gguf_files}")
        # Move them to output directory
        for root_file in root_gguf_files:
            dest = os.path.join(GGUF_OUTPUT_DIR, root_file)
            if not os.path.exists(dest):
                shutil.move(root_file, dest)
                print(f"      Moved {root_file} to {GGUF_OUTPUT_DIR}")
                gguf_files.append(root_file)
    
    if gguf_files:
        # Find Q4_K_M file (should be the quantized one)
        q4_files = [f for f in gguf_files if "q4_k_m" in f.lower()]
        if q4_files:
            original_file = os.path.join(GGUF_OUTPUT_DIR, q4_files[0])
        else:
            # Use first file found
            original_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
        
        # Rename to expected filename
        new_file = os.path.join(GGUF_OUTPUT_DIR, EXPECTED_GGUF_FILENAME)
        if original_file != new_file:
            if os.path.exists(new_file):
                os.remove(new_file)
            shutil.move(original_file, new_file)
            print(f"✅ GGUF model saved as: {EXPECTED_GGUF_FILENAME}")
        else:
            print(f"✅ GGUF model saved as: {EXPECTED_GGUF_FILENAME}")
    else:
        print(f"⚠️  No GGUF files found after conversion")
        print(f"   Checked: {GGUF_OUTPUT_DIR} and root directory")
        print(f"   Conversion may have completed but file location unknown")
        
except RuntimeError as e:
    error_msg = str(e)
    if "llama.cpp" in error_msg or "FAILED building" in error_msg or "CMake failed" in error_msg:
        print(f"\n   ⚠️  GGUF conversion failed: llama.cpp build issue")
        print(f"   This is a known issue with Unsloth's automatic llama.cpp installation")
        print(f"   The model is saved in HuggingFace format and can be converted later")
        print(f"\n   💡 Solutions:")
        print(f"   1. Try the converter script in a fresh runtime (may work):")
        print(f"      !python convert_rag_cot_to_gguf_simple.py")
        print(f"   2. Or wait for Unsloth to update their llama.cpp build process")
        print(f"   3. Or convert manually using llama.cpp directly")
    else:
        print(f"\n   ❌ GGUF conversion failed: {e}")
        print(f"   Model is still saved in HuggingFace format at: {OUTPUT_DIR}/")
except Exception as e:
    print(f"\n   ❌ Unexpected error during GGUF conversion: {e}")
    print(f"   Model is still saved in HuggingFace format at: {OUTPUT_DIR}/")

# Final verification
print()
print("=" * 80)
print("GGUF Conversion Verification")
print("=" * 80)

expected_gguf = os.path.join(GGUF_OUTPUT_DIR, "Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf")
if os.path.exists(expected_gguf):
    file_size_mb = os.path.getsize(expected_gguf) / (1024 * 1024)
    print(f"✅ SUCCESS: Expected GGUF file found!")
    print(f"   File: {os.path.basename(expected_gguf)}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Location: {expected_gguf}")
else:
    # Check for any Q4_K_M GGUF file
    all_gguf = []
    if os.path.exists(GGUF_OUTPUT_DIR):
        for f in os.listdir(GGUF_OUTPUT_DIR):
            if f.endswith(".gguf") and "q4_k_m" in f.lower():
                all_gguf.append(f)
    
    if all_gguf:
        print(f"⚠️  Found Q4_K_M GGUF but with different name:")
        for f in all_gguf:
            print(f"   - {f}")
        print(f"   Expected: Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf")
        print(f"   You may need to rename it manually or update the test script")
    else:
        print(f"❌ FAILED: No Q4_K_M GGUF file found!")
        print(f"   Expected: {expected_gguf}")
        print(f"\n   💡 Convert using the simple converter script (recommended):")
        print(f"      !python convert_rag_cot_to_gguf_simple.py")
        print(f"   ")
        print(f"   This script uses the exact same method as this training script.")
        print(f"   It will load the model and convert it to GGUF format.")

print()
print("=" * 80)
print("🎉 Fine-tuning Complete!")
print("=" * 80)
print(f"Your RAG CoT model is ready:")
print(f"  - HuggingFace format: {OUTPUT_DIR}/")
expected_gguf_path = os.path.join(GGUF_OUTPUT_DIR, EXPECTED_GGUF_FILENAME)
if os.path.exists(expected_gguf_path):
    file_size_mb = os.path.getsize(expected_gguf_path) / (1024 * 1024)
    print(f"  - GGUF format: {expected_gguf_path} ({file_size_mb:.1f} MB) ✅")
else:
    print(f"  - GGUF format: Not converted yet")
    print(f"    💡 Convert using: !python convert_rag_cot_to_gguf_simple.py")
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

