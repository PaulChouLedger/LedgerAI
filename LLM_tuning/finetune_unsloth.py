#!/usr/bin/env python3
"""
Unsloth Fine-Tuning Script for Medical Bot on Jetson
====================================================

This script fine-tunes a language model using Unsloth for the medical chatbot.
It uses the medical_sft_dataset.json file created for supervised fine-tuning.

Requirements:
- Jetson device with CUDA support
- Unsloth installed from Jetson AI Lab PyPI index
- Medical SFT dataset (medical_sft_dataset.json in this directory)

Usage:
    cd LLM_tuning
    python3 finetune_unsloth.py --model_name llama-3.2-1b-instruct --output_dir ../models/finetuned
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from unsloth import FastLanguageModel
    from unsloth.train import TrainingArguments
    from unsloth.train.llama_trainer import LlamaTrainer
    from trl import SFTTrainer
    from datasets import Dataset
    import torch
except (ImportError, IndexError, AttributeError, ModuleNotFoundError) as e:
    error_type = type(e).__name__
    error_msg = str(e)
    
    if 'unsloth_zoo' in error_msg or ('No module named' in error_msg and 'unsloth_zoo' in error_msg):
        print("❌ unsloth_zoo is not properly installed or is incompatible.")
        print("\nSolution:")
        print("pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo")
        exit(1)
    elif 'IndexError' in error_type or 'list index out of range' in error_msg:
        print("⚠️  Unsloth patching encountered an error during import.")
        print("   This is often a non-critical compatibility issue with TRL.")
        print("   Attempting to continue - SFT should still work...")
        print("")
        # Try to import again - sometimes the error doesn't prevent usage
        try:
            from unsloth import FastLanguageModel
            from unsloth.train import TrainingArguments
            from trl import SFTTrainer
            from datasets import Dataset
            import torch
            print("✅ Imports succeeded on retry - continuing with fine-tuning")
        except Exception as e2:
            print(f"❌ Import failed: {e2}")
            error_msg = str(e2)
            if 'top_k_top_p_filtering' in error_msg:
                print("\nThis is a transformers version compatibility issue.")
                print("Solution: pip install 'transformers>=4.40.0,<4.46.0'")
            else:
                print("\nTroubleshooting steps:")
                print("1. pip install 'transformers>=4.40.0,<4.46.0'")
                print("2. pip install 'trl>=0.7.0,<0.8.0'")
                print("3. pip install --upgrade unsloth")
            exit(1)
    else:
        print(f"❌ Missing required packages. Please install unsloth and dependencies.")
        print(f"Error: {e}")
        print("\nInstallation instructions:")
        print("1. pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth-2025.7.9-py3-none-any.whl")
        print("2. pip install 'transformers>=4.40.0,<4.46.0'")
        print("3. pip install 'trl>=0.7.0,<0.8.0'")
        print("4. pip install datasets peft")
        exit(1)


def load_dataset(dataset_path: str, tokenizer=None) -> Dataset:
    """Load the medical SFT dataset from JSON file."""
    print(f"📂 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Unsloth expects text format
    # We'll format using the chat template if tokenizer is available
    # Otherwise, use a simple format that works with most models
    texts = []
    for conversation in data:
        messages = conversation.get("messages", [])
        
        # If tokenizer is available, use chat template
        if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
            try:
                # Format messages for chat template
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        "role": msg.get("role", ""),
                        "content": msg.get("content", "")
                    })
                
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append({"text": text})
            except Exception as e:
                print(f"⚠️  Chat template failed, using simple format: {e}")
                # Fallback to simple format
                conversation_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation_text += f"User: {content}\n\n"
                    elif role == "assistant":
                        conversation_text += f"Assistant: {content}\n\n"
                texts.append({"text": conversation_text.strip()})
        else:
            # Simple format without chat template
            conversation_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    conversation_text += f"User: {content}\n\n"
                elif role == "assistant":
                    conversation_text += f"Assistant: {content}\n\n"
            texts.append({"text": conversation_text.strip()})
    
    print(f"✅ Loaded {len(texts)} training examples")
    return Dataset.from_list(texts)


def setup_model(
    model_name: str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: bool = True,
):
    """Setup the model for fine-tuning with Unsloth optimizations."""
    print(f"🤖 Loading model: {model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # Use one if using gated models
    )
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    
    print("✅ Model loaded and configured")
    return model, tokenizer


def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    max_steps: int = -1,
    save_steps: int = 100,
    logging_steps: int = 10,
):
    """Train the model using Unsloth's optimized trainer."""
    print("🚀 Starting training...")
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Setup training arguments
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 on Jetson
            bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if supported
            logging_steps=logging_steps,
            optim="adamw_8bit",  # Memory-efficient optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=save_steps,
            save_total_limit=3,  # Keep only last 3 checkpoints
            report_to="none",  # Disable wandb/tensorboard
        ),
    )
    
    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"💾 GPU: {gpu_stats.name}, Memory: {gpu_stats.total_memory / 1024**3:.2f} GB")
    
    # Train
    trainer_stats = trainer.train()
    print(f"✅ Training completed!")
    print(f"📊 Training stats: {trainer_stats}")
    
    # Save model
    print(f"💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune medical bot with Unsloth on Jetson")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Model name or path (default: unsloth/Llama-3.2-1B-Instruct-bnb-4bit)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./medical_sft_dataset.json",
        help="Path to medical SFT dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/finetuned_medical",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device (default: 2, adjust based on GPU memory)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Warmup steps (default: 5)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 for epochs-based training, default: -1)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Validate dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset file not found: {args.dataset_path}")
        exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model first (needed for dataset formatting with chat template)
    model, tokenizer = setup_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # Use 4-bit quantization on Jetson
        use_gradient_checkpointing=True,
    )
    
    # Load dataset (with tokenizer for chat template formatting)
    dataset = load_dataset(args.dataset_path, tokenizer=tokenizer)
    
    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    
    print("\n🎉 Fine-tuning completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")
    print("\nTo use the fine-tuned model:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()

