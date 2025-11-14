#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Test Script for Clinical Reasoning
Simpler version for manual testing - shows reasoning in real-time

Usage in Colab:
1. Upload fine-tuned model (outputs/ or gguf_model/)
2. Run: !pip install unsloth transformers accelerate
3. Run this script and interact with the model
"""

import re
import os
import glob

# Try imports
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

import torch

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load fine-tuned model."""
    # Try Unsloth
    if UNSLOTH_AVAILABLE and os.path.exists("outputs/"):
        try:
            print("📦 Loading Unsloth model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                "outputs/",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            return model, tokenizer, "unsloth"
        except:
            pass
    
    # Try HuggingFace
    if TRANSFORMERS_AVAILABLE and os.path.exists("outputs/"):
        try:
            print("📦 Loading HuggingFace model...")
            tokenizer = AutoTokenizer.from_pretrained("outputs/")
            model = AutoModelForCausalLM.from_pretrained(
                "outputs/",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            return model, tokenizer, "transformers"
        except:
            pass
    
    # Try GGUF
    if LLAMA_CPP_AVAILABLE:
        gguf_files = glob.glob("gguf_model/*.gguf")
        if gguf_files:
            try:
                print(f"📦 Loading GGUF model: {gguf_files[0]}...")
                model = Llama(model_path=gguf_files[0], n_ctx=2048, verbose=False)
                return model, None, "gguf"
            except:
                pass
    
    raise RuntimeError("❌ Could not load model. Ensure model files exist.")

# ============================================================================
# Reasoning Detection
# ============================================================================

def highlight_reasoning(text: str) -> str:
    """Highlight reasoning patterns in text."""
    # Patterns to highlight
    patterns = [
        (r"(more|less).*(concerning|likely|consistent|suggestive)", "🔍 COMPARATIVE"),
        (r"(ruled in|supports|favors|consistent with)", "✅ RULE IN"),
        (r"(ruled out|excludes|against)", "❌ RULE OUT"),
        (r"\d+%|probability|likelihood", "📊 PROBABILITY"),
        (r"(differential|diagnosis).*(ranked|probability)", "🎯 DIFFERENTIAL"),
        (r"CLINICAL REASONING", "🧠 CLINICAL REASONING"),
        (r"associated symptom|associated finding", "🔗 ASSOCIATED"),
    ]
    
    highlighted = text
    for pattern, label in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start, end = match.span()
            highlighted = highlighted[:start] + f"[{label}]" + highlighted[start:end] + f"[/{label}]" + highlighted[end:]
    
    return highlighted

def extract_reasoning(text: str) -> dict:
    """Extract reasoning from text."""
    reasoning = {
        "has_reasoning": False,
        "sections": [],
        "patterns": {}
    }
    
    # Check for explicit reasoning markers
    if "CLINICAL REASONING" in text or "clinical reasoning" in text.lower():
        reasoning["has_reasoning"] = True
        # Extract reasoning section
        if "CLINICAL REASONING" in text:
            parts = text.split("CLINICAL REASONING")
            if len(parts) > 1:
                reasoning["sections"].append("CLINICAL REASONING" + parts[1].split("?")[0] if "?" in parts[1] else parts[1])
    
    # Check for comparative thinking
    if re.search(r"(more|less).*(concerning|likely|consistent)", text, re.IGNORECASE):
        reasoning["patterns"]["comparative"] = True
        reasoning["has_reasoning"] = True
    
    # Check for probability
    if re.search(r"\d+%|probability", text, re.IGNORECASE):
        reasoning["patterns"]["probability"] = True
        reasoning["has_reasoning"] = True
    
    # Check for differential
    if re.search(r"differential|diagnosis.*ranked", text, re.IGNORECASE):
        reasoning["patterns"]["differential"] = True
        reasoning["has_reasoning"] = True
    
    return reasoning

# ============================================================================
# Generation
# ============================================================================

def generate(model, tokenizer, messages, model_type):
    """Generate response."""
    if model_type == "unsloth":
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    elif model_type == "transformers":
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    elif model_type == "gguf":
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        response = model(prompt, max_tokens=512, temperature=0.7, stop=["User:", "System:", "\n\n"])
        return response["choices"][0]["text"].strip()
    
    return ""

# ============================================================================
# Interactive Session
# ============================================================================

def main():
    print("=" * 80)
    print("Interactive Clinical Reasoning Test")
    print("=" * 80)
    print()
    
    # Load model
    try:
        model, tokenizer, model_type = load_model()
        print(f"✅ Model loaded ({model_type})")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Initialize conversation
    messages = [{
        "role": "system",
        "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability."
    }]
    
    print("\n" + "=" * 80)
    print("Conversation Started")
    print("=" * 80)
    print("Type 'quit' to exit, 'reset' to start over, 'show' to see history")
    print()
    
    turn_count = 0
    reasoning_count = 0
    
    while True:
        # Get user input
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "reset":
            messages = [messages[0]]  # Keep system message
            turn_count = 0
            reasoning_count = 0
            print("\n🔄 Conversation reset\n")
            continue
        elif user_input.lower() == "show":
            print("\n📜 Conversation History:")
            for i, msg in enumerate(messages[1:], 1):
                role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                print(f"{i}. {role}: {content}")
            print()
            continue
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        turn_count += 1
        
        # Generate response
        print("🤖 Thinking...")
        response = generate(model, tokenizer, messages, model_type)
        
        # Analyze reasoning
        reasoning = extract_reasoning(response)
        
        # Display response
        print(f"\n🤖 Assistant: {response}")
        
        # Display reasoning analysis
        if reasoning["has_reasoning"]:
            reasoning_count += 1
            print("\n" + "🧠 CLINICAL REASONING DETECTED!")
            if reasoning["patterns"]:
                patterns = ", ".join(reasoning["patterns"].keys())
                print(f"   Patterns: {patterns}")
            if reasoning["sections"]:
                print(f"   Reasoning sections found: {len(reasoning['sections'])}")
        else:
            print("\n⚠️  No clinical reasoning detected in this response")
        
        print(f"\n📊 Reasoning detected in {reasoning_count}/{turn_count} turns ({reasoning_count/turn_count*100:.0f}%)" if turn_count > 0 else "")
        print()
        
        # Add assistant message
        messages.append({"role": "assistant", "content": response})
    
    # Final summary
    print("\n" + "=" * 80)
    print("Session Summary")
    print("=" * 80)
    print(f"Total turns: {turn_count}")
    print(f"Reasoning detected: {reasoning_count}")
    if turn_count > 0:
        print(f"Reasoning rate: {reasoning_count/turn_count*100:.0f}%")
        if reasoning_count / turn_count >= 0.5:
            print("✅ Model is using clinical reasoning!")
        elif reasoning_count / turn_count >= 0.3:
            print("⚠️  Model shows some reasoning")
        else:
            print("❌ Model needs more training on clinical reasoning")

if __name__ == "__main__":
    main()

