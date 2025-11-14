#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Clinical Reasoning in Fine-Tuned Medical LLM
Verifies that the model uses clinical reasoning patterns learned from training

To use in Colab:
1. Upload this script and your fine-tuned model (from outputs/ or gguf_model/)
2. Run: !pip install unsloth transformers accelerate
3. Run this script

The script will:
- Load your fine-tuned model
- Run test conversations
- Extract and display clinical reasoning
- Check for key reasoning patterns:
  * Comparative thinking ("more concerning for X than Y")
  * Rule-in/rule-out logic
  * Probability rankings
  * Differential diagnosis updates
  * Associated symptom reasoning
"""

import json
import re
import os
import glob
from typing import List, Dict, Optional
import torch

# Try to import Unsloth (for HuggingFace format models)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available. Will try standard transformers.")

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available.")

# Try to import llama_cpp (for GGUF format models)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available. Will try HuggingFace format.")

# ============================================================================
# Configuration
# ============================================================================

# Model paths (try in order)
MODEL_PATHS = [
    "outputs/",  # Unsloth HuggingFace format
    "gguf_model/",  # GGUF format
]

# ============================================================================
# Clinical Reasoning Pattern Detection
# ============================================================================

def detect_reasoning_patterns(text: str) -> Dict[str, bool]:
    """Detect if text contains clinical reasoning patterns."""
    patterns = {
        "comparative_thinking": bool(re.search(
            r"(more|less|rather|instead).*(concerning|likely|consistent|suggestive|indicates).*(than|compared to)",
            text, re.IGNORECASE
        )),
        "rule_in": bool(re.search(
            r"(ruled in|supports|favors|consistent with|matches|indicates|suggests).*diagnosis",
            text, re.IGNORECASE
        )),
        "rule_out": bool(re.search(
            r"(ruled out|excludes|against|not consistent|doesn't support|less likely)",
            text, re.IGNORECASE
        )),
        "probability": bool(re.search(
            r"\d+%|probability|likelihood|chance",
            text, re.IGNORECASE
        )),
        "differential": bool(re.search(
            r"(differential|diagnosis|condition).*(ranked|probability|likely|most probable)",
            text, re.IGNORECASE
        )),
        "clinical_reasoning": bool(re.search(
            r"CLINICAL REASONING|clinical reasoning|Clinical reasoning|Clinical context|CLINICAL INFORMATION|ELEMENT IDENTIFICATION",
            text, re.IGNORECASE
        )),
        "associated_symptom": bool(re.search(
            r"associated symptom|associated finding|classic.*finding",
            text, re.IGNORECASE
        )),
        "progressive_narrowing": bool(re.search(
            r"(narrow|narrowing|refine|refining).*(differential|diagnosis)",
            text, re.IGNORECASE
        )),
    }
    return patterns

def extract_reasoning_sections(text: str) -> List[str]:
    """Extract reasoning sections from model output."""
    reasoning_sections = []
    
    # Look for explicit reasoning markers
    reasoning_markers = ["CLINICAL REASONING", "Clinical reasoning", "Clinical context", "CLINICAL INFORMATION", "ELEMENT IDENTIFICATION"]
    for marker in reasoning_markers:
        if marker in text:
            # Extract everything after the marker
            parts = text.split(marker)
            for part in parts[1:]:
                # Take up to next question or end (but at least 50 chars)
                if "?" in part:
                    reasoning = part.split("?")[0]
                else:
                    reasoning = part[:500]  # Limit to 500 chars if no question mark
                if len(reasoning.strip()) > 50:  # Only add if substantial
                    reasoning_sections.append(marker + reasoning.strip())
            break  # Only process first marker found
    
    # Look for reasoning-like patterns
    if "more concerning" in text.lower() or "more likely" in text.lower():
        # Extract sentences with reasoning
        sentences = re.split(r'[.!?]\s+', text)
        reasoning_sentences = [s for s in sentences if any(
            word in s.lower() for word in [
                "more concerning", "more likely", "consistent with",
                "suggests", "indicates", "supports", "favors"
            ]
        )]
        if reasoning_sentences:
            reasoning_sections.append(" ".join(reasoning_sentences))
    
    # Look for probability/differential sections
    if "%" in text or "probability" in text.lower():
        # Extract lines with percentages
        lines = text.split("\n")
        prob_lines = [l for l in lines if "%" in l or "probability" in l.lower()]
        if prob_lines:
            reasoning_sections.append("\n".join(prob_lines))
    
    return reasoning_sections

def analyze_reasoning_quality(text: str) -> Dict:
    """Analyze the quality of clinical reasoning."""
    patterns = detect_reasoning_patterns(text)
    reasoning_sections = extract_reasoning_sections(text)
    
    score = sum(patterns.values())
    max_score = len(patterns)
    
    return {
        "score": score,
        "max_score": max_score,
        "percentage": (score / max_score * 100) if max_score > 0 else 0,
        "patterns_detected": patterns,
        "reasoning_sections": reasoning_sections,
        "has_reasoning": score >= 2,  # At least 2 patterns = has reasoning
    }

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the fine-tuned model (tries multiple formats)."""
    model = None
    tokenizer = None
    model_type = None
    
    # Try Unsloth format first
    if UNSLOTH_AVAILABLE:
        try:
            if os.path.exists("outputs/"):
                print("📦 Loading Unsloth model from outputs/...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="outputs/",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                model_type = "unsloth"
                print("✅ Loaded Unsloth model")
                return model, tokenizer, model_type
        except Exception as e:
            print(f"⚠️  Could not load Unsloth model: {e}")
    
    # Try standard HuggingFace format
    if TRANSFORMERS_AVAILABLE:
        try:
            if os.path.exists("outputs/"):
                print("📦 Loading HuggingFace model from outputs/...")
                tokenizer = AutoTokenizer.from_pretrained("outputs/")
                model = AutoModelForCausalLM.from_pretrained(
                    "outputs/",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                model_type = "transformers"
                print("✅ Loaded HuggingFace model")
                return model, tokenizer, model_type
        except Exception as e:
            print(f"⚠️  Could not load HuggingFace model: {e}")
    
    # Try GGUF format
    if LLAMA_CPP_AVAILABLE:
        try:
            gguf_files = glob.glob("gguf_model/*.gguf")
            if gguf_files:
                print(f"📦 Loading GGUF model: {gguf_files[0]}...")
                model = Llama(
                    model_path=gguf_files[0],
                    n_ctx=2048,
                    verbose=False,
                )
                model_type = "gguf"
                print("✅ Loaded GGUF model")
                return model, None, model_type
        except Exception as e:
            print(f"⚠️  Could not load GGUF model: {e}")
    
    raise RuntimeError("❌ Could not load any model format. Please ensure model files exist.")

# ============================================================================
# Inference
# ============================================================================

def generate_response(model, tokenizer, messages: List[Dict], model_type: str) -> str:
    """Generate response from model."""
    if model_type == "unsloth":
        # Unsloth format
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "transformers":
        # Standard transformers
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "gguf":
        # GGUF format
        # Format messages for llama.cpp
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
        
        response = model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["User:", "System:", "\n\n"],
        )
        
        return response["choices"][0]["text"].strip()
    
    return ""

# ============================================================================
# Test Conversations
# ============================================================================

def run_test_conversation(model, tokenizer, model_type: str, test_case: Dict):
    """Run a test conversation and analyze reasoning."""
    print("\n" + "=" * 80)
    print(f"TEST CASE: {test_case['name']}")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning."
        }
    ]
    
    conversation_history = []
    reasoning_detected = []
    
    for turn_num, turn in enumerate(test_case["turns"], 1):
        # Handle both user and assistant messages (matching training format)
        if "user" in turn:
            user_msg = turn["user"]
            messages.append({"role": "user", "content": user_msg})
            conversation_history.append(f"👤 User: {user_msg}")
            
            # Generate response
            print(f"\n📊 Turn {turn_num}:")
            print(f"👤 User: {user_msg}")
            print("🤖 Generating response...")
            
            response = generate_response(model, tokenizer, messages, model_type)
        elif "assistant" in turn:
            # Pre-written assistant message (question or empathy)
            assistant_msg = turn["assistant"]
            messages.append({"role": "assistant", "content": assistant_msg})
            conversation_history.append(f"🤖 Assistant: {assistant_msg}")
            
            print(f"\n📊 Turn {turn_num}:")
            print(f"🤖 Assistant: {assistant_msg}")
            continue  # Skip reasoning analysis for questions
        else:
            continue
        
        # Analyze reasoning
        reasoning_analysis = analyze_reasoning_quality(response)
        
        # Display response
        print(f"🤖 Assistant: {response[:200]}...")
        
        # Display reasoning analysis
        if reasoning_analysis["has_reasoning"]:
            print(f"\n✅ CLINICAL REASONING DETECTED ({reasoning_analysis['percentage']:.0f}% match)")
            print(f"   Patterns found: {sum(reasoning_analysis['patterns_detected'].values())}/{reasoning_analysis['max_score']}")
            
            detected_patterns = [k for k, v in reasoning_analysis['patterns_detected'].items() if v]
            print(f"   ✓ {', '.join(detected_patterns)}")
            
            if reasoning_analysis['reasoning_sections']:
                print(f"\n   📝 Reasoning sections:")
                for i, section in enumerate(reasoning_analysis['reasoning_sections'][:2], 1):
                    print(f"      {i}. {section[:150]}...")
            
            reasoning_detected.append({
                "turn": turn_num,
                "analysis": reasoning_analysis
            })
        else:
            print(f"\n⚠️  No clinical reasoning detected in this response")
        
        # Add assistant message
        messages.append({"role": "assistant", "content": response})
        conversation_history.append(f"🤖 Assistant: {response}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    # Count only user turns (where reasoning should be generated)
    user_turns = sum(1 for turn in test_case['turns'] if "user" in turn)
    print(f"✅ Reasoning detected in {len(reasoning_detected)}/{user_turns} user turns")
    
    if reasoning_detected:
        avg_score = sum(a["analysis"]["score"] for a in reasoning_detected) / len(reasoning_detected)
        print(f"📊 Average reasoning score: {avg_score:.1f}/{reasoning_detected[0]['analysis']['max_score']}")
        print(f"📈 Average reasoning percentage: {sum(a['analysis']['percentage'] for a in reasoning_detected) / len(reasoning_detected):.0f}%")
    
    return {
        "test_name": test_case["name"],
        "reasoning_detected": len(reasoning_detected),
        "total_turns": len(test_case["turns"]),
        "reasoning_analyses": reasoning_detected,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Clinical Reasoning Test for Fine-Tuned Medical LLM")
    print("=" * 80)
    print()
    
    # Load model
    print("📦 Loading fine-tuned model...")
    try:
        model, tokenizer, model_type = load_model()
        print(f"✅ Model loaded successfully (type: {model_type})")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test cases - Updated to include questions (matching training format)
    test_cases = [
        {
            "name": "Chest Pain - Acute MI",
            "turns": [
                {"user": "I have chest pain"},
                {"assistant": "I understand you're experiencing chest pain. I'm here to help."},
                {"assistant": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"},
                {"user": "It's new, started about an hour ago"},
                {"assistant": "How old are you?"},
                {"user": "I'm 58 years old"},
                {"assistant": "What is your biological sex?"},
                {"user": "Male"},
                {"assistant": "When did the chest pain start?"},
                {"user": "It started suddenly"},
                {"assistant": "Where exactly is the chest pain located?"},
                {"user": "In the center of my chest"},
                {"assistant": "What does the chest pain feel like? For example, is it sharp, heavy, burning, or pressure?"},
                {"user": "It feels like heavy pressure"},
            ]
        },
        {
            "name": "Abdominal Pain - Appendicitis",
            "turns": [
                {"user": "I have abdominal pain"},
                {"assistant": "I understand you're experiencing abdominal pain. I'm here to help."},
                {"assistant": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"},
                {"user": "It's new, just started"},
                {"assistant": "How old are you?"},
                {"user": "I'm 25 years old"},
                {"assistant": "What is your biological sex?"},
                {"user": "Female"},
                {"assistant": "When did it start?"},
                {"user": "It started a few hours ago"},
                {"assistant": "Where exactly is the abdominal pain located?"},
                {"user": "Lower right side"},
                {"assistant": "What does the abdominal pain feel like? For example, is it sharp, heavy, burning, or pressure?"},
                {"user": "It feels sharp"},
            ]
        },
    ]
    
    # Run tests
    results = []
    for test_case in test_cases:
        result = run_test_conversation(model, tokenizer, model_type, test_case)
        results.append(result)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    total_reasoning = sum(r["reasoning_detected"] for r in results)
    total_turns = sum(r["total_turns"] for r in results)
    
    print(f"📊 Overall Reasoning Detection: {total_reasoning}/{total_turns} turns ({total_reasoning/total_turns*100:.0f}%)")
    
    if total_reasoning / total_turns >= 0.5:
        print("✅ Model is using clinical reasoning!")
    elif total_reasoning / total_turns >= 0.3:
        print("⚠️  Model shows some reasoning, but may need more training")
    else:
        print("❌ Model is not showing clinical reasoning. Consider retraining with more examples.")

if __name__ == "__main__":
    main()

