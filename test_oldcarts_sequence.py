#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the fine-tuned model follows OLD CARTS sequence correctly.
Tests the complete flow: Empathy → Chronicity → Age → Sex → OLD CARTS

For Colab: Run this cell after training
"""

import json
import os

# Try to detect if we're in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Configuration
MODEL_PATH_GGUF = "gguf_model/model.gguf"  # Path to GGUF model
MODEL_PATH_HF = "outputs"  # Path to HuggingFace model

def install_dependencies():
    """Install required dependencies for Colab"""
    if IN_COLAB:
        print("📦 Installing dependencies for Colab...")
        try:
            import subprocess
            import sys
            
            # Install Unsloth (required for models trained with Unsloth)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth", "--quiet"])
                print("✅ unsloth installed")
            except:
                print("⚠️  unsloth installation failed")
            
            # Try installing llama-cpp-python (may fail, that's OK)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--quiet"])
                print("✅ llama-cpp-python installed")
            except:
                print("⚠️  llama-cpp-python installation failed (will use Unsloth instead)")
            
            # Install transformers (should always work)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "--quiet"])
            print("✅ transformers installed")
        except Exception as e:
            print(f"⚠️  Installation warning: {e}")

def load_model():
    """Load the fine-tuned model - supports Unsloth, GGUF, and HuggingFace"""
    # Try Unsloth first (most common after training with Unsloth)
    if os.path.exists(MODEL_PATH_HF):
        print(f"📦 Loading model from {MODEL_PATH_HF}")
        
        # Try Unsloth first (for models trained with Unsloth)
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_PATH_HF,
                max_seq_length=2048,
                dtype=None,  # Auto-detect
                load_in_4bit=False,  # Load in full precision for inference
            )
            FastLanguageModel.for_inference(model)  # Enable fast inference
            print("✅ Unsloth model loaded successfully")
            return ("unsloth", model, tokenizer)
        except ImportError:
            print("⚠️  Unsloth not available. Installing...")
            install_dependencies()
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_PATH_HF,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
                )
                FastLanguageModel.for_inference(model)
                print("✅ Unsloth model loaded successfully")
                return ("unsloth", model, tokenizer)
            except Exception as e:
                print(f"⚠️  Unsloth loading failed: {e}")
                print("   Trying standard transformers...")
        except Exception as e:
            print(f"⚠️  Unsloth loading failed: {e}")
            print("   Trying standard transformers...")
        
        # Fallback to standard transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_HF)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_HF)
            print("✅ HuggingFace model loaded successfully (standard transformers)")
            return ("hf", model, tokenizer)
        except ImportError:
            print("❌ transformers not available. Installing...")
            install_dependencies()
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_HF)
                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_HF)
                print("✅ HuggingFace model loaded successfully (standard transformers)")
                return ("hf", model, tokenizer)
            except Exception as e:
                print(f"❌ Failed to load HuggingFace model: {e}")
                return None
        except Exception as e:
            print(f"❌ Error loading HuggingFace model: {e}")
            return None
    
    # Try GGUF as fallback
    if os.path.exists(MODEL_PATH_GGUF):
        print(f"📦 Loading GGUF model from {MODEL_PATH_GGUF}")
        try:
            from llama_cpp import Llama
            model = Llama(
                model_path=MODEL_PATH_GGUF,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            print("✅ GGUF model loaded successfully")
            return ("gguf", model, None)
        except ImportError:
            print("⚠️  llama-cpp-python not available. Trying to install...")
            install_dependencies()
            try:
                from llama_cpp import Llama
                model = Llama(
                    model_path=MODEL_PATH_GGUF,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                print("✅ GGUF model loaded successfully")
                return ("gguf", model, None)
            except Exception as e:
                print(f"❌ Failed to load GGUF model: {e}")
                return None
        except Exception as e:
            print(f"❌ Error loading GGUF model: {e}")
            return None
    
    print("❌ Model not found. Please train the model first.")
    print(f"   Looked for: {MODEL_PATH_HF} or {MODEL_PATH_GGUF}")
    return None

def generate_response(model_info, messages):
    """Generate response from model"""
    model_type = model_info[0]
    
    if model_type == "unsloth":
        # Unsloth format (for models trained with Unsloth)
        model_obj = model_info[1]
        tokenizer = model_info[2]
        
        # Use tokenizer chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt")
        
        # Move to device if CUDA available
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model_obj = model_obj.cuda()
        
        outputs = model_obj.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7, 
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    elif model_type == "gguf":
        # GGUF format (llama.cpp)
        model = model_info[1]
        # Format for llama.cpp
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        
        response = model(
            formatted,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "System:", "\n\n\n"],
            echo=False
        )
        return response['choices'][0]['text'].strip()
    
    elif model_type == "hf":
        # HuggingFace format (standard transformers)
        model_obj = model_info[1]
        tokenizer = model_info[2]
        
        # Use tokenizer chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt")
        
        # Move to device if CUDA available
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model_obj = model_obj.cuda()
        
        outputs = model_obj.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7, 
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_oldcarts_sequence():
    """Test the complete OLD CARTS sequence"""
    print("=" * 80)
    print("🧪 Testing OLD CARTS Sequence")
    print("=" * 80)
    
    model_info = load_model()
    if model_info is None:
        print("\n❌ Cannot proceed without a model. Please train the model first.")
        return
    
    # System prompt (MUST match training exactly - simplified version)
    system_prompt = """You are a professional medical assistant. When a patient tells you about a symptom, follow this order:

1. Show empathy and acknowledge their concern
2. Ask if this is new or an ongoing problem
3. Ask their age
4. Ask their biological sex
5. Then ask about the symptom - one question at a time, waiting for each answer before asking the next

Ask about: when it started, where it is, how long it's been present, what it feels like, what makes it worse, what makes it better, if it spreads, if it's constant or comes and goes, and how severe it is.

Be natural and conversational. Ask only one question at a time. Do not list multiple questions. Do not mention frameworks or include instructions in your responses."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I have chest pain"}
    ]
    
    print("\n" + "=" * 80)
    print("👤 User: I have chest pain")
    print("=" * 80)
    
    # Track what we're testing
    expected_sequence = [
        "empathy",
        "chronicity",
        "age",
        "biological_sex",
        "onset",
        "location",
        "duration",
        "character",
        "aggravating",
        "alleviating",
        "radiation",
        "timing",
        "severity"
    ]
    
    # Smart response mapping - match responses to what model actually asks
    def get_appropriate_response(question_lower):
        """Return appropriate user response based on what model asks"""
        if "new issue" in question_lower or "ongoing problem" in question_lower or "prior diagnosis" in question_lower:
            return "It's new, started about an hour ago"
        elif "your age" in question_lower or "how old" in question_lower:
            return "I'm 58 years old"
        elif "biological sex" in question_lower or "male or female" in question_lower:
            return "Male"
        elif "when did" in question_lower and ("start" in question_lower or "begin" in question_lower):
            return "It started about an hour ago and came on suddenly"
        elif "where" in question_lower and ("located" in question_lower or "exactly" in question_lower):
            return "It's in the center of my chest"
        elif "how long" in question_lower or ("duration" in question_lower and "present" in question_lower):
            return "It's been constant for about an hour"
        elif "what does it feel like" in question_lower or "describe" in question_lower or ("character" in question_lower and "feel" in question_lower):
            return "It feels like pressure"
        elif "what makes it worse" in question_lower or "aggravating" in question_lower:
            return "It gets worse when I breathe deeply"
        elif "what makes it better" in question_lower or "what helps" in question_lower or "alleviating" in question_lower:
            return "Nothing really helps"
        elif "radiate" in question_lower or "spread" in question_lower or "move anywhere" in question_lower:
            return "Yes, it goes to my left arm"
        elif "constant or does it come and go" in question_lower or ("timing" in question_lower and "occur" in question_lower):
            return "It's constant"
        elif "scale of 1 to 10" in question_lower or "how severe" in question_lower or "rate" in question_lower or "how bad" in question_lower:
            return "It's about an 8 out of 10"
        elif "tried any treatment" in question_lower or "tried anything" in question_lower:
            return "No, I haven't tried anything yet"
        else:
            # Generic response if we can't match
            return "I'm not sure how to answer that"
    
    issues_found = []
    
    # Track what's been asked to detect repeats
    asked_elements = []
    expected_sequence = [
        "empathy",
        "chronicity", 
        "age",
        "biological_sex",
        "onset",
        "location",
        "duration",
        "character",
        "aggravating",
        "alleviating",
        "radiation",
        "timing",
        "severity"
    ]
    current_expected_idx = 0
    
    for turn in range(15):  # Test up to 15 turns
        print(f"\n📊 Turn {turn + 1}")
        print("-" * 80)
        
        # Generate response
        try:
            response = generate_response(model_info, messages)
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            break
        
        print(f"🟢 Model Response:")
        print(response)
        print()
        
        # Check for issues
        response_lower = response.lower()
        
        # Detect what element is being asked
        detected_element = None
        if turn == 0:
            if any(phrase in response_lower for phrase in ["understand", "here to help", "sorry to hear"]):
                detected_element = "empathy"
        elif "new issue" in response_lower or "ongoing problem" in response_lower or "prior diagnosis" in response_lower:
            detected_element = "chronicity"
        elif "your age" in response_lower or "how old" in response_lower:
            detected_element = "age"
        elif "biological sex" in response_lower or "male or female" in response_lower:
            detected_element = "biological_sex"
        elif "when did" in response_lower and ("start" in response_lower or "begin" in response_lower):
            detected_element = "onset"
        elif "where" in response_lower and ("located" in response_lower or "exactly" in response_lower):
            detected_element = "location"
        elif "how long" in response_lower or ("duration" in response_lower and "present" in response_lower):
            detected_element = "duration"
        elif "what does it feel like" in response_lower or ("character" in response_lower and "feel" in response_lower) or ("sharp" in response_lower and "pressure" in response_lower):
            detected_element = "character"
        elif "what makes it worse" in response_lower or "aggravating" in response_lower:
            detected_element = "aggravating"
        elif "what makes it better" in response_lower or "what helps" in response_lower or "alleviating" in response_lower:
            detected_element = "alleviating"
        elif "radiate" in response_lower or "spread" in response_lower or "move anywhere" in response_lower:
            detected_element = "radiation"
        elif "constant or does it come and go" in response_lower or ("timing" in response_lower and "occur" in response_lower):
            detected_element = "timing"
        elif "scale of 1 to 10" in response_lower or "how severe" in response_lower or "rate" in response_lower:
            detected_element = "severity"
        
        # Check sequence
        if detected_element:
            if detected_element in asked_elements:
                issues_found.append(f"Turn {turn + 1}: Repeats {detected_element} question (already asked)")
                print(f"❌ ISSUE: Repeats {detected_element} question!")
            else:
                asked_elements.append(detected_element)
            
            # Check if following expected sequence
            if current_expected_idx < len(expected_sequence):
                expected = expected_sequence[current_expected_idx]
                if detected_element == expected:
                    current_expected_idx += 1
                    print(f"✅ Correct sequence: {detected_element}")
                else:
                    issues_found.append(f"Turn {turn + 1}: Expected {expected}, got {detected_element}")
                    print(f"❌ ISSUE: Sequence wrong - expected {expected}, got {detected_element}")
        
        # Check for SOCRATES
        if "socrates" in response_lower:
            issues_found.append(f"Turn {turn + 1}: Mentions SOCRATES (should only use OLD CARTS)")
            print("❌ ISSUE: Mentions SOCRATES!")
        
        # Check for multiple questions
        question_count = response.count("?")
        if question_count > 1:
            issues_found.append(f"Turn {turn + 1}: Asks {question_count} questions (should ask only 1)")
            print(f"❌ ISSUE: Asks {question_count} questions (should ask only 1)")
        
        # Check for internal instructions
        if any(phrase in response_lower for phrase in [
            "wait for", "you must", "critical:", "framework", "old carts", "socrates",
            "answer all", "do not skip", "must answer"
        ]):
            issues_found.append(f"Turn {turn + 1}: Contains internal instructions/reasoning")
            print("❌ ISSUE: Contains internal instructions or framework mentions!")
        
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})
        
        # Get appropriate user response based on what model asked
        user_response = get_appropriate_response(response_lower)
        print(f"👤 User: {user_response}")
        messages.append({"role": "user", "content": user_response})
        
        # Stop if we've asked all expected elements
        if len(asked_elements) >= len(expected_sequence):
            print("\n✅ All expected elements have been asked!")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    
    if issues_found:
        print(f"❌ Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("✅ No issues found! Model follows OLD CARTS sequence correctly.")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_oldcarts_sequence()

