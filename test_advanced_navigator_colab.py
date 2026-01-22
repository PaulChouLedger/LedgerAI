#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Medical Navigator - Colab Test Script
===============================================

This script mirrors the logic of advanced_medical_navigator.py for testing in Colab.
It implements:
- Balanced initial seeding (all conditions at 0.5)
- LLM evaluates ALL conditions based on answers
- Condition rankings update after each answer (progressive scoring)
- Uses fine-tuned model's trained knowledge
- Clarification questions for ambiguous answers
- Context-aware OLD CARTS questions with examples from top 3 conditions
- Associated symptom questions based on top 3 conditions (after OLD CARTS)

Optimized for medical_sft_dataset_complex.json:
- Complex cross-organ system differentiation
- Progressive scoring with rolling differential diagnosis
- Clarification questions for ambiguous location/other OLD CARTS elements
- Associated symptom questions to differentiate top conditions
- Context-aware questions with examples from top 3 conditions
- Generalizable methodology for any medical condition

To use in Colab:
1. Upload this script and your fine-tuned model (trained on medical_sft_dataset_complex.json)
2. Run: !pip install unsloth transformers accelerate
3. Run this script
"""

import json
import re
import os
import glob
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path

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

# Try to import PEFT (needed to load LoRA adapters saved by training)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  peft not available. If you only have LoRA adapters in outputs/, loading may fail.")

# Try to import llama_cpp (for GGUF format models)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available. Will try HuggingFace format.")

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load fine-tuned model (tries multiple formats)."""
    outputs_dir = Path("outputs")

    # If outputs/ contains LoRA adapters (common for QLoRA/Unsloth), load base model + attach adapters.
    adapter_cfg = outputs_dir / "adapter_config.json"
    if adapter_cfg.exists():
        if not (UNSLOTH_AVAILABLE and PEFT_AVAILABLE):
            raise RuntimeError(
                "outputs/ appears to be LoRA adapters (adapter_config.json found), but required "
                f"libraries are missing: UNSLOTH_AVAILABLE={UNSLOTH_AVAILABLE}, PEFT_AVAILABLE={PEFT_AVAILABLE}. "
                "Install: !pip install unsloth peft transformers accelerate bitsandbytes"
            )

        with adapter_cfg.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path") or cfg.get("model_name_or_path")
        if not base_model_name:
            raise RuntimeError("adapter_config.json is missing base_model_name_or_path; cannot load base model.")

        print(f"📦 Loading base model for adapters: {base_model_name}")
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        print("🧩 Attaching LoRA adapters from outputs/ ...")
        model = PeftModel.from_pretrained(base_model, str(outputs_dir))
        model_type = "unsloth"
        print("✅ Loaded base model + LoRA adapters")
        return model, tokenizer, model_type

    # Try Unsloth/HF format from outputs/ (works only if outputs/ is a merged full model)
    if UNSLOTH_AVAILABLE:
        try:
            if outputs_dir.exists():
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
    
    # Try standard transformers
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
                    n_threads=4,
                    n_batch=256,
                    chat_format="chatml",  # Required for Qwen models
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

def generate_response(model, tokenizer, messages: List[Dict], model_type: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate response from model."""
    if model_type == "unsloth":
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "transformers":
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "gguf":
        # Use create_chat_completion for proper chat format handling (matches RAG CoT test script)
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\n\n"]
        )
        return response["choices"][0]["message"]["content"].strip()
    
    return ""

# ============================================================================
# Advanced Medical Navigator Logic (Simplified for Colab)
# ============================================================================

class SimpleMedicalNavigator:
    """Simplified navigator that mirrors advanced_medical_navigator.py logic."""
    
    # Medical categories (for reference - LLM will suggest conditions dynamically)
    # This is just used to show available categories to LLM, not to limit conditions
    CATEGORY_EXAMPLES = {
        'gastrointestinal': ['Acute Appendicitis', 'GERD', 'Acute Cholecystitis'],
        'cardiovascular': ['Acute Myocardial Infarction', 'Unstable Angina', 'Aortic Dissection'],
        'renal': ['Nephrolithiasis', 'Acute Kidney Injury', 'UTI'],
        'respiratory': ['Pneumonia', 'Asthma Exacerbation', 'Pulmonary Embolism'],
    }
    
    def __init__(self, model, tokenizer, model_type):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.condition_scores = {}
        self.condition_rankings = []
        self.conversation_context = {
            'pre_hpi': {},
            'hpi': {},
        }
        self.matched_categories = []
        # Pool tracking
        self.active_conditions = []  # Top 5 conditions
        self.reserve_conditions = []  # Conditions ranked 6+
        self.previous_active = set()  # Track promotions/demotions
        # Smart features
        self.skipped_elements = set()  # OLD CARTS elements to skip
        self.followups_asked = False  # Track if follow-ups have been asked
        # Dynamic pending clarifications for any OLD CARTS element (set via setattr)
    
    def needs_clarification(self, element: str, answer: str, chief_complaint: str) -> bool:
        """Use LLM to determine if any OLD CARTS answer needs clarification based on trained knowledge."""
        system_prompt = (
            "You are a medical expert. Determine if a patient's answer to an OLD CARTS question is specific enough "
            "for clinical reasoning, or if it needs clarification. Use your medical knowledge to assess if the answer "
            "provides sufficient detail for accurate diagnosis. Return ONLY 'yes' if clarification is needed, or 'no' if it's specific enough."
        )
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n\n"
            "Is this answer specific enough for clinical reasoning, or does it need clarification? "
            "Return ONLY 'yes' if clarification is needed, or 'no' if it's specific enough. No explanations."
        )
        
        try:
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            return "yes" in response.lower() if response else False
        except Exception:
            return False  # Default to not needing clarification if LLM fails
    
    def needs_location_clarification(self, location_answer: str, chief_complaint: str) -> bool:
        """Use LLM to determine if location answer needs clarification based on trained knowledge."""
        return self.needs_clarification("location", location_answer, chief_complaint)
    
    def ask_clarification(self, element: str, answer: str, chief_complaint: str, messages: List[Dict]) -> str:
        """Use LLM to generate appropriate clarification question for any OLD CARTS element based on trained knowledge."""
        element_labels = {
            "location": "anatomical location",
            "character": "quality or description of the symptom",
            "duration": "how long the symptom has been present",
            "timing": "whether it's constant or intermittent",
            "aggravating": "what makes it worse",
            "relieving": "what makes it better",
            "radiation": "where the symptom spreads",
            "severity": "how severe the symptom is",
            "onset": "when the symptom started"
        }
        
        element_label = element_labels.get(element, element)
        
        system_prompt = (
            "You are a medical professional conducting a medical history. "
            "When a patient gives an incomplete or ambiguous answer, you need to ask a clarifying question "
            "to get the specific information needed for accurate clinical reasoning. "
            "Use your medical knowledge to ask appropriate clarification questions. "
            "Be natural, conversational, and medically accurate. Ask only the clarification question, no other text."
        )
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element} ({element_label})\n"
            f"Patient's incomplete/ambiguous answer: '{answer}'\n\n"
            f"Ask a clarifying question to get the specific {element_label} information needed for accurate diagnosis."
        )
        
        try:
            clarification = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.4,
            )
            clarification = clarification.strip() if clarification else ""
            # Minimal fallback if LLM doesn't generate a good question
            if not clarification or len(clarification) < 20:
                return f"Can you be more specific about the {element_label}?"
            return clarification
        except Exception:
            # Minimal fallback on error
            return f"Can you be more specific about the {element_label}?"
    
    def ask_location_clarification(self, location_answer: str, chief_complaint: str, messages: List[Dict]) -> str:
        """Use LLM to generate appropriate location clarification question based on trained knowledge."""
        return self.ask_clarification("location", location_answer, chief_complaint, messages)
    
    def llm_chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Wrapper for LLM chat function."""
        return generate_response(self.model, self.tokenizer, messages, self.model_type, max_tokens, temperature)
    
    def is_medical_complaint(self, user_input: str) -> bool:
        """Check if user input contains a medical complaint or is just casual conversation"""
        user_lower = user_input.lower().strip()
        
        # Common greetings and casual phrases (not medical)
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                     'how are you', 'what\'s up', 'sup', 'greetings', 'hi there']
        
        # Check if it's just a greeting
        if user_lower in greetings or any(user_lower.startswith(g) for g in greetings):
            return False
        
        # Medical complaint indicators
        medical_keywords = [
            'pain', 'ache', 'hurt', 'sore', 'discomfort', 'symptom', 'problem', 'issue',
            'fever', 'nausea', 'vomit', 'dizzy', 'shortness', 'breath', 'cough', 'chest',
            'abdominal', 'headache', 'stomach', 'bleeding', 'blood', 'rash', 'swelling',
            'burning', 'pressure', 'tightness', 'numbness', 'tingling', 'weakness',
            'dizziness', 'fatigue', 'tired', 'unwell', 'sick', 'ill', 'feeling',
            'concerned about', 'worried about', 'having', 'experiencing', 'feeling'
        ]
        
        # Check if input contains medical keywords
        return any(keyword in user_lower for keyword in medical_keywords)
    
    def match_chief_complaint_to_categories(self, chief_complaint: str) -> List[str]:
        """Match chief complaint to medical categories using LLM's trained medical knowledge."""
        # Get available categories (just for reference - LLM knows all medical categories)
        available_categories = list(self.CATEGORY_EXAMPLES.keys())
        available_cats_str = ', '.join(available_categories)
        
        # Use LLM's trained medical knowledge to determine relevant categories
        # The model has been trained on medical knowledge and can recognize:
        # - Chest pain can be cardiac, pulmonary, OR gastrointestinal (GERD)
        # - Abdominal pain can be GI, renal, gynecological, etc.
        # - Multiple categories may be relevant
        system_prompt = (
            "You are a medical expert with extensive training in clinical reasoning. "
            "Based on the chief complaint, identify which medical categories are relevant using your medical knowledge.\n\n"
            f"Available categories: {available_cats_str}\n\n"
            "CRITICAL: Consider ALL possible causes, not just the most obvious one. You MUST include multiple categories when appropriate.\n\n"
            "EXAMPLES:\n"
            "- Chest pain: MUST include ['cardiovascular', 'respiratory', 'gastrointestinal'] because:\n"
            "  * Cardiovascular: MI, angina, aortic dissection, pericarditis\n"
            "  * Respiratory: PE, pneumonia, pneumothorax, pleuritis\n"
            "  * Gastrointestinal: GERD, peptic ulcer, esophagitis\n"
            "- Abdominal pain: MUST include ['gastrointestinal', 'renal', 'genitourinary'] because multiple systems can cause it\n"
            "- Shortness of breath: MUST include ['respiratory', 'cardiovascular']\n\n"
            "RULE: If the chief complaint could reasonably be caused by multiple organ systems, you MUST include ALL of them.\n"
            "Do NOT default to just one category. Think like a doctor building a differential diagnosis.\n\n"
            "Return ONLY valid JSON: {\"categories\": [\"category1\", \"category2\", \"category3\", ...]}\n"
            "No explanations, no other text. Just the JSON object."
        )
        
        user_prompt = (
            f"Chief complaint: '{chief_complaint}'\n\n"
            "Using your medical knowledge, identify which categories are relevant. "
            "Consider all possible causes, not just the most obvious one."
        )
        
        try:
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            
            if not response:
                print(f"[Category] LLM returned empty response, defaulting to all categories")
                return list(available_categories)
            
            # Parse JSON - handle multiple formats
            cleaned = response.strip()
            
            # Remove markdown code blocks
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline+1:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3].strip()
                    elif '```' in cleaned:
                        last_idx = cleaned.rfind('```')
                        cleaned = cleaned[:last_idx].strip()
            
            # Try to parse as single JSON object with "categories" key first
            start_idx = cleaned.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned)):
                    if cleaned[i] == '{':
                        brace_count += 1
                    elif cleaned[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    single_json = cleaned[start_idx:end_idx]
                    try:
                        parsed = json.loads(single_json)
                        if isinstance(parsed, dict) and 'categories' in parsed:
                            parsed_categories = parsed['categories']
                            if isinstance(parsed_categories, list):
                                valid_cats = [cat for cat in parsed_categories if cat in available_categories]
                                if valid_cats:
                                    # If likely multi-system complaint but too few categories, ask LLM to expand
                                    cc_lower = chief_complaint.lower()
                                    multi_system_complaints = ['chest pain', 'abdominal pain', 'shortness of breath', 'sob']
                                    needs_multi = any(ms in cc_lower for ms in multi_system_complaints)
                                    if needs_multi and len(valid_cats) < 2:
                                        expand_prompt = (
                                            f"Chief complaint: '{chief_complaint}'\n\n"
                                            "Your previous answer included too few categories for this complaint, which often spans multiple organ systems.\n"
                                            "Reconsider and return JSON with ALL plausible categories from the provided list."
                                        )
                                        try:
                                            retry = self.llm_chat(
                                                [
                                                    {"role": "system", "content": system_prompt},
                                                    {"role": "user", "content": expand_prompt},
                                                ],
                                                max_tokens=200,
                                                temperature=0.0,
                                            )
                                            cleaned_retry = retry.strip() if retry else ""
                                            if cleaned_retry.startswith('```'):
                                                first_newline = cleaned_retry.find('\n')
                                                if first_newline != -1:
                                                    cleaned_retry = cleaned_retry[first_newline+1:]
                                                    if cleaned_retry.endswith('```'):
                                                        cleaned_retry = cleaned_retry[:-3].strip()
                                                    elif '```' in cleaned_retry:
                                                        last_idx = cleaned_retry.rfind('```')
                                                        cleaned_retry = cleaned_retry[:last_idx].strip()
                                            start_idx = cleaned_retry.find('{')
                                            if start_idx != -1:
                                                brace_count = 0
                                                end_idx = start_idx
                                                for i in range(start_idx, len(cleaned_retry)):
                                                    if cleaned_retry[i] == '{':
                                                        brace_count += 1
                                                    elif cleaned_retry[i] == '}':
                                                        brace_count -= 1
                                                        if brace_count == 0:
                                                            end_idx = i + 1
                                                            break
                                                if end_idx > start_idx:
                                                    cleaned_retry = cleaned_retry[start_idx:end_idx]
                                            try:
                                                parsed_retry = json.loads(cleaned_retry)
                                                if isinstance(parsed_retry, dict) and 'categories' in parsed_retry:
                                                    retry_categories = parsed_retry['categories']
                                                    if isinstance(retry_categories, list):
                                                        valid_retry = [cat for cat in retry_categories if cat in available_categories]
                                                        if valid_retry:
                                                            print(f"[Category] LLM expanded categories to: {valid_retry}")
                                                            return valid_retry
                                            except json.JSONDecodeError:
                                                pass
                                        except Exception:
                                            pass
                                    print(f"[Category] LLM matched '{chief_complaint}' to categories: {valid_cats}")
                                    return valid_cats
                            # Alternative format: {"category_name": [conditions]} - extract category names
                            elif isinstance(parsed, dict):
                                # Model returned category-specific format - extract category names
                                extracted_categories = []
                                for key in parsed.keys():
                                    if key in available_categories:
                                        extracted_categories.append(key)
                                if extracted_categories:
                                    print(f"[Category] LLM matched '{chief_complaint}' to categories (extracted from condition format): {extracted_categories}")
                                    return extracted_categories
                    except json.JSONDecodeError:
                        pass  # Try parsing multiple JSON objects below
            
            # Try parsing multiple JSON objects (one per line or separated)
            # Format: {"category1": [...]}\n{"category2": [...]}
            extracted_categories = []
            lines = cleaned.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                start_idx = line.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(line)):
                        if line[i] == '{':
                            brace_count += 1
                        elif line[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        json_str = line[start_idx:end_idx]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict):
                                # Extract category name from key
                                for key in parsed.keys():
                                    if key in available_categories:
                                        if key not in extracted_categories:
                                            extracted_categories.append(key)
                        except json.JSONDecodeError:
                            continue
            
            if extracted_categories:
                print(f"[Category] LLM matched '{chief_complaint}' to categories (from multiple JSON objects): {extracted_categories}")
                # Check if we need more categories for multi-system complaints
                cc_lower = chief_complaint.lower()
                multi_system_complaints = ['chest pain', 'abdominal pain', 'shortness of breath', 'sob']
                needs_multi = any(ms in cc_lower for ms in multi_system_complaints)
                if needs_multi and len(extracted_categories) < 2:
                    # Try to expand
                    expand_prompt = (
                        f"Chief complaint: '{chief_complaint}'\n\n"
                        "Your previous answer included too few categories for this complaint, which often spans multiple organ systems.\n"
                        "Reconsider and return JSON with ALL plausible categories from the provided list."
                    )
                    try:
                        retry = self.llm_chat(
                            [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": expand_prompt},
                            ],
                            max_tokens=200,
                            temperature=0.0,
                        )
                        # Parse retry response (same logic as above)
                        if retry:
                            retry_cleaned = retry.strip()
                            if retry_cleaned.startswith('```'):
                                first_newline = retry_cleaned.find('\n')
                                if first_newline != -1:
                                    retry_cleaned = retry_cleaned[first_newline+1:]
                                    if retry_cleaned.endswith('```'):
                                        retry_cleaned = retry_cleaned[:-3].strip()
                            # Try to extract categories from retry
                            retry_lines = retry_cleaned.split('\n')
                            for retry_line in retry_lines:
                                retry_line = retry_line.strip()
                                if '{' in retry_line:
                                    retry_start = retry_line.find('{')
                                    retry_brace_count = 0
                                    retry_end = retry_start
                                    for i in range(retry_start, len(retry_line)):
                                        if retry_line[i] == '{':
                                            retry_brace_count += 1
                                        elif retry_line[i] == '}':
                                            retry_brace_count -= 1
                                            if retry_brace_count == 0:
                                                retry_end = i + 1
                                                break
                                    if retry_end > retry_start:
                                        try:
                                            retry_parsed = json.loads(retry_line[retry_start:retry_end])
                                            if isinstance(retry_parsed, dict):
                                                for key in retry_parsed.keys():
                                                    if key in available_categories and key not in extracted_categories:
                                                        extracted_categories.append(key)
                                        except json.JSONDecodeError:
                                            pass
                            if len(extracted_categories) >= 2:
                                print(f"[Category] LLM expanded categories to: {extracted_categories}")
                    except Exception:
                        pass
                return extracted_categories
            
            # Final fallback: try parsing the entire cleaned string as single JSON
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'categories' in parsed:
                    parsed_categories = parsed['categories']
                    if isinstance(parsed_categories, list):
                        valid_cats = [cat for cat in parsed_categories if cat in available_categories]
                        if valid_cats:
                            print(f"[Category] LLM matched '{chief_complaint}' to categories: {valid_cats}")
                            return valid_cats
            except json.JSONDecodeError as e:
                print(f"[Category] JSON parse error: {e}, defaulting to all categories")
            # If LLM fails, default will be handled after try/except
        except Exception as e:
            print(f"⚠️  Error in category matching: {e}, defaulting to all categories")
            return list(available_categories)
        
        # If we reach here, parsing succeeded but we didn't return valid categories.
        # Default to all categories (let scoring narrow down).
        print(f"[Category] Defaulting to all available categories: {available_categories}")
        return list(available_categories)
    
    def initialize_condition_scores(self, categories: List[str], chief_complaint: str = ""):
        """Initialize condition scores - LLM suggests relevant conditions dynamically."""
        # Use LLM's medical knowledge to suggest relevant conditions
        system_prompt = (
            "You are a medical expert with extensive training in clinical reasoning. "
            "Based on the chief complaint and medical categories, suggest relevant medical conditions "
            "that should be considered in the differential diagnosis.\n\n"
            "INTERNAL CHECKLIST (do not output):\n"
            "- Include common conditions (most likely)\n"
            "- Include serious 'can't-miss' conditions to rule out\n"
            "- Cover ALL selected categories (distribute conditions across them)\n"
            "- For chest pain: aim for 8–10+ total; others 6+ total\n"
            "- If your list is too short, expand before answering\n\n"
            "OUTPUT FORMAT (JSON only): {\"conditions\": [\"Condition 1\", \"Condition 2\", ...]}\n"
            "Return only the JSON object, no explanations."
        )
        
        user_prompt = (
            f"Chief complaint: '{chief_complaint}'\n"
            f"Medical categories: {', '.join(categories)}\n\n"
            "Suggest a COMPREHENSIVE list of relevant medical conditions for differential diagnosis. "
            "Include conditions from ALL categories provided. "
            "Suggest AT LEAST 8-10 conditions for chest pain, or 5-6 minimum for other complaints. "
            "Include both common and serious conditions that must be ruled out."
        )
        
        try:
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,  # More tokens for longer condition lists
                temperature=0.0,
            )
            
            if response:
                print(f"[Condition Init] 📥 LLM response (first 300 chars): {response[:300]}")
                # Parse JSON response - handle multiple formats
                cleaned = response.strip()
                
                # Remove markdown code blocks
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                        elif '```' in cleaned:
                            last_idx = cleaned.rfind('```')
                            cleaned = cleaned[:last_idx].strip()
                
                # Try to parse as single JSON object first
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        single_json = cleaned[start_idx:end_idx]
                        try:
                            parsed = json.loads(single_json)
                            if isinstance(parsed, dict) and 'conditions' in parsed:
                                suggested_conditions = parsed['conditions']
                                if isinstance(suggested_conditions, list) and suggested_conditions:
                                    # Process conditions list
                                    suggested_conditions = list(dict.fromkeys(suggested_conditions))  # Remove duplicates
                                    self.condition_scores = {cond: 0.5 for cond in suggested_conditions}
                                    print(f"\n📋 LLM suggested {len(self.condition_scores)} conditions at balanced baseline 50.0%")
                                    print(f"   Categories: {', '.join(categories)}")
                                    print(f"   Conditions: {', '.join(list(self.condition_scores.keys())[:5])}{'...' if len(self.condition_scores) > 5 else ''}")
                                    print(f"   LLM will narrow down based on answers")
                                    print(f"   Additional conditions can be added dynamically during scoring\n")
                                    self._update_rankings()
                                    self._update_condition_pools()
                                    return
                            # Alternative format: {"category": [conditions]} - extract all conditions
                            elif isinstance(parsed, dict):
                                # Model returned category-specific format - extract all conditions from all categories
                                all_conditions_from_categories = []
                                for key, value in parsed.items():
                                    if isinstance(value, list):
                                        all_conditions_from_categories.extend(value)
                                if all_conditions_from_categories:
                                    suggested_conditions = list(dict.fromkeys(all_conditions_from_categories))  # Remove duplicates
                                    self.condition_scores = {cond: 0.5 for cond in suggested_conditions}
                                    print(f"\n📋 LLM suggested {len(self.condition_scores)} conditions at balanced baseline 50.0% (from category format)")
                                    print(f"   Categories: {', '.join(categories)}")
                                    print(f"   Conditions: {', '.join(list(self.condition_scores.keys())[:5])}{'...' if len(self.condition_scores) > 5 else ''}")
                                    print(f"   LLM will narrow down based on answers")
                                    print(f"   Additional conditions can be added dynamically during scoring\n")
                                    self._update_rankings()
                                    self._update_condition_pools()
                                    return
                        except json.JSONDecodeError:
                            pass  # Try parsing multiple JSON objects below
                
                # Try parsing multiple JSON objects (category-specific format)
                # Format: {"category1": [conditions]}\n{"category2": [conditions]}
                all_conditions_from_multiple = []
                lines = cleaned.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    start_idx = line.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(line)):
                            if line[i] == '{':
                                brace_count += 1
                            elif line[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > start_idx:
                            json_str = line[start_idx:end_idx]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    # Extract conditions from category-specific format
                                    for key, value in parsed.items():
                                        if isinstance(value, list):
                                            all_conditions_from_multiple.extend(value)
                            except json.JSONDecodeError:
                                continue
                
                if all_conditions_from_multiple:
                    suggested_conditions = list(dict.fromkeys(all_conditions_from_multiple))  # Remove duplicates
                    # Check if we need more conditions
                    cc_lower = (chief_complaint or '').lower()
                    target_min = 10 if ('chest' in cc_lower and 'pain' in cc_lower) else 6
                    if len(suggested_conditions) < target_min:
                        expand_user = (
                            f"Chief complaint: '{chief_complaint}'\n"
                            f"Medical categories: {', '.join(categories)}\n\n"
                            f"Your previous list had only {len(suggested_conditions)} items. "
                            f"Re-evaluate and return JSON with at least {target_min} conditions covering common and can't-miss diagnoses across ALL categories."
                        )
                        try:
                            retry_resp = self.llm_chat(
                                [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": expand_user},
                                ],
                                max_tokens=600,
                                temperature=0.0,
                            )
                            if retry_resp:
                                retry_clean = retry_resp.strip()
                                if retry_clean.startswith('```'):
                                    first_newline = retry_clean.find('\n')
                                    if first_newline != -1:
                                        retry_clean = retry_clean[first_newline+1:]
                                        if retry_clean.endswith('```'):
                                            retry_clean = retry_clean[:-3].strip()
                                        elif '```' in retry_clean:
                                            last_idx = retry_clean.rfind('```')
                                            retry_clean = retry_clean[:last_idx].strip()
                                # Parse retry response - handle category format
                                retry_lines = retry_clean.split('\n')
                                retry_conditions = []
                                for retry_line in retry_lines:
                                    retry_line = retry_line.strip()
                                    if '{' in retry_line:
                                        rs = retry_line.find('{')
                                        bc = 0
                                        re = rs
                                        for i in range(rs, len(retry_line)):
                                            if retry_line[i] == '{':
                                                bc += 1
                                            elif retry_line[i] == '}':
                                                bc -= 1
                                                if bc == 0:
                                                    re = i + 1
                                                    break
                                        if re > rs:
                                            try:
                                                parsed_retry = json.loads(retry_line[rs:re])
                                                if isinstance(parsed_retry, dict):
                                                    if 'conditions' in parsed_retry and isinstance(parsed_retry['conditions'], list):
                                                        retry_conditions.extend(parsed_retry['conditions'])
                                                    else:
                                                        # Category format
                                                        for key, value in parsed_retry.items():
                                                            if isinstance(value, list):
                                                                retry_conditions.extend(value)
                                            except json.JSONDecodeError:
                                                pass
                                if retry_conditions:
                                    suggested_conditions = list(dict.fromkeys(retry_conditions))
                                    print(f"[Condition Init] 🔁 Expanded conditions to {len(suggested_conditions)} via LLM retry")
                        except Exception:
                            pass
                    # Start ALL suggested conditions at balanced baseline (0.5)
                    all_conditions = list(dict.fromkeys(suggested_conditions))
                    self.condition_scores = {cond: 0.5 for cond in all_conditions}
                    print(f"\n📋 LLM suggested {len(self.condition_scores)} conditions at balanced baseline 50.0% (from multiple category JSON)")
                    print(f"   Categories: {', '.join(categories)}")
                    print(f"   Conditions: {', '.join(list(self.condition_scores.keys())[:5])}{'...' if len(self.condition_scores) > 5 else ''}")
                    print(f"   LLM will narrow down based on answers")
                    print(f"   Additional conditions can be added dynamically during scoring\n")
                    self._update_rankings()
                    self._update_condition_pools()
                    return
                
                # Alternative: Handle multiple JSON objects on separate lines
                # Format: {"Condition 1"}\n{"Condition 2"}\n...
                suggested_conditions = []
                lines = cleaned.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Try to extract JSON object from line
                    start_idx = line.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(line)):
                            if line[i] == '{':
                                brace_count += 1
                            elif line[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > start_idx:
                            json_str = line[start_idx:end_idx]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    # Extract condition name from dict
                                    # Could be {"Condition Name"} or {"conditions": ["Condition"]} or just keys
                                    if 'conditions' in parsed and isinstance(parsed['conditions'], list):
                                        suggested_conditions.extend(parsed['conditions'])
                                    else:
                                        # Extract all string values from dict
                                        for key, value in parsed.items():
                                            if isinstance(value, str):
                                                suggested_conditions.append(value)
                                            elif isinstance(value, list):
                                                suggested_conditions.extend([v for v in value if isinstance(v, str)])
                                        # If no values, use keys as condition names
                                        if not suggested_conditions:
                                            suggested_conditions.extend([k for k in parsed.keys() if isinstance(k, str)])
                            except json.JSONDecodeError:
                                continue
                
                # If we found conditions in the alternative format
                if suggested_conditions:
                    # Remove duplicates and clean up
                    cc_lower = (chief_complaint or '').lower()
                    target_min = 10 if ('chest' in cc_lower and 'pain' in cc_lower) else 6
                    if len(suggested_conditions) < target_min:
                        expand_user = (
                            f"Chief complaint: '{chief_complaint}'\n"
                            f"Medical categories: {', '.join(categories)}\n\n"
                            f"Your previous list had only {len(suggested_conditions)} items. "
                            f"Re-evaluate and return JSON with at least {target_min} conditions covering common and can't-miss diagnoses across ALL categories."
                        )
                        try:
                            retry_resp = self.llm_chat(
                                [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": expand_user},
                                ],
                                max_tokens=600,
                                temperature=0.0,
                            )
                            if retry_resp:
                                retry_clean = retry_resp.strip()
                                if retry_clean.startswith('```'):
                                    first_newline = retry_clean.find('\n')
                                    if first_newline != -1:
                                        retry_clean = retry_clean[first_newline+1:]
                                        if retry_clean.endswith('```'):
                                            retry_clean = retry_clean[:-3].strip()
                                        elif '```' in retry_clean:
                                            last_idx = retry_clean.rfind('```')
                                            retry_clean = retry_clean[:last_idx].strip()
                                rs = retry_clean.find('{')
                                if rs != -1:
                                    bc = 0
                                    re = rs
                                    for i in range(rs, len(retry_clean)):
                                        if retry_clean[i] == '{':
                                            bc += 1
                                        elif retry_clean[i] == '}':
                                            bc -= 1
                                            if bc == 0:
                                                re = i + 1
                                                break
                                    if re > rs:
                                        retry_json = retry_clean[rs:re]
                                        try:
                                            parsed_retry = json.loads(retry_json)
                                            if isinstance(parsed_retry, dict) and 'conditions' in parsed_retry and isinstance(parsed_retry['conditions'], list) and parsed_retry['conditions']:
                                                suggested_conditions = parsed_retry['conditions']
                                                print(f"[Condition Init] 🔁 Expanded conditions to {len(suggested_conditions)} via LLM retry")
                                        except json.JSONDecodeError:
                                            pass
                        except Exception:
                            pass
                    suggested_conditions = list(dict.fromkeys(suggested_conditions))  # Preserve order, remove dupes
                    self.condition_scores = {cond: 0.5 for cond in suggested_conditions}
                    print(f"\n📋 LLM suggested {len(self.condition_scores)} conditions at balanced baseline 50.0%")
                    print(f"   Categories: {', '.join(categories)}")
                    print(f"   Conditions: {', '.join(list(self.condition_scores.keys())[:5])}{'...' if len(self.condition_scores) > 5 else ''}")
                    print(f"   LLM will narrow down based on answers")
                    print(f"   Additional conditions can be added dynamically during scoring\n")
                    self._update_rankings()
                    self._update_condition_pools()
                    return
                
                # If we still couldn't parse, show error
                print(f"[Condition Init] ⚠️ Failed to parse JSON in any format")
                print(f"[Condition Init] ⚠️ Extracted JSON attempt (first 200 chars): {cleaned[:200]}")
            else:
                print(f"[Condition Init] ⚠️ LLM returned empty response")
            
            # Fallback: Use example conditions from categories (limited, but better than nothing)
            print(f"[Condition Init] ⚠️ LLM condition suggestion failed, using category examples as fallback")
            all_conditions = []
            for category in categories:
                if category in self.CATEGORY_EXAMPLES:
                    all_conditions.extend(self.CATEGORY_EXAMPLES[category])
            
            if all_conditions:
                self.condition_scores = {cond: 0.5 for cond in all_conditions}
                print(f"\n📋 Seeded {len(self.condition_scores)} example conditions at balanced baseline 50.0%")
                print(f"   Categories: {', '.join(categories)}")
                print(f"   Note: Using limited example conditions. LLM can still reason about other conditions.\n")
            else:
                # Last resort: empty dict, let scoring add conditions dynamically
                self.condition_scores = {}
                print(f"\n📋 No conditions initialized - LLM will suggest during scoring\n")
            
            self._update_rankings()
            
        except Exception as e:
            print(f"⚠️  Error initializing conditions: {e}, using category examples")
            all_conditions = []
            for category in categories:
                if category in self.CATEGORY_EXAMPLES:
                    all_conditions.extend(self.CATEGORY_EXAMPLES[category])
            self.condition_scores = {cond: 0.5 for cond in all_conditions} if all_conditions else {}
            self._update_rankings()
            self._update_condition_pools()
    
    def _update_rankings(self):
        """Update condition rankings."""
        self.condition_rankings = sorted(
            self.condition_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def _update_condition_pools(self):
        """Update active and reserve condition pools, track promotions/demotions."""
        # Active = top 5, Reserve = rest
        new_active = self.condition_rankings[:5]
        new_reserve = self.condition_rankings[5:]
        
        new_active_names = {name for name, _ in new_active}
        previous_active_names = self.previous_active
        
        # Track promotions (moved into top 5)
        promotions = new_active_names - previous_active_names
        if promotions:
            print(f"\n[Pool] 🔼 PROMOTED to active ({len(promotions)}):")
            for name in promotions:
                score = next((score for n, score in new_active if n == name), 0.0)
                print(f"[Pool]   ↑ {name}: {score:.1%}")
        
        # Track demotions (moved out of top 5)
        demotions = previous_active_names - new_active_names
        if demotions:
            print(f"[Pool] 🔽 DEMOTED to reserve ({len(demotions)}):")
            for name in demotions:
                score = next((score for n, score in new_reserve if n == name), 0.0)
                print(f"[Pool]   ↓ {name}: {score:.1%}")
        
        self.active_conditions = new_active
        self.reserve_conditions = new_reserve
        self.previous_active = new_active_names
        
        # Print pool status
        print(f"\n[Pool] 📊 Condition Pool Status:")
        print(f"[Pool]   Total conditions: {len(self.condition_scores)}")
        print(f"[Pool]   Active (top 5): {len(self.active_conditions)}")
        print(f"[Pool]   Reserve: {len(self.reserve_conditions)}")
    
    def _check_for_missing_conditions(self, element: str, answer: str):
        """Check if LLM should consider additional conditions based on answer."""
        # Only check for missing conditions on key elements that might reveal new diagnoses
        if element not in ['character', 'aggravating', 'relieving', 'location']:
            return
        
        chief_complaint = self.conversation_context['pre_hpi'].get('chief_complaint', '')
        current_conditions = list(self.condition_scores.keys())
        
        # Ask LLM if there are other conditions that should be considered
        system_prompt = (
            "You are a medical expert. Based on the patient's answer, identify if there are "
            "other medical conditions that should be considered in the differential diagnosis.\n\n"
            "Return ONLY valid JSON: {\"additional_conditions\": [\"Condition 1\", \"Condition 2\", ...]}\n"
            "If no additional conditions are needed, return empty list: {\"additional_conditions\": []}\n"
            "No explanations, no other text. Just the JSON object."
        )
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n"
            f"Currently evaluating: {', '.join(current_conditions[:5])}{'...' if len(current_conditions) > 5 else ''}\n\n"
            "Are there other medical conditions that should be considered based on this answer? "
            "For example, if chest pain is 'burning' and 'worse after meals', consider GERD. "
            "If no additional conditions, return empty list."
        )
        
        try:
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            
            if response:
                # Parse JSON
                cleaned = response.strip()
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        cleaned = cleaned[start_idx:end_idx]
                
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and 'additional_conditions' in parsed:
                        additional = parsed['additional_conditions']
                        if isinstance(additional, list) and additional:
                            # Add new conditions at baseline - trust LLM's medical knowledge
                            added_count = 0
                            for cond in additional:
                                if cond not in self.condition_scores:
                                    self.condition_scores[cond] = 0.5
                                    added_count += 1
                                    print(f"[Pool] 🆕 LLM suggested additional condition: {cond}")
                            if added_count > 0:
                                print(f"[Pool] ✅ Added {added_count} new condition(s) to evaluation pool")
                                self._update_rankings()
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            # Silently fail - not critical
            pass
    
    def update_condition_scores_from_answer(self, element: str, answer: str):
        """Update condition scores based on answer - LLM evaluates ALL conditions and can add new ones."""
        # Allow LLM to suggest additional conditions that should be considered
        # This handles cases where initial suggestions missed relevant conditions (e.g., GERD for chest pain)
        self._check_for_missing_conditions(element, answer)
        
        if not self.condition_scores:
            return
        
        # Get ALL conditions, not just top 5
        all_conditions = list(self.condition_scores.keys())
        
        if not all_conditions:
            return
        
        # Build context
        chief_complaint = self.conversation_context['pre_hpi'].get('chief_complaint', '')
        conversation_context = self._build_conversation_context()
        
        # Get current rankings for context
        current_rankings = self.condition_rankings[:5]
        ranking_context = ", ".join([f"{cond} ({score:.2f})" for cond, score in current_rankings])
        
        # Ask LLM to evaluate ALL conditions using its trained medical knowledge
        system_prompt = (
            "You are a medical expert. Return ONLY valid JSON with condition names as keys and score changes as values.\n\n"
            "SCORING:\n"
            "- +0.2 to +0.3 = condition MORE likely\n"
            "- -0.2 to -0.3 = condition LESS likely\n"
            "- 0.0 = neutral/no change\n"
            "- -0.1 to +0.1 = minimal change\n\n"
            "CRITICAL RULES:\n"
            "1. When answer suggests GERD (burning, worse with eating, after eating, better with antacids/cold) → GERD/esophageal: +0.2 to +0.3, Cardiac: -0.2 to -0.3\n"
            "2. When answer suggests cardiac (crushing, pressure, radiating, worse with exertion) → Cardiac: +0.2 to +0.3, GERD/esophageal: -0.2 to -0.3\n"
            "3. ALWAYS use negative scores when answers make conditions less likely - do NOT use 0.0 for conditions that become less likely\n"
            "4. Most answers should result in score changes - avoid all 0.0 unless truly neutral\n\n"
            "Return ONLY JSON. Example: {\"Condition1\": 0.2, \"Condition2\": -0.2, \"Condition3\": 0.0}"
        )
        
        # Build context including any previous location information for combined answers
        location_context = ""
        if element == 'location' and 'location' in self.conversation_context.get('hpi', {}):
            # If this is a location answer that was combined from clarification
            prev_location = self.conversation_context['hpi'].get('location', '')
            if prev_location and ',' in answer:
                # This is a combined answer (original + clarification)
                location_context = f"\nNote: This location answer combines the original answer with clarification."
        
        # Analyze the answer to determine what it suggests
        answer_lower = answer.lower()
        is_gerd_like = any(phrase in answer_lower for phrase in ['burning', 'worse with eating', 'after eating', 'better with antacid', 'better with cold', 'acid', 'reflux'])
        is_cardiac_like = any(phrase in answer_lower for phrase in ['crushing', 'pressure', 'heavy', 'radiating', 'worse with exertion', 'worse with exercise', 'better with rest', 'nitroglycerin'])
        is_neutral = any(phrase in answer_lower for phrase in ['center', 'middle', 'central', 'sternum', 'chest']) and element == 'location'
        
        # Build focused prompt based on what the answer suggests
        if is_gerd_like and element in ['character', 'aggravating', 'relieving', 'timing', 'onset']:
            guidance = f"Answer '{answer}' STRONGLY suggests GERD/esophageal issues. You MUST increase GERD/esophageal conditions by +0.2 to +0.3 and decrease cardiac conditions by -0.2 to -0.3. Do NOT return 0.0 for these conditions."
        elif is_cardiac_like and element in ['character', 'aggravating', 'relieving', 'radiation']:
            guidance = f"Answer '{answer}' STRONGLY suggests cardiac issues. You MUST increase cardiac conditions by +0.2 to +0.3 and decrease GERD/esophageal conditions by -0.2 to -0.3. Do NOT return 0.0 for these conditions."
        elif 'after eating' in answer_lower or 'worse with eating' in answer_lower:
            guidance = f"Answer '{answer}' STRONGLY suggests GERD/esophageal (triggered by eating). You MUST increase GERD/esophageal by +0.2 to +0.3 and decrease cardiac by -0.2 to -0.3. Do NOT return 0.0."
        elif is_neutral:
            guidance = f"Answer '{answer}' is neutral (applies to multiple conditions). You may use 0.0 for most conditions, but consider small adjustments based on current rankings."
        else:
            guidance = f"Evaluate how answer '{answer}' affects each condition. Use +0.2 to +0.3 if more likely, -0.2 to -0.3 if less likely, 0.0 only if truly neutral. Most informative answers should result in score changes."
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'{location_context}\n\n"
            f"{guidance}\n\n"
            f"All conditions ({len(all_conditions)}): {', '.join(all_conditions)}\n\n"
            f"Return JSON with ALL conditions as keys and score changes (-0.3 to +0.3) as values.\n"
            f"Example: {{\"Condition1\": 0.2, \"Condition2\": -0.2, \"Condition3\": 0.0}}\n\n"
            f"CRITICAL: Return ONLY JSON. Every condition must have a score. Use negative scores when answers make conditions less likely."
        )
        
        # Show what we're evaluating
        print(f"\n[Scoring] 🔍 Evaluating {len(all_conditions)} conditions for {element} answer: '{answer}'")
        print(f"[Scoring] 📋 Conditions: {', '.join(all_conditions[:5])}{'...' if len(all_conditions) > 5 else ''}")
        
        try:
            print(f"[Scoring] 🤖 Calling LLM for score evaluation...")
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,  # Increased for evaluating all conditions
                temperature=0.0,
            )
            
            if response:
                print(f"[Scoring] 📥 Raw LLM response (first 200 chars): {response[:200]}")
                # Parse JSON response - try multiple extraction methods
                cleaned = response.strip()
                
                # Method 1: Remove markdown code blocks
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                        elif '```' in cleaned:
                            # Find last ```
                            last_idx = cleaned.rfind('```')
                            cleaned = cleaned[:last_idx].strip()
                
                # Method 2: Extract JSON object using brace matching
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        cleaned = cleaned[start_idx:end_idx]
                
                # Method 3: Remove any text before first { or after last }
                if '{' in cleaned:
                    cleaned = cleaned[cleaned.find('{'):]
                if '}' in cleaned:
                    cleaned = cleaned[:cleaned.rfind('}')+1]
                
                try:
                    score_changes = json.loads(cleaned)
                    if isinstance(score_changes, dict):
                        # Check if all scores are 0.0 (model being too conservative)
                        all_zero = all(abs(float(v)) < 0.01 for v in score_changes.values())
                        
                        # Determine if 0.0 is appropriate (neutral answers like "center of chest" are legitimately neutral)
                        answer_lower = answer.lower()
                        is_neutral_location = element == 'location' and any(phrase in answer_lower for phrase in ['center', 'middle', 'central', 'sternum'])
                        is_neutral_duration = element == 'duration' and any(phrase in answer_lower for phrase in ['few hours', 'few days', 'hours', 'days'])
                        is_neutral_severity = element == 'severity' and answer.strip().isdigit() and 4 <= int(answer.strip()) <= 6
                        
                        # Only apply fallback if answer should have impact but model returned all 0.0
                        if all_zero and len(score_changes) > 0 and not (is_neutral_location or is_neutral_duration or is_neutral_severity):
                            print(f"[Scoring] ⚠️  All scores are 0.0 but answer should have impact. Applying heuristic fallback...")
                            # Apply heuristic-based scoring as fallback
                            
                            # Heuristic: if answer suggests GERD pattern
                            if any(phrase in answer_lower for phrase in ['after eating', 'worse with eating', 'burning', 'acid', 'reflux', 'better with antacid', 'better with cold']):
                                for cond in all_conditions:
                                    if any(term in cond.lower() for term in ['gerd', 'esophageal', 'esophagitis']):
                                        score_changes[cond] = 0.2
                                    elif any(term in cond.lower() for term in ['cardiac', 'myocardial', 'angina', 'heart']):
                                        score_changes[cond] = -0.2
                            
                            # Heuristic: if answer suggests cardiac pattern
                            elif any(phrase in answer_lower for phrase in ['crushing', 'pressure', 'heavy', 'radiating', 'worse with exertion', 'worse with exercise', 'better with rest', 'nitroglycerin']):
                                for cond in all_conditions:
                                    if any(term in cond.lower() for term in ['cardiac', 'myocardial', 'angina', 'heart']):
                                        score_changes[cond] = 0.2
                                    elif any(term in cond.lower() for term in ['gerd', 'esophageal', 'esophagitis']):
                                        score_changes[cond] = -0.2
                            
                            # For other answers that should have impact, apply small changes
                            else:
                                # If answer is informative but model returned 0.0, apply minimal changes
                                if answer and len(answer.strip()) > 3:
                                    # Check current top conditions to maintain context
                                    current_top = sorted(self.condition_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                                    top_condition_names = [c[0].lower() for c in current_top]
                                    
                                    # Small boost to current top conditions if answer doesn't contradict
                                    for cond in all_conditions:
                                        if any(term in cond.lower() for term in top_condition_names):
                                            score_changes[cond] = 0.1
                            
                            print(f"[Scoring] 🔧 Applied heuristic fallback scores")
                        elif all_zero and (is_neutral_location or is_neutral_duration or is_neutral_severity):
                            print(f"[Scoring] ℹ️  All scores are 0.0 - this is appropriate for a neutral answer (e.g., central location applies to multiple conditions)")
                        
                        print(f"[Scoring] ✅ Successfully parsed JSON with {len(score_changes)} condition scores")
                        # Apply score changes to all conditions
                        updated_count = 0
                        significant_changes = []
                        
                        for condition, change in score_changes.items():
                            if condition in self.condition_scores:
                                current_score = self.condition_scores[condition]
                                # Clamp change to reasonable range
                                change_value = max(-0.3, min(0.3, float(change)))
                                new_score = max(0.0, min(1.0, current_score + change_value))
                                self.condition_scores[condition] = new_score
                                updated_count += 1
                                
                                # Log significant changes
                                if abs(change_value) >= 0.1:
                                    significant_changes.append(
                                        f"{condition}: {current_score:.3f} → {new_score:.3f} ({change_value:+.3f})"
                                    )
                                    print(f"[Scoring]   📈 {condition}: {current_score:.3f} → {new_score:.3f} ({change_value:+.3f})")
                            else:
                                # LLM suggested a new condition - trust LLM's medical knowledge
                                # Add it at baseline and apply change
                                change_value = max(-0.3, min(0.3, float(change)))
                                new_score = max(0.0, min(1.0, 0.5 + change_value))
                                self.condition_scores[condition] = new_score
                                updated_count += 1
                                print(f"[Scoring] 🆕 Added new condition: {condition} (initial score: {new_score:.3f})")
                        
                        if updated_count > 0:
                            print(f"\n[Scoring] ✅ Updated {updated_count}/{len(self.condition_scores)} conditions based on {element} answer")
                            if significant_changes:
                                print(f"[Scoring] 📊 Significant changes ({len(significant_changes)}):")
                                for change in significant_changes[:10]:  # Show top 10
                                    print(f"[Scoring]   • {change}")
                        else:
                            print(f"[Scoring] ⚠️  No conditions matched in LLM response")
                            print(f"[Scoring] ⚠️  LLM returned {len(score_changes)} conditions, but none matched session conditions")
                            print(f"[Scoring] ⚠️  LLM keys: {list(score_changes.keys())[:5]}")
                            print(f"[Scoring] ⚠️  Session keys: {list(self.condition_scores.keys())[:5]}")
                            
                        self._update_rankings()
                        self._update_condition_pools()
                        self._print_rankings()
                        
                except json.JSONDecodeError as e:
                    print(f"[Scoring] ❌ Failed to parse LLM score changes: {e}")
                    print(f"[Scoring] ⚠️  Raw response (first 500 chars): {response[:500]}")
                    print(f"[Scoring] ⚠️  Extracted JSON attempt (first 300 chars): {cleaned[:300]}")
                    print(f"[Scoring] ⚠️  This usually means LLM returned conversational text instead of JSON")
                    print(f"[Scoring] ⚠️  Check if model is fine-tuned and following JSON-only instructions")
                    print(f"[Scoring] ⚠️  SKIPPING score update - model needs to be retrained with JSON scoring examples")
                    # Don't update scores if we can't parse JSON - better to skip than use wrong data
        except Exception as e:
            print(f"[Scoring] ⚠️  Error updating scores: {e}")
    
    def _build_conversation_context(self) -> str:
        """Build conversation context for LLM."""
        parts = []
        
        pre_hpi = self.conversation_context.get('pre_hpi', {})
        if pre_hpi.get('chief_complaint'):
            parts.append(f"Chief complaint: {pre_hpi['chief_complaint']}")
        if pre_hpi.get('chronicity'):
            parts.append(f"Chronicity: {pre_hpi['chronicity']}")
        if pre_hpi.get('age'):
            parts.append(f"Age: {pre_hpi['age']}")
        if pre_hpi.get('sex'):
            parts.append(f"Biological sex: {pre_hpi['sex']}")
        
        hpi = self.conversation_context.get('hpi', {})
        hpi_labels = {
            'onset': 'Onset',
            'location': 'Location',
            'duration': 'Duration',
            'character': 'Character',
            'aggravating': 'Aggravating factors',
            'relieving': 'Relieving factors',
            'radiation': 'Radiation',
            'timing': 'Timing',
            'severity': 'Severity',
        }
        
        for key, label in hpi_labels.items():
            value = hpi.get(key)
            if value and value.strip():
                parts.append(f"{label}: {value}")
        
        if not parts:
            return "No information collected yet."
        
        return "\n".join(parts)
    
    def _print_rankings(self):
        """Print current condition rankings with pool status."""
        if not self.condition_rankings:
            print("\n[Rankings] No conditions ranked yet.")
            return
        
        print(f"\n[Rankings] 📊 Top 5 conditions (Active Pool):")
        for idx, (condition, score) in enumerate(self.active_conditions[:5], 1):
            pct = round(score * 100, 1)
            print(f"  {idx}. {condition}: {pct}%")
        
        if self.reserve_conditions:
            print(f"\n[Rankings] 📋 Reserve Pool ({len(self.reserve_conditions)} conditions):")
            for idx, (condition, score) in enumerate(self.reserve_conditions[:5], 1):
                pct = round(score * 100, 1)
                print(f"  {idx+5}. {condition}: {pct}%")
            if len(self.reserve_conditions) > 5:
                print(f"  ... and {len(self.reserve_conditions) - 5} more")
        
        print(f"\n[Rankings] 📊 Total conditions in pool: {len(self.condition_scores)}")
        print(f"[Rankings]    Active: {len(self.active_conditions)}, Reserve: {len(self.reserve_conditions)}")

# ============================================================================
# Skip Tag and Follow-up Support
# ============================================================================

def should_skip_oldcarts_element(navigator, element: str) -> bool:
    """Check if an OLD CARTS element should be skipped based on chief complaint."""
    chief_complaint = navigator.conversation_context['pre_hpi'].get('chief_complaint', '').lower()
    
    # Heuristics for common non-physical/non-localized complaints
    # These match the patterns learned from the dataset skip tags
    if "blood pressure" in chief_complaint or "hypertension" in chief_complaint or "high blood pressure" in chief_complaint:
        if element in ["location", "character", "radiation"]:
            return True
    if "palpitations" in chief_complaint or "heart racing" in chief_complaint:
        if element in ["location", "radiation"]:
            return True
    if "fatigue" in chief_complaint or "tired" in chief_complaint:
        if element in ["location", "character", "radiation"]:
            return True
    if "diarrhea" in chief_complaint or "constipation" in chief_complaint:
        if element in ["location", "character", "radiation"]:
            return True
    if "urinary frequency" in chief_complaint or "urgency" in chief_complaint or "incontinence" in chief_complaint:
        if element in ["location", "character", "radiation"]:
            return True
    if "coffee ground" in chief_complaint or "vomit" in chief_complaint:
        if element in ["location", "character"]:
            return True
    
    # Note: The fine-tuned model should naturally learn to skip irrelevant questions
    # from the skip tags in the training data. If the model generates a skip tag
    # in its response, we'll detect it and skip the question.
    # These heuristics are just a fallback.
    
    return False

def element_name_to_key(element_name: str) -> str:
    """Convert element name to key format."""
    mapping = {
        'o': 'onset', 'onset': 'onset',
        'l': 'location', 'location': 'location',
        'd': 'duration', 'duration': 'duration',
        'c': 'character', 'character': 'character',
        'a': 'aggravating', 'aggravating': 'aggravating',
        'a_alleviating': 'relieving', 'alleviating': 'relieving',
        'r': 'radiation', 'radiation': 'radiation',
        't': 'timing', 'timing': 'timing',
        's': 'severity', 'severity': 'severity',
    }
    return mapping.get(element_name.lower(), element_name.lower())

def generate_context_aware_question(navigator, element: str, symptom: str, top_conditions: List[str], chief_complaint: str) -> str:
    """Generate OLD CARTS question with examples from top 3 conditions."""
    # Use LLM to generate question with examples based on top conditions
    if not top_conditions:
        # Fallback to generic question
        questions = {
            'onset': f"When did {symptom} start? For example, suddenly, gradually, or after eating?",
            'location': f"Where exactly is {symptom} located? For example, upper abdomen, lower abdomen, or one side?",
            'duration': f"How long has {symptom} been present? For example, hours, days, or weeks?",
            'character': f"What does {symptom} feel like? For example, is it sharp, dull, burning, or pressure?",
            'aggravating': f"What makes {symptom} worse? For example, movement, eating, or breathing?",
            'relieving': f"What makes {symptom} better? For example, rest, medication, or position changes?",
            'radiation': f"Does {symptom} spread anywhere else? For example, to your arm, jaw, or back?",
            'timing': f"Is {symptom} constant or does it come and go? For example, constant or intermittent?",
            'severity': f"On a scale of 1 to 10, with 10 being the worst imaginable, how severe is {symptom}? For example, mild (1-3), moderate (4-6), or severe (7-10)?",
        }
        return questions.get(element, f"Can you tell me more about {symptom}?")
    
    # Use LLM to generate question with context-aware examples
    system_prompt = (
        "You are a medical professional. Generate an OLD CARTS question with 1-2 example answers "
        "based on the top 3 conditions in the differential diagnosis.\n\n"
        "The examples should guide the user toward answers that help differentiate between these conditions.\n\n"
        "Format: [Question text]? For example, [example1], [example2]?\n\n"
        "Return ONLY the question with examples. No explanations."
    )
    
    user_prompt = (
        f"Chief complaint: '{chief_complaint}'\n"
        f"OLD CARTS element: {element}\n"
        f"Top 3 conditions in differential: {', '.join(top_conditions)}\n\n"
        f"Generate a question about {element} with 1-2 example answers that would help differentiate between these conditions."
    )
    
    try:
        response = navigator.llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
            temperature=0.4,
        )
        if response and response.strip():
            return response.strip()
    except Exception:
        pass
    
    # Fallback to generic
    questions = {
        'onset': f"When did {symptom} start? For example, suddenly, gradually, or after eating?",
        'location': f"Where exactly is {symptom} located? For example, upper abdomen, lower abdomen, or one side?",
        'duration': f"How long has {symptom} been present? For example, hours, days, or weeks?",
        'character': f"What does {symptom} feel like? For example, is it sharp, dull, burning, or pressure?",
        'aggravating': f"What makes {symptom} worse? For example, movement, eating, or breathing?",
        'relieving': f"What makes {symptom} better? For example, rest, medication, or position changes?",
        'radiation': f"Does {symptom} spread anywhere else? For example, to your arm, jaw, or back?",
        'timing': f"Is {symptom} constant or does it come and go? For example, constant or intermittent?",
        'severity': f"On a scale of 1 to 10, with 10 being the worst imaginable, how severe is {symptom}? For example, mild (1-3), moderate (4-6), or severe (7-10)?",
    }
    return questions.get(element, f"Can you tell me more about {symptom}?")

def generate_final_summary(navigator, messages: List[Dict]):
    """Generate final summary with likely conditions, urgency levels, and next steps."""
    if not navigator.condition_rankings:
        return
    
    # Get top 5 conditions for summary
    top_conditions = navigator.condition_rankings[:5]
    if not top_conditions:
        return
    
    # Build context summary
    context_summary = navigator._build_conversation_context()
    
    # Create prompt for final summary
    conditions_text = "\n".join([f"  {i+1}. {cond}: {score*100:.1f}%" for i, (cond, score) in enumerate(top_conditions)])
    
    summary_prompt = f"""Based on the following medical history and condition rankings, provide a comprehensive clinical assessment:

{context_summary}

Top Likely Conditions (with probability scores):
{conditions_text}

CRITICAL INSTRUCTIONS:
1. Analyze the SYMPTOM PATTERN, not just the probability scores
2. Recognize when symptoms clearly point to one diagnosis over others
3. Example: Burning pain + worse with eating + better with cold water/antacids + after eating = GERD/esophageal, NOT cardiac
4. Example: Crushing pain + radiating to arm + worse with exertion = Cardiac, NOT GERD
5. If symptoms clearly point to one diagnosis, explain why other high-scoring conditions are LESS likely despite their scores

Please provide:
1. A clinical summary analyzing the SYMPTOM PATTERN and what it suggests
2. For each condition, classify urgency as one of:
   - EMERGENT: Requires immediate emergency department evaluation (life-threatening)
   - URGENT: Requires prompt medical attention within 24-48 hours (serious but not immediately life-threatening)
   - SELF-LIMITED: Can be managed at home or with routine care (mild, resolving conditions)
3. Specific next steps/recommendations based on the urgency level

Format your response as:
SUMMARY:
[Clinical analysis of symptom pattern - what the symptoms suggest, not just listing conditions]

LIKELY CONDITIONS (in order of clinical likelihood based on symptom pattern):
1. [Most Likely Condition] ([urgency level])
   - Reasoning: [Why this condition matches the symptom pattern - be specific about which symptoms support this]
   - Why other conditions are less likely: [Explain why other high-scoring conditions don't fit the pattern]
   - Next Steps: [Specific recommendations]

2. [Second Most Likely Condition] ([urgency level])
   - Reasoning: [Why this condition is possible but less likely than #1]
   - Next Steps: [Specific recommendations]

[Continue for top 3-5 conditions, but prioritize based on symptom pattern, not just scores]

IMPORTANT: 
- Analyze symptom patterns, not just probability scores
- If symptoms clearly point to GERD (burning, worse with eating, better with antacids), GERD should be #1 even if cardiac scores are high
- If symptoms clearly point to cardiac (crushing, radiating, worse with exertion), cardiac should be #1
- Be specific about urgency levels. If ANY condition is EMERGENT, clearly state that immediate emergency care is needed."""
    
    try:
        summary_response = navigator.llm_chat(
            messages + [{"role": "user", "content": summary_prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        print("\n" + "="*80)
        print("📋 FINAL ASSESSMENT & RECOMMENDATIONS")
        print("="*80)
        print(summary_response.strip())
        print("="*80)
        
        # Add to messages
        messages.append({"role": "assistant", "content": summary_response.strip()})
    except Exception as e:
        print(f"\n⚠️  Error generating final summary: {e}")
        # Fallback: simple summary
        print("\n" + "="*80)
        print("📋 FINAL ASSESSMENT & RECOMMENDATIONS")
        print("="*80)
        print("\nTop Likely Conditions:")
        for i, (cond, score) in enumerate(top_conditions[:3], 1):
            print(f"  {i}. {cond}: {score*100:.1f}%")
        print("\n⚠️  IMPORTANT: Based on your symptoms, please consult with a healthcare provider.")
        print("   If you are experiencing severe chest pain, difficulty breathing, or other")
        print("   life-threatening symptoms, seek immediate emergency medical attention.")
        print("="*80)


def ask_intelligent_followups(navigator, messages: List[Dict]):
    """Ask associated symptom questions based on top 3 conditions (matches training data format)."""
    if not navigator.condition_rankings:
        return
    
    # Get top 3 conditions for associated symptom questions
    top_3 = [cond for cond, _ in navigator.condition_rankings[:3]]
    if not top_3:
        return
    
    chief_complaint = navigator.conversation_context['pre_hpi'].get('chief_complaint', '')
    
    print(f"\n📋 Asking associated symptom questions to differentiate between top conditions:")
    for i, (cond, score) in enumerate(navigator.condition_rankings[:3], 1):
        print(f"   {i}. {cond}: {score:.1%}")
    print("=" * 80)
    
    # Ask the model to generate associated symptom questions based on top 3 conditions
    system_prompt = (
        "You are a medical professional. Based on the top 3 conditions in the differential diagnosis, "
        "ask 1-3 associated symptom questions that help differentiate between them.\n\n"
        "These questions should be specific to the top conditions and help rule in or rule out diagnoses. "
        "Examples:\n"
        "- For cardiac conditions: sweating, nausea, shortness of breath\n"
        "- For GERD: sour taste, regurgitation\n"
        "- For cholecystitis: nausea, vomiting, fever\n\n"
        "Ask only the question(s), one at a time. Be natural and conversational."
    )
    
    user_prompt = (
        f"Chief complaint: '{chief_complaint}'\n"
        f"Top 3 conditions: {', '.join(top_3)}\n\n"
        f"Ask 1-3 associated symptom questions that would help differentiate between these top 3 conditions. "
        f"Focus on symptoms that are characteristic of some conditions but not others."
    )
    
    try:
        followup_response = navigator.llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.4,
        )
        
        if followup_response and followup_response.strip():
            # Split into individual questions if multiple
            questions = [q.strip() for q in followup_response.split('?') if q.strip()]
            if questions:
                # Add question mark if missing
                first_question = questions[0]
                if not first_question.endswith('?'):
                    first_question += '?'
                
                print(f"🤖 Assistant: {first_question}")
                messages.append({"role": "assistant", "content": first_question})
                
                # Wait for user response
                user_followup = input("👤 You: ").strip()
                if user_followup and user_followup.lower() not in ['quit', 'reset', 'rankings']:
                    messages.append({"role": "user", "content": user_followup})
                    print(f"\n[Associated Symptom] Received answer - considering impact on differential")
                    # Update scores based on associated symptom answer
                    navigator.update_condition_scores_from_answer("associated_symptom", user_followup)
                    navigator._update_rankings()
                    navigator._update_condition_pools()
                    navigator._print_rankings()
    except Exception as e:
        print(f"[Associated Symptom] ⚠️  Error asking associated symptoms: {e}")

# ============================================================================
# Interactive Test
# ============================================================================

def run_interactive_test(model, tokenizer, model_type):
    """Run interactive test with advanced navigator logic."""
    print("\n" + "=" * 80)
    print("ADVANCED MEDICAL NAVIGATOR - INTERACTIVE TEST")
    print("=" * 80)
    print("\nThis test mirrors advanced_medical_navigator.py logic:")
    print("  - Balanced initial seeding (all conditions at 0.5)")
    print("  - LLM evaluates ALL conditions based on answers")
    print("  - Condition rankings update after each answer (progressive scoring)")
    print("  - Clarification questions for ambiguous answers")
    print("  - Context-aware OLD CARTS questions with examples from top 3 conditions")
    print("  - Associated symptom questions to differentiate top conditions")
    print("\n" + "=" * 80 + "\n")
    
    navigator = SimpleMedicalNavigator(model, tokenizer, model_type)
    
    # System prompt for question generation
    question_system_prompt = (
        "You are a professional medical assistant conducting a medical history. "
        "You must understand the conversation context and avoid asking redundant questions.\n\n"
        "Follow this order:\n"
        "1. Show empathy and acknowledge their concern\n"
        "2. Ask if this is new or an ongoing problem\n"
        "3. Ask their age\n"
        "4. Ask their biological sex\n"
        "5. Then ask about the symptom - one question at a time\n\n"
        "Be natural and conversational. Ask only one question at a time. "
        "Do not include internal reasoning, acknowledgments, or explanations. Only ask the question."
    )
    
    messages = [{"role": "system", "content": question_system_prompt}]
    stage = "chief_complaint"
    last_question_type = None  # Track what question was just asked
    last_question_element = None  # Track which OLD CARTS element was asked about
    
    # OLD CARTS elements in order
    oldcarts_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'radiation', 'timing', 'severity']
    
    print("👤 Start by describing your symptoms (e.g., 'I have chest pain' or 'I have abdominal pain')")
    print("   Type 'quit' to exit, 'reset' to start over, 'rankings' to see current rankings\n")
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            print("\n🔄 Resetting conversation...\n")
            navigator = SimpleMedicalNavigator(model, tokenizer, model_type)
            messages = [{"role": "system", "content": question_system_prompt}]
            stage = "chief_complaint"
            last_question_type = None
            last_question_element = None
            navigator.skipped_elements = set()
            navigator.followups_asked = False
            print("👤 Start by describing your symptoms\n")
            continue
        
        # Debug: Print current state
        print(f"[Debug] Before processing answer: last_question_type={last_question_type}, stage={stage}, pre_hpi={navigator.conversation_context['pre_hpi']}")
        
        if user_input.lower() == 'rankings':
            navigator._print_rankings()
            continue
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Handle chief complaint
        if stage == "chief_complaint":
            # First check if it's a simple greeting (before medical complaint check)
            user_lower = user_input.lower().strip()
            simple_greetings = ['hello', 'hi', 'hey', 'hey there', 'good morning', 'good afternoon', 'good evening', 'greetings', 'how are you', 'what\'s up']
            
            # Check for exact match or starts with greeting
            is_greeting = (user_lower in simple_greetings or 
                          any(user_lower == g or user_lower.startswith(g + ' ') or user_lower.startswith(g + '!') or user_lower.startswith(g + '.') 
                              for g in simple_greetings))
            
            if is_greeting:
                # It's a greeting - respond naturally WITHOUT treating as medical complaint
                print(f"[Debug] Detected greeting: '{user_input}'")
                greeting_response = "Hi there! How can I help you today? If you're experiencing any symptoms or have a medical concern, please let me know."
                print(f"🤖 Assistant: {greeting_response}")
                messages.append({"role": "assistant", "content": greeting_response})
                # Stay in chief_complaint stage, wait for actual medical complaint
                continue
            
            # Check if this is a medical complaint or just casual conversation
            if not navigator.is_medical_complaint(user_input):
                # It's a greeting or casual conversation - respond naturally
                greeting_prompt = (
                    "The user just said: '{user_input}'\n\n"
                    "This is a greeting or casual conversation, NOT a medical complaint. "
                    "Respond naturally and friendly. Wait for them to mention a medical concern "
                    "before asking any medical questions. Keep it brief and welcoming."
                ).format(user_input=user_input)
                
                greeting_response = navigator.llm_chat(
                    [{"role": "system", "content": question_system_prompt},
                     {"role": "user", "content": greeting_prompt}],
                    max_tokens=100,
                    temperature=0.7
                )
                print(f"🤖 Assistant: {greeting_response}")
                messages.append({"role": "assistant", "content": greeting_response})
                # Stay in chief_complaint stage, wait for actual medical complaint
                continue
            
            # It's a medical complaint - start assessment
            # Match categories
            categories = navigator.match_chief_complaint_to_categories(user_input)
            navigator.matched_categories = categories
            navigator.conversation_context['pre_hpi']['chief_complaint'] = user_input
            
            # Initialize condition scores - LLM suggests conditions dynamically
            navigator.initialize_condition_scores(categories, user_input)
            
            # Generate empathetic response
            empathetic_prompt = f"The patient just said: '{user_input}'\n\nShow empathy and acknowledge their concern. Be natural and conversational. Do not ask questions yet."
            empathetic_response = navigator.llm_chat(
                [{"role": "system", "content": question_system_prompt},
                 {"role": "user", "content": empathetic_prompt}],
                max_tokens=80,
                temperature=0.4
            )
            print(f"🤖 Assistant: {empathetic_response}")
            messages.append({"role": "assistant", "content": empathetic_response})
            
            # Now ask chronicity question
            stage = "chronicity"
            chronicity_prompt = "Ask if this is new or an ongoing problem. Ask only the question, no acknowledgment or reasoning."
            try:
                chronicity_response = navigator.llm_chat(
                    messages + [{"role": "user", "content": chronicity_prompt}],
                    max_tokens=80,
                    temperature=0.4
                )
                # Fallback if LLM returns empty or invalid response
                if not chronicity_response or not chronicity_response.strip():
                    chronicity_response = "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
            except Exception as e:
                print(f"[Warning] Error generating chronicity question: {e}")
                chronicity_response = "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
            
            print(f"🤖 Assistant: {chronicity_response}")
            messages.append({"role": "assistant", "content": chronicity_response})
            last_question_type = "chronicity"
            continue
        
        # Handle answers based on what question was just asked
        # Only score OLD CARTS elements, not demographics
        pre_hpi = navigator.conversation_context['pre_hpi']
        
        # Track if we just processed a demographic answer (so we skip to next question immediately)
        demographic_answered = False
        
        # Debug: Check what question was asked
        if last_question_type:
            print(f"[Debug] Processing answer for question type: {last_question_type}")
        
        if last_question_type == "chronicity":
            pre_hpi['chronicity'] = user_input
            stage = "age"
            demographic_answered = True
            last_question_type = None  # Reset so we ask age question next
            # Don't score chronicity - it's demographic, not OLD CARTS
        elif last_question_type == "age":
            pre_hpi['age'] = user_input
            stage = "sex"
            demographic_answered = True
            print(f"[Debug] Stored age: {user_input}, set stage to: {stage}, demographic_answered: {demographic_answered}")
            # Reset last_question_type AFTER we ask the next question
            # Don't score age - it's demographic, not OLD CARTS
        elif last_question_type == "sex":
            pre_hpi['sex'] = user_input
            stage = "hpi"
            demographic_answered = True
            last_question_type = None  # Reset so we don't ask sex again
            print(f"[Debug] Stored sex: {user_input}, set stage to: {stage}, demographic_answered: {demographic_answered}")
            # Don't score sex - it's demographic, not OLD CARTS
            # Continue to HPI questions below
        elif last_question_type == "hpi":
            # HPI answers - use tracked element or detect from question
            hpi = navigator.conversation_context['hpi']
            element = None
            
            # First, try to use the tracked element from when we asked the question
            # This is the most reliable method
            if last_question_element:
                element = last_question_element
                if element not in hpi:
                    hpi[element] = user_input
                # Even if already in hpi, use it for scoring (might be updating)
            else:
                # Fallback: detect which OLD CARTS element this is based on last question
                last_assistant_msg = messages[-1]["content"] if messages and messages[-1].get("role") == "assistant" else ""
                last_q_lower = last_assistant_msg.lower()
                print(f"[Debug] No tracked element, detecting from question: '{last_assistant_msg[:50]}...'")
                
                if ('when' in last_q_lower or 'start' in last_q_lower or 'onset' in last_q_lower) and 'onset' not in hpi:
                    hpi['onset'] = user_input
                    element = 'onset'
                elif ('where' in last_q_lower or 'location' in last_q_lower or 'located' in last_q_lower) and 'location' not in hpi:
                    hpi['location'] = user_input
                    element = 'location'
                elif ('feel' in last_q_lower or 'character' in last_q_lower or 'describe' in last_q_lower or 
                      'sharp' in last_q_lower or 'pressure' in last_q_lower or 'burning' in last_q_lower or 'heavy' in last_q_lower) and 'character' not in hpi:
                    hpi['character'] = user_input
                    element = 'character'
                elif ('duration' in last_q_lower or ('how long' in last_q_lower and 'present' in last_q_lower)) and 'duration' not in hpi:
                    # Only match duration if it's about "how long present", not "when did it start"
                    if 'when' not in last_q_lower and 'start' not in last_q_lower:
                        hpi['duration'] = user_input
                        element = 'duration'
                elif ('worse' in last_q_lower or 'aggravating' in last_q_lower) and 'aggravating' not in hpi:
                    hpi['aggravating'] = user_input
                    element = 'aggravating'
                elif ('better' in last_q_lower or 'relieving' in last_q_lower or 'alleviating' in last_q_lower) and 'relieving' not in hpi:
                    hpi['relieving'] = user_input
                    element = 'relieving'
                elif ('severity' in last_q_lower or 'scale' in last_q_lower or '1 to 10' in last_q_lower or 'bad' in last_q_lower) and 'severity' not in hpi:
                    hpi['severity'] = user_input
                    element = 'severity'
                elif element is None and last_question_element:
                    # Use tracked element even if already in hpi (might be updating)
                    element = last_question_element
                    hpi[element] = user_input
            
            # Update condition scores using LLM (only for OLD CARTS elements)
            if element:
                print(f"[Debug] Detected element: {element} for answer: '{user_input}'")
                # Skip scoring if answer is confused/unclear
                confused_phrases = [
                    'what', 'what?', 'huh', 'i don\'t understand', 'clarify',
                    'what do you mean', 'what does that mean', 'i don\'t know',
                    'not sure', 'unclear', 'confused', 'can you explain',
                    'what are you asking', 'repeat', 'again'
                ]
                is_confused = any(phrase in user_input.lower() for phrase in confused_phrases)
                
                if is_confused:
                    print(f"[Info] Skipping scoring for confused/clarification request: '{user_input}'")
                    # Don't store confused responses as answers - we'll re-ask the question
                    if element in hpi:
                        del hpi[element]  # Remove if it was incorrectly stored
                    # Keep last_question_element so we can re-ask
                # Check if this is a clarification answer for any pending element
                pending_key = f"pending_{element}_clarification"
                if hasattr(navigator, pending_key) and getattr(navigator, pending_key):
                    # This is the clarification answer - combine with original
                    original_answer = getattr(navigator, pending_key)
                    combined_answer = f"{original_answer}, {user_input}"
                    print(f"[Info] Combined {element}: '{combined_answer}' (original: '{original_answer}' + clarification: '{user_input}')")
                    hpi[element] = combined_answer
                    setattr(navigator, pending_key, None)  # Clear pending
                    # Score with combined answer
                    navigator.update_condition_scores_from_answer(element, combined_answer)
                    last_question_element = None
                    # Don't continue here - fall through to generate next question immediately
                
                # Use LLM to determine if clarification is needed for any OLD CARTS element
                chief_complaint = navigator.conversation_context['pre_hpi'].get('chief_complaint', '')
                if navigator.needs_clarification(element, user_input, chief_complaint):
                    # Answer is incomplete/ambiguous - ask for clarification
                    print(f"[Info] {element} answer is incomplete/ambiguous, asking for clarification")
                    # Store original answer temporarily
                    setattr(navigator, pending_key, user_input)
                    # Ask clarifying question using LLM
                    clarification = navigator.ask_clarification(element, user_input, chief_complaint, messages)
                    print(f"🤖 Assistant: {clarification}")
                    messages.append({"role": "assistant", "content": clarification})
                    # Don't reset last_question_element - we'll use it when we get the clarified answer
                    continue  # Skip to next iteration to wait for clarified answer
                else:
                    # Answer is specific enough, store it and score
                    hpi[element] = user_input
                    navigator.update_condition_scores_from_answer(element, user_input)
                    last_question_element = None
                    last_question_type = None  # Reset so question generation runs
                    print(f"[Debug] Processed {element} answer, falling through to generate next question")
                    # Don't continue here - fall through to generate next question immediately
            else:
                # If we couldn't detect the element, but we asked a question, try to use tracked element
                if last_question_element and last_question_element not in hpi:
                    # Store answer even if we couldn't detect it properly
                    hpi[last_question_element] = user_input
                    if user_input.lower() not in ['what', 'what?', 'huh', 'i don\'t understand']:
                        navigator.update_condition_scores_from_answer(last_question_element, user_input)
                    last_question_element = None
        
        # SIMPLIFIED FLOW: Ask question → Receive answer → Validate → Next question
        # Generate next question based on what's missing
        # Check stage to determine what question to ask next
        
        # If we're in HPI stage, ask HPI questions (regardless of demographic_answered flag)
        if stage == "hpi":
            # All demographics collected - ask HPI questions
            # Determine which OLD CARTS element to ask about next
            hpi = navigator.conversation_context['hpi']
            # Only include elements that are actually answered (not confused responses)
            answered_elements = {k: v for k, v in hpi.items() if v and v.strip()}
            
            # Check for skip tags - elements that should be skipped based on chief complaint
            skipped_elements = getattr(navigator, 'skipped_elements', set())
            remaining_elements = [e for e in oldcarts_elements 
                                if e not in answered_elements and e not in skipped_elements]
            
            if not remaining_elements:
                # All relevant OLD CARTS collected (some may have been skipped)
                print("\n✅ All relevant OLD CARTS elements collected!")
                if skipped_elements:
                    print(f"   (Skipped irrelevant elements: {', '.join(skipped_elements)})")
                navigator._print_rankings()
                
                # Check if we should ask intelligent follow-up questions
                if not getattr(navigator, 'followups_asked', False):
                    ask_intelligent_followups(navigator, messages)
                    navigator.followups_asked = True
                
                # Generate final summary with likely conditions, urgency, and next steps
                generate_final_summary(navigator, messages)
                
                print("\n👋 Conversation complete. Type 'reset' to start over or 'quit' to exit.")
                continue
            
            # Ask about the next element in order
            next_element = remaining_elements[0]
            last_question_element = next_element  # Track which element we're asking about
            print(f"[Debug] Asking about element: {next_element}, tracking as last_question_element")
            
            # Check if this element should be skipped (model learned from skip tags)
            if should_skip_oldcarts_element(navigator, next_element):
                print(f"[Skip] {next_element} is not relevant for this chief complaint - skipping")
                skipped_elements.add(next_element)
                navigator.skipped_elements = skipped_elements
                continue  # Skip to next element
            
            # Build context and generate question
            context_summary = navigator._build_conversation_context()
            raw_cc = navigator.conversation_context['pre_hpi'].get('chief_complaint', 'symptoms')
            chief_complaint = raw_cc.lower()
            prefixes = ["i have ", "i've got ", "i am having ", "i'm having ", "i am ", "i'm ", "my ", "i feel "]
            for prefix in prefixes:
                if chief_complaint.startswith(prefix):
                    chief_complaint = chief_complaint[len(prefix):].strip()
                    break
            chief_complaint = chief_complaint.strip(" .,!?:;")
            if not chief_complaint:
                chief_complaint = "symptoms"
            
            # Get top 3 conditions for context-aware question examples
            top_3_conditions = [cond for cond, _ in navigator.condition_rankings[:3]] if navigator.condition_rankings else []
            
            # Generate context-aware questions with examples from top 3 conditions
            base_question = generate_context_aware_question(
                navigator, next_element, chief_complaint, top_3_conditions, raw_cc
            )
            
            # Generate question using LLM
            hpi_prompt = (
                f"Context of what we already know:\n{context_summary}\n\n"
                f"You need to ask about the {next_element} of the {chief_complaint}. "
                f"IMPORTANT: You MUST ask about {next_element} specifically using this exact format: '{base_question}' "
                f"Do NOT ask about age, demographics, or information already in the context. "
                f"Do NOT use phrases like 'character of' or 'the {next_element}' - use the natural question format shown above. "
                f"Ask only one question about {next_element}."
            )
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": hpi_prompt}],
                max_tokens=120,
                temperature=0.4
            )
            
            # Clean up response
            response = response.strip()
            if 'age' in response.lower() and 'old' in response.lower() and next_element != 'age':
                response = base_question
            elif next_element == 'character' and ('character of' in response.lower() or 'the character' in response.lower()):
                response = base_question
            elif not response or len(response) < 10:
                response = base_question
            
            last_question_type = "hpi"
            print(f"🤖 Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            continue  # Continue to next iteration to wait for answer
        
        # Handle demographic questions
        elif demographic_answered:
            # We just stored a demographic answer, so skip to the next question based on stage
            if stage == "age":
                # Just answered chronicity, now ask age
                age_system = "You are a medical assistant. Generate ONLY a question asking the patient their age. Do NOT repeat or echo any previous answers. Do NOT acknowledge previous responses. Ask ONLY the age question in second person format."
                age_prompt = "Ask the patient their age using second person (e.g., 'How old are you?' or 'What is your age?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient's age'. IMPORTANT: Do NOT repeat or echo any previous user answers."
                response = navigator.llm_chat(
                    [{"role": "system", "content": age_system}] + messages[-3:] + [{"role": "user", "content": age_prompt}],
                    max_tokens=60,
                    temperature=0.4
                )
                response = response.strip()
                if response.isdigit() or (len(response) <= 3 and any(char.isdigit() for char in response)):
                    response = "How old are you?"
                elif 'patient' in response.lower() and ('age' in response.lower() or 'old' in response.lower()):
                    response = "How old are you?"
                last_question_type = "age"
                print(f"🤖 Assistant: {response}")
                messages.append({"role": "assistant", "content": response})
                continue
            elif stage == "sex":
                # Just answered age, now ask sex
                sex_system = "You are a medical assistant. Generate ONLY a question asking the patient their biological sex. Do NOT repeat or echo any previous answers. Do NOT acknowledge previous responses. Ask ONLY the sex question in second person format."
                sex_prompt = "Ask the patient their biological sex using second person (e.g., 'What is your biological sex?' or 'Are you male or female?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient' or 'is the patient male'. IMPORTANT: Do NOT repeat or echo any previous user answers."
                response = navigator.llm_chat(
                    [{"role": "system", "content": sex_system}] + messages[-3:] + [{"role": "user", "content": sex_prompt}],
                    max_tokens=60,
                    temperature=0.4
                )
                response = response.strip()
                if response.isdigit() or response.lower() in ['male', 'female', 'm', 'f']:
                    response = "What is your biological sex?"
                elif 'patient' in response.lower() and ('male' in response.lower() or 'sex' in response.lower() or 'female' in response.lower()):
                    response = "What is your biological sex?"
                last_question_type = "sex"
                print(f"🤖 Assistant: {response}")
                messages.append({"role": "assistant", "content": response})
                continue
        elif stage == "chronicity" or 'chronicity' not in pre_hpi:
            # Ask chronicity question
            if stage != "chronicity":
                stage = "chronicity"  # Set stage if not already set
            chronicity_prompt = "Ask if this is new or an ongoing problem. Ask only the question, no acknowledgment or reasoning."
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": chronicity_prompt}],
                max_tokens=80,
                temperature=0.4
            )
            last_question_type = "chronicity"
            print(f"🤖 Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            continue
        elif (stage == "age" or 'age' not in pre_hpi) and not demographic_answered and last_question_type != "age":
            # Ask age question - use second person format matching training data
            # Only ask if we didn't just answer a demographic question AND we haven't already asked the age question
            if stage != "age":
                stage = "age"  # Set stage if not already set
            # Use explicit system prompt to prevent echoing user answers
            age_system = "You are a medical assistant. Generate ONLY a question asking the patient their age. Do NOT repeat or echo any previous answers. Do NOT acknowledge previous responses. Ask ONLY the age question in second person format."
            age_prompt = "Ask the patient their age using second person (e.g., 'How old are you?' or 'What is your age?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient's age'. IMPORTANT: Do NOT repeat or echo any previous user answers."
            response = navigator.llm_chat(
                [{"role": "system", "content": age_system}] + messages[-3:] + [{"role": "user", "content": age_prompt}],
                max_tokens=60,
                temperature=0.4
            )
            # Fallback to correct format if LLM generates wrong format or echoes answer
            response = response.strip()
            # Check if response is just echoing a number (age answer)
            if response.isdigit() or (len(response) <= 3 and any(char.isdigit() for char in response)):
                # LLM echoed the age answer, use fallback
                response = "How old are you?"
            elif 'patient' in response.lower() and ('age' in response.lower() or 'old' in response.lower()):
                # LLM used third person, use correct second person format
                response = "How old are you?"
            last_question_type = "age"
            print(f"🤖 Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            continue
        elif (stage == "sex" or 'sex' not in pre_hpi) and not demographic_answered and last_question_type != "sex" and stage != "hpi" and 'sex' not in pre_hpi:
            # Ask sex question - use second person format matching training data
            # Only ask if: stage is sex OR sex not in pre_hpi, AND we didn't just answer a demographic question, 
            # AND we haven't already asked, AND stage is not hpi, AND sex is not already stored
            if stage != "sex" and stage != "hpi":
                stage = "sex"  # Set stage if not already set
            # Use explicit system prompt to prevent echoing user answers
            sex_system = "You are a medical assistant. Generate ONLY a question asking the patient their biological sex. Do NOT repeat or echo any previous answers. Do NOT acknowledge previous responses. Ask ONLY the sex question in second person format."
            sex_prompt = "Ask the patient their biological sex using second person (e.g., 'What is your biological sex?' or 'Are you male or female?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient' or 'is the patient male'. IMPORTANT: Do NOT repeat or echo any previous user answers."
            response = navigator.llm_chat(
                [{"role": "system", "content": sex_system}] + messages[-3:] + [{"role": "user", "content": sex_prompt}],
                max_tokens=60,
                temperature=0.4
            )
            # Fallback to correct format if LLM generates wrong format
            response = response.strip()
            # Check if response is just echoing the age or other answer
            if response.isdigit() or response.lower() in ['male', 'female', 'm', 'f']:
                # LLM echoed an answer, use fallback
                response = "What is your biological sex?"
            elif 'patient' in response.lower() and ('male' in response.lower() or 'sex' in response.lower() or 'female' in response.lower()):
                # LLM used third person, use correct second person format
                response = "What is your biological sex?"
            last_question_type = "sex"
            print(f"🤖 Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            continue
        # If we reach here, something went wrong - should not happen with proper flow
        else:
            print(f"[Error] Unexpected state: stage={stage}, demographic_answered={demographic_answered}, last_question_type={last_question_type}")
            print(f"[Error] pre_hpi keys: {list(pre_hpi.keys())}")
            # Try to recover by checking what's missing
            if stage == "hpi":
                # Should have been caught by first if statement
                print("[Error] Stage is hpi but didn't match first condition - this is a bug")
            elif 'chronicity' not in pre_hpi:
                stage = "chronicity"
            elif 'age' not in pre_hpi:
                stage = "age"
            elif 'sex' not in pre_hpi:
                stage = "sex"
            else:
                stage = "hpi"

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Medical Navigator - Colab Test Script")
    print("=" * 80)
    print("\nLoading model...")
    
    try:
        model, tokenizer, model_type = load_model()
        print(f"\n✅ Model loaded successfully (type: {model_type})")
        
        print("\n" + "=" * 80)
        print("Starting interactive test...")
        print("=" * 80)
        
        run_interactive_test(model, tokenizer, model_type)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

