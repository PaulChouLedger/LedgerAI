#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Medical Navigator - Colab Test Script
===============================================

This script mirrors the logic of advanced_medical_navigator.py for testing in Colab.
It implements:
- Balanced initial seeding (all conditions at 0.5)
- LLM evaluates ALL conditions based on answers
- Condition rankings update after each answer
- Uses fine-tuned model's trained knowledge

To use in Colab:
1. Upload this script and your fine-tuned model
2. Run: !pip install unsloth transformers accelerate
3. Run this script
"""

import json
import re
import os
import glob
from typing import List, Dict, Optional, Tuple
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
# Model Loading
# ============================================================================

def load_model():
    """Load fine-tuned model (tries multiple formats)."""
    # Try Unsloth format
    if UNSLOTH_AVAILABLE:
        try:
            if os.path.exists("outputs/"):
                print("📦 Loading Unsloth model from outputs/...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="outputs/",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
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
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "System:", "\n\n"],
        )
        
        return response["choices"][0]["text"].strip()
    
    return ""

# ============================================================================
# Advanced Medical Navigator Logic (Simplified for Colab)
# ============================================================================

class SimpleMedicalNavigator:
    """Simplified navigator that mirrors advanced_medical_navigator.py logic."""
    
    # Medical categories and their conditions (simplified - in production, load from guidelines)
    CATEGORY_CONDITIONS = {
        'gastrointestinal': [
            'Acute Appendicitis',
            'Acute Cholecystitis',
            'Acute Cholangitis',
            'Acute Diverticulitis',
            'Acute Gastroenteritis',
            'Gastroesophageal Reflux Disease (GERD)',
            'Acute Upper GI Bleed',
            'Acute Lower GI Bleed',
            'Bowel Obstruction',
        ],
        'cardiovascular': [
            'Acute Myocardial Infarction (Heart Attack)',
            'Unstable Angina',
            'Stable Angina',
            'Aortic Dissection',
            'Pericarditis',
            'Pulmonary Embolism',
        ],
        'renal': [
            'Nephrolithiasis (Kidney Stones)',
            'Acute Kidney Injury',
            'Acute Glomerulonephritis',
            'Urinary Tract Infection',
            'Urinary Retention',
        ],
        'respiratory': [
            'Pneumonia',
            'Asthma Exacerbation',
            'COPD Exacerbation',
            'Pneumothorax',
            'Respiratory Failure',
        ],
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
    
    def llm_chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Wrapper for LLM chat function."""
        return generate_response(self.model, self.tokenizer, messages, self.model_type, max_tokens, temperature)
    
    def match_chief_complaint_to_categories(self, chief_complaint: str) -> List[str]:
        """Match chief complaint to medical categories using LLM."""
        # Get available categories
        available_categories = list(self.CATEGORY_CONDITIONS.keys())
        available_cats_str = ', '.join(available_categories)
        
        system_prompt = (
            "You are a medical expert. Based on the chief complaint, identify which medical categories are relevant. "
            f"Categories: {available_cats_str}. "
            "Return JSON: {\"categories\": [\"category1\", \"category2\", ...]}. "
            "Include all relevant categories. Output only JSON, no other text."
        )
        
        user_prompt = f"Chief complaint: '{chief_complaint}'\n\nIdentify relevant medical categories."
        
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
                return ['gastrointestinal']  # Default fallback
            
            # Parse JSON
            cleaned = response.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline+1:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3].strip()
            
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'categories' in parsed:
                    categories = parsed['categories']
                    if isinstance(categories, list):
                        valid_categories = [cat for cat in categories if cat in available_categories]
                        if valid_categories:
                            return valid_categories
            except json.JSONDecodeError:
                pass
            
            return ['gastrointestinal']  # Default fallback
            
        except Exception as e:
            print(f"⚠️  Error in category matching: {e}")
            return ['gastrointestinal']  # Default fallback
    
    def initialize_condition_scores(self, categories: List[str]):
        """Initialize condition scores - ALL at balanced baseline (0.5)."""
        all_conditions = []
        for category in categories:
            if category in self.CATEGORY_CONDITIONS:
                all_conditions.extend(self.CATEGORY_CONDITIONS[category])
        
        # Start ALL conditions at balanced baseline (0.5)
        # This allows LLM to evaluate and narrow down based on answers
        self.condition_scores = {cond: 0.5 for cond in all_conditions}
        print(f"\n📋 Seeded {len(self.condition_scores)} conditions at balanced baseline 50.0%")
        print(f"   Categories: {', '.join(categories)}")
        print(f"   LLM will narrow down based on answers\n")
        
        self._update_rankings()
    
    def _update_rankings(self):
        """Update condition rankings."""
        self.condition_rankings = sorted(
            self.condition_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def update_condition_scores_from_answer(self, element: str, answer: str):
        """Update condition scores based on answer - LLM evaluates ALL conditions."""
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
            "You are a medical expert with extensive training in clinical reasoning. "
            "Based on the patient's answer, evaluate how it affects the likelihood of EACH condition. "
            "Use your trained medical knowledge to determine which conditions become more or less likely. "
            "Consider: classic presentations, anatomical locations, symptom patterns, and differential diagnosis logic. "
            "Return JSON: {\"condition_name\": score_change} where score_change is between -0.3 and +0.3. "
            "Positive values mean the condition is MORE likely (rule in), negative means LESS likely (rule out). "
            "Be specific: if the answer strongly supports a condition, use +0.2 to +0.3. "
            "If it strongly rules out a condition, use -0.2 to -0.3. "
            "If neutral or unclear, use small changes (-0.1 to +0.1). "
            "You MUST evaluate ALL conditions listed. Output only JSON, no other text."
        )
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n\n"
            f"All conditions to evaluate ({len(all_conditions)} total):\n"
            f"{', '.join(all_conditions)}\n\n"
            f"Current top conditions: {ranking_context if ranking_context else 'all at baseline'}\n\n"
            f"Conversation context:\n{conversation_context}\n\n"
            f"Using your trained medical knowledge, evaluate how this answer affects EACH condition. "
            f"Consider classic presentations, anatomical locations, and symptom patterns. "
            f"Return JSON with score changes for ALL {len(all_conditions)} conditions listed above."
        )
        
        try:
            response = self.llm_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,  # Increased for evaluating all conditions
                temperature=0.0,
            )
            
            if response:
                # Parse JSON response
                cleaned = response.strip()
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                
                try:
                    score_changes = json.loads(cleaned)
                    if isinstance(score_changes, dict):
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
                                        f"  {condition}: {current_score:.3f} → {new_score:.3f} ({change_value:+.3f})"
                                    )
                        
                        if updated_count > 0:
                            print(f"\n[Scoring] ✅ Updated {updated_count}/{len(all_conditions)} conditions based on {element} answer")
                            if significant_changes:
                                print("[Scoring] Significant changes:")
                                for change in significant_changes[:5]:  # Show top 5
                                    print(change)
                        else:
                            print(f"[Scoring] ⚠️  No conditions matched in LLM response")
                            
                        self._update_rankings()
                        self._print_rankings()
                        
                except json.JSONDecodeError as e:
                    print(f"[Scoring] ⚠️  Failed to parse LLM score changes: {e}")
                    print(f"[Scoring] ⚠️  Response (first 300 chars): {response[:300]}")
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
        """Print current condition rankings."""
        print(f"\n[Rankings] 📊 Top 5 conditions:")
        for idx, (condition, score) in enumerate(self.condition_rankings[:5], 1):
            pct = round(score * 100, 1)
            print(f"  {idx}. {condition}: {pct}%")

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
    print("  - Condition rankings update after each answer")
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
            print("👤 Start by describing your symptoms\n")
            continue
        
        if user_input.lower() == 'rankings':
            navigator._print_rankings()
            continue
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Handle chief complaint
        if stage == "chief_complaint":
            # Match categories
            categories = navigator.match_chief_complaint_to_categories(user_input)
            navigator.matched_categories = categories
            navigator.conversation_context['pre_hpi']['chief_complaint'] = user_input
            
            # Initialize condition scores (balanced baseline)
            navigator.initialize_condition_scores(categories)
            
            # Generate empathetic response and first question
            stage = "chronicity"
            response = navigator.llm_chat(messages, max_tokens=120, temperature=0.4)
            print(f"🤖 Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            continue
        
        # Handle answers and generate next question
        # For simplicity, detect what type of answer this is
        if 'chronicity' not in navigator.conversation_context['pre_hpi']:
            navigator.conversation_context['pre_hpi']['chronicity'] = user_input
            stage = "age"
        elif 'age' not in navigator.conversation_context['pre_hpi']:
            navigator.conversation_context['pre_hpi']['age'] = user_input
            stage = "sex"
        elif 'sex' not in navigator.conversation_context['pre_hpi']:
            navigator.conversation_context['pre_hpi']['sex'] = user_input
            stage = "hpi"
        else:
            # HPI answers - update condition scores
            # Detect which element this might be (simplified)
            hpi = navigator.conversation_context['hpi']
            if 'onset' not in hpi:
                hpi['onset'] = user_input
                element = 'onset'
            elif 'location' not in hpi:
                hpi['location'] = user_input
                element = 'location'
            elif 'character' not in hpi:
                hpi['character'] = user_input
                element = 'character'
            else:
                element = 'unknown'
            
            # Update condition scores using LLM
            if element != 'unknown':
                navigator.update_condition_scores_from_answer(element, user_input)
        
        # Generate next question
        response = navigator.llm_chat(messages, max_tokens=120, temperature=0.4)
        print(f"🤖 Assistant: {response}")
        messages.append({"role": "assistant", "content": response})

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

