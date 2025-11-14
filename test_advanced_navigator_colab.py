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
            "You MUST return ONLY valid JSON. No explanations, no text before or after the JSON.\n\n"
            "CRITICAL FORMAT REQUIREMENTS:\n"
            "- Output ONLY valid JSON (no explanations, no text before or after)\n"
            "- JSON must be an object with ALL condition names as keys and numeric scores as values\n"
            "- Example format: {\"Acute Appendicitis\": 0.2, \"Nephrolithiasis (Kidney Stones)\": -0.1, \"Acute Cholecystitis\": 0.0}\n"
            "- Each condition must be a key in the JSON object with its score change as the value\n"
            "- Scores must be between -0.3 and +0.3 (numeric values only)\n"
            "- Do NOT use any other format - only JSON object with condition names as keys and numeric values\n\n"
            "Based on the patient's answer, evaluate how it affects the likelihood of EACH condition. "
            "Use your trained medical knowledge. Consider: classic presentations, anatomical locations, symptom patterns. "
            "Positive values (+0.2 to +0.3) = condition MORE likely. Negative values (-0.2 to -0.3) = condition LESS likely. "
            "Neutral = small changes (-0.1 to +0.1)."
        )
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n\n"
            f"All conditions to evaluate ({len(all_conditions)} total):\n"
            f"{', '.join(all_conditions)}\n\n"
            f"Return ONLY valid JSON with ALL {len(all_conditions)} conditions as keys and their score changes as values.\n"
            f"Format: {{\"condition_name\": score_change, \"condition_name\": score_change, ...}}\n"
            f"Example (not actual conditions): {{\"Condition1\": 0.2, \"Condition2\": -0.1, \"Condition3\": 0.0}}\n\n"
            f"CRITICAL: Return ONLY the JSON object. No explanations, no text before or after. "
            f"Every condition listed above must be a key in the JSON with a numeric score between -0.3 and +0.3."
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
                    print(f"[Scoring] ⚠️  Response (first 500 chars): {response[:500]}")
                    print(f"[Scoring] ⚠️  Extracted JSON attempt: {cleaned[:200]}")
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
            last_question_type = "chronicity"
            continue
        
        # Handle answers based on what question was just asked
        # Only score OLD CARTS elements, not demographics
        pre_hpi = navigator.conversation_context['pre_hpi']
        
        if last_question_type == "chronicity":
            pre_hpi['chronicity'] = user_input
            stage = "age"
            # Don't score chronicity - it's demographic, not OLD CARTS
        elif last_question_type == "age":
            pre_hpi['age'] = user_input
            stage = "sex"
            # Don't score age - it's demographic, not OLD CARTS
        elif last_question_type == "sex":
            pre_hpi['sex'] = user_input
            stage = "hpi"
            # Don't score sex - it's demographic, not OLD CARTS
        elif last_question_type == "hpi":
            # HPI answers - use tracked element or detect from question
            hpi = navigator.conversation_context['hpi']
            element = None
            
            # First, try to use the tracked element from when we asked the question
            if last_question_element and last_question_element not in hpi:
                element = last_question_element
                hpi[element] = user_input
            else:
                # Fallback: detect which OLD CARTS element this is based on last question
                last_assistant_msg = messages[-1]["content"] if messages and messages[-1].get("role") == "assistant" else ""
                last_q_lower = last_assistant_msg.lower()
                
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
                # Skip scoring if answer is confused/unclear
                if user_input.lower() in ['what', 'what?', 'huh', 'i don\'t understand', 'clarify']:
                    print(f"[Info] Skipping scoring for confused response: '{user_input}'")
                else:
                    navigator.update_condition_scores_from_answer(element, user_input)
                last_question_element = None  # Reset after processing
            else:
                # If we couldn't detect the element, but we asked a question, try to use tracked element
                if last_question_element and last_question_element not in hpi:
                    # Store answer even if we couldn't detect it properly
                    hpi[last_question_element] = user_input
                    if user_input.lower() not in ['what', 'what?', 'huh', 'i don\'t understand']:
                        navigator.update_condition_scores_from_answer(last_question_element, user_input)
                    last_question_element = None
        
        # Generate next question based on what's missing
        # Check what we still need to collect
        if 'chronicity' not in pre_hpi:
            # Ask chronicity question
            chronicity_prompt = "Ask if this is new or an ongoing problem. Ask only the question, no acknowledgment or reasoning."
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": chronicity_prompt}],
                max_tokens=80,
                temperature=0.4
            )
            last_question_type = "chronicity"
        elif 'age' not in pre_hpi:
            # Ask age question - use second person format matching training data
            age_prompt = "Ask the patient their age using second person (e.g., 'How old are you?' or 'What is your age?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient's age'."
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": age_prompt}],
                max_tokens=60,
                temperature=0.4
            )
            # Fallback to correct format if LLM generates wrong format
            response = response.strip()
            if 'patient' in response.lower() and ('age' in response.lower() or 'old' in response.lower()):
                # LLM used third person, use correct second person format
                response = "How old are you?"
            last_question_type = "age"
        elif 'sex' not in pre_hpi:
            # Ask sex question - use second person format matching training data
            sex_prompt = "Ask the patient their biological sex using second person (e.g., 'What is your biological sex?' or 'Are you male or female?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient' or 'is the patient male'."
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": sex_prompt}],
                max_tokens=60,
                temperature=0.4
            )
            # Fallback to correct format if LLM generates wrong format
            response = response.strip()
            if 'patient' in response.lower() and ('male' in response.lower() or 'sex' in response.lower() or 'female' in response.lower()):
                # LLM used third person, use correct second person format
                response = "What is your biological sex?"
            last_question_type = "sex"
        else:
            # All demographics collected - ask HPI questions
            # Determine which OLD CARTS element to ask about next
            hpi = navigator.conversation_context['hpi']
            remaining_elements = [e for e in oldcarts_elements if e not in hpi]
            
            if not remaining_elements:
                # All OLD CARTS collected
                print("\n✅ All OLD CARTS elements collected!")
                navigator._print_rankings()
                print("\n👋 Conversation complete. Type 'reset' to start over or 'quit' to exit.")
                continue
            
            # Ask about the next element in order
            next_element = remaining_elements[0]
            last_question_element = next_element  # Track which element we're asking about
            
            # Build context and specific guidance
            context_summary = navigator._build_conversation_context()
            raw_cc = navigator.conversation_context['pre_hpi'].get('chief_complaint', 'symptoms')
            
            # Normalize chief complaint (remove "I have", "I'm having", etc.)
            chief_complaint = raw_cc.lower()
            prefixes = ["i have ", "i've got ", "i am having ", "i'm having ", "i am ", "i'm ", "my ", "i feel "]
            for prefix in prefixes:
                if chief_complaint.startswith(prefix):
                    chief_complaint = chief_complaint[len(prefix):].strip()
                    break
            chief_complaint = chief_complaint.strip(" .,!?:;")
            if not chief_complaint:
                chief_complaint = "symptoms"
            
            # Element-specific guidance with normalized complaint
            element_guidance = {
                'onset': f"When did the {chief_complaint} start?",
                'location': f"Where exactly is the {chief_complaint} located?",
                'duration': f"How long has the {chief_complaint} been present?",
                'character': f"What does the {chief_complaint} feel like? For example, is it sharp, heavy, burning, or pressure?",
                'aggravating': f"What makes the {chief_complaint} worse?",
                'relieving': f"What makes the {chief_complaint} better?",
                'radiation': f"Does the {chief_complaint} spread to other areas?",
                'timing': f"Is the {chief_complaint} constant or does it come and go?",
                'severity': f"On a scale from 1 to 10, how severe is the {chief_complaint}?",
            }
            
            base_question = element_guidance.get(next_element, f"Tell me about the {next_element} of {chief_complaint}.")
            
            # Use base question directly - LLM can rephrase naturally but must ask about the correct element
            hpi_prompt = (
                f"Context of what we already know:\n{context_summary}\n\n"
                f"You need to ask about the {next_element} of the {chief_complaint}. "
                f"IMPORTANT: You MUST ask about {next_element} specifically. "
                f"Do NOT ask about age, demographics, or information already in the context. "
                f"Ask only one question about {next_element}. "
                f"Example question format: {base_question}"
            )
            response = navigator.llm_chat(
                messages + [{"role": "user", "content": hpi_prompt}],
                max_tokens=120,
                temperature=0.4
            )
            
            # Clean up response - remove any weird phrasing
            response = response.strip()
            # If response seems wrong, use base question directly
            if 'age' in response.lower() and 'old' in response.lower() and next_element != 'age':
                print(f"[Warning] LLM generated wrong question, using base question instead")
                response = base_question
            elif not response or len(response) < 10:
                response = base_question
            
            last_question_type = "hpi"
        
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

