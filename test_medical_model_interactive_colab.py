#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Medical Model Testing Script for Google Colab
Tests the fine-tuned medical bot model interactively

To use in Colab:
1. Upload your fine-tuned model (from outputs/ or gguf_model/)
2. Run: !pip install unsloth transformers torch llama-cpp-python
3. Run this script
"""

import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# ============================================================================
# Auto-install dependencies (Colab)
# ============================================================================

def install_dependencies():
    """Install required packages if not available."""
    try:
        import unsloth
        print("✅ unsloth already installed")
    except ImportError:
        print("📦 Installing unsloth...")
        os.system("pip install unsloth -q")
    
    try:
        import transformers
        print("✅ transformers already installed")
    except ImportError:
        print("📦 Installing transformers...")
        os.system("pip install transformers -q")
    
    try:
        import torch
        print("✅ torch already installed")
    except ImportError:
        print("📦 Installing torch...")
        os.system("pip install torch -q")
    
    try:
        import llama_cpp
        print("✅ llama-cpp-python already installed")
    except ImportError:
        print("📦 Installing llama-cpp-python...")
        os.system("pip install llama-cpp-python -q")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: Optional[str] = None):
    """
    Load the fine-tuned model.
    Tries: Unsloth format -> HuggingFace format -> GGUF format
    """
    if model_path is None:
        # Try common paths
        possible_paths = [
            "outputs/checkpoint-*/",  # Unsloth checkpoint
            "outputs/",  # Unsloth final model
            "gguf_model/",  # GGUF format
            "models/",  # Generic models directory
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None:
        print("⚠️  No model path found. Please specify model_path or upload model.")
        return None
    
    print(f"🔍 Looking for model in: {model_path}")
    
    # Try 1: Unsloth format
    try:
        from unsloth import FastLanguageModel
        print("📦 Attempting to load with Unsloth...")
        
        # Find checkpoint or final model
        if "checkpoint" in model_path or os.path.isdir(model_path):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=2048,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)  # Enable inference mode
            print("✅ Model loaded with Unsloth")
            return model, tokenizer, "unsloth"
    except Exception as e:
        print(f"⚠️  Unsloth loading failed: {e}")
    
    # Try 2: Standard HuggingFace format
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("📦 Attempting to load with HuggingFace transformers...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model loaded with HuggingFace transformers")
        return model, tokenizer, "transformers"
    except Exception as e:
        print(f"⚠️  HuggingFace loading failed: {e}")
    
    # Try 3: GGUF format
    try:
        import llama_cpp
        print("📦 Attempting to load GGUF model...")
        
        # Find .gguf file
        gguf_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.gguf'):
                    gguf_files.append(os.path.join(root, file))
        
        if not gguf_files:
            raise FileNotFoundError("No .gguf files found")
        
        gguf_file = gguf_files[0]
        print(f"📄 Found GGUF file: {gguf_file}")
        
        llm = llama_cpp.Llama(
            model_path=gguf_file,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        print("✅ Model loaded with llama-cpp-python")
        return llm, None, "gguf"
    except Exception as e:
        print(f"⚠️  GGUF loading failed: {e}")
    
    print("❌ Failed to load model from any format")
    return None

# ============================================================================
# Chat Template
# ============================================================================

def format_chat_template(messages: List[Dict[str, str]], tokenizer) -> str:
    """Format messages using Llama-3.2 chat template."""
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback formatting
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|system|>\n{content}<|end|>\n"
            elif role == 'user':
                formatted += f"<|user|>\n{content}<|end|>\n"
            elif role == 'assistant':
                formatted += f"<|assistant|>\n{content}<|end|>\n"
        formatted += "<|assistant|>\n"
        return formatted

# ============================================================================
# Response Generation
# ============================================================================

def generate_response(
    model,
    tokenizer,
    model_type: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    """Generate response from model."""
    
    if model_type == "unsloth":
        # Unsloth inference
        inputs = tokenizer(
            format_chat_template(messages, tokenizer),
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "transformers":
        # Standard HuggingFace
        inputs = tokenizer(
            format_chat_template(messages, tokenizer),
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "gguf":
        # GGUF format
        formatted = format_chat_template(messages, None)
        response = model(
            formatted,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|end|>", "<|user|>", "\n\nUser:", "\n\n👤"],
            echo=False,
        )
        return response['choices'][0]['text'].strip()
    
    return ""

# ============================================================================
# Interactive Testing
# ============================================================================

def clean_response(response: str) -> str:
    """Clean LLM response to extract only the question/statement."""
    if not response:
        return ""
    
    # Remove markdown code blocks
    if response.startswith('```'):
        first_newline = response.find('\n')
        if first_newline != -1:
            response = response[first_newline+1:]
            if response.endswith('```'):
                response = response[:-3].strip()
    
    # Remove internal reasoning phrases
    reasoning_phrases = [
        'based on', 'according to', 'i understand that', 'it seems',
        'let me', 'i should', 'i need to', 'i will', 'i can',
        'your age is', 'you are', 'you have', 'you mentioned'
    ]
    response_lower = response.lower()
    for phrase in reasoning_phrases:
        if phrase in response_lower:
            # Try to extract text after reasoning
            idx = response_lower.find(phrase)
            if idx > 0:
                # Check if there's content before (might be valid)
                before = response[:idx].strip()
                after = response[idx + len(phrase):].strip()
                if after and len(after) > 10:  # Prefer content after reasoning
                    response = after
                elif before and len(before) > 10:  # Otherwise use before
                    response = before
    
    # Remove statements that look like assumptions (e.g., "Your age is 27")
    if response.strip().startswith(('Your ', 'You are ', 'You have ', 'You mentioned ')):
        # Convert to question if possible
        if 'is' in response.lower() or 'are' in response.lower():
            # Try to convert "Your age is 27" to "How old are you?"
            if 'age' in response.lower():
                response = "How old are you?"
            elif 'sex' in response.lower() or 'gender' in response.lower():
                response = "What is your biological sex?"
            else:
                # Generic conversion
                response = response.replace('Your ', 'What is your ', 1)
                response = response.replace('You are ', 'What is your ', 1)
                if not response.endswith('?'):
                    response = response.rstrip('.') + '?'
    
    # Extract first sentence/question
    sentences = response.split('.')
    questions = response.split('?')
    
    # Prefer question if available
    if questions and len(questions) > 1:
        return questions[0].strip() + '?'
    
    # Otherwise return first sentence
    if sentences:
        result = sentences[0].strip()
        if not result.endswith(('?', '.', '!')):
            result += '.'
        return result
    
    return response.strip()

def load_guidelines() -> Dict[str, Dict]:
    """Load medical guidelines for differential diagnosis."""
    guidelines = {}
    guidelines_path = Path("llm-medical-container/medical/guidelines")
    
    if not guidelines_path.exists():
        return guidelines
    
    for json_file in guidelines_path.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                guideline = json.load(f)
                condition_name = guideline.get('condition', '')
                if condition_name:
                    guidelines[condition_name] = guideline
        except Exception:
            pass
    
    return guidelines

def extract_chief_complaint(conversation: List[Dict[str, str]]) -> Optional[str]:
    """Extract chief complaint from conversation."""
    for msg in conversation:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            medical_keywords = [
                'pain', 'ache', 'hurt', 'sore', 'symptom', 'problem', 'issue',
                'fever', 'cough', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue',
                'bleeding', 'rash', 'swelling', 'discomfort', 'tender', 'stiff',
                'chest', 'head', 'stomach', 'abdominal', 'back', 'arm', 'leg',
                'breathing', 'shortness', 'difficulty', 'trouble', 'blood'
            ]
            for keyword in medical_keywords:
                if keyword in content:
                    # Extract the complaint
                    words = content.split()
                    for i, word in enumerate(words):
                        if keyword in word:
                            # Get surrounding context
                            start = max(0, i - 2)
                            end = min(len(words), i + 5)
                            complaint = ' '.join(words[start:end])
                            return complaint
    return None

def match_chief_complaint_to_conditions(chief_complaint: str, guidelines: Dict[str, Dict]) -> List[str]:
    """Match chief complaint to potential conditions."""
    if not chief_complaint:
        return []
    
    complaint_lower = chief_complaint.lower()
    matched = []
    
    for condition_name, guideline in guidelines.items():
        triggers = guideline.get('chief_complaint_triggers', [])
        if not triggers:
            triggers = guideline.get('key_features', {}).get('chief_complaint_triggers', [])
        
        for trigger in triggers:
            if trigger.lower() in complaint_lower or complaint_lower in trigger.lower():
                matched.append(condition_name)
                break
    
    # If no matches, return top 4 GI conditions (most common)
    if not matched:
        gi_conditions = [
            'Acute Cholecystitis',
            'Acute Gastroenteritis',
            'Acute Appendicitis',
            'gastroesophageal reflux disease (GERD)'
        ]
        for cond in gi_conditions:
            if cond in guidelines:
                matched.append(cond)
                if len(matched) >= 4:
                    break
    
    return matched[:4]  # Return top 4

def extract_oldcarts_info(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract OLD CARTS information from conversation."""
    oldcarts = {
        'onset': None,
        'location': None,
        'duration': None,
        'character': None,
        'aggravating': None,
        'alleviating': None,
        'radiation': None,
        'timing': None,
        'severity': None
    }
    
    # Look through user messages for answers
    for msg in conversation:
        if msg.get('role') != 'user':
            continue
        
        content = msg.get('content', '').lower()
        
        # Extract location
        if not oldcarts['location']:
            location_keywords = ['right', 'left', 'center', 'middle', 'upper', 'lower', 
                               'chest', 'abdomen', 'stomach', 'back', 'head', 'arm', 'leg',
                               'rib', 'shoulder', 'groin', 'belly']
            if any(kw in content for kw in location_keywords):
                oldcarts['location'] = msg.get('content', '')
        
        # Extract character
        if not oldcarts['character']:
            character_keywords = ['sharp', 'dull', 'burning', 'pressure', 'stabbing', 
                                'aching', 'throbbing', 'cramping']
            if any(kw in content for kw in character_keywords):
                oldcarts['character'] = msg.get('content', '')
        
        # Extract timing/duration
        if not oldcarts['duration']:
            if any(word in content for word in ['day', 'hour', 'week', 'month', 'ago']):
                oldcarts['duration'] = msg.get('content', '')
        
        # Extract aggravating
        if not oldcarts['aggravating']:
            if any(word in content for word in ['worse', 'aggravating', 'makes it']):
                oldcarts['aggravating'] = msg.get('content', '')
        
        # Extract alleviating
        if not oldcarts['alleviating']:
            if any(word in content for word in ['better', 'helps', 'relieves', 'alleviates']):
                oldcarts['alleviating'] = msg.get('content', '')
        
        # Extract timing
        if not oldcarts['timing']:
            if any(word in content for word in ['constant', 'comes and goes', 'intermittent', 'episodic']):
                oldcarts['timing'] = msg.get('content', '')
    
    return oldcarts

def score_condition(condition_name: str, guideline: Dict, oldcarts: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Score a condition based on OLD CARTS information."""
    score = 0.0
    reasons = []
    
    oldcarts_data = guideline.get('key_features', {}).get('structured_oldcarts', {})
    
    # Score location
    if oldcarts['location'] and 'location' in oldcarts_data:
        includes = oldcarts_data['location'].get('includes', [])
        location_lower = oldcarts['location'].lower()
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                if any(word in location_lower for word in friendly_words) or patient_friendly in location_lower:
                    score += 0.3
                    reasons.append(f"Location matches: {patient_friendly}")
                    break
    
    # Score character
    if oldcarts['character'] and 'character' in oldcarts_data:
        includes = oldcarts_data['character'].get('includes', [])
        character_lower = oldcarts['character'].lower()
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                if any(word in character_lower for word in friendly_words) or patient_friendly in character_lower:
                    score += 0.25
                    reasons.append(f"Character matches: {patient_friendly}")
                    break
    
    # Score timing
    if oldcarts['timing'] and 'timing' in oldcarts_data:
        includes = oldcarts_data['timing'].get('includes', [])
        timing_lower = oldcarts['timing'].lower()
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                if any(word in timing_lower for word in friendly_words) or patient_friendly in timing_lower:
                    score += 0.2
                    reasons.append(f"Timing matches: {patient_friendly}")
                    break
    
    # Score aggravating
    if oldcarts['aggravating'] and 'aggravating' in oldcarts_data:
        includes = oldcarts_data['aggravating'].get('includes', [])
        aggravating_lower = oldcarts['aggravating'].lower()
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                if any(word in aggravating_lower for word in friendly_words) or patient_friendly in aggravating_lower:
                    score += 0.15
                    reasons.append(f"Aggravating matches: {patient_friendly}")
                    break
    
    # Score alleviating
    if oldcarts['alleviating'] and 'relieving' in oldcarts_data:
        includes = oldcarts_data['relieving'].get('includes', [])
        alleviating_lower = oldcarts['alleviating'].lower()
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                if any(word in alleviating_lower for word in friendly_words) or patient_friendly in alleviating_lower:
                    score += 0.15
                    reasons.append(f"Alleviating matches: {patient_friendly}")
                    break
    
    return min(score, 1.0), reasons

def generate_differential_reasoning(conversation: List[Dict[str, str]], guidelines: Dict[str, Dict]) -> str:
    """Generate differential diagnosis reasoning based on conversation."""
    chief_complaint = extract_chief_complaint(conversation)
    if not chief_complaint:
        return ""
    
    conditions = match_chief_complaint_to_conditions(chief_complaint, guidelines)
    if not conditions:
        return ""
    
    oldcarts = extract_oldcarts_info(conversation)
    
    # Score each condition
    scored_conditions = []
    for condition_name in conditions:
        guideline = guidelines.get(condition_name)
        if not guideline:
            continue
        
        score, reasons = score_condition(condition_name, guideline, oldcarts)
        scored_conditions.append((condition_name, score, reasons))
    
    # Sort by score
    scored_conditions.sort(key=lambda x: x[1], reverse=True)
    
    # Generate reasoning text
    reasoning = "=" * 80 + "\n"
    reasoning += "🔍 DIFFERENTIAL DIAGNOSIS REASONING\n"
    reasoning += "=" * 80 + "\n"
    reasoning += f"\nChief Complaint: {chief_complaint}\n"
    reasoning += f"\nConsidering Conditions: {', '.join(conditions)}\n"
    reasoning += "\n" + "-" * 80 + "\n"
    reasoning += "CURRENT SCORES:\n"
    reasoning += "-" * 80 + "\n"
    
    for i, (condition, score, reasons) in enumerate(scored_conditions, 1):
        reasoning += f"\n{i}. {condition}: {score:.0%} likelihood\n"
        if reasons:
            for reason in reasons:
                reasoning += f"   ✅ {reason}\n"
        else:
            reasoning += "   ⚠️  No matches yet\n"
    
    # Show top diagnosis
    if scored_conditions:
        top_condition, top_score, top_reasons = scored_conditions[0]
        reasoning += "\n" + "-" * 80 + "\n"
        reasoning += f"🏆 TOP DIAGNOSIS: {top_condition} ({top_score:.0%} confidence)\n"
        reasoning += "-" * 80 + "\n"
    
    reasoning += "\n" + "=" * 80 + "\n"
    
    return reasoning

def track_sequence(conversation: List[Dict[str, str]]) -> Dict[str, bool]:
    """Track what parts of the sequence have been covered."""
    sequence = {
        'empathy': False,
        'chronicity': False,
        'age': False,
        'sex': False,
        'oldcarts': False,
    }
    
    for msg in conversation:
        if msg.get('role') == 'assistant':
            content = msg['content'].lower()
            # Empathy - must be first
            if not sequence['empathy'] and ('understand' in content or 'here to help' in content or 'sorry' in content):
                sequence['empathy'] = True
            # Chronicity - must come after empathy
            elif sequence['empathy'] and not sequence['chronicity']:
                if ('new' in content and ('ongoing' in content or 'before' in content or 'prior' in content)) or \
                   ('chronicity' in content):
                    sequence['chronicity'] = True
            # Age - must come after chronicity
            elif sequence['chronicity'] and not sequence['age']:
                if 'age' in content or ('old' in content and 'are you' in content) or 'how old' in content:
                    sequence['age'] = True
            # Sex - must come after age
            elif sequence['age'] and not sequence['sex']:
                if 'biological sex' in content or ('sex' in content and 'biological' in content):
                    sequence['sex'] = True
            # OLD CARTS - must come after sex
            elif sequence['sex']:
                # Any question after sex is OLD CARTS
                if '?' in msg['content'] or any(word in content for word in ['when', 'where', 'how', 'what', 'does', 'is it']):
                    sequence['oldcarts'] = True
    
    return sequence

def build_context_summary(session_state: Dict[str, Any]) -> str:
    """Build a summary of what information has already been collected."""
    parts = []
    
    if session_state.get('chief_complaint'):
        parts.append(f"Chief complaint: {session_state['chief_complaint']}")
    if session_state.get('chronicity'):
        parts.append(f"Chronicity: {session_state['chronicity']}")
    if session_state.get('age'):
        parts.append(f"Age: {session_state['age']}")
    if session_state.get('sex'):
        parts.append(f"Biological sex: {session_state['sex']}")
    
    # OLD CARTS information
    oldcarts_labels = {
        'onset': 'Onset',
        'location': 'Location',
        'duration': 'Duration',
        'character': 'Character',
        'aggravating': 'Aggravating factors',
        'alleviating': 'Relieving factors',
        'radiation': 'Radiation',
        'timing': 'Timing',
        'severity': 'Severity',
    }
    
    for key, label in oldcarts_labels.items():
        value = session_state.get('oldcarts', {}).get(key)
        if value and value.strip():
            parts.append(f"{label}: {value}")
    
    if not parts:
        return "No information collected yet."
    
    return "\n".join(parts)

def is_redundant_question(session_state: Dict[str, Any], question: str) -> bool:
    """Check if a question is redundant given what's already known."""
    question_lower = question.lower()
    
    # Check if asking about information already collected
    if session_state.get('chronicity') and any(word in question_lower for word in ['new', 'ongoing', 'before', 'prior']):
        return True
    
    if session_state.get('age') and ('age' in question_lower or 'old' in question_lower):
        return True
    
    if session_state.get('sex') and ('sex' in question_lower or 'biological' in question_lower):
        return True
    
    # Check OLD CARTS redundancy
    oldcarts = session_state.get('oldcarts', {})
    
    # If onset is known (e.g., "2 days ago"), don't ask duration again
    if oldcarts.get('onset') and any(word in question_lower for word in ['how long', 'duration', 'length of time']):
        onset = oldcarts['onset'].lower()
        if any(word in onset for word in ['day', 'hour', 'week', 'month']):
            return True
    
    # If character is already described, don't ask again
    if oldcarts.get('character') and any(word in question_lower for word in ['feel', 'describe', 'character', 'sensation']):
        return True
    
    # If timing is "constant", don't ask about frequency
    if oldcarts.get('timing') and 'constant' in oldcarts['timing'].lower():
        if any(word in question_lower for word in ['frequency', 'comes and goes', 'intermittent']):
            return True
    
    return False

def determine_next_stage(session_state: Dict[str, Any], user_input: str) -> str:
    """Determine what stage we should be in based on current state."""
    stage = session_state.get('stage', 'awaiting_chief_complaint')
    
    # Check if user has a medical complaint
    medical_keywords = [
        'pain', 'ache', 'hurt', 'sore', 'symptom', 'problem', 'issue',
        'fever', 'cough', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue',
        'bleeding', 'rash', 'swelling', 'discomfort', 'tender', 'stiff',
        'chest', 'head', 'stomach', 'abdominal', 'back', 'arm', 'leg',
        'breathing', 'shortness', 'difficulty', 'trouble', 'blood'
    ]
    
    if stage == 'awaiting_chief_complaint':
        if any(keyword in user_input.lower() for keyword in medical_keywords):
            return 'awaiting_chronicity'
        return 'awaiting_chief_complaint'
    
    if stage == 'awaiting_chronicity':
        if session_state.get('chronicity'):
            return 'awaiting_age'
        return 'awaiting_chronicity'
    
    if stage == 'awaiting_age':
        if session_state.get('age'):
            return 'awaiting_sex'
        return 'awaiting_age'
    
    if stage == 'awaiting_sex':
        if session_state.get('sex'):
            return 'hpi'
        return 'awaiting_sex'
    
    # HPI stage - continue asking OLD CARTS questions
    return 'hpi'

def extract_answer_from_user_input(session_state: Dict[str, Any], user_input: str) -> Optional[str]:
    """Extract structured answer from user input based on current stage."""
    stage = session_state.get('stage', 'awaiting_chief_complaint')
    
    if stage == 'awaiting_chronicity':
        input_lower = user_input.lower()
        if 'new' in input_lower or 'just started' in input_lower or 'recent' in input_lower:
            return 'new'
        elif 'ongoing' in input_lower or 'before' in input_lower or 'prior' in input_lower or 'chronic' in input_lower:
            return 'ongoing'
        return user_input
    
    if stage == 'awaiting_age':
        # Extract age number
        import re
        digits = re.findall(r'\d{1,3}', user_input)
        if digits:
            try:
                age = int(digits[0])
                if 0 < age <= 120:
                    return str(age)
            except ValueError:
                pass
        return user_input
    
    if stage == 'awaiting_sex':
        input_lower = user_input.lower()
        if any(word in input_lower for word in ['male', 'man', 'm']):
            return 'male'
        elif any(word in input_lower for word in ['female', 'woman', 'f']):
            return 'female'
        elif any(word in input_lower for word in ['intersex', 'non-binary', 'nonbinary']):
            return 'intersex/non-binary'
        return user_input
    
    # HPI stage - store in oldcarts
    return user_input

def interactive_test(model, tokenizer, model_type: str):
    """Interactive testing loop with structured stage management."""
    print("=" * 80)
    print("🤖 Interactive Medical Model Testing")
    print("=" * 80)
    print()
    print("Expected sequence:")
    print("  1. Empathy statement")
    print("  2. Chronicity question (new vs ongoing)")
    print("  3. Age question")
    print("  4. Biological sex question")
    print("  5. OLD CARTS questions")
    print()
    print("Commands:")
    print("  • 'quit' or 'exit' - End conversation")
    print("  • 'reset' - Start new conversation")
    print("  • 'show' - Show conversation history")
    print("  • 'sequence' - Show sequence progress")
    print("  • 'debug' or 'reasoning' - Toggle showing internal reasoning")
    print("  • 'differential' or 'diagnosis' - Toggle showing differential diagnosis reasoning")
    print("  • 'prompt' - Show the full prompt being sent to the model")
    print()
    
    # Medical system prompt (matching training)
    # IMPORTANT: Must match exactly what was used in training
    MEDICAL_SYSTEM_PROMPT = """You are a professional medical assistant. 

IMPORTANT RULES:
- ONLY ask medical questions when the patient mentions a symptom, pain, or medical concern
- If the patient is just greeting you or having casual conversation, respond naturally and wait for them to mention a medical issue
- NEVER make up or assume symptoms the patient hasn't mentioned
- NEVER make statements about the patient's information (like "Your age is 27") - always ASK questions
- Always ask questions, never make statements about patient information
- NEVER ask redundant questions about information already provided

CRITICAL SEQUENCE - You MUST follow this EXACT order for EVERY conversation. DO NOT skip any step:

STEP 1: Show empathy and acknowledge their concern (REQUIRED - do this FIRST when patient mentions a symptom)
STEP 2: Ask if this is new or an ongoing problem (REQUIRED - do this SECOND, BEFORE age)
STEP 3: Ask their age (REQUIRED - do this THIRD, AFTER chronicity)
STEP 4: Ask their biological sex (REQUIRED - do this FOURTH, AFTER age)
STEP 5: THEN and ONLY THEN ask about the symptom using OLD CARTS - one question at a time

DO NOT:
- Skip empathy, chronicity, age, or sex questions
- Ask OLD CARTS questions before completing steps 1-4
- Ask redundant questions about information already provided
- Make statements instead of asking questions

When asking OLD CARTS questions, ask about: when it started, where it is, how long it's been present, what it feels like, what makes it worse, what makes it better, if it spreads, if it's constant or comes and goes, and how severe it is.

Be natural and conversational. Ask only one question at a time. Do not list multiple questions. Do not mention frameworks or include instructions in your responses. Do not include internal reasoning, acknowledgments, or explanations. Only ask the question."""
    
    conversation = [
        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT}
    ]
    
    # Session state tracking (similar to advanced_medical_navigator)
    session_state = {
        'stage': 'awaiting_chief_complaint',
        'chief_complaint': None,
        'chronicity': None,
        'age': None,
        'sex': None,
        'oldcarts': {},
        'last_question_type': None,
    }
    
    turn = 0
    show_reasoning = False  # Toggle for showing internal reasoning
    show_differential = False  # Toggle for showing differential diagnosis
    
    # Load guidelines for differential diagnosis
    print("📚 Loading medical guidelines for differential diagnosis...")
    guidelines = load_guidelines()
    if guidelines:
        print(f"✅ Loaded {len(guidelines)} guidelines")
    else:
        print("⚠️  No guidelines found. Differential diagnosis will be limited.")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                conversation = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
                session_state = {
                    'stage': 'awaiting_chief_complaint',
                    'chief_complaint': None,
                    'chronicity': None,
                    'age': None,
                    'sex': None,
                    'oldcarts': {},
                    'last_question_type': None,
                }
                turn = 0
                print("\n🔄 Conversation reset. Starting fresh...")
                continue
            
            if user_input.lower() == 'show':
                print("\n📜 Conversation History:")
                print("-" * 80)
                for i, msg in enumerate(conversation[1:], 1):  # Skip system prompt
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    content = msg['content']
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"{i}. {role_icon} {msg['role'].upper()}: {content}")
                continue
            
            if user_input.lower() == 'sequence':
                seq = track_sequence(conversation)
                print("\n📊 Sequence Progress:")
                print("-" * 80)
                print(f"  {'✅' if seq['empathy'] else '❌'} Empathy")
                print(f"  {'✅' if seq['chronicity'] else '❌'} Chronicity")
                print(f"  {'✅' if seq['age'] else '❌'} Age")
                print(f"  {'✅' if seq['sex'] else '❌'} Biological Sex")
                print(f"  {'✅' if seq['oldcarts'] else '❌'} OLD CARTS")
                print()
                continue
            
            if user_input.lower() in ['debug', 'reasoning']:
                show_reasoning = not show_reasoning
                status = "ENABLED" if show_reasoning else "DISABLED"
                print(f"\n🔍 Internal reasoning display: {status}")
                print("   (Raw model output will be shown before cleaning)")
                continue
            
            if user_input.lower() in ['differential', 'diagnosis']:
                show_differential = not show_differential
                status = "ENABLED" if show_differential else "DISABLED"
                print(f"\n🔍 Differential diagnosis reasoning: {status}")
                print("   (Shows condition scores and reasoning after each answer)")
                continue
            
            if user_input.lower() == 'prompt':
                print("\n📝 Full Prompt Being Sent to Model:")
                print("=" * 80)
                if tokenizer:
                    formatted = format_chat_template(conversation, tokenizer)
                else:
                    formatted = format_chat_template(conversation, None)
                print(formatted)
                print("=" * 80)
                continue
            
            # Update session state based on user input
            user_lower = user_input.lower()
            stage = session_state.get('stage', 'awaiting_chief_complaint')
            
            # Extract chief complaint if in initial stage
            if stage == 'awaiting_chief_complaint':
                medical_keywords = [
                    'pain', 'ache', 'hurt', 'sore', 'symptom', 'problem', 'issue',
                    'fever', 'cough', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue',
                    'bleeding', 'rash', 'swelling', 'discomfort', 'tender', 'stiff',
                    'chest', 'head', 'stomach', 'abdominal', 'back', 'arm', 'leg',
                    'breathing', 'shortness', 'difficulty', 'trouble', 'blood'
                ]
                if any(keyword in user_lower for keyword in medical_keywords):
                    session_state['chief_complaint'] = user_input
                    session_state['stage'] = 'awaiting_chronicity'
            
            # Extract structured answers and update stage
            if stage in ['awaiting_chronicity', 'awaiting_age', 'awaiting_sex']:
                answer = extract_answer_from_user_input(session_state, user_input)
                if stage == 'awaiting_chronicity':
                    session_state['chronicity'] = answer
                    session_state['stage'] = 'awaiting_age'
                elif stage == 'awaiting_age':
                    session_state['age'] = answer
                    session_state['stage'] = 'awaiting_sex'
                elif stage == 'awaiting_sex':
                    session_state['sex'] = answer
                    session_state['stage'] = 'hpi'
            
            # Check if user mentioned a medical complaint
            medical_keywords = [
                'pain', 'ache', 'hurt', 'sore', 'symptom', 'problem', 'issue',
                'fever', 'cough', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue',
                'bleeding', 'rash', 'swelling', 'discomfort', 'tender', 'stiff',
                'chest', 'head', 'stomach', 'abdominal', 'back', 'arm', 'leg',
                'breathing', 'shortness', 'difficulty', 'trouble'
            ]
            has_medical_complaint = any(keyword in user_lower for keyword in medical_keywords)
            
            # Check conversation history for medical complaint
            for msg in conversation:
                if msg.get('role') == 'user':
                    msg_lower = msg.get('content', '').lower()
                    if any(keyword in msg_lower for keyword in medical_keywords):
                        has_medical_complaint = True
                        break
            
            # Build context-aware prompt based on stage
            current_stage = session_state.get('stage', 'awaiting_chief_complaint')
            context_summary = build_context_summary(session_state)
            
            # Create stage-specific guidance
            stage_guidance = {
                'awaiting_chief_complaint': "",
                'awaiting_chronicity': f"\n[Context: {context_summary}]\n\nYou MUST ask if this is new or an ongoing problem. This is REQUIRED before asking age.",
                'awaiting_age': f"\n[Context: {context_summary}]\n\nYou MUST ask for the patient's age. This is REQUIRED after chronicity, before sex.",
                'awaiting_sex': f"\n[Context: {context_summary}]\n\nYou MUST ask for the patient's biological sex. This is REQUIRED after age, before OLD CARTS.",
                'hpi': f"\n[Context: {context_summary}]\n\nNow ask about the symptom using OLD CARTS. Do NOT ask about information already in the context above. Ask one question at a time.",
            }
            
            # Add user message with stage guidance
            enhanced_user_input = user_input + stage_guidance.get(current_stage, "")
            conversation.append({"role": "user", "content": enhanced_user_input})
            turn += 1
            
            # Generate response
            print(f"\n📊 Turn {turn} (Stage: {current_stage})")
            print("-" * 80)
            print("🤖 Assistant: ", end="", flush=True)
            
            # If no medical complaint and user is just greeting, respond naturally
            if not has_medical_complaint and any(word in user_lower for word in ['hello', 'hi', 'hey', 'greeting']):
                # Natural greeting response
                print("Hello! How can I help you today? If you're experiencing any symptoms or have a medical concern, please let me know.")
                conversation.append({"role": "assistant", "content": "Hello! How can I help you today? If you're experiencing any symptoms or have a medical concern, please let me know."})
                continue
            
            # Show prompt if debugging
            if show_reasoning:
                print("\n" + "=" * 80)
                print("🔍 DEBUG: Full Prompt Sent to Model")
                print("=" * 80)
                if tokenizer:
                    formatted = format_chat_template(conversation, tokenizer)
                else:
                    formatted = format_chat_template(conversation, None)
                print(formatted)
                print("=" * 80)
                print()
            
            response = generate_response(
                model,
                tokenizer,
                model_type,
                conversation,
                max_tokens=150,
                temperature=0.7,
            )
            
            # Show raw response if debugging
            if show_reasoning:
                print("\n" + "=" * 80)
                print("🔍 DEBUG: Raw Model Response (Before Cleaning)")
                print("=" * 80)
                print(response)
                print("=" * 80)
                print()
            
            # Clean response
            cleaned = clean_response(response)
            if not cleaned:
                cleaned = response.strip()
            
            # Show cleaned response if debugging
            if show_reasoning:
                print("\n" + "=" * 80)
                print("🔍 DEBUG: Cleaned Response (What User Sees)")
                print("=" * 80)
                print(cleaned)
                print("=" * 80)
                print()
            
            # Check for redundant questions
            if is_redundant_question(session_state, cleaned):
                print("\n⚠️  WARNING: Model asked a redundant question! Correcting...")
                # Force correct question based on stage
                if current_stage == 'awaiting_chronicity' and not session_state.get('chronicity'):
                    cleaned = "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                elif current_stage == 'awaiting_age' and not session_state.get('age'):
                    cleaned = "How old are you?"
                elif current_stage == 'awaiting_sex' and not session_state.get('sex'):
                    cleaned = "What is your biological sex?"
            
            # Additional check: if response makes assumptions, convert to question
            if cleaned.startswith(('Your ', 'You are ', 'You have ')):
                # Try to convert to question
                if 'age' in cleaned.lower():
                    cleaned = "How old are you?"
                elif 'sex' in cleaned.lower() or 'gender' in cleaned.lower():
                    cleaned = "What is your biological sex?"
                else:
                    # Generic: ask what they need help with
                    cleaned = "How can I help you today? If you have a medical concern, please describe your symptoms."
            
            print(cleaned)
            
            # Add assistant response
            conversation.append({"role": "assistant", "content": cleaned})
            
            # Update session state based on what was asked
            cleaned_lower = cleaned.lower()
            if current_stage == 'awaiting_chronicity' and ('new' in cleaned_lower or 'ongoing' in cleaned_lower):
                session_state['last_question_type'] = 'chronicity'
            elif current_stage == 'awaiting_age' and ('age' in cleaned_lower or 'old' in cleaned_lower):
                session_state['last_question_type'] = 'age'
            elif current_stage == 'awaiting_sex' and ('sex' in cleaned_lower or 'biological' in cleaned_lower):
                session_state['last_question_type'] = 'sex'
            elif current_stage == 'hpi':
                session_state['last_question_type'] = 'oldcarts'
            
            # Show differential diagnosis reasoning if enabled
            if show_differential and guidelines:
                diff_reasoning = generate_differential_reasoning(conversation, guidelines)
                if diff_reasoning:
                    print(diff_reasoning)
            
            # Show sequence progress after each assistant response
            seq = track_sequence(conversation)
            if not all(seq.values()):
                print(f"\n📊 Progress: {'✅' if seq['empathy'] else '⏳'} Empathy | "
                      f"{'✅' if seq['chronicity'] else '⏳'} Chronicity | "
                      f"{'✅' if seq['age'] else '⏳'} Age | "
                      f"{'✅' if seq['sex'] else '⏳'} Sex | "
                      f"{'✅' if seq['oldcarts'] else '⏳'} OLD CARTS")
            
            # Keep conversation manageable
            if len(conversation) > 20:
                # Keep system prompt and last 18 messages
                conversation = [conversation[0]] + conversation[-18:]
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Medical Model Interactive Tester")
    print("=" * 80)
    print()
    
    # Install dependencies
    print("📦 Checking dependencies...")
    install_dependencies()
    print()
    
    # Load model
    print("=" * 80)
    print("Loading Model")
    print("=" * 80)
    print()
    
    result = load_model()
    if result is None:
        print("\n❌ Could not load model. Please check:")
        print("   1. Model files are uploaded to Colab")
        print("   2. Model path is correct")
        print("   3. Model format is supported (Unsloth, HuggingFace, or GGUF)")
        return
    
    model, tokenizer, model_type = result
    print(f"\n✅ Model loaded successfully (type: {model_type})")
    print()
    
    # Start interactive testing
    interactive_test(model, tokenizer, model_type)

if __name__ == "__main__":
    main()

