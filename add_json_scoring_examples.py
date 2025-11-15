#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add JSON Scoring Examples to Training Dataset

This script adds specific examples showing the model how to return JSON
for condition scoring tasks, which is critical for the scoring function.
"""

import json
import os

def create_json_scoring_examples():
    """Create examples showing JSON-only responses for condition scoring."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
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
                },
                {
                    "role": "user",
                    "content": (
                        "Chief complaint: chest pain\n"
                        "OLD CARTS element: location\n"
                        "Patient's answer: 'center of my chest'\n\n"
                        "All conditions to evaluate (5 total):\n"
                        "Acute Myocardial Infarction (Heart Attack), Unstable Angina, Stable Angina, Aortic Dissection, Pericarditis\n\n"
                        "Return ONLY valid JSON with ALL 5 conditions as keys and their score changes as values.\n"
                        "Format: {\"condition_name\": score_change, \"condition_name\": score_change, ...}\n"
                        "Example (not actual conditions): {\"Condition1\": 0.2, \"Condition2\": -0.1, \"Condition3\": 0.0}\n\n"
                        "CRITICAL: Return ONLY the JSON object. No explanations, no text before or after. "
                        "Every condition listed above must be a key in the JSON with a numeric score between -0.3 and +0.3."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\"Acute Myocardial Infarction (Heart Attack)\": 0.2, \"Unstable Angina\": 0.1, "
                        "\"Stable Angina\": 0.0, \"Aortic Dissection\": -0.2, \"Pericarditis\": -0.1}"
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical expert with extensive training in clinical reasoning. "
                        "You MUST return ONLY valid JSON. No explanations, no text before or after the JSON.\n\n"
                        "CRITICAL FORMAT REQUIREMENTS:\n"
                        "- Output ONLY valid JSON (no explanations, no text before or after)\n"
                        "- JSON must be an object with ALL condition names as keys and numeric scores as values\n"
                        "- Scores must be between -0.3 and +0.3 (numeric values only)\n"
                        "- Do NOT use any other format - only JSON object with condition names as keys and numeric values"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Chief complaint: abdominal pain\n"
                        "OLD CARTS element: location\n"
                        "Patient's answer: 'lower right side'\n\n"
                        "All conditions to evaluate (9 total):\n"
                        "Acute Appendicitis, Acute Cholecystitis, Acute Cholangitis, Acute Diverticulitis, "
                        "Acute Gastroenteritis, Gastroesophageal Reflux Disease (GERD), Acute Upper GI Bleed, "
                        "Acute Lower GI Bleed, Bowel Obstruction\n\n"
                        "Return ONLY valid JSON with ALL 9 conditions as keys and their score changes as values."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\"Acute Appendicitis\": 0.3, \"Acute Cholecystitis\": -0.2, \"Acute Cholangitis\": -0.2, "
                        "\"Acute Diverticulitis\": 0.1, \"Acute Gastroenteritis\": -0.1, "
                        "\"Gastroesophageal Reflux Disease (GERD)\": -0.2, \"Acute Upper GI Bleed\": -0.1, "
                        "\"Acute Lower GI Bleed\": 0.0, \"Bowel Obstruction\": -0.1}"
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical expert. You MUST return ONLY valid JSON. "
                        "No explanations, no text before or after the JSON."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Chief complaint: chest pain\n"
                        "OLD CARTS element: character\n"
                        "Patient's answer: 'heavy pressure'\n\n"
                        "All conditions to evaluate (5 total):\n"
                        "Acute Myocardial Infarction (Heart Attack), Unstable Angina, Stable Angina, Aortic Dissection, Pericarditis\n\n"
                        "Return ONLY valid JSON with ALL 5 conditions as keys and their score changes as values."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\"Acute Myocardial Infarction (Heart Attack)\": 0.3, \"Unstable Angina\": 0.2, "
                        "\"Stable Angina\": 0.1, \"Aortic Dissection\": -0.1, \"Pericarditis\": -0.2}"
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical expert. You MUST return ONLY valid JSON. "
                        "No explanations, no text before or after the JSON."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Chief complaint: abdominal pain\n"
                        "OLD CARTS element: onset\n"
                        "Patient's answer: 'started a few hours ago'\n\n"
                        "All conditions to evaluate (9 total):\n"
                        "Acute Appendicitis, Acute Cholecystitis, Acute Cholangitis, Acute Diverticulitis, "
                        "Acute Gastroenteritis, Gastroesophageal Reflux Disease (GERD), Acute Upper GI Bleed, "
                        "Acute Lower GI Bleed, Bowel Obstruction\n\n"
                        "Return ONLY valid JSON with ALL 9 conditions as keys and their score changes as values."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\"Acute Appendicitis\": 0.2, \"Acute Cholecystitis\": 0.1, \"Acute Cholangitis\": 0.1, "
                        "\"Acute Diverticulitis\": 0.0, \"Acute Gastroenteritis\": 0.1, "
                        "\"Gastroesophageal Reflux Disease (GERD)\": -0.2, \"Acute Upper GI Bleed\": 0.0, "
                        "\"Acute Lower GI Bleed\": 0.0, \"Bowel Obstruction\": 0.0}"
                    )
                }
            ]
        }
    ]
    
    return examples

def create_no_repeat_demographic_examples():
    """Create examples showing NOT to repeat demographic questions."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional conducting a clinical history. "
                        "CRITICAL: After collecting demographics (age, biological sex), you MUST ONLY ask OLD CARTS questions. "
                        "DO NOT ask about age, biological sex, or demographics again during HPI. "
                        "The sequence is: empathy → chronicity → age → sex → OLD CARTS questions (onset, location, duration, character, etc.). "
                        "Once you've asked age and sex, NEVER ask them again."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "It's new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "45"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "When did the chest pain start?"
                },
                {
                    "role": "user",
                    "content": "2 days ago"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "CRITICAL: After asking age and biological sex once, NEVER ask them again. "
                        "Only ask OLD CARTS questions about the symptom. "
                        "DO NOT repeat 'How old are you?' or 'What is your biological sex?' after they've been answered."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "30"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "When did the abdominal pain start?"
                },
                {
                    "role": "user",
                    "content": "a few hours ago"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "lower right side"
                },
                {
                    "role": "assistant",
                    "content": "What does the abdominal pain feel like? For example, is it sharp, heavy, burning, or pressure?"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "CRITICAL RULE: Ask age ONCE, ask biological sex ONCE, then ONLY ask OLD CARTS questions. "
                        "NEVER ask 'How old are you?' or 'What is your biological sex?' after they've been answered. "
                        "If you've already asked age and sex, the next question MUST be an OLD CARTS question (onset, location, duration, character, etc.)."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "45"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "When did the chest pain start?"
                },
                {
                    "role": "user",
                    "content": "2 days ago"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                },
                {
                    "role": "user",
                    "content": "center of my chest"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like? For example, is it sharp, heavy, burning, or pressure?"
                }
            ]
        }
    ]
    
    return examples

def create_element_detection_examples():
    """Create examples showing correct element identification."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "When a patient answers a question, you must correctly identify which OLD CARTS element their answer addresses. "
                        "CRITICAL: 'left side' or 'center of chest' = Location (L), NOT Onset (O). "
                        "'started 2 days ago' or 'began yesterday' = Onset (O), NOT Location (L). "
                        "'sharp' or 'burning' or 'pressure' = Character (C), NOT Location (L) or Onset (O)."
                    )
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                },
                {
                    "role": "user",
                    "content": "left side"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Location (L) - where the symptom is located element of OLD CARTS. "
                        "ELEMENT IDENTIFICATION: The patient's answer 'left side' is being evaluated for the Location (L) element. "
                        "CRITICAL: This is Location (L), NOT Onset (O), NOT Duration (D), NOT Character (C)."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "Correctly identify OLD CARTS elements from patient answers. "
                        "Location answers: 'left side', 'center', 'lower right', 'upper abdomen' = Location (L). "
                        "Onset answers: 'started yesterday', 'began 2 hours ago', 'came on suddenly' = Onset (O). "
                        "Character answers: 'sharp', 'burning', 'pressure', 'dull' = Character (C)."
                    )
                },
                {
                    "role": "assistant",
                    "content": "When did the chest pain start?"
                },
                {
                    "role": "user",
                    "content": "2 days ago"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Onset (O) - when the symptom started element of OLD CARTS. "
                        "ELEMENT IDENTIFICATION: The patient's answer '2 days ago' is being evaluated for the Onset (O) element. "
                        "CRITICAL: This is Onset (O), NOT Location (L), NOT Duration (D), NOT Character (C)."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical professional. Correctly identify OLD CARTS elements."
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like? For example, is it sharp, heavy, burning, or pressure?"
                },
                {
                    "role": "user",
                    "content": "burning"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Character (C) - what the symptom feels like element of OLD CARTS. "
                        "ELEMENT IDENTIFICATION: The patient's answer 'burning' is being evaluated for the Character (C) element. "
                        "CRITICAL: This is Character (C), NOT Location (L), NOT Onset (O), NOT Duration (D)."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "CRITICAL ELEMENT IDENTIFICATION RULES:\n"
                        "- If you ask 'Where exactly is [symptom] located?', the answer is ALWAYS Location (L)\n"
                        "- If you ask 'When did [symptom] start?', the answer is ALWAYS Onset (O)\n"
                        "- If you ask 'What does [symptom] feel like?', the answer is ALWAYS Character (C)\n"
                        "- Answers like 'left side', 'center', 'lower right' = Location (L)\n"
                        "- Answers like '2 days ago', 'yesterday', 'started suddenly' = Onset (O)\n"
                        "- Answers like 'sharp', 'burning', 'pressure' = Character (C)"
                    )
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                },
                {
                    "role": "user",
                    "content": "center of my chest"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Location (L) element. "
                        "The question was 'Where exactly is the chest pain located?' "
                        "The answer 'center of my chest' describes WHERE the pain is, so this is Location (L), NOT Onset (O)."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. "
                        "When you ask 'Where exactly is [symptom] located?', the answer is ALWAYS Location (L), even if the answer contains words like 'started' or 'ago'. "
                        "The question determines the element, not the answer text alone."
                    )
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                },
                {
                    "role": "user",
                    "content": "left side"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is Location (L). "
                        "I asked 'Where exactly is the chest pain located?' so the answer 'left side' is Location (L), NOT Onset (O)."
                    )
                }
            ]
        }
    ]
    
    return examples

def enhance_dataset_with_json_examples(input_path, output_path):
    """Add JSON scoring examples to the dataset."""
    print(f"📚 Loading dataset: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create new examples
    json_examples = create_json_scoring_examples()
    no_repeat_examples = create_no_repeat_demographic_examples()
    element_examples = create_element_detection_examples()
    
    # Combine
    enhanced = dataset + json_examples + no_repeat_examples + element_examples
    
    print(f"📝 Added {len(json_examples)} JSON scoring examples")
    print(f"📝 Added {len(no_repeat_examples)} no-repeat demographic examples")
    print(f"📝 Added {len(element_examples)} element detection examples")
    print(f"📊 Total conversations: {len(enhanced)}")
    
    # Save
    print(f"💾 Saving enhanced dataset: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced dataset saved!")
    return enhanced

if __name__ == "__main__":
    # Try enhanced first, then high_quality, then fallback
    input_paths = [
        "medical_sft_dataset_enhanced.json",
        "medical_sft_dataset_high_quality.json",
        "medical_sft_dataset.json"
    ]
    
    output_path = "medical_sft_dataset_enhanced.json"
    
    input_path = None
    for path in input_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if not input_path:
        print(f"❌ Error: No dataset file found. Tried: {input_paths}")
        exit(1)
    
    print(f"📂 Using input dataset: {input_path}")
    enhance_dataset_with_json_examples(input_path, output_path)
    
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Added examples for:")
    print(f"  ✅ JSON scoring format (critical for condition scoring)")
    print(f"  ✅ No repeated demographic questions")
    print(f"  ✅ Correct element detection (location vs onset vs character)")
    print(f"\n📝 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Retrain model with enhanced dataset")
    print(f"3. Test JSON scoring and element detection")

