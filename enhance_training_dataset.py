#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance Training Dataset with Negative Examples and Better Question Patterns

This script adds:
1. Negative examples (what NOT to ask)
2. More explicit OLD CARTS question format examples
3. Clearer separation between demographics and HPI questions
"""

import json
import os

def create_negative_examples():
    """Create training examples showing what NOT to ask"""
    negative_examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning.\n\nCRITICAL: After collecting demographics (age, biological sex), you MUST only ask OLD CARTS questions. Do NOT ask about age, biological sex, or demographics again during HPI."
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
                },
                {
                    "role": "user",
                    "content": "center of my chest"
                },
                {
                    "role": "assistant",
                    "content": "How long has the chest pain been present?"
                },
                {
                    "role": "user",
                    "content": "for the past 2 days"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like? For example, is it sharp, heavy, burning, or pressure?"
                }
            ]
        }
    ]
    return negative_examples

def create_question_format_examples():
    """Create examples showing correct OLD CARTS question formats"""
    question_examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning.\n\nCRITICAL QUESTION FORMATS:\n- Onset (O): 'When did [symptom] start?' or 'When did it start?'\n- Location (L): 'Where exactly is the [symptom] located?'\n- Duration (D): 'How long has the [symptom] been present?'\n- Character (C): 'What does the [symptom] feel like? For example, is it sharp, heavy, burning, or pressure?'\n- Aggravating (A): 'What makes the [symptom] worse?'\n- Alleviating (A): 'What makes the [symptom] better?'\n- Radiation (R): 'Does the [symptom] spread to other areas?'\n- Timing (T): 'Is the [symptom] constant or does it come and go?'\n- Severity (S): 'On a scale from 1 to 10, how severe is the [symptom]?'\n\nDO NOT ask about age, biological sex, or demographics during HPI. These are only asked once at the beginning."
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
                    "content": "It's new"
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
                    "content": "started a few hours ago"
                },
                {
                    "role": "assistant",
                    "content": "CLINICAL REASONING: This is the Onset (O) - when the symptom started element of OLD CARTS.\nELEMENT IDENTIFICATION: The patient's answer 'started a few hours ago' is being evaluated for the Onset (O) element.\nCRITICAL: This is Onset (O), NOT any other OLD CARTS element (NOT Location (L), NOT Duration (D), NOT Character (C)...).\nPatient reported 'started a few hours ago' for ONSET (O). This onset pattern supports Acute Appendicitis over Chronic conditions.\n\nRULED IN (increased probability):\n• Acute Appendicitis: Onset (O) 'started a few hours ago' matches clinical pattern. Likelihood increased to 35%.\n\nCURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n1. Acute Appendicitis: 35% probability (MOST PROBABLE)\n   → Key finding: Onset (O) 'started a few hours ago' is CLASSIC for Acute Appendicitis\n2. Acute Cholecystitis: 20% probability (less likely)\n3. Bowel Obstruction: 15% probability (less likely)\n\nNEXT STEP: Continue OLD CARTS assessment to further narrow differential."
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
                    "content": "CLINICAL REASONING: This is the Location (L) - where the symptom is located element of OLD CARTS.\nELEMENT IDENTIFICATION: The patient's answer 'lower right side' is being evaluated for the Location (L) element.\nCRITICAL: This is Location (L), NOT any other OLD CARTS element (NOT Onset (O), NOT Duration (D), NOT Character (C)...).\nPatient reported 'lower right side' for LOCATION (L). This location pattern is CLASSIC for Acute Appendicitis (RLQ).\n\nRULED IN (increased probability):\n• Acute Appendicitis: Location (L) 'lower right side' matches clinical pattern. Likelihood increased to 60%.\n\nCURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n1. Acute Appendicitis: 60% probability (MOST PROBABLE)\n   → Key finding: Location (L) 'lower right side' is CLASSIC for Acute Appendicitis (RLQ)\n2. Acute Cholecystitis: 15% probability (less likely - RUQ location)\n3. Bowel Obstruction: 10% probability (less likely)\n\nNEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                },
                {
                    "role": "assistant",
                    "content": "How long has the abdominal pain been present?"
                },
                {
                    "role": "user",
                    "content": "for the past few hours"
                },
                {
                    "role": "assistant",
                    "content": "CLINICAL REASONING: This is the Duration (D) - how long the symptom has been present element of OLD CARTS.\nELEMENT IDENTIFICATION: The patient's answer 'for the past few hours' is being evaluated for the Duration (D) element.\nCRITICAL: This is Duration (D), NOT any other OLD CARTS element (NOT Onset (O), NOT Location (L), NOT Character (C)...).\nPatient reported 'for the past few hours' for DURATION (D). This duration pattern supports Acute Appendicitis.\n\nRULED IN (increased probability):\n• Acute Appendicitis: Duration (D) 'for the past few hours' matches clinical pattern. Likelihood increased to 70%.\n\nCURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n1. Acute Appendicitis: 70% probability (MOST PROBABLE)\n   → Key finding: Duration (D) 'for the past few hours' is CLASSIC for Acute Appendicitis\n2. Acute Cholecystitis: 10% probability (less likely)\n3. Bowel Obstruction: 5% probability (less likely)\n\nNEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                },
                {
                    "role": "assistant",
                    "content": "What does the abdominal pain feel like? For example, is it sharp, heavy, burning, or pressure?"
                }
            ]
        }
    ]
    return question_examples

def enhance_dataset(input_path, output_path):
    """Enhance existing dataset with negative examples and better question patterns"""
    print(f"📚 Loading dataset: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Add negative examples
    negative_examples = create_negative_examples()
    question_examples = create_question_format_examples()
    
    enhanced = dataset + negative_examples + question_examples
    
    print(f"📝 Added {len(negative_examples)} negative examples")
    print(f"📝 Added {len(question_examples)} question format examples")
    print(f"📊 Total conversations: {len(enhanced)}")
    
    # Save enhanced dataset
    print(f"💾 Saving enhanced dataset: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced dataset saved!")
    return enhanced

if __name__ == "__main__":
    input_path = "medical_sft_dataset_high_quality.json"
    output_path = "medical_sft_dataset_enhanced.json"
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        exit(1)
    
    enhance_dataset(input_path, output_path)
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Update train_medical_bot_colab.py to use {output_path}")
    print(f"3. Retrain the model with enhanced dataset")

