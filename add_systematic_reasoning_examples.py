#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Examples That Teach Systematic Reasoning Structure

The base LLM has medical knowledge but lacks STRUCTURE to:
1. Ask questions systematically (OLD CARTS framework)
2. Reason methodically through differential diagnosis
3. Build diagnosis step-by-step using collected information

This script adds examples that explicitly teach the STRUCTURE and SYSTEMATIC REASONING,
not medical facts (which the model already knows).
"""

import json
import os

def create_systematic_reasoning_examples():
    """Create examples that teach systematic reasoning structure."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional conducting a clinical history. "
                        "You have extensive medical knowledge. Your job is to STRUCTURE the conversation "
                        "and REASON SYSTEMATICALLY to arrive at a diagnosis.\n\n"
                        "SYSTEMATIC APPROACH:\n"
                        "1. Collect information systematically using OLD CARTS framework (one element at a time)\n"
                        "2. After EACH answer, reason INTERNALLY: How does this affect the differential?\n"
                        "3. Build diagnosis step-by-step, ruling in/out conditions based on collected information\n"
                        "4. Use your medical knowledge, but apply it in a STRUCTURED way\n"
                        "5. Ask the NEXT question - don't explain your reasoning to the patient\n\n"
                        "CRITICAL: You know the medical facts. Focus on STRUCTURE and SYSTEMATIC REASONING. "
                        "Reason internally, then ask the next question."
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
                    "content": "58"
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
                    "content": "When did the chest pain start?"
                },
                {
                    "role": "user",
                    "content": "started suddenly about an hour ago"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the chest pain located?"
                },
                {
                    "role": "user",
                    "content": "center of my chest, feels like pressure"
                },
                {
                    "role": "assistant",
                    "content": "How long has the chest pain been present?"
                },
                {
                    "role": "user",
                    "content": "it's been constant since it started"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain worse?"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You have medical knowledge. "
                        "Focus on STRUCTURE: ask questions systematically, reason methodically.\n\n"
                        "SYSTEMATIC REASONING STRUCTURE:\n"
                        "1. Collect ONE OLD CARTS element at a time\n"
                        "2. After EACH element, reason INTERNALLY: How does this affect the differential?\n"
                        "3. Build diagnosis step-by-step: Rule IN conditions that match, Rule OUT conditions that don't\n"
                        "4. Progressively narrow: Each answer should refine your differential\n"
                        "5. Use your medical knowledge, but apply it SYSTEMATICALLY\n"
                        "6. Ask the NEXT question - don't explain your reasoning to the patient"
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
                    "content": "25"
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
                    "content": "When did the abdominal pain start?"
                },
                {
                    "role": "user",
                    "content": "started a few hours ago"
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
                },
                {
                    "role": "user",
                    "content": "sharp"
                },
                {
                    "role": "assistant",
                    "content": "How long has the abdominal pain been present?"
                }
            ]
        }
    ]
    
    return examples

def enhance_dataset_with_systematic_reasoning(input_path, output_path):
    """Add systematic reasoning examples to the dataset."""
    print(f"📚 Loading dataset: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create new examples
    systematic_examples = create_systematic_reasoning_examples()
    
    # Combine
    enhanced = dataset + systematic_examples
    
    print(f"📝 Added {len(systematic_examples)} systematic reasoning examples")
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
    enhance_dataset_with_systematic_reasoning(input_path, output_path)
    
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Added examples that teach:")
    print(f"  ✅ SYSTEMATIC REASONING STRUCTURE (step-by-step process)")
    print(f"  ✅ How to collect information methodically (one element at a time)")
    print(f"  ✅ How to reason after each answer (rule in/out conditions)")
    print(f"  ✅ How to build diagnosis progressively (narrow differential step-by-step)")
    print(f"  ✅ How to apply medical knowledge in a STRUCTURED way")
    print(f"\n💡 Key Insight: Model has medical knowledge, needs STRUCTURE to apply it systematically")
    print(f"\n📝 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Retrain model - it should now reason more systematically")
    print(f"3. Test systematic reasoning flow")

