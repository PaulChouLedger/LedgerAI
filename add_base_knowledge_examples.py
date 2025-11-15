#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Examples That Leverage Base Model Medical Knowledge

This script adds examples that explicitly show the model using its base medical knowledge
(e.g., understanding anatomical relationships, medical terminology, clinical patterns)
while following the conversation format.
"""

import json
import os

def create_base_knowledge_examples():
    """Create examples that leverage base model's medical knowledge."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional with extensive medical knowledge. "
                        "You understand anatomy, medical terminology, and clinical patterns. "
                        "Use your medical knowledge to understand patient descriptions and make clinical connections. "
                        "For example, 'right upper quadrant pain' is anatomically related to liver, gallbladder, and biliary system. "
                        "'Left lower quadrant pain' is related to sigmoid colon, left ovary, and left fallopian tube. "
                        "Use this knowledge to guide your questions and reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": "I have right upper quadrant pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing right upper quadrant pain. I'm here to help."
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
                    "content": "When did the right upper quadrant pain start?"
                },
                {
                    "role": "user",
                    "content": "started a few hours ago after eating a fatty meal"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Onset (O) element. "
                        "The patient's answer 'started a few hours ago after eating a fatty meal' provides important clinical information. "
                        "Right upper quadrant pain that starts after fatty meals is CLASSICALLY associated with gallbladder issues "
                        "(Acute Cholecystitis, Biliary Colic). The timing and trigger (fatty meal) are key clinical features. "
                        "This pattern strongly supports gallbladder/biliary conditions over other causes of RUQ pain.\n\n"
                        "RULED IN (increased probability):\n"
                        "• Acute Cholecystitis: Onset (O) 'started a few hours ago after eating a fatty meal' is CLASSIC for gallbladder disease. Likelihood increased to 70%.\n"
                        "• Biliary Colic: Fatty meal trigger is characteristic. Likelihood increased to 60%.\n\n"
                        "RULED OUT (decreased probability):\n"
                        "• Acute Appendicitis: RUQ location and fatty meal trigger are NOT characteristic. Likelihood decreased to 10%.\n"
                        "• Acute Gastroenteritis: Fatty meal trigger is not typical. Likelihood decreased to 15%.\n\n"
                        "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
                        "1. Acute Cholecystitis: 70% probability (MOST PROBABLE)\n"
                        "   → Key finding: RUQ pain + fatty meal trigger is CLASSIC for Acute Cholecystitis\n"
                        "2. Biliary Colic: 60% probability\n"
                        "   → Key finding: Fatty meal trigger is characteristic of biliary issues\n"
                        "3. Acute Cholangitis: 30% probability (less likely)\n\n"
                        "NEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                    )
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the right upper quadrant pain located?"
                },
                {
                    "role": "user",
                    "content": "right upper quadrant, under my ribs"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Location (L) element. "
                        "The patient confirmed 'right upper quadrant, under my ribs' which anatomically corresponds to: "
                        "liver (right lobe), gallbladder, bile ducts, and duodenum. "
                        "This location, combined with the fatty meal trigger, strongly supports biliary/gallbladder pathology. "
                        "The anatomical location is consistent with Acute Cholecystitis.\n\n"
                        "RULED IN (increased probability):\n"
                        "• Acute Cholecystitis: Location (L) 'right upper quadrant, under my ribs' + fatty meal trigger is CLASSIC. Likelihood increased to 85%.\n\n"
                        "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
                        "1. Acute Cholecystitis: 85% probability (MOST PROBABLE)\n"
                        "   → Key finding: RUQ location + fatty meal trigger is CLASSIC for Acute Cholecystitis\n"
                        "2. Biliary Colic: 50% probability\n"
                        "3. Acute Cholangitis: 25% probability\n\n"
                        "NEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional with extensive medical knowledge. "
                        "Use your understanding of anatomy and medical terminology to make clinical connections. "
                        "For example, 'epigastric pain' is related to stomach, pancreas, and upper GI tract. "
                        "'Left lower quadrant pain' is related to sigmoid colon, left ovary, and left fallopian tube. "
                        "Use this knowledge to guide your clinical reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": "I have epigastric pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing epigastric pain. I'm here to help."
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
                    "content": "55"
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
                    "content": "When did the epigastric pain start?"
                },
                {
                    "role": "user",
                    "content": "started suddenly about 2 hours ago"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Onset (O) element. "
                        "Epigastric pain that starts suddenly is concerning and can indicate: "
                        "peptic ulcer disease, acute pancreatitis, gastroesophageal reflux, or upper GI issues. "
                        "The sudden onset in the epigastric region (anatomically: stomach, pancreas, upper GI) suggests acute pathology. "
                        "This pattern supports acute upper GI conditions.\n\n"
                        "RULED IN (increased probability):\n"
                        "• Peptic Ulcer Disease: Epigastric location + sudden onset is characteristic. Likelihood increased to 50%.\n"
                        "• Acute Pancreatitis: Epigastric location + sudden onset can indicate pancreatitis. Likelihood increased to 40%.\n\n"
                        "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
                        "1. Peptic Ulcer Disease: 50% probability (MOST PROBABLE)\n"
                        "   → Key finding: Epigastric location + sudden onset is characteristic\n"
                        "2. Acute Pancreatitis: 40% probability\n"
                        "   → Key finding: Epigastric location is consistent with pancreatic pathology\n"
                        "3. Gastroesophageal Reflux Disease (GERD): 30% probability\n\n"
                        "NEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Use your medical knowledge to understand anatomical relationships. "
                        "For example, 'left lower quadrant pain' anatomically relates to sigmoid colon, left ovary, left fallopian tube. "
                        "'Right lower quadrant pain' relates to appendix, right ovary, right fallopian tube. "
                        "Use this knowledge in your clinical reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": "I have left lower quadrant pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing left lower quadrant pain. I'm here to help."
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
                    "content": "28"
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
                    "content": "When did the left lower quadrant pain start?"
                },
                {
                    "role": "user",
                    "content": "started yesterday"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Onset (O) element. "
                        "Left lower quadrant pain that started yesterday suggests: "
                        "diverticulitis (sigmoid colon is in LLQ), ovarian issues (left ovary), or gynecological conditions. "
                        "The LLQ location anatomically corresponds to sigmoid colon, left ovary, and left fallopian tube. "
                        "This pattern supports conditions affecting these structures.\n\n"
                        "RULED IN (increased probability):\n"
                        "• Acute Diverticulitis: LLQ location + recent onset is CLASSIC for sigmoid diverticulitis. Likelihood increased to 60%.\n"
                        "• Ovarian Torsion: LLQ location in a 28-year-old female + recent onset is concerning. Likelihood increased to 40%.\n\n"
                        "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
                        "1. Acute Diverticulitis: 60% probability (MOST PROBABLE)\n"
                        "   → Key finding: LLQ location is CLASSIC for sigmoid diverticulitis\n"
                        "2. Ovarian Torsion: 40% probability\n"
                        "   → Key finding: LLQ location + female + recent onset is concerning for ovarian pathology\n"
                        "3. Acute Appendicitis: 10% probability (less likely - typically RLQ)\n\n"
                        "NEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Use your medical knowledge to understand clinical patterns. "
                        "For example, 'substernal chest pain' relates to cardiac, esophageal, or mediastinal structures. "
                        "'Pleuritic chest pain' (worse with breathing) relates to pleural, pulmonary, or chest wall issues. "
                        "Use this knowledge to guide your differential diagnosis."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain that's worse when I breathe"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain that's worse when you breathe. I'm here to help."
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
                    "content": "35"
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
                    "content": "started this morning"
                },
                {
                    "role": "assistant",
                    "content": (
                        "CLINICAL REASONING: This is the Onset (O) element. "
                        "Chest pain that's worse with breathing (pleuritic) suggests: "
                        "pleural inflammation, pulmonary embolism, pneumonia, or chest wall issues. "
                        "Pleuritic pain is a key clinical feature that helps differentiate cardiac from pulmonary causes. "
                        "This pattern supports pulmonary/pleural conditions over cardiac causes.\n\n"
                        "RULED IN (increased probability):\n"
                        "• Pulmonary Embolism: Pleuritic chest pain (worse with breathing) is a CLASSIC feature. Likelihood increased to 50%.\n"
                        "• Pneumonia: Pleuritic pain can occur with pneumonia. Likelihood increased to 40%.\n"
                        "• Pleuritis: Pleuritic pain is the defining feature. Likelihood increased to 45%.\n\n"
                        "RULED OUT (decreased probability):\n"
                        "• Acute Myocardial Infarction: Pleuritic pain is NOT typical for MI (MI pain is usually pressure, not worse with breathing). Likelihood decreased to 20%.\n\n"
                        "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
                        "1. Pleuritis: 45% probability (MOST PROBABLE)\n"
                        "   → Key finding: Pleuritic pain (worse with breathing) is CLASSIC for pleural inflammation\n"
                        "2. Pulmonary Embolism: 50% probability\n"
                        "   → Key finding: Pleuritic pain is a key feature of PE\n"
                        "3. Pneumonia: 40% probability\n\n"
                        "NEXT STEP: Continue OLD CARTS assessment to further narrow differential."
                    )
                }
            ]
        }
    ]
    
    return examples

def enhance_dataset_with_base_knowledge(input_path, output_path):
    """Add base knowledge examples to the dataset."""
    print(f"📚 Loading dataset: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create new examples
    base_knowledge_examples = create_base_knowledge_examples()
    
    # Combine
    enhanced = dataset + base_knowledge_examples
    
    print(f"📝 Added {len(base_knowledge_examples)} base knowledge examples")
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
    enhance_dataset_with_base_knowledge(input_path, output_path)
    
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Added examples that:")
    print(f"  ✅ Leverage base model's medical knowledge (anatomy, terminology)")
    print(f"  ✅ Show anatomical relationships (RUQ → liver/gallbladder, LLQ → sigmoid/ovary)")
    print(f"  ✅ Use clinical patterns (pleuritic pain → pulmonary, fatty meal → gallbladder)")
    print(f"  ✅ Demonstrate using medical knowledge in reasoning")
    print(f"\n📝 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Retrain model - it should now leverage base medical knowledge better")
    print(f"3. Test with anatomical terms (RUQ, LLQ, epigastric, pleuritic)")

