#!/usr/bin/env python3
"""
Generate high-quality medical conversation dataset for fine-tuning.
Focuses on teaching correct element identification and condition matching.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

# Configuration
GUIDELINES_DIR = "llm-medical-container/medical/guidelines"
OUTPUT_FILE = "medical_sft_dataset_high_quality.json"

# OLD CARTS element mapping with explicit names
OLD_CARTS_ELEMENTS = {
    "onset": {"name": "Onset (O)", "full_name": "Onset (O) - when the symptom started"},
    "location": {"name": "Location (L)", "full_name": "Location (L) - where the symptom is located"},
    "duration": {"name": "Duration (D)", "full_name": "Duration (D) - how long the symptom has been present"},
    "character": {"name": "Character (C)", "full_name": "Character (C) - what the symptom feels like (NOT Distribution, NOT Intensity)"},
    "aggravating": {"name": "Aggravating factors (A)", "full_name": "Aggravating factors (A) - what makes it worse"},
    "alleviating": {"name": "Alleviating factors (A)", "full_name": "Alleviating factors (A) - what makes it better"},
    "radiation": {"name": "Radiation (R)", "full_name": "Radiation (R) - where the symptom spreads"},
    "timing": {"name": "Timing (T)", "full_name": "Timing (T) - constant or intermittent"},
    "severity": {"name": "Severity (S)", "full_name": "Severity (S) - how severe the symptom is"}
}

def load_all_guidelines() -> Dict[str, List[Dict]]:
    """Load all guideline files organized by organ system."""
    guidelines = {}
    base_path = Path(GUIDELINES_DIR)
    
    if not base_path.exists():
        print(f"❌ Guidelines directory not found: {GUIDELINES_DIR}")
        return guidelines
    
    for system_dir in base_path.iterdir():
        if system_dir.is_dir():
            system_name = system_dir.name
            guidelines[system_name] = []
            
            for guideline_file in system_dir.glob("*.json"):
                try:
                    with open(guideline_file, 'r', encoding='utf-8') as f:
                        guideline = json.load(f)
                        guidelines[system_name].append(guideline)
                except Exception as e:
                    print(f"⚠️  Error loading {guideline_file}: {e}")
    
    return guidelines

def get_differential_conditions(condition_name: str, all_guidelines: Dict[str, List[Dict]]) -> List[str]:
    """Get 2-3 differential diagnoses for a condition."""
    differentials = []
    
    # Extract organ system from condition name
    condition_lower = condition_name.lower()
    
    # Find similar conditions in same organ system
    for system, conditions in all_guidelines.items():
        for cond in conditions:
            cond_name = cond.get("condition", "")
            if cond_name.lower() != condition_lower:
                # Add if same organ system or similar presentation
                if system in condition_name or any(term in cond_name.lower() for term in condition_lower.split()[:2]):
                    differentials.append(cond_name)
                    if len(differentials) >= 3:
                        break
        if len(differentials) >= 3:
            break
    
    # Add common differentials based on condition type
    if "mi" in condition_lower or "myocardial" in condition_lower:
        differentials.extend(["Unstable Angina", "Pulmonary Embolism", "Costochondritis"])
    elif "appendicitis" in condition_lower:
        differentials.extend(["Acute Cholecystitis", "Acute Gastroenteritis", "Ovarian Torsion"])
    elif "cholecystitis" in condition_lower:
        differentials.extend(["Acute Appendicitis", "Acute Cholangitis", "Hepatitis"])
    
    return list(set(differentials))[:3]

def get_patient_friendly_answer(element: str, guideline: Dict) -> Optional[str]:
    """Get a patient-friendly answer for an OLD CARTS element from guideline."""
    oldcarts = guideline.get("key_features", {}).get("structured_oldcarts", {})
    element_data = oldcarts.get(element)
    
    if not element_data:
        return None
    
    includes = element_data.get("includes", [])
    if not includes:
        return None
    
    # Prefer emergent answers if available
    emergent_answers = [inc for inc in includes if inc.get("emergent")]
    if emergent_answers:
        return random.choice(emergent_answers).get("patient_friendly")
    
    return random.choice(includes).get("patient_friendly")

def generate_reasoning_for_element(
    element: str,
    answer: str,
    target_condition: str,
    differential_conditions: List[str],
    guideline: Dict
) -> str:
    """Generate explicit clinical reasoning that teaches correct element identification and condition matching."""
    
    element_info = OLD_CARTS_ELEMENTS.get(element, {})
    element_name = element_info.get("name", element)
    element_full_name = element_info.get("full_name", element)
    
    # Get other conditions (exclude target)
    other_conditions = [c for c in differential_conditions if c != target_condition]
    if not other_conditions:
        other_conditions = ["Other conditions"]
    
    other_cond = other_conditions[0] if other_conditions else "Other conditions"
    
    # Start with EXPLICIT element identification
    all_elements = ["Onset (O)", "Location (L)", "Duration (D)", "Character (C)", "Aggravating factors (A)", "Alleviating factors (A)", "Radiation (R)", "Timing (T)", "Severity (S)"]
    other_elements = [e for e in all_elements if e != element_name]
    
    reasoning_parts = [
        f"CLINICAL REASONING: This is the {element_full_name} element of OLD CARTS.",
        f"ELEMENT IDENTIFICATION: The patient's answer '{answer}' is being evaluated for the {element_name} element.",
        f"CRITICAL: This is {element_name}, NOT any other OLD CARTS element (NOT {', NOT '.join(other_elements[:3])}...)."
    ]
    
    # Add element-specific reasoning with explicit condition matching
    if element == "onset":
        reasoning_parts.append(
            f"Patient reported '{answer}' for ONSET (O). This onset pattern is CLASSIC for {target_condition}. "
            f"The onset '{answer}' strongly supports {target_condition} over {other_cond}."
        )
    elif element == "location":
        answer_lower = answer.lower()
        if "lower right" in answer_lower or "rlq" in answer_lower:
            if "appendicitis" in target_condition.lower():
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). RLQ location is CLASSIC for {target_condition}. "
                    f"CRITICAL: {target_condition} ALWAYS presents in RLQ. {other_cond} presents in RUQ (right upper quadrant), NOT RLQ. "
                    f"This location definitively supports {target_condition}, NOT {other_cond}."
                )
            else:
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). This location pattern supports {target_condition} over {other_cond}."
                )
        elif "upper right" in answer_lower or "ruq" in answer_lower:
            if "cholecystitis" in target_condition.lower() or "cholangitis" in target_condition.lower():
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). RUQ location is CLASSIC for {target_condition}. "
                    f"CRITICAL: {target_condition} ALWAYS presents in RUQ. {other_cond} presents in RLQ (right lower quadrant), NOT RUQ. "
                    f"This location definitively supports {target_condition}, NOT {other_cond}."
                )
            else:
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). This location pattern supports {target_condition} over {other_cond}."
                )
        elif "center" in answer_lower or "central" in answer_lower or "breastbone" in answer_lower:
            if "myocardial" in target_condition.lower() or "mi" in target_condition.lower():
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). Central/retrosternal location is CLASSIC for {target_condition}. "
                    f"CRITICAL: {target_condition} typically presents in central chest. {other_cond} (if cardiac) may also present centrally, "
                    f"but the combination with other findings strongly supports {target_condition}."
                )
            else:
                reasoning_parts.append(
                    f"Patient reported '{answer}' for LOCATION (L). This location pattern supports {target_condition} over {other_cond}."
                )
        else:
            reasoning_parts.append(
                f"Patient reported '{answer}' for LOCATION (L). This location pattern supports {target_condition} over {other_cond}."
            )
    elif element == "character":
        answer_lower = answer.lower()
        reasoning_parts.append(
            f"Patient reported '{answer}' for CHARACTER (C). "
            f"CRITICAL: This is CHARACTER (what it feels like), NOT Distribution, NOT Intensity, NOT Location."
        )
        if "heavy" in answer_lower or "pressure" in answer_lower or "crushing" in answer_lower:
            if "myocardial" in target_condition.lower() or "mi" in target_condition.lower():
                reasoning_parts.append(
                    f"The character '{answer}' is CLASSIC for {target_condition}. Heavy/pressure/crushing character strongly supports "
                    f"{target_condition} over {other_cond}, which typically presents with different character (e.g., sharp for MSK, burning for GERD)."
                )
            else:
                reasoning_parts.append(
                    f"The character '{answer}' supports {target_condition} over {other_cond}."
                )
        elif "sharp" in answer_lower:
            reasoning_parts.append(
                f"The character '{answer}' is characteristic of {target_condition}. Sharp character supports "
                f"{target_condition} over {other_cond}."
            )
        else:
            reasoning_parts.append(
                f"The character '{answer}' supports {target_condition} over {other_cond}."
            )
    elif element == "duration":
        reasoning_parts.append(
            f"Patient reported '{answer}' for DURATION (D). This duration pattern supports {target_condition} over {other_cond}."
        )
    elif element == "aggravating":
        reasoning_parts.append(
            f"Patient reported '{answer}' for AGGRAVATING FACTORS (A). This aggravating factor supports {target_condition} over {other_cond}."
        )
    elif element == "alleviating":
        reasoning_parts.append(
            f"Patient reported '{answer}' for ALLEVIATING FACTORS (A). This alleviating factor supports {target_condition} over {other_cond}."
        )
    elif element == "radiation":
        reasoning_parts.append(
            f"Patient reported '{answer}' for RADIATION (R). This radiation pattern supports {target_condition} over {other_cond}."
        )
    elif element == "timing":
        reasoning_parts.append(
            f"Patient reported '{answer}' for TIMING (T). This timing pattern supports {target_condition} over {other_cond}."
        )
    elif element == "severity":
        reasoning_parts.append(
            f"Patient reported '{answer}' for SEVERITY (S). This severity level supports {target_condition} over {other_cond}."
        )
    
    # Calculate probability (increasing with each element)
    base_prob = 0.30
    element_weights = {
        "onset": 0.05, "location": 0.10, "duration": 0.05,
        "character": 0.15, "aggravating": 0.10, "alleviating": 0.10,
        "radiation": 0.10, "timing": 0.05, "severity": 0.10
    }
    new_probability = min(0.99, base_prob + element_weights.get(element, 0.05))
    
    # RULED IN section
    reasoning_parts.append(f"\nRULED IN (increased probability):")
    reasoning_parts.append(
        f"• {target_condition}: {element_name} '{answer}' matches clinical pattern. Likelihood increased to {new_probability:.0%}."
    )
    
    # CURRENT DIFFERENTIAL DIAGNOSIS with explicit ranking
    reasoning_parts.append(f"\nCURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):")
    reasoning_parts.append(f"1. {target_condition}: {new_probability:.0%} probability (MOST PROBABLE)")
    reasoning_parts.append(f"   → Key finding: {element_name} '{answer}' is CLASSIC for {target_condition}")
    reasoning_parts.append(f"   → {target_condition} is the most likely diagnosis based on this {element_name} finding")
    
    for i, cond in enumerate(other_conditions[:2], 2):
        prob = max(0.05, 0.5 - (i-1) * 0.15)
        reasoning_parts.append(f"{i}. {cond}: {prob:.0%} probability (less likely)")
        reasoning_parts.append(f"   → {cond} is less likely because this {element_name} pattern is more characteristic of {target_condition}")
    
    reasoning_parts.append(
        f"\nNEXT STEP: Continue OLD CARTS assessment to further narrow differential. {target_condition} remains the most probable diagnosis."
    )
    
    return "\n".join(reasoning_parts)

def create_conversation_for_condition(guideline: Dict, all_guidelines: Dict[str, List[Dict]]) -> Dict:
    """Create a complete conversation for a condition following the example format."""
    condition_name = guideline.get("condition", "Unknown Condition")
    chief_complaints = guideline.get("chief_complaint_triggers", [])
    
    if not chief_complaints:
        chief_complaints = ["symptoms"]
    
    chief_complaint = random.choice(chief_complaints)
    oldcarts = guideline.get("key_features", {}).get("structured_oldcarts", {})
    
    # Get differential conditions
    differential_conditions = get_differential_conditions(condition_name, all_guidelines)
    
    messages = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning."
    })
    
    # Initial user complaint
    messages.append({
        "role": "user",
        "content": f"I have {chief_complaint}"
    })
    
    # Empathy
    messages.append({
        "role": "assistant",
        "content": f"I understand you're experiencing {chief_complaint}. I'm here to help."
    })
    
    # Chronicity question
    messages.append({
        "role": "assistant",
        "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
    })
    
    # User: new/ongoing
    messages.append({
        "role": "user",
        "content": "It's new, started recently"
    })
    
    # Age question
    messages.append({
        "role": "assistant",
        "content": "How old are you?"
    })
    
    # User: age (NO REASONING)
    age = random.randint(25, 75)
    messages.append({
        "role": "user",
        "content": f"I'm {age} years old"
    })
    
    # Biological sex question
    messages.append({
        "role": "assistant",
        "content": "What is your biological sex?"
    })
    
    # User: sex (NO REASONING)
    sex = random.choice(["Male", "Female"])
    messages.append({
        "role": "user",
        "content": sex
    })
    
    # OLD CARTS sequence
    oldcarts_order = ["onset", "location", "duration", "character", "aggravating", "alleviating", "radiation", "timing", "severity"]
    
    for element in oldcarts_order:
        if element not in oldcarts:
            continue
        
        # Get question
        question_tags = oldcarts[element].get("question_tags", [])
        if "sensory" in question_tags:
            question = f"What does the {chief_complaint} feel like? For example, is it sharp, heavy, burning, or pressure?"
        elif element == "onset":
            question = "When did it start?"
        elif element == "location":
            question = f"Where exactly is the {chief_complaint} located?"
        elif element == "duration":
            question = "How long has it been present?"
        elif element == "character":
            question = f"What does the {chief_complaint} feel like? For example, is it sharp, heavy, burning, or pressure?"
        elif element == "aggravating":
            question = "What makes it worse?"
        elif element == "alleviating":
            question = "What makes it better?"
        elif element == "radiation":
            question = "Does it spread anywhere else?"
        elif element == "timing":
            question = "Is it constant or does it come and go?"
        elif element == "severity":
            question = "On a scale of 1 to 10, with 10 being the worst imaginable, how severe is it?"
        else:
            continue
        
        messages.append({
            "role": "assistant",
            "content": question
        })
        
        # Get patient answer
        answer = get_patient_friendly_answer(element, guideline)
        if not answer:
            # Fallback answer
            fallbacks = {
                "onset": "started a few hours ago",
                "location": "in the affected area",
                "duration": "for several hours",
                "character": "sharp",
                "aggravating": "movement",
                "alleviating": "rest",
                "radiation": "no",
                "timing": "constant",
                "severity": "about 7 out of 10"
            }
            answer = fallbacks.get(element, "yes")
        
        messages.append({
            "role": "user",
            "content": answer
        })
        
        # Generate reasoning
        reasoning = generate_reasoning_for_element(
            element, answer, condition_name, differential_conditions, guideline
        )
        messages.append({
            "role": "assistant",
            "content": reasoning
        })
    
    # Final diagnostic reasoning
    final_reasoning = f"""FINAL DIAGNOSTIC REASONING:

Based on complete OLD CARTS assessment, the most probable diagnosis is {condition_name} (99% probability).

CLINICAL REASONING:
1. Chief complaint recognized: {chief_complaint}
2. Initial differential considered: {', '.join([condition_name] + differential_conditions[:2])}
3. Progressive narrowing through OLD CARTS:
   • Patient's symptom pattern is more consistent with {condition_name} than {differential_conditions[0] if differential_conditions else 'other conditions'}
   • Key findings support {condition_name} over alternative diagnoses

RANKED DIFFERENTIAL DIAGNOSIS:
1. {condition_name}: 99% probability (MOST PROBABLE)
2. {differential_conditions[0] if differential_conditions else 'Other conditions'}: 10% probability
3. {differential_conditions[1] if len(differential_conditions) > 1 else 'Other conditions'}: 5% probability

CONCLUSION: The clinical presentation most strongly supports {condition_name}.
This conclusion is based on systematic collection and analysis of OLD CARTS elements, with each answer progressively ruling in {condition_name} and ruling out alternative diagnoses."""
    
    messages.append({
        "role": "assistant",
        "content": final_reasoning
    })
    
    return {"messages": messages}

def create_element_inference_examples(guideline: Dict, all_guidelines: Dict[str, List[Dict]]) -> List[Dict]:
    """Create additional training examples where model learns to infer element from answer alone."""
    condition_name = guideline.get("condition", "Unknown Condition")
    oldcarts = guideline.get("key_features", {}).get("structured_oldcarts", {})
    differential_conditions = get_differential_conditions(condition_name, all_guidelines)
    
    inference_examples = []
    
    # Create 2-3 inference examples per condition (one for each key element)
    key_elements = ["location", "character", "onset"]  # Most important for inference
    
    for element in key_elements:
        if element not in oldcarts:
            continue
        
        # Get answer
        answer = get_patient_friendly_answer(element, guideline)
        if not answer:
            fallbacks = {
                "onset": "started a few hours ago",
                "location": "in the center of my chest",
                "character": "sharp"
            }
            answer = fallbacks.get(element, "yes")
        
        # Create inference example - answer without explicit question
        messages = [
            {
                "role": "system",
                "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning.\n\nCRITICAL: You must identify which OLD CARTS element the patient's answer corresponds to based on the answer content and conversation context."
            },
            {
                "role": "user",
                "content": f"I have {condition_name.lower()}"
            },
            {
                "role": "assistant",
                "content": f"I understand you're experiencing {condition_name.lower()}. I'm here to help."
            },
            {
                "role": "user",
                "content": answer  # Answer without explicit question
            }
        ]
        
        # Generate reasoning with explicit element inference
        element_info = OLD_CARTS_ELEMENTS.get(element, {})
        element_name = element_info.get("name", element)
        element_full_name = element_info.get("full_name", element)
        
        # Inference reasoning - explicitly states how we know which element this is
        inference_reasoning = f"""CLINICAL REASONING: Based on the patient's answer '{answer}', this corresponds to the {element_full_name} element of OLD CARTS.

ELEMENT INFERENCE: The answer '{answer}' indicates this is the {element_name} element because:
"""
        
        if element == "onset":
            inference_reasoning += f"- The answer describes when the symptom started (e.g., '{answer}' indicates timing of onset)\n"
            inference_reasoning += f"- Keywords like 'started', 'began', 'came on' indicate onset information\n"
        elif element == "location":
            inference_reasoning += f"- The answer describes where the symptom is located (e.g., '{answer}' indicates anatomical location)\n"
            inference_reasoning += f"- Location answers typically include anatomical terms, body parts, or directional descriptors\n"
        elif element == "character":
            inference_reasoning += f"- The answer describes what the symptom feels like (e.g., '{answer}' indicates sensory quality)\n"
            inference_reasoning += f"- Character answers describe sensations like sharp, dull, burning, pressure, etc.\n"
        
        inference_reasoning += f"\nELEMENT IDENTIFICATION: The patient's answer '{answer}' is being evaluated for the {element_name} element.\n"
        inference_reasoning += f"CRITICAL: This is {element_name}, NOT any other OLD CARTS element.\n"
        
        # Add standard reasoning (skip the first line which is redundant)
        standard_reasoning = generate_reasoning_for_element(
            element, answer, condition_name, differential_conditions, guideline
        )
        
        # Extract the reasoning parts after "CLINICAL REASONING:" line
        standard_lines = standard_reasoning.split("\n")
        # Skip first 3 lines (CLINICAL REASONING, ELEMENT IDENTIFICATION, CRITICAL)
        # and take the rest
        standard_rest = "\n".join(standard_lines[3:]) if len(standard_lines) > 3 else standard_reasoning
        
        # Combine inference + standard reasoning
        full_reasoning = inference_reasoning + "\n" + standard_rest
        
        messages.append({
            "role": "assistant",
            "content": full_reasoning
        })
        
        inference_examples.append({"messages": messages})
    
    return inference_examples

def main():
    print("=" * 80)
    print("Generating High-Quality Medical Conversation Dataset")
    print("=" * 80)
    print()
    
    # Load all guidelines
    print("📚 Loading medical guidelines...")
    all_guidelines = load_all_guidelines()
    
    total_conditions = sum(len(conds) for conds in all_guidelines.values())
    print(f"✅ Loaded {total_conditions} conditions from {len(all_guidelines)} organ systems")
    print()
    
    # Generate conversations
    print("🔧 Creating conversations with explicit element identification and condition matching...")
    print()
    
    conversations = []
    
    for system_name, conditions in all_guidelines.items():
        print(f"📋 Processing {system_name} system ({len(conditions)} conditions)...")
        for guideline in conditions:
            condition_name = guideline.get("condition", "Unknown")
            try:
                # Create standard conversation
                conversation = create_conversation_for_condition(guideline, all_guidelines)
                conversations.append(conversation)
                print(f"  ✅ Created conversation for: {condition_name}")
                
                # Create element inference examples (teaches model to infer element from answer alone)
                inference_examples = create_element_inference_examples(guideline, all_guidelines)
                conversations.extend(inference_examples)
                if inference_examples:
                    print(f"  ✅ Created {len(inference_examples)} element inference examples for: {condition_name}")
            except Exception as e:
                print(f"  ❌ Error creating conversation for {condition_name}: {e}")
        print()
    
    # Save dataset
    print(f"💾 Saving {len(conversations)} conversations to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved successfully!")
    print()
    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"Total conversations: {len(conversations)}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Key features:")
    print("  ✅ Explicit element identification (CHARACTER vs Distribution vs Intensity)")
    print("  ✅ Explicit condition matching (RLQ = Appendicitis, NOT Cholecystitis)")
    print("  ✅ No reasoning for demographics (age/sex)")
    print("  ✅ Progressive differential diagnosis with probability rankings")
    print("  ✅ Element inference training (model learns to infer element from answer alone)")
    print("=" * 80)

if __name__ == "__main__":
    main()

