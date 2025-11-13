#!/usr/bin/env python3
"""
Create high-quality medical training dataset with interleaved differential diagnosis reasoning.

This script:
1. Uses medical guidelines to create realistic conversations
2. Includes 2-3 diseases per organ system
3. Adds interleaved reasoning after each OLD CARTS answer
4. Shows progressive narrowing of differential diagnosis
5. Ends with most probable diagnosis
6. Handles both sensory and visual complaints
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Paths
GUIDELINES_DIR = "llm-medical-container/medical/guidelines"
OUTPUT_DATASET = "medical_sft_dataset_high_quality.json"

# Medical system prompt (matching training)
MEDICAL_SYSTEM_PROMPT = """You are a professional medical assistant. 

IMPORTANT RULES:
- ONLY ask medical questions when the patient mentions a symptom, pain, or medical concern
- If the patient is just greeting you or having casual conversation, respond naturally and wait for them to mention a medical issue
- NEVER make up or assume symptoms the patient hasn't mentioned
- NEVER make statements about the patient's information (like "Your age is 27") - always ASK questions
- Always ask questions, never make statements about patient information

When a patient tells you about a symptom or medical concern, follow this order:

1. Show empathy and acknowledge their concern
2. Ask if this is new or an ongoing problem
3. Ask their age
4. Ask their biological sex
5. Then ask about the symptom - one question at a time, waiting for each answer

Ask about: when it started, where it is, how long it's been present, what it feels like, what makes it worse, what makes it better, if it spreads, if it's constant or comes and goes, and how severe it is.

Be natural and conversational. Ask only one question at a time. Do not list multiple questions. Do not mention frameworks or include instructions in your responses. Do not include internal reasoning, acknowledgments, or explanations. Only ask the question.

CRITICAL: You MUST ask chronicity (new vs ongoing) BEFORE asking age. Do NOT skip chronicity. Do NOT ask medical questions unless the patient has mentioned a symptom."""

def load_all_guidelines() -> Dict[str, Dict]:
    """Load all guideline JSON files."""
    guidelines = {}
    guidelines_path = Path(GUIDELINES_DIR)
    
    if not guidelines_path.exists():
        return guidelines
    
    for json_file in guidelines_path.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                guideline = json.load(f)
                condition_name = guideline.get('condition', '')
                if condition_name:
                    guidelines[condition_name] = guideline
        except Exception as e:
            pass
    
    return guidelines

def get_organ_system_conditions(guidelines: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Organize conditions by organ system based on guideline file paths."""
    organ_systems = {
        'GI': [],
        'CARDIO': [],
        'RESPIRATORY': [],
        'NEURO': [],
        'RENAL': [],
        'GU': [],
        'MSK': [],
        'DERM': []
    }
    
    # Map condition names to organ systems based on guideline file paths
    guidelines_path = Path(GUIDELINES_DIR)
    
    for json_file in guidelines_path.rglob("*.json"):
        try:
            # Extract organ system from path (e.g., GI/GI_Acute_Cholecystitis.json -> GI)
            parts = json_file.parts
            if len(parts) >= 2:
                system = parts[-2]  # Parent directory name
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    guideline = json.load(f)
                    condition_name = guideline.get('condition', '')
                    
                    if condition_name and system in organ_systems:
                        organ_systems[system].append(condition_name)
        except Exception as e:
            pass
    
    return organ_systems

def select_differential_conditions(condition: str, all_conditions: List[str], guidelines: Dict[str, Dict], count: int = 3) -> List[str]:
    """Select conditions for differential diagnosis based on similar chief complaints."""
    target_guideline = guidelines.get(condition)
    if not target_guideline:
        return all_conditions[:count]
    
    target_triggers = set(target_guideline.get('key_features', {}).get('chief_complaint_triggers', []))
    
    # Score conditions by shared triggers
    scored = []
    for other_condition in all_conditions:
        if other_condition == condition:
            continue
        other_guideline = guidelines.get(other_condition)
        if not other_guideline:
            continue
        
        other_triggers = set(other_guideline.get('key_features', {}).get('chief_complaint_triggers', []))
        overlap = len(target_triggers & other_triggers)
        scored.append((other_condition, overlap))
    
    # Sort by overlap and take top conditions
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [condition] + [c[0] for c in scored[:count-1]]
    
    # Fill remaining slots if needed
    while len(selected) < count and len(selected) < len(all_conditions):
        for c in all_conditions:
            if c not in selected:
                selected.append(c)
                break
    
    return selected[:count]

def extract_patient_friendly_answer(guideline: Dict, element: str, oldcarts: Dict) -> Optional[str]:
    """Extract a patient-friendly answer from guideline for a specific OLD CARTS element."""
    if element not in oldcarts:
        return None
    
    includes = oldcarts[element].get('includes', [])
    if not includes:
        return None
    
    # Return first patient-friendly answer
    for item in includes:
        patient_friendly = item.get('patient_friendly', '')
        if patient_friendly:
            return patient_friendly
    
    return None

def generate_reasoning_for_answer(
    element: str,
    answer: str,
    conditions: List[str],
    guidelines: Dict[str, Dict],
    current_scores: Dict[str, float],
    excluded: set
) -> Tuple[str, Dict[str, float], set]:
    """Generate reasoning showing how answer affects differential diagnosis."""
    reasoning = f"REASONING: Based on the answer about {element}:\n"
    
    new_scores = current_scores.copy()
    new_excluded = excluded.copy()
    ruled_in = []
    ruled_out = []
    answer_lower = answer.lower()
    
    for condition_name in conditions:
        if condition_name in new_excluded:
            continue
        
        guideline = guidelines.get(condition_name)
        if not guideline:
            continue
        
        oldcarts = guideline.get('key_features', {}).get('structured_oldcarts', {})
        if element not in oldcarts:
            continue
        
        # Check includes (rule in)
        includes = oldcarts[element].get('includes', [])
        best_match = None
        best_confidence = 0.0
        
        for item in includes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly:
                # More precise matching: check if answer contains key descriptive words
                friendly_words = [w for w in patient_friendly.split() if len(w) > 3]
                match_score = 0
                
                # Full phrase match (strongest)
                if patient_friendly in answer_lower:
                    match_score = 1.0
                # Key word matches
                elif friendly_words:
                    matches = sum(1 for word in friendly_words if word in answer_lower)
                    match_score = matches / len(friendly_words) if friendly_words else 0
                
                if match_score > 0.5:  # Require at least 50% word match
                    confidence = 0.3 if element == 'location' else 0.25 if element == 'character' else 0.2 if element == 'timing' else 0.15
                    confidence *= match_score  # Scale by match quality
                    
                    if confidence > best_confidence:
                        best_match = patient_friendly
                        best_confidence = confidence
        
        if best_match:
            new_scores[condition_name] = min(new_scores.get(condition_name, 0.0) + best_confidence, 1.0)  # Cap at 100%
            ruled_in.append(f"• {condition_name}: {element.capitalize()} '{best_match}' matches pattern (+{best_confidence:.0%})")
        
        # Check excludes (rule out)
        excludes = oldcarts[element].get('excludes', [])
        for item in excludes:
            patient_friendly = item.get('patient_friendly', '').lower()
            if patient_friendly and patient_friendly in answer_lower:
                new_excluded.add(condition_name)
                new_scores[condition_name] = 0.0
                ruled_out.append(f"• {condition_name}: Answer contradicts pattern (excludes '{patient_friendly}')")
                break
    
    if not ruled_in and not ruled_out:
        return "", current_scores, excluded
    
    if ruled_in:
        reasoning += "\nRULED IN (increased likelihood):\n"
        reasoning += "\n".join(ruled_in)
    
    if ruled_out:
        reasoning += "\n\nRULED OUT:\n"
        reasoning += "\n".join(ruled_out)
    
    # Show current differential
    active = {k: min(v, 1.0) for k, v in new_scores.items() if k not in new_excluded and v > 0}  # Cap at 100%
    if active:
        sorted_conditions = sorted(active.items(), key=lambda x: x[1], reverse=True)
        reasoning += "\n\nCURRENT DIFFERENTIAL (top 3):\n"
        for i, (cond, score) in enumerate(sorted_conditions[:3], 1):
            reasoning += f"{i}. {cond}: {min(score, 1.0):.0%} likelihood\n"
    
    reasoning += "\n\nNEXT STEP: Ask about the next OLD CARTS element to further narrow the differential."
    
    return reasoning, new_scores, new_excluded

def create_conversation_for_condition(
    target_condition: str,
    guidelines: Dict[str, Dict],
    all_conditions: List[str]
) -> Optional[Dict]:
    """Create a complete conversation with reasoning for a specific condition."""
    guideline = guidelines.get(target_condition)
    if not guideline:
        return None
    
    # Select differential conditions
    differential = select_differential_conditions(target_condition, all_conditions, guidelines, count=4)
    
    # Get chief complaint
    triggers = guideline.get('chief_complaint_triggers', [])
    if not triggers:
        # Check nested location
        key_features = guideline.get('key_features', {})
        if isinstance(key_features, dict):
            triggers = key_features.get('chief_complaint_triggers', [])
    
    if not triggers:
        return None
    
    chief_complaint = triggers[0] if isinstance(triggers, list) else list(triggers)[0] if triggers else None
    if not chief_complaint:
        return None
    
    # Build system prompt with conditions
    conditions_text = ", ".join(differential)
    system_prompt = f"""You are a professional medical assistant conducting a medical history. 

The patient is reporting {chief_complaint}. You are considering these possible conditions: {conditions_text}.

CRITICAL: You MUST follow this EXACT sequence for EVERY conversation. DO NOT skip any step:

STEP 1: Show empathy and acknowledge their concern (REQUIRED - do this FIRST)
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

Be natural and conversational. Ask only one question at a time. Do not mention the conditions or your reasoning to the patient."""
    
    # Build conversation
    messages = [{"role": "system", "content": system_prompt}]
    
    # Chief complaint
    messages.append({"role": "user", "content": f"I have {chief_complaint}"})
    
    # Empathy
    messages.append({"role": "assistant", "content": f"I understand you're experiencing {chief_complaint}. I'm here to help."})
    
    # Chronicity
    oldcarts = guideline.get('key_features', {}).get('structured_oldcarts', {})
    onset = oldcarts.get('onset', {}).get('includes', [])
    is_chronic = any('chronic' in item.get('medical', '').lower() for item in onset)
    
    messages.append({
        "role": "assistant",
        "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
    })
    messages.append({
        "role": "user",
        "content": "It's ongoing" if is_chronic else "It's new"
    })
    
    # Age
    messages.append({"role": "assistant", "content": "How old are you?"})
    age = random.randint(25, 75)
    messages.append({"role": "user", "content": f"I'm {age} years old"})
    
    # Sex
    sex_preference = guideline.get('sex', 'both')
    if sex_preference == 'male':
        sex = 'Male'
    elif sex_preference == 'female':
        sex = 'Female'
    else:
        sex = random.choice(['Male', 'Female'])
    
    messages.append({"role": "assistant", "content": "What is your biological sex?"})
    messages.append({"role": "user", "content": sex})
    
    # Initialize scores and excluded
    condition_scores = {cond: 0.0 for cond in differential}
    excluded_conditions = set()
    last_element_reasoned = None
    
    # OLD CARTS sequence
    oldcarts_sequence = ['location', 'onset', 'character', 'aggravating', 'alleviating', 'radiation', 'timing', 'severity']
    
    for element in oldcarts_sequence:
        if element not in oldcarts:
            continue
        
        # Generate question
        if element == 'location':
            question = f"Where exactly is the {chief_complaint.split()[-1]} located?"
        elif element == 'onset':
            question = "When did it start?"
        elif element == 'character':
            is_visual = any(word in chief_complaint.lower() for word in ['blood', 'rash', 'stool', 'urine', 'discharge', 'appearance'])
            if is_visual:
                question = f"What does the {chief_complaint.split()[-1]} look like? For example, is it red, dark, bright, or something else?"
            else:
                question = f"What does the {chief_complaint.split()[-1]} feel like? For example, is it sharp, heavy, burning, or pressure?"
        elif element == 'aggravating':
            question = "What makes it worse?"
        elif element == 'alleviating':
            question = "What makes it better?"
        elif element == 'radiation':
            question = "Does it spread anywhere else?"
        elif element == 'timing':
            question = "Is it constant or does it come and go?"
        elif element == 'severity':
            question = "On a scale of 1 to 10, with 10 being the worst imaginable, how severe is it?"
        else:
            continue
        
        messages.append({"role": "assistant", "content": question})
        
        # Get answer from guideline
        answer = extract_patient_friendly_answer(guideline, element, oldcarts)
        if not answer:
            # Generate generic answer
            if element == 'location':
                answer = "In my abdomen"
            elif element == 'onset':
                answer = "Started a few hours ago"
            elif element == 'character':
                answer = "It's sharp"
            elif element == 'aggravating':
                answer = "Movement makes it worse"
            elif element == 'alleviating':
                answer = "Rest helps"
            elif element == 'radiation':
                answer = "No, it stays in one place"
            elif element == 'timing':
                answer = "It's constant"
            elif element == 'severity':
                answer = "About a 6 out of 10"
            else:
                continue
        
        messages.append({"role": "user", "content": answer})
        
        # Add reasoning (only once per element)
        if element != last_element_reasoned:
            reasoning, condition_scores, excluded_conditions = generate_reasoning_for_answer(
                element, answer, differential, guidelines, condition_scores, excluded_conditions
            )
            
            if reasoning and ('RULED IN' in reasoning or 'RULED OUT' in reasoning):
                messages.append({"role": "assistant", "content": reasoning})
                last_element_reasoned = element
    
    # Final diagnostic reasoning
    active = {k: v for k, v in condition_scores.items() if k not in excluded_conditions and v > 0}
    if active:
        sorted_conditions = sorted(active.items(), key=lambda x: x[1], reverse=True)
        top_condition = sorted_conditions[0]
        top_score = min(top_condition[1], 1.0)  # Cap at 100%
        
        final_reasoning = f"""FINAL DIAGNOSTIC REASONING:

Based on the complete OLD CARTS assessment:

TOP DIAGNOSIS: {top_condition[0]} ({top_score:.0%} confidence)

REASONING PATH:
1. Started with differential: {', '.join(differential[:3])}...
2. Each answer progressively narrowed the differential:
"""
        
        # Add key factors
        for cond, score in sorted_conditions[:3]:
            final_reasoning += f"   • {cond}: {min(score, 1.0):.0%} likelihood\n"
        
        final_reasoning += f"""
3. Pattern recognition identified {top_condition[0]} as most consistent
4. Key matching factors from OLD CARTS:
   • Location, character, timing, and aggravating/alleviating factors all support {top_condition[0]}

CONCLUSION: The symptom pattern most strongly supports {top_condition[0]}.
This diagnosis is based on the systematic collection and analysis of OLD CARTS elements, 
with each answer helping to rule in or rule out conditions in the differential diagnosis."""
        
        messages.append({"role": "assistant", "content": final_reasoning})
    
    return {"messages": messages}

def main():
    print("=" * 80)
    print("Creating High-Quality Medical Training Dataset")
    print("=" * 80)
    print()
    
    # Load guidelines
    print("📚 Loading medical guidelines...")
    guidelines = load_all_guidelines()
    print(f"✅ Loaded {len(guidelines)} guidelines")
    print()
    
    # Organize by organ system
    organ_systems = get_organ_system_conditions(guidelines)
    all_conditions = list(guidelines.keys())
    
    print("📊 Organ System Coverage:")
    for system, conditions in organ_systems.items():
        if conditions:
            print(f"  {system}: {len(conditions)} conditions")
    print()
    
    # Create conversations
    print("🔧 Creating high-quality conversations with reasoning...")
    conversations = []
    
    # For each organ system, create 5 conversations
    for system, conditions in organ_systems.items():
        if not conditions:
            continue
        
        print(f"\n📋 Processing {system} system ({len(conditions)} conditions available)...")
        
        # Select 5 conditions per system (prioritize diverse chief complaints)
        # Group by chief complaint to ensure variety
        selected = []
        seen_complaints = set()
        
        # First pass: prioritize unique chief complaints
        for condition in conditions:
            guideline = guidelines.get(condition)
            if not guideline:
                continue
            
            triggers = guideline.get('chief_complaint_triggers', [])
            if not triggers:
                key_features = guideline.get('key_features', {})
                if isinstance(key_features, dict):
                    triggers = key_features.get('chief_complaint_triggers', [])
            
            if triggers:
                chief_complaint = triggers[0] if isinstance(triggers, list) else list(triggers)[0] if triggers else None
                if chief_complaint and chief_complaint not in seen_complaints:
                    selected.append(condition)
                    seen_complaints.add(chief_complaint)
                    if len(selected) >= 5:  # 5 examples per system
                        break
        
        # If we still need more, add any remaining (up to 5 total)
        if len(selected) < 5:
            for condition in conditions:
                if condition not in selected and len(selected) < 5:
                    selected.append(condition)
                    if len(selected) >= 5:
                        break
        
        for condition in selected:
            conv = create_conversation_for_condition(condition, guidelines, all_conditions)
            if conv:
                conversations.append(conv)
                print(f"  ✅ Created conversation for: {condition}")
    
    print(f"\n✅ Created {len(conversations)} high-quality conversations")
    print()
    
    # Save dataset
    print(f"💾 Saving to {OUTPUT_DATASET}...")
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(conversations)} conversations")
    print()
    
    print("=" * 80)
    print("✅ High-Quality Dataset Created!")
    print("=" * 80)
    print()
    print("Features:")
    print("  ✅ 5 diseases per organ system (8 systems = up to 40 conversations)")
    print("  ✅ Interleaved reasoning after each OLD CARTS answer")
    print("  ✅ Progressive narrowing of differential diagnosis")
    print("  ✅ Rule-in/rule-out logic")
    print("  ✅ Final diagnostic conclusion")
    print("  ✅ Handles both sensory and visual complaints")
    print("  ✅ Comprehensive coverage: CARDIO, RESPIRATORY, NEURO, RENAL, GU, MSK, DERM, GI")
    print()
    print("Each conversation includes:")
    print("  • Complete OLD CARTS sequence")
    print("  • Reasoning after each answer showing rule-in/rule-out")
    print("  • Updated differential diagnosis after each question")
    print("  • Final diagnosis with confidence score")
    print("  • Structured system prompts with condition context")

if __name__ == "__main__":
    main()

