#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance Medical Dataset with Smart Features
==========================================
- Uses medical_sft_dataset_enhanced.json as foundation
- Adds smart OLD CARTS question selection (skips irrelevant elements)
- Adds British slang variations for UK market
- Maintains all clinical reasoning and structure
"""

import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# ============================================================================
# British Slang Variations
# ============================================================================

BRITISH_SLANG_MAPPINGS = {
    # Pain descriptions
    "hurting": "proper painful",
    "killing me": "really hurting",
    "really bad": "quite bad",
    "awful": "rubbish",
    "terrible": "not great",
    
    # Temporal phrases
    "came on": "came on",
    "started": "started",
    "began": "began",
    "out of nowhere": "out of the blue",
    "all of a sudden": "all of a sudden",
    
    # Location phrases
    "right here": "over here",
    "over here": "round here",
    "in this spot": "in this bit",
    "right there": "over there",
    
    # Duration phrases
    "been going on": "been going on",
    "sticking around": "hanging about",
    "hanging on": "hanging about",
    
    # Character phrases
    "feels like": "feels a bit like",
    "kind of like": "sort of",
    "sort of": "a bit",
    "really": "quite",
    "pretty": "a bit",
    
    # Aggravating phrases
    "really messes me up": "proper sets it off",
    "kills me": "makes it bad",
    "makes it worse": "makes it worse",
    "aggravates it": "makes it flare up",
    
    # Alleviating phrases (mostly same)
    "helps": "helps",
    "makes it better": "makes it better",
    "eases it": "eases it",
    "takes the edge off": "takes the edge off",
    
    # Severity phrases
    "really bad": "quite bad",
    "pretty bad": "not too bad",
    "not too bad": "manageable",
    "killing me": "proper painful",
    
    # Common phrases
    # Note: "I have" -> "I've got" is handled separately to avoid conflicts
    "I'm": "I am",
    # "I've": "I have",  # Removed - conflicts with "I've got" conversion
    "can't": "cannot",
    "don't": "do not",
    "won't": "will not",
}

BRITISH_SPECIFIC_PHRASES = {
    "bloody": ["bloody", "blooming", "ruddy"],
    "very": ["very", "quite", "rather"],
    "really": ["really", "proper", "quite"],
    "a lot": ["a lot", "quite a bit", "a fair bit"],
    "okay": ["okay", "alright", "fine"],
    "sure": ["sure", "certainly", "of course"],
}

# ============================================================================
# OLD CARTS Relevance Detection
# ============================================================================

def determine_relevant_oldcarts_from_conversation(conversation: Dict) -> Dict[str, bool]:
    """
    Determine which OLD CARTS elements are relevant based on conversation content.
    Analyzes the chief complaint and existing questions in the conversation.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return {element: True for element in ["O", "L", "D", "C", "A_aggravating", "A_alleviating", "R", "T", "S"]}
    
    # Find chief complaint
    chief_complaint = ""
    for msg in messages:
        if msg.get("role") == "user" and "I have" in msg.get("content", ""):
            chief_complaint = msg.get("content", "").lower()
            break
    
    if not chief_complaint:
        # Default: all relevant
        return {element: True for element in ["O", "L", "D", "C", "A_aggravating", "A_alleviating", "R", "T", "S"]}
    
    # Default: most elements relevant
    relevant = {
        "O": True,
        "L": False,
        "D": True,
        "C": False,
        "A_aggravating": True,
        "A_alleviating": True,
        "R": False,
        "T": True,
        "S": True,
    }
    
    # Check for systemic conditions (no location)
    systemic_keywords = [
        'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
        'diabetes', 'polyuria', 'polydipsia', 'polyphagia', 'fatigue', 'dizziness',
        'depression', 'anxiety', 'insomnia', 'elevated blood pressure', 'palpitations'
    ]
    
    # Check for non-sensory symptoms (no character)
    non_sensory_keywords = [
        'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
        'polyuria', 'polydipsia', 'polyphagia', 'constipation', 'urinary incontinence',
        'insomnia', 'difficulty falling asleep', 'difficulty maintaining sleep'
    ]
    
    # Check for non-radiating symptoms
    non_radiating_keywords = [
        'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
        'diabetes', 'fatigue', 'polyuria', 'polydipsia', 'constipation',
        'urinary incontinence', 'insomnia', 'depression', 'anxiety', 'dizziness'
    ]
    
    # Check if location is relevant
    if any(keyword in chief_complaint for keyword in systemic_keywords):
        relevant["L"] = False
    else:
        # Pain, discomfort, and sensory symptoms typically have location
        pain_keywords = ['pain', 'ache', 'discomfort', 'sore', 'tender', 'burning', 'stinging', 'headache', 'back pain']
        if any(keyword in chief_complaint for keyword in pain_keywords):
            relevant["L"] = True
        
        # Skin conditions typically have location
        skin_keywords = ['rash', 'dermatitis', 'acne', 'lesion', 'eruption', 'skin']
        if any(keyword in chief_complaint for keyword in skin_keywords):
            relevant["L"] = True
        
        # Respiratory symptoms may have location
        respiratory_keywords = ['cough', 'shortness of breath', 'dyspnea', 'wheezing', 'chest tightness']
        if any(keyword in chief_complaint for keyword in respiratory_keywords):
            relevant["L"] = True
        
        # GI symptoms may have location
        gi_keywords = ['abdominal', 'stomach', 'belly', 'heartburn', 'nausea', 'vomiting']
        if any(keyword in chief_complaint for keyword in gi_keywords):
            relevant["L"] = True
    
    # Check if character is relevant
    if any(keyword in chief_complaint for keyword in non_sensory_keywords):
        relevant["C"] = False
    else:
        # Pain and sensory symptoms have character
        pain_keywords = ['pain', 'ache', 'discomfort', 'sore', 'burning', 'stinging', 'itching']
        if any(keyword in chief_complaint for keyword in pain_keywords):
            relevant["C"] = True
    
    # Check if radiation is relevant
    if any(keyword in chief_complaint for keyword in non_radiating_keywords):
        relevant["R"] = False
    else:
        # Pain can radiate
        pain_keywords = ['pain', 'ache', 'discomfort', 'chest pain', 'back pain']
        if any(keyword in chief_complaint for keyword in pain_keywords):
            relevant["R"] = True
    
    return relevant

def extract_oldcarts_elements_from_conversation(conversation: Dict) -> Dict[str, bool]:
    """Extract which OLD CARTS elements are actually asked in the conversation."""
    messages = conversation.get("messages", [])
    asked_elements = {
        "O": False,
        "L": False,
        "D": False,
        "C": False,
        "A_aggravating": False,
        "A_alleviating": False,
        "R": False,
        "T": False,
        "S": False,
    }
    
    # Patterns to detect OLD CARTS questions
    patterns = {
        "O": [r"when did", r"when did it start", r"how did it begin", r"onset"],
        "L": [r"where", r"location", r"located"],
        "D": [r"how long", r"duration", r"been present"],
        "C": [r"what does.*feel", r"character", r"sharp.*dull.*burning"],
        "A_aggravating": [r"what makes.*worse", r"aggravates", r"triggers"],
        "A_alleviating": [r"what makes.*better", r"helps", r"relieves"],
        "R": [r"spread", r"radiate", r"goes.*else"],
        "T": [r"constant.*intermittent", r"comes and goes", r"timing"],
        "S": [r"scale.*1.*10", r"how severe", r"severity"],
    }
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            for element, element_patterns in patterns.items():
                if any(re.search(pattern, content) for pattern in element_patterns):
                    asked_elements[element] = True
    
    return asked_elements

# ============================================================================
# British Slang Conversion
# ============================================================================

def convert_to_british_slang(text: str) -> str:
    """Convert American English to British English slang."""
    if not text:
        return text
    
    result = text
    
    # First, handle "I have" -> "I've got" (most common conversion)
    # Only convert when "I have" is followed by a noun (possession), not "been" (present perfect)
    # Check if it's already "I've got" to avoid double conversion
    if "I've got" not in result and "I have got" not in result:
        # Match "I have" followed by a noun (not a verb like "been", "had", "seen")
        # Use a function to properly replace just "I have" with "I've got"
        def replace_i_have(match):
            # match.group(0) is the full match, e.g., "I have coffee"
            # We want to replace "I have" with "I've got"
            rest = match.group(0)[6:]  # Everything after "I have" (6 chars)
            return "I've got" + rest
        
        # Match "I have" + space + word (but not verbs like "been", "had", etc.)
        result = re.sub(
            r'\bI have\s+(?!been|had|seen|done|gone|taken|given|made|said|got|felt|heard|known|told|shown)\w+',
            replace_i_have,
            result,
            flags=re.IGNORECASE
        )
        
        # Also handle standalone "I have" at start of sentence
        result = re.sub(r'^I have\s+(?!been|had|seen)', "I've got ", result, flags=re.IGNORECASE)
    
    # Apply direct mappings (excluding "I have" which we already handled)
    # Also protect "I've got" from being converted back
    for american, british in BRITISH_SLANG_MAPPINGS.items():
        if american == "I have":  # Skip, already handled
            continue
        # Protect "I've got" from being converted (e.g., if "I've" -> "I have" mapping exists)
        if "I've got" in result and american in ["I've"]:
            continue  # Skip mappings that would break "I've got"
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(american) + r'\b'
        result = re.sub(pattern, british, result, flags=re.IGNORECASE)
    
    # Apply British-specific phrases
    for phrase, alternatives in BRITISH_SPECIFIC_PHRASES.items():
        if phrase in result.lower():
            # Randomly choose a British alternative
            british_alt = random.choice(alternatives)
            pattern = r'\b' + re.escape(phrase) + r'\b'
            result = re.sub(pattern, british_alt, result, flags=re.IGNORECASE)
    
    # Other British contractions and phrases
    british_replacements = [
        ("I'm having", "I am having"),
        ("can't", "cannot"),
        ("won't", "will not"),
    ]
    
    for american, british in british_replacements:
        result = result.replace(american, british)
    
    return result

def create_british_variant(conversation: Dict) -> Dict:
    """Create a British slang variant of a conversation."""
    british_conv = {
        "messages": [],
        "variant": "british",
        "original": True,  # Mark as variant
    }
    
    # Copy metadata if present
    for key in ["organ_system", "diagnosis", "icd10", "conversation_number"]:
        if key in conversation:
            british_conv[key] = conversation[key]
    
    messages = conversation.get("messages", [])
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        # Convert user messages and some assistant messages to British slang
        # Don't convert clinical reasoning or system prompts
        if role == "user":
            # Convert user responses to British slang
            british_content = convert_to_british_slang(content)
            british_conv["messages"].append({
                "role": role,
                "content": british_content
            })
        elif role == "assistant":
            # Only convert conversational parts, not clinical reasoning
            if "CLINICAL REASONING" in content or "FINAL DIAGNOSTIC REASONING" in content:
                # Keep clinical reasoning as-is
                british_conv["messages"].append(msg)
            elif content.startswith("I understand") or "Is this a new issue" in content:
                # Convert empathetic/initial questions to British
                british_content = convert_to_british_slang(content)
                british_conv["messages"].append({
                    "role": role,
                    "content": british_content
                })
            else:
                # Regular questions - convert to British
                british_content = convert_to_british_slang(content)
                british_conv["messages"].append({
                    "role": role,
                    "content": british_content
                })
        else:
            # System messages - keep as-is
            british_conv["messages"].append(msg)
    
    return british_conv

# ============================================================================
# Add Skip Tags
# ============================================================================

def add_skip_tags_to_conversation(conversation: Dict) -> Dict:
    """Add skip tags for irrelevant OLD CARTS elements."""
    enhanced_conv = {
        "messages": [],
        "smart_features": True,
        "relevant_oldcarts": {},
    }
    
    # Copy metadata
    for key in ["organ_system", "diagnosis", "icd10", "conversation_number", "variant"]:
        if key in conversation:
            enhanced_conv[key] = conversation[key]
    
    # Determine relevant elements
    relevant_elements = determine_relevant_oldcarts_from_conversation(conversation)
    enhanced_conv["relevant_oldcarts"] = relevant_elements
    
    # Extract which elements are actually asked
    asked_elements = extract_oldcarts_elements_from_conversation(conversation)
    
    messages = conversation.get("messages", [])
    new_messages = []
    
    # Find chief complaint
    chief_complaint = ""
    for msg in messages:
        if msg.get("role") == "user" and "I have" in msg.get("content", ""):
            chief_complaint = msg.get("content", "")
            break
    
    # Extract diagnosis from clinical reasoning if available
    diagnosis = "Unknown"
    for msg in messages:
        if "CLINICAL REASONING" in msg.get("content", ""):
            # Try to extract diagnosis from reasoning
            content = msg.get("content", "")
            match = re.search(r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", content)
            if match:
                diagnosis = match.group(1)
                break
    
    i = 0
    skip_tags_added = set()  # Track which elements we've already added skip tags for
    
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content", "")
        
        # Skip if this is already a skip tag we added
        if role == "assistant" and content.startswith("[SKIP:") and msg.get("metadata", {}).get("skip"):
            # Check if we've already added a skip tag for this element
            element_match = re.search(r"\[SKIP:([^\]]+)\]", content)
            if element_match:
                element = element_match.group(1)
                if element in skip_tags_added:
                    # Already processed - skip this duplicate
                    i += 1
                    continue
                skip_tags_added.add(element)
        
        # Check if this is an OLD CARTS question
        is_oldcarts_question = False
        element = None
        
        if role == "assistant" and "CLINICAL REASONING" not in content and "FINAL" not in content and not content.startswith("[SKIP:"):
            # Check which element this question is about
            content_lower = content.lower()
            if re.search(r"when did", content_lower):
                element = "O"
            elif re.search(r"where", content_lower):
                element = "L"
            elif re.search(r"how long", content_lower):
                element = "D"
            elif re.search(r"what does.*feel", content_lower) or re.search(r"character", content_lower):
                element = "C"
            elif re.search(r"what makes.*worse", content_lower) or re.search(r"aggravates", content_lower):
                element = "A_aggravating"
            elif re.search(r"what makes.*better", content_lower) or re.search(r"helps", content_lower):
                element = "A_alleviating"
            elif re.search(r"spread", content_lower) or re.search(r"radiate", content_lower):
                element = "R"
            elif re.search(r"constant.*intermittent", content_lower) or re.search(r"comes and goes", content_lower):
                element = "T"
            elif re.search(r"scale.*1.*10", content_lower) or re.search(r"how severe", content_lower):
                element = "S"
            
            if element:
                is_oldcarts_question = True
        
        # If this is an irrelevant OLD CARTS question, add skip tag and remove question/reasoning
        if is_oldcarts_question and element and not relevant_elements.get(element, True) and element not in skip_tags_added:
            # Add skip message
            skip_msg = {
                "role": "assistant",
                "content": f"[SKIP:{element}] This OLD CARTS element is not relevant for this chief complaint and should be skipped.",
                "metadata": {
                    "skip": True,
                    "element": element,
                    "reason": f"Not relevant for chief complaint: {chief_complaint}"
                }
            }
            new_messages.append(skip_msg)
            skip_tags_added.add(element)
            
            # Skip the question itself
            i += 1
            
            # Skip user answer if present
            if i < len(messages) and messages[i].get("role") == "user":
                i += 1
            
            # Skip any clinical reasoning that follows for this element
            # Look ahead for clinical reasoning about this specific element
            while i < len(messages):
                next_msg = messages[i]
                next_content = next_msg.get("content", "")
                
                # Check if this is clinical reasoning for the skipped element
                if "CLINICAL REASONING" in next_content:
                    # Check if this reasoning is about the skipped element
                    element_patterns = {
                        "O": r"Onset\s*\(O\)",
                        "L": r"Location\s*\(L\)",
                        "D": r"Duration\s*\(D\)",
                        "C": r"Character\s*\(C\)",
                        "A_aggravating": r"Aggravating",
                        "A_alleviating": r"Alleviating",
                        "R": r"Radiation\s*\(R\)",
                        "T": r"Timing\s*\(T\)",
                        "S": r"Severity\s*\(S\)"
                    }
                    
                    pattern = element_patterns.get(element)
                    if pattern and re.search(pattern, next_content, re.IGNORECASE):
                        # This is reasoning for the skipped element - skip it
                        i += 1
                        continue
                    else:
                        # This is reasoning for a different element - keep it
                        break
                elif next_content.startswith("[SKIP:"):
                    # Another skip tag - stop here
                    break
                else:
                    # Not clinical reasoning or skip tag - stop skipping
                    break
            
            continue
        
        # Otherwise, keep the message
        new_messages.append(msg)
        i += 1
    
    enhanced_conv["messages"] = new_messages
    return enhanced_conv

# ============================================================================
# Main Processing
# ============================================================================

def process_enhanced_dataset(input_file: str, output_file: str):
    """Process the enhanced dataset with smart features and British variants."""
    
    print("=" * 80)
    print("Enhancing Medical Dataset with Smart Features")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    # Load enhanced dataset
    print("Loading enhanced dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} conversations")
    print()
    
    # Process conversations
    enhanced_data = []
    british_variants = []
    
    print("Processing conversations...")
    for idx, conversation in enumerate(data):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(data)} conversations...")
        
        # Add skip tags to original
        enhanced_conv = add_skip_tags_to_conversation(conversation)
        enhanced_conv["original_index"] = idx
        enhanced_conv["variant"] = "american"  # Mark original as American
        enhanced_data.append(enhanced_conv)
        
        # Create British variant
        british_conv = create_british_variant(conversation)
        british_conv = add_skip_tags_to_conversation(british_conv)
        british_conv["original_index"] = idx
        british_conv["variant"] = "british"
        british_variants.append(british_conv)
    
    print(f"✅ Processed {len(enhanced_data)} original conversations")
    print(f"✅ Created {len(british_variants)} British variants")
    print()
    
    # Combine datasets
    combined_data = enhanced_data + british_variants
    print(f"✅ Combined dataset: {len(combined_data)} total conversations")
    print(f"   - American: {len(enhanced_data)}")
    print(f"   - British: {len(british_variants)}")
    print()
    
    # Save enhanced dataset
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced dataset saved!")
    print()
    print("=" * 80)
    print("Enhancement Complete!")
    print("=" * 80)
    print(f"Total conversations: {len(combined_data)}")
    print(f"Features added:")
    print(f"  ✅ Smart OLD CARTS question selection (skip irrelevant elements)")
    print(f"  ✅ British slang variations for UK market")
    print(f"  ✅ Relevance metadata for each conversation")
    print("=" * 80)

if __name__ == "__main__":
    input_file = "medical_sft_dataset_enhanced.json"
    output_file = "medical_sft_dataset_enhanced_smart.json"
    
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found!")
        exit(1)
    
    process_enhanced_dataset(input_file, output_file)

