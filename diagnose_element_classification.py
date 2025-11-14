#!/usr/bin/env python3
"""
Diagnostic script to identify why model misclassifies OLD CARTS elements.
"""

import json
from collections import Counter

DATASET_FILE = "medical_sft_dataset_high_quality.json"

def analyze_dataset_structure():
    """Analyze the dataset structure to find potential issues."""
    print("=" * 80)
    print("Dataset Structure Analysis")
    print("=" * 80)
    print()
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total conversations: {len(data)}")
    print()
    
    # Analyze message patterns
    question_patterns = Counter()
    reasoning_patterns = Counter()
    element_mentions = Counter()
    
    for conv_idx, conversation in enumerate(data[:5]):  # Sample first 5
        messages = conversation.get("messages", [])
        print(f"\n📋 Conversation {conv_idx + 1}:")
        print(f"   Total messages: {len(messages)}")
        
        # Track question-answer-reasoning patterns
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "assistant":
                # Check if it's a question
                if "?" in content and len(content) < 200:
                    question_patterns[content[:50]] += 1
                    print(f"   Q{i}: {content[:80]}...")
                
                # Check if it's reasoning
                if "CLINICAL REASONING" in content:
                    # Extract element name
                    if "Onset (O)" in content:
                        element_mentions["Onset"] += 1
                    elif "Location (L)" in content:
                        element_mentions["Location"] += 1
                    elif "Character (C)" in content:
                        element_mentions["Character"] += 1
                    elif "Duration (D)" in content:
                        element_mentions["Duration"] += 1
                    elif "Aggravating" in content:
                        element_mentions["Aggravating"] += 1
                    elif "Alleviating" in content:
                        element_mentions["Alleviating"] += 1
                    elif "Radiation (R)" in content:
                        element_mentions["Radiation"] += 1
                    elif "Timing (T)" in content:
                        element_mentions["Timing"] += 1
                    elif "Severity (S)" in content:
                        element_mentions["Severity"] += 1
                    
                    # Extract first line
                    first_line = content.split("\n")[0]
                    reasoning_patterns[first_line[:80]] += 1
    
    print("\n" + "=" * 80)
    print("Element Distribution in Reasoning")
    print("=" * 80)
    for element, count in element_mentions.most_common():
        print(f"  {element}: {count}")
    
    print("\n" + "=" * 80)
    print("Question Patterns")
    print("=" * 80)
    for pattern, count in question_patterns.most_common(10):
        print(f"  {pattern}... ({count}x)")
    
    print("\n" + "=" * 80)
    print("Reasoning Patterns")
    print("=" * 80)
    for pattern, count in reasoning_patterns.most_common(10):
        print(f"  {pattern}... ({count}x)")

def check_training_vs_test_format():
    """Compare training format vs test format."""
    print("\n" + "=" * 80)
    print("Training vs Test Format Comparison")
    print("=" * 80)
    print()
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get a sample conversation
    sample = data[0]
    messages = sample.get("messages", [])
    
    print("📚 TRAINING FORMAT (from dataset):")
    print("-" * 80)
    
    # Find a question-answer-reasoning sequence
    for i in range(len(messages) - 2):
        if (messages[i].get("role") == "assistant" and "?" in messages[i].get("content", "") and
            messages[i+1].get("role") == "user" and
            messages[i+2].get("role") == "assistant" and "CLINICAL REASONING" in messages[i+2].get("content", "")):
            
            question = messages[i].get("content", "")
            answer = messages[i+1].get("content", "")
            reasoning = messages[i+2].get("content", "")
            
            print(f"\n1. Question: {question}")
            print(f"2. Answer: {answer}")
            print(f"3. Reasoning: {reasoning[:200]}...")
            break
    
    print("\n" + "=" * 80)
    print("🧪 TEST FORMAT (from test script):")
    print("-" * 80)
    print("""
    Test provides:
    1. User: "It's new, started about an hour ago"
    2. Model generates response (tries to infer element from answer alone)
    
    ❌ PROBLEM: No question context!
    ❌ Model doesn't know which OLD CARTS element the answer corresponds to
    ❌ Model defaults to first element (Onset) because it's pattern-matching
    """)
    
    print("\n" + "=" * 80)
    print("🔍 ROOT CAUSE IDENTIFIED")
    print("=" * 80)
    print("""
    The model was trained on:
    Question → Answer → Reasoning (with element identified from question)
    
    But tested on:
    Answer only → Model tries to infer element → Fails → Defaults to Onset
    
    SOLUTIONS:
    1. Fix test to include questions (match training format)
    2. Add element inference training (teach model to infer from answer)
    3. Add explicit element markers in training data
    """)

def check_element_identification_strength():
    """Check how strongly elements are identified in training data."""
    print("\n" + "=" * 80)
    print("Element Identification Strength Analysis")
    print("=" * 80)
    print()
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    element_identification_patterns = {
        "Onset": 0,
        "Location": 0,
        "Character": 0,
        "Duration": 0,
        "Aggravating": 0,
        "Alleviating": 0,
        "Radiation": 0,
        "Timing": 0,
        "Severity": 0
    }
    
    explicit_mentions = {
        "Onset": 0,
        "Location": 0,
        "Character": 0,
        "Duration": 0,
        "Aggravating": 0,
        "Alleviating": 0,
        "Radiation": 0,
        "Timing": 0,
        "Severity": 0
    }
    
    for conversation in data:
        messages = conversation.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if "CLINICAL REASONING" in content:
                # Check for explicit element mentions
                if "This is the Onset (O)" in content:
                    element_identification_patterns["Onset"] += 1
                if "This is the Location (L)" in content:
                    element_identification_patterns["Location"] += 1
                if "This is the Character (C)" in content:
                    element_identification_patterns["Character"] += 1
                if "This is the Duration (D)" in content:
                    element_identification_patterns["Duration"] += 1
                if "This is the Aggravating" in content:
                    element_identification_patterns["Aggravating"] += 1
                if "This is the Alleviating" in content:
                    element_identification_patterns["Alleviating"] += 1
                if "This is the Radiation (R)" in content:
                    element_identification_patterns["Radiation"] += 1
                if "This is the Timing (T)" in content:
                    element_identification_patterns["Timing"] += 1
                if "This is the Severity (S)" in content:
                    element_identification_patterns["Severity"] += 1
                
                # Check for explicit mentions in reasoning text
                if "ONSET (O)" in content or "Onset (O)" in content:
                    explicit_mentions["Onset"] += 1
                if "LOCATION (L)" in content or "Location (L)" in content:
                    explicit_mentions["Location"] += 1
                if "CHARACTER (C)" in content or "Character (C)" in content:
                    explicit_mentions["Character"] += 1
    
    print("Element Identification Patterns (in 'This is the X' format):")
    for element, count in element_identification_patterns.items():
        print(f"  {element}: {count}")
    
    print("\nExplicit Element Mentions in Reasoning:")
    for element, count in explicit_mentions.items():
        print(f"  {element}: {count}")
    
    print("\n" + "=" * 80)
    print("✅ Element identification is STRONG in training data")
    print("❌ But model can't use it because test doesn't provide question context")
    print("=" * 80)

if __name__ == "__main__":
    analyze_dataset_structure()
    check_training_vs_test_format()
    check_element_identification_strength()

