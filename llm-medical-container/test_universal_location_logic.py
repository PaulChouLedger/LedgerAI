#!/usr/bin/env python3
"""
Test Universal Anatomical Competition Logic
Tests the pure similarity-based approach for location competition detection
"""

import sys
import os

# Set required environment variables if not set
if 'LLM_TEMPERATURE_SIMPLE' not in os.environ:
    os.environ['LLM_TEMPERATURE_SIMPLE'] = '0.1'
if 'LLM_TEMPERATURE_COMPLEX' not in os.environ:
    os.environ['LLM_TEMPERATURE_COMPLEX'] = '0.3'
if 'LLM_MODEL_SIMPLE' not in os.environ:
    os.environ['LLM_MODEL_SIMPLE'] = 'llama-1b'
if 'LLM_MODEL_COMPLEX' not in os.environ:
    os.environ['LLM_MODEL_COMPLEX'] = 'mistral-7b'

sys.path.insert(0, os.path.dirname(__file__))

from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine

def test_universal_location_competition():
    """Test universal location competition with 'right side' pain"""
    
    print("=" * 80)
    print("TEST: Universal Anatomical Competition Logic")
    print("=" * 80)
    print()
    
    # Fix guidelines directory path for local testing
    from pathlib import Path
    base_dir = Path(__file__).parent
    guidelines_path = base_dir / 'medical' / 'guidelines'
    if guidelines_path.exists():
        print(f"✅ Using guidelines from: {guidelines_path}")
    else:
        print(f"⚠️  Guidelines not found at: {guidelines_path}")
    
    # Initialize engine with correct guidelines directory
    engine = AdaptiveDiagnosticEngine(guidelines_dir=str(guidelines_path))
    engine.debug_mode = True
    
    # Simulate initial complaint
    complaint = "i have abodminal pain"
    print(f"📋 Initial Complaint: '{complaint}'")
    print()
    
    # Start assessment
    result = engine.start_assessment(complaint)
    print(f"\n1️⃣  First Question: {result.get('question', 'N/A')}")
    print()
    
    # Answer demographics
    demo_result = engine.process_answer("35")
    print(f"\n2️⃣  Second Question: {demo_result.get('question', 'N/A')}")
    print()
    
    # Answer sex
    sex_result = engine.process_answer("female")
    print(f"\n3️⃣  Third Question: {sex_result.get('question', 'N/A')}")
    print()
    
    # Answer timing
    timing_result = engine.process_answer("2 hours ago")
    print(f"\n4️⃣  Fourth Question: {timing_result.get('question', 'N/A')}")
    print()
    
    # NOW answer with location: "right side"
    location_answer = "right side"
    print(f"💬 Location Answer: '{location_answer}'")
    print()
    
    # Process the location answer
    answer_result = engine.process_answer(location_answer)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    if 'question' in answer_result:
        print(f"📝 System Response: {answer_result['question']}")
        print()
        if 'clarification_needed' in answer_result.get('debug', {}).get('competition', {}):
            comp = answer_result['debug']['competition']
            if comp.get('has_competition'):
                print(f"✅ COMPETITION DETECTED: {comp.get('competing_areas', [])}")
            else:
                print(f"✅ NO COMPETITION: Answer accepted")
    else:
        print("❌ ERROR: No response generated")
    
    print()
    print("=" * 80)
    
    return answer_result

if __name__ == "__main__":
    result = test_universal_location_competition()
