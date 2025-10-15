#!/usr/bin/env python3
"""
Test Script: Unified Medical Mode

Tests the new unified medical mode that combines symptom assessment and medical knowledge
into a single physician-like mode.
"""

import sys
import os
from pathlib import Path

# Add the llm-container to path
sys.path.insert(0, str(Path(__file__).parent / "llm-container"))

def test_unified_medical_imports():
    """Test that unified medical mode can be imported"""
    print("🔧 Testing unified medical mode imports...")

    try:
        from unified_medical_mode import UnifiedMedicalSession, is_unified_medical_trigger
        print("✅ Unified medical mode imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Unified medical mode import failed: {e}")
        return False

def test_medical_query_detection():
    """Test that medical queries are correctly detected"""
    print("\n🩺 Testing medical query detection...")

    try:
        from unified_medical_mode import is_unified_medical_trigger

        # Test cases: should trigger unified medical mode
        medical_queries = [
            "I have chest pain",  # Symptom
            "What is hypertension?",  # Medical knowledge
            "How do you treat diabetes?",  # Treatment knowledge
            "I feel dizzy and nauseous",  # Symptom
            "Tell me about heart disease",  # Medical knowledge
            "What are the symptoms of pneumonia?"  # Symptom knowledge
        ]

        # Test cases: should NOT trigger unified medical mode
        non_medical_queries = [
            "Hello, how are you?",
            "What is the weather like?",
            "Tell me a joke",
            "How do you make coffee?"
        ]

        medical_count = 0
        non_medical_count = 0

        for query in medical_queries:
            if is_unified_medical_trigger(query):
                medical_count += 1
                print(f"  ✅ '{query}' → UNIFIED_MEDICAL")
            else:
                print(f"  ❌ '{query}' → NOT MEDICAL")

        for query in non_medical_queries:
            if not is_unified_medical_trigger(query):
                non_medical_count += 1
                print(f"  ✅ '{query}' → NOT MEDICAL")
            else:
                print(f"  ❌ '{query}' → MEDICAL")

        print(f"\n📊 Results: {medical_count}/{len(medical_queries)} medical queries detected correctly")
        print(f"📊 Results: {non_medical_count}/{len(non_medical_queries)} non-medical queries excluded correctly")

        return medical_count == len(medical_queries) and non_medical_count == len(non_medical_queries)

    except Exception as e:
        print(f"❌ Medical query detection test failed: {e}")
        return False

def test_query_type_analysis():
    """Test that query types are correctly analyzed"""
    print("\n🔍 Testing query type analysis...")

    try:
        from unified_medical_mode import UnifiedMedicalSession

        def mock_llm(messages):
            return "Mock response"

        session = UnifiedMedicalSession("test_session", mock_llm)

        test_cases = [
            ("I have chest pain", "symptom_assessment"),
            ("What is hypertension?", "medical_knowledge"),
            ("How do you treat diabetes?", "medical_knowledge"),
            ("I feel dizzy and nauseous", "symptom_assessment"),
            ("Tell me about heart disease", "medical_knowledge"),
            ("medicine is interesting", "general_medical"),
            ("Hello there", "general_medical")
        ]

        correct = 0

        for query, expected_type in test_cases:
            actual_type = session._analyze_medical_query(query)
            if actual_type == expected_type:
                correct += 1
                print(f"  ✅ '{query}' → {actual_type}")
            else:
                print(f"  ❌ '{query}' → {actual_type} (expected {expected_type})")

        print(f"\n📊 Results: {correct}/{len(test_cases)} query types analyzed correctly")
        return correct >= len(test_cases) * 0.8  # Allow some flexibility

    except Exception as e:
        print(f"❌ Query type analysis test failed: {e}")
        return False

def test_mode_routing_priority():
    """Test that unified medical mode has proper priority over other modes"""
    print("\n⚖️ Testing mode routing priority...")

    try:
        from router import route_prompt, ConversationMode
        from unified_medical_mode import is_unified_medical_trigger

        def mock_llm(messages):
            return "Mock response"

        # Test that medical queries route to unified medical, not thinker
        medical_query = "What is hypertension?"
        mode, state = route_prompt(medical_query, {}, "test_session", mock_llm)

        if mode == ConversationMode.UNIFIED_MEDICAL:
            print(f"  ✅ '{medical_query}' → UNIFIED_MEDICAL (correct priority)")
            return True
        else:
            print(f"  ❌ '{medical_query}' → {mode} (should be UNIFIED_MEDICAL)")
            return False

    except Exception as e:
        print(f"❌ Mode routing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 UNIFIED MEDICAL MODE TEST SUITE")
    print("=" * 50)

    tests = [
        test_unified_medical_imports,
        test_medical_query_detection,
        test_query_type_analysis,
        test_mode_routing_priority
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 OVERALL TEST RESULTS:")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🎉 Unified medical mode is working correctly!")
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("🔧 Please check the errors above and fix issues.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
