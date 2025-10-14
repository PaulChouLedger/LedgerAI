#!/usr/bin/env python3
"""
Test Script: Medical Symptom Routing to Enhanced Clinician

Tests that medical symptoms like "chest pain", "abdominal pain", etc.
are routed to enhanced clinician mode instead of rigid triage mode.
"""

import sys
import json
from pathlib import Path

# Add the llm-container to path
sys.path.append(str(Path(__file__).parent / "llm-container"))

def test_medical_symptom_routing():
    """Test that medical symptoms route to clinician mode"""

    # Import router components
    try:
        from router import route_prompt, ConversationMode
        from clinician import is_clinician_trigger
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test cases: medical symptoms that should trigger clinician mode
    medical_symptoms = [
        "I have chest pain",
        "I feel abdominal pain",
        "I have a headache",
        "I'm experiencing shortness of breath",
        "My stomach hurts",
        "I have pain in my back",
        "I'm feeling dizzy",
        "I have nausea and vomiting",
        "My joints ache",
        "I have a fever",
        "I'm bleeding",
        "I have swelling in my legs"
    ]

    # Test cases: knowledge questions that should NOT trigger clinician mode
    knowledge_questions = [
        "What is chest pain?",
        "How do you treat abdominal pain?",
        "Tell me about headaches",
        "What causes shortness of breath?",
        "Explain stomach pain"
    ]

    print("🩺 TESTING MEDICAL SYMPTOM ROUTING")
    print("=" * 50)

    # Mock LLM chat function (not used in routing decision)
    def mock_llm_chat(messages):
        return "Mock response"

    # Test medical symptoms
    print("\n📋 MEDICAL SYMPTOMS (should route to CLINICIAN mode):")
    clinician_count = 0
    triage_count = 0

    for symptom in medical_symptoms:
        print(f"\nTesting: '{symptom}'")

        # Check if clinician trigger detects it
        is_clinician = is_clinician_trigger(symptom.lower())
        print(f"  → Clinician trigger: {is_clinician}")

        # Test routing
        mode, state = route_prompt(symptom.lower(), {}, "test_session", mock_llm_chat)

        if mode == ConversationMode.CLINICIAN:
            clinician_count += 1
            print(f"  ✅ Routed to: {mode}")
        elif mode == ConversationMode.TRIAGE:
            triage_count += 1
            print(f"  ❌ Routed to: {mode} (should be clinician)")
        else:
            print(f"  ⚠️ Routed to: {mode}")

    # Test knowledge questions
    print("\n\n📚 KNOWLEDGE QUESTIONS (should NOT trigger clinician mode):")
    thinker_count = 0

    for question in knowledge_questions:
        print(f"\nTesting: '{question}'")

        # Check if clinician trigger detects it (should be False)
        is_clinician = is_clinician_trigger(question.lower())
        print(f"  → Clinician trigger: {is_clinician}")

        if not is_clinician:
            thinker_count += 1
            print("  ✅ Correctly NOT triggering clinician mode")
        else:
            print("  ❌ Incorrectly triggering clinician mode")

    # Summary
    print("\n" + "=" * 50)
    print("📊 ROUTING TEST RESULTS:")
    print(f"Medical symptoms → Clinician mode: {clinician_count}/{len(medical_symptoms)}")
    print(f"Medical symptoms → Triage mode: {triage_count}/{len(medical_symptoms)}")
    print(f"Knowledge questions → Correctly excluded: {thinker_count}/{len(knowledge_questions)}")

    # Check results
    success = (clinician_count >= len(medical_symptoms) * 0.8 and  # At least 80% routed correctly
               triage_count == 0 and  # No medical symptoms routed to triage
               thinker_count == len(knowledge_questions))  # All knowledge questions excluded

    if success:
        print("\n✅ MEDICAL SYMPTOM ROUTING TEST PASSED!")
        print("Medical symptoms are correctly routed to enhanced clinician mode.")
    else:
        print("\n❌ MEDICAL SYMPTOM ROUTING TEST FAILED!")
        print("Some medical symptoms are not being routed to clinician mode.")

    return success

def test_enhanced_clinician_import():
    """Test that enhanced clinician can be imported"""
    print("\n🔧 TESTING ENHANCED CLINICIAN IMPORT")
    print("=" * 50)

    try:
        from enhanced_clinician import EnhancedClinicianSession
        print("✅ Enhanced clinician imported successfully")

        # Try to initialize it
        def mock_llm(messages):
            return "Mock LLM response"

        clinician = EnhancedClinicianSession("test_session", "I have chest pain", mock_llm)
        print("✅ Enhanced clinician initialized successfully")

        return True

    except ImportError as e:
        print(f"❌ Enhanced clinician import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced clinician initialization failed: {e}")
        return False

def test_feature_flags():
    """Test that feature flags are set correctly"""
    print("\n⚙️ TESTING FEATURE FLAGS")
    print("=" * 50)

    try:
        from router import USE_CLINICIAN_MODE, ENABLE_MEDICAL_SYMPTOM_ROUTING

        print(f"USE_CLINICIAN_MODE: {USE_CLINICIAN_MODE}")
        print(f"ENABLE_MEDICAL_SYMPTOM_ROUTING: {ENABLE_MEDICAL_SYMPTOM_ROUTING}")

        if USE_CLINICIAN_MODE and ENABLE_MEDICAL_SYMPTOM_ROUTING:
            print("✅ Feature flags are correctly enabled")
            return True
        else:
            print("❌ Feature flags are not properly enabled")
            return False

    except ImportError as e:
        print(f"❌ Could not import feature flags: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 MEDICAL SYMPTOM ROUTING TEST SUITE")
    print("Testing enhanced clinician mode integration")

    tests = [
        test_feature_flags,
        test_enhanced_clinician_import,
        test_medical_symptom_routing
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
        print("🎉 Enhanced clinician mode is ready for production!")
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("🔧 Please check the errors above and fix issues.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
