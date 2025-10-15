#!/usr/bin/env python3
"""
Enhanced Clinician Demo - Showing Intelligent Medical Questioning

Demonstrates how the enhanced clinician system asks appropriate follow-up questions
for symptoms like chest pain, using RAG-driven medical knowledge.
"""

import sys
import time
from typing import Callable
from pathlib import Path

# Import enhanced clinician system
sys.path.insert(0, str(Path(__file__).parent.parent / "llm-container"))
try:
    from enhanced_clinician import EnhancedClinicianSession
except ImportError:
    print("Error: Could not import enhanced clinician system")
    sys.exit(1)

def mock_llm_chat_for_chest_pain(messages) -> str:
    """
    Mock LLM that provides realistic responses for chest pain assessment
    """
    system_content = messages[0]['content'] if messages else ""

    # Opening response
    if "first most important question" in system_content.lower():
        return """OPENING: I understand you're experiencing chest pain, and I want to help assess this properly. Chest pain can have various causes and requires careful evaluation.
IMPRESSION: Based on your description, this warrants thorough assessment to determine the underlying cause.
FIRST_QUESTION: Can you describe the chest pain in more detail - where exactly is it located, and does it radiate to your arm, neck, or back?"""

    # Question generation response
    elif "next most clinically relevant question" in system_content.lower():
        # Simulate different questions based on context
        if "pressure" in system_content.lower() and "arm" in system_content.lower():
            return """QUESTION: Does the pain worsen with exertion or activity, or is it constant regardless of what you're doing?
RATIONALE: This helps determine if the chest pain is exertional, which is important for cardiac vs non-cardiac causes."""

        elif "exertion" in system_content.lower():
            return """QUESTION: On a scale of 1-10, how would you rate the severity of your chest pain right now?
RATIONALE: Severity assessment helps prioritize urgency and guides initial management decisions."""

        elif "scale" in system_content.lower():
            return """QUESTION: Have you experienced any associated symptoms like shortness of breath, nausea, sweating, or dizziness?
RATIONALE: Associated symptoms help narrow down potential causes and assess overall clinical picture."""

        elif "shortness of breath" in system_content.lower():
            return """QUESTION: Do you have any history of heart disease, high blood pressure, diabetes, or smoking?
RATIONALE: Risk factors help assess probability of cardiac causes and guide diagnostic priorities."""

        else:
            return """QUESTION: When did this chest pain first start, and has it happened before?
RATIONALE: Onset timing and recurrence pattern help differentiate acute vs chronic conditions."""

    # Assessment summary response
    elif "comprehensive assessment" in system_content.lower():
        return """ASSESSMENT: Based on your description of central chest pressure radiating to the left arm, starting during exertion, this raises concern for possible cardiac etiology given the classic presentation and risk factors.
RECOMMENDATIONS: Immediate evaluation at emergency department or urgent care, ECG, cardiac enzymes, consider stress test or angiography if indicated.
URGENCY: Emergent"""

    else:
        return """QUESTION: Can you tell me more about your symptoms?
RATIONALE: Need more information for proper assessment."""

def demonstrate_chest_pain_assessment():
    """Demonstrate intelligent chest pain assessment"""
    print("🏥 ENHANCED CLINICIAN DEMO: Chest Pain Assessment")
    print("=" * 60)

    # Initialize clinician session
    session = EnhancedClinicianSession(
        session_id="demo_chest_pain",
        chief_complaint="I have chest pain",
        llm_chat_fn=mock_llm_chat_for_chest_pain
    )

    # Simulate conversation
    print("\n📋 CHIEF COMPLAINT: I have chest pain")
    print("-" * 40)

    # First response from clinician
    response1 = session.start_enhanced_assessment()
    print(f"Doctor: {response1}")

    # Simulate patient responses
    patient_responses = [
        "The pain is in the center of my chest, feels like pressure",
        "It radiates to my left arm and neck",
        "It started about 2 hours ago during exercise",
        "I have a history of high blood pressure"
    ]

    for i, patient_response in enumerate(patient_responses, 1):
        print(f"\n📝 Exchange {i}:")
        print(f"Patient: {patient_response}")

        doctor_response = session.process_symptom_response(patient_response)
        print(f"Doctor: {doctor_response}")

        if session.assessment_complete:
            print("
✅ Assessment complete!"            break

        # Small delay for realistic conversation flow
        time.sleep(1)

    # Show final assessment findings
    print(f"\n📊 ASSESSMENT SUMMARY:")
    print(f"Urgency Level: {session.urgency_level.upper()}")
    print(f"Key Findings: {list(session.symptom_findings.keys())}")
    print(f"Risk Factors: {list(session.risk_factors.keys())}")

def mock_llm_chat_for_headache(messages) -> str:
    """Mock LLM for headache assessment"""
    system_content = messages[0]['content'] if messages else ""

    if "first most important question" in system_content.lower():
        return """OPENING: I understand you're experiencing a headache, and I'll help evaluate this systematically.
IMPRESSION: Headaches can have many causes from benign to serious, so thorough assessment is important.
FIRST_QUESTION: Can you describe the headache - is it on one side, both sides, and do you have any associated symptoms like nausea or vision changes?"""

    elif "next most clinically relevant question" in system_content.lower():
        if "one side" in system_content.lower():
            return """QUESTION: How severe is the headache on a scale of 1-10, and does it worsen with light, noise, or movement?
RATIONALE: Severity and aggravating factors help assess migraine vs other headache types."""

        elif "nausea" in system_content.lower():
            return """QUESTION: Have you had any recent head trauma, vision changes, weakness, or neurological symptoms?
RATIONALE: Red flag symptoms help rule out serious causes like bleeding or infection."""

        else:
            return """QUESTION: How long have you had this headache, and does it respond to any medications?
RATIONALE: Duration and treatment response help differentiate acute vs chronic conditions."""

    elif "comprehensive assessment" in system_content.lower():
        return """ASSESSMENT: Your unilateral headache with nausea suggests possible migraine, though other causes should be considered if red flags are present.
RECOMMENDATIONS: Rest in dark quiet room, consider migraine medications if diagnosed before, follow up with primary care if persistent.
URGENCY: Routine"""

    else:
        return """QUESTION: Can you describe your headache in more detail?
RATIONALE: Need more information for proper assessment."""

def demonstrate_headache_assessment():
    """Demonstrate intelligent headache assessment"""
    print("\n" + "=" * 60)
    print("🏥 ENHANCED CLINICIAN DEMO: Headache Assessment")
    print("=" * 60)

    session = EnhancedClinicianSession(
        session_id="demo_headache",
        chief_complaint="I have a headache",
        llm_chat_fn=mock_llm_chat_for_headache
    )

    print("\n📋 CHIEF COMPLAINT: I have a headache")
    print("-" * 40)

    response1 = session.start_enhanced_assessment()
    print(f"Doctor: {response1}")

    patient_responses = [
        "It's on one side of my head",
        "I feel nauseous with it",
        "It's about a 7 out of 10 severity",
        "It gets worse with light and noise"
    ]

    for i, patient_response in enumerate(patient_responses, 1):
        print(f"\n📝 Exchange {i}:")
        print(f"Patient: {patient_response}")

        doctor_response = session.process_symptom_response(patient_response)
        print(f"Doctor: {doctor_response}")

        if session.assessment_complete:
            break

        time.sleep(1)

    print(f"\n📊 ASSESSMENT SUMMARY:")
    print(f"Urgency Level: {session.urgency_level.upper()}")
    print(f"Key Findings: {list(session.symptom_findings.keys())}")

def demonstrate_breathing_assessment():
    """Demonstrate intelligent shortness of breath assessment"""
    print("\n" + "=" * 60)
    print("🏥 ENHANCED CLINICIAN DEMO: Shortness of Breath Assessment")
    print("=" * 60)

    def mock_llm_chat_for_breathing(messages) -> str:
        system_content = messages[0]['content'] if messages else ""

        if "first most important question" in system_content.lower():
            return """OPENING: I understand you're experiencing shortness of breath, which can be concerning. Let's evaluate this systematically.
IMPRESSION: Respiratory symptoms require careful assessment to determine cause and severity.
FIRST_QUESTION: When did the shortness of breath start, and is it worse when lying down or with activity?"""

        elif "next most clinically relevant question" in system_content.lower():
            if "lying down" in system_content.lower():
                return """QUESTION: Do you have any chest pain, swelling in your legs, or history of heart problems?
RATIONALE: Orthopnea with associated symptoms suggests possible cardiac causes."""

            elif "activity" in system_content.lower():
                return """QUESTION: On a scale of 1-10, how severe is your shortness of breath, and can you walk up stairs?
RATIONALE: Functional assessment helps determine impact on daily activities."""

            else:
                return """QUESTION: Have you had any fever, cough, recent travel, or sick contacts?
RATIONALE: Recent illness or exposures help assess infectious vs other causes."""

        elif "comprehensive assessment" in system_content.lower():
            return """ASSESSMENT: Your shortness of breath with orthopnea suggests possible cardiac or pulmonary etiology requiring evaluation.
RECOMMENDATIONS: Seek medical evaluation, consider chest X-ray, ECG, possible echocardiography if cardiac concerns.
URGENCY: Urgent"""

        else:
            return """QUESTION: Can you describe your breathing difficulty in more detail?
RATIONALE: Need more information for proper assessment."""

    session = EnhancedClinicianSession(
        session_id="demo_breathing",
        chief_complaint="I have shortness of breath",
        llm_chat_fn=mock_llm_chat_for_breathing
    )

    print("\n📋 CHIEF COMPLAINT: I have shortness of breath")
    print("-" * 40)

    response1 = session.start_enhanced_assessment()
    print(f"Doctor: {response1}")

    patient_responses = [
        "It started yesterday and is worse when lying down",
        "I also have some chest discomfort",
        "It's about a 6 out of 10 severity",
        "I have a history of heart problems"
    ]

    for i, patient_response in enumerate(patient_responses, 1):
        print(f"\n📝 Exchange {i}:")
        print(f"Patient: {patient_response}")

        doctor_response = session.process_symptom_response(patient_response)
        print(f"Doctor: {doctor_response}")

        if session.assessment_complete:
            break

        time.sleep(1)

    print(f"\n📊 ASSESSMENT SUMMARY:")
    print(f"Urgency Level: {session.urgency_level.upper()}")
    print(f"Key Findings: {list(session.symptom_findings.keys())}")

def demonstrate_comparison_with_rigid_triage():
    """Show the difference between rigid triage and enhanced clinician"""
    print("\n" + "=" * 60)
    print("🔍 COMPARISON: Rigid Triage vs Enhanced Clinician")
    print("=" * 60)

    print("\n📋 RIGID TRIAGE APPROACH:")
    print("❌ Predefined questions in fixed order")
    print("❌ No understanding of symptom relationships")
    print("❌ Cannot adapt based on patient responses")
    print("❌ Generic questions for all conditions")
    print("❌ No clinical reasoning or differential diagnosis")

    print("\n✅ ENHANCED CLINICIAN APPROACH:")
    print("✅ Context-aware questions based on symptoms")
    print("✅ Adapts questioning based on findings")
    print("✅ Uses medical knowledge for appropriate follow-ups")
    print("✅ Considers urgency and clinical significance")
    print("✅ Provides differential diagnosis and recommendations")

    print("\n🎯 KEY ADVANTAGES FOR CHEST PAIN:")
    print("• Recognizes cardiac risk factors automatically")
    print("• Prioritizes questions based on clinical urgency")
    print("• Considers radiation patterns and quality of pain")
    print("• Assesses associated symptoms appropriately")
    print("• Provides evidence-based urgency determination")

def main():
    """Run all demonstrations"""
    print("🚀 ENHANCED CLINICIAN SYSTEM DEMONSTRATION")
    print("Showing intelligent medical questioning for various symptoms")

    try:
        # Chest pain demo (most critical)
        demonstrate_chest_pain_assessment()

        # Headache demo (common complaint)
        demonstrate_headache_assessment()

        # Breathing demo (respiratory concern)
        demonstrate_breathing_assessment()

        # Comparison
        demonstrate_comparison_with_rigid_triage()

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe enhanced clinician system demonstrates:")
        print("• Intelligent, context-aware medical questioning")
        print("• Appropriate follow-up based on clinical findings")
        print("• Evidence-based urgency assessment")
        print("• Physician-like differential diagnosis")
        print("• RAG-driven medical knowledge integration")

        print("\n💡 This system can effectively replace rigid triage.py")
        print("with sophisticated, adaptive medical assessment.")

    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
