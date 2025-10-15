#!/usr/bin/env python3
"""
Enhanced Clinician Mode - RAG-Driven Medical Assessment

Replaces rigid triage.py with sophisticated physician-like assessment:
- Intelligent symptom analysis using medical RAG
- Context-aware follow-up questions
- Differential diagnosis generation
- Evidence-based recommendations
- Adaptive questioning based on findings

Perfect for symptoms like chest pain, where appropriate follow-up questions are crucial.
"""

import os
import sys
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from difflib import SequenceMatcher
from pathlib import Path

# Import medical RAG system (defensive imports)
sys.path.append(str(Path(__file__).parent.parent))
try:
    from clinician_rag import ClinicianRAG, search_clinician_info
    MEDICAL_RAG_AVAILABLE = True
except ImportError:
    MEDICAL_RAG_AVAILABLE = False
    ClinicianRAG = None
    search_clinician_info = None
    print("Warning: Could not import medical RAG modules")

try:
    from medical_data_ingestion import MedicalDataIngester
    MEDICAL_DATA_AVAILABLE = True
except ImportError:
    MEDICAL_DATA_AVAILABLE = False
    MedicalDataIngester = None
    print("Warning: Could not import medical data ingester")

class EnhancedClinicianSession:
    """
    Advanced clinician session with RAG-driven medical reasoning
    """

    def __init__(self, session_id: str, chief_complaint: str, llm_chat_fn: Callable):
        self.session_id = session_id
        self.chief_complaint = chief_complaint
        self.llm_chat_fn = llm_chat_fn

        # Enhanced clinical state
        self.conversation_history = []
        self.symptom_findings = {}  # Detailed symptom analysis
        self.physical_findings = {}  # Physical exam findings
        self.risk_factors = {}      # Patient risk factors
        self.vital_signs = {}       # Vital signs data
        self.medications = []       # Current medications
        self.allergies = []         # Known allergies
        self.past_medical_history = []  # PMH

        # Differential diagnosis tracking
        self.differential_diagnoses = []
        self.current_focus = None   # Current diagnostic focus
        self.urgency_level = "routine"  # routine, urgent, emergent

        # RAG integration
        self.clinician_rag = None
        self._initialize_medical_rag()

        print(f"[Enhanced Clinician] 🩺 Enhanced clinician initialized (RAG: {MEDICAL_RAG_AVAILABLE})")

        # Question generation state
        self.questions_asked = []
        self.pending_questions = []
        self.assessment_complete = False

        print(f"[Enhanced Clinician] 🩺 Starting enhanced diagnostic session for: '{chief_complaint}'")

    def _initialize_medical_rag(self):
        """Initialize medical RAG system"""
        if not MEDICAL_RAG_AVAILABLE:
            print("[Enhanced Clinician] ⚠️ Medical RAG not available")
            return

        try:
            if ClinicianRAG is not None:
                self.clinician_rag = ClinicianRAG()
                print("[Enhanced Clinician] ✅ Medical RAG initialized")
        except Exception as e:
            print(f"[Enhanced Clinician] ⚠️ Could not initialize medical RAG: {e}")
            self.clinician_rag = None

    def start_enhanced_assessment(self) -> str:
        """
        Start intelligent medical assessment with RAG-driven questioning

        Returns:
            Opening statement with first intelligent question
        """
        print(f"[Enhanced Clinician] 🔍 Analyzing chief complaint: '{self.chief_complaint}'")

        # Use medical RAG to understand the chief complaint
        rag_context = self._get_medical_context_for_symptom(self.chief_complaint)

        # Generate intelligent opening response
        opening_prompt = f"""
You are an experienced physician conducting a medical assessment. The patient reports: "{self.chief_complaint}"

Based on current medical knowledge and guidelines, provide:
1. A professional, empathetic opening statement
2. Your initial clinical impression
3. The first most important question to ask for proper assessment

Available medical context:
{rag_context}

Respond in this format:
OPENING: [Your empathetic opening statement]
IMPRESSION: [Brief clinical impression]
FIRST_QUESTION: [Specific, medically-relevant question]
"""

        try:
            response = self.llm_chat_fn([{"role": "system", "content": opening_prompt}])
            return self._parse_opening_response(response)
        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error generating opening: {e}")
            return self._fallback_opening()

    def process_symptom_response(self, user_response: str) -> str:
        """
        Process user's response and generate next intelligent question

        Args:
            user_response: Patient's response to previous question

        Returns:
            Next question or assessment summary
        """
        # Store the response
        self.conversation_history.append({
            'role': 'patient',
            'content': user_response,
            'timestamp': datetime.now().isoformat()
        })

        # Analyze the response for clinical findings
        self._analyze_patient_response(user_response)

        # If we have enough information, provide assessment
        if self._should_complete_assessment():
            return self._generate_assessment_summary()

        # Otherwise, generate next intelligent question
        return self._generate_next_question()

    def _get_medical_context_for_symptom(self, symptom: str) -> str:
        """Get relevant medical context for a symptom using RAG"""
        if not self.clinician_rag:
            return "No medical context available (RAG not initialized)"

        try:
            # Search for symptom-specific information
            if hasattr(self.clinician_rag, 'search_medical_info'):
                results = self.clinician_rag.search_medical_info(symptom, k=3)

                if results:
                    if hasattr(self.clinician_rag, 'get_medical_context'):
                        context = self.clinician_rag.get_medical_context(symptom, results)
                        return context[:1000]  # Limit context length
                    else:
                        return f"Found {len(results)} relevant medical results"
                else:
                    return "Limited medical context found for this symptom"
            else:
                return "Medical RAG search not available"

        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error getting medical context: {e}")
            return "Error retrieving medical context"

    def _parse_opening_response(self, response: str) -> str:
        """Parse LLM response to extract opening statement and first question"""
        try:
            lines = response.split('\n')
            opening = ""
            impression = ""
            first_question = ""

            for line in lines:
                line = line.strip()
                if line.startswith('OPENING:'):
                    opening = line[8:].strip()
                elif line.startswith('IMPRESSION:'):
                    impression = line[11:].strip()
                elif line.startswith('FIRST_QUESTION:'):
                    first_question = line[15:].strip()

            if first_question:
                # Store the question for tracking
                self.questions_asked.append(first_question)

                return f"{opening}\n\n{impression}\n\n{first_question}"
            else:
                return self._fallback_opening()

        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error parsing response: {e}")
            return self._fallback_opening()

    def _fallback_opening(self) -> str:
        """Fallback opening when RAG is unavailable"""
        fallback_questions = {
            "chest pain": "Can you describe the chest pain - where exactly is it located, and does it radiate to your arm, neck, or back?",
            "headache": "Can you describe the headache - is it on one side, both sides, and do you have any associated symptoms like nausea or vision changes?",
            "abdominal pain": "Can you point to where the abdominal pain is located and tell me if it's constant or comes and goes?",
            "shortness of breath": "When did the shortness of breath start, and is it worse when lying down or with activity?",
            "dizziness": "Can you describe the dizziness - is it spinning, lightheadedness, and do you feel like you might fall?"
        }

        # Find matching fallback question
        for symptom, question in fallback_questions.items():
            if symptom in self.chief_complaint.lower():
                return f"I understand you're experiencing {self.chief_complaint}. To help assess this properly, {question}"

        # Generic fallback
        return f"I understand you're experiencing {self.chief_complaint}. To provide you with the best care, I need to ask some questions. Can you describe this symptom in more detail?"

    def _analyze_patient_response(self, response: str):
        """Analyze patient response for clinical findings"""
        response_lower = response.lower()

        # Extract symptom characteristics
        symptom_patterns = {
            'chest_pain': {
                'location': ['left', 'right', 'center', 'across', 'middle'],
                'quality': ['sharp', 'dull', 'burning', 'crushing', 'pressure', 'tightness'],
                'radiation': ['arm', 'neck', 'back', 'jaw', 'shoulder'],
                'triggers': ['exertion', 'eating', 'breathing', 'movement'],
                'severity': ['mild', 'moderate', 'severe', 'worst', 'unbearable']
            },
            'respiratory': {
                'sob': ['shortness of breath', 'difficulty breathing', 'breathless'],
                'cough': ['cough', 'coughing'],
                'sputum': ['phlegm', 'sputum', 'mucus']
            },
            'cardiac': {
                'palpitations': ['racing heart', 'palpitations', 'irregular heartbeat'],
                'syncope': ['fainting', 'passing out', 'loss of consciousness']
            }
        }

        # Analyze based on chief complaint
        if 'chest' in self.chief_complaint.lower() and 'pain' in self.chief_complaint.lower():
            self._analyze_chest_pain_response(response_lower)
        elif 'breath' in self.chief_complaint.lower():
            self._analyze_respiratory_response(response_lower)
        elif 'headache' in self.chief_complaint.lower():
            self._analyze_headache_response(response_lower)

        # Extract vital information
        self._extract_vital_information(response_lower)

    def _analyze_chest_pain_response(self, response: str):
        """Analyze response for chest pain characteristics"""
        findings = {}

        # Location analysis
        if any(word in response for word in ['left', 'right', 'center']):
            findings['location'] = 'specific'
        elif 'across' in response or 'middle' in response:
            findings['location'] = 'central'

        # Quality analysis
        if any(word in response for word in ['sharp', 'stabbing']):
            findings['quality'] = 'sharp'
        elif any(word in response for word in ['dull', 'ache']):
            findings['quality'] = 'dull'
        elif any(word in response for word in ['pressure', 'tightness', 'crushing']):
            findings['quality'] = 'pressure'

        # Radiation
        if any(word in response for word in ['arm', 'neck', 'back', 'jaw', 'shoulder']):
            findings['radiation'] = 'present'

        # Triggers
        if 'exertion' in response or 'activity' in response:
            findings['exertional'] = True

        # Duration
        duration_patterns = [
            (r'(\d+)\s*(minute|min)', 'minutes'),
            (r'(\d+)\s*(hour|hr)', 'hours'),
            (r'(\d+)\s*(day|days)', 'days')
        ]

        for pattern, unit in duration_patterns:
            match = re.search(pattern, response)
            if match:
                findings['duration'] = f"{match.group(1)} {unit}"

        self.symptom_findings.update(findings)

        # Update urgency based on findings
        if findings.get('quality') == 'pressure' and findings.get('radiation') == 'present':
            self.urgency_level = 'urgent'
        elif 'exertional' in findings:
            self.urgency_level = 'urgent'

    def _analyze_respiratory_response(self, response: str):
        """Analyze respiratory symptom response"""
        findings = {}

        if 'shortness of breath' in response or 'difficulty breathing' in response:
            findings['sob_severity'] = 'significant'

        if 'lying down' in response:
            findings['orthopnea'] = True

        if 'activity' in response or 'exertion' in response:
            findings['exertional_dyspnea'] = True

        self.symptom_findings.update(findings)

    def _analyze_headache_response(self, response: str):
        """Analyze headache response"""
        findings = {}

        if 'one side' in response or 'unilateral' in response:
            findings['unilateral'] = True

        if 'nausea' in response or 'vomiting' in response:
            findings['associated_nausea'] = True

        if 'vision' in response:
            findings['visual_changes'] = True

        self.symptom_findings.update(findings)

    def _extract_vital_information(self, response: str):
        """Extract vital signs and other key information"""
        # Age
        age_match = re.search(r'(\d{2})\s*years?\s*old', response)
        if age_match:
            self.vital_signs['age'] = int(age_match.group(1))

        # Gender
        if 'male' in response or 'man' in response:
            self.vital_signs['gender'] = 'male'
        elif 'female' in response or 'woman' in response:
            self.vital_signs['gender'] = 'female'

        # Risk factors
        risk_factors = ['smoking', 'diabetes', 'hypertension', 'high cholesterol', 'family history']
        for risk in risk_factors:
            if risk in response:
                self.risk_factors[risk] = True

    def _should_complete_assessment(self) -> bool:
        """Determine if we have sufficient information for assessment"""
        # Basic criteria for completing assessment
        min_responses = 3  # Need at least 3 exchanges
        has_key_findings = len(self.symptom_findings) >= 2

        return (len(self.conversation_history) >= min_responses and has_key_findings) or self.assessment_complete

    def _generate_next_question(self) -> str:
        """Generate next intelligent question using medical RAG"""
        # Build context from current findings
        context = self._build_assessment_context()

        # Get relevant medical information
        medical_context = ""
        if self.clinician_rag and hasattr(self.clinician_rag, 'search_medical_info'):
            try:
                results = self.clinician_rag.search_medical_info(context, k=3)
                if results and hasattr(self.clinician_rag, 'get_medical_context'):
                    medical_context = self.clinician_rag.get_medical_context(context, results)[:800]
            except Exception as e:
                print(f"[Enhanced Clinician] ❌ Error getting medical context: {e}")

        # Generate next question using LLM + medical context
        question_prompt = f"""
You are an experienced physician conducting a medical assessment. Based on the current findings:

CHIEF COMPLAINT: {self.chief_complaint}
CURRENT FINDINGS: {json.dumps(self.symptom_findings, indent=2)}
VITAL SIGNS: {json.dumps(self.vital_signs, indent=2)}
RISK FACTORS: {json.dumps(self.risk_factors, indent=2)}

MEDICAL CONTEXT:
{medical_context}

QUESTIONS ALREADY ASKED:
{chr(10).join(self.questions_asked[-3:])}

Generate the next most clinically relevant question. Consider:
1. What information is missing for proper assessment?
2. What would help differentiate between possible diagnoses?
3. What red flag symptoms should be ruled out?

Provide your response in this format:
QUESTION: [Your specific, medically-relevant question]
RATIONALE: [Brief explanation of why this question is important]
"""

        try:
            response = self.llm_chat_fn([{"role": "system", "content": question_prompt}])
            question, rationale = self._parse_question_response(response)

            if question:
                self.questions_asked.append(question)
                return f"{question}\n\n[Medical rationale: {rationale}]"
            else:
                return self._fallback_question()

        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error generating question: {e}")
            return self._fallback_question()

    def _parse_question_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract question and rationale"""
        try:
            lines = response.split('\n')
            question = ""
            rationale = ""

            for line in lines:
                line = line.strip()
                if line.startswith('QUESTION:'):
                    question = line[9:].strip()
                elif line.startswith('RATIONALE:'):
                    rationale = line[10:].strip()

            return question, rationale

        except Exception:
            return "", ""

    def _fallback_question(self) -> str:
        """Fallback question when LLM fails"""
        fallback_questions = [
            "Can you rate the severity of your symptoms on a scale of 1-10?",
            "When did these symptoms first start?",
            "Have you experienced these symptoms before?",
            "Are there any factors that make the symptoms better or worse?",
            "Do you have any other symptoms along with this?",
            "Have you seen a doctor about this before?"
        ]

        # Avoid repeating recent questions
        recent_questions = set(self.questions_asked[-2:]) if len(self.questions_asked) >= 2 else set()

        for question in fallback_questions:
            if question not in recent_questions:
                return question

        return "Can you tell me more about how this is affecting you?"

    def _build_assessment_context(self) -> str:
        """Build context string for medical RAG queries"""
        context_parts = [self.chief_complaint]

        if self.symptom_findings:
            context_parts.append(f"Findings: {', '.join(self.symptom_findings.keys())}")

        if self.risk_factors:
            context_parts.append(f"Risk factors: {', '.join(self.risk_factors.keys())}")

        if self.vital_signs:
            context_parts.append(f"Vital signs: {', '.join(f'{k}={v}' for k, v in self.vital_signs.items())}")

        return " ".join(context_parts)

    def _generate_assessment_summary(self) -> str:
        """Generate comprehensive assessment summary"""
        self.assessment_complete = True

        # Generate differential diagnosis using medical RAG
        differential = self._generate_differential_diagnosis()

        # Create comprehensive assessment
        summary_prompt = f"""
You are an experienced physician providing a medical assessment summary.

PATIENT CHIEF COMPLAINT: {self.chief_complaint}

CLINICAL FINDINGS:
- Symptoms: {json.dumps(self.symptom_findings, indent=2)}
- Risk Factors: {json.dumps(self.risk_factors, indent=2)}
- Vital Signs: {json.dumps(self.vital_signs, indent=2)}

DIFFERENTIAL DIAGNOSIS:
{differential}

Provide a comprehensive assessment including:
1. Primary concerns
2. Recommended next steps
3. Urgency level
4. Patient education

Format your response as:
ASSESSMENT: [Your clinical assessment]
RECOMMENDATIONS: [Specific recommendations]
URGENCY: [Routine/Urgent/Emergent]
"""

        try:
            response = self.llm_chat_fn([{"role": "system", "content": summary_prompt}])
            return self._format_assessment_response(response)
        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error generating summary: {e}")
            return self._fallback_assessment()

    def _generate_differential_diagnosis(self) -> str:
        """Generate differential diagnosis using medical RAG"""
        if not self.clinician_rag or not hasattr(self.clinician_rag, 'search_medical_info'):
            return "Differential diagnosis not available (RAG not initialized)"

        try:
            context = self._build_assessment_context()
            results = self.clinician_rag.search_medical_info(context, k=5)

            if results and hasattr(self.clinician_rag, 'get_medical_context'):
                return self.clinician_rag.get_medical_context(context, results)[:600]
            else:
                return "Limited differential diagnosis information available"

        except Exception as e:
            print(f"[Enhanced Clinician] ❌ Error generating differential: {e}")
            return "Error generating differential diagnosis"

    def _format_assessment_response(self, response: str) -> str:
        """Format the assessment response for patient communication"""
        try:
            lines = response.split('\n')
            assessment = ""
            recommendations = ""
            urgency = "routine"

            for line in lines:
                line = line.strip()
                if line.startswith('ASSESSMENT:'):
                    assessment = line[11:].strip()
                elif line.startswith('RECOMMENDATIONS:'):
                    recommendations = line[16:].strip()
                elif line.startswith('URGENCY:'):
                    urgency = line[8:].strip().lower()

            # Create patient-friendly response
            urgency_text = {
                'routine': 'can be addressed through regular medical care',
                'urgent': 'should be evaluated by a healthcare provider within 24-48 hours',
                'emergent': 'requires immediate medical attention'
            }

            response_parts = []
            if assessment:
                response_parts.append(f"Based on our conversation, here's my assessment:\n\n{assessment}")

            if recommendations:
                response_parts.append(f"\nRecommendations:\n{recommendations}")

            response_parts.append(f"\nUrgency Level: This {urgency_text.get(urgency, 'requires medical evaluation')}.")

            if urgency == 'emergent':
                response_parts.append("\n⚠️ EMERGENCY: Please seek immediate medical care or call emergency services.")

            return "\n".join(response_parts)

        except Exception as e:
            return self._fallback_assessment()

    def _fallback_assessment(self) -> str:
        """Fallback assessment when LLM fails"""
        urgency_msg = {
            'emergent': "\n⚠️ EMERGENCY: Please seek immediate medical care.",
            'urgent': "\nPlease see a healthcare provider within 24-48 hours.",
            'routine': "\nThis can be addressed through regular medical care."
        }

        return f"""Based on your description of {self.chief_complaint}, I recommend discussing these symptoms with your healthcare provider.

Key findings to share:
- {', '.join(self.symptom_findings.keys()) if self.symptom_findings else 'Symptoms as described'}

{urgency_msg.get(self.urgency_level, '')}

Please consult with a medical professional for proper evaluation and diagnosis."""

def run_enhanced_clinician_assessment(chief_complaint: str, llm_chat_fn: Callable, session_id: str = None) -> str:
    """
    Run complete enhanced clinician assessment

    Args:
        chief_complaint: Patient's chief complaint
        llm_chat_fn: LLM chat function
        session_id: Optional session identifier

    Returns:
        Complete assessment response
    """
    session = EnhancedClinicianSession(session_id or f"session_{int(time.time())}", chief_complaint, llm_chat_fn)

    # Start assessment
    response1 = session.start_enhanced_assessment()
    print(f"Doctor: {response1}")

    # Simulate conversation (in real implementation, this would be interactive)
    # For demonstration, we'll simulate a chest pain assessment

    if "chest pain" in chief_complaint.lower():
        # Simulate patient responses for chest pain
        patient_responses = [
            "The pain is in the center of my chest, feels like pressure",
            "It radiates to my left arm and neck",
            "It started about 2 hours ago during exercise",
            "I have a history of high blood pressure"
        ]

        for patient_response in patient_responses:
            print(f"\nPatient: {patient_response}")
            doctor_response = session.process_symptom_response(patient_response)
            print(f"Doctor: {doctor_response}")

            if session.assessment_complete:
                break

    return "Enhanced clinician assessment complete"

if __name__ == "__main__":
    # Test the enhanced clinician system
    def mock_llm_chat(messages):
        # Mock LLM response for testing
        return """OPENING: I understand you're experiencing chest pain, and I want to help assess this properly.
IMPRESSION: Chest pain can have various causes and requires careful evaluation.
FIRST_QUESTION: Can you describe the chest pain - where exactly is it located, and does it radiate to your arm, neck, or back?"""

    test_complaint = "I have chest pain"
    print(f"Testing enhanced clinician with: {test_complaint}")
    print("-" * 50)

    result = run_enhanced_clinician_assessment(test_complaint, mock_llm_chat)
    print(f"\n{result}")
