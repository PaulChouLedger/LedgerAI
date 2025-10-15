#!/usr/bin/env python3
"""
Dynamic Medical Assistant - RAG-Powered Clinical Assessment

Uses real medical guidelines retrieved from RAG to dynamically assess patients,
rather than following rigid decision trees.

Architecture:
1. User reports chief complaint
2. RAG retrieves relevant guidelines
3. LLM generates contextual questions based on guidelines
4. Tracks symptoms, red flags, and urgency
5. Generates diagnosis, urgency level, and disposition
"""

import re
import json
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class MedicalAssessmentState:
    """State tracking for dynamic medical assessment"""
    session_id: str
    chief_complaint: str = ""
    symptoms_collected: List[Dict] = field(default_factory=list)
    red_flags_detected: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    responses_received: List[str] = field(default_factory=list)
    differential_diagnosis: Dict[str, float] = field(default_factory=dict)  # diagnosis -> confidence
    urgency_score: float = 0.0  # 0-10 scale
    category: str = "unknown"
    guidelines_used: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed: bool = False

class DynamicMedicalAssistant:
    """
    RAG-powered dynamic medical assessment system
    
    Replaces rigid triage with intelligent, guideline-based questioning
    """
    
    def __init__(self, rag_service_url: str = "http://localhost:11435"):
        self.rag_url = rag_service_url
        self.active_sessions: Dict[str, MedicalAssessmentState] = {}
        
        # Urgency thresholds
        self.URGENCY_EMERGENCY = 8.0  # Call 911
        self.URGENCY_URGENT = 6.0     # ER within 1-2 hours
        self.URGENCY_SEMI_URGENT = 4.0  # See doctor within 24 hours
        # Below 4.0 = routine (schedule appointment)
        
        print("[DynamicMed] ✅ Dynamic Medical Assistant initialized")
    
    def start_assessment(self, session_id: str, chief_complaint: str) -> MedicalAssessmentState:
        """
        Start a new dynamic medical assessment
        
        Args:
            session_id: Unique session identifier
            chief_complaint: User's initial complaint (e.g., "I have chest pain")
            
        Returns:
            Assessment state object
        """
        state = MedicalAssessmentState(
            session_id=session_id,
            chief_complaint=chief_complaint
        )
        
        # Extract primary symptom and category
        state.category = self._categorize_complaint(chief_complaint)
        
        self.active_sessions[session_id] = state
        
        print(f"[DynamicMed] 🏥 Starting assessment for: {chief_complaint}")
        print(f"[DynamicMed]    Category: {state.category}")
        
        return state
    
    def _categorize_complaint(self, complaint: str) -> str:
        """Categorize complaint by medical specialty"""
        complaint_lower = complaint.lower()
        
        categories = {
            'cardiovascular': ['chest pain', 'heart', 'palpitation', 'shortness of breath'],
            'respiratory': ['cough', 'breathing', 'dyspnea', 'wheezing'],
            'gastrointestinal': ['stomach', 'abdominal', 'nausea', 'vomit', 'diarrhea', 'pancreatitis'],
            'neurological': ['headache', 'dizzy', 'seizure', 'numbness', 'weakness'],
            'musculoskeletal': ['back pain', 'joint', 'muscle', 'bone'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in complaint_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def get_relevant_guidelines(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant medical guidelines from RAG
        
        Args:
            query: Medical query or symptom description
            category: Optional category filter
            
        Returns:
            List of relevant guideline chunks from RAG
        """
        try:
            # Add category to query for better retrieval
            enhanced_query = query
            if category and category != 'general':
                enhanced_query = f"{category} medical guidelines: {query}"
            
            response = requests.post(
                f"{self.rag_url}/rag/search",
                json={"query": enhanced_query, "top_k": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                print(f"[DynamicMed] 📚 Retrieved {len(results)} guideline chunks from RAG")
                return results
            else:
                print(f"[DynamicMed] ⚠️ RAG search failed: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[DynamicMed] ❌ Error retrieving guidelines: {e}")
            return []
    
    def generate_next_question(self, 
                               session_id: str, 
                               llm_chat_fn) -> Optional[str]:
        """
        Generate the next diagnostic question based on current state and guidelines
        
        Uses RAG-retrieved guidelines + LLM to intelligently determine
        what to ask next
        
        Args:
            session_id: Session identifier
            llm_chat_fn: LLM chat function for question generation
            
        Returns:
            Next question to ask, or None if assessment complete
        """
        state = self.active_sessions.get(session_id)
        if not state:
            return None
        
        # Build context from current state
        context_summary = self._build_assessment_context(state)
        
        # Retrieve relevant guidelines
        search_query = f"{state.chief_complaint} {' '.join([s['symptom'] for s in state.symptoms_collected])}"
        guidelines = self.get_relevant_guidelines(search_query, state.category)
        
        if not guidelines:
            print("[DynamicMed] ⚠️ No guidelines found - using general medical knowledge")
        
        # Extract guideline content
        guideline_text = "\n\n".join([g['text'] for g in guidelines[:3]])  # Top 3 chunks
        
        # Prompt LLM to generate next question
        prompt = f"""You are a medical professional conducting a patient assessment. 

CHIEF COMPLAINT: {state.chief_complaint}
CATEGORY: {state.category}

SYMPTOMS COLLECTED SO FAR:
{self._format_symptoms(state.symptoms_collected)}

RED FLAGS DETECTED: {', '.join(state.red_flags_detected) if state.red_flags_detected else 'None'}

RELEVANT MEDICAL GUIDELINES:
{guideline_text}

QUESTIONS ALREADY ASKED:
{self._format_questions(state.questions_asked, state.responses_received)}

Based on the medical guidelines and current information, generate the SINGLE MOST IMPORTANT next question to ask the patient.

RULES:
1. Ask ONE clear, specific question
2. Focus on high-yield diagnostic information
3. Prioritize red flag symptoms if not already assessed
4. Don't repeat questions already asked
5. Use simple, patient-friendly language
6. If sufficient information gathered, respond with: "ASSESSMENT_COMPLETE"

NEXT QUESTION:"""

        # Get LLM response
        response = llm_chat_fn([{"role": "user", "content": prompt}])
        
        # Extract question
        question = response.strip()
        
        # Check if assessment is complete
        if "ASSESSMENT_COMPLETE" in question.upper():
            state.completed = True
            return None
        
        # Clean up question
        question = question.replace("NEXT QUESTION:", "").strip()
        
        # Track question
        state.questions_asked.append(question)
        
        print(f"[DynamicMed] 🔍 Generated question: {question}")
        
        return question
    
    def process_response(self, 
                        session_id: str, 
                        user_response: str,
                        llm_chat_fn) -> Dict:
        """
        Process user's response and update assessment state
        
        Args:
            session_id: Session identifier
            user_response: User's answer to the last question
            llm_chat_fn: LLM function for analysis
            
        Returns:
            Dict with extracted information and updated urgency
        """
        state = self.active_sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}
        
        # Store response
        state.responses_received.append(user_response)
        
        # Analyze response for symptoms and red flags
        analysis = self._analyze_response(user_response, state, llm_chat_fn)
        
        # Update state
        if analysis.get('symptoms'):
            state.symptoms_collected.extend(analysis['symptoms'])
        
        if analysis.get('red_flags'):
            state.red_flags_detected.extend(analysis['red_flags'])
            # Red flags increase urgency
            state.urgency_score += len(analysis['red_flags']) * 2.0
        
        # Update urgency based on response severity
        state.urgency_score += analysis.get('urgency_delta', 0.0)
        
        # Cap urgency at 10
        state.urgency_score = min(state.urgency_score, 10.0)
        
        print(f"[DynamicMed] 📊 Updated state:")
        print(f"[DynamicMed]    Symptoms: {len(state.symptoms_collected)}")
        print(f"[DynamicMed]    Red flags: {len(state.red_flags_detected)}")
        print(f"[DynamicMed]    Urgency: {state.urgency_score:.1f}/10")
        
        return analysis
    
    def _analyze_response(self, 
                         response: str, 
                         state: MedicalAssessmentState,
                         llm_chat_fn) -> Dict:
        """
        Analyze user response to extract medical information
        
        Uses LLM to extract symptoms, severity, red flags
        """
        prompt = f"""Analyze this patient response for medical information:

QUESTION ASKED: {state.questions_asked[-1] if state.questions_asked else 'Initial complaint'}
PATIENT RESPONSE: {response}
CHIEF COMPLAINT: {state.chief_complaint}

Extract the following (respond in JSON format):
1. Symptoms mentioned (list)
2. Red flags/emergency indicators (list)
3. Severity level (1-10)
4. Urgency delta (how much this increases urgency: -2 to +3)

Example response:
{{
  "symptoms": ["crushing chest pain", "radiating to left arm"],
  "red_flags": ["diaphoresis", "shortness of breath"],
  "severity": 8,
  "urgency_delta": 2.5
}}

JSON RESPONSE:"""

        llm_response = llm_chat_fn([{"role": "user", "content": prompt}])
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                return analysis
            else:
                print("[DynamicMed] ⚠️ Could not parse LLM analysis")
                return {"symptoms": [], "red_flags": [], "severity": 5, "urgency_delta": 0.0}
        except Exception as e:
            print(f"[DynamicMed] ❌ Error parsing analysis: {e}")
            return {"symptoms": [], "red_flags": [], "severity": 5, "urgency_delta": 0.0}
    
    def generate_diagnosis_and_disposition(self, 
                                           session_id: str,
                                           llm_chat_fn) -> Dict:
        """
        Generate final diagnosis, urgency assessment, and disposition
        
        Args:
            session_id: Session identifier
            llm_chat_fn: LLM function for diagnosis generation
            
        Returns:
            Dict with diagnosis, urgency, disposition
        """
        state = self.active_sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}
        
        # Retrieve comprehensive guidelines for diagnosis
        search_query = f"{state.chief_complaint} diagnosis differential {' '.join([s.get('symptom', '') for s in state.symptoms_collected])}"
        guidelines = self.get_relevant_guidelines(search_query, state.category)
        guideline_text = "\n\n".join([g['text'] for g in guidelines[:5]])
        
        # Build comprehensive assessment summary
        assessment_summary = f"""CHIEF COMPLAINT: {state.chief_complaint}

SYMPTOMS COLLECTED:
{self._format_symptoms(state.symptoms_collected)}

RED FLAGS DETECTED: {', '.join(state.red_flags_detected) if state.red_flags_detected else 'None'}

CONVERSATION HISTORY:
{self._format_questions(state.questions_asked, state.responses_received)}

RELEVANT MEDICAL GUIDELINES:
{guideline_text}
"""
        
        # Generate diagnosis
        diagnosis_prompt = f"""{assessment_summary}

Based on the medical guidelines and patient information above, provide:

1. DIFFERENTIAL DIAGNOSIS: List 3-5 most likely diagnoses (with confidence %)
2. URGENCY LEVEL: Rate 1-10 (1=routine, 10=life-threatening emergency)
3. DISPOSITION: Recommend next steps (call 911, ER, urgent care, PCP appointment)
4. RATIONALE: Brief clinical reasoning

Respond in JSON format:
{{
  "differential": [
    {{"diagnosis": "Acute MI", "confidence": 70}},
    {{"diagnosis": "Unstable angina", "confidence": 20}},
    {{"diagnosis": "Costochondritis", "confidence": 10}}
  ],
  "urgency": 9,
  "disposition": "EMERGENCY - Call 911 immediately. Possible heart attack.",
  "rationale": "Patient presents with crushing chest pain radiating to left arm with diaphoresis - classic MI presentation. Requires immediate emergency evaluation."
}}

JSON RESPONSE:"""
        
        llm_response = llm_chat_fn([{"role": "user", "content": diagnosis_prompt}])
        
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                diagnosis = json.loads(json_match.group(0))
                
                # Update state
                for dx in diagnosis.get('differential', []):
                    state.differential_diagnosis[dx['diagnosis']] = dx['confidence']
                
                state.urgency_score = diagnosis.get('urgency', state.urgency_score)
                state.completed = True
                
                print(f"[DynamicMed] ✅ Assessment complete:")
                print(f"[DynamicMed]    Primary Dx: {diagnosis['differential'][0]['diagnosis'] if diagnosis.get('differential') else 'Unknown'}")
                print(f"[DynamicMed]    Urgency: {diagnosis.get('urgency', 0)}/10")
                print(f"[DynamicMed]    Disposition: {diagnosis.get('disposition', 'Unknown')}")
                
                return diagnosis
            else:
                print("[DynamicMed] ⚠️ Could not parse diagnosis")
                return self._generate_fallback_diagnosis(state)
        except Exception as e:
            print(f"[DynamicMed] ❌ Error generating diagnosis: {e}")
            return self._generate_fallback_diagnosis(state)
    
    def _generate_fallback_diagnosis(self, state: MedicalAssessmentState) -> Dict:
        """Generate basic diagnosis when LLM fails"""
        # Use urgency score to determine disposition
        if state.urgency_score >= self.URGENCY_EMERGENCY:
            disposition = "EMERGENCY - Seek immediate medical attention (call 911)"
        elif state.urgency_score >= self.URGENCY_URGENT:
            disposition = "URGENT - Visit emergency room within 1-2 hours"
        elif state.urgency_score >= self.URGENCY_SEMI_URGENT:
            disposition = "SEMI-URGENT - See doctor within 24 hours"
        else:
            disposition = "ROUTINE - Schedule appointment with primary care physician"
        
        return {
            "differential": [{"diagnosis": state.chief_complaint, "confidence": 50}],
            "urgency": state.urgency_score,
            "disposition": disposition,
            "rationale": "Assessment based on reported symptoms and red flags"
        }
    
    def _build_assessment_context(self, state: MedicalAssessmentState) -> str:
        """Build context summary for LLM"""
        return f"""
Chief Complaint: {state.chief_complaint}
Category: {state.category}
Symptoms: {len(state.symptoms_collected)}
Red Flags: {len(state.red_flags_detected)}
Questions Asked: {len(state.questions_asked)}
Urgency Score: {state.urgency_score:.1f}/10
"""
    
    def _format_symptoms(self, symptoms: List[Dict]) -> str:
        """Format symptoms list for display"""
        if not symptoms:
            return "  (none collected yet)"
        
        return "\n".join([f"  - {s.get('symptom', s)}" for s in symptoms])
    
    def _format_questions(self, questions: List[str], responses: List[str]) -> str:
        """Format Q&A history"""
        if not questions:
            return "  (no questions asked yet)"
        
        formatted = []
        for i, (q, r) in enumerate(zip(questions, responses), 1):
            formatted.append(f"  Q{i}: {q}")
            formatted.append(f"  A{i}: {r}")
        
        return "\n".join(formatted)
    
    def get_disposition_recommendation(self, urgency_score: float) -> Dict:
        """
        Get disposition recommendation based on urgency score
        
        Args:
            urgency_score: 0-10 urgency rating
            
        Returns:
            Dict with disposition, timeline, and instructions
        """
        if urgency_score >= self.URGENCY_EMERGENCY:
            return {
                "level": "EMERGENCY",
                "action": "Call 911 immediately",
                "timeline": "NOW",
                "color": "red",
                "icon": "🚨"
            }
        elif urgency_score >= self.URGENCY_URGENT:
            return {
                "level": "URGENT",
                "action": "Visit emergency room",
                "timeline": "Within 1-2 hours",
                "color": "orange",
                "icon": "⚠️"
            }
        elif urgency_score >= self.URGENCY_SEMI_URGENT:
            return {
                "level": "SEMI-URGENT",
                "action": "See doctor today",
                "timeline": "Within 24 hours",
                "color": "yellow",
                "icon": "⏰"
            }
        else:
            return {
                "level": "ROUTINE",
                "action": "Schedule appointment",
                "timeline": "Within 1 week",
                "color": "green",
                "icon": "📅"
            }
    
    def run_interactive_assessment(self, 
                                   session_id: str,
                                   chief_complaint: str,
                                   llm_chat_fn,
                                   max_questions: int = 8):
        """
        Run a complete interactive medical assessment
        
        This is a demo/test function showing the full workflow
        
        Args:
            session_id: Unique session ID
            chief_complaint: Patient's initial complaint
            llm_chat_fn: LLM chat function
            max_questions: Maximum questions to ask
            
        Returns:
            Final diagnosis and disposition
        """
        print("\n" + "="*80)
        print("  🏥 DYNAMIC MEDICAL ASSESSMENT")
        print("="*80 + "\n")
        
        # Start assessment
        state = self.start_assessment(session_id, chief_complaint)
        
        # Ask questions dynamically
        for i in range(max_questions):
            question = self.generate_next_question(session_id, llm_chat_fn)
            
            if not question:  # Assessment complete
                break
            
            print(f"\n[Q{i+1}] {question}")
            
            # In real use, this would come from voice/text input
            # For testing, you'd integrate with your listener/GUI
            user_response = input("[A] ")
            
            # Process response
            self.process_response(session_id, user_response, llm_chat_fn)
            
            # Check if emergency detected
            if state.urgency_score >= self.URGENCY_EMERGENCY:
                print(f"\n🚨 EMERGENCY DETECTED - Stopping assessment")
                break
        
        # Generate final diagnosis
        print("\n" + "="*80)
        print("  🩺 GENERATING DIAGNOSIS")
        print("="*80 + "\n")
        
        diagnosis = self.generate_diagnosis_and_disposition(session_id, llm_chat_fn)
        
        # Display results
        self._display_diagnosis(diagnosis)
        
        return diagnosis
    
    def _display_diagnosis(self, diagnosis: Dict):
        """Display diagnosis in formatted output"""
        print("\n" + "="*80)
        print("  📋 CLINICAL ASSESSMENT RESULTS")
        print("="*80 + "\n")
        
        # Differential diagnosis
        print("DIFFERENTIAL DIAGNOSIS:")
        for dx in diagnosis.get('differential', []):
            confidence = dx.get('confidence', 0)
            print(f"  {confidence:2d}% - {dx.get('diagnosis', 'Unknown')}")
        
        print(f"\nURGENCY: {diagnosis.get('urgency', 0)}/10")
        
        # Disposition
        disposition_info = self.get_disposition_recommendation(diagnosis.get('urgency', 0))
        print(f"\nDISPOSITION: {disposition_info['icon']} {disposition_info['level']}")
        print(f"  Action: {disposition_info['action']}")
        print(f"  Timeline: {disposition_info['timeline']}")
        
        # Rationale
        print(f"\nCLINICAL REASONING:")
        print(f"  {diagnosis.get('rationale', 'See assessment above')}")
        
        print("\n" + "="*80 + "\n")


# === Demo/Testing Functions ===

def demo_chest_pain_assessment():
    """Demo the dynamic assessment with a chest pain scenario"""
    
    # Mock LLM function for testing
    def mock_llm(messages):
        # In real use, this would call your LLM
        # For demo, return placeholder
        return "Is the pain crushing or squeezing in nature?"
    
    assistant = DynamicMedicalAssistant()
    
    # Run assessment
    diagnosis = assistant.run_interactive_assessment(
        session_id="demo_001",
        chief_complaint="I have chest pain",
        llm_chat_fn=mock_llm,
        max_questions=5
    )
    
    return diagnosis


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  🏥 DYNAMIC MEDICAL ASSISTANT - Demo Mode")
    print("="*80)
    print("\n  This is a demo of the dynamic medical assessment system.")
    print("  For full functionality, integrate with LLM and RAG containers.\n")
    print("="*80 + "\n")
    
    demo_chest_pain_assessment()

