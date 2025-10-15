#!/usr/bin/env python3
"""
Unified Medical Mode - Physician-Like Medical Assistant

Combines enhanced clinician (symptom assessment) and thinker (medical knowledge)
into a single, seamless physician-like mode that can:

1. Assess symptoms when users report them ("I have chest pain")
2. Answer medical knowledge questions ("What is hypertension?")
3. Provide comprehensive medical guidance
4. Maintain clinical context across interactions
5. Use evidence-based medical knowledge

This creates a true physician-like experience for all medical interactions.
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

# Import medical RAG for knowledge queries
try:
    from medical_rag import MedicalRAG, get_medical_rag, get_medical_messages
    MEDICAL_RAG_AVAILABLE = True
    print("[Unified Medical] ✅ Medical RAG imported successfully")
except ImportError as e:
    MEDICAL_RAG_AVAILABLE = False
    print(f"[Unified Medical] ⚠️ Medical RAG not available: {e}")

# Import dynamic medical assistant for guideline-based assessment
try:
    # Copy dynamic_medical_assistant.py to llm-container directory
    # For now, we'll implement dynamic assessment within this file
    DYNAMIC_ASSESSMENT_AVAILABLE = True
except Exception as e:
    DYNAMIC_ASSESSMENT_AVAILABLE = False
    print(f"[Unified Medical] ⚠️ Dynamic assessment not available: {e}")

# Load shared medical terms from centralized file (used by both Whisper and LLM)
MEDICAL_TERMS = {}
MEDICAL_TERMS_FILE = "/app/medical_terms.json"

def _load_medical_terms():
    """Load medical terms from shared JSON file"""
    global MEDICAL_TERMS
    try:
        with open(MEDICAL_TERMS_FILE, 'r') as f:
            MEDICAL_TERMS = json.load(f)
        # Flatten all terms into a single list for fast keyword matching
        all_terms = []
        for category, terms in MEDICAL_TERMS.items():
            all_terms.extend(terms)
        MEDICAL_TERMS['_all_terms_flat'] = list(set(all_terms))  # Deduplicate
        print(f"[Unified Medical] ✅ Loaded {len(MEDICAL_TERMS['_all_terms_flat'])} medical terms from shared file")
    except Exception as e:
        print(f"[Unified Medical] ⚠️ Could not load medical terms: {e}")
        print("[Unified Medical] ⚠️ Falling back to suffix-based detection only")
        MEDICAL_TERMS['_all_terms_flat'] = []

# Load medical terms on module import
_load_medical_terms()

class DynamicAssessmentState:
    """State tracking for dynamic RAG-powered medical assessment"""
    def __init__(self, chief_complaint: str):
        self.chief_complaint = chief_complaint
        self.symptoms_collected = []
        self.red_flags_detected = []
        self.questions_asked = []
        self.responses_received = []
        self.urgency_score = 0.0  # 0-10 scale
        self.category = "unknown"
        self.completed = False

class UnifiedMedicalSession:
    """
    Unified medical assistant that handles both symptom assessment and medical knowledge
    
    Now supports TWO assessment modes:
    1. RIGID TRIAGE: JSON-based decision tree (baseline/fallback)
    2. DYNAMIC ASSESSMENT: RAG-powered guideline-based questioning ⭐ NEW
    """

    def __init__(self, session_id: str, llm_chat_fn: Callable):
        self.session_id = session_id
        self.llm_chat_fn = llm_chat_fn
        
        # Session state
        self.conversation_history = []
        self.current_context = "general"  # "assessment", "knowledge", "general"
        self.active_assessment = None  # EnhancedClinicianSession if doing assessment
        self.dynamic_assessment = None  # DynamicAssessmentState for guideline-based assessment
        self.medical_rag = None
        
        # Assessment mode selection
        self.use_dynamic_assessment = True  # Set to True to use RAG-powered assessment

        # Medical knowledge state
        self.last_medical_query = None
        self.medical_context = {}

        # Initialize medical RAG
        self._initialize_medical_rag()

        print(f"[Unified Medical] 🩺 Starting unified medical session: {session_id}")

    def _initialize_medical_rag(self):
        """Initialize medical RAG for knowledge queries"""
        if MEDICAL_RAG_AVAILABLE:
            try:
                self.medical_rag = get_medical_rag()
                print("[Unified Medical] ✅ Medical RAG initialized")
            except Exception as e:
                print(f"[Unified Medical] ⚠️ Medical RAG initialization failed: {e}")
                self.medical_rag = None
        else:
            print("[Unified Medical] ⚠️ Medical RAG not available")

    def process_medical_query(self, user_input: str) -> str:
        """
        Process any medical-related query (symptoms or knowledge questions)

        Args:
            user_input: User's medical query

        Returns:
            Physician-like response
        """
        # Store the query
        self.conversation_history.append({
            'role': 'patient',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Analyze the query type
        query_type = self._analyze_medical_query(user_input)

        print(f"[Unified Medical] 🔍 Query type: {query_type}")

        if query_type == "symptom_assessment":
            return self._handle_symptom_assessment(user_input)
        elif query_type == "medical_knowledge":
            return self._handle_medical_knowledge(user_input)
        else:
            return self._handle_general_medical(user_input)

    def _analyze_medical_query(self, query: str) -> str:
        """
        Analyze query to determine if it's symptom assessment, knowledge question, or general medical

        Returns:
            "symptom_assessment", "medical_knowledge", or "general_medical"
        """
        query_lower = query.lower()

        # Check for symptom assessment patterns (first-person medical complaints)
        symptom_patterns = [
            r'\bi have\b', r'\bi\'m having\b', r'\bim having\b',
            r'\bi feel\b', r'\bi\'m feeling\b', r'\bim feeling\b',
            r'\bmy .+ (hurt|ache|pain)', r'\bi experience\b',
            r'\bi\'m experiencing\b', r'\bim experiencing\b',
            r'\bi suffer from\b', r'\bi\'m suffering from\b'
        ]

        if any(re.search(pattern, query_lower) for pattern in symptom_patterns):
            return "symptom_assessment"

        # Check for medical knowledge questions
        knowledge_indicators = [
            "what is", "what are", "what does", "what do",
            "how is", "how are", "how does", "how do",
            "why is", "why are", "why does", "why do",
            "when is", "when are", "when does", "when do",
            "where is", "where are", "where does", "where do",
            "who is", "who are", "who does", "who do",
            "tell me about", "explain", "describe", "define",
            "can you", "could you", "would you", "will you"
        ]

        if any(indicator in query_lower for indicator in knowledge_indicators):
            # Check if it's about medical topics using shared medical terms
            # Use the centralized medical_terms.json file (shared with Whisper container)
            if MEDICAL_TERMS.get('_all_terms_flat'):
                if any(keyword in query_lower for keyword in MEDICAL_TERMS['_all_terms_flat']):
                    return "medical_knowledge"
            # Fallback to suffix-based detection if terms not loaded
            else:
                # Medical term suffixes (catches pancreatitis, hepatitis, etc.)
                medical_suffixes = [
                    r'\w+itis\b', r'\w+osis\b', r'\w+emia\b', r'\w+pathy\b',
                    r'\w+ology\b', r'\w+oma\b', r'\w+algia\b'
                ]
                if any(re.search(pattern, query_lower) for pattern in medical_suffixes):
                    return "medical_knowledge"

        # Check for general medical topics
        if any(term in query_lower for term in ["medicine", "medical", "health", "clinical", "patient", "doctor"]):
            return "general_medical"

        return "general_medical"

    def _handle_symptom_assessment(self, symptom_query: str) -> str:
        """
        Handle symptom assessment using DYNAMIC guideline-based questioning
        
        New approach: Uses RAG-retrieved medical guidelines to ask intelligent,
        contextual questions rather than following rigid decision trees
        """
        print(f"[Unified Medical] 🩺 Handling symptom assessment: {symptom_query}")
        
        # Use dynamic RAG-powered assessment (new approach)
        if self.use_dynamic_assessment:
            return self._handle_dynamic_assessment(symptom_query)
        
        # Fallback to rigid triage if dynamic assessment disabled
        if ENHANCED_CLINICIAN_AVAILABLE:
            try:
                # Create or continue enhanced clinician session
                if self.active_assessment is None:
                    self.active_assessment = EnhancedClinicianSession(
                        self.session_id, symptom_query, self.llm_chat_fn
                    )
                    self.current_context = "assessment"

                response = self.active_assessment.process_symptom_response(symptom_query)

                # Check if assessment is complete
                if self.active_assessment.assessment_complete:
                    self.current_context = "general"
                    self.active_assessment = None

                return response

            except Exception as e:
                print(f"[Unified Medical] ❌ Enhanced clinician failed: {e}")
                return self._fallback_to_knowledge_response(symptom_query)

        return self._fallback_to_knowledge_response(symptom_query)
    
    def _handle_dynamic_assessment(self, symptom_query: str) -> str:
        """
        Handle symptom assessment using dynamic guideline-based questioning
        
        This is the NEW approach that uses RAG to retrieve medical guidelines
        and asks intelligent, contextual questions
        """
        # Initialize dynamic assessment if needed
        if self.dynamic_assessment is None:
            print("[Dynamic] 🏥 Starting dynamic guideline-based assessment")
            self.dynamic_assessment = DynamicAssessmentState(chief_complaint=symptom_query)
            self.current_context = "assessment"
            
            # Categorize the complaint
            self.dynamic_assessment.category = self._categorize_complaint(symptom_query)
            
            # Retrieve relevant guidelines
            guidelines = self._get_medical_guidelines(symptom_query, self.dynamic_assessment.category)
            
            # Generate first question
            return self._generate_dynamic_question(guidelines)
        
        # Continue existing assessment
        else:
            # Store previous response
            self.dynamic_assessment.responses_received.append(symptom_query)
            
            # Analyze response for urgency and red flags
            self._analyze_patient_response(symptom_query)
            
            # Check if assessment should be completed
            if self._should_complete_assessment():
                return self._generate_dynamic_diagnosis()
            
            # Get updated guidelines based on new information
            enhanced_query = f"{self.dynamic_assessment.chief_complaint} {symptom_query}"
            guidelines = self._get_medical_guidelines(enhanced_query, self.dynamic_assessment.category)
            
            # Generate next question
            return self._generate_dynamic_question(guidelines)
    
    def _categorize_complaint(self, complaint: str) -> str:
        """Categorize complaint by medical specialty"""
        complaint_lower = complaint.lower()
        
        categories = {
            'cardiovascular': ['chest pain', 'heart', 'palpitation', 'shortness of breath', 'chest'],
            'respiratory': ['cough', 'breathing', 'dyspnea', 'wheezing', 'lung'],
            'gastrointestinal': ['stomach', 'abdominal', 'nausea', 'vomit', 'diarrhea', 'pancreatitis', 'belly'],
            'neurological': ['headache', 'dizzy', 'seizure', 'numbness', 'weakness', 'head'],
            'musculoskeletal': ['back pain', 'joint', 'muscle', 'bone'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in complaint_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _get_medical_guidelines(self, query: str, category: str = None) -> List[Dict]:
        """Retrieve medical guidelines from RAG"""
        try:
            # Use existing medical RAG search
            if self.medical_rag:
                results = self.medical_rag.search_medical_info(query, k=5)
                return results if results else []
            else:
                # Fallback to general RAG
                response = requests.post(
                    "http://localhost:11435/rag/search",
                    json={"query": f"medical guideline {category} {query}", "top_k": 5},
                    timeout=10
                )
                if response.status_code == 200:
                    return response.json().get('results', [])
        except Exception as e:
            print(f"[Dynamic] ❌ Error retrieving guidelines: {e}")
        
        return []
    
    def _generate_dynamic_question(self, guidelines: List[Dict]) -> str:
        """
        Generate next diagnostic question using RAG-retrieved guidelines
        
        This is the core of dynamic assessment - uses real medical guidelines
        to ask intelligent, contextual questions
        """
        state = self.dynamic_assessment
        
        # Build context for LLM
        guideline_text = "\n\n".join([g.get('text', '') for g in guidelines[:3]])
        
        prompt = f"""You are conducting a medical assessment. Generate the next diagnostic question.

CHIEF COMPLAINT: {state.chief_complaint}
CATEGORY: {state.category}

SYMPTOMS SO FAR: {', '.join([str(s) for s in state.symptoms_collected]) if state.symptoms_collected else 'None'}
RED FLAGS: {', '.join(state.red_flags_detected) if state.red_flags_detected else 'None'}

MEDICAL GUIDELINES:
{guideline_text}

QUESTIONS ALREADY ASKED: {len(state.questions_asked)}

Generate ONE clear, specific question to gather critical diagnostic information. Focus on:
1. Red flag symptoms if not yet assessed
2. Severity and character of symptoms
3. Duration and onset
4. Associated symptoms

Keep it conversational and patient-friendly. Just ask the question, nothing else.

QUESTION:"""
        
        # Get LLM response
        response = self.llm_chat_fn([{"role": "user", "content": prompt}])
        question = response.strip()
        
        # Track question
        state.questions_asked.append(question)
        
        return question
    
    def _analyze_patient_response(self, response: str):
        """Analyze patient response for symptoms and urgency indicators"""
        state = self.dynamic_assessment
        response_lower = response.lower()
        
        # Check for emergency keywords (red flags)
        emergency_keywords = [
            'crushing', 'severe', 'worst', 'unbearable', 'radiating',
            'sweating', 'dizzy', 'faint', 'confused', 'can\'t breathe',
            'blood', 'bleeding', 'unconscious', 'chest pressure'
        ]
        
        for keyword in emergency_keywords:
            if keyword in response_lower:
                if keyword not in state.red_flags_detected:
                    state.red_flags_detected.append(keyword)
                    state.urgency_score += 1.5
        
        # Check for positive severe responses
        if any(word in response_lower for word in ['yes', 'yeah', 'yep']) and len(state.questions_asked) > 0:
            last_question_lower = state.questions_asked[-1].lower()
            if any(word in last_question_lower for word in ['severe', 'emergency', 'urgent', 'crushing', 'radiating']):
                state.urgency_score += 1.0
        
        # Store as symptom
        state.symptoms_collected.append({
            'symptom': response,
            'context': state.questions_asked[-1] if state.questions_asked else 'initial'
        })
    
    def _should_complete_assessment(self) -> bool:
        """Determine if enough information gathered to complete assessment"""
        state = self.dynamic_assessment
        
        # Complete if:
        # 1. Emergency detected (urgency >= 8)
        if state.urgency_score >= 8.0:
            return True
        
        # 2. Sufficient questions asked (5-8 questions typical)
        if len(state.questions_asked) >= 8:
            return True
        
        # 3. Multiple red flags detected
        if len(state.red_flags_detected) >= 3:
            return True
        
        return False
    
    def _generate_dynamic_diagnosis(self) -> str:
        """
        Generate diagnosis and disposition using collected information + guidelines
        """
        state = self.dynamic_assessment
        
        # Retrieve comprehensive guidelines for diagnosis
        search_query = f"{state.chief_complaint} diagnosis differential {' '.join([s.get('symptom', '') for s in state.symptoms_collected])}"
        guidelines = self._get_medical_guidelines(search_query, state.category)
        guideline_text = "\n\n".join([g.get('text', '') for g in guidelines[:5]])
        
        diagnosis_prompt = f"""You are a physician completing a medical assessment.

CHIEF COMPLAINT: {state.chief_complaint}
CATEGORY: {state.category}

ASSESSMENT HISTORY:
{self._format_qa_history(state)}

SYMPTOMS COLLECTED: {', '.join([str(s.get('symptom', s)) for s in state.symptoms_collected])}
RED FLAGS: {', '.join(state.red_flags_detected) if state.red_flags_detected else 'None'}

RELEVANT MEDICAL GUIDELINES:
{guideline_text}

Provide a concise clinical assessment with:
1. Most likely diagnosis
2. Urgency level (1-10)
3. Recommended next steps (disposition)

Be direct and actionable. Format as natural physician guidance.

ASSESSMENT:"""
        
        diagnosis_response = self.llm_chat_fn([{"role": "user", "content": diagnosis_prompt}])
        
        # Mark assessment as complete
        state.completed = True
        self.current_context = "general"
        self.dynamic_assessment = None
        
        return diagnosis_response
    
    def _format_qa_history(self, state: DynamicAssessmentState) -> str:
        """Format question/answer history for LLM"""
        history = []
        for i, (q, a) in enumerate(zip(state.questions_asked, state.responses_received), 1):
            history.append(f"Q{i}: {q}")
            history.append(f"A{i}: {a}")
        return "\n".join(history) if history else "(no questions yet)"

    def _handle_medical_knowledge(self, knowledge_query: str) -> str:
        """Handle medical knowledge questions using RAG"""
        print(f"[Unified Medical] 📚 Handling medical knowledge: {knowledge_query}")

        if self.medical_rag:
            try:
                # Use medical RAG for knowledge queries
                results = self.medical_rag.search_medical_info(knowledge_query, k=3)

                if results:
                    # Generate physician-like response
                    context = self.medical_rag.get_medical_context(knowledge_query, results)

                    response_prompt = f"""
You are a knowledgeable physician providing medical information. A patient has asked: "{knowledge_query}"

Based on current medical knowledge:

{context}

Provide a comprehensive, evidence-based response that:
1. Directly answers the question
2. Includes relevant medical context
3. Mentions evidence levels when applicable
4. Provides practical guidance
5. Uses professional but accessible language

IMPORTANT: Always include appropriate medical disclaimers and recommend consulting healthcare providers for personal health concerns.
"""

                    response = self.llm_chat_fn([{"role": "system", "content": response_prompt}])

                    # Store for context
                    self.last_medical_query = knowledge_query
                    self.current_context = "knowledge"

                    return response

            except Exception as e:
                print(f"[Unified Medical] ❌ Medical RAG failed: {e}")

        return self._fallback_to_general_response(knowledge_query)

    def _handle_general_medical(self, general_query: str) -> str:
        """Handle general medical queries"""
        print(f"[Unified Medical] 💬 Handling general medical: {general_query}")

        return self._fallback_to_general_response(general_query)

    def _fallback_to_knowledge_response(self, query: str) -> str:
        """Fallback response using medical knowledge when assessment fails"""
        print(f"[Unified Medical] 🔄 Falling back to knowledge response for: {query}")

        if self.medical_rag:
            try:
                results = self.medical_rag.search_medical_info(query, k=2)
                if results:
                    context = self.medical_rag.get_medical_context(query, results)

                    response_prompt = f"""
You are a physician providing guidance. The user mentioned: "{query}"

Available medical information:
{context}

Provide a helpful, professional response that addresses their concern while noting this is for informational purposes only.
"""

                    return self.llm_chat_fn([{"role": "system", "content": response_prompt}])
            except Exception as e:
                print(f"[Unified Medical] ❌ Knowledge fallback failed: {e}")

        return "I'm sorry, I encountered an issue processing your medical query. For your safety, please consult with a healthcare professional for any medical concerns."

    def _fallback_to_general_response(self, query: str) -> str:
        """General fallback response"""
        response_prompt = f"""
You are a helpful medical assistant. The user asked: "{query}"

Provide a helpful response. If this appears to be a medical concern, gently suggest consulting a healthcare professional.

Remember: You are not a substitute for professional medical advice.
"""

        return self.llm_chat_fn([{"role": "system", "content": response_prompt}])

    def get_session_summary(self) -> Dict:
        """Get summary of current medical session"""
        return {
            'session_id': self.session_id,
            'current_context': self.current_context,
            'has_active_assessment': self.active_assessment is not None,
            'conversation_length': len(self.conversation_history),
            'last_query': self.last_medical_query,
            'medical_rag_available': MEDICAL_RAG_AVAILABLE,
            'enhanced_clinician_available': ENHANCED_CLINICIAN_AVAILABLE
        }

# Global unified medical session
unified_medical_session = None

def get_unified_medical_session(session_id: str, llm_chat_fn: Callable) -> UnifiedMedicalSession:
    """Get or create unified medical session"""
    global unified_medical_session
    if unified_medical_session is None or unified_medical_session.session_id != session_id:
        unified_medical_session = UnifiedMedicalSession(session_id, llm_chat_fn)
    return unified_medical_session

def is_unified_medical_trigger(prompt: str) -> bool:
    """
    Determine if a prompt should trigger unified medical mode using intelligent keyword search

    Triggers for:
    - Medical symptoms ("I have chest pain")
    - Medical knowledge questions ("What is hypertension?")
    - General medical topics ("medicine", "health", etc.")

    Args:
        prompt: User prompt

    Returns:
        True if should use unified medical mode
    """
    prompt_lower = prompt.lower()

    # Medical symptom patterns (first-person statements)
    symptom_patterns = [
        r'\bi have\b', r'\bi\'m having\b', r'\bim having\b',
        r'\bi feel\b', r'\bi\'m feeling\b', r'\bim feeling\b',
        r'\bmy .+ (hurt|ache|pain)', r'\bi experience\b',
        r'\bi\'m experiencing\b', r'\bim experiencing\b'
    ]

    if any(re.search(pattern, prompt_lower) for pattern in symptom_patterns):
        return True

    # Medical knowledge questions
    knowledge_indicators = [
        "what is", "what are", "what does", "what do",
        "how is", "how are", "how does", "how do",
        "why is", "why are", "why does", "why do",
        "when is", "when are", "when does", "when do",
        "where is", "where are", "where does", "where do",
        "who is", "who are", "who does", "who do",
        "tell me about", "explain", "describe", "define"
    ]

    if any(indicator in prompt_lower for indicator in knowledge_indicators):
        # Use intelligent fast keyword search to detect medical topics
        # Latency: ~0.0001 seconds (100 microseconds) - negligible!
        if _is_medical_topic_fast(prompt_lower):
            return True

    # General medical topics
    if any(term in prompt_lower for term in ["medicine", "medical", "health", "clinical", "patient", "doctor"]):
        return True

    return False


def _is_medical_topic_fast(text: str) -> bool:
    """
    Lightning-fast keyword-based medical topic detection
    
    Performance: ~100 microseconds (0.0001 seconds) - negligible latency!
    Much faster than LLM classification which takes 0.5-2 seconds.
    
    Uses intelligent patterns to detect medical content:
    1. Medical term suffixes (-itis, -osis, -emia, -pathy, -ology, etc.)
    2. Anatomical terms (heart, lung, brain, liver, etc.)
    3. Common medical conditions
    4. Medical procedures and tests
    
    Args:
        text: Lowercased text to analyze
        
    Returns:
        True if text contains medical content
    """
    # Use shared medical terms from centralized medical_terms.json
    # This file is shared between Whisper (for transcription hints) and LLM (for routing)
    if MEDICAL_TERMS.get('_all_terms_flat'):
        # Fast keyword check using pre-loaded terms (O(n) where n = query words)
        words = set(text.split())
        medical_terms_set = set(MEDICAL_TERMS['_all_terms_flat'])
        
        # Check if any word in query matches known medical terms
        if words & medical_terms_set:  # Set intersection - very fast!
            return True
    
    # Fallback: Medical term suffix patterns (catches terms not in our list)
    # This catches pancreatitis, hepatitis, etc. even if not in medical_terms.json
    medical_suffixes = [
        r'\w+itis\b',      # pancreatitis, hepatitis, arthritis, bronchitis, etc.
        r'\w+osis\b',      # cirrhosis, osteoporosis, thrombosis, psychosis, etc.
        r'\w+emia\b',      # anemia, septicemia, leukemia, hypoglycemia, etc.
        r'\w+pathy\b',     # neuropathy, myopathy, cardiomyopathy, etc.
        r'\w+trophy\b',    # dystrophy, hypertrophy, atrophy, etc.
        r'\w+plasia\b',    # dysplasia, hyperplasia, neoplasia, aplasia, etc.
        r'\w+ectomy\b',    # appendectomy, mastectomy, hysterectomy, etc.
        r'\w+otomy\b',     # tracheotomy, lobotomy, laparotomy, etc.
        r'\w+scopy\b',     # endoscopy, colonoscopy, bronchoscopy, etc.
        r'\w+ology\b',     # cardiology, neurology, oncology, radiology, etc.
        r'\w+ologist\b',   # cardiologist, neurologist, oncologist, etc.
        r'\w+algia\b',     # neuralgia, myalgia, arthralgia, cephalgia, etc.
    ]
    
    for pattern in medical_suffixes:
        if re.search(pattern, text):
            return True
    
    return False


def handle_unified_medical_response(prompt: str, session_id: str, llm_chat_fn: Callable):
    """
    Handle medical queries through unified medical mode

    Args:
        prompt: User prompt
        session_id: Session identifier
        llm_chat_fn: LLM chat function

    Returns:
        Medical response (could be messages list for streaming or direct response)
    """
    session = get_unified_medical_session(session_id, llm_chat_fn)
    return session.process_medical_query(prompt)


def get_unified_medical_messages(prompt: str, session_id: str) -> list:
    """
    Get LLM messages for unified medical mode with RAG augmentation
    
    Args:
        prompt: User prompt
        session_id: Session identifier
        
    Returns:
        List of messages for LLM (with RAG context if available)
    """
    # Try to use Medical RAG for enhanced responses
    if MEDICAL_RAG_AVAILABLE:
        try:
            print("[Unified Medical] 📚 Using Medical RAG for query")
            return get_medical_messages(prompt)
        except Exception as e:
            print(f"[Unified Medical] ⚠️ Medical RAG failed, using fallback: {e}")
    
    # Fallback: Build basic medical assistant prompt
    print("[Unified Medical] ⚠️ Using fallback prompt (no RAG)")
    system_prompt = f"""You are a helpful medical assistant. The user asked: "{prompt}"

Provide a helpful response. If this appears to be a medical concern, gently suggest consulting a healthcare professional.

Remember: You are not a substitute for professional medical advice."""
    
    return [{"role": "system", "content": system_prompt}]

if __name__ == "__main__":
    # Test the unified medical mode
    def mock_llm_chat(messages):
        return "This is a mock physician response to medical queries."

    # Test different types of medical queries
    test_queries = [
        "I have chest pain",  # Symptom assessment
        "What is hypertension?",  # Medical knowledge
        "How do you treat diabetes?",  # Treatment knowledge
        "What are the symptoms of pneumonia?",  # Symptom knowledge
        "I feel dizzy and nauseous",  # Symptom assessment
        "Tell me about heart disease"  # General medical knowledge
    ]

    print("🩺 UNIFIED MEDICAL MODE TEST")
    print("=" * 50)

    session = UnifiedMedicalSession("test_session", mock_llm_chat)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        query_type = session._analyze_medical_query(query)
        print(f"🔍 Type: {query_type}")

        response = session.process_medical_query(query)
        print(f"🩺 Response: {response[:100]}{'...' if len(response) > 100 else ''}")

    print(f"\n📊 Session Summary: {session.get_session_summary()}")
