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

class UnifiedMedicalSession:
    """
    Unified medical assistant that handles both symptom assessment and medical knowledge
    """

    def __init__(self, session_id: str, llm_chat_fn: Callable):
        self.session_id = session_id
        self.llm_chat_fn = llm_chat_fn
        
        # Session state
        self.conversation_history = []
        self.current_context = "general"  # "assessment", "knowledge", "general"
        self.active_assessment = None  # EnhancedClinicianSession if doing assessment
        self.medical_rag = None

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
            # Check if it's about medical topics
            medical_keywords = [
                # Medical conditions and diseases
                "hypertension", "diabetes", "cancer", "pneumonia", "asthma", "copd",
                "heart disease", "cardiovascular", "myocardial", "infarction", "stroke",
                "alzheimer", "parkinson", "arthritis", "osteoporosis", "anemia",
                "depression", "anxiety", "schizophrenia", "bipolar", "adhd",
                "hypothyroid", "hyperthyroid", "kidney", "liver", "pancreas", "pancreatitis",
                "hepatitis", "cirrhosis", "nephritis", "gastritis", "colitis", "bronchitis",
                "meningitis", "encephalitis", "appendicitis", "diverticulitis", "cholecystitis",
                # Medical symptoms and signs
                "symptom", "treatment", "diagnosis", "medication", "therapy",
                "clinical", "medical", "health", "disease", "condition", "disorder",
                "syndrome", "infection", "inflammation", "chronic", "acute",
                "pain", "fever", "cough", "nausea", "dizziness", "headache",
                "chest", "abdominal", "heart", "lung", "brain", "blood", "pressure",
                "fatigue", "weakness", "numbness", "tingling", "swelling", "rash",
                "bleeding", "bruising", "seizure", "paralysis", "tremor", "shaking",
                # Medical procedures and tests
                "surgery", "biopsy", "endoscopy", "colonoscopy", "mammogram",
                "x-ray", "ct scan", "mri", "ultrasound", "blood test", "urine test",
                # Medical specialties
                "cardiology", "neurology", "oncology", "dermatology", "psychiatry",
                "pediatrics", "gynecology", "ophthalmology", "orthopedics",
                # General medical terms
                "patient", "doctor", "physician", "nurse", "hospital", "clinic",
                "prescription", "dosage", "side effect", "contraindication",
                "allergy", "immune", "vaccine", "vaccination", "antibody"
            ]

            if any(keyword in query_lower for keyword in medical_keywords):
                return "medical_knowledge"

        # Check for general medical topics
        if any(term in query_lower for term in ["medicine", "medical", "health", "clinical", "patient", "doctor"]):
            return "general_medical"

        return "general_medical"

    def _handle_symptom_assessment(self, symptom_query: str) -> str:
        """Handle symptom assessment using enhanced clinician"""
        print(f"[Unified Medical] 🩺 Handling symptom assessment: {symptom_query}")

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
    # Medical term suffix patterns (catches thousands of medical terms automatically!)
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
    
    # Check medical suffixes first (catches most medical terms automatically)
    for pattern in medical_suffixes:
        if re.search(pattern, text):
            return True
    
    # Anatomical terms (organs and body systems)
    anatomical_terms = {
        "heart", "lung", "brain", "liver", "kidney", "pancreas", "stomach",
        "intestine", "colon", "bladder", "prostate", "uterus", "ovary",
        "thyroid", "adrenal", "pituitary", "spleen", "gallbladder",
        "esophagus", "trachea", "bronch", "alveol", "artery", "vein",
        "muscle", "bone", "joint", "tendon", "ligament", "cartilage",
        "nerve", "spinal", "cerebral", "cardiac", "pulmonary", "hepatic",
        "renal", "gastric", "intestinal", "vascular", "lymph", "blood"
    }
    
    # Fast set membership check (O(1) average case)
    words = set(text.split())
    if words & anatomical_terms:  # Set intersection - very fast!
        return True
    
    # Common medical conditions (high-frequency terms)
    common_conditions = {
        "diabetes", "hypertension", "cancer", "stroke", "asthma",
        "pneumonia", "infection", "sepsis", "shock", "trauma",
        "fracture", "bleeding", "hemorrhage", "embolism", "thrombosis",
        "malignant", "benign", "tumor", "cyst", "abscess"
    }
    
    if words & common_conditions:
        return True
    
    # Medical symptoms and procedures (core vocabulary)
    medical_core = {
        "symptom", "pain", "fever", "cough", "nausea", "vomiting",
        "diarrhea", "constipation", "fatigue", "weakness", "dizziness",
        "headache", "migraine", "seizure", "paralysis", "numbness",
        "treatment", "therapy", "medication", "drug", "antibiotic",
        "surgery", "diagnosis", "test", "vaccine", "disease", "disorder"
    }
    
    if words & medical_core:
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
