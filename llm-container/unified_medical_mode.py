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
import requests
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
MEDICAL_TERMS_FILE = "/shared/medical_terms.json"  # Mounted from repo_root/shared/

def _load_medical_terms():
    """Load medical terms from shared mounted volume (single source of truth)"""
    global MEDICAL_TERMS
    
    if not os.path.exists(MEDICAL_TERMS_FILE):
        raise FileNotFoundError(
            f"Medical terms file not found: {MEDICAL_TERMS_FILE}\n"
            f"Ensure shared/ directory is mounted in docker-compose.yml:\n"
            f"  volumes:\n"
            f"    - ../shared:/shared\n"
            f"And that shared/medical_terms.json exists in repo"
        )
    
    with open(MEDICAL_TERMS_FILE, 'r') as f:
        MEDICAL_TERMS = json.load(f)
    
    # Flatten all terms into a single list for fast keyword matching
    all_terms = []
    for category, terms in MEDICAL_TERMS.items():
        all_terms.extend(terms)
    MEDICAL_TERMS['_all_terms_flat'] = list(set(all_terms))  # Deduplicate
    
    print(f"[Unified Medical] ✅ Loaded {len(MEDICAL_TERMS['_all_terms_flat'])} medical terms from {MEDICAL_TERMS_FILE}")
    print(f"[Unified Medical] ✅ Categories: {', '.join([k for k in MEDICAL_TERMS.keys() if k != '_all_terms_flat'])}")

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

        # Check if we have an active dynamic assessment in progress
        if self.dynamic_assessment and not self.dynamic_assessment.completed:
            print(f"[Unified Medical] 🔄 Continuing active dynamic assessment (Q{len(self.dynamic_assessment.questions_asked)})")
            return self._handle_dynamic_assessment(user_input)
        
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
            # Use general RAG for medical guidelines
            enhanced_query = f"medical guideline {category} {query}" if category else f"medical guideline {query}"
            
            response = requests.post(
                "http://localhost:11435/rag/search",
                json={"query": enhanced_query, "top_k": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                print(f"[Dynamic] 📚 Retrieved {len(results)} guideline chunks from RAG")
                return results
            else:
                print(f"[Dynamic] ⚠️ RAG search failed: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[Dynamic] ❌ Error retrieving guidelines: {e}")
            import traceback
            traceback.print_exc()
        
        return []
    
    def _generate_dynamic_question(self, guidelines: List[Dict]) -> str:
        """
        Generate next diagnostic question using RAG-retrieved guidelines
        
        This is the core of dynamic assessment - uses real medical guidelines
        to ask intelligent, contextual questions
        """
        state = self.dynamic_assessment
        
        # Build context for LLM (truncate each guideline to prevent token overflow)
        MAX_GUIDELINE_LENGTH = 250  # characters per guideline (reduced for safety)
        truncated_guidelines = [g.get('text', '')[:MAX_GUIDELINE_LENGTH] + '...' if len(g.get('text', '')) > MAX_GUIDELINE_LENGTH else g.get('text', '') for g in guidelines[:2]]  # Use only 2 chunks
        guideline_text = "\n\n".join(truncated_guidelines)
        
        print(f"[Dynamic] 📝 Guideline text length: {len(guideline_text)} chars")
        
        # Simplified prompt to reduce token usage and improve reliability
        guideline_snippet = guideline_text[:400] if len(guideline_text) > 400 else guideline_text  # Further truncate if needed
        
        # Build context from previous Q&A (in conversational format, not Q/A labels)
        qa_history = ""
        if state.questions_asked and state.responses_received:
            recent_qa = list(zip(state.questions_asked[-2:], state.responses_received[-2:]))  # Last 2 Q&A pairs
            for q, a in recent_qa:
                qa_history += f"Asked: {q}\nThey said: {a}\n"
        
        # Determine what hasn't been asked yet
        asked_topics = set()
        if state.questions_asked:
            for q in state.questions_asked:
                q_lower = q.lower()
                if 'location' in q_lower or 'where' in q_lower:
                    asked_topics.add('location')
                if 'duration' in q_lower or 'long' in q_lower or 'when' in q_lower or 'start' in q_lower:
                    asked_topics.add('duration')
                if 'severity' in q_lower or 'bad' in q_lower or 'scale' in q_lower:
                    asked_topics.add('severity')
                if 'other' in q_lower or 'associated' in q_lower or 'symptom' in q_lower:
                    asked_topics.add('associated')
        
        # Build prompt with explicit "don't repeat" instruction
        topics_to_ask = []
        if 'location' not in asked_topics:
            topics_to_ask.append('Location of pain')
        if 'severity' not in asked_topics:
            topics_to_ask.append('Severity (1-10 scale)')
        if 'duration' not in asked_topics:
            topics_to_ask.append('When it started')
        if 'associated' not in asked_topics:
            topics_to_ask.append('Other symptoms')
        
        next_topics = ', '.join(topics_to_ask[:2]) if topics_to_ask else 'Character of pain, aggravating factors'
        
        print(f"[Dynamic] 📋 Asked topics: {asked_topics}, Next: {next_topics}")
        
        # Build a simple, clear prompt
        if qa_history:
            prompt = f"""Patient has: {state.chief_complaint}

{qa_history}
Ask ONE follow-up question about: {next_topics}

DON'T mention specific conditions or diagnoses. Just ask about their symptoms.
Question:"""
        else:
            prompt = f"""Patient has: {state.chief_complaint}

Ask ONE question about: {next_topics}

Question:"""
        
        print(f"[Dynamic] 📝 Prompt length: {len(prompt)} chars, approx {len(prompt)//4} tokens")
        
        # Get LLM response (llm_chat now returns string, not dict)
        question = self.llm_chat_fn(
            [{"role": "user", "content": prompt}],
            max_tokens=80,      # Enough for a medical question
            temperature=0.4,    # Lower temperature for more consistent output
            stream=False
        )
        
        # Clean up question - remove meta-commentary and Q/A formatting
        if question:
            # Remove Q#: prefix and anything after it
            question = re.sub(r'^Q\d+:\s*', '', question)  # Remove "Q4: "
            question = re.sub(r'\n\nA\d+:.*$', '', question, flags=re.DOTALL)  # Remove "A4: ..." if present
            
            # Remove common meta phrases
            question = re.sub(r'^(Here\'s a|I can ask|Let me ask|I\'d like to ask|A good question would be)[^:]*:\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^"(.+)"$', r'\1', question)  # Remove surrounding quotes
            
            # Remove references to "guidelines" and make conversational
            question = re.sub(r',?\s*according to (medical |the )?guidelines?', '', question, flags=re.IGNORECASE)
            question = re.sub(r',?\s*based on (medical |the )?guidelines?', '', question, flags=re.IGNORECASE)
            
            # Fix awkward medical phrasings FIRST (before removing condition names)
            question = re.sub(r'\bwhat is the most common symptom of your [a-z]+\b', 'Are you experiencing any other symptoms', question, flags=re.IGNORECASE)
            question = re.sub(r'\bthe most common symptom of\b', 'any other symptoms besides', question, flags=re.IGNORECASE)
            question = re.sub(r'\bwhat is the most common symptom\b', 'Are you experiencing any other symptoms', question, flags=re.IGNORECASE)
            
            # Remove specific condition names that leak from guidelines
            condition_names = [
                'pancreatitis', 'appendicitis', 'cholecystitis', 'diverticulitis',
                'gastroenteritis', 'peptic ulcer', 'gallstone', 'kidney stone',
                'heart attack', 'stroke', 'seizure', 'pneumonia', 'asthma',
                'COPD', 'diabetes', 'hypertension'
            ]
            
            for condition in condition_names:
                # Remove the condition name entirely along with possessive/article
                question = re.sub(rf'\byour {condition}\b', '', question, flags=re.IGNORECASE)
                question = re.sub(rf'\bthe {condition}\b', '', question, flags=re.IGNORECASE)
                question = re.sub(rf'\b{condition}\b', '', question, flags=re.IGNORECASE)
            
            # Clean up extra spaces and awkward phrasings from removals
            question = re.sub(r'\s{2,}', ' ', question)  # Multiple spaces → single space
            question = re.sub(r'\s+\?', '?', question)  # Space before question mark
            question = re.sub(r'\bfrom\s*\?', '?', question)  # "from ?" → "?"
            question = re.sub(r'\bof\s+\?', '?', question)  # "of ?" → "?"
            question = re.sub(r'\bbesides\s*\?', '?', question)  # "besides ?" → "?"
            
            # If multi-line, take only the first line (should be the question)
            if '\n' in question:
                question = question.split('\n')[0].strip()
            
            question = question.strip()
            
            # Final sanity check - if question is too short or doesn't make sense, use fallback
            if len(question) < 10 or not question.endswith('?'):
                print(f"[Dynamic] ⚠️ Malformed question after cleaning: '{question}', using fallback")
                question = "Are you experiencing any other symptoms?"
        
        # Validate output - check for garbage/repetitive content
        if question and len(question) > 10:
            from collections import Counter
            char_counts = Counter(question.lower())
            most_common = char_counts.most_common(1)[0][1] if char_counts else 0
            if most_common / len(question) > 0.3:  # >30% same character = garbage
                print(f"[Dynamic] ⚠️ Garbage output detected, using fallback question")
                question = "Can you describe where the pain is located and how severe it is on a scale of 1-10?"
        
        # Check if question was already asked (repeated)
        if state.questions_asked:
            for prev_q in state.questions_asked:
                # Simple similarity check - if >70% of words match, it's a repeat
                q_words = set(question.lower().split())
                prev_words = set(prev_q.lower().split())
                if len(q_words & prev_words) / max(len(q_words), 1) > 0.7:
                    print(f"[Dynamic] ⚠️ Repeated question detected, using fallback")
                    # Use fallback based on what hasn't been asked
                    fallback_questions = [
                        "Where exactly is the pain located in your abdomen?",
                        "On a scale of 1 to 10, how severe is the pain?",
                        "Are you experiencing any nausea, vomiting, or fever?",
                        "Does the pain get worse with movement or eating?",
                        "Have you had any changes in bowel movements or appetite?"
                    ]
                    # Pick first fallback that wasn't asked
                    for fallback in fallback_questions:
                        if all(len(set(fallback.lower().split()) & set(pq.lower().split())) / len(set(fallback.lower().split())) < 0.7 for pq in state.questions_asked):
                            question = fallback
                            break
                    break
        
        print(f"[Dynamic] ❓ Generated question: {question}")
        
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
        
        # Truncate each guideline to prevent token overflow
        MAX_GUIDELINE_LENGTH = 300  # characters per guideline
        truncated_guidelines = [g.get('text', '')[:MAX_GUIDELINE_LENGTH] + '...' if len(g.get('text', '')) > MAX_GUIDELINE_LENGTH else g.get('text', '') for g in guidelines[:3]]
        guideline_text = "\n\n".join(truncated_guidelines)
        
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
        
        # Get diagnosis from LLM (llm_chat now returns string, not dict)
        diagnosis_response = self.llm_chat_fn(
            [{"role": "user", "content": diagnosis_prompt}],
            max_tokens=150,     # Enough for diagnosis + disposition
            temperature=0.3,    # Low temperature for clinical accuracy
            stream=False
        )
        
        # Validate output - check for garbage/repetitive content
        if diagnosis_response and len(diagnosis_response) > 10:
            from collections import Counter
            char_counts = Counter(diagnosis_response.lower())
            most_common = char_counts.most_common(1)[0][1] if char_counts else 0
            if most_common / len(diagnosis_response) > 0.3:  # >30% same character = garbage
                print(f"[Dynamic] ⚠️ Garbage diagnosis detected, using fallback")
                diagnosis_response = "Based on your symptoms, I recommend seeing a healthcare provider for proper evaluation. If symptoms worsen or you experience severe pain, seek immediate medical attention."
        
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
