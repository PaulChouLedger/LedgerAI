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

# Import adaptive diagnostic engine for guideline-based assessment
try:
    from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
    ADAPTIVE_ENGINE_AVAILABLE = True
    print("[Unified Medical] ✅ Adaptive diagnostic engine imported successfully")
except ImportError as e:
    ADAPTIVE_ENGINE_AVAILABLE = False
    print(f"[Unified Medical] ⚠️ Adaptive engine not available: {e}")


class RAGEmbeddingAPI:
    """
    Wrapper for RAG container's embedding service
    Provides same interface as SentenceTransformer but uses API calls
    """
    def __init__(self, rag_url: str = "http://localhost:11435"):
        self.rag_url = rag_url
    
    def encode(self, texts: List[str]) -> List:
        """
        Generate embeddings via RAG container API
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (numpy arrays)
        
        Raises:
            RuntimeError if embedding service fails
        """
        import numpy as np
        import requests
        
        response = requests.post(
            f"{self.rag_url}/embed",
            json={"texts": texts},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get('embeddings', [])
            # Convert to numpy arrays
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        else:
            raise RuntimeError(f"RAG embed API returned status {response.status_code}")

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
        self.active_assessment = None  # Legacy - kept for compatibility
        self.dynamic_assessment = None  # Legacy - kept for compatibility
        self.medical_rag = None
        
        # NEW: Adaptive diagnostic engine (with LLM + RAG embeddings for semantic similarity)
        self.adaptive_engine = None
        
        if ADAPTIVE_ENGINE_AVAILABLE:
            try:
                # Use RAG container's embedding service (no local model needed)
                embedding_api = RAGEmbeddingAPI()
                
                # Initialize adaptive engine with embeddings from RAG container
                self.adaptive_engine = AdaptiveDiagnosticEngine(
                    llm_chat_fn=self.llm_chat_fn,
                    embedding_model=embedding_api
                )
                print("[Unified Medical] ✅ Adaptive engine initialized with LLM + RAG embeddings")
            except Exception as e:
                print(f"[Unified Medical] ⚠️ Failed to initialize adaptive engine: {e}")
        
        # Assessment mode selection
        self.use_adaptive_engine = True  # Use new adaptive engine (not rigid triage)

        # Medical knowledge state
        self.last_medical_query = None
        self.medical_context = {}

        # Initialize medical RAG
        self._initialize_medical_rag()

        print(f"[Unified Medical] 🩺 Starting unified medical session: {session_id}")
    
    def _save_assessment_state(self):
        """Persist dynamic assessment state to session file"""
        if not self.dynamic_assessment:
            return
        
        # Import triage's state management
        from triage import load_state, save_state
        
        state = load_state(self.session_id)
        
        # Save dynamic assessment data
        state['dynamic_assessment'] = {
            'chief_complaint': self.dynamic_assessment.chief_complaint,
            'category': self.dynamic_assessment.category,
            'questions_asked': self.dynamic_assessment.questions_asked,
            'responses_received': self.dynamic_assessment.responses_received,
            'symptoms_collected': self.dynamic_assessment.symptoms_collected,
            'red_flags_detected': self.dynamic_assessment.red_flags_detected,
            'urgency_score': self.dynamic_assessment.urgency_score,
            'completed': self.dynamic_assessment.completed
        }
        
        save_state(state, self.session_id)
        print(f"[Unified Medical] 💾 Saved assessment state (Q{len(self.dynamic_assessment.questions_asked)})")
    
    def _load_assessment_state(self):
        """Restore dynamic assessment state from session file"""
        from triage import load_state
        
        state = load_state(self.session_id)
        assessment_data = state.get('dynamic_assessment')
        
        if assessment_data and not assessment_data.get('completed'):
            print(f"[Unified Medical] 📂 Restoring assessment state (Q{len(assessment_data.get('questions_asked', []))})")
            
            # Recreate DynamicAssessmentState object
            self.dynamic_assessment = DynamicAssessmentState(
                chief_complaint=assessment_data['chief_complaint']
            )
            self.dynamic_assessment.category = assessment_data.get('category', 'unknown')
            self.dynamic_assessment.questions_asked = assessment_data.get('questions_asked', [])
            self.dynamic_assessment.responses_received = assessment_data.get('responses_received', [])
            self.dynamic_assessment.symptoms_collected = assessment_data.get('symptoms_collected', [])
            self.dynamic_assessment.red_flags_detected = assessment_data.get('red_flags_detected', [])
            self.dynamic_assessment.urgency_score = assessment_data.get('urgency_score', 0.0)
            self.dynamic_assessment.completed = assessment_data.get('completed', False)
            
            self.current_context = "assessment"
            print(f"[Unified Medical] ✅ Restored: {len(self.dynamic_assessment.questions_asked)} questions asked, {len(self.dynamic_assessment.responses_received)} responses received")

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

        # PRIORITY 1: Check if adaptive engine has active assessment
        if self.adaptive_engine and self.adaptive_engine.status in ["questioning", "red_flag_screening"]:
            print(f"[Unified Medical] 🔄 Continuing active adaptive assessment (status: {self.adaptive_engine.status})")
            return self._handle_symptom_assessment(user_input)
        
        # PRIORITY 2: Check if we have an active dynamic assessment in progress (legacy)
        if self.dynamic_assessment and not self.dynamic_assessment.completed:
            print(f"[Unified Medical] 🔄 Continuing active dynamic assessment (Q{len(self.dynamic_assessment.questions_asked)})")
            return self._handle_dynamic_assessment(user_input)
        
        # PRIORITY 3: Analyze the query type for NEW queries only
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

        # Check for symptom assessment (any mention of medical symptoms)
        # Simple keyword-based - no grammar restrictions
        symptom_keywords = [
            'pain', 'ache', 'hurt', 'sore', 'nausea', 'vomit', 'fever',
            'cough', 'bleeding', 'dizzy', 'headache', 'chest', 'abdomen',
            'stomach', 'belly', 'breathing', 'swelling', 'rash', 'fatigue',
            'weakness', 'numbness', 'tingling', 'burning'
        ]
        
        if any(symptom in query_lower for symptom in symptom_keywords):
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
        Handle symptom assessment using ADAPTIVE guideline-based questioning
        
        New approach: Uses adaptive diagnostic engine with multi-guideline scoring,
        intelligent question selection, and natural language understanding
        """
        print(f"[Unified Medical] 🩺 Handling symptom assessment: {symptom_query}")

        # Use adaptive engine (new approach)
        if self.use_adaptive_engine and self.adaptive_engine:
            try:
                # Check if assessment is active
                if self.adaptive_engine.status == "idle":
                    # Start new assessment
                    print("[Adaptive] 🚀 Starting new adaptive assessment")
                    response = self.adaptive_engine.start_assessment(symptom_query)
                else:
                    # Continue existing assessment
                    print("[Adaptive] 🔄 Continuing adaptive assessment")
                    response = self.adaptive_engine.process_answer(symptom_query)
                
                # Handle response
                if response.get('success'):
                    if response.get('status') == 'diagnosed':
                        # Diagnosis reached!
                        print(f"[Adaptive] ✅ Diagnosis: {response.get('diagnosis')}")
                        return response.get('message', 'Assessment complete.')
                    else:
                        # Return next question
                        return response.get('question', 'Can you tell me more?')
                else:
                    # Error or no match
                    return response.get('message', 'I need more information to help you.')
            
            except Exception as e:
                print(f"[Adaptive] ❌ Adaptive engine failed: {e}")
                import traceback
                traceback.print_exc()
                return "I'm having trouble processing your symptoms. Please provide more details."
        
        # Fallback to old dynamic assessment if adaptive engine not available
        if hasattr(self, 'use_dynamic_assessment') and self.use_dynamic_assessment:
            return self._handle_dynamic_assessment(symptom_query)
        
        # Final fallback
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
            # Check if user is trying to exit/change topic
            exit_attempts = ['nevermind', 'never mind', 'forget it', 'cancel', 'stop', 'exit', 'quit']
            if any(phrase in symptom_query.lower() for phrase in exit_attempts):
                return "I understand you want to stop, but for your safety, I need to complete the assessment. Let me ask just a few more questions to ensure you get the right care. " + \
                       (self.dynamic_assessment.questions_asked[-1] if self.dynamic_assessment.questions_asked else "Where is the pain located?")
            
            # Validate the answer is reasonable
            if not self._is_valid_answer(symptom_query):
                print(f"[Dynamic] ⚠️ Invalid/incomplete answer: '{symptom_query}' - RE-ASKING (not advancing)")
                # Re-ask the same question with clarification
                last_question = self.dynamic_assessment.questions_asked[-1] if self.dynamic_assessment.questions_asked else None
                if last_question:
                    # Clean the question of any Q# prefix before re-asking
                    clean_question = re.sub(r'^\s*Q\d+\s*:\s*', '', last_question)
                    print(f"[Dynamic] 🔁 Re-asking: '{clean_question}'")
                    return f"I didn't quite catch that. {clean_question}"
                else:
                    raise ValueError("No previous question to re-ask")
            
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
    
    def _is_valid_answer(self, answer: str) -> bool:
        """
        Validate if answer is substantive enough to be a real medical response
        
        Rejects:
        - Very short answers with no content ("on the", "time.", ".")
        - Filler words only
        - Empty or whitespace-only
        
        Accepts:
        - Numeric responses ("7", "eight")
        - Yes/no
        - Location descriptions ("upper right", "center")
        - Time descriptions ("yesterday", "2 days ago")
        - Symptom descriptions ("nauseous", "vomiting")
        """
        if not answer or len(answer.strip()) < 2:
            return False
        
        answer_lower = answer.lower().strip()
        
        # Remove punctuation for checking
        answer_clean = re.sub(r'[^\w\s]', '', answer_lower)
        
        # Very short and not a valid short answer
        if len(answer_clean) < 3:
            # Allow single valid words
            valid_short_answers = ['yes', 'no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
            if answer_clean not in valid_short_answers and not answer_clean.isdigit():
                return False
        
        # Check for common filler/incomplete phrases
        invalid_patterns = [
            r'^(on the|in the|at the|to the|from the)$',
            r'^(time|day|night|ago|hour|minute)$',  # Time words without context
            r'^(and|but|or|so|then|well|um|uh)$',  # Conjunctions/fillers alone
            r'^(it|this|that|there|here)$',  # Pronouns without context
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, answer_clean):
                return False
        
        # Accept if it has at least 2 words OR is a number OR is yes/no
        word_count = len(answer_clean.split())
        if word_count >= 2 or answer_clean.isdigit() or answer_clean in ['yes', 'no', 'yup', 'nope', 'yeah', 'nah']:
            return True
        
        return False
    
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
        """
        Retrieve TOP DIFFERENTIAL DIAGNOSES from RAG
        
        Returns multiple guidelines (top 3-5 differentials) so LLM can ask
        discriminating questions to narrow them down.
        
        Example:
        - "Abdominal pain" → Returns Appendicitis, Pancreatitis, Cholecystitis
        - LLM asks: "Where is the pain? RUQ, RLQ, epigastric?"
        - Answer "RLQ" → LLM focuses on Appendicitis guideline
        
        This is how real clinicians think - differential diagnosis.
        """
        try:
            # Search SPECIFICALLY for medical guideline documents
            # Use the exact header format to ensure we get medical guidelines, not business docs
            search_query = f"DIAGNOSTIC GUIDELINE gastrointestinal {query}" if category else f"DIAGNOSTIC GUIDELINE {query}"
            
            print(f"[Dynamic] 🔍 Searching RAG with query: '{search_query}'")
            
            response = requests.post(
                "http://localhost:11435/rag/search",
                json={
                    "query": search_query, 
                    "top_k": 50,  # Get many chunks to cover multiple guidelines
                    "disable_keyword_filter": True,  # MUST disable to get all guideline chunks
                    "min_score": 0.0
                },
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"[Dynamic] ⚠️ RAG search failed: HTTP {response.status_code}")
                return []
            
            all_results = response.json().get('results', [])
            print(f"[Dynamic] 📚 Stage 1: Retrieved {len(all_results)} chunks for initial guideline identification")
            
            # STAGE 2: Extract guideline names from results and fetch complete guidelines
            import re
            guideline_names = set()
            
            for result in all_results:
                text = result.get('text', '')
                
                # Find guideline name in chunk
                if 'DIAGNOSTIC GUIDELINE:' in text:
                    match = re.search(r'DIAGNOSTIC GUIDELINE:\s*([^\n]+)', text)
                    if match:
                        guideline_name = match.group(1).strip()
                        guideline_names.add(guideline_name)
            
            if not guideline_names:
                print(f"[Dynamic] ⚠️ No guidelines found in search results")
                return all_results[:10]
            
            # Get top 3 differential diagnoses
            top_differentials = list(guideline_names)[:3]
            
            print(f"[Dynamic] 🎯 Top {len(top_differentials)} differential diagnoses identified:")
            for i, gname in enumerate(top_differentials, 1):
                print(f"[Dynamic]    {i}. {gname}")
            
            # STAGE 3: Retrieve ALL chunks from these guidelines using metadata
            all_guideline_chunks = []
            
            for guideline_name in top_differentials:
                try:
                    # Use new metadata-based endpoint to get ALL chunks
                    response2 = requests.get(
                        f"http://localhost:11435/rag/guideline/{guideline_name}",
                        timeout=10
                    )
                    
                    if response2.status_code == 200:
                        guideline_data = response2.json()
                        chunks_from_guideline = guideline_data.get('results', [])
                        
                        print(f"[Dynamic] 📋 Retrieved {len(chunks_from_guideline)} chunks from {guideline_name}")
                        all_guideline_chunks.extend(chunks_from_guideline)
                    else:
                        print(f"[Dynamic] ⚠️ Failed to get chunks for {guideline_name}: HTTP {response2.status_code}")
                        
                except Exception as e:
                    print(f"[Dynamic] ⚠️ Error fetching chunks for {guideline_name}: {e}")
            
            if all_guideline_chunks:
                print(f"[Dynamic] 📚 Total chunks from all differentials: {len(all_guideline_chunks)}")
                return all_guideline_chunks
            else:
                print(f"[Dynamic] ⚠️ Metadata retrieval failed - falling back to search results")
                return all_results[:20]
                
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
        
        # Extract clinical guidance from guidelines
        # Look for key clinical information across all retrieved chunks
        diagnostic_section = ""
        
        # Combine all guideline chunks into one text for better extraction
        combined_text = ""
        for guideline in guidelines[:10]:  # Use up to 10 chunks to get full guideline
            combined_text += guideline.get('text', '') + "\n\n"
        
        print(f"[Dynamic] 📝 Combined guideline text: {len(combined_text)} chars from {len(guidelines)} chunks")
        
        # Debug: Show what we actually got from RAG
        if len(combined_text) > 0:
            preview = combined_text[:300].replace('\n', ' ')
            print(f"[Dynamic] 📄 Guideline preview: {preview}...")
        else:
            print(f"[Dynamic] ❌ WARNING: No guideline text retrieved!")
        
        # CRITICAL CHECK: Ensure we got meaningful content
        if len(combined_text) < 500:
            print(f"[Dynamic] ❌ FATAL: Insufficient guideline content ({len(combined_text)} chars)")
            print(f"[Dynamic] ❌ RAG returned only {len(guidelines)} chunks")
            raise ValueError(f"Insufficient guideline content: {len(combined_text)} chars from {len(guidelines)} chunks")
        
        if 'QUESTION 1:' not in combined_text and 'DIAGNOSTIC' not in combined_text:
            print(f"[Dynamic] ❌ FATAL: No diagnostic questions found in guideline text")
            print(f"[Dynamic] ❌ Content preview: {combined_text[:500]}")
            raise ValueError("No diagnostic questions found in retrieved guidelines")
        
        # Try to extract diagnostic questioning section
        if 'DIAGNOSTIC QUESTIONING STRATEGY' in combined_text or 'QUESTION 1:' in combined_text:
            # Find the diagnostic questions
            question_blocks = []
            
            # Look for QUESTION 1, 2, 3 patterns
            for i in range(1, 5):  # QUESTION 1-4
                q_marker = f'QUESTION {i}:'
                if q_marker in combined_text:
                    q_start = combined_text.find(q_marker)
                    
                    # Find end (next question or 300 chars)
                    q_end_markers = [
                        combined_text.find(f'QUESTION {i+1}:', q_start),
                        combined_text.find('---', q_start + 50),
                        combined_text.find('RED FLAG', q_start),
                        q_start + 350
                    ]
                    q_end = min([m for m in q_end_markers if m > q_start and m != -1], default=q_start + 350)
                    
                    block = combined_text[q_start:q_end].strip()
                    if len(block) > 30:  # Valid block
                        question_blocks.append(block)
            
            if question_blocks:
                diagnostic_section = '\n\n'.join(question_blocks[:3])  # Use first 3 questions
                print(f"[Dynamic] ✅ Extracted {len(question_blocks)} diagnostic question blocks")
        
        # NO FALLBACK: Fail if no diagnostic section found
        if not diagnostic_section:
            print(f"[Dynamic] ❌ FATAL: No diagnostic questions extracted from guidelines")
            print(f"[Dynamic] ❌ Combined text length: {len(combined_text)}")
            print(f"[Dynamic] ❌ Contains 'QUESTION 1:': {'QUESTION 1:' in combined_text}")
            print(f"[Dynamic] ❌ Contains 'DIAGNOSTIC QUESTIONING': {'DIAGNOSTIC QUESTIONING' in combined_text}")
            raise ValueError("Failed to extract diagnostic questions from guidelines")
        
        print(f"[Dynamic] 📝 Final diagnostic section: {len(diagnostic_section)} chars")
        
        # Build context from previous Q&A
        qa_history = ""
        if state.questions_asked and state.responses_received:
            recent_qa = list(zip(state.questions_asked[-2:], state.responses_received[-2:]))
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
        
        print(f"[Dynamic] 📋 Asked topics: {asked_topics}")
        
        # Build prompt using diagnostic section from guideline (no fallback)
        if qa_history:
            prompt = f"""Patient: {state.chief_complaint}

Previous:
{qa_history}
Clinical guidance:
{diagnostic_section}

Based on the guidance above, ask the NEXT logical question. Don't repeat what's already asked.
Question:"""
        else:
            prompt = f"""Patient: {state.chief_complaint}

Clinical guidance:
{diagnostic_section}

Based on the guidance, ask the FIRST diagnostic question.
Question:"""
        
        print(f"[Dynamic] 📝 Prompt length: {len(prompt)} chars, approx {len(prompt)//4} tokens")
        
        # Get LLM response (llm_chat now returns string, not dict)
        question = self.llm_chat_fn(
            [{"role": "user", "content": prompt}],
            max_tokens=80,      # Enough for a medical question
            temperature=0.4,    # Lower temperature for more consistent output
            stream=False
        )
        
        print(f"[Dynamic] 🤖 Raw LLM output: '{question}'")
        
        # Clean up question - remove meta-commentary and Q/A formatting
        if question:
            original_question = question  # For debugging
            
            # Remove Q#: prefix (be more aggressive with whitespace)
            question = re.sub(r'^\s*Q\d+\s*:\s*', '', question)  # Remove "Q4: " or " Q4 : "
            question = re.sub(r'\n+A\d+:.*$', '', question, flags=re.DOTALL)  # Remove "A4: ..." if present
            
            if original_question != question:
                print(f"[Dynamic] 🧹 Cleaned Q/A prefix: '{original_question[:50]}...' → '{question[:50]}...'")
            
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
            
            # Final sanity check - if question is too short or doesn't make sense, FAIL
            if len(question) < 10 or not question.endswith('?'):
                print(f"[Dynamic] ❌ FATAL: Malformed question after cleaning: '{question}'")
                raise ValueError(f"LLM generated malformed question: '{question}'")
        
        # Validate output - check for garbage/repetitive content  
        if question and len(question) > 10:
            from collections import Counter
            char_counts = Counter(question.lower())
            most_common = char_counts.most_common(1)[0][1] if char_counts else 0
            if most_common / len(question) > 0.3:  # >30% same character = garbage
                print(f"[Dynamic] ❌ FATAL: Garbage output detected (char '{char_counts.most_common(1)[0][0]}' appears {most_common}/{len(question)} times)")
                raise ValueError(f"LLM generated garbage: {question[:100]}")
        
        # Check if question was already asked (repeated)
        if state.questions_asked:
            for prev_q in state.questions_asked:
                # Simple similarity check - if >70% of words match, it's a repeat
                q_words = set(question.lower().split())
                prev_words = set(prev_q.lower().split())
                if len(q_words & prev_words) / max(len(q_words), 1) > 0.7:
                    print(f"[Dynamic] ❌ FATAL: LLM repeated question")
                    print(f"[Dynamic] ❌ Previous: '{prev_q}'")
                    print(f"[Dynamic] ❌ New:      '{question}'")
                    raise ValueError(f"LLM repeated question: {question}")
        
        # Final log with cleaned question
        print(f"[Dynamic] ❓ Final question (after all cleaning): {question}")
        
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
        
        # Minimum questions required before diagnosis
        MIN_QUESTIONS_FOR_DIAGNOSIS = 4
        
        # Complete if:
        # 1. Emergency detected (urgency >= 8) AND at least minimum questions asked
        if state.urgency_score >= 8.0 and len(state.questions_asked) >= MIN_QUESTIONS_FOR_DIAGNOSIS:
            print(f"[Dynamic] ⚠️ Emergency detected (urgency={state.urgency_score}) - completing assessment")
            return True
        
        # 2. Multiple red flags detected AND sufficient questions
        if len(state.red_flags_detected) >= 2 and len(state.questions_asked) >= MIN_QUESTIONS_FOR_DIAGNOSIS:
            print(f"[Dynamic] ⚠️ Red flags detected ({len(state.red_flags_detected)}) - completing assessment")
            return True
        
        # 3. Sufficient questions asked (6-8 questions for thorough assessment)
        if len(state.questions_asked) >= 6:
            print(f"[Dynamic] ✅ Sufficient information gathered ({len(state.questions_asked)} questions) - ready for diagnosis")
            return True
        
        # 4. Don't complete too early
        if len(state.questions_asked) < MIN_QUESTIONS_FOR_DIAGNOSIS:
            print(f"[Dynamic] 🔄 Need more info ({len(state.questions_asked)}/{MIN_QUESTIONS_FOR_DIAGNOSIS} questions) - continuing assessment")
            return False
        
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
        
        diagnosis_prompt = f"""Complete medical assessment:

Chief complaint: {state.chief_complaint}
Category: {state.category}

Assessment Q&A:
{self._format_qa_history(state)}

Symptoms: {', '.join([str(s.get('symptom', s)) for s in state.symptoms_collected])}
Red flags: {', '.join(state.red_flags_detected) if state.red_flags_detected else 'None'}

Guidelines:
{guideline_text}

Provide:
1. Most likely diagnosis (or top 2-3 differentials)
2. Urgency (1-10 scale)
3. Specific next steps: See doctor today? Go to ER? Call 911? Home care OK?

Be direct. Provide clear medical advice.

Assessment:"""
        
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
                print(f"[Dynamic] ❌ FATAL: Garbage diagnosis detected (char '{char_counts.most_common(1)[0][0]}' appears {most_common}/{len(diagnosis_response)} times)")
                raise ValueError(f"LLM generated garbage diagnosis: {diagnosis_response[:100]}")
        
        # Mark assessment as complete
        state.completed = True
        self.current_context = "general"
        self.dynamic_assessment = None
        
        # Clear from session state to allow mode switching
        from triage import load_state, save_state
        session_state = load_state(self.session_id)
        if 'dynamic_assessment' in session_state:
            del session_state['dynamic_assessment']
        if 'mode' in session_state:
            del session_state['mode']
        save_state(session_state, self.session_id)
        print(f"[Dynamic] ✅ Assessment complete - session state cleared")
        
        # Add completion message to let user know they can ask other questions now
        completion_note = "\n\nThis completes your medical assessment. Feel free to ask me anything else."
        
        return diagnosis_response + completion_note
    
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

    # Simple, flexible approach: Check for medical keywords
    # Load from medical_terms.json (shared across all containers)
    medical_keywords = _get_medical_keywords()
    
    # Check for medical keywords (fuzzy/partial match)
    for keyword in medical_keywords:
        if keyword.lower() in prompt_lower:
            print(f"[Unified Medical] 🎯 Medical keyword: '{keyword}'")
        return True

    # Also check built-in common symptom terms (always medical)
    common_symptoms = [
        'pain', 'ache', 'hurt', 'sore', 'nausea', 'nauseous', 'vomit', 'vomiting',
        'fever', 'cough', 'coughing', 'bleeding', 'bleed', 'dizzy', 'dizziness',
        'headache', 'migraine', 'chest', 'abdomen', 'abdominal', 'stomach', 'belly',
        'shortness of breath', 'breathing', 'breath', 'swelling', 'swollen', 'rash',
        'fatigue', 'tired', 'weakness', 'numbness', 'tingling', 'burning'
    ]
    
    for symptom in common_symptoms:
        if symptom in prompt_lower:
            print(f"[Unified Medical] 🎯 Common symptom: '{symptom}'")
            return True

    # General medical context
    if any(term in prompt_lower for term in ["medical", "diagnosis", "treatment", "symptom"]):
        return True

    return False


def _get_medical_keywords() -> list:
    """
    Get comprehensive medical keywords from:
    1. medical_terms.json (organized by organ system)
    2. All synonym files (LLM container)
    
    Returns flattened list of all medical terms
    """
    all_keywords = []
    
    try:
        import json
        from pathlib import Path
        
        # 1. Load from shared medical_terms.json (organized by organ system)
        medical_terms_path = Path('/shared/medical_terms.json')
        if medical_terms_path.exists():
            with open(medical_terms_path, 'r') as f:
                data = json.load(f)
            
            # File is organized: {specialty: [terms], specialty: [terms], ...}
            for specialty, terms in data.items():
                # Skip metadata and non-list fields
                if specialty in ['metadata', 'proper_names'] or not isinstance(terms, list):
                    continue
                
                all_keywords.extend(terms)
        
        # 2. Load from all synonym files
        synonym_dir = Path('/app/synonyms')
        if synonym_dir.exists():
            for synonym_file in synonym_dir.glob('*_synonyms.json'):
                try:
                    with open(synonym_file, 'r') as f:
                        synonyms = json.load(f)
                    
                    # Each file is a dict where keys and values are medical terms
                    for term, variants in synonyms.items():
                        all_keywords.append(term)
                        if isinstance(variants, list):
                            all_keywords.extend(variants)
                
                except Exception as e:
                    print(f"[Unified Medical] ⚠️ Error loading {synonym_file.name}: {e}")
        
        # Remove duplicates
        unique_keywords = list(set([k.lower() for k in all_keywords]))
        
        if unique_keywords:
            print(f"[Unified Medical] 📚 Loaded {len(unique_keywords)} medical terms from all sources")
        
        return unique_keywords
        
    except Exception as e:
        print(f"[Unified Medical] ⚠️ Error loading medical keywords: {e}")
        return []


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
    
    # Load persisted assessment state if it exists
    session._load_assessment_state()
    
    # Process the query
    response = session.process_medical_query(prompt)
    
    # Save assessment state for next request
    session._save_assessment_state()
    
    return response


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
