#!/usr/bin/env python3
"""
Aura Clinician Mode - RAG-Powered Diagnostic Reasoning

Unlike the hardcoded Triage system, Clinician mode:
- Uses RAG to search multi-organ medical guidelines
- Generates intelligent, context-aware questions
- Thinks like a doctor with differential diagnosis
- Adapts questioning based on findings
- Provides comprehensive analysis

This is the next evolution of medical AI conversation.
"""

import requests
import re
from typing import List, Dict, Any, Optional, Callable
from difflib import SequenceMatcher

class ClinicianSession:
    """
    Manages a clinician-mode diagnostic conversation
    Uses RAG to access medical guidelines and generate intelligent questions
    """
    
    def __init__(self, session_id: str, chief_complaint: str, llm_chat_fn: Callable):
        self.session_id = session_id
        self.chief_complaint = chief_complaint
        self.llm_chat_fn = llm_chat_fn  # Direct LLM access, no HTTP
        self.conversation_history = []
        self.findings = {}
        self.differential_diagnoses = []
        self.current_focus = None  # What organ system/condition we're investigating
        
    def start_session(self) -> str:
        """
        Initialize diagnostic conversation
        
        Returns:
            Opening statement and first intelligent question
        """
        print(f"[Clinician] 🩺 Starting diagnostic session for: '{self.chief_complaint}'")
        
        # Search RAG for relevant medical guidelines
        guidelines = self._search_medical_guidelines(self.chief_complaint)
        
        if not guidelines:
            print(f"[Clinician] ⚠️ No medical guidelines found, using general approach")
            return self._fallback_opening()
        
        print(f"[Clinician] 📚 Found {len(guidelines)} relevant guidelines")
        
        # Analyze chief complaint and generate intelligent first question
        return self._generate_opening_response(guidelines)
    
    def process_response(self, user_response: str) -> str:
        """
        Process user's response and generate next intelligent question
        
        Args:
            user_response: What the user said
            
        Returns:
            Next question or diagnostic conclusion
        """
        # Store response
        self.conversation_history.append({
            'role': 'patient',
            'content': user_response
        })
        
        # Extract findings from response
        self._extract_findings(user_response)
        
        # Search RAG for relevant info based on accumulated findings
        context = self._build_current_context()
        guidelines = self._search_medical_guidelines(context)
        
        # Generate next intelligent question using LLM + RAG
        return self._generate_next_question(guidelines)
    
    def _search_medical_guidelines(self, query: str) -> List[Dict[str, Any]]:
        """
        Search RAG for relevant medical guidelines
        
        Args:
            query: Clinical query
            
        Returns:
            List of relevant guideline chunks
        """
        try:
            response = requests.post(
                "http://localhost:11435/rag/search",
                json={"query": query, "k": 5},  # Get more results for clinical context
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"[Clinician] 📚 RAG search: {len(results)} guidelines found for '{query[:50]}...'")
                return results
            
        except Exception as e:
            print(f"[Clinician] ⚠️ RAG search failed: {e}")
        
        return []
    
    def _generate_opening_response(self, guidelines: List[Dict[str, Any]]) -> str:
        """
        Generate intelligent opening question based on medical guidelines
        
        Args:
            guidelines: RAG results with relevant medical info
            
        Returns:
            Opening statement and first question
        """
        # Build context from guidelines
        context = "\n\n".join([g.get('text', g.get('chunk', '')) for g in guidelines[:3]])
        
        # Use LLM to generate intelligent opening
        system_prompt = """You are a skilled clinician taking a medical history.
Be conversational, empathetic, and clinically sound. Ask only ONE focused question.
Provide:
1. A brief acknowledgment of their concern (1 sentence)
2. The MOST important first question to ask (guided by clinical guidelines)"""

        user_prompt = f"""Patient's chief complaint: "{self.chief_complaint}"

Relevant medical guidelines:
{context}"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM directly (non-streaming for simplicity)
            response = self.llm_chat_fn(
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                stream=False
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else self._fallback_opening()
            
        except Exception as e:
            print(f"[Clinician] ❌ Error generating opening: {e}")
            return self._fallback_opening()
    
    def _generate_next_question(self, guidelines: List[Dict[str, Any]]) -> str:
        """
        Generate next intelligent question based on conversation so far
        
        Args:
            guidelines: Current RAG results
            
        Returns:
            Next question or diagnostic conclusion
        """
        # Build conversation context
        conversation_text = self._format_conversation_history()
        
        # Build guideline context
        guideline_context = "\n\n".join([g.get('text', g.get('chunk', '')) for g in guidelines[:3]])
        
        system_prompt = """You are a skilled clinician conducting a diagnostic interview.
Be conversational, empathetic, and clinically sound.

Based on the patient's responses and clinical guidelines:
1. What is the NEXT most important question to ask?
2. Are you ready to provide a differential diagnosis?

If more information is needed, ask ONE focused, clinically relevant question.
If you have enough information, provide a brief differential diagnosis and recommendation."""

        user_prompt = f"""Chief complaint: "{self.chief_complaint}"

Conversation so far:
{conversation_text}

Relevant medical guidelines:
{guideline_context}"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM directly (non-streaming for simplicity)
            response = self.llm_chat_fn(
                messages=messages,
                max_tokens=250,
                temperature=0.7,
                stream=False
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else "I apologize, but I'm having trouble processing right now. Could you describe your symptoms again?"
            
        except Exception as e:
            print(f"[Clinician] ❌ Error generating question: {e}")
            return "I apologize, but I'm having trouble processing right now. Could you describe your symptoms again?"
    
    def _extract_findings(self, response: str):
        """
        Extract clinical findings from user's response
        
        Args:
            response: User's answer
        """
        # Simple extraction - can be enhanced later
        response_lower = response.lower()
        
        # Common clinical findings
        if "severe" in response_lower:
            self.findings['severity'] = 'severe'
        elif "moderate" in response_lower:
            self.findings['severity'] = 'moderate'
        elif "mild" in response_lower:
            self.findings['severity'] = 'mild'
        
        # Time-based
        if any(time in response_lower for time in ["hour", "hours", "today", "this morning"]):
            self.findings['onset'] = 'acute'
        elif any(time in response_lower for time in ["day", "days", "week", "weeks"]):
            self.findings['onset'] = 'subacute'
        elif any(time in response_lower for time in ["month", "months", "year", "years"]):
            self.findings['onset'] = 'chronic'
        
        # Location-specific findings can be added as needed
        
        print(f"[Clinician] 📋 Current findings: {self.findings}")
    
    def _build_current_context(self) -> str:
        """
        Build current clinical context for RAG search
        
        Returns:
            Summary of chief complaint + findings
        """
        context_parts = [self.chief_complaint]
        
        if self.findings:
            findings_text = ", ".join([f"{k}: {v}" for k, v in self.findings.items()])
            context_parts.append(findings_text)
        
        return " | ".join(context_parts)
    
    def _format_conversation_history(self) -> str:
        """
        Format conversation history for LLM context
        
        Returns:
            Formatted conversation
        """
        lines = [f"Chief Complaint: {self.chief_complaint}"]
        
        for i, exchange in enumerate(self.conversation_history):
            role = "Patient" if exchange['role'] == 'patient' else "Clinician"
            lines.append(f"{role}: {exchange['content']}")
        
        return "\n".join(lines)
    
    def _fallback_opening(self) -> str:
        """
        Fallback opening when RAG has no results
        
        Returns:
            Generic but professional opening
        """
        return f"I understand you're experiencing {self.chief_complaint}. Can you tell me when this started and how severe it is?"


# === Helper Functions ===

def is_clinician_trigger(prompt: str) -> bool:
    """
    Determine if a prompt should trigger Clinician mode
    
    Clinician triggers:
    - Chief complaints ("I have chest pain")
    - Symptom descriptions
    - But NOT simple questions ("What is chest pain?")
    
    Args:
        prompt: Normalized prompt
        
    Returns:
        True if should use Clinician mode
    """
    prompt_lower = prompt.lower()
    
    # Knowledge queries go to THINKER mode, not CLINICIAN
    knowledge_indicators = ["what is", "who is", "tell me about", "explain", "describe"]
    if any(indicator in prompt_lower for indicator in knowledge_indicators):
        return False
    
    # First-person symptom statements → CLINICIAN
    first_person_patterns = [
        r'\bi have\b', r'\bi\'m having\b', r'\bim having\b',
        r'\bi feel\b', r'\bi\'m feeling\b', r'\bim feeling\b',
        r'\bmy .+ (hurt|ache|pain)', r'\bi experience\b',
        r'\bi\'m experiencing\b', r'\bim experiencing\b'
    ]
    
    if any(re.search(pattern, prompt_lower) for pattern in first_person_patterns):
        return True
    
    # Symptom keywords without question context → CLINICIAN
    symptom_keywords = ["pain", "ache", "hurt", "dizzy", "nausea", "vomiting", 
                       "fever", "cough", "bleeding", "swelling"]
    has_symptom = any(keyword in prompt_lower for keyword in symptom_keywords)
    
    # Check if it's a question about the symptom (goes to THINKER instead)
    is_question = any(q in prompt_lower for q in ["what is", "why do", "how does", "when should"])
    
    return has_symptom and not is_question


def create_clinician_session(session_id: str, chief_complaint: str, llm_chat_fn: Callable) -> ClinicianSession:
    """
    Create a new clinician diagnostic session
    
    Args:
        session_id: Unique session identifier
        chief_complaint: Patient's initial complaint
        llm_chat_fn: Direct LLM chat function
        
    Returns:
        New ClinicianSession instance
    """
    return ClinicianSession(session_id, chief_complaint, llm_chat_fn)

