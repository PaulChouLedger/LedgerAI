#!/usr/bin/env python3
"""
Hybrid Medical Assistant - Natural, Context-Aware Medical Conversations

This is a complete redesign from scratch, focusing on:
1. Natural, human-like conversations
2. Hybrid LLM/Rules/ML approach for ALL interactions
3. Context-aware question generation using guidelines + FAISS
4. Smart anatomical understanding (e.g., "right side" → "right upper quadrant")
5. Dynamic, fluid conversation flow (not rigid)

Architecture:
- Conversation Manager: Tracks context and flow
- Hybrid Extractor: LLM + Rules + ML for understanding patient responses
- Question Generator: Context-aware, guideline-driven questions
- FAISS Assistant: Semantic matching to guidelines for context
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

# Optional ML dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ConversationContext:
    """Manages conversation state and context"""
    
    def __init__(self):
        self.messages = []  # Full conversation history
        self.chief_complaint = None
        self.demographics = {}
        self.extracted_info = {}  # OLDCARTS and other clinical info
        self.active_guidelines = []  # Currently relevant guidelines
        self.asked_questions = []  # Track what we've asked
        self.conversation_phase = "greeting"  # greeting, chief_complaint, assessment, followup
        self.context_hints = []  # Context clues for next question
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation"""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
    
    def get_recent_context(self, n: int = 5) -> List[Dict]:
        """Get last N messages for context"""
        return self.messages[-n:]
    
    def get_conversation_summary(self) -> str:
        """Get a natural language summary of the conversation"""
        summary_parts = []
        
        if self.chief_complaint:
            summary_parts.append(f"Chief complaint: {self.chief_complaint}")
        
        if self.demographics:
            demo_str = ", ".join([f"{k}: {v}" for k, v in self.demographics.items()])
            summary_parts.append(f"Demographics: {demo_str}")
        
        if self.extracted_info:
            info_str = ", ".join([f"{k}: {v}" for k, v in self.extracted_info.items()])
            summary_parts.append(f"Clinical info: {info_str}")
        
        return ". ".join(summary_parts) if summary_parts else "New conversation"


class HybridExtractor:
    """
    Hybrid extractor using LLM + Rules + ML for understanding patient responses.
    
    Handles:
    - Anatomical locations (smart mapping: "right side" → "right upper quadrant")
    - Clinical information (OLDCARTS elements)
    - Demographics
    - Intent detection
    """
    
    def __init__(self, llm_chat_fn=None, embedding_model=None, medical_rules_path=None):
        self.llm_chat_fn = llm_chat_fn
        self.embedding_model = embedding_model
        
        # Load medical rules
        if medical_rules_path is None:
            medical_rules_path = os.path.join(
                os.path.dirname(__file__),
                "config", "medical_rules.json"
            )
        self.medical_rules = self._load_medical_rules(medical_rules_path)
    
    def _load_medical_rules(self, path: str) -> Dict:
        """Load medical_rules.json"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[HybridExtractor] ⚠️ Failed to load medical_rules.json: {e}")
            return {}
    
    def extract_anatomical_location(self, text: str, context: ConversationContext) -> Dict[str, str]:
        """
        Smart anatomical extraction with context awareness.
        
        Examples:
        - "pain on my right side" → {'horizontal': 'right', 'quadrant': 'right_upper'}
        - "lower right" → {'horizontal': 'right', 'vertical': 'lower', 'quadrant': 'right_lower'}
        - "around my belly button" → {'vague': True, 'region': 'periumbilical'}
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        components = {}
        
        # Rule-based extraction first (fast)
        anatomical = self.medical_rules.get('anatomical_components', {})
        
        # Extract quadrant patterns
        quadrant_patterns = anatomical.get('quadrant_patterns', {})
        for quadrant_key, patterns in quadrant_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                components['quadrant'] = quadrant_key
                parts = quadrant_key.split('_')
                if len(parts) >= 2:
                    components['horizontal'] = parts[0]
                    components['vertical'] = parts[1]
                break
        
        # Extract horizontal (left/right)
        if 'horizontal' not in components:
            horizontal = anatomical.get('directional_keywords', {}).get('horizontal', {})
            for direction, keywords in horizontal.items():
                if any(keyword in text_lower for keyword in keywords):
                    components['horizontal'] = direction
                    break
        
        # Extract vertical (upper/lower)
        if 'vertical' not in components:
            vertical = anatomical.get('directional_keywords', {}).get('vertical', {})
            for direction, keywords in vertical.items():
                if any(keyword in text_lower for keyword in keywords):
                    components['vertical'] = direction
                    break
        
        # Smart inference: Only infer quadrant if we have enough information
        # "right side" alone is ambiguous - could be upper or lower quadrant
        # Only infer if we have vertical information or LLM provides context
        if 'horizontal' in components and 'quadrant' not in components:
            # Use LLM for smart inference if available (LLM can use context to determine)
            if self.llm_chat_fn:
                inferred = self._infer_quadrant_with_llm(text, components, context)
                if inferred and 'quadrant' in inferred:
                    components.update(inferred)
            
            # Only infer quadrant if we have vertical information
            # "right side" alone → just horizontal="right" (no quadrant assumption)
            # "upper right" or "right upper" → quadrant="right_upper"
            if 'quadrant' not in components and 'vertical' in components:
                # We have both horizontal and vertical, can infer quadrant
                horizontal = components.get('horizontal')
                vertical = components.get('vertical')
                if horizontal and vertical:
                    components['quadrant'] = f"{horizontal}_{vertical}"
        
        # Extract special regions
        region_keywords = anatomical.get('region_keywords', {})
        for region, keywords in region_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                components['region'] = region
                break
        
        # Check vague/bilateral
        vague_keywords = anatomical.get('directional_keywords', {}).get('vague', {}).get('vague', [])
        if any(keyword in text_lower for keyword in vague_keywords):
            components['vague'] = True
        
        bilateral_keywords = anatomical.get('directional_keywords', {}).get('bilateral', {}).get('bilateral', [])
        if any(keyword in text_lower for keyword in bilateral_keywords):
            components['bilateral'] = True
        
        return components
    
    def _infer_quadrant_with_llm(self, text: str, components: Dict, context: ConversationContext) -> Optional[Dict]:
        """Use LLM to infer missing anatomical details"""
        if not self.llm_chat_fn:
            return None
        
        try:
            system_msg = """You are a medical assistant. Infer anatomical location details from patient descriptions.

Given the patient's description and partial components, infer the most likely anatomical location.
IMPORTANT: Do NOT assume a quadrant if only horizontal direction is provided (e.g., "right side" could be upper OR lower).
Only infer quadrant if you have enough context (e.g., "upper right", "near ribs", "lower abdomen").

Return ONLY a JSON object with inferred components, or empty object if uncertain.

Examples:
- "pain on my right side" + {horizontal: "right"} → {} (uncertain - could be upper or lower)
- "upper right" + {horizontal: "right"} → {quadrant: "right_upper", vertical: "upper"}
- "lower right" + {horizontal: "right", vertical: "lower"} → {quadrant: "right_lower"}
- "right side near my ribs" + {horizontal: "right"} → {quadrant: "right_upper", vertical: "upper"} (ribs = upper)
- "around my belly button" → {vague: true, region: "periumbilical"}"""
            
            user_msg = f"""Patient said: "{text}"
Current components: {json.dumps(components)}
Context: {context.get_conversation_summary()}

Infer missing anatomical details (quadrant, vertical, etc.) as JSON:"""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse JSON
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
            
            inferred = json.loads(response_clean)
            return inferred if isinstance(inferred, dict) else {}
            
        except Exception as e:
            print(f"[HybridExtractor] ⚠️ LLM inference failed: {e}")
            return None
    
    def extract_clinical_info(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Extract clinical information from patient response.
        
        Uses hybrid approach:
        1. Rules for structured data (dates, numbers)
        2. LLM for natural language understanding
        3. Context awareness for disambiguation
        """
        extracted = {}
        
        # Use LLM for natural language understanding
        if self.llm_chat_fn:
            extracted = self._extract_with_llm(text, context)
        
        # Rule-based extraction for structured data
        extracted.update(self._extract_structured_data(text))
        
        return extracted
    
    def _extract_with_llm(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract clinical info using LLM"""
        try:
            system_msg = """You are a medical assistant extracting clinical information from patient responses.

Extract relevant clinical information and return as JSON with these possible keys:
- onset: when symptoms started (e.g., "2 days ago", "yesterday")
- duration: how long symptoms last (e.g., "few minutes", "constant")
- character: description of symptom (e.g., "sharp", "dull", "burning")
- severity: severity level (1-10 or description)
- timing: constant, intermittent, etc.
- associated: other symptoms mentioned
- aggravating: what makes it worse
- relieving: what makes it better
- progression: getting worse, better, same

Return ONLY valid JSON, use null for missing info:"""
            
            user_msg = f"""Patient said: "{text}"
Conversation context: {context.get_conversation_summary()}

Extract clinical information as JSON:"""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse JSON
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
            
            extracted = json.loads(response_clean)
            return extracted if isinstance(extracted, dict) else {}
            
        except Exception as e:
            print(f"[HybridExtractor] ⚠️ LLM extraction failed: {e}")
            return {}
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data using rules (dates, numbers, etc.)"""
        extracted = {}
        
        # Extract numbers (for severity, duration, etc.)
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            # Could be severity, age, duration, etc.
            extracted['numeric_values'] = [int(n) for n in numbers]
        
        # Extract time references
        time_patterns = {
            'days_ago': r'(\d+)\s*days?\s*ago',
            'hours_ago': r'(\d+)\s*hours?\s*ago',
            'minutes_ago': r'(\d+)\s*minutes?\s*ago',
            'weeks_ago': r'(\d+)\s*weeks?\s*ago'
        }
        
        for key, pattern in time_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                extracted[key] = int(match.group(1))
        
        return extracted


class FAISSGuidelineAssistant:
    """
    Uses FAISS to match conversation context to relevant guideline information.
    Helps generate context-aware questions.
    """
    
    def __init__(self, medical_rule_engine=None):
        self.medical_rule_engine = medical_rule_engine
    
    def find_relevant_guideline_terms(self, 
                                      patient_text: str, 
                                      element: str,
                                      active_guidelines: List[Dict],
                                      threshold: float = 0.65) -> List[Tuple[str, float]]:
        """
        Find relevant terms from guidelines using FAISS semantic matching.
        
        Returns:
            List of (term, similarity_score) tuples
        """
        if not self.medical_rule_engine:
            return []
        
        try:
            # Use medical_rule_engine's FAISS matching
            matches = self.medical_rule_engine.find_matching_terms_faiss(
                patient_text,
                element,
                threshold=threshold,
                return_scores=True,
                active_condition_names={g.get('name') for g in active_guidelines}
            )
            
            return matches if isinstance(matches, list) else []
            
        except Exception as e:
            print(f"[FAISSAssistant] ⚠️ FAISS matching failed: {e}")
            return []
    
    def get_missing_information_hints(self, 
                                     context: ConversationContext,
                                     active_guidelines: List[Dict]) -> List[str]:
        """
        Analyze guidelines to determine what information is missing.
        Returns hints for what to ask next.
        """
        hints = []
        
        # Check what OLDCARTS elements are missing
        extracted = context.extracted_info
        oldcarts_elements = ['onset', 'location', 'duration', 'character', 
                            'aggravating', 'relieving', 'timing', 'severity', 'associated']
        
        missing = [elem for elem in oldcarts_elements if elem not in extracted]
        
        # For each missing element, check if guidelines have relevant terms
        for element in missing:
            # Collect terms from active guidelines
            terms = []
            for guideline in active_guidelines:
                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                element_data = structured.get(element, {})
                includes = element_data.get('includes', [])
                for include in includes:
                    patient_friendly = include.get('patient_friendly', '')
                    if patient_friendly:
                        terms.append(patient_friendly)
            
            if terms:
                hints.append({
                    'element': element,
                    'available_terms': terms[:5],  # Top 5 terms
                    'priority': self._calculate_priority(element, context)
                })
        
        # Sort by priority
        hints.sort(key=lambda x: x['priority'], reverse=True)
        
        return hints
    
    def _calculate_priority(self, element: str, context: ConversationContext) -> float:
        """Calculate priority for asking about an element"""
        priority_map = {
            'location': 1.0,
            'character': 0.9,
            'onset': 0.8,
            'severity': 0.7,
            'duration': 0.6,
            'timing': 0.5,
            'aggravating': 0.4,
            'relieving': 0.4,
            'associated': 0.3
        }
        
        base_priority = priority_map.get(element, 0.5)
        
        # Boost if chief complaint suggests it's important
        if context.chief_complaint:
            complaint_lower = context.chief_complaint.lower()
            if 'blood' in complaint_lower and element == 'character':
                base_priority += 0.2
            if 'pain' in complaint_lower and element == 'location':
                base_priority += 0.2
        
        return base_priority


class QuestionGenerator:
    """
    Generates natural, context-aware questions using guidelines and FAISS.
    """
    
    def __init__(self, llm_chat_fn=None, faiss_assistant=None):
        self.llm_chat_fn = llm_chat_fn
        self.faiss_assistant = faiss_assistant
    
    def generate_next_question(self, 
                               context: ConversationContext,
                               active_guidelines: List[Dict]) -> str:
        """
        Generate the next natural question based on context and guidelines.
        
        Examples:
        - Chief complaint "bloody diarrhea" → "Can you tell me about the color of the stool?"
        - "pain on right side" → "Is the pain sharp or dull?"
        - After location → "How long have you had this pain?"
        """
        # Get hints about what's missing
        hints = self.faiss_assistant.get_missing_information_hints(context, active_guidelines) if self.faiss_assistant else []
        
        # Use LLM to generate natural question
        if self.llm_chat_fn:
            question = self._generate_with_llm(context, hints, active_guidelines)
            if question:
                return question
        
        # Fallback to rule-based question
        return self._generate_rule_based_question(context, hints)
    
    def _generate_with_llm(self, 
                          context: ConversationContext,
                          hints: List[Dict],
                          active_guidelines: List[Dict]) -> Optional[str]:
        """Generate question using LLM for natural language"""
        try:
            # Build context about what we know and what we need
            known_info = context.get_conversation_summary()
            missing_elements = [h['element'] for h in hints[:3]]  # Top 3 missing
            
            # Get relevant terms from guidelines
            guideline_context = []
            for hint in hints[:2]:  # Top 2 hints
                element = hint['element']
                terms = hint.get('available_terms', [])
                if terms:
                    guideline_context.append(f"For {element}, relevant options: {', '.join(terms[:3])}")
            
            system_msg = """You are a medical assistant having a natural conversation with a patient.

Generate a SINGLE, natural, conversational question that:
1. Feels human and empathetic
2. Asks about missing clinical information
3. Uses the provided guideline terms naturally (don't list them all, just incorporate naturally)
4. Flows naturally from the conversation context
5. Is specific and helpful

DO NOT:
- Ask multiple questions at once
- Use medical jargon
- Sound robotic or formal
- List options like "Is it A, B, or C?"

Examples:
- Good: "Can you tell me about the color of the stool? Is it bright red or darker?"
- Good: "How long have you been experiencing this pain?"
- Bad: "What is the character of the pain? Is it sharp, dull, or burning?"
- Bad: "Please describe the onset, duration, and character of your symptoms."

Return ONLY the question, no explanations:"""
            
            user_msg = f"""Conversation so far:
{known_info}

What we still need to know: {', '.join(missing_elements)}

{chr(10).join(guideline_context) if guideline_context else 'No specific guideline terms available.'}

Generate a natural, conversational question:"""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=100,
                temperature=0.7  # More creative for natural questions
            )
            
            question = response.strip()
            # Clean up
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            
            return question if len(question) > 10 else None
            
        except Exception as e:
            print(f"[QuestionGenerator] ⚠️ LLM question generation failed: {e}")
            return None
    
    def _generate_rule_based_question(self, context: ConversationContext, hints: List[Dict]) -> str:
        """Fallback rule-based question generation"""
        if not hints:
            return "Can you tell me more about your symptoms?"
        
        top_hint = hints[0]
        element = top_hint['element']
        terms = top_hint.get('available_terms', [])
        
        # Simple question templates
        templates = {
            'location': "Where exactly is the {symptom} located?",
            'character': "Can you describe what the {symptom} feels like?",
            'onset': "When did the {symptom} start?",
            'duration': "How long does the {symptom} last?",
            'severity': "On a scale of 1 to 10, how severe is the {symptom}?",
            'timing': "Is the {symptom} constant or does it come and go?",
            'associated': "Are you experiencing any other symptoms?",
            'aggravating': "What makes the {symptom} worse?",
            'relieving': "What makes the {symptom} better?"
        }
        
        symptom = context.chief_complaint or "symptom"
        template = templates.get(element, "Can you tell me more about the {symptom}?")
        
        question = template.format(symptom=symptom)
        
        # Add terms if available
        if terms and len(terms) <= 3:
            question += f" For example, is it {', '.join(terms)}?"
        
        return question


class HybridMedicalAssistant:
    """
    Main medical assistant class - orchestrates all components for natural conversations.
    """
    
    def __init__(self, 
                 llm_chat_fn=None,
                 embedding_model=None,
                 medical_rule_engine=None,
                 guidelines_dir=None):
        self.llm_chat_fn = llm_chat_fn
        self.embedding_model = embedding_model
        self.medical_rule_engine = medical_rule_engine
        
        # Initialize components
        self.extractor = HybridExtractor(llm_chat_fn, embedding_model)
        self.faiss_assistant = FAISSGuidelineAssistant(medical_rule_engine)
        self.question_generator = QuestionGenerator(llm_chat_fn, self.faiss_assistant)
        
        # Load guidelines
        if guidelines_dir is None:
            guidelines_dir = os.path.join(
                os.path.dirname(__file__),
                "medical", "guidelines"
            )
        self.guidelines = self._load_guidelines(guidelines_dir)
        
        # Active sessions
        self.sessions = {}  # session_id -> ConversationContext
    
    def _load_guidelines(self, guidelines_dir: str) -> Dict[str, List[Dict]]:
        """Load all guidelines from directory"""
        guidelines = {}
        
        try:
            for category_dir in Path(guidelines_dir).iterdir():
                if category_dir.is_dir():
                    category_guidelines = []
                    for guideline_file in category_dir.glob("*.json"):
                        try:
                            with open(guideline_file, 'r') as f:
                                guideline = json.load(f)
                                category_guidelines.append(guideline)
                        except Exception as e:
                            print(f"[HybridAssistant] ⚠️ Failed to load {guideline_file}: {e}")
                    
                    if category_guidelines:
                        guidelines[category_dir.name] = category_guidelines
            
            print(f"[HybridAssistant] ✅ Loaded {sum(len(g) for g in guidelines.values())} guidelines from {len(guidelines)} categories")
        except Exception as e:
            print(f"[HybridAssistant] ⚠️ Failed to load guidelines: {e}")
        
        return guidelines
    
    def process_message(self, 
                       session_id: str,
                       user_message: str) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
        Returns:
            {
                'response': str,
                'status': str,  # 'greeting', 'assessment', 'complete', etc.
                'debug': dict
            }
        """
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext()
        
        context = self.sessions[session_id]
        context.add_message('user', user_message)
        
        # Determine conversation phase
        phase = self._determine_phase(context, user_message)
        context.conversation_phase = phase
        
        # Process based on phase
        if phase == "greeting":
            response = self._handle_greeting(context, user_message)
        elif phase == "chief_complaint":
            response = self._handle_chief_complaint(context, user_message)
        elif phase == "assessment":
            response = self._handle_assessment(context, user_message)
        else:
            response = self._handle_followup(context, user_message)
        
        context.add_message('assistant', response['response'], response.get('metadata', {}))
        
        return response
    
    def _determine_phase(self, context: ConversationContext, message: str) -> str:
        """Determine current conversation phase"""
        if not context.chief_complaint:
            if context.conversation_phase == "greeting":
                return "chief_complaint"
            return "greeting"
        
        if context.conversation_phase in ["greeting", "chief_complaint"]:
            return "assessment"
        
        return "assessment"
    
    def _handle_greeting(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle greeting phase"""
        if self.llm_chat_fn:
            system_msg = """You are Aura, a friendly and helpful medical AI assistant. 
Respond to greetings warmly and briefly. Mention that you're here to help with medical questions or symptom assessment.
Keep it conversational and inviting. Respond in 1-2 short sentences."""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": message}],
                max_tokens=100,
                temperature=0.7
            )
            
            greeting = response.strip()
        else:
            greeting = "Hello! I'm Aura, your medical AI assistant. How can I help you today?"
        
        return {
            'response': greeting,
            'status': 'greeting',
            'metadata': {}
        }
    
    def _handle_chief_complaint(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle chief complaint extraction"""
        # Extract chief complaint using LLM
        if self.llm_chat_fn:
            system_msg = """You are a medical assistant. Extract the chief complaint (main symptom or concern) from the patient's message.
Return ONLY the chief complaint in simple terms, or 'none' if unclear."""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": message}],
                max_tokens=50,
                temperature=0.1
            )
            
            chief_complaint = response.strip().lower()
            if chief_complaint == 'none' or not chief_complaint:
                chief_complaint = message.lower()
        else:
            chief_complaint = message.lower()
        
        context.chief_complaint = chief_complaint
        
        # Find relevant guidelines
        active_guidelines = self._find_relevant_guidelines(chief_complaint)
        context.active_guidelines = active_guidelines
        
        # Generate context-aware first question based on chief complaint
        # Example: "bloody diarrhea" → ask about color and episodes naturally
        first_question = self._generate_context_aware_first_question(chief_complaint, active_guidelines)
        
        # Generate empathetic response
        if self.llm_chat_fn:
            system_msg = """You are a medical assistant. The patient just described their chief complaint.
Respond with a brief, empathetic acknowledgment (1 sentence). Keep it conversational and human."""
            
            user_msg = f"""Patient's chief complaint: {chief_complaint}

Generate a brief empathetic acknowledgment:"""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=50,
                temperature=0.7
            )
            
            empathetic_response = response.strip()
        else:
            empathetic_response = f"I understand you're experiencing {chief_complaint}."
        
        # Combine empathetic response with first question
        assistant_response = f"{empathetic_response} {first_question}"
        
        return {
            'response': assistant_response,
            'status': 'assessment',
            'metadata': {
                'chief_complaint': chief_complaint,
                'active_guidelines': [g.get('name', g.get('condition', 'Unknown')) for g in active_guidelines]
            }
        }
    
    def _generate_context_aware_first_question(self, chief_complaint: str, active_guidelines: List[Dict]) -> str:
        """
        Generate context-aware first question based on chief complaint.
        
        Examples:
        - "bloody diarrhea" → "Can you tell me about the color of the stool? How many episodes have you had?"
        - "chest pain" → "Where exactly is the pain located?"
        - "abdominal pain" → "Can you describe what the pain feels like?"
        """
        complaint_lower = chief_complaint.lower()
        
        # Check guidelines for relevant character/associated terms
        character_terms = []
        associated_terms = []
        timing_terms = []
        
        for guideline in active_guidelines[:3]:  # Check top 3 guidelines
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            
            # Character terms
            char_data = structured.get('character', {}).get('includes', [])
            for item in char_data[:3]:  # Top 3
                pf = item.get('patient_friendly', '')
                if pf:
                    character_terms.append(pf)
            
            # Associated terms
            assoc_data = structured.get('associated', {}).get('includes', [])
            for item in assoc_data[:3]:
                pf = item.get('patient_friendly', '')
                if pf:
                    associated_terms.append(pf)
            
            # Timing terms
            timing_data = structured.get('timing', {}).get('includes', [])
            for item in timing_data[:2]:
                pf = item.get('patient_friendly', '')
                if pf:
                    timing_terms.append(pf)
        
        # Remove duplicates while preserving order
        character_terms = list(dict.fromkeys(character_terms))[:3]
        associated_terms = list(dict.fromkeys(associated_terms))[:3]
        timing_terms = list(dict.fromkeys(timing_terms))[:2]
        
        # Generate question based on chief complaint type
        if self.llm_chat_fn:
            system_msg = """You are a medical assistant. Generate a SINGLE, natural, conversational question based on the chief complaint.

The question should:
1. Be specific to the chief complaint
2. Feel natural and human (not robotic)
3. Ask about the most relevant clinical information first
4. Incorporate guideline terms naturally if provided (don't list them all)

Examples:
- Chief complaint: "bloody diarrhea" → "Can you tell me about the color of the stool? How many episodes have you had?"
- Chief complaint: "chest pain" → "Where exactly is the pain located?"
- Chief complaint: "abdominal pain" → "Can you describe what the pain feels like?"

Return ONLY the question, no explanations:"""
            
            guideline_context = ""
            if character_terms:
                guideline_context += f"Relevant character descriptions: {', '.join(character_terms[:2])}. "
            if timing_terms:
                guideline_context += f"Relevant timing: {', '.join(timing_terms[:2])}. "
            
            user_msg = f"""Chief complaint: "{chief_complaint}"
{guideline_context if guideline_context else ''}

Generate a natural first question:"""
            
            response = self.llm_chat_fn(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=100,
                temperature=0.7
            )
            
            question = response.strip()
            # Clean up
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            
            return question if len(question) > 10 else "Can you tell me more about it?"
        else:
            # Fallback rule-based questions
            if 'blood' in complaint_lower or 'bleed' in complaint_lower:
                if character_terms:
                    return f"Can you tell me about the color? For example, is it {character_terms[0]}?"
                return "Can you tell me about the color and how many episodes you've had?"
            elif 'pain' in complaint_lower:
                if 'chest' in complaint_lower:
                    return "Where exactly is the pain located?"
                return "Can you describe what the pain feels like?"
            else:
                return "Can you tell me more about it?"
    
    def _handle_assessment(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle assessment phase - extract info and ask next question"""
        # Extract clinical information
        clinical_info = self.extractor.extract_clinical_info(message, context)
        context.extracted_info.update(clinical_info)
        
        # Extract anatomical location if mentioned
        location_components = self.extractor.extract_anatomical_location(message, context)
        if location_components:
            context.extracted_info['location'] = location_components
        
        # Generate next question
        next_question = self.question_generator.generate_next_question(
            context,
            context.active_guidelines
        )
        
        return {
            'response': next_question,
            'status': 'assessment',
            'metadata': {
                'extracted_info': context.extracted_info,
                'active_guidelines': [g.get('name') for g in context.active_guidelines]
            }
        }
    
    def _handle_followup(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle follow-up questions"""
        return self._handle_assessment(context, message)
    
    def _find_relevant_guidelines(self, chief_complaint: str) -> List[Dict]:
        """Find relevant guidelines based on chief complaint"""
        relevant = []
        
        # Simple keyword matching for now
        complaint_lower = chief_complaint.lower()
        
        for category, guidelines in self.guidelines.items():
            for guideline in guidelines:
                # Check chief complaint triggers
                triggers = guideline.get('data', {}).get('chief_complaint_triggers', [])
                for trigger in triggers:
                    if trigger.lower() in complaint_lower or complaint_lower in trigger.lower():
                        relevant.append(guideline)
                        break
        
        return relevant[:10]  # Limit to top 10

