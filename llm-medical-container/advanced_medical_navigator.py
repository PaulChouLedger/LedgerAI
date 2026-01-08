#!/usr/bin/env python3
"""
Simplified Advanced Medical Navigator - LLM-Driven
===================================================

Conversation flow:
    1. Capture chief complaint → LLM identifies relevant conditions
    2. LLM empathetic acknowledgement + chronicity question
    3. Collect demographics: age, biological sex
    4. OLD CARTS assessment - LLM generates questions naturally
    5. Condition rankings updated after each answer
    6. Diagnosis ready once OLD CARTS complete

This file relies entirely on the fine-tuned LLM for all logic.
No hardcoded rules, weights, or thresholds - LLM handles everything.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Any


class AdvancedMedicalNavigator:
    """LLM-driven navigator - simplified to rely on fine-tuned model."""

    # ----------- Configuration -------------------------------------------------

    PRE_HPI_ORDER = ["chief_complaint", "chronicity", "age", "sex"]

    HPI_ELEMENTS = [
        "onset",
        "location",
        "duration",
        "character",
        "aggravating",
        "relieving",
        "radiation",
        "timing",
        "severity",
        "associated",
    ]

    CATEGORY_TO_SYSTEM = {
        'gastrointestinal': 'GI',
        'cardiovascular': 'CARDIO',
        'respiratory': 'RESPIRATORY',
        'neurological': 'NEURO',
        'musculoskeletal': 'MSK',
        'renal': 'RENAL',
        'genitourinary': 'GU',
        'dermatological': 'DERM',
        'endocrine': 'ENDO',
    }

    # LLM parameter configuration
    LLM_TEMPERATURE_QUESTIONS = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.4'))
    LLM_TEMPERATURE_EMPATHETIC = float(os.environ.get('LLM_TEMPERATURE_EMPATHETIC', '0.4'))
    LLM_TEMPERATURE_SUMMARY = float(os.environ.get('LLM_TEMPERATURE_SUMMARY', '0.25'))
    # Standardized to 200 to match llm-container (can be overridden via LLM_NUM_PREDICT env var)
    LLM_MAX_TOKENS_QUESTIONS = int(os.environ.get('LLM_NUM_PREDICT', '200'))
    LLM_MAX_TOKENS_EMPATHETIC = int(os.environ.get('LLM_MAX_TOKENS_EMPATHETIC', '60'))
    LLM_MAX_TOKENS_CHRONICITY = int(os.environ.get('LLM_MAX_TOKENS_CHRONICITY', '50'))
    LLM_MAX_TOKENS_SUMMARY = int(os.environ.get('LLM_MAX_TOKENS_SUMMARY', '400'))

    # System prompts - simplified for fine-tuned model
    QUESTION_SYSTEM_PROMPT = (
        "You are a professional medical assistant conducting a medical history. "
        "You must understand the conversation context and avoid asking redundant questions.\n\n"
        "CRITICAL RULES:\n"
        "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
        "- If the patient's response is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
        "- Instead, politely ask the patient to clarify or repeat their answer.\n"
        "- Never invent facts, symptoms, diagnoses, or medical details.\n\n"
        "IMPORTANT RULES:\n"
        "- If the patient already provided information, do NOT ask about it again\n"
        "- Pay attention to what has already been discussed in the conversation\n"
        "- Ask about NEW information only, not information already provided\n\n"
        "Follow this order:\n"
        "1. Show empathy and acknowledge their concern\n"
        "2. Ask if this is new or an ongoing problem\n"
        "3. Ask their age\n"
        "4. Ask their biological sex\n"
        "5. Then ask about the symptom - one question at a time, waiting for each answer\n\n"
        "Ask about: when it started, where it is, how long it's been present, what it feels like, "
        "what makes it worse, what makes it better, if it spreads, if it's constant or comes and goes, and how severe it is.\n\n"
        "Be natural and conversational. Ask only one question at a time. Do not list multiple questions. "
        "Do not mention frameworks or include instructions in your responses. "
        "CRITICAL: Do NOT include any of the following in your response:\n"
        "- Internal reasoning (e.g., 'This is clinical reasoning to identify...')\n"
        "- Clinical notes (e.g., 'CLINICAL NOTE:', 'Note:', etc.)\n"
        "- Explanations of what you're doing\n"
        "- Acknowledgments or filler text\n\n"
        "ONLY output the question itself. Nothing else. No reasoning, no notes, no explanations."
    )

    EMPATHETIC_SYSTEM_PROMPT = (
        "You are a professional medical assistant. Show empathy and acknowledge the patient's concern. "
        "Be natural and conversational. Do not ask questions yet.\n\n"
        "CRITICAL: Only provide logical, factual responses. If the patient's statement is unclear or doesn't make sense, "
        "politely ask them to clarify rather than forcing a response."
    )

    CHRONICITY_SYSTEM_PROMPT = (
        "You are a professional medical assistant. Ask if this is new or an ongoing problem. "
        "Be natural and conversational. Ask only one question. "
        "Do not include internal reasoning, acknowledgments, or explanations. Only ask the question.\n\n"
        "CRITICAL: Only provide logical, factual responses. If the patient's response is unclear or doesn't make sense, "
        "politely ask them to clarify rather than forcing a response."
    )

    SUMMARY_SYSTEM_PROMPT = (
        "You are a clinical assistant creating a concise medical history summary. "
        "Format as clear bullet points with proper medical terminology.\n\n"
        "REQUIRED FORMAT:\n"
        "• Chief complaint: [complaint]\n"
        "• Demographics: Age [X], [Sex]\n"
        "• History of present illness (OLD CARTS): [key findings]\n"
        "• Associated symptoms: [if any]\n"
        "• Top differential diagnoses: [condition 1], [condition 2], [condition 3]\n\n"
        "CRITICAL FORMATTING RULES:\n"
        "- Use bullet points with '•' symbol\n"
        "- Start each bullet with a category label followed by a colon\n"
        "- For OLD CARTS section, list key findings (onset, location, character, etc.) - NOT 'Chief complaint is...'\n"
        "- Do NOT repeat 'OLD CARTS:' before each finding\n"
        "- Use 'OLD CARTS' only when referring to the framework itself\n"
        "- Group OLD CARTS findings under one bullet point\n"
        "- Include all provided information, but be concise\n"
        "- Do not include clarification questions or confused responses\n"
        "- If information is missing, omit that section rather than saying 'Unknown'\n\n"
        "EXAMPLE FORMAT:\n"
        "• Chief complaint: Chest pain\n"
        "• Demographics: Age 37, Male\n"
        "• History of present illness: Onset 2 days ago, central chest location, pressure character, worsened with exertion, relieved with rest, radiates to left arm and neck\n"
        "• Associated symptoms: Shortness of breath, nausea\n"
        "• Top differential diagnoses: Acute Myocardial Infarction, Unstable Angina, Pulmonary Embolism\n\n"
        "CRITICAL: Only use factual information provided. Do not invent details. Keep format consistent."
    )

    GREETING_RESPONSES = (
        "Hi there! I'm here to help. What symptoms are you experiencing today?",
        "Hello! Let me know what brings you in today so I can assist.",
    )

    # Condition ranking configuration
    CLEAR_LEAD_MARGIN = 0.08  # Margin for clear leader in rankings

    # ----------- Session container -------------------------------------------

    @dataclass
    class MedicalSession:
        session_id: str
        created_at: datetime = field(default_factory=datetime.utcnow)
        stage: str = "awaiting_chief_complaint"
        messages: List[Dict[str, str]] = field(default_factory=list)
        pending: Optional[Dict[str, str]] = None
        context: Dict[str, Dict] = field(default_factory=lambda: {
            'pre_hpi': {},
            'hpi': {},
            'pmh': {},
            'matched_categories': [],
        })
        condition_scores: Dict[str, float] = field(default_factory=dict)
        condition_rankings: List[Tuple[str, float]] = field(default_factory=list)
        active_conditions: List[Tuple[str, float]] = field(default_factory=list)
        reserve_conditions: List[Tuple[str, float]] = field(default_factory=list)
        previous_active: set = field(default_factory=set)
        oldcarts_remaining: List[str] = field(default_factory=list)
        completed: bool = False
        last_field: Optional[str] = None

    # ----------- Lifecycle ----------------------------------------------------

    def __init__(self, llm_chat_fn, embedding_model=None):
        """
        Initialize Advanced Medical Navigator - LLM-only approach.
        
        Args:
            llm_chat_fn: LLM chat function for all decisions
            embedding_model: Optional (not used in simplified version)
        """
        self.llm_chat_fn = llm_chat_fn
        self.sessions: Dict[str, AdvancedMedicalNavigator.MedicalSession] = {}
        self._captured_debug_output: List[str] = []

    # ----------- Public API ---------------------------------------------------

    def process_message(self, session_id: str, user_message: str, stream: bool = False) -> Dict[str, any]:
        """
        Process user message and generate response.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            stream: If True, returns a generator that yields tokens for the final response.
                    If False, returns a dict with the complete response (blocking).
        
        Returns:
            If stream=False: Dict with response
            If stream=True: Generator that yields (response_dict, token_stream) where
                           token_stream yields tokens as they're generated
        """
        self._captured_debug_output = []
        session = self._get_or_create_session(session_id)
        session.messages.append({"role": "user", "content": user_message})
        if len(session.messages) > 50:
            session.messages = session.messages[-50:]

        if session.stage == "awaiting_chief_complaint":
            response = self._handle_initial_complaint(session, user_message)
            if stream:
                # For streaming, yield the response dict first, then stream tokens
                response_text = response.get('response', '') or response.get('message', '') or response.get('question', '')
                if response_text:
                    def token_stream():
                        # Stream the response word-by-word for immediate TTS start
                        words = response_text.split()
                        for word in words:
                            yield word + " "
                        yield ""  # End marker
                    return response, token_stream()
                else:
                    return response, iter([])
            return response

        if session.pending:
            response = self._store_answer(session, session.pending, user_message)
            if response:
                if stream:
                    response_text = response.get('response', '') or response.get('message', '') or response.get('question', '')
                    if response_text:
                        def token_stream():
                            words = response_text.split()
                            for word in words:
                                yield word + " "
                            yield ""
                        return response, token_stream()
                    else:
                        return response, iter([])
                return response

        if session.completed:
            follow_up = "Thanks for the update. If anything changes, let me know."
            session.messages.append({"role": "assistant", "content": follow_up})
            response = self._wrap_response(session, follow_up, status="complete")
            if stream:
                def token_stream():
                    words = follow_up.split()
                    for word in words:
                        yield word + " "
                    yield ""
                return response, token_stream()
            return response

        next_prompt = self._determine_next_question(session)
        if next_prompt:
            # STREAMING: For the final question, stream it token-by-token
            if stream:
                # Generate question with streaming - returns a generator
                question_stream = self._generate_question_streaming(
                    session, next_prompt['section'], next_prompt['field'], 
                    next_prompt.get('guidance', '')
                )
                # Accumulate full response for session storage
                # Use a list to store accumulated text (can be modified in nested function)
                accumulated = [""]
                def token_stream():
                    # Accumulate tokens as they're yielded
                    for token in question_stream:
                        accumulated[0] += token
                        yield token
                    # After streaming completes, store in session
                    full_question = accumulated[0].strip()
                    if full_question:
                        session.pending = next_prompt
                        session.pending['prompt'] = full_question
                        session.messages.append({"role": "assistant", "content": full_question})
                
                session.pending = next_prompt  # Set pending early
                # Build response dict (will contain accumulated text after streaming)
                response_dict = self._wrap_response(session, "", metadata={
                    'section': session.stage,
                    'field': next_prompt['field'],
                })
                return response_dict, token_stream()  # Return dict and generator
            else:
                # Non-streaming (blocking)
                question_text = self._generate_question(
                    session, next_prompt['section'], next_prompt['field'],
                    next_prompt.get('guidance', '')
                )
                session.pending = next_prompt
                session.pending['prompt'] = question_text
                session.messages.append({"role": "assistant", "content": question_text})
                return self._wrap_response(session, question_text, metadata={
                'section': session.stage,
                'field': next_prompt['field'],
            })

        # Summary generation (blocking, no streaming needed)
        summary = self._generate_summary(session)
        session.completed = True
        session.messages.append({"role": "assistant", "content": summary})
        response = self._wrap_response(session, summary, status="complete", metadata={'summary': True})
        if stream:
            def token_stream():
                words = summary.split()
                for word in words:
                    yield word + " "
                yield ""
            return response, token_stream()
        return response

    # ----------- Stage handlers ----------------------------------------------

    def _handle_initial_complaint(self, session: "MedicalSession", text: str) -> Dict[str, any]:
        # Check if this is a greeting or casual conversation
        if self._is_greeting(text) or not self._is_medical_complaint(text):
            # It's a greeting or casual conversation - respond naturally
            if self.llm_chat_fn:
                greeting_prompt = (
                    f"The user just said: '{text}'\n\n"
                    "This is a greeting or casual conversation, NOT a medical complaint. "
                    "Respond naturally and friendly. Wait for them to mention a medical concern "
                    "before asking any medical questions. Keep it brief and welcoming."
                )
                reply = self.llm_chat_fn(
                    [
                        {"role": "system", "content": self.QUESTION_SYSTEM_PROMPT},
                        {"role": "user", "content": greeting_prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                reply = self._clean_llm_response(reply, fallback=self.GREETING_RESPONSES[0])
            else:
                reply = self.GREETING_RESPONSES[0]
            
            self._capture_debug(f"[Navigator] 🙋 Greeting/casual conversation detected: '{text}'")
            session.messages.append({"role": "assistant", "content": reply})
            return self._wrap_response(session, reply, status="awaiting_chief_complaint")

        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (LLM-ONLY)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{text}'")
        
        # LLM identifies relevant conditions from chief complaint
        categories = self._match_chief_complaint_to_category_llm(text)
        if not categories:
            apology = (
                "I'm not sure I caught that. Could you tell me a bit more about what's bothering you, "
                "like 'I have stomach pain' or 'I'm feeling short of breath'?"
            )
            self._capture_debug(
                f"[Engine] ❌ Unable to match chief complaint '{text}'. Requesting clarification."
            )
            session.stage = "awaiting_chief_complaint"
            return self._wrap_response(session, apology, status="awaiting_chief_complaint")
        
        session.context['matched_categories'] = categories
        primary_category = categories[0]
        if len(categories) == 1:
            self._capture_debug(f"[Engine] 🎯 Category: {primary_category}")
        else:
            self._capture_debug(f"[Engine] 🎯 Categories: {', '.join(categories)}")

        # Initialize condition scores - LLM suggests relevant conditions dynamically
        session.condition_scores = self._initialize_condition_scores_llm(categories, text)
        
        self._capture_debug(f"[Engine] 📋 Initialized {len(session.condition_scores)} conditions at balanced baseline 50.0% - LLM will narrow down based on answers")

        self._apply_rule_outs(session)

        session.stage = "awaiting_chronicity"
        session.context['pre_hpi']['chief_complaint'] = text

        empathetic = self._generate_empathetic_statement(text)
        chronicity_prompt = self._generate_chronicity_question()

        session.pending = {
            'section': 'pre_hpi',
            'field': 'chronicity',
            'prompt': chronicity_prompt,
        }
        session.messages.append({"role": "assistant", "content": empathetic})
        session.messages.append({"role": "assistant", "content": chronicity_prompt})
        combined_message = f"{empathetic} {chronicity_prompt}"
        return self._wrap_response(session, combined_message, metadata={'stage': 'pre_hpi'})

    # ----------- Question selection ------------------------------------------

    def _determine_next_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        if session.stage in {"awaiting_sex", "pre_hpi"} and session.context['pre_hpi'].get('sex'):
            session.stage = "hpi"
            if not session.oldcarts_remaining:
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)

        if session.stage == "awaiting_chronicity":
            if not session.context['pre_hpi'].get('chronicity'):
                return None
            session.stage = "awaiting_age"

        if session.stage == "awaiting_age":
            if not session.context['pre_hpi'].get('age'):
                prompt = self._generate_question(
                    session=session,
                    section='pre_hpi',
                    field='age',
                    guidance="Ask the patient their age using second person (e.g., 'How old are you?' or 'What is your age?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient's age'."
                )
                if not prompt.strip().endswith('?'):
                    prompt = prompt.rstrip('.') + '?'
                # Fallback to correct format if LLM generates wrong format
                prompt_cleaned = prompt.strip()
                if 'patient' in prompt_cleaned.lower() and ('age' in prompt_cleaned.lower() or 'old' in prompt_cleaned.lower()):
                    # LLM used third person, use correct second person format
                    prompt = "How old are you?"
                return {'section': 'pre_hpi', 'field': 'age', 'prompt': prompt}
            session.stage = "awaiting_sex"

        if session.stage == "awaiting_sex":
            if session.context['pre_hpi'].get('sex'):
                session.stage = "hpi"
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
            else:
                prompt = self._generate_question(
                    session=session,
                    section='pre_hpi',
                    field='sex',
                    guidance="Ask the patient their biological sex using second person (e.g., 'What is your biological sex?' or 'Are you male or female?'). Ask only the question, no acknowledgment or reasoning. Do NOT use third person like 'the patient' or 'is the patient male'."
                )
                if not prompt.strip().endswith('?'):
                    prompt = prompt.rstrip('.') + '?'
                # Fallback to correct format if LLM generates wrong format
                prompt_cleaned = prompt.strip()
                if 'patient' in prompt_cleaned.lower() and ('male' in prompt_cleaned.lower() or 'sex' in prompt_cleaned.lower() or 'female' in prompt_cleaned.lower()):
                    # LLM used third person, use correct second person format
                    prompt = "What is your biological sex?"
                return {'section': 'pre_hpi', 'field': 'sex', 'prompt': prompt}

        if session.stage == "hpi":
            return self._next_oldcarts_question(session)

            return None

    def _is_redundant_question(self, session: "MedicalSession", element: str) -> bool:
        """Check if asking about this element would be redundant - LLM handles this."""
        hpi = session.context.get('hpi', {})
        
        # If we already have an answer for this element, it's redundant
        if hpi.get(element) and hpi[element].strip():
            return True
        
        # Basic redundancy checks - LLM should handle most of this
        if element == 'duration' and hpi.get('onset'):
            onset = hpi['onset'].lower()
            if any(word in onset for word in ['day', 'days', 'week', 'weeks', 'hour', 'hours', 'minute', 'minutes']):
                return True
        
        if element == 'frequency' and hpi.get('timing'):
            timing = hpi['timing'].lower()
            if 'constant' in timing or 'continuous' in timing:
                return True
        
        return False

    def _next_oldcarts_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        # Always regenerate oldcarts_remaining to ensure it's in sync with answered fields
        # This prevents re-asking fields that were answered (including after clarifications)
        session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
        
        if not session.oldcarts_remaining:
            session.stage = "pmh"
            return None

        element = None
        # Get answered fields (exclude confused responses)
        answered = {
            key for key, value in session.context['hpi'].items()
            if value and value.strip() and not self._is_confused_response(value)
        }
        
        # Check if we have relevance information from training data
        # This would be set if we're using a fine-tuned model that learned skip patterns
        chief_complaint = session.context['pre_hpi'].get('chief_complaint', '').lower()
        
        while session.oldcarts_remaining:
            candidate = session.oldcarts_remaining.pop(0)
            # Double-check: skip if already answered (safety check)
            if candidate in answered:
                self._capture_debug(f"[HPI] ⏭️ Skipping {candidate} - already answered in hpi context")
                continue
            if self._is_redundant_question(session, candidate):
                self._capture_debug(f"[HPI] ⏭️ Skipping {candidate} - redundant question")
                continue
            
            # Check if this element should be skipped based on chief complaint
            if self._should_skip_oldcarts_element(session, candidate, chief_complaint):
                self._capture_debug(f"[HPI] ⏭️ Skipping {candidate} - not relevant for this complaint")
                continue
            
            element = candidate
            break
        
        if element is None:
            session.stage = "pmh"
            return None
        
        cc_subject = self._normalize_subject_for_questions(session.context['pre_hpi'].get('chief_complaint'))
        session.last_field = element

        # Determine if we should use pronouns (after first HPI question)
        answered_hpi_count = len([k for k in session.context['hpi'].keys() if session.context['hpi'][k] and session.context['hpi'][k].strip()])
        use_pronoun = answered_hpi_count > 0  # Use pronouns after first answer

        if element == 'associated':
            # LLM handles associated symptoms naturally
            guidance = "Ask about any other symptoms the patient might be experiencing. Ask only one question." if use_pronoun else f"Ask about any other symptoms the patient might have along with {cc_subject}. Ask only one question."
            prompt = self._generate_question(
                session=session,
                section='hpi',
                field='associated',
                guidance=guidance,
                use_pronoun=use_pronoun,
                cc_subject=cc_subject
            )
            return {
                'section': 'hpi',
                'field': element,
                'prompt': prompt,
            }

        # Generate question naturally using LLM with pronoun awareness
        guidance = f"Ask about the {element}. Ask only one question, no acknowledgment or reasoning." if use_pronoun else f"Ask about the {element} of {cc_subject}. Ask only one question, no acknowledgment or reasoning."
        prompt = self._generate_question(
            session=session,
            section='hpi',
            field=element,
            guidance=guidance,
            use_pronoun=use_pronoun,
            cc_subject=cc_subject
        )
        return {
            'section': 'hpi',
            'field': element,
            'prompt': prompt,
        }
    
    def _should_skip_oldcarts_element(self, session: "MedicalSession", element: str, chief_complaint: str) -> bool:
        """
        Determine if an OLD CARTS element should be skipped based on chief complaint.
        Uses heuristics based on training data patterns.
        
        The fine-tuned model should learn these patterns from the dataset,
        but we provide heuristics as a fallback.
        """
        chief_lower = chief_complaint.lower()
        
        # Location (L) - skip for systemic conditions
        if element == 'location':
            systemic_keywords = [
                'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
                'diabetes', 'polyuria', 'polydipsia', 'polyphagia', 'fatigue', 'dizziness',
                'depression', 'anxiety', 'insomnia', 'difficulty falling asleep'
            ]
            if any(keyword in chief_lower for keyword in systemic_keywords):
                return True
        
        # Character (C) - skip for non-sensory symptoms
        if element == 'character':
            non_sensory_keywords = [
                'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
                'polyuria', 'polydipsia', 'polyphagia', 'constipation', 'urinary incontinence',
                'insomnia', 'difficulty falling asleep', 'difficulty maintaining sleep'
            ]
            if any(keyword in chief_lower for keyword in non_sensory_keywords):
                return True
        
        # Radiation (R) - skip for non-radiating symptoms
        if element == 'radiation':
            non_radiating_keywords = [
                'hypertension', 'high blood pressure', 'hyperlipidemia', 'elevated cholesterol',
                'diabetes', 'fatigue', 'polyuria', 'polydipsia', 'constipation',
                'urinary incontinence', 'insomnia', 'depression', 'anxiety', 'dizziness'
            ]
            if any(keyword in chief_lower for keyword in non_radiating_keywords):
                return True
        
        return False

    # ----------- Answer persistence & scoring --------------------------------

    def _store_answer(self, session: "MedicalSession", pending: Dict[str, str], answer: str) -> Optional[Dict[str, Any]]:
        section, field = pending['section'], pending['field']
        text = answer.strip()
        
        if section == 'pre_hpi':
            if field == 'age':
                valid, normalized_value, validation_message = self._validate_age_answer(text)
                if not valid:
                    session.pending = pending
                    session.messages.append({"role": "assistant", "content": validation_message})
                    return self._wrap_response(
                        session,
                        validation_message,
                        status="validation_error",
                        metadata={'field': field, 'section': section, 'validation_error': True},
                    )
                session.context['pre_hpi'][field] = normalized_value
                session.stage = "awaiting_sex"
                session.pending = None
                return None  # No next prompt - wait for sex answer
            elif field == 'sex':
                valid, normalized_value, validation_message = self._validate_sex_answer(text)
                if not valid:
                    session.pending = pending
                    session.messages.append({"role": "assistant", "content": validation_message})
                    return self._wrap_response(
                        session,
                        validation_message,
                        status="validation_error",
                        metadata={'field': field, 'section': section, 'validation_error': True},
                    )
                session.context['pre_hpi'][field] = normalized_value
                session.stage = "hpi"
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
                session.pending = None
                next_prompt = self._next_oldcarts_question(session)
                if next_prompt:
                    session.pending = next_prompt
                    session.messages.append({"role": "assistant", "content": next_prompt['prompt']})
                    return self._wrap_response(
                        session,
                        next_prompt['prompt'],
                        metadata={
                            'section': next_prompt['section'],
                            'field': next_prompt['field'],
                        },
                    )
                return None
            else:
                # For other pre_hpi fields (e.g., chronicity), store raw text and adjust stage
                session.context['pre_hpi'][field] = text
                if field == 'chronicity':
                    session.stage = "awaiting_age"
                session.pending = None
                return None
        
        if section == 'hpi':
            # Check if this is a confused/clarification request
            if self._is_confused_response(text):
                self._capture_debug(f"[HPI] ⚠️ Confused/clarification request detected: '{text}'")
                # Don't store confused responses as answers - we'll re-ask the question
                if field in session.context['hpi']:
                    del session.context['hpi'][field]  # Remove if it was incorrectly stored
                
                # Provide clearer clarification for specific fields
                if field == 'severity':
                    cc_subject = self._normalize_subject_for_questions(session.context['pre_hpi'].get('chief_complaint'))
                    answered_hpi_count = len([k for k in session.context['hpi'].keys() if session.context['hpi'][k] and session.context['hpi'][k].strip()])
                    use_pronoun = answered_hpi_count > 0
                    symptom_ref = "it" if use_pronoun else cc_subject
                    
                    clarification = f"On a scale from 1 to 10, with 1 being very mild and 10 being the worst possible, how severe is {symptom_ref}? Just give me a number between 1 and 10."
                    session.pending = pending  # Keep the pending state
                    session.messages.append({"role": "assistant", "content": clarification})
                    return self._wrap_response(
                        session,
                        clarification,
                        metadata={
                            'section': pending['section'],
                            'field': pending['field'],
                            'clarification': True,
                        },
                    )
                else:
                    # For other fields, re-ask with the same question
                    session.pending = pending  # Keep the pending state
                    return self._wrap_response(
                        session,
                        pending['prompt'],  # Re-use the same question
                        metadata={
                            'section': pending['section'],
                            'field': pending['field'],
                            'reasking': True,
                        },
                    )
            
            # Validate that the answer is appropriate for the question being asked
            is_appropriate, validation_message = self._validate_hpi_answer_appropriateness(session, field, text, pending['prompt'])
            if not is_appropriate:
                self._capture_debug(f"[HPI] ⚠️ Answer not appropriate for question: '{text}' for field '{field}'")
                # Don't store inappropriate answers - ask for clarification
                if field in session.context['hpi']:
                    del session.context['hpi'][field]  # Remove if it was incorrectly stored
                
                # Provide clarification message
                clarification_msg = validation_message or pending['prompt']
                session.pending = pending  # Keep the pending state
                session.messages.append({"role": "assistant", "content": clarification_msg})
                return self._wrap_response(
                    session,
                    clarification_msg,
                    metadata={
                        'section': pending['section'],
                        'field': pending['field'],
                        'validation_error': True,
                    },
                )
            
            # Store answer (not a confused response and appropriate for the question)
            session.context['hpi'][field] = text
            self._capture_debug(f"[HPI] ✅ Stored answer for {field}: {text}")
            
            # Remove from remaining list if present
            if field in session.oldcarts_remaining:
                session.oldcarts_remaining = [e for e in session.oldcarts_remaining if e != field]
                self._capture_debug(f"[HPI] ✅ Removed {field} from oldcarts_remaining list")
            
            # Also ensure it's not in the list by regenerating if needed (safety check)
            # This ensures answered fields are never re-asked
            if field not in session.context['hpi'] or not session.context['hpi'][field] or not session.context['hpi'][field].strip():
                self._capture_debug(f"[HPI] ⚠️ Warning: Field {field} was supposed to be stored but isn't in hpi context")
            
            # Update condition scores using LLM reasoning
            self._update_condition_scores_from_answer(session, field, text)
            
            # Get next OLD CARTS question
            session.pending = None
            next_prompt = self._next_oldcarts_question(session)
            if next_prompt:
                session.pending = next_prompt
                session.messages.append({"role": "assistant", "content": next_prompt['prompt']})
                return self._wrap_response(
                    session,
                    next_prompt['prompt'],
                    metadata={
                        'section': next_prompt['section'],
                        'field': next_prompt['field'],
                    },
                )
        
            return None

    def _update_condition_scores_from_answer(self, session: "MedicalSession", element: str, answer: str) -> None:
        """Update condition scores based on answer - LLM evaluates ALL conditions and can add new ones."""
        if not self.llm_chat_fn:
            return
        
        # Check for missing conditions that should be added (e.g., GERD for chest pain)
        self._check_for_missing_conditions(session, element, answer)
        
        # Get ALL conditions in the session, not just top 5
        # This allows LLM to discover the correct condition even if it wasn't initially high
        all_conditions = list(session.condition_scores.keys())
        
        if not all_conditions:
            return
        
        # Build context
        chief_complaint = session.context['pre_hpi'].get('chief_complaint', '')
        conversation_context = self._build_conversation_context(session)
        
        # Get current rankings for context (but evaluate ALL conditions)
        current_rankings = session.condition_rankings[:10] if session.condition_rankings else []
        ranking_context = ", ".join([f"{cond} ({score:.2f})" for cond, score in current_rankings[:5]])
        
        # Ask LLM to evaluate ALL conditions using its trained medical knowledge
        # CRITICAL: Use very strong JSON-only prompt to prevent conversational responses
        system_prompt = (
            "You are a medical expert with extensive training in clinical reasoning. "
            "You MUST return ONLY valid JSON. No explanations, no text before or after the JSON.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- If the patient's answer is unclear, nonsensical, or doesn't make logical sense, return neutral scores (0.0) for all conditions.\n"
            "- Never invent symptoms, diagnoses, or medical details that aren't in the patient's answer.\n\n"
            "CRITICAL FORMAT REQUIREMENTS:\n"
            "- Output ONLY valid JSON (no explanations, no text before or after)\n"
            "- JSON must be an object with ALL condition names as keys and numeric scores as values\n"
            "- Example format: {\"Acute Appendicitis\": 0.2, \"Nephrolithiasis (Kidney Stones)\": -0.1, \"Acute Cholecystitis\": 0.0}\n"
            "- Each condition must be a key in the JSON object with its score change as the value\n"
            "- Scores must be between -0.3 and +0.3 (numeric values only)\n"
            "- Do NOT use any other format - only JSON object with condition names as keys and numeric values\n\n"
            "Based on the patient's answer, evaluate how it affects the likelihood of EACH condition. "
            "Use your trained medical knowledge. Consider: classic presentations, anatomical locations, symptom patterns.\n\n"
            "CRITICAL: Know classic symptom patterns:\n"
            "- GERD: Burning chest pain, WORSE when laying down (especially after meals), better with antacids/sitting up\n"
            "- Cardiac: Chest pain worse with exertion, better with rest, NOT typically worse when laying down\n"
            "- Pleuritic: Chest pain worse with breathing, coughing\n"
            "- Biliary: Right upper quadrant pain, worse after fatty meals\n"
            "- Appendicitis: Right lower quadrant pain, worse with movement\n\n"
            "Positive values (+0.2 to +0.3) = condition MORE likely (answer matches classic pattern). "
            "Negative values (-0.2 to -0.3) = condition LESS likely (answer contradicts classic pattern). "
            "Neutral = small changes (-0.1 to +0.1)."
        )
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n\n"
            f"All conditions to evaluate ({len(all_conditions)} total):\n"
            f"{', '.join(all_conditions)}\n\n"
            f"Return ONLY valid JSON with ALL {len(all_conditions)} conditions as keys and their score changes as values.\n"
            f"Format: {{\"condition_name\": score_change, \"condition_name\": score_change, ...}}\n"
            f"Example (not actual conditions): {{\"Condition1\": 0.2, \"Condition2\": -0.1, \"Condition3\": 0.0}}\n\n"
            f"CRITICAL: Return ONLY the JSON object. No explanations, no text before or after. "
            f"Every condition listed above must be a key in the JSON with a numeric score between -0.3 and +0.3."
        )
        
        self._capture_debug(f"[Scoring] 🔍 Evaluating {len(all_conditions)} conditions for {element} answer: '{answer}'")
        self._capture_debug(f"[Scoring] 📋 Conditions: {', '.join(all_conditions[:5])}{'...' if len(all_conditions) > 5 else ''}")
        
        try:
            self._capture_debug(f"[Scoring] 🤖 Calling LLM for score evaluation...")
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,  # Reduced for faster responses while maintaining condition evaluation quality
                temperature=0.0,  # Deterministic for scoring
            )
            
            if response:
                self._capture_debug(f"[Scoring] 📥 Raw LLM response (first 200 chars): {response[:200]}")
                # Parse JSON response - try multiple extraction methods
                cleaned = response.strip()
                
                # Method 1: Remove markdown code blocks
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                        elif '```' in cleaned:
                            # Find last ```
                            last_idx = cleaned.rfind('```')
                            cleaned = cleaned[:last_idx].strip()
                
                # Method 2: Extract JSON object using brace matching
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        cleaned = cleaned[start_idx:end_idx]
                
                # Method 3: Remove any text before first { or after last }
                if '{' in cleaned:
                    cleaned = cleaned[cleaned.find('{'):]
                if '}' in cleaned:
                    cleaned = cleaned[:cleaned.rfind('}')+1]
                
                try:
                    score_changes = json.loads(cleaned)
                    if isinstance(score_changes, dict):
                        self._capture_debug(f"[Scoring] ✅ Successfully parsed JSON with {len(score_changes)} condition scores")
                        # Apply score changes to all conditions
                        updated_count = 0
                        significant_changes = []
                        for condition, change in score_changes.items():
                            if condition in session.condition_scores:
                                current_score = session.condition_scores[condition]
                                # Clamp change to reasonable range
                                change_value = max(-0.3, min(0.3, float(change)))
                                new_score = max(0.0, min(1.0, current_score + change_value))
                                session.condition_scores[condition] = new_score
                                updated_count += 1
                                # Log significant changes
                                if abs(change_value) >= 0.1:
                                    significant_changes.append(
                                        f"{condition}: {current_score:.3f} → {new_score:.3f} ({change_value:+.3f})"
                                    )
                                    self._capture_debug(
                                        f"[Scoring]   📈 {condition}: {current_score:.3f} → {new_score:.3f} ({change_value:+.3f})"
                                    )
                            else:
                                # LLM suggested a new condition - add it at baseline and apply change
                                change_value = max(-0.3, min(0.3, float(change)))
                                new_score = max(0.0, min(1.0, 0.5 + change_value))
                                session.condition_scores[condition] = new_score
                                updated_count += 1
                                self._capture_debug(f"[Scoring] 🆕 Added new condition: {condition} (initial score: {new_score:.3f})")
                        
                        if updated_count > 0:
                            self._capture_debug(f"[Scoring] ✅ Updated {updated_count}/{len(session.condition_scores)} conditions based on {element} answer")
                            if significant_changes:
                                self._capture_debug(f"[Scoring] 📊 Significant changes ({len(significant_changes)}):")
                                for change in significant_changes[:10]:  # Show top 10
                                    self._capture_debug(f"[Scoring]   • {change}")
                        else:
                            self._capture_debug(f"[Scoring] ⚠️ No conditions matched in LLM response")
                            self._capture_debug(f"[Scoring] ⚠️ LLM returned {len(score_changes)} conditions, but none matched session conditions")
                            self._capture_debug(f"[Scoring] ⚠️ LLM keys: {list(score_changes.keys())[:5]}")
                            self._capture_debug(f"[Scoring] ⚠️ Session keys: {list(session.condition_scores.keys())[:5]}")
                except json.JSONDecodeError as e:
                    self._capture_debug(f"[Scoring] ❌ Failed to parse LLM score changes: {e}")
                    self._capture_debug(f"[Scoring] ⚠️ Raw response (first 500 chars): {response[:500]}")
                    self._capture_debug(f"[Scoring] ⚠️ Extracted JSON attempt (first 300 chars): {cleaned[:300]}")
                    self._capture_debug(f"[Scoring] ⚠️ This usually means LLM returned conversational text instead of JSON")
                    self._capture_debug(f"[Scoring] ⚠️ Check if model is fine-tuned and following JSON-only instructions")
        except Exception as e:
            self._capture_debug(f"[Scoring] ⚠️ Error updating scores: {e}")
        
        # Update rankings
        self._apply_rule_outs(session)

    def _build_conversation_context(self, session: "MedicalSession") -> str:
        """Build conversation context for LLM."""
        parts = []
        
        pre_hpi = session.context.get('pre_hpi', {})
        if pre_hpi.get('chief_complaint'):
            parts.append(f"Chief complaint: {pre_hpi['chief_complaint']}")
        if pre_hpi.get('chronicity'):
            parts.append(f"Chronicity: {pre_hpi['chronicity']}")
        if pre_hpi.get('age'):
            parts.append(f"Age: {pre_hpi['age']}")
        if pre_hpi.get('sex'):
            parts.append(f"Biological sex: {pre_hpi['sex']}")
        
        hpi = session.context.get('hpi', {})
        hpi_labels = {
            'onset': 'Onset',
            'location': 'Location',
            'duration': 'Duration',
            'character': 'Character',
            'aggravating': 'Aggravating factors',
            'relieving': 'Relieving factors',
            'radiation': 'Radiation',
            'timing': 'Timing',
            'severity': 'Severity',
        }
        
        for key, label in hpi_labels.items():
            value = hpi.get(key)
            if value and value.strip():
                parts.append(f"{label}: {value}")
        
        if not parts:
            return "No information collected yet."
        
        return "\n".join(parts)

    # ----------- Chief complaint matching ------------------------------------

    def _apply_rule_outs(self, session: "MedicalSession") -> None:
        """Update condition rankings - keep this for ranking system."""
        sorted_scores = sorted(session.condition_scores.items(), key=lambda x: x[1], reverse=True)
        session.condition_rankings = sorted_scores
        self._update_condition_pools(session)
        self._log_rankings(session)

    def _match_chief_complaint_to_category_llm(self, chief_complaint: str) -> List[str]:
        """Match chief complaint to medical categories using LLM only."""
        if not self.llm_chat_fn:
            # Fallback: return all categories
            return list(self.CATEGORY_TO_SYSTEM.keys())
        
        # Get available categories from default categories
        available_categories = set(self.CATEGORY_TO_SYSTEM.keys())
        
        available_cats_str = ', '.join(sorted(available_categories))
        
        system_prompt = (
            "You are a medical expert with extensive training in clinical reasoning. "
            "Based on the chief complaint, identify which medical categories are relevant using your medical knowledge.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- If the chief complaint is unclear, nonsensical, or doesn't make logical sense, return an empty categories array: {\"categories\": []}\n"
            "- Never invent medical conditions or categories that don't logically relate to the chief complaint.\n\n"
            "CRITICAL: Consider ALL possible causes, not just the most obvious one. You MUST include multiple categories when appropriate.\n\n"
            "EXAMPLES:\n"
            "- Chest pain: MUST include ['cardiovascular', 'respiratory', 'gastrointestinal'] because:\n"
            "  * Cardiovascular: MI, angina, aortic dissection, pericarditis\n"
            "  * Respiratory: PE, pneumonia, pneumothorax, pleuritis\n"
            "  * Gastrointestinal: GERD, peptic ulcer, esophagitis\n"
            "- Abdominal pain: MUST include ['gastrointestinal', 'renal', 'genitourinary'] because multiple systems can cause it\n"
            "- Shortness of breath: MUST include ['respiratory', 'cardiovascular']\n\n"
            "RULE: If the chief complaint could reasonably be caused by multiple organ systems, you MUST include ALL of them.\n"
            "Do NOT default to just one category. Think like a doctor building a differential diagnosis.\n\n"
            f"Available categories: {available_cats_str}\n\n"
            "Return ONLY valid JSON: {\"categories\": [\"category1\", \"category2\", \"category3\", ...]}\n"
            "No explanations, no other text. Just the JSON object."
        )
        
        user_prompt = (
            f"Chief complaint: '{chief_complaint}'\n\n"
            "Using your medical knowledge, identify which categories are relevant. "
            "Consider all possible causes, not just the most obvious one."
        )
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            
            if not response:
                self._capture_debug(f"[Category] LLM returned empty response, defaulting to all categories")
                return list(available_categories) if available_categories else ['gastrointestinal']
            
            # Parse JSON - use robust parsing
            cleaned = response.strip()
            
            # Remove markdown code blocks
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline+1:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3].strip()
                    elif '```' in cleaned:
                        last_idx = cleaned.rfind('```')
                        cleaned = cleaned[:last_idx].strip()
            
            # Extract JSON using brace matching
            start_idx = cleaned.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned)):
                    if cleaned[i] == '{':
                        brace_count += 1
                    elif cleaned[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx]
            
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'categories' in parsed:
                    parsed_categories = parsed['categories']
                    if isinstance(parsed_categories, list):
                        valid_cats = [cat for cat in parsed_categories if cat in available_categories]
                        if valid_cats:
                            self._capture_debug(f"[Category] LLM matched '{chief_complaint}' to categories: {valid_cats}")
                            return valid_cats
                        else:
                            self._capture_debug(f"[Category] LLM returned invalid categories: {parsed_categories}, defaulting to all categories")
                    else:
                        self._capture_debug(f"[Category] LLM returned non-list categories: {parsed_categories}, defaulting to all categories")
                else:
                    self._capture_debug(f"[Category] Failed to parse categories from LLM response, defaulting to all categories")
            except json.JSONDecodeError as e:
                self._capture_debug(f"[Category] JSON parse error: {e}, defaulting to all categories")
            
            # If LLM fails, default to all categories (let scoring narrow down)
            self._capture_debug(f"[Category] Defaulting to all available categories: {list(available_categories)}")
            return list(available_categories) if available_categories else ['gastrointestinal']
            
        except Exception as e:
            self._capture_debug(f"⚠️  Error in category matching: {e}, defaulting to all categories")
            return list(available_categories) if available_categories else ['gastrointestinal']
 
    # ----------- Guidance builders -------------------------------------------


    def _initialize_condition_scores_llm(self, categories: List[str], chief_complaint: str) -> Dict[str, float]:
        """Initialize condition scores - LLM suggests relevant conditions dynamically."""
        if not self.llm_chat_fn:
            # No LLM available - return empty dict
            return {}
        
        # Use LLM's medical knowledge to suggest relevant conditions
        system_prompt = (
            "You are a medical expert with extensive training in clinical reasoning. "
            "Based on the chief complaint and medical categories, suggest relevant medical conditions "
            "that should be considered in the differential diagnosis.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- If the chief complaint or categories are unclear or don't make logical sense, return an empty conditions array: {\"conditions\": []}\n"
            "- Never invent medical conditions that don't logically relate to the chief complaint and categories.\n"
            "- Only suggest conditions that are medically relevant and factual.\n\n"
            "CRITICAL: You MUST suggest a COMPREHENSIVE differential diagnosis. Include:\n"
            "1. Common conditions (most likely)\n"
            "2. Serious conditions that must be ruled out (can't miss diagnoses)\n"
            "3. Conditions from ALL relevant categories provided\n\n"
            "MINIMUM REQUIREMENTS:\n"
            "- For chest pain with 3 categories: Suggest AT LEAST 8-10 conditions (3-4 per category)\n"
            "- For abdominal pain: Suggest AT LEAST 6-8 conditions\n"
            "- For other complaints: Suggest AT LEAST 5-6 conditions\n\n"
            "EXAMPLES:\n"
            "- Chest pain with ['cardiovascular', 'respiratory', 'gastrointestinal']:\n"
            "  * Cardiovascular: Acute MI, Unstable Angina, Stable Angina, Aortic Dissection, Pericarditis\n"
            "  * Respiratory: Pulmonary Embolism, Pneumonia, Pneumothorax, Pleuritis\n"
            "  * Gastrointestinal: GERD, Peptic Ulcer Disease\n"
            "  Total: 10+ conditions\n\n"
            "Use your medical knowledge to identify conditions that could cause these symptoms.\n\n"
            "CRITICAL FORMAT REQUIREMENT:\n"
            "Return ONLY valid JSON in this EXACT format: {\"conditions\": [\"Condition 1\", \"Condition 2\", \"Condition 3\", ...]}\n"
            "Example: {\"conditions\": [\"Acute Myocardial Infarction (Heart Attack)\", \"Unstable Angina\", \"Aortic Dissection\", \"Pulmonary Embolism\", \"Pneumonia\", \"GERD\", \"Peptic Ulcer Disease\", \"Pericarditis\"]}\n"
            "DO NOT return multiple JSON objects on separate lines.\n"
            "DO NOT return just condition names without the array.\n"
            "Return ONLY the single JSON object. No explanations, no other text."
        )
        
        available_conditions_str = "LLM will suggest based on medical knowledge"
        
        user_prompt = (
            f"Chief complaint: '{chief_complaint}'\n"
            f"Medical categories: {', '.join(categories)}\n"
            f"Available conditions (examples): {available_conditions_str}\n\n"
            "Suggest a COMPREHENSIVE list of relevant medical conditions for differential diagnosis. "
            "Include conditions from ALL categories provided. "
            "Suggest AT LEAST 8-10 conditions for chest pain, or 5-6 minimum for other complaints. "
            "Include both common and serious conditions that must be ruled out. "
            "You can suggest conditions from the examples above, or other conditions you know about."
        )
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,  # Reduced for faster responses while handling longer condition lists
                temperature=0.0,
            )
            
            if response:
                self._capture_debug(f"[Condition Init] 📥 LLM response (first 300 chars): {response[:300]}")
                # Parse JSON response - handle multiple formats
                cleaned = response.strip()
                
                # Remove markdown code blocks
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                        elif '```' in cleaned:
                            last_idx = cleaned.rfind('```')
                            cleaned = cleaned[:last_idx].strip()
                
                # Try to parse as single JSON object first
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        single_json = cleaned[start_idx:end_idx]
                        try:
                            parsed = json.loads(single_json)
                            if isinstance(parsed, dict) and 'conditions' in parsed:
                                suggested_conditions = parsed['conditions']
                                if isinstance(suggested_conditions, list) and suggested_conditions:
                                    # Start ALL suggested conditions at balanced baseline (0.5)
                                    condition_scores = {cond: 0.5 for cond in suggested_conditions}
                                    self._capture_debug(f"[Engine] 📋 LLM suggested {len(condition_scores)} conditions at balanced baseline 50.0%")
                                    self._capture_debug(f"[Engine]    Conditions: {', '.join(list(condition_scores.keys())[:5])}{'...' if len(condition_scores) > 5 else ''}")
                                    return condition_scores
                        except json.JSONDecodeError:
                            pass  # Try alternative parsing below
                
                # Alternative: Handle multiple JSON objects on separate lines
                # Format: {"Condition 1"}\n{"Condition 2"}\n...
                suggested_conditions = []
                lines = cleaned.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Try to extract JSON object from line
                    start_idx = line.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(line)):
                            if line[i] == '{':
                                brace_count += 1
                            elif line[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > start_idx:
                            json_str = line[start_idx:end_idx]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    # Extract condition name from dict
                                    # Could be {"Condition Name"} or {"conditions": ["Condition"]} or just keys
                                    if 'conditions' in parsed and isinstance(parsed['conditions'], list):
                                        suggested_conditions.extend(parsed['conditions'])
                                    else:
                                        # Extract all string values from dict
                                        for key, value in parsed.items():
                                            if isinstance(value, str):
                                                suggested_conditions.append(value)
                                            elif isinstance(value, list):
                                                suggested_conditions.extend([v for v in value if isinstance(v, str)])
                                        # If no values, use keys as condition names
                                        if not suggested_conditions:
                                            suggested_conditions.extend([k for k in parsed.keys() if isinstance(k, str)])
                            except json.JSONDecodeError:
                                continue
                
                # If we found conditions in the alternative format
                if suggested_conditions:
                    # Remove duplicates and clean up
                    suggested_conditions = list(dict.fromkeys(suggested_conditions))  # Preserve order, remove dupes
                    condition_scores = {cond: 0.5 for cond in suggested_conditions}
                    self._capture_debug(f"[Engine] 📋 LLM suggested {len(condition_scores)} conditions at balanced baseline 50.0%")
                    self._capture_debug(f"[Engine]    Conditions: {', '.join(list(condition_scores.keys())[:5])}{'...' if len(condition_scores) > 5 else ''}")
                    return condition_scores
                
                # If we still couldn't parse, show error
                self._capture_debug(f"[Condition Init] ⚠️ Failed to parse JSON in any format")
                self._capture_debug(f"[Condition Init] ⚠️ Extracted JSON attempt (first 200 chars): {cleaned[:200]}")
            else:
                self._capture_debug(f"[Condition Init] ⚠️ LLM returned empty response")
            
            # LLM failed - return empty dict
            self._capture_debug(f"[Condition Init] ⚠️ LLM condition suggestion failed, returning empty conditions")
            return {}
            
        except Exception as e:
            self._capture_debug(f"⚠️  Error initializing conditions: {e}, returning empty conditions")
            return {}


    def _normalize_subject_for_questions(self, text: Optional[str]) -> str:
        """Normalize chief complaint to extract core symptom phrase for use in questions."""
        if not text:
            return 'your symptoms'
        subject = text.strip()
        lowered = subject.lower()
        
        # Remove common conversational prefixes
        prefixes = [
            "i have ", "i've got ", "i am having ", "i'm having ",
            "i am ", "i'm ", "my ", "i feel ", "i've been having ",
            "i've been experiencing ", "i'm experiencing ", "i'm feeling ",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                subject = subject[len(prefix):].strip()
                break
        
        # Clean up punctuation and whitespace
        subject = subject.strip(" .,!?:;")
        
        # Extract core symptom if there are multiple clauses
        # Example: "i am having chest pain pain" -> "chest pain"
        words = subject.split()
        if len(words) > 5:
            # If too long, try to extract the key symptom phrase
            # Look for common symptom patterns (noun phrases)
            common_symptoms = ['pain', 'ache', 'discomfort', 'pressure', 'tightness', 
                             'nausea', 'dizziness', 'shortness', 'breath', 'cough',
                             'fever', 'headache', 'stomach', 'chest', 'back', 'arm',
                             'leg', 'joint', 'muscle', 'throat', 'rash', 'swelling']
            symptom_words = [w for w in words if any(symptom in w.lower() for symptom in common_symptoms)]
            if symptom_words:
                # Extract phrase around symptom words
                indices = [i for i, w in enumerate(words) if any(symptom in w.lower() for symptom in common_symptoms)]
                if indices:
                    start = max(0, indices[0] - 1)
                    end = min(len(words), indices[-1] + 2)
                    subject = ' '.join(words[start:end])
        
        # Remove duplicate words (e.g., "chest pain pain" -> "chest pain")
        words = subject.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()
        subject = ' '.join(cleaned_words) if cleaned_words else 'your symptoms'
        
        subject = subject.strip(" .,!?:;")
        if not subject or len(subject) < 2:
            return 'your symptoms'
        return subject.lower()  # Return lowercase for consistent use in questions
    
    def _get_base_question_for_element(self, element: str, chief_complaint: str, use_pronoun: bool = False) -> str:
        """Get base question format for OLD CARTS element.
        
        Args:
            element: OLD CARTS element name
            chief_complaint: Normalized chief complaint
            use_pronoun: If True, use 'it' instead of repeating chief complaint
        """
        # Use pronoun for better natural language after first question
        symptom_ref = "it" if use_pronoun else chief_complaint
        
        element_guidance = {
            'onset': f"When did {symptom_ref} start?" if use_pronoun else f"When did the {chief_complaint} start?",
            'location': f"Where exactly is {symptom_ref} located?" if use_pronoun else f"Where exactly is the {chief_complaint} located?",
            'duration': f"How long has {symptom_ref} been present?" if use_pronoun else f"How long has the {chief_complaint} been present?",
            'character': f"What does {symptom_ref} feel like?" if use_pronoun else f"What does the {chief_complaint} feel like?",
            'aggravating': f"What makes {symptom_ref} worse?" if use_pronoun else f"What makes the {chief_complaint} worse?",
            'relieving': f"What makes {symptom_ref} better?" if use_pronoun else f"What makes the {chief_complaint} better?",
            'radiation': f"Does {symptom_ref} spread to other areas?" if use_pronoun else f"Does the {chief_complaint} spread to other areas?",
            'timing': f"Is {symptom_ref} constant or does it come and go?" if use_pronoun else f"Is the {chief_complaint} constant or does it come and go?",
            'severity': f"On a scale from 1 to 10, with 1 being very mild and 10 being the worst possible, how severe is {symptom_ref}?" if use_pronoun else f"On a scale from 1 to 10, with 1 being very mild and 10 being the worst possible, how severe is the {chief_complaint}?",
            'associated': f"Are there any other symptoms you're experiencing?" if use_pronoun else f"Are there any other symptoms you're experiencing along with the {chief_complaint}?",
        }
        
        default = f"Tell me about the {element} of {symptom_ref}." if use_pronoun else f"Tell me about the {element} of {chief_complaint}."
        return element_guidance.get(element, default)
    
    def _validate_hpi_question(self, question: str, element: str, chief_complaint: str, use_pronoun: bool = False) -> str:
        """Validate and fix HPI questions to avoid nonsensical phrasing."""
        question_lower = question.lower()
        
        # Check for nonsensical questions like "how old is your chest pain?"
        if 'old' in question_lower and 'age' not in question_lower and element != 'age':
            # LLM generated wrong question, use base question
            self._capture_debug(f"[LLM] ⚠️ Detected nonsensical question, using base question instead")
            return self._get_base_question_for_element(element, chief_complaint, use_pronoun=use_pronoun)
        
        # Check for awkward character phrasing
        if element == 'character' and ('character of' in question_lower or 'the character' in question_lower):
            # LLM generated awkward phrasing for character - use base question
            self._capture_debug(f"[LLM] ⚠️ Detected awkward character question, using base question instead")
            return self._get_base_question_for_element(element, chief_complaint, use_pronoun=use_pronoun)
        
        # Check if question is too short or empty
        if not question or len(question) < 10:
            return self._get_base_question_for_element(element, chief_complaint, use_pronoun=use_pronoun)
        
        return question

    def _ordered_oldcarts_elements(self, session: "MedicalSession") -> List[str]:
        """Get ordered OLD CARTS elements - simplified, LLM handles priority."""
        ordered = self.HPI_ELEMENTS.copy()
        # Only include elements that are actually answered (not confused responses)
        answered_elements = {k: v for k, v in session.context['hpi'].items() if v and v.strip() and not self._is_confused_response(v)}
        answered = set(answered_elements.keys())
        filtered = [element for element in ordered if element not in answered]
        return filtered

    # ----------- LLM helpers -------------------------------------------------

    def _generate_question(
        self,
        session: "MedicalSession",
        section: str,
        field: str,
        guidance: str,
        use_pronoun: bool = False,
        cc_subject: Optional[str] = None,
    ) -> str:
        if not self.llm_chat_fn:
            return guidance or f"Tell me about {field}."
        
        cc = session.context['pre_hpi'].get('chief_complaint', 'your symptoms') or 'your symptoms'
        if cc_subject is None:
            cc_subject = self._normalize_subject_for_questions(cc)
        context_summary = self._build_conversation_context(session)
        
        # Build conversation context
        conversation_context = []
        recent_messages = session.messages[-6:] if len(session.messages) > 6 else session.messages
        for msg in recent_messages:
            if msg.get('role') in ['assistant', 'user']:
                conversation_context.append({"role": msg['role'], "content": msg['content']})
        
        if section == 'hpi':
            # Get base question format for this element (with pronoun if appropriate)
            base_question = self._get_base_question_for_element(field, cc_subject, use_pronoun=use_pronoun)
            
            # Build natural guidance based on whether we're using pronouns
            if use_pronoun:
                user_prompt = f"""Context of what we already know:
{context_summary}

You need to ask about the {field}. 
IMPORTANT: Use natural language with pronouns (like "it" or "your symptoms") instead of repeating the complaint. 
Use this as a guide: '{base_question}'
Do NOT ask about age, demographics, or information already in the context. 
Do NOT use awkward phrases like 'character of' or 'the {field}'. 
CRITICAL: Output ONLY the question. No reasoning, no notes, no explanations, no "CLINICAL NOTE:" prefixes.
Ask only one natural, conversational question about {field}."""
            else:
                user_prompt = f"""Context of what we already know:
{context_summary}

You need to ask about the {field} of the {cc_subject}. 
IMPORTANT: Ask about {field} using natural language. Use this as a guide: '{base_question}' 
Do NOT ask about age, demographics, or information already in the context. 
Do NOT use awkward phrases like 'character of' or 'the {field}'. 
CRITICAL: Output ONLY the question. No reasoning, no notes, no explanations, no "CLINICAL NOTE:" prefixes.
Ask only one natural, conversational question about {field}."""
        elif section == 'pre_hpi':
            user_prompt = guidance
        else:
            user_prompt = f"Ask about {field}. Ask only one question."
        
        messages = [{"role": "system", "content": self.QUESTION_SYSTEM_PROMPT}]
        if conversation_context:
            messages.extend(conversation_context)
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.llm_chat_fn(
            messages,
            max_tokens=self.LLM_MAX_TOKENS_QUESTIONS,
            temperature=self.LLM_TEMPERATURE_QUESTIONS,
        )
        
        self._capture_debug(f"[LLM] ❓ Question prompt:\n{user_prompt}")
        self._capture_debug(f"[LLM] ❓ Raw question response: {response}")
        
        cleaned = self._clean_llm_response(response)
        if not cleaned:
            # Fall back to base question if LLM doesn't generate good question
            cleaned = base_question if section == 'hpi' else (guidance or f"Tell me about {field}.")
        
        if not cleaned.endswith('?'):
            cleaned = cleaned.rstrip('.') + '?'
        
        # Quality check: detect nonsensical questions
        if section == 'hpi':
            cleaned = self._validate_hpi_question(cleaned, field, cc_subject, use_pronoun=use_pronoun)
        
        return cleaned

    def _generate_question_streaming(
        self,
        session: "MedicalSession",
        section: str,
        field: str,
        guidance: str,
    ):
        """
        Generate question with streaming - yields tokens as they're generated.
        Returns a generator that yields token strings as they come from the LLM.
        
        NOTE: This is a generator - it must be consumed to get the tokens.
        The full response is accumulated internally, but tokens are yielded immediately.
        """
        if not self.llm_chat_fn:
            fallback = guidance or f"Tell me about {field}."
            # Yield fallback token-by-token
            for token in fallback.split():
                yield token + " "
            return
        
        cc = session.context['pre_hpi'].get('chief_complaint', 'your symptoms') or 'your symptoms'
        context_summary = self._build_conversation_context(session)
        
        # Build conversation context
        conversation_context = []
        recent_messages = session.messages[-6:] if len(session.messages) > 6 else session.messages
        for msg in recent_messages:
            if msg.get('role') in ['assistant', 'user']:
                conversation_context.append({"role": msg['role'], "content": msg['content']})
        
        if section == 'hpi':
            base_question = self._get_base_question_for_element(field, cc)
            user_prompt = f"""Context of what we already know:
{context_summary}

You need to ask about the {field} of the {cc}. 
IMPORTANT: You MUST ask about {field} specifically using this exact format: '{base_question}' 
Do NOT ask about age, demographics, or information already in the context. 
Do NOT use phrases like 'character of' or 'the {field}' - use the natural question format shown above. 
Ask only one question about {field}."""
        elif section == 'pre_hpi':
            user_prompt = guidance
        else:
            user_prompt = f"Ask about {field}. Ask only one question."
        
        messages = [{"role": "system", "content": self.QUESTION_SYSTEM_PROMPT}]
        if conversation_context:
            messages.extend(conversation_context)
        messages.append({"role": "user", "content": user_prompt})
        
        self._capture_debug(f"[LLM] ❓ Question prompt (streaming):\n{user_prompt}")
        
        # Call LLM with streaming enabled
        try:
            stream = self.llm_chat_fn(
                messages,
                max_tokens=self.LLM_MAX_TOKENS_QUESTIONS,
                temperature=self.LLM_TEMPERATURE_QUESTIONS,
                stream=True,  # Enable streaming
            )
            
            # Buffer tokens until we have complete words, then yield words
            # This prevents awkward sub-word splits like "G all bl adder" → "Gallbladder"
            token_buffer = ""
            for chunk in stream:
                if isinstance(chunk, dict):
                    # Extract content from chunk (OpenAI-style format)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            token_buffer += content
                elif isinstance(chunk, str):
                    token_buffer += chunk
                
                # Check if buffer contains complete words (space or punctuation indicates word boundary)
                # Yield complete words, keep remaining sub-word pieces in buffer
                # Optimized: use regex to find first boundary instead of multiple find() calls
                while token_buffer:
                    # Find first word boundary using regex (more efficient than multiple find() calls)
                    match = re.search(r'[ \.\,\!\?\:\;\n\-\(\)\[\]]', token_buffer)
                    if match:
                        boundary_pos = match.start()
                        # Found word boundary - yield complete word(s) up to boundary (inclusive)
                        word = token_buffer[:boundary_pos + 1]
                        token_buffer = token_buffer[boundary_pos + 1:]
                        if word.strip():  # Only yield non-empty words
                            yield word
                    else:
                        # No word boundary yet - keep buffering
                        break
            
            # After stream ends, yield any remaining buffered content
            if token_buffer.strip():
                yield token_buffer
        except Exception as e:
            print(f"[Navigator] ⚠️ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to non-streaming if streaming fails
            try:
                response = self.llm_chat_fn(
                    messages,
                    max_tokens=self.LLM_MAX_TOKENS_QUESTIONS,
                    temperature=self.LLM_TEMPERATURE_QUESTIONS,
                    stream=False,
                )
                full_response = response or (guidance or f"Tell me about {field}.")
                # Yield fallback word-by-word
                for token in full_response.split():
                    yield token + " "
            except Exception as e2:
                print(f"[Navigator] ❌ Fallback also failed: {e2}")
                # Last resort fallback
                fallback = guidance or f"Tell me about {field}."
                for token in fallback.split():
                    yield token + " "

    def _generate_empathetic_statement(self, chief_complaint: str) -> str:
        if not self.llm_chat_fn:
            return "I understand you're experiencing that. I'm here to help."
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.EMPATHETIC_SYSTEM_PROMPT},
                {"role": "user", "content": f"I have {chief_complaint}"},
            ],
            max_tokens=self.LLM_MAX_TOKENS_EMPATHETIC,
            temperature=self.LLM_TEMPERATURE_EMPATHETIC,
        )
        return self._clean_llm_response(response, fallback="I understand you're experiencing that. I'm here to help.")

    def _generate_chronicity_question(self) -> str:
        if not self.llm_chat_fn:
            return "Is this a new problem or something you've experienced before?"
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.CHRONICITY_SYSTEM_PROMPT},
                {"role": "user", "content": "Ask if this is new or an ongoing problem."},
            ],
            max_tokens=self.LLM_MAX_TOKENS_CHRONICITY,
            temperature=self.LLM_TEMPERATURE_QUESTIONS,
        )
        return self._clean_llm_response(response, fallback="Is this a new problem or something you've experienced before?")

    def _generate_summary(self, session: "MedicalSession") -> str:
        if not self.llm_chat_fn:
            return "History collection complete."
        
        pre = session.context['pre_hpi']
        
        # Filter out confused responses from HPI context before generating summary
        hpi_filtered = {
            k: v for k, v in session.context['hpi'].items()
            if v and v.strip() and not self._is_confused_response(v)
        }
        
        rankings = session.condition_rankings[:3]
        ranking_text = ", ".join(f"{name} ({score:.0%})" for name, score in rankings) if rankings else "No ranked conditions yet"
        
        # Format HPI data for summary
        hpi_items = []
        if hpi_filtered:
            for key, value in hpi_filtered.items():
                if key == 'onset':
                    hpi_items.append(f"Onset: {value}")
                elif key == 'location':
                    hpi_items.append(f"Location: {value}")
                elif key == 'duration':
                    hpi_items.append(f"Duration: {value}")
                elif key == 'character':
                    hpi_items.append(f"Character: {value}")
                elif key == 'aggravating':
                    hpi_items.append(f"Aggravating factors: {value}")
                elif key == 'relieving':
                    hpi_items.append(f"Relieving factors: {value}")
                elif key == 'radiation':
                    hpi_items.append(f"Radiation: {value}")
                elif key == 'timing':
                    hpi_items.append(f"Timing: {value}")
                elif key == 'severity':
                    hpi_items.append(f"Severity: {value}")
                elif key == 'associated':
                    hpi_items.append(f"Associated symptoms: {value}")
        
        hpi_summary = ", ".join(hpi_items) if hpi_items else "Not collected"
        
        user_prompt = (
            f"Create a clinical summary with the following information:\n\n"
            f"Chief complaint: {pre.get('chief_complaint', 'Not stated')}\n"
            f"Chronicity: {pre.get('chronicity', 'Unknown')}\n"
            f"Age: {pre.get('age', 'Unknown')}\n"
            f"Biological sex: {pre.get('sex', 'Unknown')}\n"
            f"History of present illness (OLD CARTS findings): {hpi_summary}\n"
            f"Top differential diagnoses: {ranking_text}\n\n"
            f"Format as clear bullet points following the required format. "
            f"Group OLD CARTS findings under 'History of present illness' - do NOT list each finding as a separate bullet starting with 'OLD CARTS:'."
        )
        
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.LLM_MAX_TOKENS_SUMMARY,
            temperature=self.LLM_TEMPERATURE_SUMMARY,
        )
        return response.strip() if response else "History collection complete."

    # ----------- Validation / Clarification ----------------------------------

    def _clean_llm_response(self, text: Optional[str], fallback: str = "") -> str:
        """Clean LLM response to extract only the question, removing all clinical notes and reasoning."""
        if not text:
            return fallback
        cleaned = text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```'):
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline+1:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3].strip()
        
        cleaned = cleaned.strip('"').strip("'")
        
        # Remove clinical notes, reasoning, and internal comments
        # Patterns to remove: "CLINICAL NOTE:", "Note:", "Reasoning:", etc.
        reasoning_prefixes = [
            r'CLINICAL NOTE:\s*',
            r'NOTE:\s*',
            r'Note:\s*',
            r'REASONING:\s*',
            r'Reasoning:\s*',
            r'This is clinical reasoning[^?]*\?',
            r'This is clinical reasoning[^.]*\.',
            r'Clinical reasoning[^?]*\?',
            r'Clinical reasoning[^.]*\.',
            r'Internal note[^?]*\?',
            r'Internal note[^.]*\.',
        ]
        for pattern in reasoning_prefixes:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove any text before the first question mark if it contains reasoning keywords
        # Split by newlines first to handle multi-line responses
        lines = cleaned.split('\n')
        question_lines = []
        found_question = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are clearly reasoning/notes (before question)
            if not found_question:
                reasoning_keywords = [
                    'clinical note', 'note:', 'reasoning:', 'this is clinical',
                    'internal reasoning', 'identify the next question',
                    'to identify', 'clinical reasoning'
                ]
                if any(keyword in line.lower() for keyword in reasoning_keywords):
                    # Check if this line itself contains a question
                    if '?' in line:
                        # Extract just the question part after the reasoning
                        # Try to find the actual question
                        parts = re.split(r'[?!]', line)
                        for part in parts:
                            if part.strip() and any(q_word in part.lower() for q_word in ['how', 'what', 'when', 'where', 'who', 'which', 'are', 'is', 'do', 'can']):
                                question_lines.append(part.strip() + '?')
                                found_question = True
                                break
                    continue  # Skip this reasoning line
            
            # If we found a question, include this line
            if found_question or '?' in line:
                question_lines.append(line)
                found_question = True
        
        # Reconstruct cleaned text from question lines
        if question_lines:
            cleaned = ' '.join(question_lines)
        
        # Split into sentences and find the question
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        question = None
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip sentences that are reasoning/notes
            if any(keyword in sentence.lower() for keyword in ['clinical note', 'this is clinical', 'to identify', 'clinical reasoning']):
                continue
            if sentence and sentence.endswith('?'):
                question = sentence
                break
            if sentence and any(word in sentence.lower() for word in ['how', 'what', 'when', 'where', 'who', 'which', 'are you', 'is this', 'do you', 'can you']):
                question = sentence.rstrip('.!') + '?'
                break
        
        if question:
            # Remove any remaining reasoning phrases
            reasoning_phrases = [
                r'now i have.*?which helps',
                r'thank you.*?which helps',
                r'for our records',
                r'this helps with',
                r'clinical note.*',
                r'this is clinical.*',
                r'to identify.*',
            ]
            for phrase in reasoning_phrases:
                question = re.sub(phrase, '', question, flags=re.IGNORECASE)
            question = re.sub(r'\s+', ' ', question).strip()
            return question or fallback
        
        # Last resort: try to extract any question-like content
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
        # Remove any remaining reasoning prefixes
        for pattern in reasoning_prefixes:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        return cleaned or fallback

    def _validate_age_answer(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        cleaned = text.strip()
        lowered = cleaned.lower()
        if not cleaned:
            return False, None, "I just need a number for your age. How old are you?"
        if lowered in {"unknown", "unsure", "not sure", "prefer not to say"}:
            return True, "unspecified", None
        digits = re.findall(r"\d{1,3}", cleaned)
        if digits:
            try:
                age_value = int(digits[0])
                if 0 < age_value <= 120:
                    return True, str(age_value), None
            except ValueError:
                pass
        return False, None, "Please enter your age as a number between 1 and 120. How old are you?"

    def _validate_sex_answer(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        cleaned = text.strip().lower()
        if not cleaned:
            return False, None, "For documentation, is your biological sex male, female, or intersex/non-binary?"

        mappings = {
            'male': {'male', 'm', 'man', 'boy'},
            'female': {'female', 'f', 'woman', 'girl'},
            'intersex/non-binary': {'intersex', 'nonbinary', 'non-binary', 'nb', 'enby'},
            'unspecified': {'prefer not to say', 'decline', 'unknown', 'unsure'},
        }

        for normalized, variants in mappings.items():
            if cleaned in variants:
                return True, normalized, None

        return False, None, (
            "Just to be sure, please tell me your biological sex as male, female, "
            "or intersex/non-binary."
        )

    # ----------- Utilities ----------------------------------------------------

    def _wrap_response(self, session: "MedicalSession", message: str, status: str = "question", metadata: Optional[Dict] = None) -> Dict[str, any]:
        return {
            'response': message,
            'status': status,
            'metadata': metadata or {},
            'debug': {
                'engine': self._format_engine_debug(session),
                'internal': self._captured_debug_output[-50:],
            },
        }

    def _get_or_create_session(self, session_id: str) -> "MedicalSession":
        if session_id not in self.sessions:
            self.sessions[session_id] = AdvancedMedicalNavigator.MedicalSession(session_id)
        return self.sessions[session_id]
    
    def _is_greeting(self, text: str) -> bool:
        if not text:
            return False
        normalized = re.sub(r"[^a-zA-Z\s]", "", text).strip().lower()
        greetings = {'hi', 'hello', 'hey', 'hey there', 'good morning', 'good afternoon', 'good evening'}
        return normalized in greetings
    
    def _is_medical_complaint(self, user_input: str) -> bool:
        """Check if user input contains a medical complaint or is just casual conversation"""
        user_lower = user_input.lower().strip()
        
        # Common greetings and casual phrases (not medical)
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                     'how are you', 'what\'s up', 'sup', 'greetings', 'hi there']
        
        # Check if it's just a greeting
        if user_lower in greetings or any(user_lower.startswith(g) for g in greetings):
            return False
        
        # Medical complaint indicators
        medical_keywords = [
            'pain', 'ache', 'hurt', 'sore', 'discomfort', 'symptom', 'problem', 'issue',
            'fever', 'nausea', 'vomit', 'dizzy', 'shortness', 'breath', 'cough', 'chest',
            'abdominal', 'headache', 'stomach', 'bleeding', 'blood', 'rash', 'swelling',
            'burning', 'pressure', 'tightness', 'numbness', 'tingling', 'weakness',
            'dizziness', 'fatigue', 'tired', 'unwell', 'sick', 'ill', 'feeling',
            'concerned about', 'worried about', 'having', 'experiencing', 'feeling'
        ]
        
        # Check if input contains medical keywords
        return any(keyword in user_lower for keyword in medical_keywords)
    
    def _is_confused_response(self, text: str) -> bool:
        """Check if user response is a confused/clarification request"""
        text_lower = text.lower().strip()
        confused_phrases = [
            'what', 'what?', 'huh', 'i don\'t understand', 'clarify',
            'what do you mean', 'what does that mean', 'i don\'t know',
            'not sure', 'unclear', 'confused', 'can you explain',
            'what are you asking', 'repeat', 'again', 'sorry',
            'could you', 'can you clarify', 'i\'m not sure',
            'like out of', 'scale of', 'do you mean'
        ]
        
        # Check for clarification patterns (e.g., "what do you mean? like out of a scale of 1-10?")
        if any(phrase in text_lower for phrase in confused_phrases):
            return True
        
        # Check if it's a question asking for clarification (ends with ? and contains clarification words)
        if text_lower.endswith('?') and any(word in text_lower for word in ['mean', 'asking', 'scale', 'rate', 'score']):
            return True
        
        return False
    
    def _validate_hpi_answer_appropriateness(self, session: "MedicalSession", field: str, answer: str, question: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the answer is appropriate for the OLD CARTS field being asked.
        Returns: (is_appropriate, clarification_message_if_needed)
        """
        # Quick heuristic check for common valid responses before LLM validation
        # This prevents false negatives for obviously valid answers
        answer_lower = answer.lower().strip()
        
        # For onset field, accept common time-related phrases immediately
        if field == 'onset':
            valid_time_phrases = [
                'this morning', 'today', 'yesterday', 'tomorrow',  # relative times
                'ago', 'hours ago', 'days ago', 'weeks ago', 'months ago',  # relative durations
                'last week', 'last month', 'last year',  # relative periods
                'when i woke up', 'woke up', 'when i got up', 'got up',  # wake-related
                'earlier', 'just now', 'a few', 'a couple', 'recently',  # vague but valid
                'morning', 'afternoon', 'evening', 'night',  # time of day
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',  # days of week
            ]
            # Check if answer contains any valid time phrase
            if any(phrase in answer_lower for phrase in valid_time_phrases):
                return True, None
            # Also accept if it starts with a number (e.g., "2 days ago", "3 hours ago")
            import re
            if re.match(r'^\d+\s+(hour|day|week|month|minute)', answer_lower):
                return True, None
        
        if not self.llm_chat_fn:
            # Fallback: basic heuristic check
            return self._heuristic_answer_validation(session, field, answer)
        
        # Use LLM to validate answer appropriateness
        # Get context of what's already been answered
        answered_fields = {}
        for key, value in session.context['hpi'].items():
            if value and value.strip():
                answered_fields[key] = value
        
        # Build validation prompt with field-specific guidance
        field_guidance = {
            'onset': "VALID answers include: time-related phrases like 'this morning', 'today', 'yesterday', '2 days ago', 'last week', 'a few hours ago', 'when I woke up', etc. These are ALL acceptable ways to describe when a symptom started.",
            'location': "VALID answers include: body locations like 'center of chest', 'upper right abdomen', 'left arm', 'behind the eyes', etc.",
            'character': "VALID answers include: descriptive words like 'sharp', 'dull', 'pressure', 'burning', 'stabbing', 'aching', etc.",
            'aggravating': "VALID answers include: activities or factors that worsen symptoms like 'walking', 'eating', 'lying down', 'deep breathing', etc.",
            'relieving': "VALID answers include: activities or factors that improve symptoms like 'resting', 'sitting up', 'taking medication', 'applying heat', etc.",
            'duration': "VALID answers include: how long symptoms last like 'constant', 'comes and goes', 'lasts 10 minutes', 'all day', etc.",
            'timing': "VALID answers include: patterns like 'constant', 'intermittent', 'only at night', 'comes and goes', etc.",
            'severity': "VALID answers include: numbers on a scale of 1-10, or descriptive words like 'mild', 'moderate', 'severe'.",
            'radiation': "VALID answers include: whether and where symptoms spread like 'yes, to my left arm', 'no', 'radiates to my jaw', etc.",
        }
        
        field_guidance_text = field_guidance.get(field, "The answer should directly answer what was asked.")
        
        validation_prompt = f"""You are a medical assistant validating patient answers.

Question asked: "{question}"
Field type: {field}
Patient's answer: "{answer}"

Previous answers provided:
{chr(10).join([f"- {k}: {v}" for k, v in answered_fields.items()])}

Determine if the patient's answer is appropriate for this question. 

{field_guidance_text}

The answer should:
- Directly answer what was asked
- Not be a duplicate of an answer already given for a different field
- Make logical sense for the question type

Examples of inappropriate answers:
- Answering "pressure" to "what makes it worse?" when "pressure" was already given as the character (what it feels like)
- Answering a location to a question about aggravating factors
- Answering the character description to a question about location

CRITICAL: For onset field, accept ANY time-related response including: "this morning", "today", "yesterday", "2 days ago", "last week", "a few hours ago", "when I woke up", "earlier", etc. These are ALL valid.

Return ONLY valid JSON: {{"appropriate": true/false, "reason": "brief reason"}}
If not appropriate, also suggest what type of answer would be expected."""
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": "You are a medical assistant. Return only valid JSON. No explanations outside JSON."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=150,
                temperature=0.0,
            )
            
            # Parse JSON response
            cleaned = response.strip()
            # Remove markdown code blocks if present
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline+1:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3].strip()
            
            # Extract JSON
            start_idx = cleaned.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned)):
                    if cleaned[i] == '{':
                        brace_count += 1
                    elif cleaned[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx]
            
            import json
            validation_result = json.loads(cleaned)
            is_appropriate = validation_result.get("appropriate", True)
            
            if not is_appropriate:
                reason = validation_result.get("reason", "This doesn't seem to answer the question asked.")
                # Generate clarification message
                clarification = self._generate_validation_clarification(field, question, answer, reason)
                self._capture_debug(f"[Validation] ❌ Answer '{answer}' not appropriate for {field}: {reason}")
                return False, clarification
            
            return True, None
            
        except Exception as e:
            self._capture_debug(f"[Validation] ⚠️ LLM validation failed, using heuristic: {e}")
            # Fallback to heuristic
            return self._heuristic_answer_validation(session, field, answer)
    
    def _heuristic_answer_validation(self, session: "MedicalSession", field: str, answer: str) -> Tuple[bool, Optional[str]]:
        """Heuristic-based validation as fallback when LLM validation fails."""
        answer_lower = answer.lower().strip()
        
        # Check if answer matches what was already given for another field
        for other_field, other_value in session.context['hpi'].items():
            if other_field != field and other_value and other_value.strip():
                other_lower = other_value.lower().strip()
                # If answer is very similar to a previous answer, it's likely inappropriate
                if answer_lower == other_lower and len(answer_lower) > 3:
                    # Check if fields are different types
                    if (field in ['character'] and other_field in ['aggravating', 'relieving']) or \
                       (field in ['aggravating', 'relieving'] and other_field in ['character']):
                        # Generate clarification based on field
                        clarification = self._generate_validation_clarification(field, "", answer, "This seems like a repeat of a previous answer.")
                        return False, clarification  # Likely repeating answer for wrong field
                    if (field in ['location'] and other_field not in ['location', 'radiation']) or \
                       (field not in ['location', 'radiation'] and other_field == 'location'):
                        if answer_lower in other_lower or other_lower in answer_lower:
                            clarification = self._generate_validation_clarification(field, "", answer, "This doesn't seem to match the question type.")
                            return False, clarification  # Location answer used for non-location question
        
        return True, None
    
    def _generate_validation_clarification(self, field: str, question: str, answer: str, reason: str) -> str:
        """Generate a clarification message when answer is inappropriate."""
        # Create field-specific clarification prompts
        field_clarifications = {
            'aggravating': "I need to know what makes your symptoms worse. For example: activities like walking or exercising, positions like lying down, eating certain foods, etc. What makes it worse?",
            'relieving': "I need to know what makes your symptoms better. For example: resting, sitting up, taking medication, applying heat or ice, etc. What makes it better?",
            'location': "I need to know where exactly the symptom is located on your body. For example: 'center of chest', 'upper right abdomen', 'left arm', etc. Where exactly is it located?",
            'character': "I need to know what the symptom feels like. For example: sharp, dull, pressure, burning, stabbing, aching, etc. What does it feel like?",
            'radiation': "I need to know if the symptom spreads or radiates to other areas. For example: 'radiates to left arm', 'spreads to jaw', 'goes down my leg', etc. Does it spread or radiate anywhere?",
            'onset': "I need to know when the symptom started. For example: '2 days ago', 'this morning', 'last week', etc. When did it start?",
            'duration': "I need to know how long the symptom lasts. For example: 'constant', 'comes and goes', 'lasts 10 minutes', etc. How long does it last?",
            'timing': "I need to know if the symptom is constant or intermittent. For example: 'constant', 'comes and goes', 'only at night', etc. Is it constant or does it come and go?",
            'severity': "I need a number on a scale from 1 to 10, where 1 is very mild and 10 is the worst possible. How severe is it?",
            'associated': "I need to know about any other symptoms you're experiencing. What other symptoms are you having?",
        }
        
        # Use field-specific clarification if available
        if field in field_clarifications:
            return field_clarifications[field]
        
        # Generic clarification
        return f"I'm asking about {field}. {reason} Could you please provide a different answer? {question}"

    # ----------- Debug helpers ----------------------------------------------

    def _capture_debug(self, message: str) -> None:
        self._captured_debug_output.append(message)
    
    def _format_engine_debug(self, session: "MedicalSession") -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("[Engine] 🧠 ENGINE DEBUG OUTPUT")
        lines.append("=" * 80)
        lines.append(f"[Engine] 🎯 Conditions: Active={len(session.active_conditions)}, Reserve={len(session.reserve_conditions)}")
        
        pre_filled, pre_missing = self._get_pre_hpi_status(session)
        lines.append(f"[Engine] 👤 Demographics collected: {', '.join(pre_filled) if pre_filled else 'none'}")
        lines.append(f"[Engine] 📝 Demographics missing: {', '.join(pre_missing) if pre_missing else 'none'}")

        satisfied, missing, current = self._get_oldcarts_status(session)
        coverage = ''.join(e[0].upper() if e in satisfied else '_' for e in self.HPI_ELEMENTS)
        lines.append(f"[Engine] 📋 OLDCARTS: {coverage} ({len(satisfied)}/{len(self.HPI_ELEMENTS)})")
        lines.append(f"[Engine] ✅ Satisfied: {', '.join(satisfied) if satisfied else 'none'}")
        lines.append(f"[Engine] ❔ Missing: {', '.join(missing) if missing else 'none'}")
        if current:
            lines.append(f"[Engine] 🔍 Currently asking: {current}")
        
        lines.append(self._format_rankings_debug(session))
        return '\n'.join(lines)

    def _format_rankings_debug(self, session: "MedicalSession") -> str:
        lines = []
        lines.append("[Engine] 📊 UPDATED RANKINGS:")
        for idx, (name, score) in enumerate(session.active_conditions[:5], start=1):
            pct = round(score * 100, 1)
            lines.append(f"[Engine]   {idx}. {name}: {pct}% 📋")
        lines.append("")
        lines.append(f"[Engine] 🔄 Pool status: Active={len(session.active_conditions)}, Reserve={len(session.reserve_conditions)}, Ruled out=0")
        return '\n'.join(lines)

    def _update_condition_pools(self, session: "MedicalSession") -> None:
        """Update active and reserve condition pools based on rankings."""
        active = session.condition_rankings[:5]
        reserve = session.condition_rankings[5:]
        previous_active = session.previous_active
        current_active = {name for name, _ in active}
        promotions = current_active - previous_active
        demotions = previous_active - current_active

        if promotions:
            self._capture_debug(f"\n[Pool] 🔼 PROMOTED to active ({len(promotions)}):")
            for name in promotions:
                score = next((score for cond, score in active if cond == name), 0.0)
                pct = round(score * 100, 1)
                self._capture_debug(f"[Pool]   ↑ {name}: {pct:.1%}")

        if demotions:
            self._capture_debug(f"[Pool] 🔽 DEMOTED to reserve ({len(demotions)}):")
            for name in demotions:
                score = next((score for cond, score in reserve if cond == name), 0.0)
                pct = round(score * 100, 1)
                self._capture_debug(f"[Pool]   ↓ {name}: {pct:.1%}")

        session.active_conditions = active
        session.reserve_conditions = reserve
        session.previous_active = current_active
        
        # Print pool status
        self._capture_debug(f"\n[Pool] 📊 Condition Pool Status:")
        self._capture_debug(f"[Pool]   Total conditions: {len(session.condition_scores)}")
        self._capture_debug(f"[Pool]   Active (top 5): {len(session.active_conditions)}")
        self._capture_debug(f"[Pool]   Reserve: {len(session.reserve_conditions)}")
    
    def _check_for_missing_conditions(self, session: "MedicalSession", element: str, answer: str) -> None:
        """Check if LLM should consider additional conditions based on answer."""
        # Only check for missing conditions on key elements that might reveal new diagnoses
        if element not in ['character', 'aggravating', 'relieving', 'location']:
            return
        
        chief_complaint = session.context['pre_hpi'].get('chief_complaint', '')
        current_conditions = list(session.condition_scores.keys())
        
        if not self.llm_chat_fn:
            return
        
        # Ask LLM if there are other conditions that should be considered
        system_prompt = (
            "You are a medical expert. Based on the patient's answer, identify if there are "
            "other medical conditions that should be considered in the differential diagnosis.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- If the patient's answer is unclear, nonsensical, or doesn't make logical sense, return empty list: {\"additional_conditions\": []}\n"
            "- Never invent medical conditions that don't logically relate to the patient's answer.\n\n"
            "Return ONLY valid JSON: {\"additional_conditions\": [\"Condition 1\", \"Condition 2\", ...]}\n"
            "If no additional conditions are needed, return empty list: {\"additional_conditions\": []}\n"
            "No explanations, no other text. Just the JSON object."
        )
        
        user_prompt = (
            f"Chief complaint: {chief_complaint}\n"
            f"OLD CARTS element: {element}\n"
            f"Patient's answer: '{answer}'\n"
            f"Currently evaluating: {', '.join(current_conditions[:5])}{'...' if len(current_conditions) > 5 else ''}\n\n"
            "Are there other medical conditions that should be considered based on this answer? "
            "For example, if chest pain is 'burning' and 'worse after meals', consider GERD. "
            "If no additional conditions, return empty list."
        )
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            
            if response:
                # Parse JSON
                cleaned = response.strip()
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline+1:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3].strip()
                
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        cleaned = cleaned[start_idx:end_idx]
                
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and 'additional_conditions' in parsed:
                        additional = parsed['additional_conditions']
                        if isinstance(additional, list) and additional:
                            # Add new conditions at baseline
                            added_count = 0
                            for cond in additional:
                                if cond not in session.condition_scores:
                                    session.condition_scores[cond] = 0.5
                                    added_count += 1
                                    self._capture_debug(f"[Pool] 🆕 LLM suggested additional condition: {cond}")
                            if added_count > 0:
                                self._capture_debug(f"[Pool] ✅ Added {added_count} new condition(s) to evaluation pool")
                                self._apply_rule_outs(session)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            # Silently fail - not critical
            pass

    def _log_rankings(self, session: "MedicalSession") -> None:
        self._capture_debug("\n[Engine] 📊 UPDATED RANKINGS:")
        for idx, (name, score) in enumerate(session.active_conditions[:5], start=1):
            pct = round(score * 100, 1)
            self._capture_debug(f"[Engine]   {idx}. {name}: {pct}% 📋")

    def _get_oldcarts_status(self, session: "MedicalSession") -> Tuple[List[str], List[str], Optional[str]]:
        hpi_answers = session.context.get('hpi', {})
        satisfied = [element for element in self.HPI_ELEMENTS if element in hpi_answers and hpi_answers[element]]
        missing = [element for element in self.HPI_ELEMENTS if element not in hpi_answers or not hpi_answers[element]]
        current = None
        if session.pending and session.pending.get('section') == 'hpi':
            current = session.pending.get('field')
        return satisfied, missing, current

    def _get_pre_hpi_status(self, session: "MedicalSession") -> Tuple[List[str], List[str]]:
        pre = session.context.get('pre_hpi', {})
        collected = [item for item in self.PRE_HPI_ORDER if pre.get(item)]
        missing = [item for item in self.PRE_HPI_ORDER if item not in collected]
        return collected, missing
