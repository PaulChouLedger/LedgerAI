#!/usr/bin/env python3
"""
Advanced Medical Navigator (LLM-only, Guideline-aware)
======================================================

Conversation flow:
    1. Capture chief complaint → LLM-based matching to medical categories
       • LLM matches chief complaint to guideline categories using medical knowledge
       • Loads guideline categories & seeds condition scores.
    2. LLM empathetic acknowledgement + chronicity question (new vs known w/ prior Dx)
    3. Collect demographics: age, biological sex
    4. OLDCARTS assessment using guideline terms & weights per category
       • LLM crafts questions with injected options
       • Responses scored via LLM against patient-friendly terms
       • Clarifying questions generated when multiple / no matches
    5. Rankings update after every element; diagnosis ready once OLDCARTS complete.

This file uses LLM-only approach for all matching and scoring. No FAISS or medical_rule_engine
is required - LLM handles all semantic matching, fuzzy correction, and anatomical reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import random
from typing import Dict, List, Optional, Tuple, Any


class AdvancedMedicalNavigator:
    """LLM-driven navigator augmented with guideline intelligence."""

    # ----------- Configuration -------------------------------------------------

    PRE_HPI_ORDER = ["chief_complaint", "chronicity", "age", "sex"]
    PRE_HPI_PROMPTS = {
        "chronicity": "Determine if the problem is new or ongoing and whether there is a prior diagnosis.",
        "age": "Ask for the patient's age (single number).",
        "sex": "Ask for the patient's biological sex for medical documentation.",
    }

    HPI_ELEMENTS = [
        "onset",
        "abruptness",
        "location",
        "duration",
        "frequency",
        "character",
        "aggravating",
        "relieving",
        "timing",
        "severity",
        "associated",
        "red_flags",
    ]

    HPI_BASE_GUIDANCE = {
        "onset": "When did this {cc} start?",
        "abruptness": "Did it come on suddenly or build up gradually?",
        "location": "Where exactly is your {cc} located?",
        "duration": "How long does each episode typically last?",
        "frequency": "Is it constant or does it come and go?",
        "character": "How would you describe what it feels like?",
        "aggravating": "What tends to make it worse?",
        "relieving": "What tends to make it better?",
        "timing": "Does it come on at specific times or during certain activities?",
        "severity": "On a scale from 1 to 10, how bad is it?",
        "associated": "Have you noticed any other symptoms along with it?",
        "red_flags": "Have you experienced any urgent warning signs?",
    }

    CATEGORY_TO_SYSTEM = {
        'gastrointestinal': 'GI',
        'cardiovascular': 'CARDIO',
        'respiratory': 'PULMONARY',
        'neurological': 'NEURO',
        'musculoskeletal': 'MSK',
        'renal': 'RENAL',
        'genitourinary': 'GU',
        'gynecological': 'GYN',
        'dermatological': 'DERM',
    }

    # LLM-only matching thresholds
    CHIEF_COMPLAINT_LLM_THRESHOLD = 0.5  # Minimum LLM confidence for category match
    RULE_OUT_THRESHOLD = 0.05
    LLM_MATCH_ACCEPT_THRESHOLD = 0.5  # Minimum LLM score for term match

    PMH_ELEMENTS = ["pmh", "psh", "meds_allergies"]
    PMH_PROMPTS = {
        "pmh": "Do you have any existing medical conditions?",
        "psh": "Have you had any surgeries in the past?",
        "meds_allergies": "What medications do you take, and do you have any medication allergies?",
    }

    QUESTION_SYSTEM_PROMPT = (
        "You are a compassionate medical assistant conducting a medical interview."
        " Use the guidance to craft one concise, patient-friendly question."
        " Return ONLY the question text (no prefixes, no reasoning)."
        " Keep it ≤20 words and honor any provided options."
        " Do NOT include phrases like 'Here is a friendly question' or similar meta commentary."
    )

    EMPATHETIC_SYSTEM_PROMPT = (
        "You are an empathetic medical assistant. Craft a 1–2 sentence acknowledgment"
        " that validates the patient's concern and expresses willingness to help."
        " Do not mention checking vitals, running tests, or any clinical actions—you are offering"
        " emotional support only. Provide a single compassionate sentence and do not ask follow-up questions."
    )

    CHRONICITY_SYSTEM_PROMPT = (
        "You are a medical assistant. Ask the patient directly whether the problem is new or ongoing and"
        " whether a prior diagnosis exists. Respond with exactly one simple question addressed to the patient."
        " Do not mention laboratory tests, vital signs, additional steps, or introduce the question with phrases like"
        " 'Here is the question'."
    )

    SUMMARY_SYSTEM_PROMPT = (
        "You are a clinical assistant. Produce ≤6 bullet points summarising demographics, chief complaint,"
        " focused OLDCARTS facts, PMH/meds/allergies, and top ranked differentials with urgency."
    )

    GREETING_RESPONSES = (
        "Hi there! I'm here to help. What symptoms are you experiencing today?",
        "Hello! Let me know what brings you in today so I can assist.",
    )

    # Weights shared with adaptive engine for consistency
    CATEGORY_ELEMENT_WEIGHTS = {
        'gastrointestinal': {
            'location': 0.65,
            'character': 0.25,
            'aggravating': 0.40,
            'relieving': 0.40,
            'onset': 0.31,
            'abruptness': 0.30,
            'timing': 0.30,
            'duration': 0.29,
            'frequency': 0.29,
            'severity': 0.20,
            'associated': 0.5,
        },
        'cardiovascular': {
            'character': 0.65,
            'location': 0.30,
            'aggravating': 0.65,
            'relieving': 0.35,
            'onset': 0.30,
            'timing': 0.25,
            'duration': 0.30,
            'severity': 0.25,
            'associated': 0.30,
            'abruptness': 0.30,
            'frequency': 0.28,
        },
        'respiratory': {
            'character': 0.35,
            'location': 0.30,
            'aggravating': 0.30,
            'relieving': 0.25,
            'onset': 0.30,
            'timing': 0.25,
            'duration': 0.25,
            'severity': 0.25,
            'associated': 0.35,
            'abruptness': 0.28,
            'frequency': 0.26,
        },
    }

    DEFAULT_ELEMENT_WEIGHT = 0.30
    CLEAR_LEAD_MARGIN = 0.08

    # Auto-selection: if one match is clearly best, use it without clarification
    AUTO_SELECT_BEST_THRESHOLD = 0.95  # Best score must be >= 0.95
    AUTO_SELECT_MARGIN = 0.25  # Best must be >= 0.25 higher than next best
    AUTO_SELECT_NEXT_MAX = 0.7  # Or next best must be < 0.7
    LOCATION_HIGH_CONFIDENCE_THRESHOLD = 0.9  # High confidence LLM score for location matching

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
            'guideline_terms': {},
            'guideline_includes': {},
            'matched_categories': [],
            'clarifications': {},
            'associated_state': {},
            'red_flag_state': {},
            'debug': {},
        })
        condition_scores: Dict[str, float] = field(default_factory=dict)
        condition_rankings: List[Tuple[str, float]] = field(default_factory=list)
        active_conditions: List[Tuple[str, float]] = field(default_factory=list)
        reserve_conditions: List[Tuple[str, float]] = field(default_factory=list)
        previous_active: set = field(default_factory=set)
        oldcarts_remaining: List[str] = field(default_factory=list)
        completed: bool = False
        last_field: Optional[str] = None

    def _extract_chief_complaint_descriptors(self, complaint: str) -> Dict[str, bool]:
        complaint_lower = (complaint or "").lower()
        keywords = {
            "sensory": ["ache", "aching", "pain", "sharp", "stabbing", "burning", "tender", "sore", "cramping", "cramp", "tingling", "numb"],
            "visual": ["bleeding", "blood", "color", "discolor", "rash", "lesion", "swelling", "bruise", "ooze", "discharge"],
        }
        descriptors = {"sensory": False, "visual": False}
        for tag, words in keywords.items():
            if any(word in complaint_lower for word in words):
                descriptors[tag] = True
        return descriptors

    # ----------- Lifecycle ----------------------------------------------------

    def __init__(self, llm_chat_fn):
        """
        Initialize Advanced Medical Navigator with LLM-only approach.
        
        Args:
            llm_chat_fn: LLM chat function for all matching and scoring
        """
        self.llm_chat_fn = llm_chat_fn
        self.sessions: Dict[str, AdvancedMedicalNavigator.MedicalSession] = {}
        self._captured_debug_output: List[str] = []
        self.guidelines_dir = self._resolve_guidelines_dir()
        self.enabled_categories = self._get_enabled_categories()
        self.all_guidelines: Dict[str, Dict] = {}
        self.chief_complaint_triggers_data: List[Dict] = []  # For LLM matching
        self._chief_complaint_condition_seed: Dict[str, float] = {}

        if self.guidelines_dir:
            self._load_guidelines()
            self._build_chief_complaint_triggers()
        else:
            self._capture_debug("[Navigator] ⚠️ No guidelines directory found. Chief complaint matching may be limited.")

    # ----------- Public API ---------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> Dict[str, any]:
        self._captured_debug_output = []
        session = self._get_or_create_session(session_id)
        session.messages.append({"role": "user", "content": user_message})
        if len(session.messages) > 50:
            session.messages = session.messages[-50:]

        if session.stage == "awaiting_chief_complaint":
            return self._handle_initial_complaint(session, user_message)

        if session.pending:
            response = self._store_answer(session, session.pending, user_message)
            if response:
                return response

        if session.completed:
            follow_up = "Thanks for the update. If anything changes, let me know."
            session.messages.append({"role": "assistant", "content": follow_up})
            return self._wrap_response(session, follow_up, status="complete")

        next_prompt = self._determine_next_question(session)
        if next_prompt:
            session.pending = next_prompt
            session.messages.append({"role": "assistant", "content": next_prompt['prompt']})
            return self._wrap_response(session, next_prompt['prompt'], metadata={
                'section': session.stage,
                'field': next_prompt['field'],
            })

        summary = self._generate_summary(session)
        session.completed = True
        session.messages.append({"role": "assistant", "content": summary})
        return self._wrap_response(session, summary, status="complete", metadata={'summary': True})

    # ----------- Stage handlers ----------------------------------------------

    def _handle_initial_complaint(self, session: "MedicalSession", text: str) -> Dict[str, any]:
        if self._is_greeting(text):
            reply = self.GREETING_RESPONSES[0]
            self._capture_debug(f"[Navigator] 🙋 Greeting detected: '{text}'")
            return self._wrap_response(session, reply, status="awaiting_chief_complaint")

        # LLM-only approach: no FAISS, no medical_rule_engine, no embedding_model
        # LLM handles fuzzy correction, semantic matching, and category assignment
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (ADVANCED NAVIGATOR - LLM-ONLY)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{text}'")
        
        categories = self._match_chief_complaint_to_category_llm(text)
        if not categories:
            apology = (
                "I'm not sure I caught that. Could you tell me a bit more about what's bothering you, "
                "like 'I have stomach pain' or 'I'm feeling short of breath'?"
            )
            self._capture_debug(
                f"[Engine] ❌ Unable to match chief complaint '{text}' to guidelines. Requesting clarification."
            )
            session.stage = "awaiting_chief_complaint"
            return self._wrap_response(session, apology, status="awaiting_chief_complaint")
        
        session.context['matched_categories'] = categories
        primary_category = categories[0] if categories else 'gastrointestinal'
        if len(categories) == 1:
            self._capture_debug(f"[Engine] 🎯 Category: {primary_category}")
        else:
            self._capture_debug(f"[Engine] 🎯 Categories: {', '.join(categories)}")

        seed_scores = self._chief_complaint_condition_seed or {}
        session.condition_scores = {}

        self._capture_debug(
            f"[Engine] 🧾 Chief complaint seed map: { {k: round(v, 3) for k, v in seed_scores.items()} }"
            if seed_scores else "[Engine] 🧾 Chief complaint seed map: {}"
        )

        for cond in self._get_conditions_for_categories(categories):
            base = seed_scores.get(cond)
            if base is not None:
                seeded_value = max(0.5, float(base))
                session.condition_scores[cond] = seeded_value
                self._capture_debug(f"[Engine] 🔧 Seeded {cond} at {seeded_value:.3f} from chief complaint match")
            else:
                session.condition_scores[cond] = 0.5
                self._capture_debug(f"[Engine] 🔧 Seeded {cond} at baseline 0.500 (no chief complaint match)")

        if seed_scores:
            seeded_sorted = sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)
            top_preview = ', '.join(f"{name}: {score:.3f}" for name, score in seeded_sorted[:5])
            self._capture_debug(f"[Engine] 📊 Chief complaint seeding (top {min(5, len(seeded_sorted))}): {top_preview}")
        else:
            self._capture_debug(f"[Engine] 📋 No direct chief complaint guideline matches; seeded {len(session.condition_scores)} conditions at baseline 50.0%")

        self._apply_rule_outs(session)

        session.stage = "awaiting_chronicity"
        session.context['pre_hpi']['chief_complaint'] = text
        session.context['guideline_terms']['chief_complaint_terms'] = self._get_element_includes(session, 'chief_complaint') if hasattr(self, '_get_element_includes') else []
        session.context['guideline_terms']['chief_complaint_descriptors'] = self._extract_chief_complaint_descriptors(text)

        empathetic = self._generate_empathetic_statement(text)
        chronicity_prompt = self._generate_chronicity_question()

        session.pending = {
            'section': 'pre_hpi',
            'field': 'chronicity',
            'guidance': self.PRE_HPI_PROMPTS['chronicity'],
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
                prompt = "Thank you. For our records, how old are you?"
                return {'section': 'pre_hpi', 'field': 'age', 'prompt': prompt, 'guidance': self.PRE_HPI_PROMPTS['age']}
            session.stage = "awaiting_sex"

        if session.stage == "awaiting_sex":
            if session.context['pre_hpi'].get('sex'):
                session.stage = "hpi"
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
            else:
                prompt = "And for medical documentation, what is your biological sex?"
                return {'section': 'pre_hpi', 'field': 'sex', 'prompt': prompt, 'guidance': self.PRE_HPI_PROMPTS['sex']}

        if session.stage == "hpi":
            return self._next_oldcarts_question(session)

        if session.stage == "pmh":
            for field in self.PMH_ELEMENTS:
                if field not in session.context['pmh']:
                    prompt = self._generate_question(session, 'pmh', field, self.PMH_PROMPTS[field])
                    return {'section': 'pmh', 'field': field, 'prompt': prompt, 'guidance': self.PMH_PROMPTS[field]}
            session.stage = "complete"
            return None

        # Skip pre-HPI checks if we're already past pre-HPI stage
        if session.stage not in {"awaiting_chronicity", "awaiting_age", "awaiting_sex", "pre_hpi"}:
            return None

        remaining_pre_hpi = [field for field in self.PRE_HPI_ORDER if not session.context['pre_hpi'].get(field)]
        if remaining_pre_hpi:
            next_field = remaining_pre_hpi[0]
            if next_field == 'chronicity':
                session.stage = "awaiting_chronicity"
                prompt = self._generate_chronicity_question()
                return {
                    'section': 'pre_hpi',
                    'field': 'chronicity',
                    'prompt': prompt,
                    'guidance': self.PRE_HPI_PROMPTS['chronicity'],
                }
            if next_field == 'age':
                session.stage = "awaiting_age"
                prompt = "Thank you. For our records, how old are you?"
                return {
                    'section': 'pre_hpi',
                    'field': 'age',
                    'prompt': prompt,
                    'guidance': self.PRE_HPI_PROMPTS['age'],
                }
            if next_field == 'sex':
                session.stage = "awaiting_sex"
                prompt = "And for medical documentation, what is your biological sex?"
                return {
                    'section': 'pre_hpi',
                    'field': 'sex',
                    'prompt': prompt,
                    'guidance': self.PRE_HPI_PROMPTS['sex'],
                }

        return None

    def _next_oldcarts_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        if not session.oldcarts_remaining:
            session.stage = "pmh"
            return self._determine_next_question(session)

        element = None
        answered = {key for key, value in session.context['hpi'].items() if value}
        while session.oldcarts_remaining:
            candidate = session.oldcarts_remaining.pop(0)
            if candidate in answered:
                continue
            element = candidate
            break
        if element is None:
            session.stage = "pmh"
            return self._determine_next_question(session)
        cc_subject = self._normalize_subject_for_questions(session.context['pre_hpi'].get('chief_complaint'))
        session.last_field = element

        if element == 'associated':
            question_info = self._prepare_next_associated_question(session)
            if not question_info:
                session.context['hpi']['associated'] = 'none reported'
                return self._next_oldcarts_question(session)
            return question_info

        if element == 'red_flags':
            question_info = self._prepare_next_red_flag_question(session)
            if not question_info:
                session.context['hpi']['red_flags'] = 'none reported'
                return self._next_oldcarts_question(session)
            return question_info

        guidance, base_question, options = self._build_oldcarts_guidance(session, element, cc_subject)
        prompt = self._generate_question(
            session=session,
            section='hpi',
            field=element,
            guidance=guidance,
            base_question=base_question,
            options=options,
        )
        return {
            'section': 'hpi',
            'field': element,
            'prompt': prompt,
            'guidance': guidance,
            'base_question': base_question,
            'options': options,
        }

    # ----------- Answer persistence & scoring --------------------------------

    def _store_answer(self, session: "MedicalSession", pending: Dict[str, str], answer: str) -> Optional[Dict[str, Any]]:
        section, field = pending['section'], pending['field']
        text = answer.strip()
        if section == 'pre_hpi':
            debug_ctx = session.context.setdefault('debug', {})
            validation_message = None
            normalized_value = text
            if field == 'age':
                valid, normalized_value, validation_message = self._validate_age_answer(text)
                if not valid:
                    debug_ctx['last_validation_error'] = validation_message
                    self._capture_debug(f"[Pre-HPI] ❌ Invalid age response '{text}'")
                    session.pending = pending
                    session.messages.append({"role": "assistant", "content": validation_message})
                    return self._wrap_response(
                        session,
                        validation_message,
                        status="validation_error",
                        metadata={'field': field, 'section': section, 'validation_error': True},
                    )
                debug_ctx.pop('last_validation_error', None)
                self._capture_debug(f"[Pre-HPI] ✅ Recorded age: {normalized_value}")
            elif field == 'sex':
                valid, normalized_value, validation_message = self._validate_sex_answer(text)
                if not valid:
                    debug_ctx['last_validation_error'] = validation_message
                    self._capture_debug(f"[Pre-HPI] ❌ Invalid sex response '{text}'")
                    session.pending = pending
                    session.messages.append({"role": "assistant", "content": validation_message})
                    return self._wrap_response(
                        session,
                        validation_message,
                        status="validation_error",
                        metadata={'field': field, 'section': section, 'validation_error': True},
                    )
                debug_ctx.pop('last_validation_error', None)
                self._capture_debug(f"[Pre-HPI] ✅ Recorded sex: {normalized_value}")
            else:
                debug_ctx.pop('last_validation_error', None)
            session.context['pre_hpi'][field] = normalized_value
            if field == 'chronicity':
                session.stage = "awaiting_age"
            elif field == 'age':
                session.stage = "awaiting_sex"
            elif field == 'sex':
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
            session.pending = None
            return None
        elif section == 'hpi':
            if self._is_confused_response(text):
                clarification = self._clarify_element_question(session, field, pending)
                session.pending = pending
                return self._wrap_response(session, clarification, metadata={
                    'section': 'hpi',
                    'field': field,
                    'clarification': True,
                })

            session.context['hpi'][field] = text
            response = self._score_oldcarts_answer(session, pending, field, text)
            if response:
                return response
            if field in session.oldcarts_remaining:
                session.oldcarts_remaining = [e for e in session.oldcarts_remaining if e != field]
            session.pending = None
        elif section == 'pmh':
            session.context['pmh'][field] = text
            session.pending = None
        return None

    def _score_oldcarts_answer(
        self,
        session: "MedicalSession",
        pending: Dict[str, str],
        element: str,
        answer: str,
    ) -> Optional[Dict[str, Any]]:
        # LLM-only approach: no medical_rule_engine, no embedding_model, no FAISS
        # LLM handles all matching, fuzzy correction, and anatomical reasoning
        requires_clarification = self._requires_clarification(element)

        sequence_result: Optional[Dict[str, Any]] = None
        follow_up_question: Optional[Dict[str, Any]] = None
        answer_to_score: Optional[str] = answer
        if element == 'associated':
            sequence_result = self._handle_associated_answer(session, answer)
        elif element == 'red_flags':
            sequence_result = self._handle_red_flag_answer(session, answer)

        if sequence_result is not None:
            answer_to_score = sequence_result.get('score_text')
            follow_up_question = sequence_result.get('next_question')
            positives = sequence_result.get('positives', [])
            if element == 'associated':
                if positives:
                    session.context['hpi']['associated'] = ', '.join(positives)
                if sequence_result.get('completed'):
                    session.context['hpi']['associated'] = ', '.join(positives) if positives else 'none reported'
            elif element == 'red_flags':
                if positives:
                    session.context['hpi']['red_flags'] = ', '.join(positives)
                if sequence_result.get('completed'):
                    session.context['hpi']['red_flags'] = ', '.join(positives) if positives else 'none reported'

            if answer_to_score is None:
                if follow_up_question:
                    session.pending = follow_up_question
                    session.messages.append({"role": "assistant", "content": follow_up_question['prompt']})
                    return self._wrap_response(
                        session,
                        follow_up_question['prompt'],
                        metadata=follow_up_question,
                    )
                return None

            answer = answer_to_score

        scoring_element = element
        if sequence_result is not None and element == 'red_flags':
            source_element = sequence_result.get('source_element')
            if source_element:
                scoring_element = source_element

        # LLM-only approach: get ALL terms from guidelines and let LLM decide which match
        all_element_terms = self._get_all_terms_for_element(session, scoring_element)
        if all_element_terms:
            # LLM reviews all terms - no pre-filtering needed
            matches = all_element_terms
            term_scores = {term: 0.0 for term in all_element_terms}  # Placeholder, LLM will set actual scores
            self._capture_debug(f"[LLM] 🔍 {scoring_element}: Sending {len(all_element_terms)} terms to LLM for review")
            self._capture_debug(f"[LLM] 🔍 {scoring_element}: Terms: {all_element_terms}")
        else:
            # Fallback: no terms found in guidelines (shouldn't happen, but handle gracefully)
            self._capture_debug(f"[LLM] ⚠️ {scoring_element}: No terms found in guidelines")
            matches = []
            term_scores = {}

        # LLM threshold is 0.5 for all elements (LLM decides what matches)
        llm_threshold = 0.5
        matches, term_scores, review_rows, review_meta = self._llm_refine_matches(
            session,
            scoring_element,
            answer,
            matches,
            term_scores,
            llm_threshold,  # Use LLM threshold for all elements
        )
        debug_ctx = session.context.setdefault('debug', {})
        if review_meta.get('invoked'):
            self._log_llm_match_review(scoring_element, answer, review_rows)
            if not review_rows:
                self._capture_debug(f"[LLM] ⚠️ Match review completed but no review rows generated for '{answer}' in {scoring_element}")
            debug_ctx['last_llm_review'] = {
                'element': scoring_element,
                'answer': answer,
                'rows': review_rows,
                'requested_terms': review_meta.get('requested_terms', []),
                'raw_response': review_meta.get('raw_response'),
                'had_scores': review_meta.get('had_scores', False),
            }
        else:
            reason = review_meta.get('reason', 'unknown')
            self._capture_debug(f"[LLM] ⚠️ Match review not invoked for '{answer}' in {scoring_element}: {reason}")
            debug_ctx.pop('last_llm_review', None)

        if pending and pending.get('clarification'):
            session.context['clarifications'].pop(element, None)
        elif not requires_clarification:
            session.context['clarifications'].pop(element, None)

        analysis = None
        if scoring_element == 'location':
            analysis = self._analyze_location_answer(session, element, answer, matches, term_scores)
            self._log_location_analysis(session, analysis)
            if analysis:
                boosted_matches = analysis.get('boosted_matches')
                if boosted_matches:
                    matches = boosted_matches
                boosted_scores = analysis.get('boosted_term_scores')
                if boosted_scores:
                    term_scores = boosted_scores
        else:
            self._log_llm_scores(scoring_element, answer, matches, term_scores)

        # LLM-only approach: no term_embeddings or synonym mapping needed - LLM handles synonym matching
        # Build condition_similarities directly from term_scores and guideline terms
        condition_similarities: Dict[str, float] = {}

        for term in matches:
            score = term_scores.get(term)
            if score is None:
                score = term_scores.get(term.lower())
            if score is None:
                continue
            
            # Find which conditions this term belongs to from guidelines
            conditions = []
            for guideline_name, guideline in self.all_guidelines.items():
                structured = self._structured_oldcarts(guideline)
                element_data = structured.get(scoring_element, {})
                includes = element_data.get('includes', []) if isinstance(element_data, dict) else []
                for item in includes:
                    if isinstance(item, dict):
                        patient_term = item.get('patient_friendly', '')
                    else:
                        patient_term = item
                    if isinstance(patient_term, str) and patient_term.lower() == term.lower():
                        conditions.append(guideline_name)
                        break
            if not conditions:
                    continue
                
            for cond in conditions:
                prev = condition_similarities.get(cond, 0.0)
                condition_similarities[cond] = max(prev, score)

        if analysis and requires_clarification:
            clarification_pending = pending.get('clarification') if pending else False
            clarification = self._maybe_request_location_clarification(
                session=session,
                element=element,
                answer=answer,
                analysis=analysis,
                clarification_just_asked=clarification_pending,
            )
            if clarification:
                session.pending = clarification
                session.messages.append({"role": "assistant", "content": clarification['prompt']})
                return self._wrap_response(
                    session,
                    clarification['prompt'],
                    metadata={
                        'section': 'hpi',
                        'field': element,
                        'clarification': True,
                        'mode': clarification.get('mode'),
                    },
                )

        if not condition_similarities:
            self._capture_debug(f"[Scoring] ⚪ No guideline matches for {element} → '{answer}'")
            self._apply_rule_outs(session)
            return None

        weight = self._get_element_weight(session, element)
        for cond, similarity in condition_similarities.items():
            prior = session.condition_scores.get(cond, 0.5)
            blended = prior + weight * (similarity - prior)
            session.condition_scores[cond] = blended
            self._capture_debug(
                f"[Scoring] 📊 {cond}: old={prior:.3f}, similarity={similarity:.3f}, weight={weight:.2f}, new={blended:.3f}"
            )

        self._apply_rule_outs(session)

        if follow_up_question:
            session.pending = follow_up_question
            session.messages.append({"role": "assistant", "content": follow_up_question['prompt']})
            return self._wrap_response(
                session,
                follow_up_question['prompt'],
                metadata=follow_up_question,
            )

        session.last_field = None
        return None
    
    # ----------- Chief complaint matching ------------------------------------

    def _apply_rule_outs(self, session: "MedicalSession") -> None:
        sorted_scores = sorted(session.condition_scores.items(), key=lambda x: x[1], reverse=True)
        remaining = [(cond, score) for cond, score in sorted_scores if score >= self.RULE_OUT_THRESHOLD]
        ruled_out = [(cond, score) for cond, score in sorted_scores if score < self.RULE_OUT_THRESHOLD]

        session.condition_rankings = remaining
        self._capture_debug(f"[Rule Out] 📉 Ruled out {len(ruled_out)} guidelines, {len(remaining)} remaining")
        self._update_condition_pools(session)
        self._log_rankings(session)

    def _log_llm_scores(
        self,
        element: str,
        answer: str,
        matches: List[str],
        term_scores: Dict[str, float],
    ) -> None:
        """Log LLM scores for element matching (LLM-only approach)."""
        sorted_scores = dict(sorted(term_scores.items(), key=lambda x: x[1], reverse=True))
        self._capture_debug(f"[LLM] 🔍 LLM scores for '{answer}' in {element}: {sorted_scores}")
        if matches:
            self._capture_debug(f"[LLM] ✅ Matched terms for {element}: {matches}")
        else:
            self._capture_debug(f"[LLM] ⚠️ No patient-friendly terms matched {element} for '{answer}'")

    def _log_llm_match_review(
        self,
        element: str,
        answer: str,
        review_rows: List[Tuple[str, float, float, float]],
    ) -> None:
        """Log LLM match review results (LLM-only approach, no FAISS)."""
        if not review_rows:
            return
        self._capture_debug(f"[LLM] 🔍 Match review for '{answer}' in {element}:")
        for term, unused_score, llm_score, final_score in review_rows:
            # Note: first score is always 0.0 (FAISS bypassed), second is LLM score, third is final (LLM-only)
            self._capture_debug(
                f"[LLM]   • {term}: LLM={llm_score:.3f}, final={final_score:.3f}"
            )

    def _collect_guidelines_for_session(self, session: "MedicalSession") -> List[Dict[str, Any]]:
        categories = session.context.get('matched_categories') or ['gastrointestinal']
        conditions = set(session.condition_scores.keys())
        collected: List[Dict[str, Any]] = []
        seen: set = set()
        for category in categories:
            for name, guideline in self._get_guidelines_by_category(category).items():
                condition_name = guideline.get('condition', name)
                if condition_name in conditions and condition_name not in seen:
                    collected.append(guideline)
                    seen.add(condition_name)
        return collected

    def _get_all_location_terms(self, session: "MedicalSession") -> List[str]:
        """Get all location terms from all guidelines in session"""
        return self._get_all_terms_for_element(session, 'location')
    
    def _get_all_terms_for_element(self, session: "MedicalSession", element: str) -> List[str]:
        """Get all terms for any element from all guidelines in session"""
        guidelines = self._collect_guidelines_for_session(session)
        all_terms = []
        seen = set()
        for guideline in guidelines:
            structured = self._structured_oldcarts(guideline)
            element_data = structured.get(element, {})
            includes = element_data.get('includes', []) if isinstance(element_data, dict) else []
            for item in includes:
                if isinstance(item, dict):
                    patient_term = item.get('patient_friendly')
                else:
                    patient_term = item
                if not isinstance(patient_term, str):
                    continue
                normalized = patient_term.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    all_terms.append(patient_term.strip())
        return all_terms

    # FAISS scoring removed - using LLM-only approach for all matching

    def _structured_oldcarts(self, guideline: Dict[str, Any]) -> Dict[str, Any]:
        structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
        if not structured:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
        return structured or {}

    def _llm_refine_matches(
        self,
        session: "MedicalSession",
        element: str,
        answer: str,
        matches: List[str],
        term_scores: Dict[str, float],
        threshold: float,
    ) -> Tuple[
        List[str],
        Dict[str, float],
        List[Tuple[str, float, float, float]],
        Dict[str, Any],
    ]:
        if not self.llm_chat_fn:
            self._capture_debug(f"[LLM] ⚠️ Match review skipped for '{answer}' in {element}: LLM function not available")
            return matches, term_scores, [], {'invoked': False, 'reason': 'no_llm_function'}
        
        if not matches:
            self._capture_debug(f"[LLM] ⚠️ Match review skipped for '{answer}' in {element}: No terms to review")
            return matches, term_scores, [], {'invoked': False, 'reason': 'no_matches'}

        unique_matches: List[str] = []
        for term in matches:
            if term not in unique_matches:
                unique_matches.append(term)

        # For ALL elements, send ALL terms to LLM (no limit, no FAISS filtering)
        # LLM is the decision maker for all elements
        limited = unique_matches  # Send all terms to LLM
        self._capture_debug(f"[LLM] 🔍 {element}: Sending ALL {len(limited)} terms to LLM for decision")

        if not limited:
            self._capture_debug(f"[LLM] ⚠️ Match review skipped for '{answer}' in {element}: No terms to review")
            return matches, term_scores, [], {
                'invoked': False,
                'reason': 'no_terms',
                'requested_terms': [],
            }

        self._capture_debug(f"[LLM] 🔍 Reviewing {len(limited)} terms for '{answer}' in {element}: {limited}")
        # Build alias map with multiple normalization strategies
        alias_map = {}
        for term in limited:
            # Exact lowercase match
            alias_map[term.lower()] = term
            # Normalized match (remove quotes, extra spaces)
            normalized = term.lower().strip().strip('"').strip("'").strip()
            alias_map[normalized] = term
            # Also add original term
            alias_map[term] = term
        
        candidate_lines = "\n".join(f"- {term}" for term in limited)
        
        # Create element-specific prompts (universal, condition-agnostic)
        element_prompts = {
            'location': (
                f"Chief complaint: {session.context['pre_hpi'].get('chief_complaint', 'unknown')}\n"
                f"Patient's description of symptom location: '{answer}'\n\n"
                "You are a medical expert evaluating anatomical location terms. "
                "Match the patient's description to the medical location terms based on ANATOMICAL ACCURACY.\n\n"
                "Use your medical knowledge to determine if the patient's description refers to the same anatomical location "
                "as each medical term. Consider:\n"
                "- Same body region (head, chest, abdomen, limbs, etc.)\n"
                "- Same anatomical structure (organ, muscle, bone, etc.)\n"
                "- Same relative position (left/right, upper/lower, anterior/posterior, etc.)\n"
                "- Anatomical synonyms and equivalent descriptions\n\n"
                "Location terms to evaluate:\n"
                f"{candidate_lines}\n\n"
                "Return ONLY valid JSON with ALL terms as keys and their scores as values.\n"
                "Format: {\"term1\": score1, \"term2\": score2, \"term3\": score3, ...}\n"
                "You MUST include ALL terms listed above with numeric scores (0.0 to 1.0).\n\n"
                "Scoring guidelines:\n"
                "- 1.0 = Exact anatomical match (same body region, structure, and relative position)\n"
                "- 0.0 = Completely different anatomical location (different body region or structure)\n"
                "- 0.1-0.4 = Distantly related or opposite locations (same body part but different side/position)\n"
                "- For location matching, be precise: use 1.0 for matches, 0.0 for clear mismatches\n\n"
                "Example format (not actual terms - your terms are listed above):\n"
                "{\n"
                '  "medical_term_1": 1.0,\n'
                '  "medical_term_2": 0.0,\n'
                '  "medical_term_3": 0.0\n'
                "}\n\n"
                "CRITICAL: Return a JSON object with ALL terms above as keys, each with a numeric score (0.0-1.0)."
            ),
        }
        
        # Use element-specific prompt if available, otherwise use generic prompt
        if element in element_prompts:
            user_prompt = element_prompts[element]
        else:
            user_prompt = (
                f"Chief complaint: {session.context['pre_hpi'].get('chief_complaint', 'unknown')}\n"
                f"Patient statement: '{answer}'\n\n"
                f"You are a medical expert evaluating {element} terms. "
                "Match the patient's statement to the medical guideline terms based on semantic meaning and medical relevance.\n\n"
                f"{element.replace('_', ' ').title()} terms to evaluate:\n"
                f"{candidate_lines}\n\n"
                "Return ONLY valid JSON with ALL terms as keys and their scores as values.\n"
                "Format: {\"term1\": score1, \"term2\": score2, \"term3\": score3, ...}\n"
                "You MUST include ALL terms listed above with numeric scores (0.0 to 1.0).\n\n"
                "Scoring:\n"
                "- 1.0 = Exact semantic match (same meaning)\n"
                "- 0.0 = No match (different meaning)\n"
                "- 0.5-0.9 = Partial or related match\n\n"
                "Example:\n"
                "{\n"
                '  "term1": 1.0,\n'
                '  "term2": 0.0,\n'
                '  "term3": 0.7\n'
                "}\n\n"
                "CRITICAL: Return a JSON object with ALL terms above as keys, each with a numeric score (0.0-1.0)."
            )
        
        # Use element-specific system prompt, otherwise use generic
        if element == 'location':
            system_prompt = (
                "You are a medical expert specializing in anatomical location assessment. "
                "Your task is to evaluate whether patient-described locations match medical location terms "
                "based on ANATOMICAL ACCURACY and your comprehensive anatomical knowledge.\n\n"
                "Use your medical expertise to determine if descriptions refer to the same anatomical location. "
                "Consider body regions, anatomical structures, relative positions, and anatomical terminology. "
                "Be precise: match exact locations, distinguish different body regions, and recognize anatomical synonyms.\n\n"
                "CRITICAL FORMAT REQUIREMENTS:\n"
                "- Output ONLY valid JSON (no explanations, no text before or after)\n"
                "- JSON must be an object with ALL terms as keys and numeric scores (0.0-1.0) as values\n"
                "- Example format: {\"term1\": 1.0, \"term2\": 0.0, \"term3\": 0.0}\n"
                "- Each term must be a key in the JSON object with its score as the value\n"
                "- Scores: 1.0 = exact anatomical match, 0.0 = different location\n"
                "- Do NOT use any other format - only JSON object with term keys and numeric values"
            )
        else:
            system_prompt = (
                "You are a medical expert evaluating patient statements against medical guideline terms. "
                "Your task is to determine semantic equivalence and medical relevance.\n\n"
                "CRITICAL FORMAT REQUIREMENTS:\n"
                "- Output ONLY valid JSON (no explanations, no text before or after)\n"
                "- JSON must be an object with ALL terms as keys and numeric scores (0.0-1.0) as values\n"
                "- Example format: {\"term1\": 1.0, \"term2\": 0.5, \"term3\": 0.0}\n"
                "- Each term must be a key in the JSON object with its score as the value\n"
                "- Scores: 1.0 = exact match, 0.0 = no match, 0.5-0.9 = partial match\n"
                "- Do NOT use any other format - only JSON object with term keys and numeric values"
            )
        
        self._capture_debug(f"[LLM] 🔍 Calling LLM for {element} match review...")
        self._capture_debug(f"[LLM] 🔍 System prompt: {system_prompt[:150]}...")
        self._capture_debug(f"[LLM] 🔍 User prompt (first 500 chars): {user_prompt[:500]}...")
        if len(user_prompt) > 500:
            self._capture_debug(f"[LLM] 🔍 User prompt length: {len(user_prompt)} chars (truncated in log)")
        
        # Also print to console for debugging
        print(f"[LLM] 🔍 Calling LLM for {element} match review - {len(limited)} terms")
        print(f"[LLM] 🔍 Terms: {limited[:5]}...")
        
        try:
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=250,  # More tokens for all elements (since we're sending all terms)
                temperature=0.0,
            )
            print(f"[LLM] ✅ LLM returned response (type: {type(response).__name__}, length: {len(response) if response else 0})")
            if response:
                print(f"[LLM] 🔍 Raw response (first 300 chars): {response[:300]}")
            else:
                print(f"[LLM] ⚠️ LLM returned EMPTY response")
            self._capture_debug(f"[LLM] ✅ LLM function returned response (type: {type(response).__name__})")
        except Exception as e:
            print(f"[LLM] ❌ LLM function raised exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self._capture_debug(f"[LLM] ❌ LLM function raised exception: {type(e).__name__}: {e}")
            response = None
        
        if response:
            self._capture_debug(f"[LLM] 🔍 Match review raw response (first 500 chars): {response[:500]}")
            if len(response) > 500:
                self._capture_debug(f"[LLM] 🔍 ... (response truncated, total length: {len(response)} chars)")
        else:
            self._capture_debug(f"[LLM] ⚠️ Match review raw response: EMPTY or None")

        llm_scores: Dict[str, float] = {}
        had_scores = False
        parse_error = None
        
        if not response:
            self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: LLM returned EMPTY response")
        else:
            parsed = None
            try:
                parsed = json.loads(response)
                self._capture_debug(f"[LLM] ✅ Successfully parsed JSON with {len(parsed) if isinstance(parsed, dict) else 'unknown'} keys")
            except json.JSONDecodeError as e:
                parse_error = str(e)
                print(f"[LLM] ⚠️ JSON parse error: {parse_error}")
                self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: JSON parse error: {parse_error}")
                self._capture_debug(f"[LLM] ⚠️ Raw response (first 200 chars): {response[:200]}")
                # Try to extract JSON from response using a more robust method
                # Look for JSON object boundaries (start with {, end with })
                start_idx = response.find('{')
                if start_idx != -1:
                    # Try to find matching closing brace by counting braces
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(response)):
                        if response[i] == '{':
                            brace_count += 1
                        elif response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        try:
                            extracted_json = response[start_idx:end_idx]
                            parsed = json.loads(extracted_json)
                            parse_error = None
                            print(f"[LLM] ✅ Successfully extracted and parsed JSON from response")
                            self._capture_debug(f"[LLM] ✅ Extracted and parsed JSON from response")
                        except json.JSONDecodeError as e2:
                            parse_error = str(e2)
                            parsed = None
                            print(f"[LLM] ⚠️ Failed to parse extracted JSON: {parse_error}")
                            self._capture_debug(f"[LLM] ⚠️ Failed to parse extracted JSON: {parse_error}")
                            self._capture_debug(f"[LLM] ⚠️ Extracted JSON (first 300 chars): {extracted_json[:300]}")
                    else:
                        print(f"[LLM] ⚠️ Could not find matching closing brace in JSON")
                        self._capture_debug(f"[LLM] ⚠️ Could not find matching closing brace in JSON")
                else:
                    print(f"[LLM] ⚠️ No opening brace found in response")
                    self._capture_debug(f"[LLM] ⚠️ No opening brace found in response")
            
            if parse_error:
                self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: Failed to parse JSON. Error: {parse_error}")
                self._capture_debug(f"[LLM] ⚠️ Full response: {response}")
            elif parsed is None:
                self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: No JSON found in response")
                self._capture_debug(f"[LLM] ⚠️ Response content: {response}")
            elif not isinstance(parsed, dict):
                self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: Response is not a JSON object. Type: {type(parsed)}")
                self._capture_debug(f"[LLM] ⚠️ Response content: {response}")
            elif isinstance(parsed, dict):
                llm_returned_keys = list(parsed.keys())
                expected_keys = list(limited)
                self._capture_debug(f"[LLM] ✅ Parsed JSON with {len(parsed)} keys")
                print(f"[LLM] ✅ Parsed JSON with {len(parsed)} keys: {llm_returned_keys}")
                self._capture_debug(f"[LLM] 🔍 LLM returned keys: {llm_returned_keys[:10]}")
                self._capture_debug(f"[LLM] 🔍 Expected keys (sample): {expected_keys[:10]}")
                
                # Detect wrong format: LLM returned generic keys like "term" instead of actual term names
                wrong_format_detected = False
                if len(parsed) == 1 and "term" in parsed:
                    wrong_format_detected = True
                    print(f"[LLM] ❌ WRONG FORMAT DETECTED: LLM returned {{\"term\": \"{parsed.get('term')}\"}} instead of {{\"term_name\": score}}")
                    self._capture_debug(f"[LLM] ❌ WRONG FORMAT: LLM returned generic key 'term' with value '{parsed.get('term')}' instead of term names as keys")
                    self._capture_debug(f"[LLM] ❌ Expected format: {{\"medical_term_1\": 1.0, \"medical_term_2\": 0.0, \"medical_term_3\": 0.0, ...}}")
                    self._capture_debug(f"[LLM] ❌ LLM should return ALL terms as keys with numeric scores (0.0-1.0) as values")
                
                # Check if any values are strings instead of numbers
                string_values = [(k, v) for k, v in parsed.items() if isinstance(v, str)]
                if string_values:
                    print(f"[LLM] ⚠️ Found {len(string_values)} non-numeric values (should be numbers): {string_values[:3]}")
                    self._capture_debug(f"[LLM] ⚠️ Found {len(string_values)} keys with string values (should be numeric scores): {string_values[:5]}")
                
                # Check how many keys match expected terms
                matching_keys = [k for k in llm_returned_keys if k in expected_keys or k.lower() in [t.lower() for t in expected_keys]]
                if len(matching_keys) == 0 and len(llm_returned_keys) > 0:
                    print(f"[LLM] ⚠️ NO MATCHING KEYS: LLM returned keys {llm_returned_keys} but expected keys like {expected_keys[:3]}")
                    wrong_format_detected = True
                
                for key, value in parsed.items():
                    if not isinstance(value, (int, float)):
                        print(f"[LLM] ⚠️ Skipping '{key}': value is {type(value).__name__} (not numeric): {value}")
                        self._capture_debug(f"[LLM] ⚠️ Skipping '{key}': value is {type(value).__name__} (not numeric): {value}")
                    continue
                    term_key = key.strip()
                    score = max(0.0, min(1.0, float(value)))
                    
                    # Try multiple matching strategies
                    canonical = None
                    # Strategy 1: Exact lowercase match
                    canonical = alias_map.get(term_key.lower())
                    # Strategy 2: Normalized match (remove quotes, extra spaces)
                    if not canonical:
                        normalized_key = term_key.lower().strip().strip('"').strip("'").strip()
                        canonical = alias_map.get(normalized_key)
                    # Strategy 3: Try exact match
                    if not canonical:
                        canonical = alias_map.get(term_key)
                    # Strategy 4: Try fuzzy match (find closest term)
                    if not canonical:
                        # Find closest term by checking if key contains term or term contains key
                        for term in limited:
                            if term_key.lower() in term.lower() or term.lower() in term_key.lower():
                                canonical = term
                                self._capture_debug(f"[LLM] ✅ Fuzzy matched '{term_key}' to '{canonical}'")
                                break
                    
                    if canonical:
                        llm_scores[canonical] = score
                        self._capture_debug(f"[LLM] ✅ Scored '{canonical}' (from LLM key '{term_key}'): {score:.3f}")
                    else:
                        # If no match found, still store it (might be a valid term we didn't send)
                        llm_scores[term_key] = score
                        self._capture_debug(f"[LLM] ⚠️ LLM key '{term_key}' not found in expected terms, storing as-is with score {score:.3f}")
                    had_scores = True
                
                if not had_scores:
                    self._capture_debug(f"[LLM] ⚠️ Match review for '{answer}' in {element}: LLM returned JSON with {len(parsed)} keys but NO valid numeric scores")
                    self._capture_debug(f"[LLM] ⚠️ JSON keys and sample values: {list(parsed.items())[:5]}")
                    self._capture_debug(f"[LLM] ⚠️ Expected terms: {limited[:10]}")
            else:
                    self._capture_debug(f"[LLM] ✅ Successfully extracted {len(llm_scores)} LLM scores")
                    # Log which terms were matched
                    matched_terms = [term for term in limited if term in llm_scores]
                    unmatched_terms = [term for term in limited if term not in llm_scores]
                    if matched_terms:
                        self._capture_debug(f"[LLM] ✅ Matched terms: {matched_terms[:5]}")
                    if unmatched_terms:
                        self._capture_debug(f"[LLM] ⚠️ Unmatched terms: {unmatched_terms[:5]}")

        refined_scores = dict(term_scores)
        review_rows: List[Tuple[str, float, float, float]] = []

        # For ALL elements, use LLM scores exclusively (100% weight)
        # FAISS is bypassed completely - LLM is the decision maker for all elements
        use_llm_only = True  # All elements now use LLM exclusively
        
        # If LLM didn't return scores, we can't proceed (LLM is required for all elements)
        if not had_scores:
            self._capture_debug(f"[LLM] ⚠️ {element}: LLM did not return scores for '{answer}'. Cannot determine matches without LLM decision.")
            return [], refined_scores, [], {
                'invoked': True,
                'requested_terms': limited,
                'raw_response': response,
                'had_scores': False,
            }

        for term in limited:
            # LLM-only approach: FAISS is bypassed, so first score is always 0.0 (kept for tuple compatibility)
            unused_score = 0.0  # Placeholder (FAISS bypassed in LLM-only approach)
            llm_score = llm_scores.get(term, llm_scores.get(term.lower()))
            
            # For ALL elements, use LLM score exclusively
            if llm_score is not None:
                refined_scores[term] = llm_score
                review_rows.append((term, unused_score, llm_score, llm_score))
            else:
                # If LLM didn't score it, set to 0.0 (no match)
                refined_scores[term] = 0.0
                review_rows.append((term, unused_score, 0.0, 0.0))

        # Order terms by refined score for filtering
        ordered = sorted(limited, key=lambda t: refined_scores.get(t, 0.0), reverse=True)
        
        # For ALL elements, use LLM's decision threshold (0.5 or higher means match)
        blended_threshold = 0.5  # LLM score >= 0.5 means match for all elements
        
        filtered = [term for term in ordered if refined_scores.get(term, 0.0) >= blended_threshold]

        if not filtered:
            best_term = max(ordered, key=lambda t: refined_scores.get(t, 0.0))
            if refined_scores.get(best_term, 0.0) >= 0.5:
                filtered = [best_term]

        review_meta = {
            'invoked': True,
            'requested_terms': limited,
            'raw_response': response,
            'had_scores': had_scores,
        }

        return filtered, refined_scores, review_rows, review_meta

    def _analyze_location_answer(
        self,
        session: "MedicalSession",
        element: str,
        answer: str,
        matches: List[str],
        term_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        guidelines = self._collect_guidelines_for_session(session)
        total_guidelines = len(guidelines)
        answer_lower = answer.lower()

        # LLM-only approach: no anatomical filtering needed - LLM handles anatomical reasoning
        # Step 1: collect terms from all guidelines
        all_terms_patient: Dict[str, Dict[str, str]] = {}
        medical_to_patient: Dict[str, str] = {}
        term_to_guidelines: Dict[str, List[str]] = {}
        for guideline in guidelines:
            condition_name = guideline.get('condition', guideline.get('name', 'Unknown'))
            structured = self._structured_oldcarts(guideline)
            location_data = structured.get('location', {})
            includes = location_data.get('includes', []) if isinstance(location_data, dict) else []
            for item in includes:
                if isinstance(item, dict):
                    patient_term = item.get('patient_friendly')
                    medical_term = item.get('medical')
                else:
                    patient_term = item
                    medical_term = item
                if not isinstance(patient_term, str):
                    continue
                normalized_pf = patient_term.strip()
                if not normalized_pf:
                    continue
                key = normalized_pf.lower()
                all_terms_patient[key] = {
                    'patient_friendly': normalized_pf,
                    'medical': medical_term.strip() if isinstance(medical_term, str) else normalized_pf,
                }
                medical_to_patient[all_terms_patient[key]['medical'].lower()] = normalized_pf
                term_to_guidelines.setdefault(key, []).append(condition_name)

        # Step 3: satisfied terms (patient friendly)
        semantic_set = {m.lower() for m in matches}
        satisfied_pf_terms = [all_terms_patient[key]['patient_friendly'] for key in all_terms_patient if key in semantic_set]

        # Map to medical terms and deduplicate
        satisfied_medical_terms = []
        seen_medical = set()
        for key in all_terms_patient:
            if key in semantic_set:
                med = all_terms_patient[key]['medical']
                med_key = med.lower()
                if med_key not in seen_medical:
                    seen_medical.add(med_key)
                    satisfied_medical_terms.append(med)

        priority_conditions = {
            cond for cond, score in session.condition_scores.items() if score > (0.5 + 1e-6)
        }
        if priority_conditions and satisfied_medical_terms:
            filtered_med_terms: List[str] = []
            fallback_med_terms: List[str] = []
            for med in satisfied_medical_terms:
                patient_term = medical_to_patient.get(med.lower(), med)
                conds = term_to_guidelines.get(patient_term.lower(), [])
                if conds and not any(cond in priority_conditions for cond in conds):
                    fallback_med_terms.append(med)
                else:
                    filtered_med_terms.append(med)

            if filtered_med_terms:
                satisfied_medical_terms = filtered_med_terms
            elif fallback_med_terms:
                satisfied_medical_terms = fallback_med_terms

        # Missing medical terms if none satisfied
        # LLM-only approach: no anatomical filtering needed - LLM already handled anatomical reasoning in scoring
        missing_medical_terms = []
        if not satisfied_medical_terms:
            unsatisfied_keys = [key for key in all_terms_patient if key not in semantic_set]
            unsatisfied_medical = []
            seen_unsatisfied = set()
            for key in unsatisfied_keys:
                med = all_terms_patient[key]['medical']
                med_key = med.lower()
                if med_key not in seen_unsatisfied:
                    seen_unsatisfied.add(med_key)
                    unsatisfied_medical.append(med)
            # Rank by LLM scores
            scored_missing = []
            for med in unsatisfied_medical:
                pf = medical_to_patient.get(med.lower(), med)
                score = term_scores.get(pf, term_scores.get(pf.lower(), 0.0))
                scored_missing.append((med, score))
            scored_missing.sort(key=lambda x: x[1], reverse=True)
            missing_medical_terms = [med for med, _ in scored_missing[:5]]

        sorted_scores = dict(sorted(term_scores.items(), key=lambda x: x[1], reverse=True))
        term_breakdown = []
        # For ALL elements, use LLM threshold (0.5) since LLM scores are used exclusively
        # FAISS is bypassed completely - LLM is the decision maker for all elements
        threshold = 0.5  # LLM threshold for all elements
        boosted_matches: Optional[List[str]] = None
        boosted_term_scores: Optional[Dict[str, float]] = None
        for key, meta in sorted(all_terms_patient.items()):
            patient_term = meta['patient_friendly']
            score = sorted_scores.get(patient_term, sorted_scores.get(patient_term.lower(), 0.0))
            term_breakdown.append({
                'term': patient_term,
                'score': score,
                'in_semantic': patient_term.lower() in semantic_set,
                'term_in_answer': patient_term.lower() in answer_lower,
                'answer_in_term': answer_lower in patient_term.lower(),
                'medical': meta['medical'],
            })

        if element == 'location':
            # For location, use LLM scores exclusively - check for high confidence (>= 0.95)
            # or use the LLM threshold (0.5) for matches
            high_conf_keys: List[str] = []
            for key, meta in all_terms_patient.items():
                pf = meta['patient_friendly']
                score = sorted_scores.get(pf, sorted_scores.get(pf.lower(), 0.0))
                # For location, high confidence is 0.95 (very confident LLM match)
                if score >= 0.95:
                    high_conf_keys.append(key)

            if high_conf_keys:
                high_conf_matches: List[str] = []
                high_conf_med_terms: List[str] = []
                seen_high = set()
                for key in high_conf_keys:
                    pf_term = all_terms_patient[key]['patient_friendly']
                    med_term = all_terms_patient[key]['medical']
                    high_conf_matches.append(pf_term)
                    med_lower = med_term.lower()
                    if med_lower not in seen_high:
                        seen_high.add(med_lower)
                        high_conf_med_terms.append(med_term)

                # LLM-only approach: no anatomical filtering needed - LLM already handled anatomical reasoning in scoring
                if high_conf_med_terms:
                    satisfied_medical_terms = high_conf_med_terms
                    matches = high_conf_matches
                    semantic_set = {term.lower() for term in matches}
                    for entry in term_breakdown:
                        entry['in_semantic'] = entry['term'].lower() in semantic_set
                    boosted_matches = high_conf_matches
                    boosted_term_scores = {pf: sorted_scores.get(pf, sorted_scores.get(pf.lower(), self.LOCATION_HIGH_CONFIDENCE_THRESHOLD)) for pf in high_conf_matches}

        def _unique(sequence: List[str]) -> List[str]:
            seen_local: set = set()
            result: List[str] = []
            for item in sequence:
                key_local = item.lower()
                if key_local not in seen_local:
                    seen_local.add(key_local)
                    result.append(item)
            return result

        def _filter_by_condition_scores(options: List[str]) -> List[str]:
            if not options:
                return options
            baseline = 0.5 - 1e-6
            filtered: List[str] = []
            skipped: List[str] = []
            for opt in options:
                conds = term_to_guidelines.get(opt.lower(), [])
                if not conds:
                    filtered.append(opt)
                    continue
                if any(session.condition_scores.get(cond, 0.5) > baseline for cond in conds):
                    filtered.append(opt)
                else:
                    skipped.append(opt)
            if skipped:
                self._capture_debug(
                    f"[Clarification] ⚖️ Skipping options tied to baseline scores: {skipped}"
                )
            if filtered:
                return filtered
            return options

        satisfied_options = _filter_by_condition_scores(
            _unique([medical_to_patient.get(med.lower(), med) for med in satisfied_medical_terms])
        )

        missing_options = _filter_by_condition_scores(
            _unique([medical_to_patient.get(med.lower(), med) for med in missing_medical_terms])
        )

        all_terms_list = sorted([meta['patient_friendly'] for meta in all_terms_patient.values()])
        
        return {
            'answer': answer,
            'threshold': threshold,
            'total_guidelines': total_guidelines,
            'all_terms': all_terms_list,
            'active_terms': [],
            'semantic_matches': matches,
            'llm_scores': sorted_scores,  # LLM scores for all terms (LLM-only approach)
            'term_breakdown': term_breakdown,
            'satisfied_medical_terms': satisfied_medical_terms,
            'satisfied_options': satisfied_options,
            'missing_medical_terms': missing_medical_terms,
            'missing_options': missing_options,
            'term_to_guidelines': term_to_guidelines,
            'boosted_matches': boosted_matches,
            'boosted_term_scores': boosted_term_scores,
        }

    def _maybe_request_location_clarification(
        self,
        session: "MedicalSession",
        element: str,
        answer: str,
        analysis: Dict[str, Any],
        clarification_just_asked: bool,
    ) -> Optional[Dict[str, Any]]:
        if clarification_just_asked:
            return None

        satisfied = analysis.get('satisfied_medical_terms', [])
        missing = analysis.get('missing_medical_terms', [])

        if len(satisfied) == 1:
            self._capture_debug("[Clarification] ✅ Exactly one satisfied term - no clarification needed")
            session.context['clarifications'].pop(element, None)
            return None

        if len(satisfied) >= 2:
            # Check if one match is clearly best (auto-selection logic)
            # Get LLM scores for satisfied options
            scores = analysis.get('boosted_term_scores') or analysis.get('llm_scores', {})
            satisfied_options = analysis.get('satisfied_options', [])
            
            # Build list of (option, score) tuples for satisfied options
            option_scores = []
            for option in satisfied_options:
                score = scores.get(option, scores.get(option.lower(), 0.0))
                option_scores.append((option, score))
            
            # Sort by score (highest first)
            option_scores.sort(key=lambda x: x[1], reverse=True)
            
            if len(option_scores) >= 2:
                best_option, best_score = option_scores[0]
                next_best_score = option_scores[1][1]
                
                # Check if best match is clearly best
                best_is_clear = (
                    best_score >= self.AUTO_SELECT_BEST_THRESHOLD and
                    (
                        (best_score - next_best_score) >= self.AUTO_SELECT_MARGIN or
                        next_best_score < self.AUTO_SELECT_NEXT_MAX
                    )
                )
                
                if best_is_clear:
                    self._capture_debug(
                        f"[Clarification] ✅ Auto-selecting best match: '{best_option}' "
                        f"(score={best_score:.3f}, next_best={next_best_score:.3f})"
                    )
                    # Find medical term for best option using term_breakdown
                    # Then verify it's in satisfied_medical_terms
                    term_breakdown = analysis.get('term_breakdown', [])
                    satisfied_set = {med.lower() for med in satisfied}
                    best_medical = None
                    
                    for entry in term_breakdown:
                        patient_term = entry.get('term', '').lower()
                        medical_term = entry.get('medical', '')
                        if patient_term == best_option.lower() and medical_term.lower() in satisfied_set:
                            best_medical = medical_term
                            break
                    
                    if not best_medical:
                        # Fallback: find first satisfied medical term that might correspond
                        # by checking if any satisfied option matches best_option
                        for med in satisfied:
                            # Try to find corresponding patient-friendly term
                            for entry in term_breakdown:
                                if entry.get('medical', '').lower() == med.lower():
                                    if entry.get('term', '').lower() == best_option.lower():
                                        best_medical = med
                                        break
                            if best_medical:
                                break
                    
                    if best_medical:
                        # Update analysis to only include best match
                        analysis['satisfied_medical_terms'] = [best_medical]
                        analysis['satisfied_options'] = [best_option]
                        self._capture_debug(f"[Clarification] ✅ Auto-selected: '{best_option}' → {best_medical}")
                    else:
                        # Last resort: use first satisfied medical term
                        if satisfied:
                            analysis['satisfied_medical_terms'] = [satisfied[0]]
                            analysis['satisfied_options'] = [best_option]
                            self._capture_debug(f"[Clarification] ✅ Auto-selected (fallback): '{best_option}' → {satisfied[0]}")
                    
                    session.context['clarifications'].pop(element, None)
                    return None
            
            # If no clear winner, trigger clarification
            self._capture_debug(f"[Clarification] 🔍 {len(satisfied)} satisfied medical terms - generating clarification with satisfied context")
            options = satisfied_options[:5]
            question = self._build_clarifying_question(element, answer, options, satisfied_context=True)
            clar_data = {
                'section': 'hpi',
                'field': element,
                'prompt': question,
                'guidance': 'clarification',
                'clarification': True,
                'mode': 'satisfied',
                'options': options,
            }
            session.context['clarifications'][element] = clar_data
            return clar_data

        if not satisfied and missing:
            self._capture_debug("[Clarification] 🔍 No satisfied terms - generating clarification with missing terms")
            options = analysis.get('missing_options', [])[:5]
            question = self._build_clarifying_question(element, answer, options, satisfied_context=False)
            clar_data = {
                'section': 'hpi',
                'field': element,
                'prompt': question,
                'guidance': 'clarification',
                'clarification': True,
                'mode': 'missing',
                'options': options,
            }
            session.context['clarifications'][element] = clar_data
            return clar_data

        session.context['clarifications'].pop(element, None)
        return None

    def _build_clarifying_question(
        self,
        element: str,
        answer: str,
        options: List[str],
        satisfied_context: bool,
    ) -> str:
        if not options:
            return "Could you describe the location a bit more specifically?"

        friendly_options = ', '.join(options)
        if satisfied_context:
            return (
                f"I heard a few possible locations based on what you said ({friendly_options}). "
                f"Which one matches your {element} the best?"
            )
        return (
            f"I'm still not sure exactly where you feel it. "
            f"Would you say it's more like {friendly_options}?"
        )

    def _log_location_analysis(self, session: "MedicalSession", analysis: Dict[str, Any]) -> None:
        answer = analysis['answer']
        matches = analysis.get('semantic_matches', [])
        term_breakdown = analysis.get('term_breakdown', [])
        threshold = analysis.get('threshold', 0.6)
        llm_scores = analysis.get('llm_scores', {})  # LLM scores for all terms (LLM-only approach)

        self._capture_debug(
            f"[Location Analysis] 📍 Checking satisfaction against ALL {analysis.get('total_guidelines', 0)} guidelines (active + reserve)"
        )
        self._capture_debug(
            f"[Location Analysis] 📍 All includes terms from {len(analysis.get('all_terms', []))} total guidelines: {analysis.get('all_terms', [])}"
        )
        self._capture_debug(f"[Location Analysis] 📝 Patient answer: '{answer}'")
        self._capture_debug(f"[LLM] 🔍 LLM scores for '{answer}' in location: {llm_scores}")
        self._capture_debug(f"[Location Analysis] 🔍 LLM found {len(matches)} matches above threshold ({threshold}): {matches}")
        self._capture_debug(f"[Location Analysis]   - semantic_matches_set ({len(matches)} terms): {matches}")
        self._capture_debug(f"[Location Analysis]   - LLM scores ({len(llm_scores)} terms): {llm_scores}")

        normalized_answer = answer.lower().strip()
        semantic_set = {m.lower() for m in matches}

        checked_terms: List[str] = []
        satisfied_terms_log: List[str] = []
        unsatisfied_terms_log: List[str] = []

        for entry in term_breakdown:
            term = entry['term']
            score = entry['score']
            in_semantic = entry['in_semantic']
            term_lower = term.lower()

            self._capture_debug(f"[Location Analysis] 🔍 Checking term: '{term}' (patient answer: '{answer}')")
            self._capture_debug(
                f"[Location Analysis]   LLM check: in semantic_matches_set={term_lower in semantic_set}, score={score}"
            )
            self._capture_debug(
                f"[Location Analysis]   LLM score for '{term}': {score:.3f} (threshold={threshold})"
            )
            if score < threshold:
                self._capture_debug(
                    f"[Location Analysis]   ⚠️ LLM score {score:.3f} < {threshold} threshold, should NOT be in semantic_matches_set"
                )
            if in_semantic:
                self._capture_debug(f"[Location Analysis]   ✅ '{term}' satisfied")
                satisfied_terms_log.append(term)
            else:
                self._capture_debug(f"[Location Analysis]   ❌ '{term}' not satisfied")
                unsatisfied_terms_log.append(term)
            checked_terms.append(term)

        self._capture_debug(
            f"[Location Analysis] 📋 Terms checked ({len(checked_terms)}): {checked_terms}"
        )
        if satisfied_terms_log:
            self._capture_debug(
                f"[Location Analysis] ✅ Satisfied terms: {satisfied_terms_log}"
            )
        if unsatisfied_terms_log:
            self._capture_debug(
                f"[Location Analysis] ❌ Unsatisfied terms: {unsatisfied_terms_log}"
            )

    def _match_chief_complaint_to_category_llm(self, chief_complaint: str) -> List[str]:
        """Match chief complaint to medical categories using LLM only."""
        if not self.chief_complaint_triggers_data:
            self._capture_debug("[Engine] ⚠️ No chief complaint triggers available - defaulting to gastrointestinal")
            return ['gastrointestinal']

        # Group triggers by category
        triggers_by_category: Dict[str, List[Dict]] = {}
        triggers_by_condition: Dict[str, Dict] = {}
        for trigger_data in self.chief_complaint_triggers_data:
            category = trigger_data.get('category', 'gastrointestinal')
            condition = trigger_data.get('condition', '')
            if category not in triggers_by_category:
                triggers_by_category[category] = []
            triggers_by_category[category].append(trigger_data)
            if condition:
                triggers_by_condition[condition] = trigger_data

        # Build LLM prompt
        available_categories = list(triggers_by_category.keys())
        category_summary = []
        for category, triggers in triggers_by_category.items():
            trigger_texts = [t['trigger'] for t in triggers[:5]]  # Sample triggers
            category_summary.append(f"- {category}: {', '.join(trigger_texts)}")

        system_prompt = (
            "You are a medical expert matching patient chief complaints to medical categories. "
            "Use your medical knowledge to determine which categories match the patient's complaint. "
            "Consider synonyms, related terms, and medical context.\n\n"
            "CRITICAL FORMAT REQUIREMENTS:\n"
            "- Output ONLY valid JSON (no explanations, no text before or after)\n"
            "- JSON must be an object with categories as keys and confidence scores (0.0-1.0) as values\n"
            "- Example format: {\"gastrointestinal\": 0.9, \"cardiovascular\": 0.3}\n"
            "- Scores: 1.0 = exact match, 0.0 = no match, 0.5-0.9 = partial match\n"
            "- Include ALL categories with scores, even if 0.0\n"
            "- Also include a \"conditions\" key with condition names and their scores\n"
            "- Example: {\"gastrointestinal\": 0.9, \"cardiovascular\": 0.0, \"conditions\": {\"GERD\": 0.9, \"Appendicitis\": 0.2}}\n"
            "- Do NOT use any other format - only JSON object"
        )

        user_prompt = (
            f"Patient chief complaint: '{chief_complaint}'\n\n"
            f"Available medical categories:\n"
            f"{chr(10).join(category_summary)}\n\n"
            f"Match the patient's chief complaint to the appropriate medical categories and specific conditions. "
            f"Use your medical knowledge to handle misspellings, synonyms, and related terms. "
            f"Return a JSON object with category scores and condition scores."
        )

        try:
            self._capture_debug(f"[Engine] 🔍 LLM matching chief complaint '{chief_complaint}' to categories")
            response = self.llm_chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            
            if not response:
                self._capture_debug("[Engine] ⚠️ LLM returned empty response for chief complaint matching")
                return ['gastrointestinal']

            # Parse JSON response
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    self._capture_debug(f"[Engine] ⚠️ Failed to parse LLM response: {response[:200]}")
                    return ['gastrointestinal']

            # Extract category scores
            category_scores: Dict[str, float] = {}
            condition_scores: Dict[str, float] = {}
            
            # Get conditions if provided
            conditions_dict = parsed.get('conditions', {})
            if isinstance(conditions_dict, dict):
                for condition, score in conditions_dict.items():
                    if isinstance(score, (int, float)):
                        condition_scores[condition] = max(0.0, min(1.0, float(score)))

            # Get category scores (exclude 'conditions' key)
            for key, value in parsed.items():
                if key == 'conditions':
                    continue
                if key in available_categories and isinstance(value, (int, float)):
                    category_scores[key] = max(0.0, min(1.0, float(value)))

            self._capture_debug(f"[Engine] 🔍 LLM category scores: {category_scores}")
            if condition_scores:
                self._capture_debug(f"[Engine] 🔍 LLM condition scores: {dict(list(condition_scores.items())[:5])}")

            # Filter categories by threshold
            matched_categories = [
                cat for cat, score in category_scores.items()
                if score >= self.CHIEF_COMPLAINT_LLM_THRESHOLD
            ]

            if not matched_categories:
                # If no categories above threshold, use top category
                if category_scores:
                    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                    top_cat, top_score = sorted_cats[0]
                    if top_score > 0.3:  # Lower threshold for fallback
                        matched_categories = [top_cat]
                        self._capture_debug(f"[Engine] ⚠️ No categories above threshold, using top category: {top_cat} ({top_score:.3f})")
                    else:
                        self._capture_debug(f"[Engine] ⚠️ No confident category match found, defaulting to gastrointestinal")
                        matched_categories = ['gastrointestinal']
                else:
                    self._capture_debug(f"[Engine] ⚠️ No category scores found, defaulting to gastrointestinal")
                    matched_categories = ['gastrointestinal']

            # Store condition seeds
            self._chief_complaint_condition_seed = condition_scores
            if condition_scores:
                top_preview = ', '.join(
                    f"{name}: {score:.3f}" for name, score in sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                self._capture_debug(f"[Engine] 📌 Chief complaint condition seeds: {top_preview}")

            if len(matched_categories) == 1:
                self._capture_debug(f"[Engine] 🎯 Category matched: {matched_categories[0]} (LLM score: {category_scores.get(matched_categories[0], 0.0):.3f})")
            else:
                scores = ', '.join(f"{cat} ({category_scores.get(cat, 0.0):.3f})" for cat in matched_categories)
                self._capture_debug(f"[Engine] 🎯 Multiple categories matched: {scores}")

            return matched_categories

        except Exception as e:
            self._capture_debug(f"[Engine] ❌ Error in LLM chief complaint matching: {e}")
            import traceback
            traceback.print_exc()
            return ['gastrointestinal']
 
     # ----------- Guidance builders -------------------------------------------

    def _resolve_guidelines_dir(self) -> Optional[Path]:
        candidates = [
            Path("/app/medical/guidelines"),
            Path("medical/guidelines"),
            Path(__file__).resolve().parent / "medical" / "guidelines",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _get_enabled_categories(self) -> List[str]:
        enabled_env = os.environ.get('ENABLED_MEDICAL_CATEGORIES', '').strip()
        if not enabled_env:
            return []
        return [cat.strip().upper() for cat in enabled_env.split(',') if cat.strip()]

    def _load_guidelines(self) -> None:
        if not self.guidelines_dir or not self.guidelines_dir.exists():
            return
        
        loaded = 0
        skipped = 0

        for json_file in sorted(self.guidelines_dir.glob("**/*.json")):
            try:
                organ_system = json_file.parent.name
                if self.enabled_categories and organ_system.upper() not in self.enabled_categories:
                    skipped += 1
                    continue
                with json_file.open('r') as f:
                    guideline = json.load(f)
                condition_name = guideline.get('condition', json_file.stem)
                guideline['organ_system'] = organ_system
                self.all_guidelines[condition_name] = guideline
                loaded += 1
            except Exception as e:
                self._capture_debug(f"[Navigator] ⚠️ Failed to load guideline {json_file.name}: {e}")

        self._capture_debug(f"[Navigator] 📚 Loaded {loaded} guidelines ({skipped} skipped)")

    def _build_chief_complaint_triggers(self) -> None:
        """Build chief complaint triggers list for LLM matching (no FAISS index needed)."""
        self.chief_complaint_triggers_data = []

        for name, guideline in self.all_guidelines.items():
            trigger_list = guideline.get('chief_complaint_triggers', [])
            category = self._get_guideline_category(guideline)
            for trigger in trigger_list:
                if not trigger:
                    continue
                self.chief_complaint_triggers_data.append({
                    'trigger': trigger,
                    'category': category,
                    'condition': name,
                })

        if not self.chief_complaint_triggers_data:
            self._capture_debug("[Navigator] ⚠️ No chief complaint triggers found in guidelines")
        else:
            self._capture_debug(f"[Navigator] ✅ Collected {len(self.chief_complaint_triggers_data)} chief complaint triggers for LLM matching")

    def _get_guideline_category(self, guideline: Dict) -> str:
        organ_system = guideline.get('organ_system', '')
        for category, system in self.CATEGORY_TO_SYSTEM.items():
            if system.upper() == organ_system.upper():
                return category
        return 'gastrointestinal'

    def _get_guidelines_by_category(self, category: str) -> Dict[str, Dict]:
        if category == 'ALL':
            return self.all_guidelines

        target_system = self.CATEGORY_TO_SYSTEM.get(category.lower(), category.upper())
        filtered = {}
        for name, guideline in self.all_guidelines.items():
            organ_system = guideline.get('organ_system', '')
            if organ_system.upper() == target_system or target_system in organ_system.upper():
                filtered[name] = guideline
        if not filtered:
            return self.all_guidelines
        return filtered

    def _get_conditions_for_categories(self, categories: List[str]) -> List[str]:
        if not categories:
            categories = ['gastrointestinal']
        conditions = set()
        for category in categories:
            for name, guideline in self._get_guidelines_by_category(category).items():
                condition_name = guideline.get('condition', name)
                if condition_name:
                    conditions.add(condition_name)
        return sorted(list(conditions))

    def _top_condition_names(self, session: "MedicalSession", limit: int = 5) -> List[str]:
        names: List[str] = []
        rankings = session.condition_rankings or []
        for condition, _score in rankings:
            if condition and condition not in names:
                names.append(condition)
            if len(names) >= limit:
                break
        if len(names) < limit:
            sorted_scores = sorted(session.condition_scores.items(), key=lambda item: item[1], reverse=True)
            for condition, _ in sorted_scores:
                if condition and condition not in names:
                    names.append(condition)
                if len(names) >= limit:
                    break
        return names

    def _build_oldcarts_guidance(self, session: "MedicalSession", element: str, cc: str) -> Tuple[str, str, List[str]]:
        includes = self._get_element_includes(session, element)
        base_template = self.HPI_BASE_GUIDANCE[element]
        base_question = base_template.replace('{cc}', cc)
        character_summary = self._character_tag_summary(session)
        character_tags = character_summary.get('tags', set())
        dominant_tag = character_summary.get('dominant')
        allowed_conditions = self._priority_condition_set(session)
 
        if element == 'character':
            subject = cc if cc else 'symptoms'
            plural = subject.endswith('s')
            feel_verb = 'feel' if plural else 'feels'
            look_verb = 'look' if plural else 'looks'
            do_verb = 'do' if plural else 'does'
            visual = 'visual' in character_tags
            sensory = 'sensory' in character_tags
            if visual and not sensory:
                base_question = f"What {do_verb} your {subject} {look_verb} like?"
            elif visual and sensory:
                base_question = f"How would you describe how your {subject} {feel_verb} or {look_verb}?"
            else:
                base_question = f"How would you describe what your {subject} {feel_verb} like?"
        else:
            base_question = base_template.replace('{cc}', cc)
 
        inject_options = element != 'severity'
        if element == 'location' and dominant_tag == 'visual':
            inject_options = False
 
        sample_terms: List[str] = []
        debug_entries: List[Dict[str, Any]] = []
        if inject_options:
            sample_entries = self._select_guidance_entries(session, element, includes, limit=2, debug_entries=debug_entries)
            sample_terms = [entry['patient_friendly'] for entry in sample_entries if entry.get('patient_friendly')]
        if debug_entries:
            self._capture_debug(f"[Guidance] 📋 Option candidates ({element}): {debug_entries}")
 
        if sample_terms:
            top_condition = next(iter(self._priority_condition_set(session)), None)
            if top_condition:
                top_terms = [term for term in sample_terms if any(entry for entry in sample_entries if entry['patient_friendly'] == term and entry.get('condition') == top_condition)]
                if top_terms:
                    sample_terms = [top_terms[0]] + [term for term in sample_terms if term != top_terms[0]]
            options = ', '.join(sample_terms)
            guidance = (
                f"Create exactly two sentences. Sentence 1 must be the open-ended question: '{base_question}'. "
                f"Sentence 2 should gently offer examples starting with 'You can mention things like' followed by up to two of these options: {options}. "
                "Use only the options provided verbatim and do not invent additional examples or wording. "
                "Keep both sentences short, friendly, and avoid adding extra options or clauses."
            )
            return guidance, base_question, sample_terms
 
        guidance = (
            f"Ask exactly one friendly, open-ended sentence: '{base_question}'. Do not add examples or extra sentences."
        )
        return guidance, base_question, []
 
    def _build_yes_no_guidance(self, field: str, term: Optional[str]) -> str:
        term_text = term or "this symptom"
        if field == 'associated':
            context = "an associated symptom that might support a diagnosis"
        elif field == 'red_flags':
            context = "an urgent warning sign that could indicate an emergency"
        else:
            context = "this symptom"

        return (
            f"Create exactly one short, patient-friendly yes/no question to ask whether the patient has experienced \"{term_text}\". "
            f"Keep it conversational and supportive, but do not add explanations, examples, or multiple sentences. "
            f"Focus on confirming if the patient currently has {context}. "
            "Return only the question ending with a question mark."
        )

    def _ensure_binary_prompt_format(self, prompt: str, fallback: str) -> str:
        if not prompt:
            return fallback
        prompt = prompt.strip()
        prompt = prompt.split('\n')[0].strip()
        if prompt.endswith('yes/no'):
            prompt = prompt[:-6].rstrip()
        if not prompt.endswith('?'):
            prompt = prompt.rstrip('.')
            prompt = prompt + '?'
        return f"{prompt} (yes/no)"

    def _compose_binary_question(
        self,
        session: "MedicalSession",
        field: str,
        entry: Dict[str, Any],
        mode: str,
        fallback_template: str,
    ) -> Dict[str, Any]:
        term = (entry.get('patient_term') or '').strip()
        if not term:
            term = 'this symptom'
        guidance = self._build_yes_no_guidance(field, term)
        prompt = self._generate_question(
            session=session,
            section='hpi',
            field=field,
            guidance=guidance,
            base_question=None,
            options=None,
        )
        prompt = self._ensure_binary_prompt_format(prompt, f"Have you noticed {term}? (yes/no)")

        return {
            'section': 'hpi',
            'field': field,
            'prompt': prompt,
            'guidance': guidance,
            'base_question': None,
            'options': [],
            'mode': mode,
        }

    def _last_exchange(self, session: "MedicalSession") -> Tuple[Optional[str], Optional[str]]:
        last_user = None
        last_assistant = None
        for msg in reversed(session.messages):
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user' and last_user is None:
                last_user = content
            elif role == 'assistant' and last_assistant is None:
                last_assistant = content
            if last_user and last_assistant:
                break
        return last_assistant, last_user

    def _collect_character_tags(self, session: "MedicalSession") -> set:
        summary = self._character_tag_summary(session)
        return summary.get('tags', set())

    def _character_tag_summary(self, session: "MedicalSession") -> Dict[str, Any]:
        cache = session.context['guideline_terms'].get('character_tag_summary')
        if cache is not None:
            return cache

        includes = self._get_element_includes(session, 'character')
        top_conditions = [name.lower() for name in self._top_condition_names(session, limit=5)]
        top_counts = {'visual': 0, 'sensory': 0}
        all_counts = {'visual': 0, 'sensory': 0}
        tag_set: set = set()

        for entry in includes:
            condition_name = (entry.get('condition') or '').lower()
            entry_tags = entry.get('question_tags') or []
            normalized_tags = []
            for tag in entry_tags:
                if isinstance(tag, str):
                    tag_lower = tag.lower()
                    if tag_lower in ('visual', 'sensory'):
                        normalized_tags.append(tag_lower)
                        tag_set.add(tag_lower)
                        all_counts[tag_lower] += 1
                        if condition_name in top_conditions:
                            top_counts[tag_lower] += 1

        dominant_tag = None
        if sum(top_counts.values()) > 0:
            if top_counts['visual'] > top_counts['sensory']:
                dominant_tag = 'visual'
            elif top_counts['sensory'] > top_counts['visual']:
                dominant_tag = 'sensory'
        if dominant_tag is None and sum(all_counts.values()) > 0:
            if all_counts['visual'] > all_counts['sensory']:
                dominant_tag = 'visual'
            elif all_counts['sensory'] > all_counts['visual']:
                dominant_tag = 'sensory'

        summary = {
            'tags': set(tag_set),
            'top_counts': top_counts,
            'all_counts': all_counts,
            'dominant': dominant_tag,
        }
        session.context['guideline_terms']['character_tag_summary'] = summary
        session.context['guideline_terms']['character_tags'] = summary['tags']
        return summary

    def _select_guidance_entries(self, session: "MedicalSession", element: str, entries: List[Dict[str, Any]], limit: int = 2, debug_entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Select patient-friendly terms for question guidance, preferring emergent conditions."""
        if limit <= 0 or not entries:
            if debug_entries is not None:
                debug_entries.append({'note': 'no unique entries after filtering'})
            return []
 
        sorted_conditions = sorted(session.condition_scores.items(), key=lambda item: item[1], reverse=True)
        allowed_conditions = self._priority_condition_set(session)
        top_condition_name = sorted_conditions[0][0] if sorted_conditions else None
        if top_condition_name and top_condition_name not in allowed_conditions:
            allowed_conditions = allowed_conditions | {top_condition_name}
        top_conditions = [name for name, _ in sorted_conditions[:5]]
        top_condition_set = {name.lower() for name in top_conditions}
 
        if debug_entries is not None and top_conditions:
            debug_entries.append({
                'top_ranked_conditions': top_conditions,
                'allowed_conditions': list(allowed_conditions),
            })
 
        unique_entries: List[Dict[str, Any]] = []
        seen_pf = set()
        for entry in entries:
            pf = entry.get('patient_friendly')
            if not isinstance(pf, str):
                continue
            cleaned = pf.strip()
            if not cleaned:
                continue
            # Determine if this entry belongs to a guideline whose character terms carry sensory/visual tags
            guideline_condition = entry.get('condition')
            guideline_character_labels = self._get_guideline_character_tags(session, guideline_condition)
            if element == 'location' and 'visual' in guideline_character_labels:
                if debug_entries is not None:
                    debug_entries.append({
                        'term': cleaned,
                        'condition': guideline_condition,
                        'reason': 'excluded_visual_character',
                        'character_tags': list(guideline_character_labels),
                    })
                continue
            key = cleaned.lower()
            if key in seen_pf:
                continue
            seen_pf.add(key)
            copy_entry = dict(entry)
            copy_entry['patient_friendly'] = cleaned
            condition_name = copy_entry.get('condition')
            condition_lower = condition_name.lower() if isinstance(condition_name, str) else None
            condition_score = session.condition_scores.get(condition_name, 0.5) if condition_name else 0.5
            copy_entry['condition_score'] = condition_score
            copy_entry['from_priority_condition'] = bool(condition_name in allowed_conditions)
            copy_entry['from_top_condition'] = bool(condition_lower and condition_lower in top_condition_set)
            copy_entry['character_tags'] = list(guideline_character_labels)
            unique_entries.append(copy_entry)
 
        if not unique_entries:
            if debug_entries is not None:
                debug_entries.append({'note': 'no unique entries after filtering'})
            return []
 
        filtered_entries = [entry for entry in unique_entries if not allowed_conditions or entry.get('condition') in allowed_conditions]
        if filtered_entries:
            unique_entries = filtered_entries

        def split_priority(pool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            priority = [entry for entry in pool if entry.get('from_priority_condition')]
            secondary = [entry for entry in pool if entry.get('from_top_condition') and entry not in priority]
            rest = [entry for entry in pool if entry not in priority and entry not in secondary]
            return priority, secondary, rest
 
        emergent_entries = [entry for entry in unique_entries if entry.get('emergent_term') or entry.get('condition_urgency') == 'emergent']
        urgent_entries = [entry for entry in unique_entries if entry.get('condition_urgency') == 'urgent' and entry not in emergent_entries]
        other_entries = [entry for entry in unique_entries if entry not in emergent_entries and entry not in urgent_entries]
 
        selected: List[Dict[str, Any]] = []
 
        def choose_from(pool: List[Dict[str, Any]]):
            nonlocal selected
            if len(selected) >= limit or not pool:
                return
            primary_pool, secondary_pool, fallback_pool = split_priority(pool)
            for candidate_pool in (primary_pool, secondary_pool, fallback_pool):
                if len(selected) >= limit:
                    break
                available = [entry for entry in candidate_pool if entry not in selected]
                if not available:
                    continue
                available_sorted = sorted(
                    available,
                    key=lambda entry: (
                        -(1 if entry.get('emergent_term') or entry.get('condition_urgency') == 'emergent' else 0),
                        entry.get('condition_score', 0.5),
                    ),
                    reverse=True,
                )
                needed = limit - len(selected)
                selected.extend(available_sorted[:needed])
                if len(selected) >= limit:
                    break
 
        choose_from(emergent_entries)
        choose_from(urgent_entries)
        choose_from(other_entries)
 
        baseline = 0.5 + 1e-6
        high_conf_selected = [entry for entry in selected if entry.get('condition_score', 0.5) > baseline]
        if high_conf_selected:
            selected = high_conf_selected
        if top_condition_name and not any(entry.get('condition') == top_condition_name for entry in selected):
            top_entries = [entry for entry in unique_entries if entry.get('condition') == top_condition_name]
            if top_entries:
                top_entry = max(top_entries, key=lambda e: e.get('condition_score', 0.5))
                selected = [top_entry] + [entry for entry in selected if entry.get('condition') != top_condition_name]
        selected = selected[:limit]
 
        if debug_entries is not None:
            for entry in selected:
                debug_entries.append({
                    'term': entry.get('patient_friendly'),
                    'condition': entry.get('condition'),
                    'character_tags': entry.get('character_tags'),
                    'from_top_condition': entry.get('from_top_condition'),
                    'selected': True,
                })
 
        return selected

    def _get_guideline_character_tags(self, session: "MedicalSession", condition_name: Optional[str]) -> set:
        if not condition_name:
            return set()
        character_includes = self._get_element_includes(session, 'character')
        tags = set()
        for entry in character_includes:
            if entry.get('condition') != condition_name:
                continue
            entry_tags = entry.get('question_tags') or []
            for tag in entry_tags:
                if isinstance(tag, str):
                    tags.add(tag.lower())
        return tags

    def _get_element_includes(self, session: "MedicalSession", element: str) -> List[Dict[str, Any]]:
        cache = session.context.setdefault('guideline_includes', {})
        if element in cache:
            return cache[element]
        categories = session.context.get('matched_categories') or ['gastrointestinal']
        if element == 'red_flags':
            collected = self._collect_emergent_entries(session, categories)
            cache[element] = collected
            return collected
        collected: List[Dict[str, Any]] = []
        for category in categories:
            guidelines = self._get_guidelines_by_category(category)
            for guideline in guidelines.values():
                condition_name = guideline.get('condition') or guideline.get('data', {}).get('condition') or guideline.get('name', 'Unknown')
                condition_urgency = guideline.get('urgency') or guideline.get('data', {}).get('urgency', '')
                condition_prevalence = guideline.get('prevalence') or guideline.get('data', {}).get('prevalence', '')
                structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                if not structured:
                    structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                element_data = structured.get(element, {})
                include_items = element_data.get('includes', []) if isinstance(element_data, dict) else []
                for entry in include_items:
                    if isinstance(entry, dict):
                        patient_term = entry.get('patient_friendly') or entry.get('medical')
                        medical_term = entry.get('medical')
                        tags = entry.get('question_tags')
                        if tags is None:
                            tags = entry.get('question_tag')
                        if isinstance(tags, str):
                            tag_list = [tags.lower()]
                        elif isinstance(tags, (list, tuple, set)):
                            tag_list = [str(tag).lower() for tag in tags if isinstance(tag, str)]
                        else:
                            tag_list = []
                        emergent_term = bool(entry.get('emergent'))
                        collected.append({
                            'patient_friendly': patient_term.strip() if isinstance(patient_term, str) else '',
                            'medical': medical_term.strip() if isinstance(medical_term, str) else '',
                            'question_tags': tag_list,
                            'emergent_term': emergent_term,
                            'condition': condition_name,
                            'condition_urgency': condition_urgency,
                            'condition_prevalence': condition_prevalence,
                        })
                    elif isinstance(entry, str):
                        stripped = entry.strip()
                        collected.append({
                            'patient_friendly': stripped,
                            'medical': stripped,
                            'question_tags': [],
                            'emergent_term': False,
                            'condition': condition_name,
                            'condition_urgency': condition_urgency,
                            'condition_prevalence': condition_prevalence,
                        })
        cache[element] = collected
        return collected

    def _collect_emergent_entries(self, session: "MedicalSession", categories: List[str]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen: set = set()
        for category in categories:
            guidelines = self._get_guidelines_by_category(category)
            for guideline in guidelines.values():
                condition_name = guideline.get('condition') or guideline.get('data', {}).get('condition') or guideline.get('name', 'Unknown')
                condition_urgency = guideline.get('urgency') or guideline.get('data', {}).get('urgency', '')
                condition_prevalence = guideline.get('prevalence') or guideline.get('data', {}).get('prevalence', '')
                structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                if not structured:
                    structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                for element_name, element_data in structured.items():
                    include_items = element_data.get('includes', []) if isinstance(element_data, dict) else []
                    for entry in include_items:
                        if not isinstance(entry, dict):
                            continue
                        if not entry.get('emergent'):
                            continue
                        patient_term = entry.get('patient_friendly') or entry.get('medical')
                        medical_term = entry.get('medical')
                        if not isinstance(patient_term, str):
                            continue
                        cleaned = patient_term.strip()
                        if not cleaned:
                            continue
                        key = (cleaned.lower(), condition_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        entries.append({
                            'patient_friendly': cleaned,
                            'medical': medical_term.strip() if isinstance(medical_term, str) else '',
                            'question_tags': [],
                            'emergent_term': True,
                            'condition': condition_name,
                            'condition_urgency': condition_urgency,
                            'condition_prevalence': condition_prevalence,
                            'source_element': element_name,
                        })
        if not entries:
            defaults = [
                "fever over 103°F",
                "difficulty breathing",
                "fainting or passing out",
                "severe chest pain",
            ]
            for term in defaults:
                entries.append({
                    'patient_friendly': term,
                    'medical': term,
                    'question_tags': [],
                    'emergent_term': True,
                    'condition': 'General emergency',
                    'condition_urgency': 'emergent',
                    'condition_prevalence': 'unknown',
                    'source_element': 'default',
                })
        return entries

    def _guideline_terms_for_element(self, session: "MedicalSession", element: str) -> List[str]:
        cache = session.context['guideline_terms']
        if element in cache:
            return cache[element]
        includes = self._get_element_includes(session, element)
        terms: List[str] = []
        seen = set()
        for entry in includes:
            term = entry.get('patient_friendly')
            if isinstance(term, str):
                cleaned = term.strip()
                if cleaned and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    terms.append(cleaned)
        cache[element] = terms
        return terms

    def _normalize_subject_for_questions(self, text: Optional[str]) -> str:
        if not text:
            return 'symptoms'
        subject = text.strip()
        lowered = subject.lower()
        prefixes = [
            "i have ",
            "i've got ",
            "i am having ",
            "i'm having ",
            "i am ",
            "i'm ",
            "my ",
            "i feel ",
            "i've been having ",
            "i been having ",
            "i've had ",
            "i had ",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                subject = subject[len(prefix):].strip()
                break
        subject = subject.strip(" .,!?:;")
        if not subject:
            return 'symptoms'
        return subject

    def _ordered_oldcarts_elements(self, session: "MedicalSession") -> List[str]:
        ordered = sorted(self.HPI_ELEMENTS, key=lambda e: self._get_element_weight(session, e), reverse=True)
        ordered_list = ordered.copy()
        if 'associated' in ordered_list:
            ordered_list.remove('associated')
        if 'red_flags' in ordered_list:
            ordered_list.remove('red_flags')
        if 'associated' in ordered:
            ordered_list.append('associated')
        if 'red_flags' in self.HPI_ELEMENTS:
            ordered_list.append('red_flags')
        answered = {key for key, value in session.context['hpi'].items() if value}
        filtered = [element for element in ordered_list if element not in answered]
        return filtered

    def _get_element_weight(self, session: "MedicalSession", element: str) -> float:
        categories = session.context['matched_categories'] or ['gastrointestinal']
        best = self.DEFAULT_ELEMENT_WEIGHT
        for cat in categories:
            cat_weights = self.CATEGORY_ELEMENT_WEIGHTS.get(cat.lower(), {})
            if element in cat_weights:
                best = max(best, cat_weights[element])
        return best

    # ----------- LLM helpers -------------------------------------------------

    def _generate_question(
        self,
        session: "MedicalSession",
        section: str,
        field: str,
        guidance: str,
        base_question: Optional[str] = None,
        options: Optional[List[str]] = None,
    ) -> str:
        if section == 'hpi' and field == 'severity':
            raw_subject = session.context['pre_hpi'].get('chief_complaint')
            subject = self._normalize_subject_for_questions(raw_subject)
            if subject.startswith(('your ', 'the ', 'this ')):
                subject_phrase = subject
            else:
                subject_phrase = f"your {subject}"
            guidance = (
                f"Ask the patient to rate how bad {subject_phrase} is on a scale from 1 to 10. "
                "Return a single concise question ending with a question mark."
            )
            base_question = f"On a scale from 1 to 10, how bad is {subject_phrase} right now?"
        if not self.llm_chat_fn:
            return base_question or guidance or ""
        cc = session.context['pre_hpi'].get('chief_complaint', 'your symptoms') or 'your symptoms'
        last_assistant, last_user = self._last_exchange(session)
        if section == 'hpi':
            exchange_lines = []
            if last_assistant:
                exchange_lines.append(f"assistant: {last_assistant}")
            if last_user:
                exchange_lines.append(f"user: {last_user}")
            recent = "\n".join(exchange_lines)
            element_instruction = (
                f"Ask exactly one question about the '{field}' aspect of the chief complaint. "
                "Do not repeat demographic questions (age, sex, chronicity) or acknowledge prior demographic answers. "
                "Do not repeat previous questions verbatim."
            )
        else:
            exchange_lines = []
            if last_assistant:
                exchange_lines.append(f"assistant: {last_assistant}")
            if last_user:
                exchange_lines.append(f"user: {last_user}")
            recent = "\n".join(exchange_lines)
            element_instruction = ""
        if section == 'pre_hpi' and field == 'chief_complaint':
            guidance = (
                "Greet the patient warmly (e.g., 'Hi there, it's nice to meet you.') and ask what brings them in today"
                " and for how long. Return one friendly sentence combining greeting and question."
            )
        extra_rules = ""
        if section == 'hpi':
            extra_rules = (
                "Address only the specified OLDCARTS element for the chief complaint."
            )
        elif section == 'pre_hpi' and field in {'age', 'sex'}:
            extra_rules = "Ask this demographic question plainly in one sentence without additional commentary."
        user_prompt = (
            f"Section: {section}\n"
            f"Field: {field}\n"
            f"Chief complaint: {cc}\n"
            f"Guidance: {guidance}\n"
            f"Element instructions: {element_instruction}\n"
            f"Additional instructions: {extra_rules}\n"
            f"Recent conversation:\n{recent}\n"
            "Return one concise question that follows the guidance without any preface."
        )
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=60,
            temperature=0.5,
        )
        self._capture_debug(f"[LLM] ❓ Question prompt:\n{user_prompt}")
        self._capture_debug(f"[LLM] ❓ Raw question response: {response}")
        cleaned = self._clean_llm_response(response)
        if not cleaned:
            raise ValueError(f"LLM returned empty question for {field}")
        if base_question and base_question.lower() not in cleaned.lower():
            cleaned = base_question
        if section == 'hpi' and options:
            question_part = base_question or cleaned
            question_part = question_part.strip()
            if not question_part.endswith('?'):
                question_part = question_part.rstrip('.') + '?'
            option_text = ', '.join(options)
            cleaned = f"{question_part} You can mention things like: {option_text}."
        elif not cleaned.endswith('?'):
            cleaned = cleaned.rstrip('.') + '?'
        return cleaned

    def _generate_empathetic_statement(self, chief_complaint: str) -> str:
        if not self.llm_chat_fn:
            return "I'm here to help you with that."
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.EMPATHETIC_SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient concern: {chief_complaint}"},
            ],
            max_tokens=80,
            temperature=0.4,
        )
        self._capture_debug(f"[LLM] ❤️ Empathy prompt: Patient concern: {chief_complaint}")
        self._capture_debug(f"[LLM] ❤️ Empathy response: {response}")
        return self._clean_llm_response(response, fallback="I'm sorry you're dealing with that." )

    def _generate_chronicity_question(self) -> str:
        if not self.llm_chat_fn:
            return "Is this a new problem or something you've experienced before?"
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.CHRONICITY_SYSTEM_PROMPT},
                {"role": "user", "content": "Ask if the problem is new or ongoing, and if there's a prior diagnosis."},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        self._capture_debug("[LLM] 🕒 Chronicity prompt issued.")
        self._capture_debug(f"[LLM] 🕒 Chronicity response: {response}")
        return self._clean_llm_response(response, fallback="Is this a new problem or something you've had before with a diagnosis?")

    def _generate_summary(self, session: "MedicalSession") -> str:
        if not self.llm_chat_fn:
            return "History collection complete."
        pre = session.context['pre_hpi']
        hpi = session.context['hpi']
        pmh = session.context['pmh']
        rankings = session.condition_rankings[:3]
        ranking_text = ", ".join(f"{name} ({score:.0%})" for name, score in rankings) if rankings else "No ranked conditions yet"
        user_prompt = (
            f"Chief complaint: {pre.get('chief_complaint', 'Not stated')}\n"
            f"Chronicity: {pre.get('chronicity', 'Unknown')}\n"
            f"Age: {pre.get('age', 'Unknown')}\n"
            f"Biological sex: {pre.get('sex', 'Unknown')}\n"
            f"OLDCARTS responses: {hpi}\n"
            f"PMH/PSH/Meds/Allergies: {pmh}\n"
            f"Top differentials: {ranking_text}\n"
            "Summarise as bullet points."
        )
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        self._capture_debug(f"[LLM] 📝 Summary prompt:\n{user_prompt}")
        self._capture_debug(f"[LLM] 📝 Summary response: {response}")
        return response.strip() if response else "History collection complete."

    # ----------- Validation / Clarification ----------------------------------

    def _clean_llm_response(self, text: Optional[str], fallback: str = "") -> str:
        if not text:
            return fallback
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", text.strip())
        cleaned = cleaned.strip('"')
        return cleaned or fallback

    def _requires_clarification(self, element: str) -> bool:
        no_clarification_elements = {
            'onset',
            'progression',
            'duration',
            'timing',
            'severity',
            'associated',
            'character',
            'abruptness',
            'frequency',
            'red_flags',
        }
        return element not in no_clarification_elements

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
            'intersex/non-binary': {'intersex', 'nonbinary', 'non-binary', 'nb', 'enby', 'genderqueer'},
            'unspecified': {'prefer not to say', 'decline', 'undisclosed', 'unknown', 'unsure', 'not sure'},
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
        greetings = {'hi', 'hello', 'hey', 'hey there', 'good morning', 'good afternoon', 'good evening', 'hola', 'sup', 'yo'}
        return normalized in greetings

    # ----------- Debug helpers ----------------------------------------------

    def _capture_debug(self, message: str) -> None:
        self._captured_debug_output.append(message)
    
    def _format_engine_debug(self, session: "MedicalSession") -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("[Telegram] 🧠 ENGINE DEBUG OUTPUT")
        lines.append("=" * 80)
        lines.append(f"[Engine] 🎯 Structured guidelines: Active={len(session.active_conditions)}, Reserve={len(session.reserve_conditions)}")
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
        debug_ctx = session.context.get('debug', {})
        validation_error = debug_ctx.get('last_validation_error')
        if validation_error:
            lines.append(f"[Engine] ⚠️ Validation: {validation_error}")
        review = debug_ctx.get('last_llm_review')
        if review:
            rows = review.get('rows', [])
            requested = review.get('requested_terms', [])
            raw_response = review.get('raw_response', '')
            had_scores = review.get('had_scores', False)
            lines.append(f"[LLM] 🔍 Match review ({review['element']}) for '{review['answer']}':")
            lines.append(f"[LLM]   • Requested terms: {requested if requested else 'none'}")
            if rows:
                for term, unused_score, llm_score, final_score in rows:
                    # Note: first score is always 0.0 (FAISS bypassed), second is LLM score, third is final (LLM-only)
                    lines.append(
                        f"[LLM]   • {term}: LLM={llm_score:.3f}, final={final_score:.3f}"
                    )
            else:
                lines.append("[LLM]   • LLM returned no scores.")
                if raw_response:
                    lines.append(f"[LLM]   • Raw response (first 300 chars): {raw_response[:300]}")
                else:
                    lines.append("[LLM]   • Raw response: EMPTY or not captured")
                lines.append(f"[LLM]   • Had scores: {had_scores}")
                lines.append("[LLM]   • Note: LLM-only approach - no FAISS, all matching done by LLM")
        lines.append(self._format_rankings_debug(session))
        return '\n'.join(lines)

    def _format_rankings_debug(self, session: "MedicalSession") -> str:
        lines = []
        lines.append("[Engine] 📊 UPDATED RANKINGS:")
        for idx, (name, score) in enumerate(session.active_conditions[:5], start=1):
            pct = round(score * 100, 1)
            lines.append(f"[Engine]   {idx}. {name}: {pct}% 📋")
            lines.append(f"[Scoring] 🏆 Top {idx}: {name}")
            lines.append(f"[Scoring]   📊 Score: {pct}%")
            lines.append(f"[Scoring]   🎯 ML Confidence: High similarity match")
            lines.append(f"[Scoring]   🚨 Urgency: unknown")
        lines.append("")
        lines.append(f"[Engine] 🔄 Pool status: Active={len(session.active_conditions)}, Reserve={len(session.reserve_conditions)}, Ruled out=0")
        lines.append("[Scoring] 📊 Final statistics:")
        lines.append(f"[Scoring]   🎯 Active Conditions: {len(session.active_conditions)}")
        lines.append(f"[Scoring]   📋 Reserve Conditions: {len(session.reserve_conditions)}")
        lines.append(f"[Scoring]   ❌ Ruled Out: 0")
        total = len(session.active_conditions) + len(session.reserve_conditions)
        lines.append(f"[Scoring]   📈 Total Processed: {total}")
        return '\n'.join(lines)

    def _update_condition_pools(self, session: "MedicalSession") -> None:
        active = session.condition_rankings[:5]
        reserve = session.condition_rankings[5:]
        previous_active = session.previous_active
        current_active = {name for name, _ in active}
        promotions = current_active - previous_active
        demotions = previous_active - current_active

        if promotions:
            self._capture_debug("\n[Engine] 🔼 PROMOTED to active:")
            for name in promotions:
                score = next((score for cond, score in active if cond == name), 0.0)
                pct = round(score * 100, 1)
                self._capture_debug(f"[Engine]   ↑ {name} (score: {pct}%)")

        if demotions:
            self._capture_debug("\n[Engine] 🔽 DEMOTED to reserve:")
            for name in demotions:
                score = next((score for cond, score in reserve if cond == name), 0.0)
                pct = round(score * 100, 1)
                self._capture_debug(f"[Engine]   ↓ {name} (score: {pct}%)")

        session.active_conditions = active
        session.reserve_conditions = reserve
        session.previous_active = current_active

    def _log_rankings(self, session: "MedicalSession") -> None:
        self._capture_debug("\n[Engine] 📊 UPDATED RANKINGS:")
        for idx, (name, score) in enumerate(session.active_conditions[:5], start=1):
            pct = round(score * 100, 1)
            self._capture_debug(f"[Engine]   {idx}. {name}: {pct}% 📋")
            self._capture_debug(f"[Scoring] 🏆 Top {idx}: {name}")
            self._capture_debug(f"[Scoring]   📊 Score: {pct}%")
            self._capture_debug(f"[Scoring]   📋 Prevalence: unknown")
            self._capture_debug(f"[Scoring]   🎯 ML Confidence: High similarity match")
            self._capture_debug(f"[Scoring]   🚨 Urgency: unknown")
        self._capture_debug(f"\n[Engine] 🔄 Pool status: Active={len(session.active_conditions)}, Reserve={len(session.reserve_conditions)}, Ruled out=0")
        self._capture_debug("[Scoring] 📊 Final statistics:")
        self._capture_debug(f"[Scoring]   🎯 Active Conditions: {len(session.active_conditions)}")
        self._capture_debug(f"[Scoring]   📋 Reserve Conditions: {len(session.reserve_conditions)}")
        self._capture_debug(f"[Scoring]   ❌ Ruled Out: 0")
        total = len(session.active_conditions) + len(session.reserve_conditions)
        self._capture_debug(f"[Scoring]   📈 Total Processed: {total}")

    def _extract_similarity(self, matches: List[Dict], condition: str) -> float:
        if not matches:
            return 0.0
        for match in matches:
            cond = match.get('condition') or match.get('name')
            if cond == condition:
                return float(match.get('score', 0.0))
        return 0.0

    def _get_oldcarts_status(self, session: "MedicalSession") -> Tuple[List[str], List[str], Optional[str]]:
        hpi_answers = session.context.get('hpi', {})
        satisfied = [element for element in self.HPI_ELEMENTS if element in hpi_answers]
        missing = [element for element in self.HPI_ELEMENTS if element not in hpi_answers]
        current = None
        if session.pending and session.pending.get('section') == 'hpi':
            current = session.pending.get('field')
        return satisfied, missing, current

    def _get_pre_hpi_status(self, session: "MedicalSession") -> Tuple[List[str], List[str]]:
        pre = session.context.get('pre_hpi', {})
        collected = [item for item in self.PRE_HPI_ORDER if pre.get(item)]
        missing = [item for item in self.PRE_HPI_ORDER if item not in collected]
        return collected, missing

    def _is_confused_response(self, text: str) -> bool:
        if not text:
            return False
        normalized = text.strip().lower()
        confusion_markers = [
            "what do you mean",
            "i don't understand",
            "can you explain",
            "not sure",
            "clarify",
        ]
        if any(marker in normalized for marker in confusion_markers):
            return True
        if normalized.endswith('?') and len(normalized) <= 60:
            return True
        return False

    def _clarify_element_question(self, session: "MedicalSession", element: str, pending: Optional[Dict[str, Any]] = None) -> str:
        pending = pending or session.pending or {}
        prompt_text = pending.get('prompt')
        base_question = pending.get('base_question')
        options = pending.get('options')
        cc_subject = self._normalize_subject_for_questions(session.context['pre_hpi'].get('chief_complaint'))

        if not base_question and element in self.HPI_BASE_GUIDANCE:
            base_question = self.HPI_BASE_GUIDANCE[element].replace('{cc}', cc_subject)

        if not prompt_text and base_question:
            prompt_text = base_question if base_question.endswith('?') else f"{base_question}?"
            if options:
                option_text = ', '.join(options[:2])
                prompt_text = f"{prompt_text} You can mention things like: {option_text}."

        if element == 'red_flags':
            includes = self._get_element_includes(session, 'red_flags')
            emergent_terms = [entry['patient_friendly'] for entry in includes if entry.get('emergent_term')]
            if not emergent_terms:
                emergent_terms = [entry['patient_friendly'] for entry in includes[:3]]
            emergent_terms = [term for term in emergent_terms if term]
            if emergent_terms:
                examples = ', '.join(emergent_terms[:4])
                guidance = f"By urgent warning signs I mean things like {examples}."
                if prompt_text:
                    return f"{guidance} {prompt_text}"
                return f"{guidance} Have you noticed any of those?"
            if prompt_text:
                return f"I'm checking for any severe or alarming symptoms. {prompt_text}"
            return "I'm asking if you've noticed any severe or alarming symptoms that might need urgent attention."

        if prompt_text:
            return f"Sure—I'm asking: {prompt_text}"

        return "Could you share a bit more detail about that?"

    def _filter_options_by_condition_scores(
        self,
        session: "MedicalSession",
        element: str,
        options: List[str],
        term_to_conditions: Optional[Dict[str, List[str]]] = None,
        allowed_conditions: Optional[set] = None,
    ) -> List[str]:
        if not options:
            return options
 
        baseline = 0.5 + 1e-6
        prioritized: List[str] = []
        fallback: List[str] = []
        skipped: List[str] = []
        mapping = term_to_conditions or {}
        top_condition = None
        if allowed_conditions:
            top_condition = next(iter(sorted(allowed_conditions, key=lambda name: session.condition_scores.get(name, 0.0), reverse=True)), None)
        elif session.condition_scores:
            top_condition = max(session.condition_scores.items(), key=lambda item: item[1])[0]
 
        for opt in options:
            conds = mapping.get(opt.lower(), [])
            if allowed_conditions:
                conds = [cond for cond in conds if cond in allowed_conditions]
            if not conds:
                fallback.append(opt)
                continue
            if any(session.condition_scores.get(cond, 0.5) > baseline for cond in conds):
                prioritized.append(opt)
            else:
                skipped.append(opt)
 
        if skipped:
            self._capture_debug(
                f"[Clarification] ⚖️ Skipping options for {element} tied to baseline scores: {skipped}"
            )
 
        if top_condition:
            top_options = [opt for opt in prioritized if top_condition in mapping.get(opt.lower(), [])]
            if not top_options:
                extra_top = [opt for opt in fallback if top_condition in mapping.get(opt.lower(), [])]
                if extra_top:
                    prioritized = [extra_top[0]] + prioritized
                else:
                    for opt, conds in mapping.items():
                        if conds and top_condition in conds and opt not in prioritized:
                            prioritized = [opt] + prioritized
                            break
 
        if prioritized:
            return prioritized
        if fallback:
            return fallback
 
        return options

    def _ensure_associated_state(self, session: "MedicalSession") -> Dict[str, Any]:
        state = session.context.setdefault('associated_state', {})
        state.setdefault('queue', [])
        state.setdefault('index', 0)
        state.setdefault('positives', [])
        state.setdefault('negatives', [])
        state.setdefault('current', None)
        return state

    def _prepare_associated_queue(self, session: "MedicalSession") -> None:
        state = self._ensure_associated_state(session)
        if state['queue']:
            return

        includes = self._get_element_includes(session, 'associated')
        baseline = 0.5 + 1e-6
        priority_bucket: List[Dict[str, Any]] = []
        fallback_bucket: List[Dict[str, Any]] = []
        seen_terms: set = set()

        target_condition = None
        sorted_scores = sorted(session.condition_scores.items(), key=lambda item: item[1], reverse=True)
        if sorted_scores:
            target_condition = sorted_scores[0][0]

        filtered_entries = []
        if target_condition:
            for entry in includes:
                if entry.get('condition') == target_condition:
                    filtered_entries.append(entry)
        if not filtered_entries:
            filtered_entries = includes

        for entry in filtered_entries:
            patient_term = (entry.get('patient_friendly') or entry.get('medical') or '').strip()
            if not patient_term:
                continue
            key = patient_term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            condition_name = entry.get('condition')
            score = session.condition_scores.get(condition_name, 0.5) if condition_name else 0.5
            item = {
                'patient_term': patient_term,
                'medical_term': entry.get('medical') or patient_term,
                'condition': condition_name,
                'score': score,
                'emergent': bool(entry.get('emergent_term') or entry.get('condition_urgency') == 'emergent'),
            }
            if score > baseline:
                priority_bucket.append(item)
            else:
                fallback_bucket.append(item)

        def sort_entries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(
                items,
                key=lambda x: (
                    -(1 if x['emergent'] else 0),
                    -x['score'],
                    x['patient_term'],
                ),
            )

        queue = sort_entries(priority_bucket) + sort_entries(fallback_bucket)
        state['queue'] = queue
        state['index'] = 0
        state['positives'] = []
        state['negatives'] = []
        state['current'] = None

    def _prepare_next_associated_question(self, session: "MedicalSession") -> Optional[Dict[str, Any]]:
        self._prepare_associated_queue(session)
        state = self._ensure_associated_state(session)
        queue = state.get('queue', [])
        index = state.get('index', 0)

        if index >= len(queue):
            return None

        entry = queue[index]
        state['current'] = entry
        return self._compose_binary_question(
            session=session,
            field='associated',
            entry=entry,
            mode='associated_sequence',
            fallback_template="Have you noticed {term}? (yes/no)",
        )

    def _advance_associated_queue(self, session: "MedicalSession") -> Optional[Dict[str, Any]]:
        state = self._ensure_associated_state(session)
        state['index'] = state.get('index', 0) + 1
        if state['index'] >= len(state.get('queue', [])):
            state['current'] = None
        return None
        entry = state['queue'][state['index']]
        state['current'] = entry
        return self._compose_binary_question(
            session=session,
            field='associated',
            entry=entry,
            mode='associated_sequence',
            fallback_template="Have you noticed {term}? (yes/no)",
        )

    def _handle_associated_answer(
        self,
        session: "MedicalSession",
        answer: str,
    ) -> Dict[str, Any]:
        state = self._ensure_associated_state(session)
        entry = state.get('current')
        if not entry:
            return {'completed': True}

        normalized = answer.strip().lower()
        positive_markers = {'yes', 'y', 'yeah', 'yep', 'affirmative', 'correct', 'sure', 'absolutely', 'definitely'}
        negative_markers = {'no', 'n', 'not really', 'nope', 'nah', 'none', 'negative'}

        positive = False
        negative = False
        if normalized in positive_markers or normalized.startswith('yes'):
            positive = True
        elif normalized in negative_markers or normalized.startswith('no'):
            negative = True

        result: Dict[str, Any] = {
            'score_text': None,
            'next_question': None,
            'completed': False,
        }

        if positive:
            state['positives'].append(entry['patient_term'])
            hpi_assoc = session.context['hpi'].get('associated')
            if not isinstance(hpi_assoc, list):
                hpi_assoc = []
            if entry['patient_term'] not in hpi_assoc:
                hpi_assoc.append(entry['patient_term'])
            session.context['hpi']['associated'] = hpi_assoc
            result['score_text'] = entry['patient_term']
        elif negative:
            state['negatives'].append(entry['patient_term'])
        else:
            # Treat ambiguous answer as positive evidence text for scoring
            result['score_text'] = answer

        next_question = self._advance_associated_queue(session)
        if next_question:
            result['next_question'] = next_question
        else:
            result['completed'] = True
            state['current'] = None
            state['queue'] = []
            state['index'] = 0

        result['positives'] = list(state.get('positives', []))
        result['negatives'] = list(state.get('negatives', []))

        return result

    def _ensure_red_flag_state(self, session: "MedicalSession") -> Dict[str, Any]:
        state = session.context.setdefault('red_flag_state', {})
        state.setdefault('queue', [])
        state.setdefault('index', 0)
        state.setdefault('positives', [])
        state.setdefault('negatives', [])
        state.setdefault('current', None)
        return state

    def _prepare_red_flag_queue(self, session: "MedicalSession") -> None:
        state = self._ensure_red_flag_state(session)
        if state['queue']:
            return

        includes = self._get_element_includes(session, 'red_flags')
        seen_terms: set = set()
        queue: List[Dict[str, Any]] = []

        for entry in includes:
            patient_term = (entry.get('patient_friendly') or entry.get('medical') or '').strip()
            if not patient_term:
                continue
            key = patient_term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            condition_name = entry.get('condition')
            score = session.condition_scores.get(condition_name, 0.5) if condition_name else 0.5
            queue.append({
                'patient_term': patient_term,
                'medical_term': entry.get('medical') or patient_term,
                'condition': condition_name,
                'score': score,
                'urgency': entry.get('condition_urgency', ''),
                'source_element': entry.get('source_element') or 'associated',
            })

        if not queue:
            defaults = [
                "high fever over 103°F",
                "shortness of breath",
                "fainting or passing out",
                "severe chest pain",
            ]
            for term in defaults:
                queue.append({
                    'patient_term': term,
                    'medical_term': term,
                    'condition': 'General emergency',
                    'score': 0.5,
                    'urgency': 'emergent',
                    'source_element': 'associated',
                })

        queue.sort(key=lambda item: (-item['score'], item['patient_term']))

        state['queue'] = queue
        state['index'] = 0
        state['positives'] = []
        state['negatives'] = []
        state['current'] = queue[0] if queue else None

    def _prepare_next_red_flag_question(self, session: "MedicalSession") -> Optional[Dict[str, Any]]:
        self._prepare_red_flag_queue(session)
        state = self._ensure_red_flag_state(session)
        queue = state.get('queue', [])
        index = state.get('index', 0)

        if index >= len(queue):
            return None

        entry = queue[index]
        state['current'] = entry
        return self._compose_binary_question(
            session=session,
            field='red_flags',
            entry=entry,
            mode='red_flag_sequence',
            fallback_template="Have you experienced {term}? (yes/no)",
        )

    def _advance_red_flag_queue(self, session: "MedicalSession") -> Optional[Dict[str, Any]]:
        state = self._ensure_red_flag_state(session)
        state['index'] = state.get('index', 0) + 1
        queue = state.get('queue', [])
        if state['index'] >= len(queue):
            state['current'] = None
            return None
        entry = queue[state['index']]
        state['current'] = entry
        return self._compose_binary_question(
            session=session,
            field='red_flags',
            entry=entry,
            mode='red_flag_sequence',
            fallback_template="Have you experienced {term}? (yes/no)",
        )

    def _handle_red_flag_answer(
        self,
        session: "MedicalSession",
        answer: str,
    ) -> Dict[str, Any]:
        state = self._ensure_red_flag_state(session)
        entry = state.get('current')
        if not entry:
            return {'completed': True}

        normalized = answer.strip().lower()
        positive_markers = {'yes', 'y', 'yeah', 'yep', 'affirmative', 'correct', 'sure', 'absolutely', 'definitely', 'i have'}
        negative_markers = {'no', 'n', 'not really', 'nope', 'nah', 'none', 'negative'}

        positive = False
        negative = False
        if normalized in positive_markers or normalized.startswith('yes'):
            positive = True
        elif normalized in negative_markers or normalized.startswith('no'):
            negative = True

        result: Dict[str, Any] = {
            'score_text': None,
            'next_question': None,
            'completed': False,
        }

        if positive:
            state['positives'].append(entry['patient_term'])
            result['score_text'] = entry['patient_term']
        elif negative:
            state['negatives'].append(entry['patient_term'])
        else:
            result['score_text'] = answer

        next_question = self._advance_red_flag_queue(session)
        if next_question:
            result['next_question'] = next_question
        else:
            result['completed'] = True
            state['current'] = None
            state['queue'] = []
            state['index'] = 0

        result['positives'] = list(state.get('positives', []))
        result['negatives'] = list(state.get('negatives', []))
        result['source_element'] = entry.get('source_element')

        return result

    def _priority_condition_set(self, session: "MedicalSession") -> set:
        baseline = 0.5 + 1e-6
        sorted_conditions = sorted(session.condition_scores.items(), key=lambda item: item[1], reverse=True)
        if not sorted_conditions:
            return set()

        top_name, top_score = sorted_conditions[0]
        second_score = sorted_conditions[1][1] if len(sorted_conditions) > 1 else None

        if second_score is None or top_score - second_score >= self.CLEAR_LEAD_MARGIN:
            return {top_name}

        priority = {name for name, score in sorted_conditions if score > baseline}
        if priority:
            return priority

        if len(sorted_conditions) > 1:
            return {top_name, sorted_conditions[1][0]}

        return {top_name}
