#!/usr/bin/env python3
"""
Advanced Medical Navigator (Guideline-aware, LLM-first)
=======================================================

Conversation flow:
    1. Capture chief complaint → fuzzy + semantic trigger matching (same logic as adaptive engine)
       • Loads guideline categories & seeds condition scores.
    2. LLM empathetic acknowledgement + chronicity question (new vs known w/ prior Dx)
    3. Collect demographics: age, biological sex
    4. OLDCARTS assessment using guideline terms & weights per category
       • LLM crafts questions with injected options
       • Responses scored via FAISS against patient-friendly terms
       • Clarifying questions generated when multiple / no matches
    5. Rankings update after every element; diagnosis ready once OLDCARTS complete.

This file intentionally focuses on conversational logic; heavy lifting (FAISS, anatomical
rules, guideline storage) relies on `ml.medical_rule_engine.MedicalRuleEngine`.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import faiss
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
        "location",
        "duration",
        "character",
        "aggravating",
        "relieving",
        "timing",
        "severity",
        "associated",
        "red_flags",
    ]

    HPI_BASE_GUIDANCE = {
        "onset": "When did this start, and did it come on suddenly or gradually?",
        "location": "Where exactly is your {cc} located?",
        "duration": "How long does each episode typically last?",
        "character": "How would you describe what it feels like?",
        "aggravating": "What tends to make it worse?",
        "relieving": "What tends to make it better?",
        "timing": "Does it happen at particular times or during specific activities?",
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

    CHIEF_COMPLAINT_FAISS_THRESHOLD = 0.6
    CHIEF_COMPLAINT_NEAR_MISS_UPPER = 0.5
    CHIEF_COMPLAINT_NEAR_MISS_LOWER = 0.4
    CHIEF_COMPLAINT_FUZZY_THRESHOLD = 0.55

    RULE_OUT_THRESHOLD = 0.05

    PMH_ELEMENTS = ["pmh", "psh", "meds_allergies"]
    PMH_PROMPTS = {
        "pmh": "Do you have any existing medical conditions?",
        "psh": "Have you had any surgeries in the past?",
        "meds_allergies": "What medications do you take, and do you have any medication allergies?",
    }

    QUESTION_SYSTEM_PROMPT = (
        "You are a compassionate medical assistant conducting a medical interview."
        " Use the guidance to craft one friendly question."
        " Return ONLY the question text (no prefixes, no reasoning)."
        " Keep it ≤20 words and honor any provided options."
    )

    EMPATHETIC_SYSTEM_PROMPT = (
        "You are an empathetic medical assistant. Craft a 1–2 sentence acknowledgment"
        " that validates the patient's concern and expresses willingness to help."
        " Do not mention checking vitals, running tests, or any clinical actions—you are offering"
        " emotional support only."
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
            'character': 0.20,
            'aggravating': 0.30,
            'relieving': 0.30,
            'onset': 0.25,
            'timing': 0.25,
            'duration': 0.25,
            'severity': 0.20,
            'associated': 0.25,
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
        },
    }

    DEFAULT_ELEMENT_WEIGHT = 0.30

    RELAXED_LOCATION_THRESHOLD = 0.55
    RELAXED_LOCATION_MARGIN = 0.02
    LOCATION_HIGH_CONFIDENCE_THRESHOLD = 0.9

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
        })
        condition_scores: Dict[str, float] = field(default_factory=dict)
        condition_rankings: List[Tuple[str, float]] = field(default_factory=list)
        active_conditions: List[Tuple[str, float]] = field(default_factory=list)
        reserve_conditions: List[Tuple[str, float]] = field(default_factory=list)
        previous_active: set = field(default_factory=set)
        oldcarts_remaining: List[str] = field(default_factory=list)
        completed: bool = False

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

    def __init__(self, llm_chat_fn, medical_rule_engine=None, embedding_model=None):
        self.llm_chat_fn = llm_chat_fn
        self.medical_rule_engine = medical_rule_engine
        self.embedding_model = embedding_model
        self.sessions: Dict[str, AdvancedMedicalNavigator.MedicalSession] = {}
        self._captured_debug_output: List[str] = []
        self.guidelines_dir = self._resolve_guidelines_dir()
        self.enabled_categories = self._get_enabled_categories()
        self.all_guidelines: Dict[str, Dict] = {}
        self.chief_complaint_triggers_data: List[Dict] = []
        self.chief_complaint_triggers_index = None

        if self.guidelines_dir:
            self._load_guidelines()
            self._build_chief_complaint_index()
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

        if not self.medical_rule_engine or not self.embedding_model:
            raise ValueError("Medical rule engine with embedding model is required for chief complaint matching.")

        complaint = self._fuzzy_correct(text)
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (ADVANCED NAVIGATOR)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{complaint}'")
        categories = self._match_chief_complaint_to_category(complaint)
        if not categories:
            apology = (
                "I'm not sure I caught that. Could you tell me a bit more about what's bothering you, "
                "like 'I have stomach pain' or 'I'm feeling short of breath'?"
            )
            self._capture_debug(
                f"[Engine] ❌ Unable to match chief complaint '{complaint}' to guidelines. Requesting clarification."
            )
            session.stage = "awaiting_chief_complaint"
            return self._wrap_response(session, apology, status="awaiting_chief_complaint")
        session.context['matched_categories'] = categories
        primary_category = categories[0] if categories else 'gastrointestinal'
        if len(categories) == 1:
            self._capture_debug(f"[Engine] 🎯 Category: {primary_category}")
        else:
            self._capture_debug(f"[Engine] 🎯 Categories: {', '.join(categories)}")
        self.medical_rule_engine.set_active_category(categories if len(categories) > 1 else primary_category)

        # seed condition scores
        for cond in self._get_conditions_for_categories(categories):
            session.condition_scores.setdefault(cond, 0.5)
        self._capture_debug(f"[Engine] 📋 Seeded {len(session.condition_scores)} conditions at 50.0% confidence")

        session.stage = "awaiting_chronicity"
        session.context['pre_hpi']['chief_complaint'] = complaint
        session.context['guideline_terms']['chief_complaint_terms'] = self._get_element_includes(session, 'chief_complaint') if hasattr(self, '_get_element_includes') else []
        session.context['guideline_terms']['chief_complaint_descriptors'] = self._extract_chief_complaint_descriptors(complaint)

        empathetic = self._generate_empathetic_statement(complaint)
        chronicity_prompt = self._generate_chronicity_question()

        session.pending = {
            'section': 'pre_hpi',
            'field': 'chronicity',
            'guidance': self.PRE_HPI_PROMPTS['chronicity'],
            'prompt': chronicity_prompt,
        }
        session.messages.append({"role": "assistant", "content": empathetic})
        session.messages.append({"role": "assistant", "content": chronicity_prompt})
        return self._wrap_response(session, f"{empathetic}\n\n{chronicity_prompt}", metadata={'stage': 'pre_hpi'})

    # ----------- Question selection ------------------------------------------

    def _determine_next_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        if session.stage == "awaiting_chronicity":
            if session.context['pre_hpi'].get('chronicity'):
                session.stage = "awaiting_age"
            else:
                return None

        if session.stage == "awaiting_age":
            if session.context['pre_hpi'].get('age'):
                session.stage = "awaiting_sex"
            else:
                prompt = "Thank you. For our records, how old are you?"
                return {'section': 'pre_hpi', 'field': 'age', 'prompt': prompt, 'guidance': self.PRE_HPI_PROMPTS['age']}
            session.stage = "awaiting_sex"
            prompt = "And for medical documentation, what is your biological sex?"
            return {'section': 'pre_hpi', 'field': 'sex', 'prompt': prompt, 'guidance': self.PRE_HPI_PROMPTS['sex']}

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

    def _next_oldcarts_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        if not session.oldcarts_remaining:
            session.stage = "pmh"
            return self._determine_next_question(session)

        element = session.oldcarts_remaining.pop(0)
        cc_subject = self._normalize_subject_for_questions(session.context['pre_hpi'].get('chief_complaint'))
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
            session.context['pre_hpi'][field] = text
            if field == 'chronicity':
                session.stage = "awaiting_age"
            elif field == 'age':
                session.stage = "awaiting_sex"
            elif field == 'sex':
                session.stage = "hpi"
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
            session.pending = None
            return None
        elif section == 'hpi':
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
        if not self.medical_rule_engine:
            return None
        requires_clarification = self._requires_clarification(element)
        matches = self.medical_rule_engine.find_matching_terms_faiss(
            prompt=answer,
            element=element,
            threshold=0.6,
            return_scores=True,
        )
        term_scores = getattr(self.medical_rule_engine, '_last_faiss_scores', {}) or {}

        if pending and pending.get('clarification'):
            session.context['clarifications'].pop(element, None)
        elif not requires_clarification:
            session.context['clarifications'].pop(element, None)

        analysis = None
        if element == 'location':
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
            self._log_generic_faiss(element, answer, matches, term_scores)

        index_data = self.medical_rule_engine.term_embeddings.get(element, {}) if hasattr(self.medical_rule_engine, 'term_embeddings') else {}
        term_to_conditions = index_data.get('term_to_conditions', {})
        synonym_to_medical = index_data.get('synonym_to_medical', {})

        condition_similarities: Dict[str, float] = {}

        for term in matches:
            score = term_scores.get(term)
            if score is None:
                score = term_scores.get(term.lower())
            mapped_term = synonym_to_medical.get(term) or synonym_to_medical.get(term.lower())
            if score is None and mapped_term:
                score = term_scores.get(mapped_term)
            if score is None:
                continue
            
            conditions = term_to_conditions.get(term)
            if not conditions and mapped_term:
                conditions = term_to_conditions.get(mapped_term)
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

    def _log_generic_faiss(
        self,
        element: str,
        answer: str,
        matches: List[str],
        term_scores: Dict[str, float],
    ) -> None:
        sorted_scores = dict(sorted(term_scores.items(), key=lambda x: x[1], reverse=True))
        self._capture_debug(f"[FAISS] 🔍 Scores for '{answer}' in {element}: {sorted_scores}")
        if matches:
            self._capture_debug(f"[FAISS] ✅ Matched terms for {element}: {matches}")
        else:
            self._capture_debug(f"[FAISS] ⚠️ No patient-friendly terms matched {element} for '{answer}'")

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

    def _structured_oldcarts(self, guideline: Dict[str, Any]) -> Dict[str, Any]:
        structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
        if not structured:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
        return structured or {}

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
        patient_components = {}
        if self.medical_rule_engine:
            patient_components = self.medical_rule_engine._extract_anatomical_components(answer_lower)

        # Step 1: anatomical filtering on guidelines
        filtered_guidelines = []
        if patient_components and self.medical_rule_engine:
            for guideline in guidelines:
                anatomical_type = self.medical_rule_engine._get_anatomical_type_from_guideline(guideline)
                if anatomical_type:
                    term_components = self.medical_rule_engine._map_anatomical_type_to_components(anatomical_type)
                    if term_components and self.medical_rule_engine._are_anatomical_opposites(patient_components, term_components):
                        continue
                filtered_guidelines.append(guideline)
        else:
            filtered_guidelines = guidelines

        # Step 2: collect terms
        all_terms_patient: Dict[str, Dict[str, str]] = {}
        medical_to_patient: Dict[str, str] = {}
        term_to_guidelines: Dict[str, List[str]] = {}
        for guideline in filtered_guidelines:
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

        # Anatomical filtering of satisfied medical terms
        if self.medical_rule_engine and patient_components:
            filtered_satisfied = []
            for med in satisfied_medical_terms:
                med_components = self.medical_rule_engine._extract_anatomical_components(med.lower())
                if not med_components:
                    filtered_satisfied.append(med)
                continue
                if not self.medical_rule_engine._are_anatomical_opposites(patient_components, med_components):
                    filtered_satisfied.append(med)
            satisfied_medical_terms = filtered_satisfied

        # Missing medical terms if none satisfied
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
            if self.medical_rule_engine and patient_components:
                filtered_missing = []
                for med in unsatisfied_medical:
                    med_components = self.medical_rule_engine._extract_anatomical_components(med.lower())
                    if med_components and self.medical_rule_engine._are_anatomical_opposites(patient_components, med_components):
                        continue
                    filtered_missing.append(med)
                unsatisfied_medical = filtered_missing
            # rank by FAISS scores if available
            scored_missing = []
            for med in unsatisfied_medical:
                pf = medical_to_patient.get(med.lower(), med)
                score = term_scores.get(pf, term_scores.get(pf.lower(), 0.0))
                scored_missing.append((med, score))
            scored_missing.sort(key=lambda x: x[1], reverse=True)
            missing_medical_terms = [med for med, _ in scored_missing[:5]]

        sorted_scores = dict(sorted(term_scores.items(), key=lambda x: x[1], reverse=True))
        term_breakdown = []
        threshold = 0.6
        boosted_matches: Optional[List[str]] = None
        boosted_term_scores: Optional[Dict[str, float]] = None
        for key, meta in sorted(all_terms_patient.items()):
            patient_term = meta['patient_friendly']
            score = sorted_scores.get(patient_term, sorted_scores.get(patient_term.lower(), 0.0))
            term_breakdown.append({
                'term': patient_term,
                'score': score,
                'in_semantic': patient_term.lower() in semantic_set,
            })

        if element == 'location':
            high_conf_keys: List[str] = []
            for key, meta in all_terms_patient.items():
                pf = meta['patient_friendly']
                score = sorted_scores.get(pf, sorted_scores.get(pf.lower(), 0.0))
                if score >= self.LOCATION_HIGH_CONFIDENCE_THRESHOLD:
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

                # Reapply anatomical filtering to the high-confidence set
                if self.medical_rule_engine and patient_components:
                    filtered_high_conf = []
                    seen_filtered = set()
                    for med in high_conf_med_terms:
                        med_components = self.medical_rule_engine._extract_anatomical_components(med.lower())
                        if med_components and self.medical_rule_engine._are_anatomical_opposites(patient_components, med_components):
                            continue
                        med_lower = med.lower()
                        if med_lower not in seen_filtered:
                            seen_filtered.add(med_lower)
                            filtered_high_conf.append(med)
                    high_conf_med_terms = filtered_high_conf

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

        satisfied_options = _unique([medical_to_patient.get(med.lower(), med) for med in satisfied_medical_terms])

        missing_options = _unique([medical_to_patient.get(med.lower(), med) for med in missing_medical_terms])

        all_terms_list = sorted([meta['patient_friendly'] for meta in all_terms_patient.values()])
        
        return {
            'answer': answer,
            'threshold': threshold,
            'total_guidelines': total_guidelines,
            'all_terms': all_terms_list,
            'active_terms': [],
            'semantic_matches': matches,
            'faiss_scores': sorted_scores,
            'term_breakdown': term_breakdown,
            'satisfied_medical_terms': satisfied_medical_terms,
            'satisfied_options': satisfied_options,
            'missing_medical_terms': missing_medical_terms,
            'missing_options': missing_options,
            'patient_components': patient_components,
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
            self._capture_debug(f"[Clarification] 🔍 {len(satisfied)} satisfied medical terms - generating clarification with satisfied context")
            options = analysis.get('satisfied_options', [])[:5]
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
        faiss_scores = analysis.get('faiss_scores', {})

        self._capture_debug(
            f"[Location Analysis] 📍 Checking satisfaction against ALL {analysis.get('total_guidelines', 0)} guidelines (active + reserve)"
        )
        self._capture_debug(
            f"[Location Analysis] 📍 All includes terms from {len(analysis.get('all_terms', []))} total guidelines: {analysis.get('all_terms', [])}"
        )
        self._capture_debug(f"[Location Analysis] 📝 Patient answer: '{answer}'")
        self._capture_debug(f"[FAISS] 🔍 Scores for '{answer}' in location: {faiss_scores}")
        self._capture_debug(f"[Location Analysis] 🔍 FAISS found {len(matches)} matches above threshold: {matches}")
        self._capture_debug(f"[Location Analysis]   - semantic_matches_set ({len(matches)} terms): {matches}")
        self._capture_debug(f"[Location Analysis]   - faiss_scores ({len(faiss_scores)} terms): {faiss_scores}")
        self._capture_debug(f"[Location Analysis]   - Raw FAISS scores ({len(faiss_scores)} terms): {faiss_scores}")

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
                f"[Location Analysis]   Step 3 - FAISS check: in semantic_matches_set={term_lower in semantic_set}, score={score}"
            )
            self._capture_debug(
                f"[Location Analysis]   Step 3 - Raw FAISS score for '{term}': {score:.3f} (threshold={threshold})"
            )
            if score < threshold:
                self._capture_debug(
                    f"[Location Analysis]   ⚠️ Raw score {score:.3f} < {threshold} threshold, should NOT be in semantic_matches_set"
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

    def _fuzzy_correct(self, text: str) -> str:
        if not self.medical_rule_engine or not hasattr(self.medical_rule_engine, 'fuzzy_correct_medical_terms'):
            return text
        return self.medical_rule_engine.fuzzy_correct_medical_terms(text, similarity_threshold=0.6)

    def _match_chief_complaint_to_category(self, chief_complaint: str) -> List[str]:
        if not self.chief_complaint_triggers_index or not self.chief_complaint_triggers_data:
            self._capture_debug("[Engine] ⚠️ Chief complaint trigger index unavailable - defaulting to gastrointestinal")
            return ['gastrointestinal']
        if not self.embedding_model:
            self._capture_debug("[Engine] ⚠️ No embedding model available - defaulting to gastrointestinal")
            return ['gastrointestinal']

        try:
            query_embedding = self.embedding_model.encode([chief_complaint.lower().strip()])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ Failed to encode chief complaint: {e}")
            return ['gastrointestinal']

        k = min(10, len(self.chief_complaint_triggers_data))
        similarities, indices = self.chief_complaint_triggers_index.search(query_embedding, k)

        self._capture_debug(f"[Engine] 🔍 FAISS search for '{chief_complaint}' (threshold: {self.CHIEF_COMPLAINT_FAISS_THRESHOLD})")

        category_scores = {}
        near_miss_candidates = []

        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= len(self.chief_complaint_triggers_data):
                continue
            trigger_data = self.chief_complaint_triggers_data[idx]
            trigger_text = trigger_data.get('trigger', '')
            category = trigger_data.get('category', 'gastrointestinal')
            status = "✅ ABOVE" if sim >= self.CHIEF_COMPLAINT_FAISS_THRESHOLD else "⚠️ BELOW"
            self._capture_debug(f"[Engine]   - '{trigger_text}' ({category}): {sim:.4f} {status} threshold")

            if sim >= self.CHIEF_COMPLAINT_FAISS_THRESHOLD:
                category_scores[category] = max(category_scores.get(category, 0.0), sim)
            elif sim >= self.CHIEF_COMPLAINT_NEAR_MISS_UPPER:
                category_scores[category] = max(category_scores.get(category, 0.0), sim)
            elif sim >= self.CHIEF_COMPLAINT_NEAR_MISS_LOWER:
                near_miss_candidates.append((trigger_data, sim))

        if not category_scores and near_miss_candidates:
            self._capture_debug(f"[Engine] ⚠️ No matches above threshold. Trying fuzzy matching on {len(near_miss_candidates)} candidates.")
            best_category = None
            best_score = 0.0
            cleaned = chief_complaint.lower().strip()
            for trigger_data, faiss_score in near_miss_candidates:
                trigger_text = trigger_data.get('trigger', '').lower()
                similarity = SequenceMatcher(None, cleaned, trigger_text).ratio()
                combined = (faiss_score * 0.6) + (similarity * 0.4)
                if combined >= self.CHIEF_COMPLAINT_FUZZY_THRESHOLD and combined > best_score:
                    best_score = combined
                    best_category = trigger_data.get('category', 'gastrointestinal')
            if best_category:
                self._capture_debug(f"[Engine] ✅ Fuzzy matched to category '{best_category}' (score: {best_score:.3f})")
                return [best_category]

        if not category_scores:
            self._capture_debug(f"[Engine] ⚠️ No category match found for chief complaint '{chief_complaint}'.")
            return []

        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        best_category, best_score = sorted_categories[0]
        matched = [best_category]

        for category, score in sorted_categories[1:]:
            if best_score - score < 0.1 and score >= self.CHIEF_COMPLAINT_FAISS_THRESHOLD:
                matched.append(category)

        if len(matched) == 1:
            self._capture_debug(f"[Engine] 🎯 Category matched via chief complaint: {best_category} (score: {best_score:.3f})")
        else:
            scores = ', '.join(f"{cat} ({category_scores[cat]:.3f})" for cat in matched)
            self._capture_debug(f"[Engine] 🎯 Multiple categories matched via chief complaint: {scores}")

        return matched
 
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

    def _build_chief_complaint_index(self) -> None:
        if not self.embedding_model:
            self._capture_debug("[Navigator] ⚠️ No embedding model available for chief complaint index")
            return
        triggers = []
        self.chief_complaint_triggers_data = []

        for name, guideline in self.all_guidelines.items():
            trigger_list = guideline.get('chief_complaint_triggers', [])
            category = self._get_guideline_category(guideline)
            for trigger in trigger_list:
                if not trigger:
                    continue
                triggers.append(trigger)
                self.chief_complaint_triggers_data.append({
                    'trigger': trigger,
                    'category': category,
                    'condition': name,
                })

        if not triggers:
            self._capture_debug("[Navigator] ⚠️ No chief complaint triggers found in guidelines")
            return

        try:
            embeddings = self.embedding_model.encode(triggers)
            embeddings = np.asarray(embeddings, dtype='float32')
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.chief_complaint_triggers_index = index
            self._capture_debug(f"[Navigator] ✅ Chief complaint trigger index built with {len(triggers)} triggers")
        except Exception as e:
            self._capture_debug(f"[Navigator] ❌ Failed to build chief complaint trigger index: {e}")
            self.chief_complaint_triggers_index = None

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

    def _build_oldcarts_guidance(self, session: "MedicalSession", element: str, cc: str) -> Tuple[str, str, List[str]]:
        includes = self._get_element_includes(session, element)
        base_template = self.HPI_BASE_GUIDANCE[element]
        base_question = base_template.replace('{cc}', cc)
        character_tags: set = set()
 
        stored_character_tags = session.context['guideline_terms'].get('character_tags')
        if stored_character_tags is None and element == 'character':
            stored_character_tags = set()
            for entry in includes:
                tags = entry.get('question_tags') or []
                for tag in tags:
                    if isinstance(tag, str):
                        stored_character_tags.add(tag.lower())
            session.context['guideline_terms']['character_tags'] = stored_character_tags
        elif stored_character_tags is None:
            character_includes = self._get_element_includes(session, 'character')
            stored_character_tags = set()
            for entry in character_includes:
                tags = entry.get('question_tags') or []
                for tag in tags:
                    if isinstance(tag, str):
                        stored_character_tags.add(tag.lower())
            session.context['guideline_terms']['character_tags'] = stored_character_tags

        character_tags = stored_character_tags or set()
 
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
 
        character_tags = stored_character_tags or set()
        chief_sensory = 'sensory' in character_tags
        chief_visual = 'visual' in character_tags
        sample_entries = self._select_guidance_entries(includes, limit=2, chief_sensory=chief_sensory, chief_visual=chief_visual)
        sample_terms = [entry['patient_friendly'] for entry in sample_entries if entry.get('patient_friendly')]
 
        if sample_terms:
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
 
    def _select_guidance_entries(self, entries: List[Dict[str, Any]], limit: int = 2, chief_sensory: bool = False, chief_visual: bool = False) -> List[Dict[str, Any]]:
        """Select patient-friendly terms for question guidance, preferring emergent conditions."""
        if limit <= 0 or not entries:
            return []
 
        unique_entries: List[Dict[str, Any]] = []
        seen_pf = set()
        for entry in entries:
            pf = entry.get('patient_friendly')
            if not isinstance(pf, str):
                continue
            cleaned = pf.strip()
            if not cleaned:
                continue
            tags = entry.get('question_tags') or []
            if chief_sensory and 'visual' in tags:
                continue
            if chief_visual and 'sensory' in tags:
                continue
            key = cleaned.lower()
            if key in seen_pf:
                continue
            seen_pf.add(key)
            copy_entry = dict(entry)
            copy_entry['patient_friendly'] = cleaned
            unique_entries.append(copy_entry)
 
        if not unique_entries:
            return []
 
        emergent_entries = [entry for entry in unique_entries if entry.get('emergent_term') or entry.get('condition_urgency') == 'emergent']
        urgent_entries = [entry for entry in unique_entries if entry.get('condition_urgency') == 'urgent' and entry not in emergent_entries]
        other_entries = [entry for entry in unique_entries if entry not in emergent_entries and entry not in urgent_entries]
 
        selected: List[Dict[str, Any]] = []
 
        def choose_from(pool: List[Dict[str, Any]]):
            nonlocal selected
            if len(selected) >= limit or not pool:
                return
            available = [entry for entry in pool if entry not in selected]
            if not available:
                return
            k = min(limit - len(selected), len(available))
            if k <= 0:
                return
            selected.extend(random.sample(available, k=k))
 
        choose_from(emergent_entries)
        choose_from(urgent_entries)
        choose_from(other_entries)
 
        return selected

    def _get_element_includes(self, session: "MedicalSession", element: str) -> List[Dict[str, Any]]:
        cache = session.context.setdefault('guideline_includes', {})
        if element in cache:
            return cache[element]
        categories = session.context.get('matched_categories') or ['gastrointestinal']
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
        return ordered.copy()

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
        if not self.llm_chat_fn:
            question = base_question or guidance
            if section == 'hpi' and options:
                option_text = ', '.join(options)
                if question.endswith('?'):
                    prefix = question
                else:
                    prefix = question.rstrip('.') + '?'
                question = f"{prefix} You can mention things like: {option_text}."
            return question
        cc = session.context['pre_hpi'].get('chief_complaint', 'your symptoms') or 'your symptoms'
        recent = '\n'.join(f"{m['role']}: {m['content']}" for m in session.messages[-6:])
        if section == 'pre_hpi' and field == 'chief_complaint':
            guidance = (
                "Greet the patient warmly (e.g., 'Hi there, it's nice to meet you.') and ask what brings them in today"
                " and for how long. Return one friendly sentence combining greeting and question."
            )
        user_prompt = (
            f"Section: {section}\n"
            f"Field: {field}\n"
            f"Chief complaint: {cc}\n"
            f"Guidance: {guidance}\n"
            f"Recent conversation:\n{recent}\n"
            "Produce one friendly question."
        )
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=60,
            temperature=0.5,
        )
        cleaned = self._clean_llm_response(response)
        if not cleaned:
            raise ValueError(f"LLM returned empty question for {field}")
        if base_question and base_question.lower() not in cleaned.lower():
            cleaned = base_question
        if section == 'hpi' and options:
            if "you can mention things like" not in cleaned.lower():
                option_text = ', '.join(options)
                if cleaned.endswith('?'):
                    prefix = cleaned
                else:
                    prefix = cleaned.rstrip('.') + '?'
                cleaned = f"{prefix} You can mention things like: {option_text}."
        return cleaned

    def _generate_empathetic_statement(self, chief_complaint: str) -> str:
        if not self.llm_chat_fn:
            return "I'm here to help."
        response = self.llm_chat_fn(
            [
                {"role": "system", "content": self.EMPATHETIC_SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient concern: {chief_complaint}"},
            ],
            max_tokens=80,
            temperature=0.4,
        )
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
        }
        return element not in no_clarification_elements

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
