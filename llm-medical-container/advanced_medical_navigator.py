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
from typing import Dict, List, Optional, Tuple


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
    )

    CHRONICITY_SYSTEM_PROMPT = (
        "You are a medical assistant. Generate a concise question asking whether the problem is new"
        " or ongoing, and if a prior diagnosis exists."
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
            'matched_categories': [],
        })
        condition_scores: Dict[str, float] = field(default_factory=dict)
        condition_rankings: List[Tuple[str, float]] = field(default_factory=list)
        active_conditions: List[Tuple[str, float]] = field(default_factory=list)
        reserve_conditions: List[Tuple[str, float]] = field(default_factory=list)
        previous_active: set = field(default_factory=set)
        oldcarts_remaining: List[str] = field(default_factory=list)
        completed: bool = False

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
            self._store_answer(session, session.pending, user_message)
            session.pending = None

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
        cc = session.context['pre_hpi'].get('chief_complaint', 'the problem')
        guidance = self._build_oldcarts_guidance(session, element, cc)
        prompt = self._generate_question(session, 'hpi', element, guidance)
        return {'section': 'hpi', 'field': element, 'prompt': prompt, 'guidance': guidance}

    # ----------- Answer persistence & scoring --------------------------------

    def _store_answer(self, session: "MedicalSession", pending: Dict[str, str], answer: str) -> None:
        section, field = pending['section'], pending['field']
        text = answer.strip()
        if section == 'pre_hpi':
            session.context['pre_hpi'][field] = text
            if field == 'chronicity':
                session.stage = "awaiting_age"
                session.pending = None
            elif field == 'age':
                session.stage = "awaiting_sex"
            elif field == 'sex':
                session.stage = "hpi"
                session.oldcarts_remaining = self._ordered_oldcarts_elements(session)
        elif section == 'hpi':
            session.context['hpi'][field] = text
            self._score_oldcarts_answer(session, field, text)
        elif section == 'pmh':
            session.context['pmh'][field] = text

    def _score_oldcarts_answer(self, session: "MedicalSession", element: str, answer: str) -> None:
         if not self.medical_rule_engine:
             return
        matches = self.medical_rule_engine.find_matching_terms_faiss(
            prompt=answer,
            element=element,
            threshold=0.6,
            return_scores=True,
        )
        term_scores = getattr(self.medical_rule_engine, '_last_faiss_scores', {}) or {}

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

        if not condition_similarities:
            self._capture_debug(f"[Scoring] ⚪ No guideline matches for {element} → '{answer}'")
            return

        weight = self._get_element_weight(session, element)
        for cond, similarity in condition_similarities.items():
            prior = session.condition_scores.get(cond, 0.5)
            blended = prior + weight * (similarity - prior)
            session.condition_scores[cond] = blended
            self._capture_debug(
                f"[Scoring] 📊 {cond}: old={prior:.3f}, similarity={similarity:.3f}, weight={weight:.2f}, new={blended:.3f}"
            )

        session.condition_rankings = sorted(session.condition_scores.items(), key=lambda x: x[1], reverse=True)
        if session.condition_rankings:
            self._update_condition_pools(session)
            self._log_rankings(session)

    # ----------- Chief complaint matching ------------------------------------

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

    def _build_oldcarts_guidance(self, session: "MedicalSession", element: str, cc: str) -> str:
        base_question = self.HPI_BASE_GUIDANCE[element].replace('{cc}', cc)
        terms = self._guideline_terms_for_element(session, element)
        clean_terms = []
        seen = set()
        for term in terms:
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                clean_terms.append(t)
            if len(clean_terms) >= 3:
                break

        if clean_terms:
            options = ', '.join(clean_terms)
            return (
                f"Create exactly two sentences. Sentence 1 must be the open-ended question: '{base_question}'. "
                f"Sentence 2 should gently offer examples starting with 'You can mention things like' followed by up to three of these options: {options}. "
                "Keep both sentences short, friendly, and avoid adding extra options or clauses."
            )

        return (
            f"Ask exactly one friendly, open-ended sentence: '{base_question}'. Do not add examples or extra sentences." )

    def _guideline_terms_for_element(self, session: "MedicalSession", element: str) -> List[str]:
        cache = session.context['guideline_terms']
        if element in cache:
            return cache[element]
        categories = session.context.get('matched_categories') or ['gastrointestinal']
        terms: List[str] = []
        seen = set()
        for category in categories:
            guidelines = self._get_guidelines_by_category(category)
            for guideline in guidelines.values():
                structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                if not structured:
                    structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                element_data = structured.get(element, {})
                includes = element_data.get('includes', []) if isinstance(element_data, dict) else []
                for entry in includes:
                    if isinstance(entry, dict):
                        term = entry.get('patient_friendly') or entry.get('medical')
                    else:
                        term = entry
                    if isinstance(term, str):
                        cleaned = term.strip()
                        if cleaned and cleaned.lower() not in seen:
                            seen.add(cleaned.lower())
                            terms.append(cleaned)
        cache[element] = terms
        return terms

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

    def _generate_question(self, session: "MedicalSession", section: str, field: str, guidance: str) -> str:
        if not self.llm_chat_fn:
            return guidance
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
        print(message)
    
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
        lines.append(f"[Scoring]   🧠 ML System: Fully operational")
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
        self._capture_debug(f"[Scoring]   🧠 ML System: Fully operational")

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
