#!/usr/bin/env python3
"""
Advanced Medical Navigator (LLM-first implementation)
====================================================

Collects a focused history using the "Concise Universal History Taking Template".
Sections:
    I.  Patient Identification & Chief Complaint
    II. History of Present Illness (OLDCARTS + red flags)
    III. Past Medical History / Surgeries / Medications & Allergies

All questions are conversational templates (atlas-style). Responses are stored and
summarised via the injected LLM when the history is complete.
"""

from datetime import datetime
from typing import Dict, Optional, List


class AdvancedMedicalNavigator:
    """LLM-driven navigator that walks through a concise universal history."""

    # ---------------------------------------------------------------------
    # Section 0: Configuration - Template prompts for each element
    # ---------------------------------------------------------------------

    IDENTIFICATION_PROMPT = (
        "To get started, could you please tell me your name and date of birth?"
    )

    CHIEF_COMPLAINT_PROMPT = (
        "What brings you in today, and how long have you been dealing with it?"
    )

    HPI_ORDER = [
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

    HPI_PROMPTS = {
        "onset": "When did this start, and did it come on suddenly or gradually?",
        "location": "Where exactly is your {cc}? Does it spread or move anywhere else?",
        "duration": "Is it constant or does it come and go? How long does an episode usually last?",
        "character": "How would you describe what it feels like?",
        "aggravating": "What tends to make it worse?",
        "relieving": "What makes it better? Have you tried anything that helps?",
        "timing": "Does it happen at a particular time or only during certain activities?",
        "severity": "On a scale of 1 to 10, how bad is it? How much does it affect your day?",
        "associated": "Have you noticed any other symptoms along with it?",
        "red_flags": "Have you had any dizziness, fever, or trouble breathing?",
    }

    PMH_ORDER = ["pmh", "psh", "meds_allergies"]

    PMH_PROMPTS = {
        "pmh": "Do you have any existing medical conditions I should know about?",
        "psh": "Have you had any surgeries in the past?",
        "meds_allergies": "Are you taking any medications, and do you have any medication allergies?",
    }

    SUMMARY_SYSTEM_PROMPT = (
        "You are a clinical assistant. Given structured history data, create a brief,"
        " professional summary covering identification, chief complaint, focused HPI,"
        " and PMH/medications/allergies. Keep it concise (<= 6 bullet points)."
    )

    QUESTION_SYSTEM_PROMPT = (
        "You are a compassionate medical assistant conducting a patient interview."
        " Use the provided guidance to craft one natural question in plain text."
        " Do not include explanations or prefaces—return only the question itself."
        " Keep it under 20 words and reference the supplied chief complaint when helpful."
    )

    PRE_HPI_ORDER = [
        "chief_complaint",
        "chronicity",
        "age",
        "sex",
    ]

    PRE_HPI_GUIDANCE = {
        "chief_complaint": "Please ask the patient what brings them in today and how long it has been going on.",
        "chronicity": "Determine if the problem is new or ongoing and whether there is a prior diagnosis.",
        "age": "Ask for the patient's age, stated as a single number.",
        "sex": "Ask for the patient's biological sex for medical documentation.",
    }

    # ---------------------------------------------------------------------
    # Section 1: Session container
    # ---------------------------------------------------------------------

    class MedicalSession:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.created_at = datetime.now()
            self.messages: List[Dict[str, str]] = []
            self.section: str = "pre_hpi"
            self.pending: Optional[Dict[str, str]] = None
            self.context: Dict[str, Dict] = {
                "pre_hpi": {},
                "hpi": {},
                "pmh": {},
            }
            self.completed: bool = False

    # ---------------------------------------------------------------------
    # Section 2: Initialisation
    # ---------------------------------------------------------------------

    def __init__(self, llm_chat_fn, medical_rule_engine=None, embedding_model=None):
        self.llm_chat_fn = llm_chat_fn
        self.sessions: Dict[str, AdvancedMedicalNavigator.MedicalSession] = {}

    # ---------------------------------------------------------------------
    # Section 3: Public interface
    # ---------------------------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> Dict[str, any]:
        session = self._get_or_create_session(session_id)
        session.messages.append({"role": "user", "content": user_message})

        if session.pending:
            if not self._is_valid_answer(session, session.pending, user_message):
                clarification = self._clarify_prompt(session.pending)
                session.messages.append({"role": "assistant", "content": clarification})
                return {
                    "response": clarification,
                    "status": "question",
                    "metadata": {
                        "section": session.section,
                        "field": session.pending["field"],
                        "clarification": True,
                    },
                }
            self._store_answer(session, session.pending, user_message)
            session.pending = None

        if session.completed:
            follow_up = "Thank you. If anything changes, feel free to let me know."
            session.messages.append({"role": "assistant", "content": follow_up})
            return {
                "response": follow_up,
                "status": "complete",
                "metadata": {"section": "complete"},
            }

        next_item = self._determine_next_question(session)
        if next_item:
            prompt = next_item["prompt"]
            session.pending = next_item
            session.messages.append({"role": "assistant", "content": prompt})
            return {
                "response": prompt,
                "status": "question",
                "metadata": {
                    "section": session.section,
                    "field": next_item["field"],
                },
            }

        summary = self._generate_summary(session)
        session.completed = True
        session.messages.append({"role": "assistant", "content": summary})
        return {
            "response": summary,
            "status": "complete",
            "metadata": {
                "section": "complete",
                "summary": True,
            },
        }

    # ---------------------------------------------------------------------
    # Section 4: Question selection helpers
    # ---------------------------------------------------------------------

    def _determine_next_question(self, session: "MedicalSession") -> Optional[Dict[str, str]]:
        if session.section == "pre_hpi":
            pre_hpi_context = session.context.setdefault("pre_hpi", {})
            for field in self.PRE_HPI_ORDER:
                if field not in pre_hpi_context:
                    guidance = self.PRE_HPI_GUIDANCE[field]
                    prompt = self._generate_question(session, "pre_hpi", field, guidance)
                    return {
                        "section": "pre_hpi",
                        "field": field,
                        "prompt": prompt,
                    }
            session.section = "hpi"

        if session.section == "hpi":
            pre_hpi_context = session.context.setdefault("pre_hpi", {})
            cc = pre_hpi_context.get("chief_complaint", "the issue")
            for element in self.HPI_ORDER:
                if element not in session.context["hpi"]:
                    template = self.HPI_PROMPTS[element]
                    prompt_text = template.replace("{cc}", cc)
                    prompt = self._generate_question(session, "hpi", element, prompt_text)
                    return {
                        "section": "hpi",
                        "field": element,
                        "prompt": prompt,
                    }
            session.section = "pmh"

        if session.section == "pmh":
            for field in self.PMH_ORDER:
                if field not in session.context["pmh"]:
                    template = self.PMH_PROMPTS[field]
                    prompt = self._generate_question(session, "pmh", field, template)
                    return {
                        "section": "pmh",
                        "field": field,
                        "prompt": prompt,
                    }
            session.section = "complete"

        return None

    # ---------------------------------------------------------------------
    # Section 5: Answer storage
    # ---------------------------------------------------------------------

    def _store_answer(self, session: "MedicalSession", pending: Dict[str, str], answer: str) -> None:
        section = pending["section"]
        field = pending["field"]

        if section == "pre_hpi":
            session.context.setdefault("pre_hpi", {})[field] = answer.strip()
        elif section == "hpi":
            session.context["hpi"][field] = answer.strip()
        elif section == "pmh":
            session.context["pmh"][field] = answer.strip()

    # ---------------------------------------------------------------------
    # Section 6: Session management
    # ---------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> "MedicalSession":
        if session_id not in self.sessions:
            self.sessions[session_id] = AdvancedMedicalNavigator.MedicalSession(session_id)
        return self.sessions[session_id]

    # ---------------------------------------------------------------------
    # Section 7: Summarisation
    # ---------------------------------------------------------------------

    def _generate_summary(self, session: "MedicalSession") -> str:
        if not self.llm_chat_fn:
            return "History collection complete."

        pre_hpi_context = session.context.get("pre_hpi", {})
        age = pre_hpi_context.get("age", "Not provided")
        sex = pre_hpi_context.get("sex", "Not provided")
        cc = pre_hpi_context.get("chief_complaint", "Not stated")
        chronicity = pre_hpi_context.get("chronicity", "Not provided")
        hpi_parts = session.context.get("hpi", {})
        pmh_parts = session.context.get("pmh", {})

        user_prompt = (
            f"Chief complaint: {cc}\n"
            f"Chronicity: {chronicity}\n"
            f"Age: {age}\n"
            f"Biological sex: {sex}\n\n"
            f"HPI (OLDCARTS): {hpi_parts}\n"
            f"PMH/PSH/Meds/Allergies: {pmh_parts}\n"
            "Provide a focused clinical summary."
        )

        llm_response = self.llm_chat_fn(
            [{"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}],
            max_tokens=300,
            temperature=0.2,
        )

        return llm_response.strip() if llm_response else "History collection complete."

    def _clarify_prompt(self, pending: Dict[str, str]) -> str:
        prompt = pending.get("prompt", "Could you tell me more?")
        return f"Sorry for the confusion. I was asking: {prompt}"

    def _is_valid_answer(self, session: "MedicalSession", pending: Dict[str, str], answer: str) -> bool:
        if not answer or not answer.strip():
            return False
        if not self.llm_chat_fn:
            return True
        prompt = (
            "Question asked: " + pending.get("prompt", "") + "\n"
            "Patient replied: " + answer + "\n\n"
            "Should the reply be accepted? Answer ONLY with YES or NO."
        )
        llm_result = self.llm_chat_fn(
            [
                {"role": "system", "content": "You are a medical assistant validating patient responses."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        if not llm_result:
            return True
        return llm_result.strip().upper().startswith("Y")

    def _generate_question(
        self,
        session: "MedicalSession",
        section: str,
        field: str,
        guidance: Optional[str] = None,
    ) -> str:
        if not self.llm_chat_fn:
            return guidance or "Could you tell me more about that?"

        pre_hpi_context = session.context.setdefault("pre_hpi", {})
        cc = pre_hpi_context.get("chief_complaint", "your symptoms") or "your symptoms"
        previous = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in session.messages[-6:]
        )
        guidance_text = guidance or self.CHIEF_COMPLAINT_PROMPT
        if section == "pre_hpi" and field == "chief_complaint" and "chief_complaint" not in pre_hpi_context:
            guidance_text = (
                "Greet the patient warmly (e.g., 'Hi there, it's nice to meet you.')"
                " and then ask what brings them in today and for how long."
                " Return a single sentence combining the greeting and the question."
            )
        user_prompt = (
            f"Section: {section}\n"
            f"Field: {field}\n"
            f"Chief complaint: {cc}\n"
            f"Guidance: {guidance_text}\n"
            f"Recent conversation:\n{previous}\n"
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
        return response.strip() if response else guidance_text
