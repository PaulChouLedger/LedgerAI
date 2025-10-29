#!/usr/bin/env python3
"""
Adaptive Diagnostic Engine - Minimal Universal Version

UNIVERSAL FLOW:
1. Chief complaint → Unified function with chief complaint synonyms → Match category → Narrow guidelines
2. Parse prompt → Detect answered OLDCARTS elements
3. For each element:
   - Location: Extract location → Filter using medical_rules.json → Parse user response array → 
     Compare to structured_oldcarts → Generate clarifying question if missing → Use unified function to score
   - Other elements: Use unified function directly
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from thinking_fillers import get_filler

# Import modular RAG client
from rag import get_rag_client
import numpy as np

# Import fuzzy medical matcher for typo correction
from fuzzy_medical_matcher import FuzzyMedicalMatcher

class RAGEmbeddingAPI:
    """Wrapper for RAG client's embedding service"""
    def __init__(self, rag_url: str = "http://localhost:11435"):
        self.rag_client = get_rag_client()
    
    def encode(self, texts: List[str]) -> List:
        embeddings = self.rag_client.embed(texts)
        if embeddings:
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        else:
            raise RuntimeError(f"RAG embedding failed")

# RAG client availability check
try:
    rag_api = RAGEmbeddingAPI()
    test_embedding = rag_api.encode(["test"])
    RAG_API_AVAILABLE = True
    rag_client = get_rag_client()
    print(f"[Engine] ✅ RAG client available - using {rag_client.get_mode()}")
except Exception as e:
    RAG_API_AVAILABLE = False
    print(f"[Engine] ⚠️ RAG client not available: {e}")


class AdaptiveDiagnosticEngine:
    """Minimal universal diagnostic engine"""
    
    def __init__(self, guidelines_dir: str = None, llm_chat_fn=None, embedding_model=None, llm_chat_simple_fn=None):
        # Auto-detect guidelines directory
        if guidelines_dir is None:
            if os.path.exists("/app/medical/guidelines"):
                guidelines_dir = "/app/medical/guidelines"
            elif os.path.exists("medical/guidelines"):
                guidelines_dir = "medical/guidelines"
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "medical", "guidelines")):
                guidelines_dir = os.path.join(os.path.dirname(__file__), "medical", "guidelines")
            else:
                raise RuntimeError("Could not find medical guidelines directory")
        
        self.guidelines_dir = Path(guidelines_dir)
        self.llm_chat_fn = llm_chat_fn
        self.llm_chat_simple_fn = llm_chat_simple_fn or llm_chat_fn
        self.embedding_model = embedding_model
        
        # Configuration
        self.temperature_simple = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.7'))
        self.temperature_complex = float(os.environ.get('LLM_TEMPERATURE_COMPLEX', '0.7'))
        
        # Initialize debug capture
        self._captured_debug_output = []
        self.current_category = None
        
        # Initialize Medical Rule Engine
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
            self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized")
        except ImportError:
            try:
                import sys
                sys.path.append('/app/ml')
                from medical_rule_engine import MedicalRuleEngine
                self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
                self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized (alternative path)")
        except ImportError:
                self.medical_rule_engine = None
                self._capture_debug(f"[Engine] ⚠️ Medical Rule Engine not available")
        
        # Initialize fuzzy matcher
        self.fuzzy_matcher = FuzzyMedicalMatcher()
        
        # RAG API
        self.rag_api = RAGEmbeddingAPI() if RAG_API_AVAILABLE else None
        self.use_rag_api = RAG_API_AVAILABLE
        
        # Load guidelines
        self.all_guidelines = {}
        self._load_guidelines()
        
        # Initialize assessment state
        self.reset_assessment()
    
    def _load_guidelines(self):
        """Load all JSON guideline files"""
        if not self.guidelines_dir.exists():
            return
        
        for json_file in sorted(self.guidelines_dir.glob("**/*.json")):
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                    name = guideline.get('condition', json_file.stem)
                    # Store organ system from directory structure
                    organ_system = json_file.parent.name if json_file.parent != self.guidelines_dir else "Other"
                    guideline['organ_system'] = organ_system  # Store for filtering
                    self.all_guidelines[name] = guideline
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to load {json_file.name}: {e}")
        
    def _capture_debug(self, message: str):
        """Capture debug output"""
        self._captured_debug_output.append(message)
        print(message)
    
    def reset_assessment(self):
        """Reset for new patient"""
        self.active_guidelines = []
        self.reserve_pool = []
        self.ruled_out = []
        self.chief_complaint = None
        self.status = "idle"
        self.conversation_history = []
        self.demographics = {}
        self.oldcarts_covered = {'O': False, 'L': False, 'D': False, 'C': False, 'A': False, 'R': False, 'T': False, 'S': False}
        self.oldcarts_analysis = None
        self.clarification_count = {}
        self.diagnosed_condition = None
        self.red_flags_present = []
        self.red_flag_index = 0
        self.MAX_ACTIVE = 5
        self.RULE_OUT_THRESHOLD = 0.05
    
    def _get_debug_info(self, last_answer: str = None) -> Dict:
        """Build debug information"""
        num_questions = len([item for item in self.conversation_history if item['type'] == 'question'])
        covered_count = sum(self.oldcarts_covered.values())
        coverage_str = ''.join([k if v else '_' for k, v in self.oldcarts_covered.items()])
        
        return {
            'demographics': self.demographics,
            'question_number': num_questions,
            'oldcarts_coverage': coverage_str,
            'active_differentials': [
                {'rank': i+1, 'name': g['name'], 'score': g['score']}
                for i, g in enumerate(self.active_guidelines[:5])
            ]
        }
    
    def start_assessment(self, chief_complaint: str) -> Dict[str, Any]:
        """
        UNIVERSAL FLOW:
        1. Unified function with chief complaint synonyms → match category → narrow guidelines
        2. Parse prompt to detect answered OLDCARTS elements
        3. Process each element individually using same logic as _process_clinical_answer
        """
        self._capture_debug(f"\n{'='*80}")
        self._capture_debug(f"[Engine] 🚀 NEW ASSESSMENT (UNIVERSAL FLOW)")
        self._capture_debug(f"{'='*80}")
        self._capture_debug(f"[Engine] Chief Complaint: '{chief_complaint}'")
        
        # STEP 1: Unified function with chief complaint synonyms → match category
        category = self._match_chief_complaint_to_category(chief_complaint)
        self.current_category = category
        self._capture_debug(f"[Engine] 🎯 Category: {category}")
        
        # STEP 2: Narrow down guidelines
        matched_guidelines = self._get_all_guidelines_in_category(category)
        self._capture_debug(f"[Engine] 📊 Found {len(matched_guidelines)} guidelines")
        
        # STEP 3: Parse prompt to detect answered OLDCARTS elements
        oldcarts_analysis = self._parse_prompt_against_structured_oldcarts(chief_complaint, matched_guidelines)
        self._capture_debug(f"[Engine] 🔍 OLDCARTS Analysis: {oldcarts_analysis}")
        
        # Initialize assessment
        self.reset_assessment()
        self.chief_complaint = chief_complaint
        self.status = "questioning"
        self.active_guidelines = matched_guidelines[:self.MAX_ACTIVE]
        self.reserve_pool = matched_guidelines[self.MAX_ACTIVE:]
        
        # Store OLDCARTS analysis for use in questioning
        self.oldcarts_analysis = oldcarts_analysis
        
        # Don't process detected elements here - just ask first question
        # Processing will happen when user answers each question
        return self._generate_ml_first_question_with_demographics()
    
    def _match_chief_complaint_to_category(self, chief_complaint: str) -> str:
        """Use unified function with chief complaint synonyms → match category"""
        category_to_system = {
            'gastrointestinal': 'GI', 'cardiovascular': 'CARDIO',
            'respiratory': 'PULMONARY', 'neurological': 'NEURO',
            'musculoskeletal': 'MSK', 'renal': 'RENAL',
            'genitourinary': 'GU', 'gynecological': 'GYN',
            'dermatological': 'DERM'
        }
        
        best_match = None
        best_score = 0.0
        
        for category, organ_system in category_to_system.items():
            synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
            synonym_path = os.path.join(os.path.dirname(__file__), synonym_file)
            
            if os.path.exists(synonym_path):
                try:
                    with open(synonym_path, 'r') as f:
                        synonyms = json.load(f)
                    
                    if 'chief_complaint' in synonyms:
                        for standard_term, synonym_list in synonyms['chief_complaint'].items():
                            for synonym in synonym_list:
                                if synonym.lower() in chief_complaint.lower():
                                    score = len(synonym) / len(chief_complaint)
                                    if score > best_score:
                                        best_score = score
                                        best_match = category
                except Exception:
                    pass
        
        if best_match:
            return best_match
        
        # Fallback to substring matching
        return self._categorize_complaint_by_substring(chief_complaint)
    
    def _categorize_complaint_by_substring(self, complaint: str) -> str:
        """Fallback category detection"""
        complaint_lower = complaint.lower()
        organ_keywords = {
            'GI': ['abdominal', 'stomach', 'belly', 'gut', 'bowel'],
            'CARDIO': ['chest', 'heart', 'cardiac'],
            'NEURO': ['head', 'headache', 'brain'],
            'MSK': ['back', 'joint', 'muscle', 'bone'],
            'RENAL': ['kidney', 'urinary', 'bladder'],
            'DERM': ['skin', 'rash'],
            'GYN': ['pelvic', 'menstrual'],
            'GU': ['prostate', 'testicular'],
            'PULMONARY': ['lung', 'breathing', 'respiratory']
        }
        
        for organ, keywords in organ_keywords.items():
            if any(keyword in complaint_lower for keyword in keywords):
                return organ.lower()
        
        return 'gastrointestinal'  # Default
    
    def _get_all_guidelines_in_category(self, category: str) -> List[Dict]:
        """Get all guidelines in category"""
        relevant_guidelines = self._get_guidelines_by_category(category)
        
        matched_guidelines = []
        for name, guideline in relevant_guidelines.items():
            matched_guidelines.append({
                'name': name,
                'score': 0.5,
                'data': guideline
            })
        
        # Sort by prevalence
        matched_guidelines.sort(key=lambda x: x['data'].get('prevalence_score', 0), reverse=True)
        return matched_guidelines
    
    def _get_guidelines_by_category(self, category: str) -> Dict:
        """Get guidelines filtered by category using directory structure"""
        if category == 'ALL':
            return self.all_guidelines
        
        category_map = {
            'gastrointestinal': 'GI',
            'cardiovascular': 'CARDIO',
            'respiratory': 'PULMONARY',
            'neurological': 'NEURO',
            'musculoskeletal': 'MSK',
            'renal': 'RENAL',
            'genitourinary': 'GU',
            'gynecological': 'GYN',
            'dermatological': 'DERM'
        }
        
        target_organ = category_map.get(category.lower(), category.upper())
        filtered = {}
        
        for name, guideline in self.all_guidelines.items():
            organ_system = guideline.get('organ_system', '')
            # Match by directory name (e.g., 'GI', 'CARDIO', etc.)
            if organ_system == target_organ or target_organ in organ_system.upper():
                filtered[name] = guideline
        
        if not filtered:
            self._capture_debug(f"[Engine] ⚠️ No guidelines found for {category}, using all guidelines")
            return self.all_guidelines
        
        self._capture_debug(f"[Engine] ✅ Filtered {len(self.all_guidelines)} → {len(filtered)} guidelines for {category}")
        return filtered
    
    def _parse_prompt_against_structured_oldcarts(self, prompt: str, guidelines: List[Dict]) -> Dict[str, Any]:
        """Parse prompt against structured OLDCARTS to determine what's already answered"""
        if not guidelines:
                return {
                'answered_components': {},
                'missing_components': ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
            }
        
        # Collect all 'includes' terms from guidelines
        all_includes = {
            'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
            'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set()
        }
        
        for guideline in guidelines:
            structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            for element, data in structured.items():
                if isinstance(data, dict) and 'includes' in data:
                    if element in all_includes:
                        for term in data['includes']:
                            all_includes[element].add(term.lower())
        
        # Detect which elements are present in prompt (using whole word matching)
        answered_components = {}
        prompt_lower = prompt.lower()
        
        # Common words to exclude (too generic, cause false positives)
        exclude_words = {'pain', 'ache', 'hurt', 'sore'}
        
        for element, expected_terms in all_includes.items():
            for term in expected_terms:
                        term_lower = term.lower()
                # Skip single generic words that cause false positives
                if term_lower in exclude_words:
                    continue
                
                # Use whole word matching (more specific)
                # Check if term appears as whole word/phrase, not substring
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, prompt_lower):
                    if element not in answered_components:
                        answered_components[element] = []
                    answered_components[element].append(term)
                            break
                        
        all_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
        answered_elements = list(answered_components.keys())
        missing_elements = [element for element in all_elements if element not in answered_elements]
        
        return {
            'answered_components': answered_components,
            'missing_components': missing_elements
        }
    
    def _process_clinical_answer(self, answer: str) -> Dict[str, Any]:
        """Score guidelines using unified similarity function"""
        # Get last question
        last_q = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question':
                last_q = item
                break
        
        oldcarts_element = last_q.get('oldcarts') if last_q else None
        
        if not oldcarts_element:
            return self._ask_next_clinical_question()
        
        # Handle demographics
        if last_q.get('focus') == 'demographics':
            if 'sex' not in self.demographics:
                question = "What is your biological sex?"
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'oldcarts': 'demographics',
                    'focus': 'demographics'
                })
                return {
                    'success': True,
                    'question': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'Male', 'callback_data': 'sex_male'},
                        {'text': 'Female', 'callback_data': 'sex_female'}
                    ]
                }
                return self._ask_next_clinical_question()
        
        # Handle onset (documentation only)
        if oldcarts_element == 'onset':
            self.oldcarts_covered['O'] = True
            return self._ask_next_clinical_question()
        
        # Score guidelines
        if not self.embedding_model:
            return {'success': False, 'message': 'Embedding model not available'}
        
        all_guidelines = self.active_guidelines + self.reserve_pool
        
        # STEP 1: Filter guidelines using medical_rules.json (location only)
        category = self.current_category or 'gastrointestinal'
                category_to_system = {
                    'gastrointestinal': 'GI', 'cardiovascular': 'CARDIO',
                    'respiratory': 'PULMONARY', 'neurological': 'NEURO',
                    'musculoskeletal': 'MSK', 'renal': 'RENAL',
                    'genitourinary': 'GU', 'gynecological': 'GYN',
                    'dermatological': 'DERM'
                }
        organ_system = category_to_system.get(category, 'GI')
        
        if oldcarts_element == 'location' and self.medical_rule_engine:
            self._capture_debug(f"[Engine] 🏥 Filtering guidelines using medical_rules.json")
            all_guidelines = self.medical_rule_engine.filter_guidelines_by_location(answer, all_guidelines, organ_system)
        
        # STEP 2: Score all guidelines using unified function
        for g in all_guidelines:
            classic = g['data'].get('key_features', {}).get('classic_presentation', '')
            oldcarts_section = self._extract_oldcarts_section(classic, oldcarts_element)
            
            if not oldcarts_section:
                continue
            
            structured_oldcarts = g['data'].get('key_features', {}).get('structured_oldcarts', {})
            condition_name = g['data'].get('condition', g['name'])
            
            # Use unified function for scoring
            if self.medical_rule_engine:
                element_data = structured_oldcarts.get(oldcarts_element)
                similarity_result = self.medical_rule_engine.compute_unified_similarity(
                    answer, oldcarts_section, condition_name, organ_system,
                    oldcarts_element, {oldcarts_element: element_data} if element_data else None
                )
                similarity = similarity_result['similarity']
            else:
                similarity = 0.5
            
            # Update score
            old_score = g['score']
            new_score = (old_score * 0.7) + (similarity * 0.3)
            g['score'] = new_score
        
        # Re-rank
        all_guidelines.sort(key=lambda x: x['score'], reverse=True)
        
        # Rule out low scores
        remaining = []
        for g in all_guidelines:
            threshold = self._get_dynamic_threshold(g['score'])
            if g['score'] >= threshold:
                remaining.append(g)
            else:
                self.ruled_out.append(g)
        
        remaining.sort(key=lambda x: x['score'], reverse=True)
        self.active_guidelines = remaining[:self.MAX_ACTIVE]
        self.reserve_pool = remaining[self.MAX_ACTIVE:]
        
        # Check if clarification needed
        clarification_count = sum(1 for item in self.conversation_history 
                                 if item.get('oldcarts') == oldcarts_element 
                                 and item.get('is_clarification'))
        
        if clarification_count < 2 and self.active_guidelines:
            try:
                missing_terms = self._analyze_missing_information(answer, oldcarts_element)
                if missing_terms:
                    question = self._generate_clarifying_question(oldcarts_element, answer, clarification_count, missing_terms)
                    self.conversation_history.append({
                        'type': 'question',
                        'question': question,
                        'oldcarts': oldcarts_element,
                        'is_clarification': True
                    })
            return {
                        'success': True,
                        'question': question,
                        'status': 'questioning'
                    }
            except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Clarification check failed: {e}")
        
        # Mark element as covered
        element_map = {'onset': 'O', 'location': 'L', 'duration': 'D', 'character': 'C',
                      'aggravating': 'A', 'relieving': 'R', 'timing': 'T', 'severity': 'S'}
        if oldcarts_element in element_map:
            self.oldcarts_covered[element_map[oldcarts_element]] = True
        
        # Continue to next question
        return self._ask_next_clinical_question()
    
    def _extract_oldcarts_section(self, classic_presentation: str, element: str) -> str:
        """Extract specific OLDCARTS section from classic_presentation"""
        element_names = {
            'onset': 'ONSET', 'location': 'LOCATION', 'duration': 'DURATION',
            'character': 'CHARACTER', 'aggravating': 'AGGRAVATING',
            'relieving': 'RELIEVING', 'timing': 'TIMING', 'severity': 'SEVERITY'
        }
        
        element_tag = element_names.get(element.lower(), element.upper())
        
        # Extract section
        if element_tag in classic_presentation:
            parts = classic_presentation.split(element_tag)
            if len(parts) > 1:
                section = parts[1].split('[')[0].split('\n')[0].strip()
                if section:
                    return section
        
        return ""
    
    def _get_dynamic_threshold(self, score: float) -> float:
        """Get dynamic threshold for ruling out"""
        if score >= 0.3:
            return 0.1
        elif score >= 0.2:
            return 0.1
        else:
            return 0.05
    
    def _analyze_missing_information(self, answer: str, oldcarts_element: str) -> List[str]:
        """Analyze what information is missing from answer"""
        if not self.active_guidelines:
            return []
        
        # Collect all includes terms from active guidelines
        all_includes = set()
        for g in self.active_guidelines:
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                all_includes.update(t.lower() for t in includes)
        
        # Check which terms are missing from answer
        answer_lower = answer.lower()
        missing = [term for term in all_includes if term not in answer_lower]
        
        return missing[:5]  # Limit to 5
    
    def _generate_clarifying_question(self, oldcarts_element: str, patient_answer: str, 
                                     clarification_count: int, missing_terms: list) -> str:
        """Generate clarifying question with missing terms"""
        if not missing_terms:
            raise ValueError(f"Cannot generate clarifying question for {oldcarts_element} - no missing terms")
        
        # Simple question generation
        if oldcarts_element == 'location':
            options = ", ".join(missing_terms[:3])
            return f"Can you be more specific? For example, is it {options}?"
        else:
            options = ", ".join(missing_terms[:3])
            return f"Can you be more specific? For example, {options}?"
    
    def _ask_next_clinical_question(self) -> Dict[str, Any]:
        """Ask next OLDCARTS question - standard order"""
        if not hasattr(self, 'oldcarts_analysis') or not self.oldcarts_analysis:
            return {'success': False, 'message': 'No OLDCARTS analysis available'}
        
        missing = self.oldcarts_analysis.get('missing_components', [])
        if not missing:
            return {'success': True, 'status': 'completed', 'message': 'Assessment complete'}
        
        # Standard OLDCARTS order
        next_element = missing[0]
        question = self._generate_oldcarts_question_for_component(next_element)
        
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'oldcarts': next_element,
            'focus': 'clinical'
        })
        
        return {
            'success': True,
            'question': question,
            'status': 'questioning'
        }
    
    def _generate_oldcarts_question_for_component(self, component: str) -> str:
        """Generate question for OLDCARTS component - standard OLDCARTS"""
        questions = {
            'onset': "When did this start? Was it sudden or gradual?",
            'location': "Where exactly is the pain located?",
            'duration': "How long does it last?",
            'character': "How would you describe the pain?",
            'aggravating': "What makes it worse?",
            'relieving': "What makes it better?",
            'timing': "When does it occur?",
            'severity': "On a scale of 1-10, how severe is it?"
        }
        return questions.get(component, f"Tell me about {component}")
    
    def _generate_ml_first_question_with_demographics(self) -> Dict[str, Any]:
        """Generate first question with demographics and empathetic statement"""
        # STEP 1: Empathetic statement (only on first question - before any questions)
        if not [item for item in self.conversation_history if item.get('type') == 'question']:
            empathetic_msg = self._generate_empathetic_statement()
            self.conversation_history.append({
                'type': 'statement',
                'message': empathetic_msg
            })
            # Return statement + age question (with pause indicator)
            age_question = "How old are you?"
            self.conversation_history.append({
                'type': 'question',
                'question': age_question,
                'focus': 'age'
            })
        return {
            'success': True,
                'message': empathetic_msg,
                'question': age_question,
                'status': 'questioning',
                'has_pause': True  # Pause between statement and question
            }
        
        # STEP 2: Age question
        if 'age' not in self.demographics:
            question = "How old are you?"
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'age'
            })
            return {
                'success': True,
                'question': question,
                'status': 'questioning'
            }
        
        # STEP 3: Sex question
        if 'sex' not in self.demographics:
            question = "What is your biological sex?"
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'oldcarts': 'demographics',
                'focus': 'sex'
            })
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'Male', 'callback_data': 'sex_male'},
                    {'text': 'Female', 'callback_data': 'sex_female'}
                ]
            }
        
        # STEP 4: Chronicity question
        if 'chronicity' not in self.demographics:
            question = "Is this a new problem or an ongoing issue?"
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'chronicity'
            })
            return {
                'success': True,
                'question': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'New Problem', 'callback_data': 'chronicity_new'},
                    {'text': 'Ongoing Issue', 'callback_data': 'chronicity_recurring'}
                ]
            }
        
        # All demographics collected, start OLDCARTS
        return self._ask_next_clinical_question()
    
    def _generate_empathetic_statement(self) -> str:
        """Generate empathetic opening statement"""
        if self.llm_chat_simple_fn:
            try:
                system_msg = "You are a compassionate medical assistant. Generate a brief, empathetic statement acknowledging the patient's chief complaint."
                user_msg = f"Patient says: '{self.chief_complaint}'\n\nGenerate a brief, empathetic statement (1-2 sentences) that acknowledges their concern and shows you're here to help."
                
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                    max_tokens=60,
                    temperature=0.7
                )
                if response and response.strip():
                    return response.strip()
        except Exception as e:
                self._capture_debug(f"[Engine] ⚠️ Failed to generate empathetic statement: {e}")
        
        # Fallback - always return something
        return f"I understand you're experiencing {self.chief_complaint}. I'm here to help figure out what's going on."
    
    def process_answer(self, user_answer: str) -> Dict[str, Any]:
        """Process user answer and continue assessment"""
        # Check if user is asking a question instead of answering
        if self._is_user_asking_question(user_answer):
            return self._handle_user_question(user_answer)
        
        # Record answer
        self.conversation_history.append({
            'type': 'answer',
            'answer': user_answer
        })
        
        # Get last question
        last_q = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question' or item['type'] == 'statement':
                last_q = item
                break
        
        # Handle statement responses (empathetic statement doesn't need user response)
        if last_q and last_q.get('type') == 'statement':
            # Statement was shown, now ask age question
            return self._generate_ml_first_question_with_demographics()
        
        # Handle demographics
        if last_q and last_q.get('focus') == 'age':
            try:
                age = int(''.join(filter(str.isdigit, user_answer)))
                self.demographics['age'] = age
                return self._generate_ml_first_question_with_demographics()
            except:
                return {'success': False, 'message': 'Please provide a valid age'}
        
        if last_q and last_q.get('focus') == 'sex':
            if 'male' in user_answer.lower() or user_answer == 'sex_male':
                self.demographics['sex'] = 'male'
            elif 'female' in user_answer.lower() or user_answer == 'sex_female':
                self.demographics['sex'] = 'female'
            return self._generate_ml_first_question_with_demographics()
        
        if last_q and last_q.get('focus') == 'chronicity':
            if 'new' in user_answer.lower() or user_answer == 'chronicity_new':
                self.demographics['chronicity'] = 'new'
            elif 'ongoing' in user_answer.lower() or 'recurring' in user_answer.lower() or user_answer == 'chronicity_recurring':
                self.demographics['chronicity'] = 'recurring'
            return self._generate_ml_first_question_with_demographics()
        
        # Handle clinical answers
        return self._process_clinical_answer(user_answer)
    
    def _is_user_asking_question(self, user_input: str) -> bool:
        """Detect if user is asking a question instead of answering"""
        question_indicators = [
            'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'can you', 'could you', 'would you', 'please explain',
            'what do you mean', 'what does that mean', 'i don\'t understand',
            'i\'m confused', 'explain', 'clarify', '?'
        ]
        
        user_lower = user_input.lower().strip()
        
        # Check for question marks
        if '?' in user_lower:
            return True
        
        # Check for question indicators at start
        for indicator in question_indicators:
            if user_lower.startswith(indicator):
                return True
        
        # Check for question patterns
        if any(indicator in user_lower for indicator in ['what do you mean', 'what does that mean', 'i don\'t understand']):
            return True
        
        return False
    
    def _handle_user_question(self, user_question: str) -> Dict[str, Any]:
        """Handle user questions using LLM with conversation context"""
        if not self.llm_chat_fn:
            return {
                'success': False,
                'message': 'I understand you have a question, but I need an answer to continue. Please try to answer the question I asked.'
            }
        
        # Build conversation context
        context_lines = []
        for item in self.conversation_history[-6:]:  # Last 6 items for context
            if item['type'] == 'question':
                context_lines.append(f"Assistant: {item['question']}")
            elif item['type'] == 'answer':
                context_lines.append(f"Patient: {item['answer']}")
        
        context = "\n".join(context_lines) if context_lines else "No previous conversation."
        
        # Get current question being asked
        last_question = None
        for item in reversed(self.conversation_history):
            if item['type'] == 'question' and item.get('focus') != 'age' and item.get('focus') != 'sex':
                last_question = item['question']
                break
        
        # Build LLM prompt
        system_msg = """You are a helpful medical assistant. The patient has asked a question during a medical assessment. 
Provide a clear, empathetic, and brief explanation to help them understand and answer the original question."""
        
        user_msg = f"""Context from recent conversation:
{context}

Current question being asked: {last_question or 'General assessment'}

Patient's question: {user_question}

Provide a helpful response that:
1. Answers their question or clarifies what they're confused about
2. Briefly restates what you need from them to continue
3. Is empathetic and encouraging

Response:"""
        
        try:
            # Use LLM to generate context-aware response
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Record the explanation as a response (not a question)
            self.conversation_history.append({
                'type': 'answer',
                'answer': user_question
            })
            
            # Just return the explanation - let conversation flow naturally
            return {
                'success': True,
                'message': response.strip(),
                'status': 'questioning',
                'is_user_question': True
            }
            
        except Exception as e:
            self._capture_debug(f"[Engine] ⚠️ LLM question handling failed: {e}")
            return {
                'success': True,
                'message': f"I understand you have a question. Let me clarify: {last_question or 'Please answer the question I asked.'}",
                'status': 'questioning'
            }
