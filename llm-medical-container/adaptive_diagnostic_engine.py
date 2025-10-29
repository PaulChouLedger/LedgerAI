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
        self.temperature = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.2'))
        
        # Initialize debug capture
        self._captured_debug_output = []
        self.current_category = None
        
        # Initialize Medical Rule Engine
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
            self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized")
            
            # Build FAISS anatomical index for fast medical_rules.json lookup
            self._build_anatomical_faiss_index()
        except ImportError:
            try:
                import sys
                sys.path.append('/app/ml')
                from medical_rule_engine import MedicalRuleEngine
                self.medical_rule_engine = MedicalRuleEngine(embedding_model=self.embedding_model)
                self._capture_debug(f"[Engine] ✅ Medical Rule Engine initialized (alternative path)")
                
                # Build FAISS anatomical index for fast medical_rules.json lookup
                self._build_anatomical_faiss_index()
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
        
        # FAISS anatomical index for fast medical_rules.json lookup
        self.anatomical_faiss_index = None
        self.anatomical_terms = []
        self.anatomical_conditions = {}
        
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
        
        # Process any detected OLDCARTS elements from the initial prompt
        answered_components = oldcarts_analysis.get('answered_components', {})
        if answered_components:
            self._capture_debug(f"[Engine] 🔄 Processing {len(answered_components)} detected elements from initial prompt")
            for element, detected_terms in answered_components.items():
                self._capture_debug(f"[Engine] 📊 Processing {element}: {detected_terms}")
                # Create a fake question entry for this element so _process_clinical_answer knows which element to process
                self.conversation_history.append({
                    'type': 'question',
                    'question': f"Tell me about {element}",
                    'oldcarts': element,
                    'focus': 'clinical'
                })
                # Store the RAW answer for context (full prompt, not just detected terms)
                self.conversation_history.append({
                    'type': 'answer',
                    'answer': chief_complaint,  # Store full raw text
                    'oldcarts': element
                })
                # Just mark the element as covered without processing through _process_clinical_answer
                # This prevents the function from returning a result that overrides the empathetic statement
                if element == 'onset':
                    self.oldcarts_covered['O'] = True
                elif element == 'location':
                    self.oldcarts_covered['L'] = True
                elif element == 'duration':
                    self.oldcarts_covered['D'] = True
                elif element == 'character':
                    self.oldcarts_covered['C'] = True
                elif element == 'aggravating':
                    self.oldcarts_covered['A'] = True
                elif element == 'relieving':
                    self.oldcarts_covered['R'] = True
                elif element == 'timing':
                    self.oldcarts_covered['T'] = True
                elif element == 'severity':
                    self.oldcarts_covered['S'] = True
                
                self._capture_debug(f"[Engine]   ✅ {element} marked as covered from initial prompt")
        
        # Start with empathetic statement + symptom gathering question
        has_shown_statement = any(item.get('type') == 'statement' for item in self.conversation_history)
        if not has_shown_statement:
            empathetic_msg = self._generate_empathetic_statement()
            
            # Generate symptom gathering question using LLM
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Generate a natural, empathetic question to ask the patient to describe more about their symptoms."
                user_msg = "Generate a natural question to ask the patient to tell you more about their symptoms. Make it empathetic and conversational. Return only the question, no other text."
                
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=40,
                    temperature=self.temperature
                )
                symptom_question = response.strip() if response and response.strip() else "Tell me more about your symptoms so I can better understand what you're experiencing."
            else:
                symptom_question = "Tell me more about your symptoms so I can better understand what you're experiencing."
            
            self.conversation_history.append({
                'type': 'statement',
                'message': empathetic_msg
            })
            self.conversation_history.append({
                'type': 'question',
                'question': symptom_question,
                'focus': 'symptom_gathering'
            })
            return {
                'success': True,
                'message': f"{empathetic_msg}\n\n{symptom_question}",
                'status': 'questioning'
            }
        else:
            # Statement already shown, proceed to demographics
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
        """Parse prompt against structured OLDCARTS using FAISS term matching + medical_rules.json"""
        if not guidelines:
            return {
                'answered_components': {},
                'missing_components': ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity'],
                'anatomical_analysis': {}
            }
        
        # STEP 1: Extract anatomical information using medical_rules.json
        anatomical_analysis = self._analyze_prompt_anatomy(prompt)
        self._capture_debug(f"[Engine] 🏥 Anatomical analysis: {anatomical_analysis}")
        
        # STEP 2: Use FAISS-based term matching with extensive synonyms
        answered_components = {}
        
        # Use FAISS to find matching terms - relies on extensive synonym files
        if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
            all_elements = ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity']
            
            for element in all_elements:
                # Use FAISS to find matching terms with semantic similarity
                matching_terms = self.medical_rule_engine.find_matching_terms_faiss(prompt, element, threshold=0.65)
                if matching_terms:
                    answered_components[element] = matching_terms
                    self._capture_debug(f"[Engine] 📍 {element}: {matching_terms}")
        
        # STEP 3: Enhanced location analysis using medical_rules.json + anatomical direction
        if anatomical_analysis.get('direction'):
            # If we detected anatomical direction but no location matches, create location matches
            if 'location' not in answered_components:
                location_terms = self._generate_anatomical_location_terms(anatomical_analysis['direction'])
                if location_terms:
                    answered_components['location'] = location_terms
                    self._capture_debug(f"[Engine] 🏥 Generated location terms from anatomical analysis: {location_terms}")
            
            # Enhance existing location matches with anatomical terms
            location_analysis = self._enhance_location_analysis(prompt, answered_components.get('location', []), anatomical_analysis)
            answered_components['location'] = location_analysis.get('enhanced_terms', answered_components.get('location', []))
            anatomical_analysis['location_enhancement'] = location_analysis
        
        answered_elements = list(answered_components.keys())
        missing_elements = [element for element in ['onset', 'location', 'duration', 'character', 'aggravating', 'relieving', 'timing', 'severity'] if element not in answered_elements]
        
        return {
            'answered_components': answered_components,
            'missing_components': missing_elements,
            'anatomical_analysis': anatomical_analysis
        }
    
    def _parse_prompt_against_structured_oldcarts_regex(self, prompt: str, guidelines: List[Dict]) -> Dict[str, Any]:
        """Fallback regex-based parsing"""
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
                    'message': question,
                    'status': 'questioning',
                    'buttons': [
                        {'text': 'Male', 'callback_data': 'sex_male'},
                        {'text': 'Female', 'callback_data': 'sex_female'}
                    ]
                }
                return self._ask_next_clinical_question()
        
        # Handle onset (documentation only)
        if oldcarts_element == 'onset':
            # Mark onset as covered and store the answer
            self.oldcarts_covered['O'] = True
            # Update missing_components list to remove onset
            if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                if 'onset' in self.oldcarts_analysis['missing_components']:
                    self.oldcarts_analysis['missing_components'].remove('onset')
            self._capture_debug(f"[Engine] ✅ onset marked as complete")
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
        
        clarification_needed = False
        if clarification_count < 2 and self.active_guidelines:
            try:
                missing_terms = self._analyze_missing_information(answer, oldcarts_element)
                # Only need clarification if there are missing terms AND no terms are satisfied
                if missing_terms:
                    # Check if any terms are satisfied (exact or semantic match)
                    satisfied_terms = self._get_satisfied_terms(answer, oldcarts_element)
                    if not satisfied_terms:
                        clarification_needed = True
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
        
        # Only mark element as covered if NO clarification needed
        if not clarification_needed:
            element_map = {'onset': 'O', 'location': 'L', 'duration': 'D', 'character': 'C',
                          'aggravating': 'A', 'relieving': 'R', 'timing': 'T', 'severity': 'S'}
            if oldcarts_element in element_map:
                self.oldcarts_covered[element_map[oldcarts_element]] = True
                # Update missing_components list to remove this element
                if self.oldcarts_analysis and 'missing_components' in self.oldcarts_analysis:
                    if oldcarts_element in self.oldcarts_analysis['missing_components']:
                        self.oldcarts_analysis['missing_components'].remove(oldcarts_element)
                self._capture_debug(f"[Engine] ✅ {oldcarts_element} marked as complete")
        else:
            self._capture_debug(f"[Engine] ⏳ {oldcarts_element} needs clarification - not marked complete")
        
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
        """Analyze what information is missing using unified function with FAISS semantic matching"""
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
        
        if not all_includes:
            return []
        
        # Use unified function to check which terms are satisfied
        satisfied_terms = set()
        answer_lower = answer.lower()
        
        # Check each term using the same logic as unified function
        for term in all_includes:
            term_satisfied = False
            
            # 1. Exact/substring matching (fast path)
            if term in answer_lower or answer_lower in term:
                term_satisfied = True
            else:
                # 2. Use FAISS semantic matching (same as unified function)
                if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
                    semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(answer, oldcarts_element, threshold=0.7)
                    if term in [t.lower() for t in semantic_matches]:
                        term_satisfied = True
                    else:
                        # 3. Direct semantic similarity check (same as unified function)
                        if self.embedding_model:
                            try:
                                embeddings = self.embedding_model.encode([answer_lower, term])
                                similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                                if similarity >= 0.7:  # Same threshold as FAISS
                                    term_satisfied = True
                            except Exception:
                                pass
            
            if term_satisfied:
                satisfied_terms.add(term)
        
        # Terms are missing if they're not satisfied
        missing = [term for term in all_includes if term not in satisfied_terms]
        
        return missing[:5]  # Limit to 5
    
    def _get_satisfied_terms(self, answer: str, oldcarts_element: str) -> set:
        """Get terms that are satisfied using unified function logic"""
        if not self.active_guidelines:
            return set()
        
        # Collect all includes terms from active guidelines
        all_includes = set()
        for g in self.active_guidelines:
            structured = g.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
            element_data = structured.get(oldcarts_element, {})
            if isinstance(element_data, dict):
                includes = element_data.get('includes', [])
                all_includes.update(t.lower() for t in includes)
        
        if not all_includes:
            return set()
        
        # Use unified function to check which terms are satisfied
        satisfied_terms = set()
        answer_lower = answer.lower()
        
        # Check each term using the same logic as unified function
        for term in all_includes:
            term_satisfied = False
            
            # 1. Exact/substring matching (fast path)
            if term in answer_lower or answer_lower in term:
                term_satisfied = True
            else:
                # 2. Use FAISS semantic matching (same as unified function)
                if self.medical_rule_engine and hasattr(self.medical_rule_engine, 'find_matching_terms_faiss'):
                    semantic_matches = self.medical_rule_engine.find_matching_terms_faiss(answer, oldcarts_element, threshold=0.7)
                    if term in [t.lower() for t in semantic_matches]:
                        term_satisfied = True
                    else:
                        # 3. Direct semantic similarity check (same as unified function)
                        if self.embedding_model:
                            try:
                                embeddings = self.embedding_model.encode([answer_lower, term])
                                similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                                if similarity >= 0.7:  # Same threshold as FAISS
                                    term_satisfied = True
                            except Exception:
                                pass
            
            if term_satisfied:
                satisfied_terms.add(term)
        
        return satisfied_terms
    
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
            'message': question,
            'status': 'questioning'
        }
    
    def _generate_oldcarts_question_for_component(self, component: str) -> str:
        """Generate question for OLDCARTS component using LLM for natural variation"""
        if self.llm_chat_simple_fn:
            system_msg = "You are a medical assistant. Generate a natural, conversational question to ask about a specific aspect of a patient's symptoms."
            user_msg = f"Generate a natural question to ask about the {component} of the patient's symptoms. Make it conversational and empathetic. Return only the question, no other text."
            
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=50,
                temperature=self.temperature
            )
            if response and response.strip():
                return response.strip()
        
        # Fallback questions
        fallback_questions = {
            'onset': "When did this start? Was it sudden or gradual?",
            'location': "Where exactly is the pain located?",
            'duration': "How long does it last?",
            'character': "How would you describe the pain?",
            'aggravating': "What makes it worse?",
            'relieving': "What makes it better?",
            'timing': "When does it occur?",
            'severity': "On a scale of 1-10, how severe is it?"
        }
        return fallback_questions.get(component, f"Tell me about {component}")
    
    def _generate_ml_first_question_with_demographics(self) -> Dict[str, Any]:
        """Generate first question with demographics and empathetic statement"""
        # STEP 1: Empathetic statement (only on first question - completely separate)
        # Show empathetic statement only once per assessment before any questions
        has_shown_statement = any(item.get('type') == 'statement' for item in self.conversation_history)
        has_asked_any_question = any(item.get('type') == 'question' for item in self.conversation_history)
        if not has_shown_statement and not has_asked_any_question:
            empathetic_msg = self._generate_empathetic_statement()
            self.conversation_history.append({
                'type': 'statement',
                'message': empathetic_msg
            })
            return {
                'success': True,
                'message': empathetic_msg,
                'status': 'questioning',
                'has_pause': True  # Pause before next question
            }
            
        # STEP 2: Age question (separate from empathetic statement)
        if 'age' not in self.demographics:
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Generate a natural, conversational question to ask for the patient's age."
                user_msg = "Generate a natural question to ask for the patient's age. Make it conversational and professional. Return only the question, no other text."
                
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=30,
                    temperature=self.temperature
                )
                question = response.strip() if response and response.strip() else "How old are you?"
            else:
                question = "How old are you?"
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'age'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning'
            }
        
        # STEP 3: Sex question
        if 'sex' not in self.demographics:
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Generate a natural, conversational question to ask for the patient's biological sex."
                user_msg = "Generate a natural question to ask for the patient's biological sex. Make it conversational and professional. Return only the question, no other text."
                
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=30,
                    temperature=self.temperature
                )
                question = response.strip() if response and response.strip() else "What is your biological sex?"
            else:
                question = "What is your biological sex?"
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'oldcarts': 'demographics',
                'focus': 'sex'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'Male', 'callback_data': 'sex_male'},
                    {'text': 'Female', 'callback_data': 'sex_female'}
                ]
            }
        
        # STEP 4: Chronicity question
        if 'chronicity' not in self.demographics:
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Generate a natural, conversational question to ask if the patient's problem is new or ongoing."
                user_msg = "Generate a natural question to ask if this is a new problem or ongoing issue. Make it conversational and professional. Return only the question, no other text."
                
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=40,
                    temperature=self.temperature
                )
                question = response.strip() if response and response.strip() else "Is this a new problem or an ongoing issue?"
            else:
                question = "Is this a new problem or an ongoing issue?"
            
            self.conversation_history.append({
                'type': 'question',
                'question': question,
                'focus': 'chronicity'
            })
            return {
                'success': True,
                'message': question,
                'status': 'questioning',
                'buttons': [
                    {'text': 'New Problem', 'callback_data': 'chronicity_new'},
                    {'text': 'Ongoing Issue', 'callback_data': 'chronicity_recurring'}
                ]
            }
        
        # All demographics collected, start OLDCARTS
        return self._ask_next_clinical_question()
    
    def _generate_empathetic_statement(self) -> str:
        """Generate empathetic opening statement using LLM"""
        if self.llm_chat_simple_fn:
            system_msg = "You are a compassionate medical assistant. Generate a brief, empathetic statement acknowledging the patient's chief complaint."
            user_msg = f"Patient says: '{self.chief_complaint}'\n\nGenerate a brief, empathetic statement (1-2 sentences) that acknowledges their concern and shows you're here to help."
            
            response = self.llm_chat_simple_fn(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=60,
                temperature=self.temperature
            )
            if response and response.strip():
                return response.strip()
        
        # Fallback template
        return f"I'm so sorry to hear that you're experiencing {self.chief_complaint}. Please know that I'm here for you and will do my best to help you feel more comfortable during your visit."
    
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
        
        # Handle symptom gathering response
        if last_q and last_q.get('focus') == 'symptom_gathering':
            # Analyze the symptom response for additional OLDCARTS information
            self._capture_debug(f"[Engine] 🔍 Analyzing symptom response: '{user_answer}'")
            
            # Parse the response for additional OLDCARTS elements
            additional_oldcarts = self._parse_prompt_against_structured_oldcarts(user_answer, list(self.all_guidelines.values()))
            answered_components = additional_oldcarts.get('answered_components', {})
            
            if answered_components:
                self._capture_debug(f"[Engine] 📊 Found additional OLDCARTS elements: {answered_components}")
                for element, detected_terms in answered_components.items():
                    # Mark as covered
                    if element == 'onset':
                        self.oldcarts_covered['O'] = True
                    elif element == 'location':
                        self.oldcarts_covered['L'] = True
                    elif element == 'duration':
                        self.oldcarts_covered['D'] = True
                    elif element == 'character':
                        self.oldcarts_covered['C'] = True
                    elif element == 'aggravating':
                        self.oldcarts_covered['A'] = True
                    elif element == 'relieving':
                        self.oldcarts_covered['R'] = True
                    elif element == 'timing':
                        self.oldcarts_covered['T'] = True
                    elif element == 'severity':
                        self.oldcarts_covered['S'] = True
                    
                    self._capture_debug(f"[Engine]   ✅ {element} marked as covered from symptom response")
            
            # Transition to demographics with a natural message using LLM
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Generate a natural, appreciative transition message before asking demographic questions."
                user_msg = "Generate a brief, natural message thanking the patient for sharing symptom information and transitioning to asking demographic questions. Make it conversational and professional. Return only the message, no other text."
                
                response = self.llm_chat_simple_fn(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=40,
                    temperature=self.temperature
                )
                transition_msg = response.strip() if response and response.strip() else "Thank you for sharing that information. Let me ask you some questions to better assist you."
            else:
                transition_msg = "Thank you for sharing that information. Let me ask you some questions to better assist you."
            
            self.conversation_history.append({
                'type': 'statement',
                'message': transition_msg
            })
            
            # Now ask for demographics
            return self._generate_ml_first_question_with_demographics()
        
        # Handle statement responses (empathetic statement doesn't need user response)
        if last_q and last_q.get('type') == 'statement':
            # Statement was shown, now ask age question (skip statement check since we already showed it)
            if 'age' not in self.demographics:
                question = "How old are you?"
                self.conversation_history.append({
                    'type': 'question',
                    'question': question,
                    'focus': 'age'
                })
                return {
                    'success': True,
                    'message': question,
                    'status': 'questioning'
                }
            else:
                # Age already collected, continue with next demographic
                return self._generate_ml_first_question_with_demographics()
        
        # Handle demographics
        if last_q and last_q.get('focus') == 'age':
            # Simple age validation - accept direct numeric answers
            age_str = user_answer.strip()
            if age_str.isdigit():
                age = int(age_str)
                if 0 <= age <= 150:
                    self.demographics['age'] = age
                    return self._generate_ml_first_question_with_demographics()
            
            # Fallback to LLM if simple validation fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Extract the patient's age from their response. Return ONLY a number between 0-150, or 'invalid' if not a valid age."
                user_msg = f"Patient said: '{user_answer}'\n\nExtract age as a number only:"
        
                response = self.llm_chat_simple_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    max_tokens=10,
                    temperature=self.temperature
                )
                
                age_str = response.strip()
                if age_str.isdigit():
                    age = int(age_str)
                    if 0 <= age <= 150:
                        self.demographics['age'] = age
                        return self._generate_ml_first_question_with_demographics()
            
            return {'success': False, 'message': 'Please provide your age as a number (e.g., 25, thirty-five, etc.)'}
        
        if last_q and last_q.get('focus') == 'sex':
            # Simple sex validation - accept direct answers
            sex_str = user_answer.strip().lower()
            if sex_str in ['male', 'female', 'm', 'f']:
                if sex_str in ['m', 'f']:
                    sex_str = 'male' if sex_str == 'm' else 'female'
                self.demographics['sex'] = sex_str
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'sex_male':
                self.demographics['sex'] = 'male'
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'sex_female':
                self.demographics['sex'] = 'female'
                return self._generate_ml_first_question_with_demographics()
            
            # Fallback to LLM if simple validation fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Extract the patient's biological sex from their response. Return ONLY 'male', 'female', or 'invalid'."
                user_msg = f"Patient said: '{user_answer}'\n\nExtract biological sex (male/female) only:"
                
                response = self.llm_chat_simple_fn(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    max_tokens=10,
                    temperature=self.temperature
                )
                
                sex_str = response.strip().lower()
                if sex_str in ['male', 'female']:
                    self.demographics['sex'] = sex_str
                    return self._generate_ml_first_question_with_demographics()
            
            return {'success': False, 'message': 'Please specify your biological sex (male or female)'}
        
        if last_q and last_q.get('focus') == 'chronicity':
            # Simple keyword matching first (more reliable than LLM)
            user_answer_lower = user_answer.lower().strip()
            
            # Check for button callbacks first
            if user_answer == 'chronicity_new':
                self.demographics['chronicity'] = 'new'
                return self._generate_ml_first_question_with_demographics()
            elif user_answer == 'chronicity_recurring':
                self.demographics['chronicity'] = 'recurring'
                return self._generate_ml_first_question_with_demographics()
            
            # Simple keyword matching
            if any(word in user_answer_lower for word in ['new', 'recent', 'today', 'yesterday', 'this week', 'sudden', 'acute']):
                self.demographics['chronicity'] = 'new'
                return self._generate_ml_first_question_with_demographics()
            elif any(word in user_answer_lower for word in ['ongoing', 'recurring', 'chronic', 'months', 'years', 'always', 'frequent', 'often']):
                self.demographics['chronicity'] = 'recurring'
                return self._generate_ml_first_question_with_demographics()
            
            # Fallback to LLM if simple matching fails
            if self.llm_chat_simple_fn:
                system_msg = "You are a medical assistant. Determine if the patient's problem is new or recurring. Return ONLY 'new', 'recurring', or 'invalid'."
                user_msg = f"Patient said: '{user_answer}'\n\nIs this a new problem or recurring/ongoing? (new/recurring):"
                
                try:
                    response = self.llm_chat_simple_fn(
                        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                        max_tokens=10,
                        temperature=self.temperature
                    )
                    
                    chronicity_str = response.strip().lower()
                    if chronicity_str in ['new', 'recurring']:
                        self.demographics['chronicity'] = chronicity_str
                        return self._generate_ml_first_question_with_demographics()
                    elif 'new' in chronicity_str:
                        self.demographics['chronicity'] = 'new'
                        return self._generate_ml_first_question_with_demographics()
                    elif 'recurring' in chronicity_str or 'ongoing' in chronicity_str:
                        self.demographics['chronicity'] = 'recurring'
                        return self._generate_ml_first_question_with_demographics()
                except Exception as e:
                    # LLM failed, fall through to error message
                    pass
            
            return {'success': False, 'message': 'Please specify if this is a new problem or ongoing issue'}
        
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
                temperature=self.temperature
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
    
    def _build_anatomical_faiss_index(self):
        """Build FAISS index for fast anatomical analysis using medical_rules.json"""
        if not self.medical_rule_engine or not self.embedding_model:
            return
        
        try:
            import faiss
            import numpy as np
            
            # Load medical_rules.json
            medical_rules = self.medical_rule_engine.medical_rules
            if not medical_rules:
                return
            
            # Collect all anatomical terms and their conditions
            anatomical_terms = []
            anatomical_conditions = {}
            
            for organ_system, anatomical_types in medical_rules.items():
                for anatomical_type, conditions in anatomical_types.items():
                    # Create terms for each anatomical type
                    if anatomical_type == 'right_only':
                        terms = ['right', 'right side', 'right sided', 'ruq', 'rlq', 'right upper quadrant', 'right lower quadrant']
                    elif anatomical_type == 'left_only':
                        terms = ['left', 'left side', 'left sided', 'luq', 'llq', 'left upper quadrant', 'left lower quadrant']
                    elif anatomical_type == 'bilateral':
                        terms = ['bilateral', 'both sides', 'both', 'symmetrical', 'all over', 'everywhere']
                    elif anatomical_type == 'midline':
                        terms = ['midline', 'center', 'central', 'middle', 'epigastric', 'suprapubic', 'periumbilical']
                    else:
                        continue
                    
                    for term in terms:
                        anatomical_terms.append(term)
                        if term not in anatomical_conditions:
                            anatomical_conditions[term] = []
                        anatomical_conditions[term].extend([(organ_system, anatomical_type, condition) for condition in conditions])
            
            if not anatomical_terms:
                return
            
            # Build FAISS index
            embeddings = self.embedding_model.encode(anatomical_terms)
            embeddings = embeddings.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            self.anatomical_faiss_index = index
            self.anatomical_terms = anatomical_terms
            self.anatomical_conditions = anatomical_conditions
            
            self._capture_debug(f"[FAISS] 🏥 Built anatomical index: {len(anatomical_terms)} terms")
            
        except Exception as e:
            self._capture_debug(f"[FAISS] ⚠️ Error building anatomical index: {e}")
    
    def _analyze_prompt_anatomy(self, prompt: str) -> Dict[str, Any]:
        """Fast anatomical analysis using FAISS + medical_rules.json"""
        if not self.anatomical_faiss_index or not self.embedding_model:
            return self._analyze_prompt_anatomy_fallback(prompt)
        
        try:
            import faiss
            import numpy as np
            
            # Encode prompt
            prompt_embedding = self.embedding_model.encode([prompt])
            prompt_embedding = prompt_embedding.astype('float32')
            faiss.normalize_L2(prompt_embedding)
            
            # Search FAISS index
            scores, indices = self.anatomical_faiss_index.search(prompt_embedding, k=5)
            
            # Analyze results
            direction = None
            organ_system = None
            compatible_conditions = set()
            incompatible_conditions = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if score >= 0.7:  # High confidence threshold
                    term = self.anatomical_terms[idx]
                    conditions = self.anatomical_conditions.get(term, [])
                    
                    # Extract direction and organ system
                    for sys, anat_type, condition in conditions:
                        if not organ_system:
                            organ_system = sys
                        
                        if not direction:
                            if anat_type == 'right_only':
                                direction = 'right'
                            elif anat_type == 'left_only':
                                direction = 'left'
                            elif anat_type == 'bilateral':
                                direction = 'bilateral'
                            elif anat_type == 'midline':
                                direction = 'midline'
                        
                        # Categorize conditions
                        if anat_type in ['right_only', 'left_only', 'bilateral', 'midline']:
                            compatible_conditions.add(condition)
                        else:
                            incompatible_conditions.add(condition)
            
            return {
                'direction': direction,
                'organ_system': organ_system,
                'compatible_conditions': list(compatible_conditions),
                'incompatible_conditions': list(incompatible_conditions),
                'confidence': float(max(scores[0])) if len(scores[0]) > 0 else 0.0
            }
            
        except Exception as e:
            self._capture_debug(f"[FAISS] ⚠️ Error in anatomical analysis: {e}")
            return self._analyze_prompt_anatomy_fallback(prompt)
    
    def _analyze_prompt_anatomy_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback anatomical analysis using simple keyword matching"""
        prompt_lower = prompt.lower()
        
        # Simple keyword matching
        if any(word in prompt_lower for word in ['right', 'ruq', 'rlq']):
            direction = 'right'
        elif any(word in prompt_lower for word in ['left', 'luq', 'llq']):
            direction = 'left'
        elif any(word in prompt_lower for word in ['bilateral', 'both sides', 'both']):
            direction = 'bilateral'
        elif any(word in prompt_lower for word in ['midline', 'center', 'central', 'middle', 'epigastric']):
            direction = 'midline'
        else:
            direction = None
        
        return {
            'direction': direction,
            'organ_system': None,
            'compatible_conditions': [],
            'incompatible_conditions': [],
            'confidence': 0.5 if direction else 0.0
        }
    
    def _enhance_location_analysis(self, prompt: str, faiss_matches: List[str], anatomical_analysis: Dict) -> Dict[str, Any]:
        """Enhance location analysis using anatomical information"""
        enhanced_terms = faiss_matches.copy()
        
        # Add anatomical terms if not already present
        direction = anatomical_analysis.get('direction')
        if direction:
            if direction == 'right':
                anatomical_terms = ['right', 'right side', 'right sided']
            elif direction == 'left':
                anatomical_terms = ['left', 'left side', 'left sided']
            elif direction == 'bilateral':
                anatomical_terms = ['bilateral', 'both sides', 'both']
            elif direction == 'midline':
                anatomical_terms = ['midline', 'center', 'central', 'middle']
            else:
                anatomical_terms = []
            
            # Add anatomical terms that aren't already in FAISS matches
            for term in anatomical_terms:
                if not any(term.lower() in match.lower() for match in enhanced_terms):
                    enhanced_terms.append(term)
        
        return {
            'enhanced_terms': enhanced_terms,
            'anatomical_direction': direction,
            'confidence': anatomical_analysis.get('confidence', 0.0)
        }
    
    def _generate_anatomical_location_terms(self, direction: str) -> List[str]:
        """Generate location terms based on anatomical direction - ONLY vague terms, not specific anatomical locations"""
        if direction == 'right':
            return ['right', 'right side', 'right sided']  # Vague terms only
        elif direction == 'left':
            return ['left', 'left side', 'left sided']  # Vague terms only
        elif direction == 'bilateral':
            return ['bilateral', 'both sides', 'both', 'symmetrical', 'all over', 'everywhere']
        elif direction == 'midline':
            return ['midline', 'center', 'central', 'middle']  # Vague terms only
        else:
            return []
