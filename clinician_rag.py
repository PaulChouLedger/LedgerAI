#!/usr/bin/env python3
"""
Clinician Mode RAG System

Specialized RAG for medical/clinical queries with:
1. Medical-specific embeddings and indexing
2. Medical terminology handling
3. Clinical context preservation
4. Evidence-based search results
5. Integration with medical data ingestion system
"""

import os
import sys
import numpy as np
import faiss
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

# Import existing RAG components (with defensive import)
try:
    sys.path.append('rag-container')
    from rag import AuraRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    AuraRAG = None
    print("Warning: Could not import main RAG components - running in limited mode")

# Import medical data ingester
try:
    from medical_data_ingestion import MedicalDataIngester
except ImportError:
    print("Warning: Could not import medical data ingester")

class ClinicianRAG:
    """
    Specialized RAG system for clinician mode with medical focus
    """

    def __init__(self,
                 medical_embeddings_dir: str = "data/medical/embeddings",
                 general_rag: Optional[AuraRAG] = None):
        """
        Initialize clinician RAG system

        Args:
            medical_embeddings_dir: Directory containing medical embeddings
            general_rag: Optional general RAG instance for fallback
        """
        self.medical_embeddings_dir = Path(medical_embeddings_dir)
        self.general_rag = general_rag

        # Medical-specific components
        self.medical_index = None
        self.medical_chunks = None
        self.medical_metadata = None
        self.medical_encoder = None

        # Medical terminology and context
        self.medical_terms = self._load_medical_terms()
        self.clinical_contexts = {}

        # Initialize medical RAG
        self._load_medical_components()

        print("🏥 Clinician RAG initialized")

    def _load_medical_components(self):
        """Load medical-specific RAG components"""
        print("🔧 Loading medical RAG components...")

        # Check if medical data exists
        if not self.medical_embeddings_dir.exists():
            print("⚠️ Medical embeddings directory not found - running in limited mode")
            return

        # Load medical FAISS index
        medical_index_path = self.medical_embeddings_dir / "medical_index.faiss"
        if medical_index_path.exists():
            try:
                import faiss
                self.medical_index = faiss.read_index(str(medical_index_path))
                print(f"✅ Loaded medical FAISS index: {self.medical_index.ntotal} vectors")
            except Exception as e:
                print(f"❌ Failed to load medical index: {e}")
        else:
            print(f"⚠️ Medical index not found at {medical_index_path}")

        # Load medical chunks
        medical_chunks_path = self.medical_embeddings_dir / "medical_chunks.npy"
        if medical_chunks_path.exists():
            try:
                self.medical_chunks = np.load(medical_chunks_path, allow_pickle=True)
                print(f"✅ Loaded {len(self.medical_chunks)} medical chunks")
            except Exception as e:
                print(f"❌ Failed to load medical chunks: {e}")
        else:
            print(f"⚠️ Medical chunks not found at {medical_chunks_path}")

        # Load medical metadata
        medical_metadata_path = self.medical_embeddings_dir / "medical_metadata.json"
        if medical_metadata_path.exists():
            try:
                with open(medical_metadata_path, 'r') as f:
                    self.medical_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.medical_metadata)} chunks")
            except Exception as e:
                print(f"❌ Failed to load medical metadata: {e}")

        # Load medical encoder (use same as general RAG for consistency)
        if self.general_rag and hasattr(self.general_rag, 'encoder') and self.general_rag.encoder is not None:
            self.medical_encoder = self.general_rag.encoder
            print("✅ Using general RAG encoder for medical queries")
        else:
            print("⚠️ No encoder available for medical queries")

    def _load_medical_terms(self) -> Dict[str, Any]:
        """Load medical terminology database"""
        # This would typically load from a comprehensive medical ontology
        # For now, using a basic set of common medical terms and their relationships
        return {
            'synonyms': {
                'heart attack': 'myocardial infarction',
                'MI': 'myocardial infarction',
                'stroke': 'cerebrovascular accident',
                'CVA': 'cerebrovascular accident',
                'high blood pressure': 'hypertension',
                'diabetes': 'diabetes mellitus',
                'DM': 'diabetes mellitus',
                'COPD': 'chronic obstructive pulmonary disease',
                'pneumonia': 'pneumonia',
                'cancer': 'malignant neoplasm',
                'tumor': 'neoplasm',
                'infection': 'infectious disease',
                'fever': 'pyrexia',
                'pain': 'algia',
                'headache': 'cephalgia',
                'chest pain': 'thoracic pain',
                'shortness of breath': 'dyspnea',
                'difficulty breathing': 'dyspnea',
                'nausea': 'nausea',
                'vomiting': 'emesis',
                'diarrhea': 'diarrhea',
                'constipation': 'constipation',
                'fatigue': 'fatigue',
                'weakness': 'asthenia',
                'dizziness': 'vertigo',
                'confusion': 'delirium',
                'seizure': 'convulsion',
                'paralysis': 'paresis',
                'numbness': 'paresthesia',
                'tingling': 'paresthesia'
            },
            'abbreviations': {
                'MI': 'myocardial infarction',
                'CAD': 'coronary artery disease',
                'CHF': 'congestive heart failure',
                'COPD': 'chronic obstructive pulmonary disease',
                'DM': 'diabetes mellitus',
                'HTN': 'hypertension',
                'CVA': 'cerebrovascular accident',
                'TIA': 'transient ischemic attack',
                'PE': 'pulmonary embolism',
                'DVT': 'deep vein thrombosis',
                'UTI': 'urinary tract infection',
                'PNA': 'pneumonia',
                'URI': 'upper respiratory infection',
                'LRI': 'lower respiratory infection',
                'GERD': 'gastroesophageal reflux disease',
                'IBD': 'inflammatory bowel disease',
                'RA': 'rheumatoid arthritis',
                'OA': 'osteoarthritis',
                'MS': 'multiple sclerosis',
                'PD': 'Parkinson disease',
                'AD': 'Alzheimer disease',
                'HIV': 'human immunodeficiency virus',
                'AIDS': 'acquired immunodeficiency syndrome',
                'TB': 'tuberculosis',
                'MRSA': 'methicillin-resistant Staphylococcus aureus',
                'VRE': 'vancomycin-resistant Enterococcus',
                'ESBL': 'extended-spectrum beta-lactamase',
                'MDR': 'multidrug-resistant',
                'XDR': 'extensively drug-resistant',
                'PDR': 'pandrug-resistant'
            }
        }

    def expand_medical_query(self, query: str) -> List[str]:
        """
        Expand medical query with synonyms and abbreviations

        Args:
            query: Original medical query

        Returns:
            List of expanded query variations
        """
        expanded_queries = [query.lower()]

        # Add synonyms
        for term, synonym in self.medical_terms['synonyms'].items():
            if term in query.lower():
                expanded_queries.append(query.lower().replace(term, synonym))

        # Add abbreviations
        for abbr, full_term in self.medical_terms['abbreviations'].items():
            if abbr.lower() in query.lower():
                expanded_queries.append(query.lower().replace(abbr.lower(), full_term))

        # Remove duplicates and return
        return list(set(expanded_queries))

    def search_medical_info(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search medical information using clinician-optimized RAG

        Args:
            query: Medical/clinical query
            k: Number of results to return

        Returns:
            List of medical search results with clinical context
        """
        if not query or not isinstance(query, str):
            return []

        print(f"🏥 Searching medical info for: '{query}'")

        # Expand query with medical terminology
        expanded_queries = self.expand_medical_query(query)
        print(f"🔍 Expanded queries: {expanded_queries}")

        # If no medical index available, fall back to general RAG
        if self.medical_index is None or self.medical_encoder is None:
            print("⚠️ No medical index available, falling back to general RAG")
            if self.general_rag:
                return self.general_rag.search(query, k)
            else:
                return []

        all_results = []

        # Search each expanded query variation
        for expanded_query in expanded_queries:
            try:
                # Encode query
                import torch
                with torch.no_grad():
                    query_embedding = self.medical_encoder.encode([expanded_query], convert_to_numpy=True, show_progress_bar=False)
                    query_embedding = query_embedding.astype(np.float32)

                # Normalize for Inner Product metric
                if self.medical_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
                    query_embedding = query_embedding / norms

                # Search medical index
                distances, indices = self.medical_index.search(query_embedding, min(k * 2, 50))  # Get more for diversity

                # Process results
                for distance, idx in zip(distances[0], indices[0]):
                    idx = int(idx)

                    if idx < len(self.medical_chunks):
                        chunk = self.medical_chunks[idx]

                        # Get metadata for this chunk
                        metadata = None
                        if self.medical_metadata and idx < len(self.medical_metadata):
                            metadata = self.medical_metadata[idx]

                        similarity_score = float(1.0 / (1.0 + distance))

                        # Apply medical relevance filtering
                        if self._is_medically_relevant(query, chunk, similarity_score):
                            result = {
                                'chunk': chunk,
                                'score': similarity_score,
                                'distance': float(distance),
                                'rank': len(all_results) + 1,
                                'metadata': metadata,
                                'query_expansion': expanded_query
                            }
                            all_results.append(result)

                            if len(all_results) >= k:
                                break

            except Exception as e:
                print(f"❌ Error searching expanded query '{expanded_query}': {e}")
                continue

        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x['score'], reverse=True)

        # Add clinical context annotations
        for result in all_results[:k]:
            result['clinical_context'] = self._add_clinical_context(result['chunk'], query)

        print(f"✅ Found {len(all_results)} medical results")
        return all_results[:k]

    def _is_medically_relevant(self, query: str, chunk: str, score: float) -> bool:
        """
        Determine if a chunk is medically relevant to the query

        Args:
            query: Original medical query
            chunk: Document chunk
            score: Similarity score

        Returns:
            True if medically relevant
        """
        # Basic relevance threshold
        if score < 0.3:
            return False

        # Check for medical keywords in both query and chunk
        medical_keywords = [
            'patient', 'treatment', 'diagnosis', 'symptom', 'medication', 'therapy',
            'clinical', 'medical', 'health', 'disease', 'condition', 'disorder',
            'syndrome', 'infection', 'inflammation', 'chronic', 'acute'
        ]

        query_lower = query.lower()
        chunk_lower = chunk.lower()

        # Must have some medical keywords
        query_has_medical = any(keyword in query_lower for keyword in medical_keywords)
        chunk_has_medical = any(keyword in chunk_lower for keyword in medical_keywords)

        if not (query_has_medical or chunk_has_medical):
            return False

        # For very specific medical queries, require higher relevance
        specific_terms = ['myocardial', 'diabetes', 'pneumonia', 'cancer', 'stroke']
        if any(term in query_lower for term in specific_terms):
            return score > 0.4

        return True

    def _add_clinical_context(self, chunk: str, query: str) -> Dict[str, Any]:
        """
        Add clinical context annotations to search results

        Args:
            chunk: Document chunk
            query: Original query

        Returns:
            Clinical context dictionary
        """
        context = {
            'evidence_level': self._assess_evidence_level(chunk),
            'clinical_relevance': self._assess_clinical_relevance(chunk, query),
            'specialty': self._identify_medical_specialty(chunk),
            'confidence': self._calculate_confidence(chunk, query)
        }

        return context

    def _assess_evidence_level(self, chunk: str) -> str:
        """Assess evidence level of medical information"""
        chunk_lower = chunk.lower()

        # Evidence indicators
        if any(term in chunk_lower for term in ['randomized controlled trial', 'rct', 'meta-analysis', 'systematic review']):
            return 'high'
        elif any(term in chunk_lower for term in ['clinical trial', 'prospective study', 'cohort study']):
            return 'moderate'
        elif any(term in chunk_lower for term in ['case series', 'retrospective', 'observational']):
            return 'low'
        elif any(term in chunk_lower for term in ['guideline', 'recommendation', 'consensus']):
            return 'guideline'
        else:
            return 'unknown'

    def _assess_clinical_relevance(self, chunk: str, query: str) -> str:
        """Assess clinical relevance to the query"""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        chunk_terms = set(re.findall(r'\b\w+\b', chunk.lower()))

        # Calculate term overlap
        overlap = len(query_terms.intersection(chunk_terms))
        total_terms = len(query_terms.union(chunk_terms))

        if total_terms == 0:
            return 'low'

        overlap_ratio = overlap / total_terms

        if overlap_ratio > 0.3:
            return 'high'
        elif overlap_ratio > 0.15:
            return 'moderate'
        else:
            return 'low'

    def _identify_medical_specialty(self, chunk: str) -> List[str]:
        """Identify relevant medical specialties"""
        chunk_lower = chunk.lower()
        specialties = []

        # Specialty keyword mapping
        specialty_keywords = {
            'cardiology': ['heart', 'cardiac', 'myocardial', 'coronary', 'angina', 'arrhythmia'],
            'pulmonology': ['lung', 'pulmonary', 'respiratory', 'breathing', 'asthma', 'copd'],
            'neurology': ['brain', 'neurological', 'seizure', 'stroke', 'paralysis', 'dementia'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'endocrine', 'metabolic'],
            'gastroenterology': ['stomach', 'intestinal', 'liver', 'digestive', 'gi'],
            'nephrology': ['kidney', 'renal', 'dialysis', 'glomerular'],
            'infectious_disease': ['infection', 'bacteria', 'virus', 'antibiotic', 'fever'],
            'emergency_medicine': ['emergency', 'acute', 'critical', 'trauma', 'resuscitation'],
            'family_medicine': ['primary', 'general', 'preventive', 'screening']
        }

        for specialty, keywords in specialty_keywords.items():
            if any(keyword in chunk_lower for keyword in keywords):
                specialties.append(specialty)

        return specialties if specialties else ['general_medicine']

    def _calculate_confidence(self, chunk: str, query: str) -> float:
        """Calculate confidence score for the result"""
        # Simple confidence based on keyword overlap and evidence level
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        chunk_terms = set(re.findall(r'\b\w+\b', chunk.lower()))

        overlap = len(query_terms.intersection(chunk_terms))
        total_terms = len(query_terms.union(chunk_terms))

        if total_terms == 0:
            return 0.0

        overlap_score = overlap / total_terms

        # Boost confidence for guideline content
        evidence_multiplier = 1.0
        if 'guideline' in chunk.lower() or 'recommendation' in chunk.lower():
            evidence_multiplier = 1.2

        return min(1.0, overlap_score * evidence_multiplier)

    def get_medical_context(self, query: str, results: List[Dict]) -> str:
        """
        Generate comprehensive medical context from search results

        Args:
            query: Original medical query
            results: Search results from clinician RAG

        Returns:
            Formatted medical context for clinician response
        """
        if not results:
            return f"No specific medical information found for: {query}"

        context_parts = []
        context_parts.append(f"MEDICAL INFORMATION FOR: {query.upper()}")
        context_parts.append("=" * 50)

        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            metadata = result.get('metadata', {})
            clinical_context = result.get('clinical_context', {})

            # Add result header
            context_parts.append(f"\n[{i}] MEDICAL EVIDENCE (Confidence: {result['score']:.2f})")
            if metadata.get('source'):
                context_parts.append(f"Source: {metadata['source']}")

            # Add evidence level and clinical relevance
            evidence_level = clinical_context.get('evidence_level', 'unknown')
            relevance = clinical_context.get('clinical_relevance', 'unknown')
            context_parts.append(f"Evidence Level: {evidence_level} | Clinical Relevance: {relevance}")

            # Add specialties
            specialties = clinical_context.get('specialty', [])
            if specialties:
                context_parts.append(f"Relevant Specialties: {', '.join(specialties)}")

            # Add the actual content
            context_parts.append(f"Content: {chunk[:500]}{'...' if len(chunk) > 500 else ''}")

        context_parts.append("\n" + "=" * 50)
        context_parts.append("IMPORTANT: This information is for educational purposes. Always consult current clinical guidelines and consider individual patient factors.")

        return "\n".join(context_parts)

    def update_medical_data(self, force_update: bool = False) -> bool:
        """
        Update medical data and rebuild embeddings

        Args:
            force_update: Force update even if recently updated

        Returns:
            True if successful
        """
        print("🔄 Updating medical data...")

        try:
            ingester = MedicalDataIngester()
            success = ingester.run_full_ingestion(force_update=force_update)

            if success:
                # Reload medical components
                self._load_medical_components()
                print("✅ Medical data updated successfully")
            else:
                print("❌ Medical data update failed")

            return success

        except Exception as e:
            print(f"❌ Error updating medical data: {e}")
            return False

def create_clinician_rag(general_rag: Optional[AuraRAG] = None) -> ClinicianRAG:
    """Factory function to create clinician RAG instance"""
    return ClinicianRAG(general_rag=general_rag)

# Global clinician RAG instance
clinician_rag_instance = None

def get_clinician_rag() -> ClinicianRAG:
    """Get or create global clinician RAG instance"""
    global clinician_rag_instance
    if clinician_rag_instance is None:
        clinician_rag_instance = ClinicianRAG()
    return clinician_rag_instance

def search_clinician_info(query: str, k: int = 5) -> str:
    """
    Search clinician information and return augmented prompt

    Args:
        query: Medical/clinical query
        k: Number of relevant chunks to retrieve

    Returns:
        Augmented prompt with medical context
    """
    clinician_rag = get_clinician_rag()
    results = clinician_rag.search_medical_info(query, k)

    if not results:
        return query

    # Generate medical context
    medical_context = clinician_rag.get_medical_context(query, results)

    # Create enhanced prompt for clinician mode
    augmented_prompt = f"""You are in CLINICIAN MODE. Provide evidence-based medical information using the following clinical data:

{medical_context}

PATIENT QUERY: {query}

IMPORTANT CLINICIAN INSTRUCTIONS:
1. Base your response on the provided medical evidence
2. Indicate evidence levels and clinical relevance
3. Mention relevant medical specialties
4. Note any limitations or uncertainties in the evidence
5. Always emphasize that this is not a substitute for professional medical judgment
6. If evidence is insufficient, clearly state this limitation

Provide a comprehensive, evidence-based clinical response:"""

    return augmented_prompt

if __name__ == "__main__":
    # Test clinician RAG
    clinician_rag = ClinicianRAG()

    # Test queries
    test_queries = [
        "What are the current guidelines for diabetes management?",
        "How do you treat myocardial infarction?",
        "What are the symptoms of pneumonia?",
        "Latest research on COVID-19 treatments"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        results = clinician_rag.search_medical_info(query, k=3)

        if results:
            context = clinician_rag.get_medical_context(query, results)
            print(f"✅ Found {len(results)} results")
            print(f"📝 Context length: {len(context)} characters")
        else:
            print("❌ No results found")
