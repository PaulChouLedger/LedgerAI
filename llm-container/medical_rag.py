#!/usr/bin/env python3
"""
Medical RAG - Specialized RAG for Medical Knowledge

Provides evidence-based medical information by searching:
1. Medical knowledge base (when available)
2. General RAG (fallback)
3. LLM general knowledge (last resort)

Designed to work with unified_medical_mode.py
"""

import os
import sys
import requests
from typing import List, Dict, Optional

# RAG service URL
RAG_SERVICE_URL = "http://localhost:11435"


class MedicalRAG:
    """
    Medical RAG system that searches for medical knowledge
    Falls back gracefully if medical database isn't available
    """
    
    def __init__(self):
        """Initialize Medical RAG"""
        self.rag_available = self._check_rag_availability()
        self.medical_embeddings_available = self._check_medical_embeddings()
        
        if self.rag_available:
            print("[Medical RAG] ✅ RAG service available (legacy data from data/parsed)")
        else:
            print("[Medical RAG] ⚠️ RAG service not available")
            
        if self.medical_embeddings_available:
            print("[Medical RAG] ✅ Medical embeddings available (used by adaptive engine for semantic matching)")
        else:
            print("[Medical RAG] ⚠️ Medical embeddings not available (using general RAG)")
    
    def _check_rag_availability(self) -> bool:
        """Check if RAG service is running"""
        try:
            response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _check_medical_embeddings(self) -> bool:
        """Check if medical embeddings exist"""
        # For now, we'll use the general embeddings
        # In the future, check for data/medical/embeddings/
        return os.path.exists("data/embeddings/index.faiss")
    
    def search_medical_knowledge(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for medical knowledge
        
        Args:
            query: Medical question
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        if not self.rag_available:
            print("[Medical RAG] ⚠️ RAG not available, returning empty results")
            return []
        
        try:
            # Search RAG (will use general embeddings for now)
            response = requests.post(
                f"{RAG_SERVICE_URL}/rag/search",
                json={"query": query, "k": k},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"[Medical RAG] 📚 Found {len(results)} results for: {query[:50]}...")
                return results
            else:
                print(f"[Medical RAG] ⚠️ Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[Medical RAG] ❌ Search error: {e}")
            return []
    
    def format_medical_context(self, query: str, results: List[Dict]) -> str:
        """
        Format search results into context for LLM
        
        Args:
            query: Original query
            results: Search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            text = result.get('text', result.get('chunk', ''))
            if text:
                context_parts.append(f"Source {i}:\n{text}")
        
        if context_parts:
            return "\n\n---\n\n".join(context_parts)
        
        return ""
    
    def build_medical_prompt(self, query: str, context: str = None) -> List[Dict]:
        """
        Build LLM messages with medical context
        
        Args:
            query: User's medical question
            context: Optional RAG context
            
        Returns:
            List of messages for LLM
        """
        if context:
            # Use RAG context
            system_prompt = f"""You are a knowledgeable medical assistant providing evidence-based information.

Based on the following medical information:

{context}

---

User question: {query}

Provide a helpful, accurate response based on the information above. Keep it concise and conversational (2-3 sentences). If this is a medical concern, remind them to consult a healthcare professional.

Remember: You provide information only, not medical advice."""
        else:
            # No RAG context - use general medical knowledge
            system_prompt = f"""You are a helpful medical assistant. The user asked: "{query}"

Provide a helpful response based on general medical knowledge. Keep it concise (2-3 sentences). If this appears to be a medical concern, gently suggest consulting a healthcare professional.

Remember: You are not a substitute for professional medical advice."""
        
        return [{"role": "system", "content": system_prompt}]
    
    def get_medical_response_messages(self, query: str) -> List[Dict]:
        """
        Get LLM messages for medical query with RAG augmentation
        
        This is the main function used by unified_medical_mode
        
        Args:
            query: User's medical question
            
        Returns:
            List of messages for LLM with RAG context (if available)
        """
        # Try to get RAG context
        results = self.search_medical_knowledge(query, k=3)
        
        if results:
            context = self.format_medical_context(query, results)
            print(f"[Medical RAG] ✅ Using RAG context ({len(results)} sources)")
            return self.build_medical_prompt(query, context)
        else:
            print(f"[Medical RAG] ⚠️ No RAG context, using general medical knowledge")
            return self.build_medical_prompt(query, None)


# Global instance
_medical_rag_instance = None


def get_medical_rag() -> MedicalRAG:
    """Get or create global Medical RAG instance"""
    global _medical_rag_instance
    if _medical_rag_instance is None:
        _medical_rag_instance = MedicalRAG()
    return _medical_rag_instance


def search_medical_info(query: str, k: int = 3) -> List[Dict]:
    """
    Convenience function to search medical knowledge
    
    Args:
        query: Medical question
        k: Number of results
        
    Returns:
        List of relevant documents
    """
    rag = get_medical_rag()
    return rag.search_medical_knowledge(query, k)


def get_medical_messages(query: str) -> List[Dict]:
    """
    Convenience function to get medical response messages with RAG
    
    Args:
        query: Medical question
        
    Returns:
        List of messages for LLM
    """
    rag = get_medical_rag()
    return rag.get_medical_response_messages(query)


if __name__ == "__main__":
    # Test medical RAG
    print("🩺 Testing Medical RAG")
    print("=" * 50)
    
    rag = MedicalRAG()
    
    test_queries = [
        "What is pancreatitis?",
        "What are the symptoms of diabetes?",
        "How do you treat hypertension?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = rag.search_medical_knowledge(query, k=2)
        print(f"✅ Found {len(results)} results")
        
        if results:
            context = rag.format_medical_context(query, results)
            print(f"📄 Context length: {len(context)} chars")
        
        messages = rag.get_medical_response_messages(query)
        print(f"💬 Generated {len(messages)} messages")
        print(f"📋 System prompt preview: {messages[0]['content'][:100]}...")

