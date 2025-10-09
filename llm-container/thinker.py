#!/usr/bin/env python3
"""
Aura THINKER Mode - Knowledge Queries with RAG

Handles:
- Information queries: "What is X?", "Who is Y?"
- Explanations: "Explain how the brain works"
- Details: "Tell me everything about diabetes"
- Knowledge requests: "What do you know about...?"

Characteristics:
- Searches RAG for relevant documents
- Provides comprehensive, detailed answers
- Insightful and thorough
- Educational tone
"""

import requests
import re
from typing import List, Dict, Any, Optional

def is_thinker_trigger(prompt: str) -> bool:
    """
    Check if prompt should trigger THINKER mode (knowledge query)
    
    Args:
        prompt: Normalized prompt (lowercase)
        
    Returns:
        True if knowledge/information query
    """
    prompt_lower = prompt.lower()
    
    # Knowledge query indicators
    knowledge_patterns = [
        "who is", "who was", "who are",
        "what is", "what was", "what are",
        "tell me about", "tell me everything", "tell me all",
        "information about", "all information", "everything about",
        "details about", "explain", "describe",
        "what do you know", "everything you know",
        "how does", "how do", "why does", "why do"
    ]
    
    return any(pattern in prompt_lower for pattern in knowledge_patterns)


def handle_thinker(prompt: str, llm_chat_fn, session_id: str = None):
    """
    Handle knowledge query with RAG search
    
    Args:
        prompt: User's knowledge query
        llm_chat_fn: Function to call LLM for response generation
        session_id: Optional session identifier
        
    Yields:
        Streamed response chunks
    """
    print(f"[THINKER] 🧠 Handling knowledge query: '{prompt[:60]}...'")
    
    # Search RAG for relevant documents
    rag_results = search_rag(prompt)
    
    if rag_results and len(rag_results) > 0:
        print(f"[THINKER] 📚 Found {len(rag_results)} relevant documents")
        
        # Build augmented prompt with RAG context
        context_parts = []
        for result in rag_results[:3]:  # Use top 3 results
            text = result.get('text', result.get('chunk', ''))
            if text:
                context_parts.append(text)
        
        if context_parts:
            context = "\n\n---\n\n".join(context_parts)
            
            augmented_prompt = f"""Based on the following information:

{context}

---

Please provide a comprehensive answer to: {prompt}

Be thorough, insightful, and educational. Synthesize the information provided."""
            
            # Use LLM to generate response
            system_msg = "I am AuraVision, an insightful AI assistant. Provide comprehensive, detailed answers based on the information provided. Be thorough and informative."
            
        else:
            augmented_prompt = prompt
            system_msg = "I am AuraVision, an insightful AI assistant. Provide thoughtful, detailed responses."
    
    else:
        print(f"[THINKER] ⚠️ No RAG results found, using general knowledge")
        augmented_prompt = prompt
        system_msg = "I am AuraVision, an insightful AI assistant. Provide thoughtful, detailed responses even without specific documents."
    
    # Generate response using LLM
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": augmented_prompt}
    ]
    
    # Stream response
    try:
        for chunk in llm_chat_fn(msgs, stream=True):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                yield token
                
    except Exception as e:
        print(f"[THINKER] ❌ Error generating response: {e}")
        yield "I apologize, but I'm having trouble accessing that information right now."


def search_rag(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search RAG system for relevant documents
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        List of relevant document chunks
    """
    try:
        response = requests.post(
            "http://localhost:11435/rag/search",
            json={"query": query, "k": k},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
    
    except Exception as e:
        print(f"[THINKER] ⚠️ RAG search failed: {e}")
    
    return []

