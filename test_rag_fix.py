#!/usr/bin/env python3
"""
Test script to verify RAG fixes work correctly
"""
import sys
import os

# Add the llm-container directory to Python path
sys.path.append('llm-container')

def test_rag_system():
    """Test the RAG system with CPU-only encoding"""
    print("🧪 Testing RAG system with CPU-only encoding...")
    
    try:
        # Import the RAG module
        from rag import smart_search_medical_info
        
        # Test queries - Dynamic RAG will decide based on:
        # 1. Query intent analysis (informational vs conversational)
        # 2. Document relevance (word overlap with actual content)
        # 3. Confidence scoring (how sure we are about the intent)
        test_queries = [
            "Who is Bob Corella?",  # Informational query - will check document relevance
            "Who is Liam Hugo?",    # Informational query - will check document relevance  
            "What is AuraVision?",  # Informational query - likely in documents
            "Tell me about LedgerAI",  # High-confidence informational request
            "What are chest pain symptoms?",  # Informational medical query
            "How to treat headaches?",  # Informational medical query
            "Hello Aura",  # Greeting - will skip RAG
            "What causes fever?",  # Informational medical query
            "How are you today?",  # Conversational - will skip RAG
            "Explain quantum computing",  # Informational but may not be in documents
            "Show me the latest research",  # Informational request
            "Thanks for helping"  # Conversational - will skip RAG
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            try:
                used_rag, result = smart_search_medical_info(query, k=3)
                
                if used_rag:
                    print(f"✅ RAG used successfully")
                    print(f"📄 Result length: {len(result)} characters")
                    print(f"📝 Sample: {result[:200]}...")
                else:
                    print(f"💬 Regular chat mode (no relevant results)")
                    print(f"📄 Original query: {result}")
                    
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
        
        print(f"\n🎉 RAG system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test RAG system: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
