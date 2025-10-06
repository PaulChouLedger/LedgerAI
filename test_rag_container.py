#!/usr/bin/env python3
"""
Test script for RAG container
"""

import requests
import time
import json

def test_rag_container():
    """Test RAG container endpoints"""
    base_url = "http://localhost:11435"
    
    print("🧪 Testing RAG Container...")
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test RAG stats
    try:
        print("\n2. Testing RAG stats...")
        response = requests.get(f"{base_url}/rag/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✅ RAG stats working")
            print(f"   Health score: {stats.get('health_score', 'N/A')}")
            print(f"   Chunks loaded: {stats.get('chunks_loaded', 'N/A')}")
            print(f"   Index loaded: {stats.get('index_loaded', 'N/A')}")
            print(f"   Encoder loaded: {stats.get('encoder_loaded', 'N/A')}")
        else:
            print(f"❌ RAG stats failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ RAG stats error: {e}")
    
    # Test RAG search
    try:
        print("\n3. Testing RAG search...")
        search_data = {
            "query": "What is the treatment for hypertension?",
            "top_k": 3
        }
        response = requests.post(f"{base_url}/rag/search", json=search_data, timeout=10)
        if response.status_code == 200:
            results = response.json()
            print("✅ RAG search working")
            print(f"   Query: {results.get('query', 'N/A')}")
            print(f"   Results count: {results.get('count', 'N/A')}")
            if results.get('results'):
                print(f"   First result: {results['results'][0][:100]}...")
        else:
            print(f"❌ RAG search failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ RAG search error: {e}")
    
    # Test text encoding
    try:
        print("\n4. Testing text encoding...")
        encode_data = {
            "text": "This is a test sentence for encoding."
        }
        response = requests.post(f"{base_url}/rag/encode", json=encode_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Text encoding working")
            print(f"   Text: {result.get('text', 'N/A')}")
            print(f"   Embedding dimension: {result.get('dimension', 'N/A')}")
        else:
            print(f"❌ Text encoding failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Text encoding error: {e}")
    
    print("\n🏁 RAG Container test completed!")

if __name__ == "__main__":
    test_rag_container()
