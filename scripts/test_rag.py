#!/usr/bin/env python3
"""
Test script for Aura RAG system
Tests FAISS-GPU functionality and retrieval performance
"""

import sys
import os
import time
import requests
import json

# Add aura-control to path
sys.path.append('/Users/rcabello/Documents/GitHub/LedgerAI/aura-control')

def test_rag_module():
    """Test RAG module directly"""
    print("🧪 Testing RAG module directly...")
    
    try:
        from rag import test_rag
        success = test_rag()
        if success:
            print("✅ RAG module test passed")
            return True
        else:
            print("❌ RAG module test failed")
            return False
    except Exception as e:
        print(f"❌ RAG module test error: {e}")
        return False

def test_rag_api():
    """Test RAG API endpoints"""
    print("🌐 Testing RAG API endpoints...")
    
    base_url = "http://localhost:11434"
    
    # Test RAG stats
    try:
        response = requests.get(f"{base_url}/rag/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ RAG stats: {stats}")
        else:
            print(f"❌ RAG stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ RAG stats error: {e}")
        return False
    
    # Test RAG search
    try:
        test_queries = [
            "chest pain symptoms",
            "dizziness causes",
            "heart attack signs"
        ]
        
        for query in test_queries:
            print(f"🔍 Testing query: '{query}'")
            response = requests.post(
                f"{base_url}/rag/search",
                json={"query": query, "k": 2},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query successful: {len(result['augmented_prompt'])} chars")
                print(f"   Augmented prompt preview: {result['augmented_prompt'][:200]}...")
            else:
                print(f"❌ Query failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ RAG search error: {e}")
        return False
    
    return True

def test_performance():
    """Test RAG performance"""
    print("⚡ Testing RAG performance...")
    
    base_url = "http://localhost:11434"
    test_query = "chest pain emergency symptoms"
    
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/rag/search",
                json={"query": test_query, "k": 3},
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                elapsed = end_time - start_time
                times.append(elapsed)
                print(f"   Test {i+1}: {elapsed:.3f}s")
            else:
                print(f"   Test {i+1}: Failed ({response.status_code})")
                return False
                
        except Exception as e:
            print(f"   Test {i+1}: Error - {e}")
            return False
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"📊 Performance: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
        
        if avg_time < 0.5:  # Less than 500ms
            print("✅ Performance: Excellent (<500ms)")
        elif avg_time < 1.0:  # Less than 1s
            print("✅ Performance: Good (<1s)")
        else:
            print("⚠️ Performance: Slow (>1s)")
    
    return True

def main():
    """Run all RAG tests"""
    print("🚀 Aura RAG System Test Suite")
    print("=" * 50)
    
    # Test 1: RAG module
    print("\n1️⃣ Testing RAG Module...")
    module_ok = test_rag_module()
    
    # Test 2: RAG API (requires running container)
    print("\n2️⃣ Testing RAG API...")
    print("   (Make sure LLM container is running: docker-compose up llm-container)")
    api_ok = test_rag_api()
    
    # Test 3: Performance
    if api_ok:
        print("\n3️⃣ Testing Performance...")
        perf_ok = test_performance()
    else:
        perf_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   RAG Module: {'✅ PASS' if module_ok else '❌ FAIL'}")
    print(f"   RAG API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"   Performance: {'✅ PASS' if perf_ok else '❌ FAIL'}")
    
    if module_ok and api_ok and perf_ok:
        print("\n🎉 All tests passed! RAG system is ready.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
