#!/usr/bin/env python3
"""
Test script for TensorRT-LLM Container
Verifies API endpoints and basic functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:11435"

def test_health():
    """Test health endpoint"""
    print("[Test] 🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"[Test] ✅ Health check passed: {data}")
            return True
        else:
            print(f"[Test] ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Test] ❌ Health check error: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("[Test] 📋 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"[Test] ✅ Models endpoint: {data}")
            return True
        else:
            print(f"[Test] ❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Test] ❌ Models endpoint error: {e}")
        return False

def test_generate():
    """Test text generation"""
    print("[Test] 🎯 Testing text generation...")
    try:
        payload = {
            "prompt": "What is the capital of France?",
            "model_name": "default",
            "max_tokens": 50
        }
        
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"[Test] ✅ Generation successful: {data['response']}")
            return True
        else:
            print(f"[Test] ❌ Generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Test] ❌ Generation error: {e}")
        return False

def main():
    """Run all tests"""
    print("[Test] 🚀 Starting TensorRT-LLM Container Tests")
    print(f"[Test] 📍 Testing against: {BASE_URL}")
    
    # Wait for container to be ready
    print("[Test] ⏳ Waiting for container to be ready...")
    time.sleep(5)
    
    tests = [
        ("Health Check", test_health),
        ("Models List", test_models),
        ("Text Generation", test_generate)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[Test] 🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"[Test] ✅ {test_name} PASSED")
        else:
            print(f"[Test] ❌ {test_name} FAILED")
    
    print(f"\n[Test] 📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[Test] 🎉 All tests passed! TensorRT-LLM container is working correctly.")
    else:
        print("[Test] ⚠️  Some tests failed. Check container logs for details.")

if __name__ == "__main__":
    main()
