#!/usr/bin/env python3
"""
Test script to verify TensorRT-LLM setup
"""

import requests
import json
import time

def test_llm_endpoint(url, test_name):
    """Test LLM endpoint"""
    print(f"\n🧪 Testing {test_name} at {url}")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    try:
        # Test chat endpoint
        test_data = {
            "prompt": "Hello, how are you?",
            "session_id": "test_session"
        }
        
        start_time = time.time()
        response = requests.post(f"{url}/chat", 
                               json=test_data, 
                               timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Chat test: {response.status_code}")
            print(f"⏱️ Response time: {end_time - start_time:.2f}s")
            print(f"📝 Response: {response.text[:100]}...")
            return True
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def main():
    print("🚀 Testing LLM Endpoints")
    print("=" * 50)
    
    # Test both endpoints
    llama_cpp_success = test_llm_endpoint("http://127.0.0.1:11434", "llama-cpp")
    tensorrt_success = test_llm_endpoint("http://127.0.0.1:11435", "TensorRT-LLM")
    
    print("\n📊 Results Summary:")
    print("=" * 50)
    print(f"llama-cpp (port 11434): {'✅ Working' if llama_cpp_success else '❌ Failed'}")
    print(f"TensorRT-LLM (port 11435): {'✅ Working' if tensorrt_success else '❌ Failed'}")
    
    if llama_cpp_success and tensorrt_success:
        print("\n🎉 Both endpoints are working! You can now test performance comparison.")
    elif llama_cpp_success:
        print("\n⚠️ Only llama-cpp is working. TensorRT-LLM may need model conversion.")
    else:
        print("\n❌ Neither endpoint is working. Check your Docker containers.")

if __name__ == "__main__":
    main()
