#!/usr/bin/env python3
"""
Quick Benchmark: llama-cpp-python vs TensorRT-LLM
Simple performance comparison script
"""

import requests
import time
import json

def test_llama_cpp(prompt: str, max_tokens: int = 50):
    """Test llama-cpp-python"""
    try:
        start_time = time.time()
        response = requests.post("http://localhost:11434/completion", json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('content', '')
            tokens = len(response_text.split())
            return {
                "success": True,
                "time": end_time - start_time,
                "tokens": tokens,
                "tps": tokens / (end_time - start_time),
                "response": response_text[:100] + "..." if len(response_text) > 100 else response_text
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_tensorrt_llm(prompt: str, max_tokens: int = 50):
    """Test TensorRT-LLM"""
    try:
        start_time = time.time()
        response = requests.post("http://localhost:11435/generate", json={
            "prompt": prompt,
            "model_name": "default",
            "max_tokens": max_tokens
        }, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            tokens = len(response_text.split())
            return {
                "success": True,
                "time": end_time - start_time,
                "tokens": tokens,
                "tps": tokens / (end_time - start_time),
                "response": response_text[:100] + "..." if len(response_text) > 100 else response_text
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_services():
    """Check if services are running"""
    print("🔍 Checking services...")
    
    # Check llama-cpp-python
    try:
        response = requests.get("http://localhost:11434/health", timeout=5)
        llama_status = response.status_code == 200
        print(f"📊 llama-cpp-python: {'✅ Running' if llama_status else '❌ Not running'}")
    except:
        llama_status = False
        print("📊 llama-cpp-python: ❌ Not running")
    
    # Check TensorRT-LLM
    try:
        response = requests.get("http://localhost:11435/health", timeout=5)
        tensorrt_status = response.status_code == 200
        print(f"📊 TensorRT-LLM: {'✅ Running' if tensorrt_status else '❌ Not running'}")
    except:
        tensorrt_status = False
        print("📊 TensorRT-LLM: ❌ Not running")
    
    return llama_status, tensorrt_status

def run_quick_benchmark():
    """Run a quick benchmark"""
    print("🚀 Quick LLM Benchmark")
    print("=" * 50)
    
    # Check services
    llama_running, tensorrt_running = check_services()
    
    if not llama_running and not tensorrt_running:
        print("❌ No services running. Start containers first:")
        print("   docker compose up llm")
        print("   cd llm-container-tensorrt && ./build_and_run.sh")
        return
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain appendicitis symptoms.",
        "Write a haiku about coding."
    ]
    
    print(f"\n🧪 Testing with {len(prompts)} prompts...")
    
    # Test llama-cpp-python
    if llama_running:
        print("\n🧠 Testing llama-cpp-python...")
        llama_times = []
        llama_tps = []
        
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt[:30]}...")
            result = test_llama_cpp(prompt)
            if result["success"]:
                llama_times.append(result["time"])
                llama_tps.append(result["tps"])
                print(f"     ✅ {result['time']:.3f}s, {result['tps']:.1f} tokens/s")
            else:
                print(f"     ❌ Error: {result['error']}")
    
    # Test TensorRT-LLM
    if tensorrt_running:
        print("\n⚡ Testing TensorRT-LLM...")
        tensorrt_times = []
        tensorrt_tps = []
        
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt[:30]}...")
            result = test_tensorrt_llm(prompt)
            if result["success"]:
                tensorrt_times.append(result["time"])
                tensorrt_tps.append(result["tps"])
                print(f"     ✅ {result['time']:.3f}s, {result['tps']:.1f} tokens/s")
            else:
                print(f"     ❌ Error: {result['error']}")
    
    # Compare results
    if llama_running and tensorrt_running and llama_times and tensorrt_times:
        print(f"\n🏁 COMPARISON")
        print("-" * 30)
        
        llama_avg_time = sum(llama_times) / len(llama_times)
        tensorrt_avg_time = sum(tensorrt_times) / len(tensorrt_times)
        speedup = llama_avg_time / tensorrt_avg_time
        
        llama_avg_tps = sum(llama_tps) / len(llama_tps)
        tensorrt_avg_tps = sum(tensorrt_tps) / len(tensorrt_tps)
        
        print(f"⏱️  Average response time:")
        print(f"   llama-cpp-python: {llama_avg_time:.3f}s")
        print(f"   TensorRT-LLM:     {tensorrt_avg_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        print(f"\n⚡ Average tokens/second:")
        print(f"   llama-cpp-python: {llama_avg_tps:.1f}")
        print(f"   TensorRT-LLM:     {tensorrt_avg_tps:.1f}")
        print(f"   Improvement: {((tensorrt_avg_tps - llama_avg_tps) / llama_avg_tps * 100):+.1f}%")
        
        if speedup > 1:
            print(f"\n🏆 TensorRT-LLM is {speedup:.2f}x faster!")
        else:
            print(f"\n🏆 llama-cpp-python is {1/speedup:.2f}x faster!")

if __name__ == "__main__":
    run_quick_benchmark()

