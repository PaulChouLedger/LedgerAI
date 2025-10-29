#!/usr/bin/env python3
"""
Benchmark Script: llama-cpp-python vs TensorRT-LLM
Compares performance between the two LLM implementations
"""

import requests
import time
import json
import statistics
import argparse
from typing import List, Dict, Any
import concurrent.futures
import threading

class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.response_times = []
        self.tokens_per_second = []
        self.total_tokens = 0
        self.total_time = 0
        self.errors = 0
        self.responses = []

class LLMBenchmark:
    def __init__(self, llama_url: str = "http://localhost:11434", tensorrt_url: str = "http://localhost:11435"):
        self.llama_url = llama_url
        self.tensorrt_url = tensorrt_url
        self.test_prompts = [
            "What is the capital of France?",
            "Explain the symptoms of appendicitis in medical terms.",
            "Write a short story about a robot learning to paint.",
            "What are the side effects of common blood pressure medications?",
            "Describe the process of photosynthesis in simple terms.",
            "What is the difference between Type 1 and Type 2 diabetes?",
            "Explain quantum computing to a 10-year-old.",
            "What are the main causes of climate change?",
            "Describe the symptoms of a heart attack.",
            "What is machine learning and how does it work?"
        ]
    
    def test_llama_cpp(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Test llama-cpp-python endpoint"""
        try:
            start_time = time.time()
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(f"{self.llama_url}/completion", json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time
                response_text = data.get('content', '')
                tokens = len(response_text.split())
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "tokens": tokens,
                    "tokens_per_second": tokens / response_time if response_time > 0 else 0,
                    "response": response_text,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "tokens": 0,
                    "tokens_per_second": 0,
                    "response": "",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "response_time": 0,
                "tokens": 0,
                "tokens_per_second": 0,
                "response": "",
                "error": str(e)
            }
    
    def test_tensorrt_llm(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Test TensorRT-LLM endpoint"""
        try:
            start_time = time.time()
            
            payload = {
                "prompt": prompt,
                "model_name": "default",
                "max_tokens": max_tokens
            }
            
            response = requests.post(f"{self.tensorrt_url}/generate", json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time
                response_text = data.get('response', '')
                tokens = len(response_text.split())
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "tokens": tokens,
                    "tokens_per_second": tokens / response_time if response_time > 0 else 0,
                    "response": response_text,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "tokens": 0,
                    "tokens_per_second": 0,
                    "response": "",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "response_time": 0,
                "tokens": 0,
                "tokens_per_second": 0,
                "error": str(e)
            }
    
    def check_services(self) -> Dict[str, bool]:
        """Check if both services are running"""
        services = {}
        
        # Check llama-cpp-python
        try:
            response = requests.get(f"{self.llama_url}/health", timeout=5)
            services['llama_cpp'] = response.status_code == 200
        except:
            services['llama_cpp'] = False
        
        # Check TensorRT-LLM
        try:
            response = requests.get(f"{self.tensorrt_url}/health", timeout=5)
            services['tensorrt_llm'] = response.status_code == 200
        except:
            services['tensorrt_llm'] = False
        
        return services
    
    def run_benchmark(self, num_iterations: int = 10, concurrent_requests: int = 1) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark"""
        print(f"[Benchmark] 🚀 Starting benchmark with {num_iterations} iterations")
        print(f"[Benchmark] 🔄 Concurrent requests: {concurrent_requests}")
        
        # Check services
        services = self.check_services()
        print(f"[Benchmark] 📊 Service status: {services}")
        
        if not services['llama_cpp']:
            print("[Benchmark] ❌ llama-cpp-python service not available")
        if not services['tensorrt_llm']:
            print("[Benchmark] ❌ TensorRT-LLM service not available")
        
        results = {
            'llama_cpp': BenchmarkResult('llama-cpp-python'),
            'tensorrt_llm': BenchmarkResult('TensorRT-LLM')
        }
        
        # Run benchmarks
        if services['llama_cpp']:
            print("[Benchmark] 🧠 Testing llama-cpp-python...")
            self._run_single_benchmark('llama_cpp', results['llama_cpp'], num_iterations, concurrent_requests)
        
        if services['tensorrt_llm']:
            print("[Benchmark] ⚡ Testing TensorRT-LLM...")
            self._run_single_benchmark('tensorrt_llm', results['tensorrt_llm'], num_iterations, concurrent_requests)
        
        return results
    
    def _run_single_benchmark(self, service_type: str, result: BenchmarkResult, num_iterations: int, concurrent_requests: int):
        """Run benchmark for a single service"""
        test_func = self.test_llama_cpp if service_type == 'llama_cpp' else self.test_tensorrt_llm
        
        def run_single_test(prompt: str):
            return test_func(prompt, max_tokens=100)
        
        # Run tests
        for i in range(num_iterations):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            if concurrent_requests > 1:
                # Run concurrent requests
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    futures = [executor.submit(run_single_test, prompt) for _ in range(concurrent_requests)]
                    for future in concurrent.futures.as_completed(futures):
                        test_result = future.result()
                        self._process_test_result(test_result, result)
            else:
                # Run single request
                test_result = run_single_test(prompt)
                self._process_test_result(test_result, result)
            
            print(f"[Benchmark] 📈 {service_type}: {i+1}/{num_iterations} completed")
    
    def _process_test_result(self, test_result: Dict[str, Any], result: BenchmarkResult):
        """Process individual test result"""
        if test_result['success']:
            result.response_times.append(test_result['response_time'])
            result.tokens_per_second.append(test_result['tokens_per_second'])
            result.total_tokens += test_result['tokens']
            result.total_time += test_result['response_time']
            result.responses.append(test_result['response'])
        else:
            result.errors += 1
            print(f"[Benchmark] ❌ Error in {result.name}: {test_result['error']}")
    
    def print_results(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark results"""
        print("\n" + "="*80)
        print("🏆 BENCHMARK RESULTS")
        print("="*80)
        
        for name, result in results.items():
            if not result.response_times:
                print(f"\n❌ {result.name}: No successful tests")
                continue
            
            print(f"\n📊 {result.name.upper()}")
            print("-" * 40)
            print(f"✅ Successful tests: {len(result.response_times)}")
            print(f"❌ Errors: {result.errors}")
            print(f"⏱️  Average response time: {statistics.mean(result.response_times):.3f}s")
            print(f"⚡ Average tokens/second: {statistics.mean(result.tokens_per_second):.2f}")
            print(f"📈 Total tokens: {result.total_tokens}")
            print(f"🕐 Total time: {result.total_time:.3f}s")
            print(f"📊 Min response time: {min(result.response_times):.3f}s")
            print(f"📊 Max response time: {max(result.response_times):.3f}s")
            print(f"📊 Median response time: {statistics.median(result.response_times):.3f}s")
        
        # Compare results
        if len(results) == 2:
            llama_result = results.get('llama_cpp')
            tensorrt_result = results.get('tensorrt_llm')
            
            if llama_result and tensorrt_result and llama_result.response_times and tensorrt_result.response_times:
                print(f"\n🏁 COMPARISON")
                print("-" * 40)
                
                llama_avg_time = statistics.mean(llama_result.response_times)
                tensorrt_avg_time = statistics.mean(tensorrt_result.response_times)
                speedup = llama_avg_time / tensorrt_avg_time if tensorrt_avg_time > 0 else 0
                
                llama_tps = statistics.mean(llama_result.tokens_per_second)
                tensorrt_tps = statistics.mean(tensorrt_result.tokens_per_second)
                tps_improvement = (tensorrt_tps - llama_tps) / llama_tps * 100 if llama_tps > 0 else 0
                
                print(f"⚡ Speed improvement: {speedup:.2f}x")
                print(f"📈 Tokens/second improvement: {tps_improvement:+.1f}%")
                
                if speedup > 1:
                    print(f"🏆 TensorRT-LLM is {speedup:.2f}x faster")
                else:
                    print(f"🏆 llama-cpp-python is {1/speedup:.2f}x faster")

def main():
    parser = argparse.ArgumentParser(description='Benchmark llama-cpp-python vs TensorRT-LLM')
    parser.add_argument('--iterations', type=int, default=10, help='Number of test iterations')
    parser.add_argument('--concurrent', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--llama-url', default='http://localhost:11434', help='llama-cpp-python URL')
    parser.add_argument('--tensorrt-url', default='http://localhost:11435', help='TensorRT-LLM URL')
    
    args = parser.parse_args()
    
    benchmark = LLMBenchmark(args.llama_url, args.tensorrt_url)
    results = benchmark.run_benchmark(args.iterations, args.concurrent)
    benchmark.print_results(results)

if __name__ == "__main__":
    main()
