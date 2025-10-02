#!/usr/bin/env python3
"""
Whisper Model Benchmarking Script

Compares latency between:
1. transcription_tuner.py (faster-whisper with distill.small)
2. whisper-container (TensorRT optimized base.en)

Tests with various audio samples and measures:
- Model loading time
- Transcription latency
- Memory usage
- Accuracy (if ground truth available)
"""

import os
import sys
import time
import json
import requests
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from typing import Dict, List, Tuple
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class WhisperBenchmark:
    def __init__(self):
        self.results = {
            'faster_whisper': {},
            'whisper_container': {},
            'comparison': {}
        }
        
        # Test audio samples
        self.test_samples = [
            "assets/voice_samples/audio1.wav",
            "assets/voice_samples/startup.wav", 
            "assets/voice_samples/welcome.wav",
            "assets/prompts/audio2.wav",
            "assets/prompts/audio3.wav"
        ]
        
        # Create test audio if samples don't exist
        self._create_test_audio()
        
    def _create_test_audio(self):
        """Create test audio samples if they don't exist"""
        import scipy.signal
        
        # Create test directory
        os.makedirs("assets/voice_samples", exist_ok=True)
        os.makedirs("assets/prompts", exist_ok=True)
        
        # Generate test audio samples
        sample_rate = 16000
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]  # Different lengths
        texts = [
            "Hello, this is a test of the transcription system.",
            "The quick brown fox jumps over the lazy dog.",
            "Medical transcription testing with complex terminology.",
            "Testing audio quality and latency performance.",
            "This is a longer sample to test transcription accuracy."
        ]
        
        for i, (duration, text) in enumerate(zip(durations, texts)):
            # Generate sine wave with speech-like characteristics
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Mix multiple frequencies to simulate speech
            audio = (np.sin(2 * np.pi * 200 * t) * 0.3 + 
                    np.sin(2 * np.pi * 400 * t) * 0.2 +
                    np.sin(2 * np.pi * 800 * t) * 0.1)
            
            # Add some noise
            audio += np.random.normal(0, 0.05, len(audio))
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save test file
            filename = f"assets/voice_samples/test_{i+1}.wav"
            sf.write(filename, audio, sample_rate)
            print(f"[Benchmark] ✅ Created test audio: {filename}")
    
    def benchmark_faster_whisper(self) -> Dict:
        """Benchmark faster-whisper with distill.small model"""
        print("\n🔬 Benchmarking faster-whisper (distill.small)...")
        
        results = {
            'model_loading_time': 0,
            'transcription_times': [],
            'memory_usage': [],
            'total_transcriptions': 0,
            'average_latency': 0,
            'errors': []
        }
        
        try:
            # Measure model loading time
            start_time = time.time()
            model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
            results['model_loading_time'] = time.time() - start_time
            print(f"[Benchmark] ⏱️ Model loading time: {results['model_loading_time']:.2f}s")
            
            # Test with each audio sample
            for i, sample_path in enumerate(self.test_samples):
                if not os.path.exists(sample_path):
                    print(f"[Benchmark] ⚠️ Sample not found: {sample_path}")
                    continue
                    
                print(f"[Benchmark] 🎵 Testing sample {i+1}: {os.path.basename(sample_path)}")
                
                # Load audio
                audio, sr = sf.read(sample_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                
                # Measure transcription time
                start_time = time.time()
                segments, _ = model.transcribe(audio, language="en", beam_size=5)
                transcription_time = time.time() - start_time
                
                # Get transcribed text
                text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                
                results['transcription_times'].append(transcription_time)
                results['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                results['total_transcriptions'] += 1
                
                print(f"[Benchmark] ⏱️ Transcription time: {transcription_time:.2f}s")
                print(f"[Benchmark] 📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
        except Exception as e:
            error_msg = f"faster-whisper error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"[Benchmark] ❌ {error_msg}")
        
        # Calculate averages
        if results['transcription_times']:
            results['average_latency'] = np.mean(results['transcription_times'])
            results['min_latency'] = np.min(results['transcription_times'])
            results['max_latency'] = np.max(results['transcription_times'])
            results['std_latency'] = np.std(results['transcription_times'])
        
        return results
    
    def benchmark_whisper_container(self, container_url="http://localhost:5000") -> Dict:
        """Benchmark whisper-container with TensorRT optimization"""
        print("\n🔬 Benchmarking whisper-container (TensorRT base.en)...")
        
        results = {
            'model_loading_time': 0,  # Not measurable for container
            'transcription_times': [],
            'memory_usage': [],
            'total_transcriptions': 0,
            'average_latency': 0,
            'errors': [],
            'container_available': False
        }
        
        try:
            # Test if container is running
            response = requests.get(f"{container_url}/health", timeout=5)
            if response.status_code == 200:
                results['container_available'] = True
                print("[Benchmark] ✅ Whisper container is running")
            else:
                print("[Benchmark] ⚠️ Whisper container health check failed")
                
        except requests.exceptions.RequestException as e:
            print(f"[Benchmark] ❌ Whisper container not available: {e}")
            results['errors'].append(f"Container not available: {str(e)}")
            return results
        
        # Test with each audio sample
        for i, sample_path in enumerate(self.test_samples):
            if not os.path.exists(sample_path):
                print(f"[Benchmark] ⚠️ Sample not found: {sample_path}")
                continue
                
            print(f"[Benchmark] 🎵 Testing sample {i+1}: {os.path.basename(sample_path)}")
            
            try:
                # Prepare audio file for upload
                with open(sample_path, 'rb') as f:
                    files = {'audio': f}
                    
                    # Measure transcription time
                    start_time = time.time()
                    response = requests.post(f"{container_url}/transcribe", files=files, timeout=30)
                    transcription_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('text', '')
                        
                        results['transcription_times'].append(transcription_time)
                        results['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                        results['total_transcriptions'] += 1
                        
                        print(f"[Benchmark] ⏱️ Transcription time: {transcription_time:.2f}s")
                        print(f"[Benchmark] 📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    else:
                        error_msg = f"Container error: {response.status_code} - {response.text}"
                        results['errors'].append(error_msg)
                        print(f"[Benchmark] ❌ {error_msg}")
                        
            except Exception as e:
                error_msg = f"Container transcription error: {str(e)}"
                results['errors'].append(error_msg)
                print(f"[Benchmark] ❌ {error_msg}")
        
        # Calculate averages
        if results['transcription_times']:
            results['average_latency'] = np.mean(results['transcription_times'])
            results['min_latency'] = np.min(results['transcription_times'])
            results['max_latency'] = np.max(results['transcription_times'])
            results['std_latency'] = np.std(results['transcription_times'])
        
        return results
    
    def run_concurrent_benchmark(self, num_threads=3) -> Dict:
        """Run concurrent transcription tests to simulate real-world usage"""
        print(f"\n🚀 Running concurrent benchmark with {num_threads} threads...")
        
        def transcribe_sample(sample_path, system_type):
            """Transcribe a single sample"""
            start_time = time.time()
            
            if system_type == "faster_whisper":
                # Use faster-whisper
                model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
                audio, sr = sf.read(sample_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                segments, _ = model.transcribe(audio, language="en", beam_size=5)
                text = " ".join([s.text.strip() for s in segments if s.text.strip()])
            else:
                # Use whisper container
                with open(sample_path, 'rb') as f:
                    files = {'audio': f}
                    response = requests.post("http://localhost:5000/transcribe", files=files, timeout=30)
                    if response.status_code == 200:
                        text = response.json().get('text', '')
                    else:
                        text = ""
            
            return {
                'system': system_type,
                'sample': os.path.basename(sample_path),
                'latency': time.time() - start_time,
                'text_length': len(text)
            }
        
        # Run concurrent tests
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            # Submit faster-whisper tasks
            for sample in self.test_samples[:2]:  # Use first 2 samples
                futures.append(executor.submit(transcribe_sample, sample, "faster_whisper"))
            
            # Submit whisper-container tasks
            for sample in self.test_samples[:2]:
                futures.append(executor.submit(transcribe_sample, sample, "whisper_container"))
            
            # Collect results
            concurrent_results = []
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    concurrent_results.append(result)
                    print(f"[Benchmark] 🧵 {result['system']}: {result['latency']:.2f}s ({result['sample']})")
                except Exception as e:
                    print(f"[Benchmark] ❌ Concurrent test failed: {e}")
        
        return {
            'concurrent_results': concurrent_results,
            'total_tests': len(concurrent_results)
        }
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n📊 Generating Benchmark Report...")
        
        # Run individual benchmarks
        faster_whisper_results = self.benchmark_faster_whisper()
        whisper_container_results = self.benchmark_whisper_container()
        
        # Run concurrent benchmark
        concurrent_results = self.run_concurrent_benchmark()
        
        # Compile results
        self.results['faster_whisper'] = faster_whisper_results
        self.results['whisper_container'] = whisper_container_results
        self.results['concurrent'] = concurrent_results
        
        # Generate comparison
        if (faster_whisper_results['average_latency'] > 0 and 
            whisper_container_results['average_latency'] > 0):
            
            faster_avg = faster_whisper_results['average_latency']
            container_avg = whisper_container_results['average_latency']
            
            if faster_avg < container_avg:
                winner = "faster-whisper"
                improvement = ((container_avg - faster_avg) / container_avg) * 100
            else:
                winner = "whisper-container"
                improvement = ((faster_avg - container_avg) / faster_avg) * 100
            
            self.results['comparison'] = {
                'winner': winner,
                'improvement_percent': improvement,
                'faster_whisper_avg': faster_avg,
                'whisper_container_avg': container_avg,
                'speed_ratio': max(faster_avg, container_avg) / min(faster_avg, container_avg)
            }
        
        # Save results
        with open('whisper_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🎯 WHISPER BENCHMARK SUMMARY")
        print("="*60)
        
        # Faster-whisper results
        fw = self.results['faster_whisper']
        print(f"\n🔬 faster-whisper (distill.small):")
        print(f"  ⏱️  Model loading: {fw['model_loading_time']:.2f}s")
        print(f"  ⏱️  Average latency: {fw['average_latency']:.2f}s")
        print(f"  📊 Transcriptions: {fw['total_transcriptions']}")
        if fw['errors']:
            print(f"  ❌ Errors: {len(fw['errors'])}")
        
        # Whisper-container results
        wc = self.results['whisper_container']
        print(f"\n🔬 whisper-container (TensorRT base.en):")
        print(f"  ⏱️  Average latency: {wc['average_latency']:.2f}s")
        print(f"  📊 Transcriptions: {wc['total_transcriptions']}")
        print(f"  🐳 Container available: {wc['container_available']}")
        if wc['errors']:
            print(f"  ❌ Errors: {len(wc['errors'])}")
        
        # Comparison
        if self.results['comparison']:
            comp = self.results['comparison']
            print(f"\n🏆 WINNER: {comp['winner']}")
            print(f"  📈 Performance improvement: {comp['improvement_percent']:.1f}%")
            print(f"  ⚡ Speed ratio: {comp['speed_ratio']:.2f}x")
            print(f"  🔬 faster-whisper avg: {comp['faster_whisper_avg']:.2f}s")
            print(f"  🐳 whisper-container avg: {comp['whisper_container_avg']:.2f}s")
        
        print(f"\n📄 Full results saved to: whisper_benchmark_results.json")
        print("="*60)

def main():
    """Main benchmark execution"""
    print("🚀 Starting Whisper Model Benchmarking...")
    print("This will compare faster-whisper (distill.small) vs whisper-container (TensorRT base.en)")
    
    # Check if whisper container is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        print("✅ Whisper container detected")
    except:
        print("⚠️ Whisper container not running - will test faster-whisper only")
        print("   Start container with: docker compose up whisper")
    
    # Run benchmark
    benchmark = WhisperBenchmark()
    results = benchmark.generate_report()
    
    print("\n🎉 Benchmark complete!")
    return results

if __name__ == "__main__":
    main()
