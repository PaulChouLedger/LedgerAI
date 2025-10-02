#!/usr/bin/env python3
"""
Quick Whisper Benchmark - Tests faster-whisper performance

This script focuses on benchmarking faster-whisper (distill.small) 
and can be extended to compare with whisper-container when available.
"""

import os
import time
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
import psutil
import json
from typing import Dict, List

class QuickWhisperBenchmark:
    def __init__(self):
        self.results = {}
        
    def create_test_audio(self, duration=3.0, sample_rate=16000):
        """Create realistic test audio with speech-like characteristics"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more realistic speech-like audio
        # Mix multiple formants (speech frequency bands)
        audio = (np.sin(2 * np.pi * 200 * t) * 0.4 +  # F0 (fundamental)
                np.sin(2 * np.pi * 400 * t) * 0.3 +   # F1 (first formant)
                np.sin(2 * np.pi * 800 * t) * 0.2 +   # F2 (second formant)
                np.sin(2 * np.pi * 1200 * t) * 0.1 +  # F3 (third formant)
                np.sin(2 * np.pi * 2000 * t) * 0.05)  # F4 (fourth formant)
        
        # Add speech-like modulation (vibrato, tremolo)
        vibrato = 1 + 0.1 * np.sin(2 * np.pi * 6 * t)  # 6Hz vibrato
        tremolo = 1 + 0.2 * np.sin(2 * np.pi * 3 * t)  # 3Hz tremolo
        audio *= vibrato * tremolo
        
        # Add realistic noise floor
        noise = np.random.normal(0, 0.05, len(audio))
        audio += noise
        
        # Add speech pauses (silence periods)
        pause_start = int(len(audio) * 0.3)
        pause_end = int(len(audio) * 0.4)
        audio[pause_start:pause_end] *= 0.1  # 10% silence
        
        # Normalize to speech levels
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio.astype(np.float32)
    
    def benchmark_faster_whisper(self, test_durations=[1.0, 2.0, 3.0, 5.0, 10.0]):
        """Benchmark faster-whisper with distill.small"""
        print("🔬 Benchmarking faster-whisper (distill.small)...")
        
        results = {
            'model_loading_time': 0,
            'transcription_times': [],
            'memory_usage': [],
            'test_durations': test_durations,
            'average_latency': 0,
            'min_latency': 0,
            'max_latency': 0,
            'std_latency': 0
        }
        
        try:
            # Measure model loading time
            print("⏳ Loading distill.small model...")
            start_time = time.time()
            model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
            results['model_loading_time'] = time.time() - start_time
            print(f"✅ Model loaded in {results['model_loading_time']:.2f}s")
            
            # Test with different audio durations
            for duration in test_durations:
                print(f"\n🎵 Testing {duration}s audio...")
                
                # Create test audio
                audio = self.create_test_audio(duration)
                
                # Measure transcription time
                start_time = time.time()
                segments, _ = model.transcribe(audio, language="en", beam_size=5)
                transcription_time = time.time() - start_time
                
                # Get transcribed text
                text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                
                results['transcription_times'].append(transcription_time)
                results['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                
                print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                print(f"📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"🧠 Memory usage: {results['memory_usage'][-1]:.1f} MB")
                
                # Calculate efficiency metrics
                efficiency = duration / transcription_time if transcription_time > 0 else 0
                print(f"⚡ Efficiency: {efficiency:.2f}x real-time")
            
            # Calculate statistics
            if results['transcription_times']:
                results['average_latency'] = np.mean(results['transcription_times'])
                results['min_latency'] = np.min(results['transcription_times'])
                results['max_latency'] = np.max(results['transcription_times'])
                results['std_latency'] = np.std(results['transcription_times'])
                
                # Calculate average efficiency
                total_duration = sum(test_durations)
                total_transcription_time = sum(results['transcription_times'])
                results['overall_efficiency'] = total_duration / total_transcription_time if total_transcription_time > 0 else 0
                
                print(f"\n📊 STATISTICS:")
                print(f"  ⏱️  Average latency: {results['average_latency']:.2f}s")
                print(f"  ⏱️  Min latency: {results['min_latency']:.2f}s")
                print(f"  ⏱️  Max latency: {results['max_latency']:.2f}s")
                print(f"  📈 Std deviation: {results['std_latency']:.2f}s")
                print(f"  ⚡ Overall efficiency: {results['overall_efficiency']:.2f}x real-time")
                
        except Exception as e:
            print(f"❌ Error during benchmarking: {e}")
            results['error'] = str(e)
        
        return results
    
    def benchmark_with_real_audio(self, audio_files=None):
        """Benchmark with real audio files if available"""
        if audio_files is None:
        # Look for existing audio files in common locations
        audio_files = []
        search_paths = [
            "assets/voice_samples",
            "assets/prompts", 
            "data/fillers",
            "shared"
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for file in os.listdir(search_path):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(search_path, file))
                        if len(audio_files) >= 3:  # Limit to 3 files
                            break
            if len(audio_files) >= 3:
                break
        
        if not audio_files:
            print("⚠️ No real audio files found, using synthetic audio only")
            return {}
        
        print(f"\n🎵 Testing with {len(audio_files)} real audio files...")
        
        results = {
            'real_audio_times': [],
            'real_audio_files': audio_files,
            'average_real_latency': 0
        }
        
        try:
            model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
            
            for i, audio_file in enumerate(audio_files[:3]):  # Test first 3 files
                print(f"\n🎵 Testing file {i+1}: {os.path.basename(audio_file)}")
                
                # Load audio
                audio, sr = sf.read(audio_file)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                
                # Measure transcription time
                start_time = time.time()
                segments, _ = model.transcribe(audio, language="en", beam_size=5)
                transcription_time = time.time() - start_time
                
                # Get transcribed text
                text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                
                results['real_audio_times'].append(transcription_time)
                
                print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                print(f"📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if results['real_audio_times']:
                results['average_real_latency'] = np.mean(results['real_audio_times'])
                print(f"\n📊 Real audio average latency: {results['average_real_latency']:.2f}s")
                
        except Exception as e:
            print(f"❌ Error with real audio: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_benchmark(self):
        """Run complete benchmark"""
        print("🚀 Starting Quick Whisper Benchmark...")
        print("Testing faster-whisper with distill.small model")
        print("="*60)
        
        # Benchmark with synthetic audio
        synthetic_results = self.benchmark_faster_whisper()
        
        # Benchmark with real audio if available
        real_audio_results = self.benchmark_with_real_audio()
        
        # Compile results
        self.results = {
            'synthetic_audio': synthetic_results,
            'real_audio': real_audio_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
        }
        
        # Save results
        with open('quick_whisper_benchmark.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🎯 QUICK WHISPER BENCHMARK SUMMARY")
        print("="*60)
        
        synthetic = self.results['synthetic_audio']
        real = self.results['real_audio']
        
        print(f"\n🔬 faster-whisper (distill.small) Results:")
        print(f"  ⏱️  Model loading: {synthetic.get('model_loading_time', 0):.2f}s")
        print(f"  ⏱️  Average latency: {synthetic.get('average_latency', 0):.2f}s")
        print(f"  ⚡ Overall efficiency: {synthetic.get('overall_efficiency', 0):.2f}x real-time")
        
        if real and 'average_real_latency' in real:
            print(f"  🎵 Real audio latency: {real['average_real_latency']:.2f}s")
        
        print(f"\n💾 Results saved to: quick_whisper_benchmark.json")
        print("="*60)

def main():
    """Main execution"""
    import sys
    
    benchmark = QuickWhisperBenchmark()
    results = benchmark.run_benchmark()
    
    print("\n🎉 Benchmark complete!")
    print("\nTo compare with whisper-container:")
    print("1. Start whisper container: docker compose up whisper")
    print("2. Run full benchmark: python scripts/benchmark_whisper.py")
    
    return results

if __name__ == "__main__":
    main()
