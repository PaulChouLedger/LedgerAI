#!/usr/bin/env python3
"""
Speech Accuracy & Latency Benchmark

Tests both faster-whisper and whisper-container with actual speech
to compare accuracy and latency across both models.
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
import tempfile

class SpeechAccuracyBenchmark:
    def __init__(self):
        self.results = {
            'faster_whisper': {},
            'whisper_container': {},
            'comparison': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Test speech samples with known ground truth
        self.test_samples = [
            {
                'file': 'assets/voice_samples/sample.wav',
                'ground_truth': 'The birds can use weight on a few of the plants. A...',
                'description': 'Natural speech sample'
            },
            {
                'file': 'assets/voice_samples/pcm1622m.wav', 
                'ground_truth': 'Heheheh heh heh heh! Heh heh! Ha ha ha ha ha ha!',
                'description': 'Laughter sample'
            },
            {
                'file': 'assets/voice_samples/startup_test.wav',
                'ground_truth': 'Startup test audio',
                'description': 'Startup test'
            }
        ]
        
        # Additional test samples if available
        self.additional_samples = [
            'assets/voice_samples/audio1.wav',
            'assets/voice_samples/startup.wav',
            'assets/voice_samples/welcome.wav',
            'assets/prompts/audio2.wav',
            'assets/prompts/audio3.wav'
        ]
    
    def calculate_accuracy(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate accuracy metrics between predicted and ground truth text"""
        predicted = predicted.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # Word-level accuracy
        pred_words = predicted.split()
        truth_words = ground_truth.split()
        
        # Calculate word overlap
        pred_set = set(pred_words)
        truth_set = set(truth_words)
        
        if len(truth_set) == 0:
            return {'word_accuracy': 0.0, 'character_accuracy': 0.0, 'similarity': 0.0}
        
        # Word accuracy (intersection over union)
        word_intersection = len(pred_set.intersection(truth_set))
        word_union = len(pred_set.union(truth_set))
        word_accuracy = word_intersection / word_union if word_union > 0 else 0.0
        
        # Character-level accuracy
        char_intersection = len(set(predicted).intersection(set(ground_truth)))
        char_union = len(set(predicted).union(set(ground_truth)))
        character_accuracy = char_intersection / char_union if char_union > 0 else 0.0
        
        # Simple similarity (common words / total words in ground truth)
        similarity = word_intersection / len(truth_set) if len(truth_set) > 0 else 0.0
        
        return {
            'word_accuracy': word_accuracy,
            'character_accuracy': character_accuracy,
            'similarity': similarity
        }
    
    def test_faster_whisper(self) -> Dict:
        """Test faster-whisper with actual speech samples"""
        print("🔬 Testing faster-whisper (distill.small) with real speech...")
        
        results = {
            'model_loading_time': 0,
            'samples': [],
            'average_latency': 0,
            'average_accuracy': 0,
            'total_samples': 0
        }
        
        try:
            # Load model
            print("⏳ Loading faster-whisper model...")
            start_time = time.time()
            model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
            results['model_loading_time'] = time.time() - start_time
            print(f"✅ Model loaded in {results['model_loading_time']:.2f}s")
            
            # Test each sample
            for i, sample in enumerate(self.test_samples):
                if not os.path.exists(sample['file']):
                    print(f"⚠️ Sample not found: {sample['file']}")
                    continue
                
                print(f"\n🎵 Testing sample {i+1}: {os.path.basename(sample['file'])}")
                print(f"📝 Ground truth: '{sample['ground_truth'][:50]}{'...' if len(sample['ground_truth']) > 50 else ''}'")
                
                # Load audio
                audio, sr = sf.read(sample['file'])
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                
                # Measure transcription time
                start_time = time.time()
                segments, _ = model.transcribe(audio, language="en", beam_size=5)
                transcription_time = time.time() - start_time
                
                # Get transcribed text
                predicted_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                
                # Calculate accuracy
                accuracy_metrics = self.calculate_accuracy(predicted_text, sample['ground_truth'])
                
                # Store results
                sample_result = {
                    'file': sample['file'],
                    'description': sample['description'],
                    'ground_truth': sample['ground_truth'],
                    'predicted': predicted_text,
                    'transcription_time': transcription_time,
                    'audio_duration': len(audio) / sr,
                    'efficiency': (len(audio) / sr) / transcription_time if transcription_time > 0 else 0,
                    'accuracy_metrics': accuracy_metrics
                }
                
                results['samples'].append(sample_result)
                results['total_samples'] += 1
                
                print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                print(f"📝 Predicted: '{predicted_text[:50]}{'...' if len(predicted_text) > 50 else ''}'")
                print(f"🎯 Word accuracy: {accuracy_metrics['word_accuracy']:.2f}")
                print(f"🎯 Similarity: {accuracy_metrics['similarity']:.2f}")
                print(f"⚡ Efficiency: {sample_result['efficiency']:.2f}x real-time")
            
            # Calculate averages
            if results['samples']:
                results['average_latency'] = np.mean([s['transcription_time'] for s in results['samples']])
                results['average_accuracy'] = np.mean([s['accuracy_metrics']['similarity'] for s in results['samples']])
                
                print(f"\n📊 faster-whisper Summary:")
                print(f"  ⏱️  Average latency: {results['average_latency']:.2f}s")
                print(f"  🎯 Average accuracy: {results['average_accuracy']:.2f}")
                print(f"  📁 Samples tested: {results['total_samples']}")
                
        except Exception as e:
            print(f"❌ faster-whisper test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_whisper_container(self) -> Dict:
        """Test whisper-container with actual speech samples"""
        print("\n🔬 Testing whisper-container (TensorRT base.en) with real speech...")
        
        results = {
            'container_available': False,
            'samples': [],
            'average_latency': 0,
            'average_accuracy': 0,
            'total_samples': 0
        }
        
        try:
            # Check container availability
            print("🔍 Checking whisper-container availability...")
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                results['container_available'] = True
                print("✅ Whisper container is running")
            else:
                print(f"⚠️ Container health check failed: {response.status_code}")
                return results
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Whisper container not available: {e}")
            results['error'] = f"Container not available: {str(e)}"
            return results
        
        # Test each sample
        for i, sample in enumerate(self.test_samples):
            if not os.path.exists(sample['file']):
                print(f"⚠️ Sample not found: {sample['file']}")
                continue
            
            print(f"\n🎵 Testing sample {i+1}: {os.path.basename(sample['file'])}")
            print(f"📝 Ground truth: '{sample['ground_truth'][:50]}{'...' if len(sample['ground_truth']) > 50 else ''}'")
            
            try:
                # Measure transcription time
                start_time = time.time()
                
                with open(sample['file'], 'rb') as f:
                    files = {'audio': f}
                    response = requests.post("http://localhost:5000/transcribe", files=files, timeout=30)
                
                transcription_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_text = result.get('text', '')
                    
                    # Calculate accuracy
                    accuracy_metrics = self.calculate_accuracy(predicted_text, sample['ground_truth'])
                    
                    # Get audio duration
                    audio, sr = sf.read(sample['file'])
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    audio_duration = len(audio) / sr
                    
                    # Store results
                    sample_result = {
                        'file': sample['file'],
                        'description': sample['description'],
                        'ground_truth': sample['ground_truth'],
                        'predicted': predicted_text,
                        'transcription_time': transcription_time,
                        'audio_duration': audio_duration,
                        'efficiency': audio_duration / transcription_time if transcription_time > 0 else 0,
                        'accuracy_metrics': accuracy_metrics
                    }
                    
                    results['samples'].append(sample_result)
                    results['total_samples'] += 1
                    
                    print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                    print(f"📝 Predicted: '{predicted_text[:50]}{'...' if len(predicted_text) > 50 else ''}'")
                    print(f"🎯 Word accuracy: {accuracy_metrics['word_accuracy']:.2f}")
                    print(f"🎯 Similarity: {accuracy_metrics['similarity']:.2f}")
                    print(f"⚡ Efficiency: {sample_result['efficiency']:.2f}x real-time")
                else:
                    print(f"❌ Transcription failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"❌ Error testing sample {i+1}: {e}")
        
        # Calculate averages
        if results['samples']:
            results['average_latency'] = np.mean([s['transcription_time'] for s in results['samples']])
            results['average_accuracy'] = np.mean([s['accuracy_metrics']['similarity'] for s in results['samples']])
            
            print(f"\n📊 whisper-container Summary:")
            print(f"  ⏱️  Average latency: {results['average_latency']:.2f}s")
            print(f"  🎯 Average accuracy: {results['average_accuracy']:.2f}")
            print(f"  📁 Samples tested: {results['total_samples']}")
        
        return results
    
    def compare_results(self) -> Dict:
        """Compare results between both models"""
        print("\n🔍 Comparing results...")
        
        fw = self.results.get('faster_whisper', {})
        wc = self.results.get('whisper_container', {})
        
        comparison = {
            'faster_whisper_available': 'error' not in fw,
            'whisper_container_available': wc.get('container_available', False),
            'winner_latency': None,
            'winner_accuracy': None,
            'latency_difference': 0,
            'accuracy_difference': 0,
            'recommendations': []
        }
        
        if comparison['faster_whisper_available'] and comparison['whisper_container_available']:
            fw_latency = fw.get('average_latency', 0)
            wc_latency = wc.get('average_latency', 0)
            fw_accuracy = fw.get('average_accuracy', 0)
            wc_accuracy = wc.get('average_accuracy', 0)
            
            # Compare latency
            if fw_latency < wc_latency:
                comparison['winner_latency'] = 'faster-whisper'
                comparison['latency_difference'] = ((wc_latency - fw_latency) / wc_latency) * 100
            else:
                comparison['winner_latency'] = 'whisper-container'
                comparison['latency_difference'] = ((fw_latency - wc_latency) / fw_latency) * 100
            
            # Compare accuracy
            if fw_accuracy > wc_accuracy:
                comparison['winner_accuracy'] = 'faster-whisper'
                comparison['accuracy_difference'] = ((fw_accuracy - wc_accuracy) / wc_accuracy) * 100
            else:
                comparison['winner_accuracy'] = 'whisper-container'
                comparison['accuracy_difference'] = ((wc_accuracy - fw_accuracy) / fw_accuracy) * 100
            
            # Generate recommendations
            if comparison['winner_latency'] == 'faster-whisper':
                comparison['recommendations'].append("faster-whisper is faster for real-time applications")
            else:
                comparison['recommendations'].append("whisper-container is faster for real-time applications")
            
            if comparison['winner_accuracy'] == 'faster-whisper':
                comparison['recommendations'].append("faster-whisper provides better accuracy")
            else:
                comparison['recommendations'].append("whisper-container provides better accuracy")
        
        return comparison
    
    def run_benchmark(self):
        """Run complete speech accuracy and latency benchmark"""
        print("🚀 Starting Speech Accuracy & Latency Benchmark")
        print("Testing both faster-whisper and whisper-container with real speech")
        print("="*70)
        
        # Test faster-whisper
        print("\n1️⃣ Testing faster-whisper...")
        self.results['faster_whisper'] = self.test_faster_whisper()
        
        # Test whisper-container
        print("\n2️⃣ Testing whisper-container...")
        self.results['whisper_container'] = self.test_whisper_container()
        
        # Compare results
        print("\n3️⃣ Comparing results...")
        self.results['comparison'] = self.compare_results()
        
        # Save results
        with open('speech_accuracy_benchmark.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("🎯 SPEECH ACCURACY & LATENCY BENCHMARK SUMMARY")
        print("="*70)
        
        fw = self.results.get('faster_whisper', {})
        wc = self.results.get('whisper_container', {})
        comp = self.results.get('comparison', {})
        
        # faster-whisper results
        if 'error' not in fw:
            print(f"\n🔬 faster-whisper (distill.small):")
            print(f"  ⏱️  Average latency: {fw.get('average_latency', 0):.2f}s")
            print(f"  🎯 Average accuracy: {fw.get('average_accuracy', 0):.2f}")
            print(f"  📁 Samples tested: {fw.get('total_samples', 0)}")
        else:
            print(f"\n❌ faster-whisper failed: {fw.get('error', 'Unknown error')}")
        
        # whisper-container results
        if wc.get('container_available'):
            print(f"\n🐳 whisper-container (TensorRT base.en):")
            print(f"  ⏱️  Average latency: {wc.get('average_latency', 0):.2f}s")
            print(f"  🎯 Average accuracy: {wc.get('average_accuracy', 0):.2f}")
            print(f"  📁 Samples tested: {wc.get('total_samples', 0)}")
        else:
            print(f"\n❌ whisper-container not available")
        
        # Comparison
        if comp.get('faster_whisper_available') and comp.get('whisper_container_available'):
            print(f"\n🏆 COMPARISON RESULTS:")
            print(f"  🚀 Speed winner: {comp.get('winner_latency', 'Unknown')}")
            print(f"  🎯 Accuracy winner: {comp.get('winner_accuracy', 'Unknown')}")
            print(f"  📈 Latency difference: {comp.get('latency_difference', 0):.1f}%")
            print(f"  📈 Accuracy difference: {comp.get('accuracy_difference', 0):.1f}%")
            
            if comp.get('recommendations'):
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in comp['recommendations']:
                    print(f"  • {rec}")
        
        print(f"\n💾 Full results saved to: speech_accuracy_benchmark.json")
        print("="*70)

def main():
    """Main execution"""
    benchmark = SpeechAccuracyBenchmark()
    results = benchmark.run_benchmark()
    
    print("\n🎉 Speech accuracy benchmark complete!")
    return results

if __name__ == "__main__":
    main()
