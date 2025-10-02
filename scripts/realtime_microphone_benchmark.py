#!/usr/bin/env python3
"""
Real-time Microphone Speech Benchmark

Tests both faster-whisper and whisper-container with live microphone input
to compare accuracy and latency in real-world usage.
"""

import os
import sys
import time
import json
import requests
import numpy as np
import soundfile as sf
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Dict, List, Tuple
import tempfile
import threading
from queue import Queue

class RealtimeMicrophoneBenchmark:
    def __init__(self):
        self.results = {
            'faster_whisper': {},
            'whisper_container': {},
            'comparison': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Focus on the other WAV files (not pcm1622m.wav)
        self.test_files = [
            {
                'file': 'assets/voice_samples/sample.wav',
                'ground_truth': 'The birds can use weight on a few of the plants. A...',
                'description': 'Natural speech sample'
            },
            {
                'file': 'assets/voice_samples/startup_test.wav',
                'ground_truth': 'Startup test audio',
                'description': 'Startup test'
            },
            {
                'file': 'assets/voice_samples/audio1.wav',
                'ground_truth': 'Audio sample 1',
                'description': 'Audio sample 1'
            },
            {
                'file': 'assets/voice_samples/startup.wav',
                'ground_truth': 'Startup audio',
                'description': 'Startup audio'
            },
            {
                'file': 'assets/voice_samples/welcome.wav',
                'ground_truth': 'Welcome message',
                'description': 'Welcome message'
            }
        ]
        
        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.recording_duration = 5.0  # seconds
        
        # Models
        self.faster_whisper_model = None
        self.whisper_container_available = False
        
    def check_whisper_container(self) -> bool:
        """Check if whisper container is available"""
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def load_faster_whisper(self):
        """Load faster-whisper model"""
        print("⏳ Loading faster-whisper model...")
        start_time = time.time()
        self.faster_whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
        loading_time = time.time() - start_time
        print(f"✅ faster-whisper loaded in {loading_time:.2f}s")
        return loading_time
    
    def transcribe_with_faster_whisper(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe audio with faster-whisper"""
        start_time = time.time()
        segments, _ = self.faster_whisper_model.transcribe(audio_data, language="en", beam_size=5)
        transcription_time = time.time() - start_time
        text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        return text, transcription_time
    
    def transcribe_with_whisper_container(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe audio with whisper container"""
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, self.sample_rate)
            
            start_time = time.time()
            try:
                with open(tmp_file.name, 'rb') as f:
                    files = {'audio': f}
                    response = requests.post("http://localhost:5000/transcribe", files=files, timeout=30)
                
                transcription_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('text', '')
                else:
                    text = f"Error: {response.status_code}"
                    
            except Exception as e:
                text = f"Error: {str(e)}"
                transcription_time = time.time() - start_time
            finally:
                os.unlink(tmp_file.name)
        
        return text, transcription_time
    
    def record_audio(self, duration: float = None) -> np.ndarray:
        """Record audio from microphone"""
        if duration is None:
            duration = self.recording_duration
            
        print(f"🎤 Recording {duration}s of audio...")
        print("   Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate), 
            samplerate=self.sample_rate, 
            channels=self.channels, 
            dtype=self.dtype
        )
        sd.wait()  # Wait until recording is finished
        
        print("✅ Recording complete!")
        return audio_data.flatten()
    
    def calculate_accuracy(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate accuracy metrics between predicted and ground truth text"""
        predicted = predicted.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        if not predicted or not ground_truth:
            return {'word_accuracy': 0.0, 'character_accuracy': 0.0, 'similarity': 0.0}
        
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
    
    def test_with_existing_files(self):
        """Test both models with existing audio files"""
        print("🔬 Testing with existing audio files...")
        
        faster_whisper_results = {
            'samples': [],
            'average_latency': 0,
            'average_accuracy': 0,
            'total_samples': 0
        }
        
        whisper_container_results = {
            'samples': [],
            'average_latency': 0,
            'average_accuracy': 0,
            'total_samples': 0
        }
        
        # Test faster-whisper
        print("\n📊 Testing faster-whisper with existing files...")
        for i, sample in enumerate(self.test_files):
            if not os.path.exists(sample['file']):
                print(f"⚠️ File not found: {sample['file']}")
                continue
                
            print(f"\n🎵 Testing {os.path.basename(sample['file'])}")
            
            # Load audio
            audio, sr = sf.read(sample['file'])
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Transcribe with faster-whisper
            text, latency = self.transcribe_with_faster_whisper(audio)
            accuracy = self.calculate_accuracy(text, sample['ground_truth'])
            
            faster_whisper_results['samples'].append({
                'file': sample['file'],
                'predicted': text,
                'latency': latency,
                'accuracy': accuracy
            })
            faster_whisper_results['total_samples'] += 1
            
            print(f"  ⏱️  Latency: {latency:.2f}s")
            print(f"  📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  🎯 Accuracy: {accuracy['similarity']:.2f}")
        
        # Test whisper-container if available
        if self.whisper_container_available:
            print("\n📊 Testing whisper-container with existing files...")
            for i, sample in enumerate(self.test_files):
                if not os.path.exists(sample['file']):
                    continue
                    
                print(f"\n🎵 Testing {os.path.basename(sample['file'])}")
                
                # Load audio
                audio, sr = sf.read(sample['file'])
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Transcribe with whisper-container
                text, latency = self.transcribe_with_whisper_container(audio)
                accuracy = self.calculate_accuracy(text, sample['ground_truth'])
                
                whisper_container_results['samples'].append({
                    'file': sample['file'],
                    'predicted': text,
                    'latency': latency,
                    'accuracy': accuracy
                })
                whisper_container_results['total_samples'] += 1
                
                print(f"  ⏱️  Latency: {latency:.2f}s")
                print(f"  📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"  🎯 Accuracy: {accuracy['similarity']:.2f}")
        
        # Calculate averages
        if faster_whisper_results['samples']:
            faster_whisper_results['average_latency'] = np.mean([s['latency'] for s in faster_whisper_results['samples']])
            faster_whisper_results['average_accuracy'] = np.mean([s['accuracy']['similarity'] for s in faster_whisper_results['samples']])
        
        if whisper_container_results['samples']:
            whisper_container_results['average_latency'] = np.mean([s['latency'] for s in whisper_container_results['samples']])
            whisper_container_results['average_accuracy'] = np.mean([s['accuracy']['similarity'] for s in whisper_container_results['samples']])
        
        return faster_whisper_results, whisper_container_results
    
    def test_with_microphone(self):
        """Test both models with live microphone input"""
        print("\n🎤 Testing with live microphone input...")
        print("This will record audio and test both models in real-time")
        
        # Record audio
        audio_data = self.record_audio()
        
        # Test faster-whisper
        print("\n🔬 Testing faster-whisper with microphone input...")
        fw_text, fw_latency = self.transcribe_with_faster_whisper(audio_data)
        print(f"⏱️  Latency: {fw_latency:.2f}s")
        print(f"📝 Text: '{fw_text}'")
        
        # Test whisper-container if available
        wc_text, wc_latency = "", 0
        if self.whisper_container_available:
            print("\n🔬 Testing whisper-container with microphone input...")
            wc_text, wc_latency = self.transcribe_with_whisper_container(audio_data)
            print(f"⏱️  Latency: {wc_latency:.2f}s")
            print(f"📝 Text: '{wc_text}'")
        else:
            print("⚠️ Whisper container not available")
        
        return {
            'faster_whisper': {'text': fw_text, 'latency': fw_latency},
            'whisper_container': {'text': wc_text, 'latency': wc_latency}
        }
    
    def run_benchmark(self):
        """Run complete real-time microphone benchmark"""
        print("🚀 Starting Real-time Microphone Speech Benchmark")
        print("Testing both faster-whisper and whisper-container with live speech")
        print("="*70)
        
        # Check whisper container availability
        self.whisper_container_available = self.check_whisper_container()
        print(f"🐳 Whisper container available: {self.whisper_container_available}")
        
        # Load faster-whisper model
        model_loading_time = self.load_faster_whisper()
        
        # Test with existing files
        print("\n1️⃣ Testing with existing audio files...")
        fw_file_results, wc_file_results = self.test_with_existing_files()
        
        # Test with microphone
        print("\n2️⃣ Testing with live microphone input...")
        mic_results = self.test_with_microphone()
        
        # Compile results
        self.results = {
            'faster_whisper': {
                'model_loading_time': model_loading_time,
                'file_tests': fw_file_results,
                'microphone_test': mic_results['faster_whisper']
            },
            'whisper_container': {
                'available': self.whisper_container_available,
                'file_tests': wc_file_results,
                'microphone_test': mic_results['whisper_container']
            },
            'comparison': {
                'faster_whisper_avg_latency': fw_file_results.get('average_latency', 0),
                'whisper_container_avg_latency': wc_file_results.get('average_latency', 0),
                'faster_whisper_avg_accuracy': fw_file_results.get('average_accuracy', 0),
                'whisper_container_avg_accuracy': wc_file_results.get('average_accuracy', 0)
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        with open('realtime_microphone_benchmark.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("🎯 REAL-TIME MICROPHONE BENCHMARK SUMMARY")
        print("="*70)
        
        fw = self.results.get('faster_whisper', {})
        wc = self.results.get('whisper_container', {})
        comp = self.results.get('comparison', {})
        
        # faster-whisper results
        print(f"\n🔬 faster-whisper (distill.small):")
        print(f"  ⏱️  Model loading: {fw.get('model_loading_time', 0):.2f}s")
        print(f"  ⏱️  Average latency: {fw.get('file_tests', {}).get('average_latency', 0):.2f}s")
        print(f"  🎯 Average accuracy: {fw.get('file_tests', {}).get('average_accuracy', 0):.2f}")
        print(f"  📁 Files tested: {fw.get('file_tests', {}).get('total_samples', 0)}")
        
        mic_fw = fw.get('microphone_test', {})
        if mic_fw.get('text'):
            print(f"  🎤 Microphone test: '{mic_fw.get('text', '')[:30]}{'...' if len(mic_fw.get('text', '')) > 30 else ''}'")
            print(f"  🎤 Mic latency: {mic_fw.get('latency', 0):.2f}s")
        
        # whisper-container results
        if wc.get('available'):
            print(f"\n🐳 whisper-container (TensorRT base.en):")
            print(f"  ⏱️  Average latency: {wc.get('file_tests', {}).get('average_latency', 0):.2f}s")
            print(f"  🎯 Average accuracy: {wc.get('file_tests', {}).get('average_accuracy', 0):.2f}")
            print(f"  📁 Files tested: {wc.get('file_tests', {}).get('total_samples', 0)}")
            
            mic_wc = wc.get('microphone_test', {})
            if mic_wc.get('text'):
                print(f"  🎤 Microphone test: '{mic_wc.get('text', '')[:30]}{'...' if len(mic_wc.get('text', '')) > 30 else ''}'")
                print(f"  🎤 Mic latency: {mic_wc.get('latency', 0):.2f}s")
        else:
            print(f"\n❌ whisper-container not available")
        
        # Comparison
        if wc.get('available'):
            fw_latency = comp.get('faster_whisper_avg_latency', 0)
            wc_latency = comp.get('whisper_container_avg_latency', 0)
            fw_accuracy = comp.get('faster_whisper_avg_accuracy', 0)
            wc_accuracy = comp.get('whisper_container_avg_accuracy', 0)
            
            print(f"\n🏆 COMPARISON:")
            if fw_latency < wc_latency:
                print(f"  🚀 Speed winner: faster-whisper ({fw_latency:.2f}s vs {wc_latency:.2f}s)")
            else:
                print(f"  🚀 Speed winner: whisper-container ({wc_latency:.2f}s vs {fw_latency:.2f}s)")
            
            if fw_accuracy > wc_accuracy:
                print(f"  🎯 Accuracy winner: faster-whisper ({fw_accuracy:.2f} vs {wc_accuracy:.2f})")
            else:
                print(f"  🎯 Accuracy winner: whisper-container ({wc_accuracy:.2f} vs {fw_accuracy:.2f})")
        
        print(f"\n💾 Full results saved to: realtime_microphone_benchmark.json")
        print("="*70)

def main():
    """Main execution"""
    benchmark = RealtimeMicrophoneBenchmark()
    results = benchmark.run_benchmark()
    
    print("\n🎉 Real-time microphone benchmark complete!")
    return results

if __name__ == "__main__":
    main()
