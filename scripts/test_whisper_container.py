#!/usr/bin/env python3
"""
Test Whisper Container Performance

Simple script to test the whisper container and measure latency
"""

import requests
import time
import os
import json
import numpy as np
import soundfile as sf

def create_test_audio(duration=3.0, sample_rate=16000):
    """Create synthetic test audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like audio
    audio = (np.sin(2 * np.pi * 200 * t) * 0.3 + 
            np.sin(2 * np.pi * 400 * t) * 0.2 +
            np.sin(2 * np.pi * 800 * t) * 0.1)
    
    # Add noise and normalize
    audio += np.random.normal(0, 0.02, len(audio))
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)

def test_whisper_container(container_url="http://localhost:5000"):
    """Test whisper container performance"""
    print("🔬 Testing Whisper Container Performance...")
    print(f"Container URL: {container_url}")
    
    results = {
        'container_available': False,
        'transcription_times': [],
        'average_latency': 0,
        'errors': []
    }
    
    try:
        # Test container availability
        print("🔍 Checking container availability...")
        response = requests.get(f"{container_url}/health", timeout=5)
        if response.status_code == 200:
            results['container_available'] = True
            print("✅ Container is running")
        else:
            print(f"⚠️ Container health check failed: {response.status_code}")
            return results
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Container not available: {e}")
        results['errors'].append(f"Container not available: {str(e)}")
        return results
    
    # Test with different audio durations
    test_durations = [1.0, 2.0, 3.0, 5.0]
    
    for duration in test_durations:
        print(f"\n🎵 Testing {duration}s audio...")
        
        # Create test audio
        audio = create_test_audio(duration)
        
        # Save to temporary file
        temp_file = f"temp_test_{duration}s.wav"
        sf.write(temp_file, audio, 16000)
        
        try:
            # Measure transcription time
            start_time = time.time()
            
            with open(temp_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{container_url}/transcribe", files=files, timeout=30)
            
            transcription_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')
                
                results['transcription_times'].append(transcription_time)
                
                print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                print(f"📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Calculate efficiency
                efficiency = duration / transcription_time if transcription_time > 0 else 0
                print(f"⚡ Efficiency: {efficiency:.2f}x real-time")
            else:
                error_msg = f"Transcription failed: {response.status_code} - {response.text}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Calculate statistics
    if results['transcription_times']:
        results['average_latency'] = np.mean(results['transcription_times'])
        results['min_latency'] = np.min(results['transcription_times'])
        results['max_latency'] = np.max(results['transcription_times'])
        results['std_latency'] = np.std(results['transcription_times'])
        
        print(f"\n📊 STATISTICS:")
        print(f"  ⏱️  Average latency: {results['average_latency']:.2f}s")
        print(f"  ⏱️  Min latency: {results['min_latency']:.2f}s")
        print(f"  ⏱️  Max latency: {results['max_latency']:.2f}s")
        print(f"  📈 Std deviation: {results['std_latency']:.2f}s")
    
    return results

def test_with_real_audio(container_url="http://localhost:5000"):
    """Test with real audio files if available"""
    print("\n🎵 Testing with real audio files...")
    
    # Look for existing audio files
    audio_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
                if len(audio_files) >= 3:  # Limit to 3 files
                    break
    
    if not audio_files:
        print("⚠️ No real audio files found")
        return {}
    
    results = {
        'real_audio_times': [],
        'real_audio_files': audio_files
    }
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n🎵 Testing file {i+1}: {os.path.basename(audio_file)}")
        
        try:
            # Measure transcription time
            start_time = time.time()
            
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{container_url}/transcribe", files=files, timeout=30)
            
            transcription_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')
                
                results['real_audio_times'].append(transcription_time)
                
                print(f"⏱️  Transcription time: {transcription_time:.2f}s")
                print(f"📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            else:
                print(f"❌ Transcription failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if results['real_audio_times']:
        results['average_real_latency'] = np.mean(results['real_audio_times'])
        print(f"\n📊 Real audio average latency: {results['average_real_latency']:.2f}s")
    
    return results

def main():
    """Main execution"""
    print("🚀 Starting Whisper Container Test...")
    print("="*50)
    
    # Test container performance
    container_results = test_whisper_container()
    
    # Test with real audio if container is available
    real_audio_results = {}
    if container_results['container_available']:
        real_audio_results = test_with_real_audio()
    
    # Compile results
    results = {
        'container_test': container_results,
        'real_audio_test': real_audio_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    with open('whisper_container_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 WHISPER CONTAINER TEST SUMMARY")
    print("="*50)
    
    if container_results['container_available']:
        print(f"✅ Container is running")
        print(f"⏱️  Average latency: {container_results['average_latency']:.2f}s")
        if real_audio_results and 'average_real_latency' in real_audio_results:
            print(f"🎵 Real audio latency: {real_audio_results['average_real_latency']:.2f}s")
    else:
        print("❌ Container not available")
        print("Start container with: docker compose up whisper")
    
    print(f"\n💾 Results saved to: whisper_container_test.json")
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()
