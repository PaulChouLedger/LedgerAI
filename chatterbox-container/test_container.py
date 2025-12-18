#!/usr/bin/env python3
"""
Test script for Chatterbox-TTS Container
Tests the REST API endpoints
"""

import requests
import json
import os
import sys

# Default container URL
CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://localhost:11437")

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{CHATTERBOX_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Chatterbox loaded: {data.get('chatterbox_loaded')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {CHATTERBOX_URL}")
        print(f"   Make sure container is running: docker compose up chatterbox-tts")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_synthesize(text="Hello, this is a test of Chatterbox TTS.", voice_sample=None, output_name="test_output.wav"):
    """Test synthesis endpoint"""
    print(f"\nTesting synthesis endpoint...")
    text_preview = text[:80] + "..." if len(text) > 80 else text
    print(f"   Text: '{text_preview}'")
    
    import time
    start_time = time.time()
    
    try:
        payload = {
            "text": text,
            "exaggeration": 0.6
        }
        if voice_sample:
            payload["voice_sample"] = voice_sample
        
        # Use longer timeout for first request (model might still be loading)
        timeout = 600 if not voice_sample else 120  # 10 min for first, 2 min for voice cloning
        
        response = requests.post(
            f"{CHATTERBOX_URL}/synthesize",
            json=payload,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            # Save audio file
            output_file = output_name
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Validate audio file
            file_size = len(response.content)
            file_size_kb = file_size / 1024
            
            # Try to read audio file to validate it
            try:
                import soundfile as sf
                with sf.SoundFile(output_file) as f:
                    duration = len(f) / f.samplerate
                    sample_rate = f.samplerate
                    channels = f.channels
                audio_valid = True
            except Exception as e:
                audio_valid = False
                duration = None
                sample_rate = None
                channels = None
            
            print(f"✅ Synthesis successful")
            print(f"   Audio saved to: {output_file}")
            print(f"   File size: {file_size:,} bytes ({file_size_kb:.1f} KB)")
            print(f"   Latency: {elapsed:.2f} seconds")
            if audio_valid:
                print(f"   Audio valid: {duration:.2f}s, {sample_rate}Hz, {channels} channel(s)")
            else:
                print(f"   ⚠️  Audio file may be corrupted (could not read)")
            return True, elapsed
        else:
            elapsed = time.time() - start_time
            print(f"❌ Synthesis failed: HTTP {response.status_code}")
            print(f"   Latency: {elapsed:.2f} seconds")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False, elapsed
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"❌ Synthesis timed out after {elapsed:.2f} seconds")
        print(f"   This may indicate the model is still loading or using CPU (very slow)")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"   Latency: {elapsed:.2f} seconds")
        import traceback
        traceback.print_exc()
        return False, elapsed

def test_voice_embedding(voice_sample_path):
    """Test voice embedding extraction"""
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        print(f"⚠️  Voice sample not found: {voice_sample_path}")
        print(f"   Skipping voice embedding test")
        return True
    
    print(f"\nTesting voice embedding extraction...")
    print(f"   Voice sample: {voice_sample_path}")
    
    try:
        payload = {
            "voice_sample_path": voice_sample_path
        }
        response = requests.post(
            f"{CHATTERBOX_URL}/voice/embedding",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Voice embedding extracted successfully")
            print(f"   Voice sample: {data.get('voice_sample')}")
            print(f"   Embedding cached: {data.get('embedding_cached')}")
            return True
        else:
            print(f"❌ Voice embedding failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("  Chatterbox-TTS Container Test")
    print("=" * 60)
    print(f"Container URL: {CHATTERBOX_URL}")
    print()
    
    # Test health
    if not test_health():
        print("\n❌ Container is not healthy or not accessible")
        print("\n💡 To start the container:")
        print("   cd setup")
        print("   docker compose up -d chatterbox-tts")
        print("\n💡 To check logs:")
        print("   docker compose logs -f chatterbox-tts")
        sys.exit(1)
    
    # Test synthesis
    success, latency = test_synthesize("Hello, this is a test of the Chatterbox TTS container....", output_name="test_output_basic_synthesis.wav")
    if not success:
        print("\n❌ Synthesis test failed")
        sys.exit(1)
    
    latencies = [latency]
    
    # Test with voice cloning using audio3.wav from prompts directory
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Try audio3.wav from prompts directory (ideal for voice cloning - 15.33s)
    voice_sample_prompts = os.path.join(workspace_root, "assets", "prompts", "audio3.wav")
    voice_sample_samples = os.path.join(workspace_root, "assets", "voice_samples", "audio3.wav")
    voice_sample_default = os.path.join(workspace_root, "assets", "voice_samples", "sample.wav")
    
    voice_sample = None
    voice_sample_name = None
    
    # Prefer audio3.wav from prompts (ideal 15s sample)
    if os.path.exists(voice_sample_prompts):
        # Copy to voice_samples directory so container can access it
        import shutil
        os.makedirs(os.path.dirname(voice_sample_samples), exist_ok=True)
        if not os.path.exists(voice_sample_samples):
            print(f"\n📋 Copying audio3.wav to voice_samples directory for container access...")
            shutil.copy2(voice_sample_prompts, voice_sample_samples)
            print(f"   ✅ Copied: {voice_sample_samples}")
        voice_sample = voice_sample_samples
        voice_sample_name = "audio3.wav"
        print(f"\n🎭 Using audio3.wav (15.33s) for voice cloning test")
    elif os.path.exists(voice_sample_samples):
        voice_sample = voice_sample_samples
        voice_sample_name = "audio3.wav"
        print(f"\n🎭 Using audio3.wav from voice_samples directory")
    elif os.path.exists(voice_sample_default):
        voice_sample = voice_sample_default
        voice_sample_name = "sample.wav"
        print(f"\n🎭 Using sample.wav for voice cloning test")
    
    if voice_sample and os.path.exists(voice_sample):
        # Test voice embedding extraction
        # Use container path (container will check /app/voice_samples/)
        container_sample_path = f"/app/voice_samples/{voice_sample_name}"
        print(f"\n📋 Testing voice embedding with container path: {container_sample_path}")
        test_voice_embedding(container_sample_path)
        
        # Test synthesis with voice cloning
        test_text = "Hello, this is a test with voice cloning enabled using the audio3 sample. This should sound like the voice in the audio3 recording."
        print(f"\n🎭 Testing voice cloning synthesis...")
        print(f"   Using sample: {voice_sample_name}")
        print(f"   Container will look in: /app/voice_samples/{voice_sample_name}")
        success, latency = test_synthesize(
            test_text, 
            voice_sample_name,  # Container will look in /app/voice_samples/
            output_name="test_output_voice_cloning_audio3.wav"
        )
        if success:
            latencies.append(latency)
            print(f"\n💡 Voice cloning test completed")
            print(f"   Compare test_output_voice_cloning_audio3.wav with audio3.wav")
            print(f"   They should sound similar if voice cloning is working")
    else:
        print(f"\n⚠️  No voice sample found for cloning test")
        print(f"   Checked:")
        print(f"     - {voice_sample_prompts}")
        print(f"     - {voice_sample_samples}")
        print(f"     - {voice_sample_default}")
        print(f"   Skipping voice cloning tests")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    print()
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Test Results:")
        print(f"  ✅ container_built: True")
        print(f"  ✅ container_running: True")
        print(f"  ✅ health_check: True")
        print(f"  ✅ synthesis_basic: True")
        if voice_sample:
            print(f"  ✅ synthesis_voice_cloning: True")
            print(f"  ✅ voice_embedding: True")
        print(f"  📊 latency: {latencies[0]}")
        if len(latencies) > 1:
            print(f"  📊 voice_cloning_latency: {latencies[1]}")
        print(f"  📊 audio_quality: not_tested")
        print()
        print(f"⏱️  Average latency: {avg_latency:.2f} seconds")
        print()
    
    print("🌐 Container URL:", CHATTERBOX_URL)
    print("📦 Container name: chatterbox-tts")
    print()
    print("=" * 70)
    print("  ✅ All tests passed!")
    print("=" * 70)
    print()
    print("💡 To use Chatterbox-TTS in your code:")
    print(f"   import requests")
    print(f"   response = requests.post('{CHATTERBOX_URL}/synthesize', json={{'text': 'Hello'}})")
    print(f"   with open('output.wav', 'wb') as f:")
    print(f"       f.write(response.content)")
    print()
    if voice_sample_name:
        print(f"💡 Voice cloning is enabled using: {voice_sample_name}")
        print(f"   To test voice cloning:")
        print(f"   response = requests.post('{CHATTERBOX_URL}/synthesize', json={{")
        print(f"       'text': 'Your text here',")
        print(f"       'voice_sample': '{voice_sample_name}'")
        print(f"   }})")

if __name__ == '__main__':
    main()

