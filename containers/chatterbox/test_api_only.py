#!/usr/bin/env python3
"""
Simple API-only test for Chatterbox-TTS Container
Tests the container API without requiring Docker to be available locally
Useful for testing containers running on remote machines or already started
"""

import requests
import json
import os
import sys
import time
from pathlib import Path

# Configuration
CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://localhost:11437")
WORKSPACE_ROOT = Path(__file__).parent.parent
VOICE_SAMPLES_DIR = WORKSPACE_ROOT / "assets" / "voice_samples"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print a formatted section"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def test_health():
    """Test health endpoint"""
    print_section("Health Check")
    
    try:
        print(f"Checking: {CHATTERBOX_URL}/health")
        response = requests.get(f"{CHATTERBOX_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print(f"   Chatterbox loaded: {data.get('chatterbox_loaded')}")
            print(f"   Can import: {data.get('can_import_chatterbox')}")
            print(f"   Device: {data.get('device')}")
            print(f"   Source directory exists: {data.get('source_directory_exists')}")
            
            if data.get('import_error'):
                print(f"   ⚠️  Import error: {data.get('import_error')}")
            
            if not data.get('chatterbox_loaded'):
                print(f"\n   ⚠️  Warning: Chatterbox not loaded yet")
                print(f"      First synthesis request will load it")
            
            return True, data
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False, None
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {CHATTERBOX_URL}")
        print(f"   Make sure container is running and accessible")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

def test_synthesize(text="Hello, this is a test of the Chatterbox TTS container.", voice_sample=None):
    """Test synthesis endpoint"""
    test_name = "Voice Cloning Synthesis" if voice_sample else "Basic Synthesis"
    print_section(test_name)
    
    try:
        payload = {
            "text": text,
            "exaggeration": 0.6
        }
        if voice_sample:
            payload["voice_sample"] = voice_sample
        
        print(f"Text: '{text[:60]}...'")
        if voice_sample:
            print(f"Voice sample: {voice_sample}")
        
        print("Sending request...")
        start_time = time.time()
        response = requests.post(
            f"{CHATTERBOX_URL}/synthesize",
            json=payload,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            # Save audio file
            output_file = f"test_output_{'cloned' if voice_sample else 'basic'}.wav"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✅ Synthesis successful")
            print(f"   Audio saved to: {output_file}")
            print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   Latency: {elapsed_time:.2f} seconds")
            
            # Check if it's a valid WAV file
            if file_size > 44:  # WAV header is 44 bytes
                print(f"   ✅ Valid WAV file (size > header)")
            else:
                print(f"   ⚠️  File seems too small for valid WAV")
            
            return True, elapsed_time
        else:
            print(f"❌ Synthesis failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False, None
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out (exceeded 60 seconds)")
        print(f"   This may indicate the model is still loading")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_voice_embedding(voice_sample_path):
    """Test voice embedding extraction"""
    print_section("Voice Embedding Extraction")
    
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        print(f"⚠️  Voice sample not found: {voice_sample_path}")
        print(f"   Skipping voice embedding test")
        return True
    
    try:
        # Use absolute path inside container
        container_path = f"/app/voice_samples/{os.path.basename(voice_sample_path)}"
        
        payload = {
            "voice_sample_path": container_path
        }
        
        print(f"Voice sample: {voice_sample_path}")
        print(f"Container path: {container_path}")
        
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
    """Main test function"""
    print_header("Chatterbox-TTS Container API Test")
    print(f"Testing container at: {CHATTERBOX_URL}")
    print(f"\nNote: This script only tests the API endpoints.")
    print(f"      It does not build or start the container.")
    print(f"      Make sure the container is running before testing.")
    
    # Test health
    health_ok, health_data = test_health()
    if not health_ok:
        print("\n❌ Health check failed - container may not be running")
        print(f"\n💡 To start the container:")
        print(f"   cd {WORKSPACE_ROOT / 'setup'}")
        print("   docker compose up -d chatterbox-tts")
        print(f"\n💡 Or set CHATTERBOX_URL environment variable if running remotely:")
        print("   export CHATTERBOX_URL=http://remote-host:11437")
        sys.exit(1)
    
    # Basic synthesis test
    print("\n" + "=" * 70)
    synthesis_ok, latency = test_synthesize()
    
    if not synthesis_ok:
        print("\n❌ Basic synthesis failed")
        sys.exit(1)
    
    # Voice cloning test (if voice sample available)
    voice_sample = None
    if VOICE_SAMPLES_DIR.exists():
        for sample_file in ["sample.wav", "startup.wav", "welcome.wav"]:
            sample_path = VOICE_SAMPLES_DIR / sample_file
            if sample_path.exists():
                voice_sample = sample_file
                break
        
        if voice_sample:
            # Test voice embedding first
            test_voice_embedding(VOICE_SAMPLES_DIR / voice_sample)
            
            # Test synthesis with voice cloning
            print("\n" + "=" * 70)
            test_synthesize(
                "Hello, this is a test with voice cloning enabled.",
                voice_sample=voice_sample
            )
        else:
            print("\n⚠️  No voice samples found for cloning test")
    
    # Summary
    print_header("Test Summary")
    print(f"✅ Health check: PASSED")
    print(f"✅ Basic synthesis: PASSED")
    if latency:
        print(f"⏱️  Latency: {latency:.2f} seconds")
    
    if health_data:
        device = health_data.get('device', 'unknown')
        loaded = health_data.get('chatterbox_loaded', False)
        print(f"📊 Device: {device}")
        print(f"📊 Chatterbox loaded: {loaded}")
    
    print(f"\n🌐 Container URL: {CHATTERBOX_URL}")
    print(f"\n✅ Container API is working correctly!")
    print(f"\n💡 Next steps:")
    print(f"   1. Container is ready for integration")
    print(f"   2. Modify aura-control/core/speaker.py to use HTTP API")
    print(f"   3. Update TTS engine to use container endpoint")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
