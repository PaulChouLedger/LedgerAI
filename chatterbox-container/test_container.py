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

def test_synthesize(text="Hello, this is a test of Chatterbox TTS.", voice_sample=None):
    """Test synthesis endpoint"""
    print(f"\nTesting synthesis endpoint...")
    print(f"   Text: '{text}'")
    
    try:
        payload = {
            "text": text,
            "exaggeration": 0.6
        }
        if voice_sample:
            payload["voice_sample"] = voice_sample
        
        response = requests.post(
            f"{CHATTERBOX_URL}/synthesize",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            # Save audio file
            output_file = "test_output.wav"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Synthesis successful")
            print(f"   Audio saved to: {output_file}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Synthesis failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    if not test_synthesize():
        print("\n❌ Synthesis test failed")
        sys.exit(1)
    
    # Test with voice cloning if sample exists
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    voice_sample = os.path.join(workspace_root, "assets", "voice_samples", "sample.wav")
    if os.path.exists(voice_sample):
        test_voice_embedding(voice_sample)
        test_synthesize("Hello, this is a test with voice cloning.", "sample.wav")
    else:
        print(f"\n⚠️  Voice sample not found at: {voice_sample}")
        print(f"   Skipping voice cloning tests")
    
    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)
    print("\n💡 To use Chatterbox-TTS in your code:")
    print(f"   import requests")
    print(f"   response = requests.post('{CHATTERBOX_URL}/synthesize', json={{'text': 'Hello'}})")
    print(f"   with open('output.wav', 'wb') as f:")
    print(f"       f.write(response.content)")

if __name__ == '__main__':
    main()

