#!/usr/bin/env python3
"""
Generate Pre-recorded Audio Samples for Thinking Fillers

This script:
1. Reads all filler phrases from thinking_fillers.py
2. Generates TTS audio for each phrase using the LLM container's TTS
3. Saves audio files as {id}.wav
4. Creates a manifest.json mapping text to audio files

IMPORTANT: Audio and text MUST be synchronized
- Chatbot will use text
- Voice will use pre-recorded audio
- Both systems output the SAME message
"""

import sys
import os
import json
import requests
from pathlib import Path

# Add parent directory to path to import thinking_fillers
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm-container'))
from thinking_fillers import get_all_fillers, FILLER_AUDIO_DIR

# TTS API endpoint (adjust if needed)
TTS_API_URL = "http://localhost:5001/tts"

def generate_filler_audio():
    """Generate all filler audio samples"""
    
    # Create output directory
    FILLER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[FillerGen] 📁 Output directory: {FILLER_AUDIO_DIR}")
    
    # Get all unique fillers
    all_fillers = get_all_fillers()
    print(f"[FillerGen] 📋 Found {len(all_fillers)} unique fillers to generate")
    
    # Generate manifest
    manifest = {
        'description': 'Pre-recorded thinking filler audio samples',
        'total_fillers': len(all_fillers),
        'fillers': []
    }
    
    success_count = 0
    fail_count = 0
    
    for filler in all_fillers:
        filler_id = filler['id']
        text = filler['text']
        output_path = FILLER_AUDIO_DIR / f"{filler_id}.wav"
        
        print(f"\n[FillerGen] 🎙️ Generating: [{filler_id}]")
        print(f"[FillerGen]    Text: \"{text}\"")
        
        try:
            # Call TTS API
            response = requests.post(
                TTS_API_URL,
                json={'text': text},
                timeout=30
            )
            
            if response.status_code == 200:
                # Save audio file
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = output_path.stat().st_size
                print(f"[FillerGen]    ✅ Saved: {output_path} ({file_size} bytes)")
                
                # Add to manifest
                manifest['fillers'].append({
                    'id': filler_id,
                    'text': text,
                    'audio_file': f"{filler_id}.wav",
                    'file_size': file_size
                })
                
                success_count += 1
            else:
                print(f"[FillerGen]    ❌ TTS API error: {response.status_code}")
                print(f"[FillerGen]       {response.text}")
                fail_count += 1
        
        except requests.exceptions.ConnectionError:
            print(f"[FillerGen]    ❌ Cannot connect to TTS API at {TTS_API_URL}")
            print(f"[FillerGen]       Make sure the LLM container is running!")
            fail_count += 1
        
        except Exception as e:
            print(f"[FillerGen]    ❌ Error: {e}")
            fail_count += 1
    
    # Save manifest
    manifest_path = FILLER_AUDIO_DIR / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[FillerGen] 📊 SUMMARY")
    print(f"{'='*80}")
    print(f"[FillerGen] ✅ Success: {success_count}/{len(all_fillers)}")
    print(f"[FillerGen] ❌ Failed:  {fail_count}/{len(all_fillers)}")
    print(f"[FillerGen] 📁 Output:   {FILLER_AUDIO_DIR}")
    print(f"[FillerGen] 📄 Manifest: {manifest_path}")
    print(f"{'='*80}\n")
    
    if success_count == len(all_fillers):
        print("[FillerGen] 🎉 All filler audio samples generated successfully!")
        return 0
    elif success_count > 0:
        print("[FillerGen] ⚠️ Some fillers generated, but some failed")
        return 1
    else:
        print("[FillerGen] ❌ No audio samples were generated")
        return 2


if __name__ == "__main__":
    print("="*80)
    print("THINKING FILLER AUDIO GENERATOR")
    print("="*80)
    print("This script generates pre-recorded audio for all filler phrases.")
    print("Make sure the LLM container is running on port 5001!")
    print("="*80 + "\n")
    
    exit_code = generate_filler_audio()
    sys.exit(exit_code)

