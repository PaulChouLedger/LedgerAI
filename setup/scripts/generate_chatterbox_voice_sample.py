#!/usr/bin/env python3
"""
Generate a voice sample from ElevenLabs for ChatterboxTTS voice cloning.

This script:
1. Generates a high-quality voice sample using ElevenLabs
2. Saves it in the correct format for ChatterboxTTS
3. Places it in the default voice samples directory

Usage:
    python setup/scripts/generate_chatterbox_voice_sample.py
"""

import os
import sys
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Load environment variables
dotenv_path = workspace_root / '.env'
load_dotenv(dotenv_path)

def generate_voice_sample():
    """Generate a voice sample from ElevenLabs for ChatterboxTTS"""
    
    # Check for ElevenLabs API key
    api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    if not api_key or api_key == "your_elevenlabs_api_key_here":
        print("❌ Error: ELEVENLABS_API_KEY not set in .env file")
        print("   Run: ./aura_config.sh and configure TTS (option 5)")
        return False
    
    # Get voice ID
    voice_id = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"
    
    try:
        from elevenlabs.client import ElevenLabs
        from pydub import AudioSegment
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Install with: pip install elevenlabs pydub")
        return False
    
    # Initialize ElevenLabs client
    try:
        client = ElevenLabs(api_key=api_key)
        print(f"✅ Connected to ElevenLabs (voice: {voice_id})")
    except Exception as e:
        print(f"❌ Failed to connect to ElevenLabs: {e}")
        return False
    
    # Text for voice sample (at least 5 seconds, preferably 10-20 seconds)
    sample_text = (
        "Hello, this is a voice sample for ChatterboxTTS voice cloning. "
        "This sample contains natural speech with varied intonation and emotion. "
        "The voice should sound clear and expressive, suitable for text to speech synthesis. "
        "This sample is designed to capture the unique characteristics of this voice."
    )
    
    print(f"\n📝 Generating voice sample...")
    print(f"   Text length: {len(sample_text)} characters")
    print(f"   Estimated duration: ~{len(sample_text.split()) / 2.5:.1f} seconds")
    
    # Output directory
    output_dir = workspace_root / "assets" / "voice_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample.wav"
    
    try:
        # Generate audio from ElevenLabs
        print(f"\n🎙️ Generating audio from ElevenLabs...")
        stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=sample_text,
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.75,  # Higher for better voice cloning
                "style": 0.5,
                "use_speaker_boost": True
            }
        )
        
        # Collect audio bytes
        audio_bytes = b"".join(stream)
        
        # Convert to WAV format
        print(f"🔄 Converting to WAV format...")
        audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
        
        # Ensure mono, 44.1kHz (ChatterboxTTS compatible)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 44100:
            audio = audio.set_frame_rate(44100)
        
        # Normalize audio levels
        audio = audio.normalize()
        
        # Export as WAV
        audio.export(str(output_path), format="wav")
        
        # Verify file
        if output_path.exists():
            file_size = output_path.stat().st_size
            duration = len(audio) / 1000.0  # Convert to seconds
            print(f"\n✅ Voice sample generated successfully!")
            print(f"   Location: {output_path}")
            print(f"   Size: {file_size / 1024:.1f} KB")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Sample rate: {audio.frame_rate} Hz")
            print(f"   Channels: {audio.channels} (mono)")
            
            if duration < 5:
                print(f"\n⚠️  Warning: Sample is less than 5 seconds ({duration:.1f}s)")
                print(f"   ChatterboxTTS works better with samples 10-20 seconds long")
            else:
                print(f"\n✅ Sample duration is good for voice cloning!")
            
            print(f"\n📋 Next steps:")
            print(f"   1. Enable ChatterboxTTS in Settings → TTS Engine")
            print(f"   2. The voice sample will be automatically used for cloning")
            print(f"   3. Test by asking AuraVision a question")
            
            return True
        else:
            print(f"❌ Error: File was not created at {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating voice sample: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ChatterboxTTS Voice Sample Generator")
    print("=" * 60)
    print()
    
    success = generate_voice_sample()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Voice sample generation complete!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Voice sample generation failed")
        print("=" * 60)
        sys.exit(1)

