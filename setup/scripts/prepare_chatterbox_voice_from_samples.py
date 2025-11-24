#!/usr/bin/env python3
"""
Prepare a voice sample for ChatterboxTTS from multiple ElevenLabs WAV files.

This script:
1. Analyzes your collection of WAV samples
2. Selects the best quality sample(s)
3. Optionally combines multiple samples for better voice cloning
4. Prepares the final sample for ChatterboxTTS caching

Usage:
    python setup/scripts/prepare_chatterbox_voice_from_samples.py [samples_directory]
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

try:
    import soundfile as sf
    import librosa
except ImportError:
    print("❌ Missing required packages: soundfile, librosa")
    print("   Install with: pip install soundfile librosa")
    sys.exit(1)

def analyze_audio_quality(file_path):
    """
    Analyze audio file quality and return metrics.
    Returns dict with quality scores.
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Calculate metrics
        duration = len(audio) / sr
        
        # Signal-to-noise ratio (simplified - higher is better)
        # Use RMS energy as proxy for signal strength
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # Dynamic range (difference between max and min)
        dynamic_range = np.max(audio) - np.min(audio)
        
        # Zero crossing rate (speech should have moderate ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        # Spectral centroid (voice should be in mid-range)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        
        # Quality score (higher is better)
        # Prefer: longer duration, higher energy, good dynamic range, moderate ZCR
        quality_score = (
            duration * 10 +  # Duration is important (5+ seconds ideal)
            rms_energy * 100 +  # Higher energy = better signal
            dynamic_range * 50 +  # Good dynamic range = clear audio
            (1.0 - abs(zcr - 0.1)) * 20  # ZCR around 0.1 is good for speech
        )
        
        return {
            'file': file_path,
            'duration': duration,
            'rms_energy': rms_energy,
            'dynamic_range': dynamic_range,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'quality_score': quality_score,
            'sample_rate': sr
        }
    except Exception as e:
        print(f"⚠️  Error analyzing {file_path}: {e}")
        return None

def find_best_samples(samples_dir, max_samples=10):
    """
    Find the best quality samples from a directory.
    Returns list of best samples sorted by quality.
    """
    print(f"🔍 Scanning {samples_dir} for WAV files...")
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(samples_dir, "*.wav"))
    wav_files.extend(glob.glob(os.path.join(samples_dir, "**", "*.wav"), recursive=True))
    
    if not wav_files:
        print(f"❌ No WAV files found in {samples_dir}")
        return []
    
    print(f"📊 Found {len(wav_files)} WAV files")
    print(f"🔧 Analyzing audio quality...")
    
    # Analyze all files
    results = []
    for i, wav_file in enumerate(wav_files):
        if (i + 1) % 100 == 0:
            print(f"   Analyzed {i + 1}/{len(wav_files)} files...")
        
        analysis = analyze_audio_quality(wav_file)
        if analysis:
            results.append(analysis)
    
    # Sort by quality score (highest first)
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"✅ Analysis complete!")
    print(f"\n📈 Top {min(max_samples, len(results))} samples:")
    for i, result in enumerate(results[:max_samples], 1):
        print(f"   {i}. {os.path.basename(result['file'])}")
        print(f"      Duration: {result['duration']:.2f}s, Quality: {result['quality_score']:.1f}")
    
    return results[:max_samples]

def combine_samples(sample_files, output_path, target_duration=15.0):
    """
    Combine multiple audio samples into one optimal file.
    Selects best segments and combines them.
    """
    print(f"\n🔧 Combining {len(sample_files)} samples...")
    
    combined_audio = []
    total_duration = 0.0
    
    for sample_file in sample_files:
        try:
            audio, sr = librosa.load(sample_file, sr=22050, mono=True)
            duration = len(audio) / sr
            
            # Add this sample
            combined_audio.append(audio)
            total_duration += duration
            
            # Stop if we have enough
            if total_duration >= target_duration:
                break
        except Exception as e:
            print(f"⚠️  Error loading {sample_file}: {e}")
            continue
    
    if not combined_audio:
        print("❌ No audio samples could be loaded")
        return False
    
    # Concatenate all samples
    final_audio = np.concatenate(combined_audio)
    
    # Normalize
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val * 0.95  # Leave headroom
    
    # Ensure minimum duration
    if len(final_audio) / sr < 5.0:
        print(f"⚠️  Combined sample is only {len(final_audio) / sr:.2f}s, padding...")
        # Pad with silence to reach 5 seconds minimum
        silence = np.zeros(int(sr * (5.0 - len(final_audio) / sr)))
        final_audio = np.concatenate([final_audio, silence])
    
    # Save
    sf.write(output_path, final_audio, sr)
    print(f"✅ Combined sample saved: {output_path}")
    print(f"   Duration: {len(final_audio) / sr:.2f}s, Sample rate: {sr}Hz")
    
    return True

def prepare_single_best_sample(best_sample, output_path):
    """
    Prepare the single best sample for ChatterboxTTS.
    """
    print(f"\n🎯 Using best single sample: {os.path.basename(best_sample['file'])}")
    
    try:
        # Load and convert to standard format
        audio, sr = librosa.load(best_sample['file'], sr=22050, mono=True)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Ensure minimum duration (pad if needed)
        min_duration = 5.0
        if len(audio) / sr < min_duration:
            print(f"⚠️  Sample is only {len(audio) / sr:.2f}s, padding to {min_duration}s...")
            silence = np.zeros(int(sr * (min_duration - len(audio) / sr)))
            audio = np.concatenate([audio, silence])
        
        # Save
        sf.write(output_path, audio, sr)
        print(f"✅ Sample prepared: {output_path}")
        print(f"   Duration: {len(audio) / sr:.2f}s, Sample rate: {sr}Hz")
        
        return True
    except Exception as e:
        print(f"❌ Error preparing sample: {e}")
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("ChatterboxTTS Voice Sample Preparation")
    print("=" * 70)
    print()
    
    # Get samples directory
    if len(sys.argv) > 1:
        samples_dir = sys.argv[1]
    else:
        samples_dir = input("Enter path to directory with ElevenLabs WAV samples: ").strip()
    
    if not os.path.isdir(samples_dir):
        print(f"❌ Error: {samples_dir} is not a valid directory")
        sys.exit(1)
    
    # Find best samples
    best_samples = find_best_samples(samples_dir, max_samples=10)
    
    if not best_samples:
        print("❌ No valid samples found")
        sys.exit(1)
    
    # Output directory
    output_dir = workspace_root / "assets" / "voice_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ask user preference
    print("\n" + "=" * 70)
    print("Select preparation method:")
    print("1. Use single best sample (fastest, recommended)")
    print("2. Combine top 3-5 samples (better quality, longer processing)")
    print("=" * 70)
    
    choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
    
    output_path = output_dir / "sample.wav"
    
    if choice == "2":
        # Combine multiple samples
        num_samples = min(5, len(best_samples))
        sample_files = [s['file'] for s in best_samples[:num_samples]]
        success = combine_samples(sample_files, output_path, target_duration=15.0)
    else:
        # Use single best sample
        success = prepare_single_best_sample(best_samples[0], output_path)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ Voice sample prepared successfully!")
        print("=" * 70)
        print(f"\n📁 Sample location: {output_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Enable ChatterboxTTS in Settings → TTS Engine")
        print(f"   2. Enable Voice Cloning toggle (if not already enabled)")
        print(f"   3. The voice embedding will be cached automatically on first use")
        print(f"   4. Test by asking AuraVision a question")
        print(f"\n💡 The cached voice embedding will be stored in:")
        print(f"   {workspace_root / 'data' / 'voice_cache'}")
        print(f"\n🎉 You're all set! The voice will be cloned and cached automatically.")
    else:
        print("\n❌ Failed to prepare voice sample")
        sys.exit(1)

if __name__ == "__main__":
    main()

