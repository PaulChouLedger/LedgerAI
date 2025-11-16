"""
Porcupine Wake Word Detection Integration
Requires: pip install pvporcupine

For Jetson (ARM64), you may need to build from source:
  git clone https://github.com/Picovoice/porcupine
  cd porcupine/binding/python
  python setup.py build_ext --inplace

Access Key:
  Get free access key from: https://console.picovoice.ai/
  Set in .env: PORCUPINE_ACCESS_KEY=your_key_here
"""

import os
import numpy as np
from dotenv import load_dotenv

# Try to import Porcupine
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    pvporcupine = None


class PorcupineWakeWord:
    """
    Porcupine wake word detection wrapper for Aura
    
    Usage:
        wake_word = PorcupineWakeWord()
        if wake_word.initialize():
            detected, confidence = wake_word.process(audio_frame)
    """
    
    def __init__(self, keyword_path=None, sensitivity=None, access_key=None):
        """
        Initialize Porcupine wake word detector.
        
        Args:
            keyword_path: Path to .ppn model file (or None for built-in/default)
            sensitivity: Detection sensitivity (0.0-1.0, default from state)
            access_key: Picovoice access key (or None to load from .env)
        """
        # Load from state module (preferred) or use provided values
        try:
            from state import get_wake_word_model_path, get_wake_word_sensitivity
            self.keyword_path = keyword_path or get_wake_word_model_path()
            self.sensitivity = sensitivity if sensitivity is not None else get_wake_word_sensitivity()
        except ImportError:
            # Fallback if state module not available
            self.keyword_path = keyword_path
            self.sensitivity = sensitivity if sensitivity is not None else 0.5
        
        # Load access key from .env if not provided
        if access_key is None:
            # Load .env from workspace root (2 levels up from this file)
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            dotenv_path = os.path.join(workspace_root, '.env')
            load_dotenv(dotenv_path)
            access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        
        self.access_key = access_key
        self.porcupine = None
        self.is_active = False
        self.frame_length = None
        self.sample_rate = None
        
    def initialize(self):
        """
        Initialize Porcupine engine.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not PORCUPINE_AVAILABLE:
            print("[Wake Word] ❌ Porcupine not available - install with: pip install pvporcupine")
            print("[Wake Word] 💡 For Jetson ARM64, you may need to build from source")
            return False
        
        # Check for access key
        if not self.access_key:
            print("[Wake Word] ❌ Porcupine access key required!")
            print("[Wake Word] 💡 Get free access key from: https://console.picovoice.ai/")
            print("[Wake Word] 💡 Set in .env: PORCUPINE_ACCESS_KEY=your_key_here")
            return False
        
        try:
            # Check if custom model path is provided
            if self.keyword_path and os.path.exists(self.keyword_path):
                # Use custom model file
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.keyword_path],
                    sensitivities=[self.sensitivity]
                )
                print(f"[Wake Word] ✅ Porcupine initialized with custom model: {self.keyword_path}")
            else:
                # Try to use built-in keywords
                try:
                    # Check available keywords (KEYWORDS is a set, not a dict)
                    if hasattr(pvporcupine, 'KEYWORDS'):
                        # KEYWORDS might be a set or dict
                        if isinstance(pvporcupine.KEYWORDS, set):
                            available_keywords = pvporcupine.KEYWORDS
                        elif isinstance(pvporcupine.KEYWORDS, dict):
                            available_keywords = set(pvporcupine.KEYWORDS.keys())
                        else:
                            available_keywords = set(pvporcupine.KEYWORDS)
                    else:
                        # Fallback: try common keywords directly
                        available_keywords = set()
                    
                    # Try common wake word phrases
                    wake_phrases = ['hey aura', 'hey aura assistant', 'aura']
                    found_keyword = None
                    
                    for phrase in wake_phrases:
                        if phrase in available_keywords:
                            found_keyword = phrase
                            break
                    
                    if found_keyword:
                        self.porcupine = pvporcupine.create(
                            access_key=self.access_key,
                            keywords=[found_keyword],
                            sensitivities=[self.sensitivity]
                        )
                        print(f"[Wake Word] ✅ Porcupine initialized with built-in keyword: '{found_keyword}'")
                    else:
                        # No built-in "hey aura" found - try fallback keywords for testing
                        fallback_keywords = ['hey siri', 'hey google', 'computer', 'porcupine', 'picovoice']
                        fallback_found = None
                        
                        for keyword in fallback_keywords:
                            if keyword in available_keywords:
                                fallback_found = keyword
                                break
                        
                        if fallback_found:
                            print(f"[Wake Word] ⚠️  'hey aura' not found, using fallback: '{fallback_found}'")
                            print(f"[Wake Word] 💡 Train custom 'hey aura' model at: https://console.picovoice.ai/")
                            self.porcupine = pvporcupine.create(
                                access_key=self.access_key,
                                keywords=[fallback_found],
                                sensitivities=[self.sensitivity]
                            )
                            print(f"[Wake Word] ✅ Porcupine initialized with fallback keyword: '{fallback_found}'")
                        else:
                            # No built-in keyword found - need custom model
                            print("[Wake Word] ❌ No built-in 'hey aura' keyword found")
                            print(f"[Wake Word] 📋 Available keywords: {sorted(list(available_keywords))[:10]}...")
                            print("[Wake Word] 💡 Options:")
                            print("[Wake Word]    1. Train custom model at: https://console.picovoice.ai/")
                            print("[Wake Word]    2. Download .ppn file and set wake_word_model_path in app_settings.json")
                            return False
                        
                except Exception as e:
                    print(f"[Wake Word] ❌ Failed to initialize with built-in keywords: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[Wake Word] 💡 Train custom model at: https://console.picovoice.ai/")
                    return False
            
            # Get required frame length and sample rate
            self.frame_length = self.porcupine.frame_length
            self.sample_rate = self.porcupine.sample_rate
            
            print(f"[Wake Word]   Frame length: {self.frame_length} samples")
            print(f"[Wake Word]   Sample rate: {self.sample_rate} Hz")
            print(f"[Wake Word]   Sensitivity: {self.sensitivity}")
            
            self.is_active = True
            return True
            
        except Exception as e:
            print(f"[Wake Word] ❌ Failed to initialize Porcupine: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process(self, audio_frame):
        """
        Process audio frame for wake word detection.
        
        Args:
            audio_frame: numpy array of audio samples (int16, float32, or float64)
                        Must match frame_length (typically 512 samples at 16kHz)
            
        Returns:
            tuple: (detected: bool, confidence: float)
        """
        if not self.is_active or self.porcupine is None:
            return False, 0.0
        
        try:
            # Convert to int16 if needed (Porcupine requires int16 PCM)
            if audio_frame.dtype == 'float32' or audio_frame.dtype == 'float64':
                # Clamp to [-1, 1] range
                audio_frame = np.clip(audio_frame, -1.0, 1.0)
                # Convert to int16
                audio_frame = (audio_frame * 32767).astype('int16')
            elif audio_frame.dtype != 'int16':
                # Convert unknown types to int16
                audio_frame = audio_frame.astype('int16')
            
            # Ensure correct length (Porcupine requires exact frame_length)
            if len(audio_frame) != self.frame_length:
                # Pad or truncate to match required length
                if len(audio_frame) < self.frame_length:
                    # Pad with zeros
                    audio_frame = np.pad(
                        audio_frame, 
                        (0, self.frame_length - len(audio_frame)),
                        mode='constant'
                    )
                else:
                    # Truncate
                    audio_frame = audio_frame[:self.frame_length]
            
            # Process frame with Porcupine
            keyword_index = self.porcupine.process(audio_frame)
            
            if keyword_index >= 0:
                # Wake word detected!
                return True, 1.0
            return False, 0.0
            
        except Exception as e:
            print(f"[Wake Word] ⚠️ Processing error: {e}")
            return False, 0.0
    
    def release(self):
        """Release Porcupine resources"""
        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                print(f"[Wake Word] ⚠️ Error releasing Porcupine: {e}")
            self.porcupine = None
            self.is_active = False


def create_wake_word_detector():
    """
    Factory function to create and initialize wake word detector.
    
    Returns:
        PorcupineWakeWord instance if successful, None otherwise
    """
    # Check if wake word is enabled (from state module)
    try:
        from state import get_wake_word_enabled
        enable_wake_word = get_wake_word_enabled()
    except ImportError:
        # Fallback: wake word disabled if state module not available
        enable_wake_word = False
    
    if not enable_wake_word:
        print("[Wake Word] ℹ️  Wake word detection disabled (toggle in Settings)")
        return None
    
    if not PORCUPINE_AVAILABLE:
        print("[Wake Word] ⚠️  Porcupine not installed - wake word detection disabled")
        print("[Wake Word] 💡 Install with: pip install pvporcupine")
        return None
    
    # Create detector
    detector = PorcupineWakeWord()
    
    # Initialize
    if detector.initialize():
        return detector
    else:
        print("[Wake Word] ⚠️  Initialization failed - wake word detection disabled")
        return None

