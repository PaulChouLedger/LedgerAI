"""
OpenWakeWord Wake Word Detection Integration

Installation:
  OpenWakeWord is a lightweight, open-source wake word detection framework
  that works natively on ARM64 (Jetson) without any manual setup.
  
  Installation:
    pip install openwakeword
  
  That's it! No API keys, no manual library downloads, no build from source.
  
  Custom Training:
    Train custom wake words with minimal data (~100-200 samples):
    https://github.com/dscripka/openWakeWord#training-custom-models
  
  Pre-trained Models:
    OpenWakeWord comes with several pre-trained models:
    - 'hey_jarvis' (recommended fallback)
    - 'hey_mycroft'
    - 'hey_fire_fox'
    - 'timer'
    - 'weather'
    - And more...
  
  GitHub: https://github.com/dscripka/openWakeWord
"""

import os
import numpy as np
from dotenv import load_dotenv

# Try to import OpenWakeWord
try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    Model = None


class OpenWakeWordDetector:
    """
    OpenWakeWord wake word detection wrapper for Aura
    
    Usage:
        wake_word = OpenWakeWordDetector()
        if wake_word.initialize():
            detected, confidence = wake_word.process(audio_frame)
    """
    
    def __init__(self, model_path=None, threshold=None):
        """
        Initialize OpenWakeWord detector.
        
        Args:
            model_path: Path to custom .onnx model file (or None for pre-trained)
            threshold: Detection threshold (0.0-1.0, default from state)
        """
        # Load from state module (preferred) or use provided values
        try:
            from state import get_wake_word_sensitivity
            # OpenWakeWord uses threshold (higher = more sensitive)
            # Convert sensitivity (0.0-1.0) to threshold (0.0-1.0)
            # Higher sensitivity = lower threshold (more sensitive)
            sensitivity = threshold if threshold is not None else get_wake_word_sensitivity()
            self.threshold = 1.0 - sensitivity if sensitivity is not None else 0.5
        except ImportError:
            # Fallback if state module not available
            self.threshold = threshold if threshold is not None else 0.5
        
        self.model_path = model_path
        self.model = None
        self.is_active = False
        self.frame_length = 1280  # OpenWakeWord uses 1280 samples at 16kHz (80ms)
        self.sample_rate = 16000
        self.wake_word_name = None
        self.available_models = []
        
    def initialize(self):
        """
        Initialize OpenWakeWord engine.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not OPENWAKEWORD_AVAILABLE:
            print("[Wake Word] ❌ OpenWakeWord not available - install with: pip install openwakeword")
            print("[Wake Word] 💡 OpenWakeWord works natively on ARM64 (Jetson) - no manual setup needed!")
            return False
        
        try:
            # Try to use custom model if provided
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model
                self.model = Model(
                    wakeword_models=[self.model_path],
                    inference_framework='onnx'
                )
                self.wake_word_name = os.path.basename(self.model_path).replace('.onnx', '')
                print(f"[Wake Word] ✅ OpenWakeWord initialized with custom model: {self.model_path}")
            else:
                # Use pre-trained models
                # Try to find "hey_aura" or similar, fallback to "hey_jarvis"
                preferred_models = ['hey_aura', 'hey_jarvis', 'hey_mycroft', 'hey_fire_fox']
                
                # Get available models from OpenWakeWord
                # OpenWakeWord has built-in models, we'll use 'hey_jarvis' as default
                # since 'hey_aura' likely doesn't exist as a pre-trained model
                self.model = Model(
                    wakeword_models=['hey_jarvis'],  # Default fallback
                    inference_framework='onnx'
                )
                self.wake_word_name = 'hey_jarvis'
                
                # Check what models are available
                # OpenWakeWord loads models from its package directory
                print(f"[Wake Word] ✅ OpenWakeWord initialized with pre-trained model: '{self.wake_word_name}'")
                print(f"[Wake Word] 💡 To use custom 'hey aura' model:")
                print(f"[Wake Word]     1. Train model: https://github.com/dscripka/openWakeWord#training-custom-models")
                print(f"[Wake Word]     2. Set wake_word_model_path in app_settings.json")
            
            # Get frame length from model (OpenWakeWord uses 1280 samples at 16kHz)
            # This is 80ms of audio
            self.frame_length = 1280
            self.sample_rate = 16000
            
            print(f"[Wake Word]   Frame length: {self.frame_length} samples ({self.frame_length/self.sample_rate*1000:.0f}ms)")
            print(f"[Wake Word]   Sample rate: {self.sample_rate} Hz")
            print(f"[Wake Word]   Threshold: {self.threshold:.2f} (lower = more sensitive)")
            
            self.is_active = True
            return True
            
        except Exception as e:
            print(f"[Wake Word] ❌ Failed to initialize OpenWakeWord: {e}")
            import traceback
            traceback.print_exc()
            print("[Wake Word] 💡 Install with: pip install openwakeword")
            print("[Wake Word] 💡 OpenWakeWord works natively on ARM64 (Jetson) - no manual setup needed!")
            return False
    
    def process(self, audio_frame):
        """
        Process audio frame for wake word detection.
        
        Args:
            audio_frame: numpy array of audio samples (float32, shape: [samples])
                        Should be 1280 samples at 16kHz for optimal performance
                        Can be any length - will be padded/truncated automatically
            
        Returns:
            tuple: (detected: bool, confidence: float)
        """
        if not self.is_active or self.model is None:
            return False, 0.0
        
        try:
            # Ensure float32 format
            if audio_frame.dtype != 'float32':
                audio_frame = audio_frame.astype('float32')
            
            # Normalize to [-1, 1] range if needed
            if audio_frame.max() > 1.0 or audio_frame.min() < -1.0:
                # Assume int16 format, convert to float32
                if audio_frame.dtype == 'int16' or audio_frame.max() > 1.0:
                    audio_frame = audio_frame.astype('float32') / 32768.0
                    audio_frame = np.clip(audio_frame, -1.0, 1.0)
            
            # Ensure correct length (OpenWakeWord expects 1280 samples)
            if len(audio_frame) != self.frame_length:
                if len(audio_frame) < self.frame_length:
                    # Pad with zeros
                    audio_frame = np.pad(
                        audio_frame, 
                        (0, self.frame_length - len(audio_frame)),
                        mode='constant'
                    )
                else:
                    # Truncate or take last N samples
                    audio_frame = audio_frame[-self.frame_length:]
            
            # Reshape for OpenWakeWord (expects [1, samples] shape)
            audio_frame = audio_frame.reshape(1, -1)
            
            # Process frame with OpenWakeWord
            # predict() returns a dict with model names as keys and confidence scores as values
            predictions = self.model.predict(audio_frame)
            
            # Get confidence for our wake word model
            if self.wake_word_name and self.wake_word_name in predictions:
                confidence = predictions[self.wake_word_name]
            elif len(predictions) > 0:
                # Use first (and likely only) model's confidence
                confidence = list(predictions.values())[0]
            else:
                confidence = 0.0
            
            # Check if confidence exceeds threshold
            # Lower threshold = more sensitive (detects more easily)
            detected = confidence >= self.threshold
            
            return detected, confidence
            
        except Exception as e:
            print(f"[Wake Word] ⚠️ Processing error: {e}")
            return False, 0.0
    
    def release(self):
        """Release OpenWakeWord resources"""
        if self.model:
            try:
                # OpenWakeWord models don't need explicit cleanup, but we'll clear the reference
                self.model = None
            except Exception as e:
                print(f"[Wake Word] ⚠️ Error releasing OpenWakeWord: {e}")
            self.is_active = False


def create_wake_word_detector():
    """
    Factory function to create and initialize wake word detector.
    
    Returns:
        OpenWakeWordDetector instance if successful, None otherwise
    """
    # Check if wake word is enabled (from state module)
    try:
        from state import get_wake_word_enabled, get_wake_word_model_path
        enable_wake_word = get_wake_word_enabled()
        model_path = get_wake_word_model_path()
    except ImportError:
        # Fallback: wake word disabled if state module not available
        enable_wake_word = False
        model_path = None
    
    if not enable_wake_word:
        print("[Wake Word] ℹ️  Wake word detection disabled (toggle in Settings)")
        return None
    
    if not OPENWAKEWORD_AVAILABLE:
        print("[Wake Word] ⚠️  OpenWakeWord not installed - wake word detection disabled")
        print("[Wake Word] 💡 Install with: pip install openwakeword")
        print("[Wake Word] 💡 OpenWakeWord works natively on ARM64 (Jetson) - no manual setup needed!")
        return None
    
    # Create detector
    detector = OpenWakeWordDetector(model_path=model_path)
    
    # Initialize
    if detector.initialize():
        return detector
    else:
        print("[Wake Word] ⚠️  Initialization failed - wake word detection disabled")
        return None
