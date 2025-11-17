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
            # OpenWakeWord uses threshold (lower = more sensitive)
            # Convert sensitivity (0.0-1.0) to threshold (0.0-1.0)
            # Higher sensitivity = lower threshold (more sensitive)
            sensitivity = threshold if threshold is not None else get_wake_word_sensitivity()
            # OpenWakeWord confidence values are typically VERY low
            # Based on observed values: 0.0 for silence, 1e-06 to 1e-05 for background, 
            # spikes to 0.1-1.0 when wake word is detected
            # Default sensitivity 0.5 should map to threshold 0.00001 (extremely sensitive)
            # Sensitivity 0.0 (least sensitive) -> threshold 0.0001 (higher threshold)
            # Sensitivity 1.0 (most sensitive) -> threshold 0.000001 (very low threshold)
            if sensitivity is not None:
                # Map sensitivity to threshold range 0.00001 to 0.001
                # OpenWakeWord values are very low, but we need to avoid false positives
                # Default sensitivity 0.5 should map to threshold 0.0005 (balanced)
                self.threshold = 0.001 - (sensitivity * 0.00099)  # Maps 0.0->0.001, 0.5->0.0005, 1.0->0.00001
            else:
                self.threshold = 0.0005  # Balanced default for OpenWakeWord (reduces false positives)
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
                # First, try to download models if they're missing
                try:
                    from openwakeword.utils import download_models
                    print("[Wake Word] 📥 Checking for model files...")
                    download_models()
                    print("[Wake Word] ✅ Models downloaded/verified")
                except Exception as download_error:
                    print(f"[Wake Word] ⚠️  Could not download models automatically: {download_error}")
                    print("[Wake Word] 💡 You may need to download models manually:")
                    print("[Wake Word]     python -c 'import openwakeword; openwakeword.utils.download_models()'")
                
                # Try to use pre-trained models
                # OpenWakeWord has built-in models, we'll use 'hey_jarvis' as default
                preferred_models = ['hey_jarvis', 'hey_mycroft', 'hey_fire_fox']
                
                # Try each model until one works
                model_initialized = False
                for model_name in preferred_models:
                    try:
                        self.model = Model(
                            wakeword_models=[model_name],
                            inference_framework='onnx'
                        )
                        self.wake_word_name = model_name
                        model_initialized = True
                        print(f"[Wake Word] ✅ OpenWakeWord initialized with pre-trained model: '{self.wake_word_name}'")
                        break
                    except Exception as model_error:
                        if "NO_SUCHFILE" in str(model_error) or "File doesn't exist" in str(model_error):
                            # Try next model
                            continue
                        else:
                            # Different error, re-raise
                            raise
                
                if not model_initialized:
                    raise Exception("No pre-trained models available. Please download models first.")
                
                print(f"[Wake Word] 💡 To use custom 'hey aura' model:")
                print(f"[Wake Word]     1. Train model: https://github.com/dscripka/openWakeWord#training-custom-models")
                print(f"[Wake Word]     2. Set wake_word_model_path in app_settings.json")
            
            # Get frame length from model (OpenWakeWord uses 1280 samples at 16kHz)
            # This is 80ms of audio
            self.frame_length = 1280
            self.sample_rate = 16000
            
            print(f"[Wake Word]   Frame length: {self.frame_length} samples ({self.frame_length/self.sample_rate*1000:.0f}ms)")
            print(f"[Wake Word]   Sample rate: {self.sample_rate} Hz")
            print(f"[Wake Word]   Threshold: {self.threshold:.6f} (lower = more sensitive)")
            print(f"[Wake Word]   Note: OpenWakeWord uses very low confidence values (typically 0.0-0.01)")
            
            self.is_active = True
            return True
            
        except Exception as e:
            error_str = str(e)
            print(f"[Wake Word] ❌ Failed to initialize OpenWakeWord: {e}")
            
            # Check if it's a missing model file error
            if "NO_SUCHFILE" in error_str or "File doesn't exist" in error_str:
                print("[Wake Word] 💡 Models are missing. Download them with:")
                print("[Wake Word]     python -c 'import openwakeword; openwakeword.utils.download_models()'")
                print("[Wake Word] 💡 Or in Python:")
                print("[Wake Word]     from openwakeword.utils import download_models")
                print("[Wake Word]     download_models()")
            else:
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
            # Debug: log audio stats on first few calls
            if not hasattr(self, '_audio_debug_count'):
                self._audio_debug_count = 0
            
            self._audio_debug_count += 1
            if self._audio_debug_count <= 5:
                print(f"[Wake Word] 🔍 Audio frame stats: shape={audio_frame.shape}, dtype={audio_frame.dtype}, "
                      f"min={audio_frame.min():.4f}, max={audio_frame.max():.4f}, "
                      f"mean={np.abs(audio_frame).mean():.4f}, rms={np.sqrt(np.mean(audio_frame**2)):.4f}")
            
            # Ensure float32 format
            if audio_frame.dtype != 'float32':
                audio_frame = audio_frame.astype('float32')
            
            # Normalize to [-1, 1] range if needed
            # Check if audio is already normalized or needs conversion
            abs_max = np.abs(audio_frame).max()
            if abs_max > 1.0:
                # Audio is likely in int16 or other integer format, normalize
                if abs_max > 32767:
                    # Very large values, might be int32 or other
                    audio_frame = audio_frame.astype('float32') / abs_max
                else:
                    # Likely int16 range
                    audio_frame = audio_frame.astype('float32') / 32768.0
                audio_frame = np.clip(audio_frame, -1.0, 1.0)
            elif abs_max < 0.01:
                # Very quiet audio - might be an issue
                if self._audio_debug_count <= 10:
                    print(f"[Wake Word] ⚠️  Very quiet audio detected: max={abs_max:.6f}")
            
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
            
            # OpenWakeWord expects 1D array (1280,) not 2D
            # Ensure it's 1D
            if audio_frame.ndim > 1:
                audio_frame = audio_frame.flatten()
            
            # Ensure it's exactly 1280 samples
            if len(audio_frame) != self.frame_length:
                if len(audio_frame) < self.frame_length:
                    audio_frame = np.pad(audio_frame, (0, self.frame_length - len(audio_frame)), mode='constant')
                else:
                    audio_frame = audio_frame[:self.frame_length]
            
            # Process frame with OpenWakeWord
            # OpenWakeWord's predict() might expect 2D array with batch dimension
            # Try both formats to see which works
            try:
                # First try: 1D array (1280,)
                predictions = self.model.predict(audio_frame)
            except Exception as e1:
                # If that fails, try 2D with batch dimension (1, 1280)
                try:
                    audio_2d = audio_frame.reshape(1, -1)
                    predictions = self.model.predict(audio_2d)
                    if self._audio_debug_count <= 3:
                        print(f"[Wake Word] 🔍 Using 2D format (1, 1280) - 1D failed: {e1}")
                except Exception as e2:
                    print(f"[Wake Word] ❌ Both formats failed - 1D: {e1}, 2D: {e2}")
                    return False, 0.0
            
            # Debug: print available keys and raw predictions on first call
            if not hasattr(self, '_printed_keys'):
                print(f"[Wake Word] 🔍 Available prediction keys: {list(predictions.keys())}")
                print(f"[Wake Word] 🔍 Looking for: '{self.wake_word_name}'")
                print(f"[Wake Word] 🔍 Raw predictions: {predictions}")
                self._printed_keys = True
            
            # Debug: show raw prediction values occasionally
            if self._audio_debug_count <= 10 or (self._audio_debug_count % 200 == 0):
                print(f"[Wake Word] 🔍 Raw predictions: {predictions}")
            
            # Get confidence for our wake word model
            # Try exact match first
            confidence = 0.0
            if self.wake_word_name and self.wake_word_name in predictions:
                confidence = predictions[self.wake_word_name]
            else:
                # Try partial match (e.g., 'hey_jarvis' might be 'hey_jarvis_v0.1' in predictions)
                for key in predictions.keys():
                    if self.wake_word_name in key or key.startswith(self.wake_word_name):
                        confidence = predictions[key]
                        break
                
                # If still no match, use first available prediction
                if confidence == 0.0 and len(predictions) > 0:
                    confidence = list(predictions.values())[0]
                    if not hasattr(self, '_warned_key_mismatch'):
                        print(f"[Wake Word] ⚠️  Model name '{self.wake_word_name}' not found in predictions")
                        print(f"[Wake Word] ⚠️  Using first available: {list(predictions.keys())[0]} = {confidence:.3f}")
                        self._warned_key_mismatch = True
            
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
