"""
OpenWakeWord Wake Word Detection Integration

OpenWakeWord is an actively maintained, open-source wake word detection framework
that works natively on ARM64 (Jetson) devices. It offers good performance and
supports custom model training.

Installation:
    pip install openwakeword

GitHub: https://github.com/dscripka/openWakeWord
"""

import os
import numpy as np
from typing import Optional, Tuple

# Import shared audio processing constants from listener
try:
    import sys
    import os
    listener_path = os.path.join(os.path.dirname(__file__), 'listener.py')
    if os.path.exists(listener_path):
        from listener import (
            SAMPLE_RATE, FRAME_SIZE, MICROPHONE_CHANNEL,
            TARGET_RMS_FOR_WHISPER, ENABLE_AUDIO_NORMALIZATION
        )
        WAKE_WORD_SAMPLE_RATE = SAMPLE_RATE
        WAKE_WORD_FRAME_SIZE = FRAME_SIZE
    else:
        WAKE_WORD_SAMPLE_RATE = 16000
        WAKE_WORD_FRAME_SIZE = 512
except ImportError:
    WAKE_WORD_SAMPLE_RATE = 16000
    WAKE_WORD_FRAME_SIZE = 512

# Default threshold (0.0-1.0, higher = less sensitive)
DEFAULT_THRESHOLD = 0.1

# Wake word model name (can be customized)
# Will automatically find variations like "hey_orah-2.onnx" when looking for "hey_orah"
DEFAULT_MODEL = "hey_orah"  # Default model, can be changed to custom model


class OpenWakeWordDetector:
    """
    OpenWakeWord-based wake word detector.
    
    Uses direct Python integration (not Wyoming container) for better reliability.
    """
    
    def __init__(self, model_name: str = None, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize OpenWakeWord detector.
        
        Args:
            model_name: Name of the wake word model (default: "hey_mycroft_v0.1")
            threshold: Detection threshold (0.0-1.0, default: 0.5)
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.threshold = threshold
        self.engine = None
        self.is_active = False
        self.frame_count = 0
        
        # OpenWakeWord expects 1280 samples per frame (80ms at 16kHz)
        # But we'll use our standard frame size and buffer accordingly
        self.required_samples = 1280  # OpenWakeWord's expected frame size
        self.audio_buffer = np.array([], dtype=np.float32)
        
        print(f"[OpenWakeWord] 🔄 Initializing OpenWakeWord detector...")
        print(f"[OpenWakeWord]    Model: {self.model_name}")
        print(f"[OpenWakeWord]    Threshold: {self.threshold}")
        
        try:
            import openwakeword
            from openwakeword import Model
            
            # Initialize the model
            # OpenWakeWord can load models by name or path
            # If no models specified, it loads all available models
            try:
                # Check if model_name is a file path
                model_path = None
                if self.model_name and os.path.exists(self.model_name):
                    # It's a file path
                    model_path = self.model_name
                    print(f"[OpenWakeWord] 📁 Loading model from path: {model_path}")
                elif self.model_name and not os.path.isabs(self.model_name):
                    # Check if it's a relative path in our models directory
                    # Priority: Check our custom directory FIRST before built-in models
                    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    custom_model_dir = os.path.join(workspace_root, 'data', 'models', 'wake_words')
                    
                    # Try with .onnx extension first
                    custom_model_path = os.path.join(custom_model_dir, f"{self.model_name}.onnx")
                    if os.path.exists(custom_model_path):
                        model_path = custom_model_path
                        print(f"[OpenWakeWord] 📁 Found custom model: {model_path}")
                    # Try without extension (in case user provided full filename)
                    elif os.path.exists(os.path.join(custom_model_dir, self.model_name)):
                        model_path = os.path.join(custom_model_dir, self.model_name)
                        print(f"[OpenWakeWord] 📁 Found custom model: {model_path}")
                    else:
                        # Try to find model variations (e.g., "hey_orah-2.onnx" when looking for "hey_orah")
                        if os.path.exists(custom_model_dir):
                            available_models = [f for f in os.listdir(custom_model_dir) if f.endswith('.onnx')]
                            if available_models:
                                print(f"[OpenWakeWord] 📁 Custom models directory: {custom_model_dir}")
                                print(f"[OpenWakeWord]    Available models: {', '.join(available_models)}")
                                
                                # Look for models that start with the model name (handles variations like "hey_orah-2.onnx")
                                matching_models = [m for m in available_models if m.startswith(self.model_name)]
                                if matching_models:
                                    # Use the first matching model (prefer exact match, then any variation)
                                    exact_match = f"{self.model_name}.onnx"
                                    if exact_match in matching_models:
                                        model_path = os.path.join(custom_model_dir, exact_match)
                                    else:
                                        model_path = os.path.join(custom_model_dir, matching_models[0])
                                    print(f"[OpenWakeWord] 📁 Found matching model: {os.path.basename(model_path)}")
                                else:
                                    print(f"[OpenWakeWord]    Looking for: {self.model_name}.onnx")
                                    print(f"[OpenWakeWord]    ⚠️  Custom model not found, will try built-in models")
                        else:
                            print(f"[OpenWakeWord] 📁 Custom models directory does not exist: {custom_model_dir}")
                            print(f"[OpenWakeWord]    Creating directory...")
                            os.makedirs(custom_model_dir, exist_ok=True)
                            print(f"[OpenWakeWord]    💡 Place your .onnx models here: {custom_model_dir}")
                
                # Try to load specific model (by name or path)
                if model_path:
                    # Load from file path
                    self.engine = Model(
                        wakeword_models=[model_path],
                        inference_framework='onnx'  # Use ONNX for better ARM64 support
                    )
                    print(f"[OpenWakeWord] ✅ Custom model loaded: {model_path}")
                else:
                    # Load by model name (built-in models)
                    self.engine = Model(
                        wakeword_models=[self.model_name] if self.model_name else None,
                        inference_framework='onnx'  # Use ONNX for better ARM64 support
                    )
                    print(f"[OpenWakeWord] ✅ Model loaded: {self.model_name or 'all available models'}")
            except Exception as e:
                error_str = str(e)
                print(f"[OpenWakeWord] ⚠️  Failed to load model '{self.model_name}': {e}")
                
                # Check if it's a missing model file error
                if "NO_SUCHFILE" in error_str or "File doesn't exist" in error_str or "melspectrogram.onnx" in error_str:
                    print(f"[OpenWakeWord] 💡 Models appear to be missing - attempting to download...")
                    print(f"[OpenWakeWord]    This may take a minute on first run...")
                    try:
                        # Use OpenWakeWord's utility function to download models
                        import openwakeword.utils
                        print(f"[OpenWakeWord]    Downloading required models (melspectrogram and wake word models)...")
                        openwakeword.utils.download_models()
                        print(f"[OpenWakeWord] ✅ Models downloaded successfully")
                        # Now try loading again
                        if model_path:
                            self.engine = Model(
                                wakeword_models=[model_path],
                                inference_framework='onnx'
                            )
                            print(f"[OpenWakeWord] ✅ Custom model loaded after download: {model_path}")
                        else:
                            self.engine = Model(
                                wakeword_models=[self.model_name] if self.model_name else None,
                                inference_framework='onnx'
                            )
                            print(f"[OpenWakeWord] ✅ Model loaded after download: {self.model_name or 'all available models'}")
                    except Exception as download_error:
                        print(f"[OpenWakeWord] ❌ Failed to download models: {download_error}")
                        print(f"[OpenWakeWord] 💡 Trying to load all available models...")
                        # Try loading all available models as fallback
                        try:
                            self.engine = Model(inference_framework='onnx')
                            # Get the first available model name
                            if hasattr(self.engine, 'models') and len(self.engine.models) > 0:
                                self.model_name = list(self.engine.models.keys())[0]
                                print(f"[OpenWakeWord] ✅ Using model: {self.model_name}")
                            else:
                                print(f"[OpenWakeWord] ✅ Model loaded (using all available)")
                        except Exception as e2:
                            print(f"[OpenWakeWord] ❌ Failed to load models: {e2}")
                            # If all attempts failed, engine is still None - will be caught below
                else:
                    print(f"[OpenWakeWord] 💡 Trying to load all available models...")
                    # Try loading all available models
                    try:
                        self.engine = Model(inference_framework='onnx')
                        # Get the first available model name
                        if hasattr(self.engine, 'models') and len(self.engine.models) > 0:
                            self.model_name = list(self.engine.models.keys())[0]
                            print(f"[OpenWakeWord] ✅ Using model: {self.model_name}")
                        else:
                            print(f"[OpenWakeWord] ✅ Model loaded (using all available)")
                    except Exception as e2:
                        print(f"[OpenWakeWord] ❌ Failed to load models: {e2}")
                        # If all attempts failed, engine is still None - will be caught below
            
            # Verify that engine was successfully initialized
            if self.engine is None:
                raise RuntimeError("Failed to initialize OpenWakeWord engine - models could not be loaded. Try running: python3 -c \"import openwakeword.utils; openwakeword.utils.download_models()\"")
            
            self.is_active = True
            print(f"[OpenWakeWord] ✅ OpenWakeWord initialized successfully")
            
        except ImportError:
            print(f"[OpenWakeWord] ❌ openwakeword package not found")
            print(f"[OpenWakeWord] 💡 Install with: pip install openwakeword")
            raise
        except Exception as e:
            print(f"[OpenWakeWord] ❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def process(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Process audio frame and check for wake word.
        
        Args:
            audio: Audio frame (float32, normalized to [-1, 1])
        
        Returns:
            Tuple of (detected: bool, confidence: float)
        """
        if not self.is_active or self.engine is None:
            return False, 0.0
        
        try:
            # OpenWakeWord expects 1280 samples (80ms at 16kHz)
            # Buffer audio until we have enough samples
            self.audio_buffer = np.concatenate([self.audio_buffer, audio])
            
            # Process when we have enough samples
            if len(self.audio_buffer) >= self.required_samples:
                # Take exactly required_samples
                frame_audio = self.audio_buffer[:self.required_samples].astype(np.float32)
                self.audio_buffer = self.audio_buffer[self.required_samples:]
                
                # OpenWakeWord expects int16 audio in range [-32768, 32767]
                # Convert from float32 [-1, 1] to int16
                audio_int16 = (frame_audio * 32767).astype(np.int16)
                
                # Get prediction
                # OpenWakeWord expects numpy array of int16 samples
                prediction = self.engine.predict(audio_int16)
                
                # prediction is a dict with model names as keys
                # Each value is a dict with 'score' (confidence) or just a float
                confidence = 0.0
                if isinstance(prediction, dict):
                    if self.model_name in prediction:
                        pred_value = prediction[self.model_name]
                        # Handle both dict format {'score': 0.5} and direct float
                        if isinstance(pred_value, dict):
                            confidence = float(pred_value.get('score', 0.0))
                        else:
                            confidence = float(pred_value)
                    elif len(prediction) > 0:
                        # Use first model's confidence if our model name doesn't match
                        first_model = list(prediction.keys())[0]
                        pred_value = prediction[first_model]
                        if isinstance(pred_value, dict):
                            confidence = float(pred_value.get('score', 0.0))
                        else:
                            confidence = float(pred_value)
                elif isinstance(prediction, (float, int)):
                    # Direct float/int confidence
                    confidence = float(prediction)
                
                detected = confidence >= self.threshold
                
                self.frame_count += 1
                
                # Debug logging (similar to Precise)
                if self.frame_count <= 10:
                    status = "🟢 DETECTED!" if detected else ("⚪ ACTIVITY" if confidence > 0.01 else "🔴 QUIET")
                    pct = (confidence / self.threshold * 100) if self.threshold > 0 else 0
                    print(f"[OpenWakeWord] {status} Confidence: {confidence:.6f} (threshold: {self.threshold:.6f}, {pct:.1f}%) - Frame {self.frame_count}")
                elif self.frame_count % 100 == 0:
                    status = "🟢 DETECTED!" if detected else ("⚪ ACTIVITY" if confidence > 0.01 else "🔴 QUIET")
                    pct = (confidence / self.threshold * 100) if self.threshold > 0 else 0
                    print(f"[OpenWakeWord] {status} Confidence: {confidence:.6f} (threshold: {self.threshold:.6f}, {pct:.1f}%) - Frame {self.frame_count}")
                elif confidence > self.threshold / 10 or confidence > 0.001:
                    status = "🟢 DETECTED!" if detected else ("⚪ ACTIVITY" if confidence > 0.01 else "🔴 QUIET")
                    pct = (confidence / self.threshold * 100) if self.threshold > 0 else 0
                    print(f"[OpenWakeWord] {status} Confidence: {confidence:.6f} (threshold: {self.threshold:.6f}, {pct:.1f}%) - Frame {self.frame_count}")
                
                if detected:
                    print(f"[OpenWakeWord] 🎤 WAKE WORD DETECTED! Confidence: {confidence:.6f}")
                
                return detected, confidence
            else:
                # Not enough samples yet, return no detection
                return False, 0.0
                
        except Exception as e:
            print(f"[OpenWakeWord] ⚠️  Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def clear_buffer(self):
        """Clear the internal audio buffer to prevent processing stale audio."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.frame_count = 0
        
        # Also try to reset the model's internal state if it has a reset method
        if self.engine is not None:
            try:
                if hasattr(self.engine, 'reset'):
                    self.engine.reset()
                    print("[OpenWakeWord] 🧹 Buffer and model state cleared")
                else:
                    print("[OpenWakeWord] 🧹 Buffer cleared")
            except Exception as e:
                print(f"[OpenWakeWord] 🧹 Buffer cleared (model reset failed: {e})")
        else:
            print("[OpenWakeWord] 🧹 Buffer cleared")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_active = False
        self.engine = None
        print("[OpenWakeWord] 🧹 Cleaned up")


def create_openwakeword_detector(model_name: str = None, threshold: float = None) -> Optional[OpenWakeWordDetector]:
    """
    Factory function to create an OpenWakeWord detector.
    
    Args:
        model_name: Optional model name (default: "hey_mycroft_v0.1")
        threshold: Optional threshold (default: from state.py or DEFAULT_THRESHOLD)
    
    Returns:
        OpenWakeWordDetector instance or None if initialization fails
    """
    try:
        # Get threshold from state if not provided
        if threshold is None:
            try:
                from state import get_wake_word_sensitivity
                # Convert sensitivity (0.0-1.0) to threshold (inverted: higher sensitivity = lower threshold)
                sensitivity = get_wake_word_sensitivity()
                threshold = 1.0 - sensitivity  # Invert: sensitivity 1.0 = threshold 0.0
            except ImportError:
                threshold = DEFAULT_THRESHOLD
        
        detector = OpenWakeWordDetector(model_name=model_name, threshold=threshold)
        return detector
    except Exception as e:
        print(f"[OpenWakeWord] ❌ Failed to create detector: {e}")
        return None

