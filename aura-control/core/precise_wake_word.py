"""
Mycroft Precise Wake Word Detection Integration

Mycroft Precise is a highly reliable wake word detection engine that works
excellently on Jetson/ARM64 devices. It's used by many production voice assistants.

Installation:
    pip install precise-runner

Download a model:
    wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb
    # Or use your custom trained model

GitHub: https://github.com/MycroftAI/mycroft-precise
"""

import os
import numpy as np
from typing import Optional, Tuple

# Try to import Precise
try:
    from precise_runner import PreciseEngine, PreciseRunner
    from precise_runner.runner import ListenerEngine
    PRECISE_AVAILABLE = True
except ImportError:
    PRECISE_AVAILABLE = False
    PreciseEngine = None
    PreciseRunner = None
    ListenerEngine = None


class PreciseWakeWordDetector:
    """
    Mycroft Precise wake word detection wrapper for Aura.
    
    Usage:
        detector = PreciseWakeWordDetector(model_path="hey-mycroft.pb")
        if detector.initialize():
            detected, confidence = detector.process(audio_frame)
    """
    
    def __init__(self, model_path: Optional[str] = None, threshold: Optional[float] = None):
        """
        Initialize Precise detector.
        
        Args:
            model_path: Path to .pb model file (or None for default)
            threshold: Detection threshold (0.0-1.0, default from state)
        """
        # Load from state module (preferred) or use provided values
        try:
            from state import get_wake_word_sensitivity, get_wake_word_model_path
            sensitivity = threshold if threshold is not None else get_wake_word_sensitivity()
            if model_path is None:
                model_path = get_wake_word_model_path()
            
            # Precise uses threshold (lower = more sensitive)
            # Map sensitivity (0.0-1.0) to threshold (0.3-0.7)
            if sensitivity is not None:
                self.threshold = 0.7 - (sensitivity * 0.4)  # Maps 0.0->0.7, 0.5->0.5, 1.0->0.3
            else:
                self.threshold = 0.5  # Balanced default
        except ImportError:
            # Fallback if state module not available
            self.threshold = threshold if threshold is not None else 0.5
        
        self.model_path = model_path
        self.engine: Optional[PreciseEngine] = None
        self.is_active = False
        self.frame_length = 2048  # Precise uses 2048 samples at 16kHz (128ms)
        self.sample_rate = 16000
        
    def initialize(self) -> bool:
        """
        Initialize Precise engine.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not PRECISE_AVAILABLE:
            print("[Wake Word] ❌ Mycroft Precise not available - install with: pip install precise-runner")
            print("[Wake Word] 💡 Precise is highly recommended for Jetson - very reliable!")
            return False
        
        try:
            # Find model file
            if self.model_path and os.path.exists(self.model_path):
                model_file = self.model_path
            else:
                # Try to find default model in common locations
                default_locations = [
                    os.path.expanduser("~/hey-mycroft.pb"),
                    os.path.expanduser("~/precise-models/hey-mycroft.pb"),
                    "/usr/local/share/precise/hey-mycroft.pb",
                    "hey-mycroft.pb",  # Current directory
                ]
                
                model_file = None
                for loc in default_locations:
                    if os.path.exists(loc):
                        model_file = loc
                        break
                
                if not model_file:
                    print("[Wake Word] ❌ Precise model file not found")
                    print("[Wake Word] 💡 Download a model:")
                    print("[Wake Word]     wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb")
                    print("[Wake Word] 💡 Or train your own: https://github.com/MycroftAI/mycroft-precise")
                    return False
            
            # Find precise-engine executable
            # For Jetson, it needs to be downloaded separately as a binary
            import shutil
            import platform
            exe_file = shutil.which('precise-engine')
            if not exe_file:
                # Try common locations including Mycroft's default location
                machine = platform.machine()
                possible_paths = [
                    '/usr/local/bin/precise-engine',
                    '/usr/bin/precise-engine',
                    os.path.expanduser('~/.local/bin/precise-engine'),
                    os.path.expanduser('~/.mycroft/precise/precise-engine/precise-engine'),
                    os.path.expanduser('~/precise-engine/precise-engine'),
                    os.path.expanduser('~/precise/precise-engine'),
                ]
                for path in possible_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        exe_file = path
                        break
            
            if not exe_file:
                print("[Wake Word] ❌ precise-engine executable not found")
                print("[Wake Word] 💡 For Jetson, you need to download the binary:")
                print("[Wake Word]     cd ~")
                print("[Wake Word]     wget https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz")
                print("[Wake Word]     tar xzf precise-all_0.3.0_aarch64.tar.gz")
                print("[Wake Word]     chmod +x precise/precise-engine")
                print("[Wake Word]     mv precise ~/.mycroft/precise/")
                print("[Wake Word] 💡 Or run: ~/LedgerAI/setup/scripts/setup_precise_wake_word.sh")
                return False
            
            # Create Precise engine
            # PreciseEngine requires exe_file and model_file
            # Precise uses 2048 samples at 16kHz (128ms chunks)
            self.engine = PreciseEngine(exe_file=exe_file, model_file=model_file, chunk_size=self.frame_length)
            
            # For manual frame processing, we use the engine directly
            # PreciseRunner is designed for automatic streaming, not manual frame feeding
            # We'll call engine.get_prediction() directly in process()
            
            self.is_active = True
            print(f"[Wake Word] ✅ Mycroft Precise initialized with model: {model_file}")
            print(f"[Wake Word] 💡 Precise is highly reliable on Jetson!")
            return True
            
        except Exception as e:
            print(f"[Wake Word] ❌ Failed to initialize Precise: {e}")
            import traceback
            print(f"[Wake Word] 🔍 Traceback: {traceback.format_exc()}")
            print("[Wake Word] 💡 Install: pip install precise-runner")
            print("[Wake Word] 💡 Download model: wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb")
            return False
    
    # Note: _on_activation callback not needed when using engine.get_prediction() directly
    
    def process(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Process audio frame for wake word detection.
        
        Args:
            audio_frame: Audio samples (numpy array, float32, 16kHz)
            
        Returns:
            Tuple[bool, float]: (detected, confidence)
        """
        if not self.is_active or not self.engine:
            return False, 0.0
        
        try:
            # Ensure correct format
            if audio_frame.dtype != np.float32:
                audio_frame = audio_frame.astype(np.float32)
            
            # Ensure correct length
            if len(audio_frame) != self.frame_length:
                if len(audio_frame) < self.frame_length:
                    audio_frame = np.pad(audio_frame, (0, self.frame_length - len(audio_frame)), mode='constant')
                else:
                    audio_frame = audio_frame[:self.frame_length]
            
            # Normalize to [-1, 1]
            abs_max = np.abs(audio_frame).max()
            if abs_max > 1.0:
                audio_frame = audio_frame / abs_max
            elif abs_max < 0.01:
                # Boost quiet audio
                gain = 0.1 / max(abs_max, 0.0001)
                audio_frame = audio_frame * min(gain, 10.0)
                audio_frame = np.clip(audio_frame, -1.0, 1.0)
            
            # Convert to int16 for Precise
            audio_int16 = (audio_frame * 32767.0).astype(np.int16)
            
            # Get prediction directly from engine
            # PreciseEngine.get_prediction() returns probability (0.0-1.0)
            try:
                prediction = self.engine.get_prediction(audio_int16.tobytes())
                confidence = float(prediction)
                
                # Check if confidence exceeds threshold
                detected = confidence >= self.threshold
                
                if detected:
                    return True, confidence
                else:
                    return False, confidence
            except Exception as pred_error:
                # Engine might return empty or invalid response
                return False, 0.0
            
        except Exception as e:
            print(f"[Wake Word] ⚠️ Processing error: {e}")
            return False, 0.0
    
    def release(self):
        """Release Precise resources."""
        if self.engine:
            try:
                # PreciseEngine cleanup if needed
                self.engine = None
            except:
                pass
        self.is_active = False
        print("[Wake Word] 🔌 Precise released")


def create_precise_wake_word_detector(model_path: Optional[str] = None) -> Optional[PreciseWakeWordDetector]:
    """
    Factory function to create and initialize Precise wake word detector.
    
    Args:
        model_path: Path to .pb model file (or None for auto-detect)
        
    Returns:
        PreciseWakeWordDetector instance if successful, None otherwise
    """
    # Check if wake word is enabled (from state module)
    try:
        from state import get_wake_word_enabled
        enable_wake_word = get_wake_word_enabled()
    except ImportError:
        enable_wake_word = False
    
    if not enable_wake_word:
        print("[Wake Word] ℹ️  Wake word detection disabled (toggle in Settings)")
        return None
    
    if not PRECISE_AVAILABLE:
        print("[Wake Word] ⚠️  Mycroft Precise not installed - wake word detection disabled")
        print("[Wake Word] 💡 Install with: pip install precise-runner")
        print("[Wake Word] 💡 Precise is highly recommended for Jetson - very reliable!")
        return None
    
    # Create detector
    detector = PreciseWakeWordDetector(model_path=model_path)
    if detector.initialize():
        return detector
    
    return None

