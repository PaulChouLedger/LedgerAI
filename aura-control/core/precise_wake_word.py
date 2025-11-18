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
import subprocess
import numpy as np
from typing import Optional, Tuple

# === Wake Word Detection Configuration ===
# Tune these values for optimal detection performance

# Default threshold (lower = more sensitive, higher = less sensitive)
# Typical range: 0.1 (very sensitive) to 0.5 (less sensitive)
# Set to 0.001 for testing (very sensitive - will trigger on most audio activity)
DEFAULT_THRESHOLD = 0.001

# Sensitivity mapping: Maps sensitivity (0.0-1.0) to threshold range
# Formula: threshold = MAX_THRESHOLD - (sensitivity * (MAX_THRESHOLD - MIN_THRESHOLD))
SENSITIVITY_MAX_THRESHOLD = 0.35  # Less sensitive (sensitivity = 0.0)
SENSITIVITY_MIN_THRESHOLD = 0.15  # More sensitive (sensitivity = 1.0)

# Audio normalization for wake word detection
# Target RMS level for consistent audio processing (same as listener.py)
WAKE_WORD_TARGET_RMS = 0.05
WAKE_WORD_MAX_GAIN = 10.0  # Maximum amplification factor to prevent distortion

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
            # Map sensitivity (0.0-1.0) to threshold range
            if sensitivity is not None:
                threshold_range = SENSITIVITY_MAX_THRESHOLD - SENSITIVITY_MIN_THRESHOLD
                self.threshold = SENSITIVITY_MAX_THRESHOLD - (sensitivity * threshold_range)
            else:
                self.threshold = DEFAULT_THRESHOLD
        except ImportError:
            # Fallback if state module not available
            self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        
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
            # NOTE: chunk_size parameter is in BYTES, not samples!
            # For int16 audio: 2048 samples = 4096 bytes
            chunk_size_bytes = self.frame_length * 2  # int16 = 2 bytes per sample
            
            # Verify executable exists and is executable before creating engine
            if not os.path.exists(exe_file):
                print(f"[Wake Word] ❌ Executable not found: {exe_file}")
                return False
            if not os.access(exe_file, os.X_OK):
                print(f"[Wake Word] ❌ Executable not executable: {exe_file}")
                print(f"[Wake Word] 💡 Fix with: chmod +x {exe_file}")
                return False
            
            # Test executable manually first
            try:
                test_result = subprocess.run(
                    [exe_file, '--help'],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
                if test_result.returncode != 0:
                    print(f"[Wake Word] ⚠️ Executable test failed (return code: {test_result.returncode})")
                    if test_result.stderr:
                        print(f"[Wake Word] 🔍 stderr: {test_result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print("[Wake Word] ⚠️ Executable test timed out")
            except Exception as test_err:
                print(f"[Wake Word] ⚠️ Executable test error: {test_err}")
            
            # Create engine
            try:
                self.engine = PreciseEngine(exe_file=exe_file, model_file=model_file, chunk_size=chunk_size_bytes)
            except Exception as engine_err:
                print(f"[Wake Word] ❌ Failed to create PreciseEngine: {engine_err}")
                import traceback
                print(f"[Wake Word] 🔍 Traceback: {traceback.format_exc()}")
                return False
            
            # Check if proc attribute exists and verify subprocess
            if not hasattr(self.engine, 'proc'):
                print("[Wake Word] ⚠️ PreciseEngine has no 'proc' attribute - checking if subprocess starts on first call...")
            
            # PreciseEngine creates subprocess lazily on first get_prediction() call
            # The subprocess creation happens inside get_prediction(), but if it fails,
            # proc might be None. Let's check the PreciseEngine source to understand this better.
            # For now, let's try to manually inspect what's happening
            
            # Check engine internals before first call
            print(f"[Wake Word] 🔍 Engine created, checking internals...")
            if hasattr(self.engine, 'proc'):
                print(f"[Wake Word] 🔍 proc before first call: {self.engine.proc}")
            if hasattr(self.engine, 'exe_file'):
                print(f"[Wake Word] 🔍 exe_file: {self.engine.exe_file}")
            if hasattr(self.engine, 'model_file'):
                print(f"[Wake Word] 🔍 model_file: {self.engine.model_file}")
            
            # Try to manually start subprocess if PreciseEngine has a start method
            if hasattr(self.engine, 'start'):
                try:
                    self.engine.start()
                    print("[Wake Word] ✅ Called engine.start()")
                except Exception as start_err:
                    print(f"[Wake Word] ⚠️ engine.start() failed: {start_err}")
            
            # Test with a dummy call to trigger subprocess creation
            try:
                dummy_audio = np.zeros(self.frame_length, dtype=np.int16)
                dummy_bytes = dummy_audio.tobytes()
                print(f"[Wake Word] 🔍 Calling get_prediction() with {len(dummy_bytes)} bytes...")
                
                # This should trigger subprocess creation
                _ = self.engine.get_prediction(dummy_bytes)
                
                # Verify subprocess was created
                if hasattr(self.engine, 'proc'):
                    if self.engine.proc is None:
                        print("[Wake Word] ❌ Subprocess is still None after get_prediction() call")
                        print("[Wake Word] 💡 This suggests subprocess creation failed silently")
                        print("[Wake Word] 💡 Check if precise-engine binary works:")
                        print(f"[Wake Word]     {exe_file} {model_file}")
                        return False
                    print("[Wake Word] ✅ Engine subprocess started successfully")
                    print(f"[Wake Word] 🔍 Subprocess PID: {self.engine.proc.pid if hasattr(self.engine.proc, 'pid') else 'N/A'}")
                else:
                    print("[Wake Word] ⚠️ PreciseEngine doesn't expose 'proc' attribute - assuming it's working")
            except Exception as init_error:
                print(f"[Wake Word] ❌ Failed to start engine subprocess: {init_error}")
                import traceback
                print(f"[Wake Word] 🔍 Traceback: {traceback.format_exc()}")
                print("[Wake Word] 💡 Troubleshooting:")
                print(f"[Wake Word]     1. Check executable: ls -la {exe_file}")
                print(f"[Wake Word]     2. Test executable: {exe_file} --help")
                print(f"[Wake Word]     3. Test with model: {exe_file} {model_file}")
                print(f"[Wake Word]     4. Check model file: ls -la {model_file}")
                print(f"[Wake Word]     5. Check permissions: chmod +x {exe_file}")
                return False
            
            # Verify subprocess is running
            if hasattr(self.engine, 'proc') and self.engine.proc is not None:
                if hasattr(self.engine.proc, 'poll') and self.engine.proc.poll() is not None:
                    print(f"[Wake Word] ❌ Engine subprocess exited with code: {self.engine.proc.returncode}")
                    if hasattr(self.engine.proc, 'stderr'):
                        try:
                            stderr_output = self.engine.proc.stderr.read()
                            if stderr_output:
                                print(f"[Wake Word] 🔍 Subprocess stderr: {stderr_output}")
                        except:
                            pass
                    return False
            
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
            
            # Normalize audio using same logic as listener.py and test_transcription.py
            # This ensures consistent audio processing across all components
            # Calculate RMS for normalization (same as normalize_audio_for_whisper in listener.py)
            rms = np.sqrt(np.mean(audio_frame**2))
            peak = np.abs(audio_frame).max()
            
            # Normalize to target RMS if audio is too quiet
            # Same logic as listener.py normalize_audio_for_whisper() function
            if rms > 0.0001:  # Only normalize if we have some signal (same threshold as listener.py)
                gain = WAKE_WORD_TARGET_RMS / max(rms, 0.0001)
                # Limit gain to avoid distortion (same as listener.py)
                gain = min(gain, WAKE_WORD_MAX_GAIN)
                audio_frame = audio_frame * gain
                # Soft clipping to prevent distortion (same as listener.py normalize_audio_for_whisper)
                audio_frame = np.clip(audio_frame, -0.95, 0.95)
            elif peak > 1.0:
                # If audio is too loud, normalize down (safety check)
                audio_frame = audio_frame / peak
            
            # Convert to int16 for Precise
            audio_int16 = (audio_frame * 32767.0).astype(np.int16)
            
            # Get prediction directly from engine
            # PreciseEngine.get_prediction() expects chunk_size bytes (we initialized with bytes)
            # We have self.frame_length samples = self.frame_length * 2 bytes
            # Engine was initialized with chunk_size = self.frame_length * 2 bytes
            # So we need to send exactly self.frame_length samples = self.frame_length * 2 bytes
            
            # Check if subprocess is running before calling get_prediction
            if hasattr(self.engine, 'proc') and self.engine.proc is None:
                print("[Wake Word] ⚠️ Subprocess not started - attempting to start...")
                # The subprocess should be created on first get_prediction() call
                # But if it's None, something went wrong during initialization
                return False, 0.0
            
            try:
                prediction = self.engine.get_prediction(audio_int16.tobytes())
                
                # Debug: log first few predictions and occasionally after
                if not hasattr(self, '_prediction_debug_count'):
                    self._prediction_debug_count = 0
                self._prediction_debug_count += 1
                
                # Try to convert prediction to float
                try:
                    confidence = float(prediction)
                except (ValueError, TypeError):
                    # Prediction might be a string or other type
                    if isinstance(prediction, bytes):
                        prediction_str = prediction.decode('utf-8', errors='ignore').strip()
                        try:
                            confidence = float(prediction_str)
                        except ValueError:
                            print(f"[Wake Word] ⚠️ Could not parse prediction: {prediction!r}")
                            return False, 0.0
                    else:
                        print(f"[Wake Word] ⚠️ Unexpected prediction type: {type(prediction)}, value: {prediction!r}")
                        return False, 0.0
                
                # Debug logging:
                # - First 10 frames (to see initial behavior)
                # - Every 100 frames (heartbeat)
                # - When confidence > 0.001 (any significant activity)
                # - When confidence > threshold/10 (getting close to detection)
                # - When confidence is rising (last confidence was lower)
                debug_this = False
                if self._prediction_debug_count <= 10:
                    debug_this = True
                elif self._prediction_debug_count % 100 == 0:
                    debug_this = True
                elif confidence > 0.001:
                    debug_this = True
                elif confidence > self.threshold / 10:
                    debug_this = True
                else:
                    # Check if confidence is rising (compared to last value)
                    if not hasattr(self, '_last_confidence'):
                        self._last_confidence = 0.0
                    if confidence > self._last_confidence * 1.5 and confidence > 0.0001:
                        debug_this = True
                    self._last_confidence = confidence
                
                if debug_this:
                    # Show status indicator based on confidence level
                    if confidence >= self.threshold:
                        status = "🟢 DETECTED!"
                    elif confidence >= self.threshold * 0.8:
                        status = "🟡 VERY CLOSE"
                    elif confidence >= self.threshold * 0.5:
                        status = "🟠 CLOSE"
                    elif confidence >= self.threshold * 0.2:
                        status = "🔵 RISING"
                    elif confidence > 0.001:
                        status = "⚪ ACTIVITY"
                    else:
                        status = "🔴 QUIET"
                    
                    print(f"[Wake Word] {status} Confidence: {confidence:.6f} (threshold: {self.threshold:.6f}, {confidence/self.threshold*100:.1f}%)")
                
                # Check if confidence exceeds threshold
                detected = confidence >= self.threshold
                
                if detected:
                    print(f"[Wake Word] 🎤 WAKE WORD DETECTED! Confidence: {confidence:.6f}")
                    return True, confidence
                else:
                    return False, confidence
            except Exception as pred_error:
                # Engine might return empty or invalid response
                if not hasattr(self, '_prediction_error_count'):
                    self._prediction_error_count = 0
                self._prediction_error_count += 1
                if self._prediction_error_count <= 3:
                    print(f"[Wake Word] ⚠️ Prediction error: {pred_error}")
                    import traceback
                    print(f"[Wake Word] 🔍 Traceback: {traceback.format_exc()}")
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

