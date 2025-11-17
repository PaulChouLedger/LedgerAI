"""
Wyoming Protocol OpenWakeWord Client

This module provides a client for the Wyoming OpenWakeWord service.
The container handles all wake word detection - we just need to communicate with it.

Uses the official Wyoming client for proper Protocol Buffers support.

Installation:
  1. Start container: cd setup && docker compose up -d wyoming-openwakeword
  2. Install client: pip install wyoming

References:
  - Wyoming Protocol: https://github.com/rhasspy/wyoming
  - Jetson Containers: https://github.com/dusty-nv/jetson-containers/tree/master/packages/smart-home/wyoming/wyoming-openwakeword
"""

import numpy as np
from typing import Optional, Tuple
import threading
import time
import asyncio

# Require official Wyoming client
# Try different import patterns as API may vary by version
WYOMING_AVAILABLE = False
AsyncWyomingClient = None
AudioChunk = None
Detection = None
_WYOMING_IMPORT_ERROR = None

try:
    # Try standard import first
    from wyoming.client import AsyncWyomingClient
    from wyoming.audio import AudioChunk
    from wyoming.wake import Detection
    WYOMING_AVAILABLE = True
except ImportError as e1:
    _WYOMING_IMPORT_ERROR = str(e1)
    try:
        # Try importing the modules and inspecting what's available
        import wyoming.client as wyoming_client
        import wyoming.audio as wyoming_audio
        import wyoming.wake as wyoming_wake
        
        # Inspect what's actually available
        client_attrs = [attr for attr in dir(wyoming_client) if not attr.startswith('_')]
        audio_attrs = [attr for attr in dir(wyoming_audio) if not attr.startswith('_')]
        wake_attrs = [attr for attr in dir(wyoming_wake) if not attr.startswith('_')]
        
        print(f"[Wyoming] 🔍 Available in wyoming.client: {client_attrs}")
        print(f"[Wyoming] 🔍 Available in wyoming.audio: {audio_attrs}")
        print(f"[Wyoming] 🔍 Available in wyoming.wake: {wake_attrs}")
        
        # Try to find the correct class names
        if 'AsyncWyomingClient' in client_attrs:
            AsyncWyomingClient = getattr(wyoming_client, 'AsyncWyomingClient')
        elif 'WyomingClient' in client_attrs:
            # Some versions might only have sync client
            print("[Wyoming] ⚠️  Only sync WyomingClient found, async not available")
            _WYOMING_IMPORT_ERROR = f"AsyncWyomingClient not found. Available classes: {client_attrs}"
        else:
            _WYOMING_IMPORT_ERROR = f"Client class not found. Available: {client_attrs}"
        
        if 'AudioChunk' in audio_attrs:
            AudioChunk = getattr(wyoming_audio, 'AudioChunk')
        else:
            _WYOMING_IMPORT_ERROR = f"{_WYOMING_IMPORT_ERROR}; AudioChunk not found in {audio_attrs}"
        
        if 'Detection' in wake_attrs:
            Detection = getattr(wyoming_wake, 'Detection')
        else:
            _WYOMING_IMPORT_ERROR = f"{_WYOMING_IMPORT_ERROR}; Detection not found in {wake_attrs}"
        
        if AsyncWyomingClient and AudioChunk and Detection:
            WYOMING_AVAILABLE = True
            print("[Wyoming] ✅ Found classes via inspection")
    except Exception as e2:
        _WYOMING_IMPORT_ERROR = f"Primary: {e1}, Inspection: {e2}"


class WyomingWakeWordClient:
    """
    Client for Wyoming OpenWakeWord service.
    
    Uses the official Wyoming client for proper Protocol Buffers support.
    """
    
    def __init__(self, host: str = "localhost", port: int = 10400):
        if not WYOMING_AVAILABLE:
            raise ImportError(
                "Wyoming client not available. Install with: pip install wyoming"
            )
        
        self.host = host
        self.port = port
        self.connected = False
        self.last_detection: Tuple[bool, float] = (False, 0.0)
        self.lock = threading.Lock()
        self.frame_length = 1280  # OpenWakeWord uses 1280 samples at 16kHz
        self.threshold = 0.5  # Default threshold (container handles this)
        self.is_active = False
        
        self.client: Optional[AsyncWyomingClient] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
    def _run_async_loop(self):
        """Run async event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def connect(self) -> bool:
        """Connect to Wyoming OpenWakeWord service."""
        try:
            # Start async event loop in background thread
            self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.loop_thread.start()
            time.sleep(0.1)  # Wait for loop to start
            
            # Connect using async client
            uri = f"tcp://{self.host}:{self.port}"
            future = asyncio.run_coroutine_threadsafe(
                self._async_connect(uri),
                self.loop
            )
            result = future.result(timeout=5.0)
            
            if result:
                self.connected = True
                self.is_active = True
                print(f"[Wyoming] ✅ Connected via official client at {self.host}:{self.port}")
                print(f"[Wyoming] 💡 Using official client: proper Protocol Buffers support, reliable")
                return True
            return False
        except Exception as e:
            print(f"[Wyoming] ❌ Connection error: {e}")
            if "Connection refused" in str(e) or isinstance(e, ConnectionRefusedError):
                print(f"[Wyoming] 💡 Start container with: cd setup && docker compose up -d wyoming-openwakeword")
            return False
    
    async def _async_connect(self, uri: str) -> bool:
        """Async connection."""
        try:
            self.client = AsyncWyomingClient.from_uri(uri)
            await self.client.connect()
            return True
        except Exception as e:
            print(f"[Wyoming] ❌ Async connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Wyoming service."""
        with self.lock:
            if self.client and self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.client.disconnect(),
                    self.loop
                )
                try:
                    future.result(timeout=2.0)
                except:
                    pass
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            self.connected = False
            self.is_active = False
            print("[Wyoming] 🔌 Disconnected")
    
    def send_audio(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Send audio frame and check for wake word detection."""
        if not self.connected or not self.client or not self.loop:
            return False, 0.0
        
        try:
            # Prepare audio
            audio_frame = self._prepare_audio(audio_frame)
            audio_int16 = (audio_frame * 32767.0).astype(np.int16)
            
            # Send via async client
            future = asyncio.run_coroutine_threadsafe(
                self._async_send_audio(audio_int16),
                self.loop
            )
            
            try:
                detected, confidence = future.result(timeout=0.01)
                with self.lock:
                    self.last_detection = (detected, confidence)
                return detected, confidence
            except:
                with self.lock:
                    return self.last_detection
        except Exception as e:
            print(f"[Wyoming] ⚠️ Error sending audio: {e}")
            return False, 0.0
    
    def _prepare_audio(self, audio_frame: np.ndarray) -> np.ndarray:
        """Prepare audio frame."""
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        if len(audio_frame) != self.frame_length:
            if len(audio_frame) < self.frame_length:
                audio_frame = np.pad(audio_frame, (0, self.frame_length - len(audio_frame)), mode='constant')
            else:
                audio_frame = audio_frame[:self.frame_length]
        
        # Normalize
        abs_max = np.abs(audio_frame).max()
        if abs_max > 1.0:
            audio_frame = audio_frame / abs_max
        elif abs_max < 0.01:
            gain = 0.1 / max(abs_max, 0.0001)
            audio_frame = audio_frame * min(gain, 10.0)
            audio_frame = np.clip(audio_frame, -1.0, 1.0)
        
        return audio_frame
    
    async def _async_send_audio(self, audio_int16: np.ndarray) -> Tuple[bool, float]:
        """Async send audio."""
        try:
            chunk = AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=audio_int16.tobytes()
            )
            await self.client.write_event(chunk)
            
            detection = await asyncio.wait_for(
                self.client.read_event(),
                timeout=0.01
            )
            
            if isinstance(detection, Detection):
                return True, detection.confidence
            return False, 0.0
        except asyncio.TimeoutError:
            return False, 0.0
        except Exception as e:
            print(f"[Wyoming] ⚠️ Async error: {e}")
            return False, 0.0
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected and self.client is not None
    
    def process(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Process audio frame (compatibility method)."""
        return self.send_audio(audio_frame)
    
    def release(self):
        """Release resources."""
        self.disconnect()


def create_wyoming_wake_word_detector(host: str = "localhost", port: int = 10400):
    """
    Create Wyoming OpenWakeWord detector client.
    
    Args:
        host: Service host (default: localhost)
        port: Service port (default: 10400)
        
    Returns:
        WyomingWakeWordClient instance or None
    """
    if not WYOMING_AVAILABLE:
        print("[Wyoming] ❌ Wyoming client not available")
        print("[Wyoming] 💡 Install with: pip install wyoming")
        # Show import error for debugging
        try:
            import sys
            print(f"[Wyoming] 🔍 Import error: {_WYOMING_IMPORT_ERROR}")
            print(f"[Wyoming] 🔍 Python executable: {sys.executable}")
            print(f"[Wyoming] 🔍 Python path (first 3): {sys.path[:3]}")
            # Try to check if wyoming is actually installed
            try:
                import wyoming
                print(f"[Wyoming] 🔍 Wyoming package found at: {wyoming.__file__}")
                print(f"[Wyoming] 🔍 But submodule import failed - check wyoming package version")
                print(f"[Wyoming] 🔍 Try: pip install --upgrade wyoming")
            except ImportError:
                print("[Wyoming] 🔍 Wyoming package not found in Python path")
                print(f"[Wyoming] 🔍 Make sure you're using the virtual environment: source ~/aura-env/bin/activate")
        except NameError:
            # _WYOMING_IMPORT_ERROR might not be defined if import failed differently
            print("[Wyoming] 🔍 Could not retrieve import error details")
        except Exception as e:
            print(f"[Wyoming] 🔍 Debug error: {e}")
        return None
    
    client = WyomingWakeWordClient(host, port)
    if client.connect():
        return client
    return None
