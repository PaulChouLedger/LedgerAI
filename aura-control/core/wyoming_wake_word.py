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
from wyoming.client import AsyncTcpClient
from wyoming.audio import AudioChunk
from wyoming.wake import Detection
from wyoming.event import Event

# Alias for compatibility with existing code
AsyncWyomingClient = AsyncTcpClient
WYOMING_AVAILABLE = True


class WyomingWakeWordClient:
    """
    Client for Wyoming OpenWakeWord service.
    
    Uses the official Wyoming client for proper Protocol Buffers support.
    """
    
    def __init__(self, host: str = "localhost", port: int = 10400):
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
        self._detection_task: Optional[asyncio.Task] = None
        
    def _run_async_loop(self):
        """Run async event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def connect(self) -> bool:
        """Connect to Wyoming OpenWakeWord service."""
        import sys
        try:
            # Check if port is accessible first
            import socket
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2)
            try:
                result = test_socket.connect_ex((self.host, self.port))
                test_socket.close()
                if result != 0:
                    print(f"[Wyoming] ❌ Port {self.port} is not accessible (connection refused)", flush=True, file=sys.stderr)
                    print(f"[Wyoming] 💡 Container may not be running or not listening on port {self.port}", flush=True, file=sys.stderr)
                    print(f"[Wyoming] 💡 Check: docker compose ps wyoming-openwakeword", flush=True, file=sys.stderr)
                    print(f"[Wyoming] 💡 Start: cd setup && docker compose up -d wyoming-openwakeword", flush=True, file=sys.stderr)
                    print(f"[Wyoming] 💡 Check logs: docker compose logs wyoming-openwakeword", flush=True, file=sys.stderr)
                    sys.stderr.flush()
                    return False
                else:
                    print(f"[Wyoming] ✅ Port {self.port} is accessible", flush=True, file=sys.stderr)
                    sys.stderr.flush()
            except Exception as sock_err:
                print(f"[Wyoming] ⚠️ Socket test failed: {sock_err}", flush=True, file=sys.stderr)
                sys.stderr.flush()
            
            print(f"[Wyoming] 🔄 Starting connection to {self.host}:{self.port}...", flush=True, file=sys.stderr)
            sys.stderr.flush()
            # Start async event loop in background thread
            self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.loop_thread.start()
            time.sleep(0.3)  # Wait for loop to start
            
            # Verify loop is running
            max_wait = 10
            for i in range(max_wait):
                if self.loop and self.loop.is_running():
                    break
                time.sleep(0.1)
            else:
                print("[Wyoming] ❌ Event loop failed to start", flush=True, file=sys.stderr)
                sys.stderr.flush()
                return False
            
            # Connect using async client
            uri = f"tcp://{self.host}:{self.port}"
            print(f"[Wyoming] 🔄 Calling _async_connect({uri})...", flush=True, file=sys.stderr)
            sys.stderr.flush()
            future = asyncio.run_coroutine_threadsafe(
                self._async_connect(uri),
                self.loop
            )
            result = future.result(timeout=10.0)
            
            if result:
                # connected and is_active are now set inside _async_connect before task starts
                print(f"[Wyoming] ✅ Connected via official client at {self.host}:{self.port}", flush=True, file=sys.stderr)
                print(f"[Wyoming] 💡 Using official client: proper Protocol Buffers support, reliable", flush=True, file=sys.stderr)
                sys.stderr.flush()
                return True
            else:
                print("[Wyoming] ❌ Connection returned False", flush=True, file=sys.stderr)
                sys.stderr.flush()
            return False
        except Exception as e:
            print(f"[Wyoming] ❌ Connection error: {e}", flush=True, file=sys.stderr)
            import traceback
            print(f"[Wyoming] 🔍 Traceback: {traceback.format_exc()}", flush=True, file=sys.stderr)
            sys.stderr.flush()
            if "Connection refused" in str(e) or isinstance(e, ConnectionRefusedError):
                print(f"[Wyoming] 💡 Start container with: cd setup && docker compose up -d wyoming-openwakeword", flush=True, file=sys.stderr)
                sys.stderr.flush()
            return False
    
    async def _async_connect(self, uri: str) -> bool:
        """Async connection."""
        try:
            # AsyncTcpClient takes host and port directly, not a URI
            # Parse URI format: tcp://host:port
            if uri.startswith("tcp://"):
                uri = uri[6:]  # Remove tcp:// prefix
            host, port_str = uri.split(":")
            port = int(port_str)
            
            # Create AsyncTcpClient with host and port
            # Check if AsyncTcpClient needs different initialization
            try:
                self.client = AsyncWyomingClient(host, port)
            except TypeError:
                # Try with URI string instead
                self.client = AsyncWyomingClient(uri)
            
            await self.client.connect()
            import sys
            print("[Wyoming] ✅ Connected to server", flush=True)
            sys.stdout.flush()
            
            # Try to read initial events from server (might send info about available models)
            try:
                initial_event = await asyncio.wait_for(
                    self.client.read_event(),
                    timeout=2.0
                )
                print(f"[Wyoming] ✅ Received initial event: {type(initial_event)}", flush=True, file=sys.stderr)
                sys.stderr.flush()
            except asyncio.TimeoutError:
                print("[Wyoming] ⏳ No initial event from server (this is OK)", flush=True, file=sys.stderr)
                sys.stderr.flush()
            except Exception as e:
                print(f"[Wyoming] ⚠️ Error reading initial event: {e}", flush=True, file=sys.stderr)
                sys.stderr.flush()
            
            # IMPORTANT: Set connected=True BEFORE starting the task
            # The task checks self.connected in its while loop
            self.connected = True
            self.is_active = True
            
            # Start background task to read detection events
            try:
                # Use stderr as well to ensure visibility
                print("[Wyoming] 🔄 Creating background detection task...", flush=True, file=sys.stderr)
                print(f"[Wyoming] 🔍 self.connected={self.connected}, self.client={self.client is not None}", flush=True, file=sys.stderr)
                sys.stderr.flush()
                self._detection_task = self.loop.create_task(self._read_detections())
                print("[Wyoming] ✅ Background detection task created", flush=True, file=sys.stderr)
                sys.stderr.flush()
                # Give it a moment to start
                await asyncio.sleep(0.1)
                # Verify task is running - wait a moment to see if it completes
                await asyncio.sleep(0.3)
                if self._detection_task.done():
                    print("[Wyoming] ⚠️ Detection task completed immediately - checking exception...", flush=True, file=sys.stderr)
                    sys.stderr.flush()
                    try:
                        # Get the result/exception - this will raise if there was an exception
                        result = self._detection_task.result()
                        print(f"[Wyoming] ⚠️ Task returned normally: {result}", flush=True, file=sys.stderr)
                        print(f"[Wyoming] 🔍 Task completed but should be running - checking state...", flush=True, file=sys.stderr)
                        print(f"[Wyoming] 🔍 self.connected={self.connected}, self.client={self.client is not None}", flush=True, file=sys.stderr)
                    except Exception as task_exc:
                        print(f"[Wyoming] ❌❌❌ TASK EXCEPTION: {task_exc}", flush=True, file=sys.stderr)
                        print(f"[Wyoming] ❌ Exception type: {type(task_exc).__name__}", flush=True, file=sys.stderr)
                        import traceback
                        print(f"[Wyoming] 🔍 FULL TRACEBACK:", flush=True, file=sys.stderr)
                        tb_str = traceback.format_exc()
                        print(tb_str, flush=True, file=sys.stderr)
                        # Also print to stdout as backup
                        print(f"\n[Wyoming] ❌ TASK CRASHED: {task_exc}", flush=True)
                        print(f"[Wyoming] 🔍 {tb_str}", flush=True)
                    sys.stderr.flush()
                else:
                    print("[Wyoming] ✅ Detection task is running", flush=True, file=sys.stderr)
                    sys.stderr.flush()
            except Exception as task_error:
                print(f"[Wyoming] ⚠️ Failed to create detection task: {task_error}", flush=True, file=sys.stderr)
                import traceback
                print(f"[Wyoming] 🔍 Traceback: {traceback.format_exc()}", flush=True, file=sys.stderr)
                sys.stderr.flush()
            
            return True
        except Exception as e:
            print(f"[Wyoming] ❌ Async connection error: {e}")
            import traceback
            print(f"[Wyoming] 🔍 Traceback: {traceback.format_exc()}")
            return False
    
    async def _read_detections(self):
        """Background task to continuously read detection events."""
        import sys
        try:
            print("[Wyoming] 🔄 Starting background detection reader...", flush=True, file=sys.stderr)
            sys.stderr.flush()
            
            # Verify client is available
            if not self.client:
                print("[Wyoming] ❌ No client available in _read_detections", flush=True, file=sys.stderr)
                sys.stderr.flush()
                return
            
            print("[Wyoming] ✅ Client available, starting read loop...", flush=True, file=sys.stderr)
            sys.stderr.flush()
            
            event_count = 0
            print("[Wyoming] ✅ Detection reader loop started, waiting for events...", flush=True, file=sys.stderr)
            sys.stderr.flush()
            
            while self.connected and self.client:
                try:
                    event = await asyncio.wait_for(
                        self.client.read_event(),
                        timeout=1.0
                    )

                    event_count += 1
                    if event_count == 1:
                        print(f"[Wyoming] ✅ Received first event: {type(event)}", flush=True, file=sys.stderr)
                        sys.stderr.flush()

                    if event:
                        # Debug: check event type
                        if not hasattr(self, '_event_types_seen'):
                            self._event_types_seen = set()
                        event_type = type(event).__name__
                        if event_type not in self._event_types_seen:
                            self._event_types_seen.add(event_type)
                            print(f"[Wyoming] 🔍 Event type: {event_type}, methods: {[m for m in dir(event) if not m.startswith('_')][:5]}", flush=True, file=sys.stderr)
                            # Try to get event data
                            try:
                                if hasattr(event, 'data'):
                                    print(f"[Wyoming] 🔍 Event data type: {type(event.data)}", flush=True, file=sys.stderr)
                                if hasattr(event, 'type'):
                                    print(f"[Wyoming] 🔍 Event type attribute: {event.type}", flush=True, file=sys.stderr)
                            except:
                                pass
                            sys.stderr.flush()

                        # Try to parse as Detection
                        try:
                            detection = Detection.from_event(event)
                            if detection:
                                with self.lock:
                                    self.last_detection = (True, detection.confidence)
                                print(f"[Wyoming] 🎤 Wake word detected! Confidence: {detection.confidence:.3f}", flush=True, file=sys.stderr)
                                sys.stderr.flush()
                        except Exception as parse_error:
                            # Not a detection event, that's OK
                            if event_count <= 3:  # Only log first few non-detection events
                                print(f"[Wyoming] 🔍 Event is not a detection: {parse_error}", flush=True, file=sys.stderr)
                                sys.stderr.flush()
                except asyncio.TimeoutError:
                    # No event received, continue
                    if event_count == 0 and self.connected:
                        # Log once that we're waiting
                        if not hasattr(self, '_waiting_logged'):
                            print("[Wyoming] ⏳ Waiting for events from server...", flush=True, file=sys.stderr)
                            sys.stderr.flush()
                            self._waiting_logged = True
                    # Log periodically that we're still waiting (every 10 seconds)
                    if not hasattr(self, '_last_wait_log'):
                        self._last_wait_log = time.time()
                    elif time.time() - self._last_wait_log > 10.0:
                        print(f"[Wyoming] ⏳ Still waiting for events... (received {event_count} events so far)", flush=True, file=sys.stderr)
                        sys.stderr.flush()
                        self._last_wait_log = time.time()
                    continue
                except Exception as e:
                    if self.connected:  # Only log if still connected
                        print(f"[Wyoming] ⚠️ Error reading detection: {e}", flush=True, file=sys.stderr)
                        import traceback
                        print(f"[Wyoming] 🔍 Traceback: {traceback.format_exc()}", flush=True, file=sys.stderr)
                        sys.stderr.flush()
                    break
        except Exception as e:
            print(f"[Wyoming] ❌ Detection reader crashed: {e}", flush=True, file=sys.stderr)
            import traceback
            print(f"[Wyoming] 🔍 Full traceback:", flush=True, file=sys.stderr)
            print(traceback.format_exc(), flush=True, file=sys.stderr)
            sys.stderr.flush()
            # Re-raise so the task shows the exception
            raise
    
    def disconnect(self):
        """Disconnect from Wyoming service."""
        with self.lock:
            # Cancel detection task
            if self._detection_task and not self._detection_task.done() and self.loop:
                self.loop.call_soon_threadsafe(self._detection_task.cancel)
            
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
        import sys
        try:
            # Create AudioChunk
            chunk = AudioChunk(
                rate=16000,
                width=2,  # 16-bit = 2 bytes
                channels=1,
                audio=audio_int16.tobytes()
            )
            # Convert AudioChunk to Event using .event() method
            event = chunk.event()
            
            # Debug: check client methods on first call
            if not hasattr(self, '_client_debugged'):
                print(f"[Wyoming] 🔍 Client type: {type(self.client)}", flush=True, file=sys.stderr)
                client_methods = [m for m in dir(self.client) if not m.startswith('_') and callable(getattr(self.client, m, None))]
                print(f"[Wyoming] 🔍 Client methods: {client_methods[:10]}...", flush=True, file=sys.stderr)  # First 10 methods
                print(f"[Wyoming] 🔍 Event type: {type(event)}", flush=True, file=sys.stderr)
                sys.stderr.flush()
                self._client_debugged = True
            
            # Write the event
            try:
                await self.client.write_event(event)
                # Log first few sends to verify audio is being sent
                if not hasattr(self, '_audio_send_count'):
                    self._audio_send_count = 0
                self._audio_send_count += 1
                if self._audio_send_count <= 3:
                    print(f"[Wyoming] ✅ Sent audio chunk {self._audio_send_count} ({len(audio_int16)} samples)", flush=True, file=sys.stderr)
                    sys.stderr.flush()
            except Exception as write_err:
                # Only log errors occasionally to avoid spam
                if not hasattr(self, '_last_write_error') or time.time() - self._last_write_error > 1.0:
                    print(f"[Wyoming] ⚠️ Error writing audio: {write_err}", flush=True, file=sys.stderr)
                    sys.stderr.flush()
                    self._last_write_error = time.time()
                raise
            
            # Detection events are read by background task _read_detections()
            # Just return the last known detection state
            with self.lock:
                return self.last_detection
        except Exception as e:
            # Only print error once per second to avoid spam
            import time
            if not hasattr(self, '_last_error_time') or time.time() - self._last_error_time > 1.0:
                print(f"[Wyoming] ⚠️ Async error: {e}")
                import traceback
                if "payload" in str(e).lower():
                    print(f"[Wyoming] 🔍 AudioChunk API issue - checking available methods...")
                    print(f"[Wyoming] 🔍 AudioChunk dir: {[x for x in dir(AudioChunk) if not x.startswith('_')]}")
                self._last_error_time = time.time()
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
    
    Uses official Wyoming client (AsyncTcpClient) for proper Protocol Buffers support.
    
    Args:
        host: Service host (default: localhost)
        port: Service port (default: 10400)
        
    Returns:
        WyomingWakeWordClient instance or None
    """
    import sys
    print(f"[Wyoming] 🔄 Creating detector for {host}:{port}...", flush=True)
    sys.stdout.flush()
    client = WyomingWakeWordClient(host, port)
    print(f"[Wyoming] 🔄 Calling client.connect()...", flush=True)
    sys.stdout.flush()
    if client.connect():
        print(f"[Wyoming] ✅ Detector created and connected", flush=True)
        sys.stdout.flush()
        return client
    else:
        print(f"[Wyoming] ❌ Detector creation failed - connect() returned False", flush=True)
        sys.stdout.flush()
    return None
