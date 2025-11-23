#!/usr/bin/env python3
"""
Background Listener - Continuously transcribes audio for memory storage
Works independently of wake word system
"""

import os
import io
import time
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
import requests
import logging
from typing import Optional, Callable
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class BackgroundListener:
    """
    Continuously listens to audio and transcribes for memory storage
    """
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 whisper_service_url: str = "http://localhost:5000",
                 sample_rate: int = 16000,
                 frame_size: int = 512,
                 device_name: str = "reSpeaker",
                 on_transcription: Optional[Callable[[str], None]] = None):
        """
        Initialize background listener
        
        Args:
            memory_manager: MemoryManager instance for storing transcriptions
            whisper_service_url: URL of Whisper transcription service
            sample_rate: Audio sample rate
            frame_size: Audio frame size
            device_name: Microphone device name
            on_transcription: Optional callback when transcription is received
        """
        self.memory_manager = memory_manager
        self.whisper_service_url = whisper_service_url
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.device_name = device_name
        self.on_transcription = on_transcription
        
        # State
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stream: Optional[sd.InputStream] = None
        self.device_index: Optional[int] = None
        
        # Audio buffer for accumulating speech
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = int(1.0 * sample_rate / frame_size)  # 1 second minimum
        self.max_silence_frames = int(2.0 * sample_rate / frame_size)  # 2 seconds silence to flush
        
        # VAD (Voice Activity Detection) thresholds
        self.vad_threshold = 0.01  # RMS threshold for speech detection
        
    def start(self):
        """Start background listening"""
        if self.running:
            logger.warning("[BackgroundListener] Already running")
            return
        
        logger.info(f"[BackgroundListener] 🔧 Starting background listener (device: {self.device_name})...")
        
        # Find device
        self.device_index = self._find_device()
        if self.device_index is None:
            logger.error(f"[BackgroundListener] ❌ Device '{self.device_name}' not found - cannot start listening")
            logger.error(f"[BackgroundListener] 💡 Available input devices:")
            try:
                devices = sd.query_devices()
                found_any = False
                for i, device in enumerate(devices):
                    if device["max_input_channels"] > 0:
                        logger.error(f"[BackgroundListener]    [{i}] {device['name']} (inputs: {device['max_input_channels']})")
                        found_any = True
                if not found_any:
                    logger.error(f"[BackgroundListener]    No input devices found!")
            except Exception as e:
                logger.error(f"[BackgroundListener]    Error listing devices: {e}")
                import traceback
                logger.error(traceback.format_exc())
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("[BackgroundListener] ✅ Started background listening - continuously transcribing all audio")
    
    def stop(self):
        """Stop background listening"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("[BackgroundListener] ✅ Stopped background listening")
    
    def _find_device(self) -> Optional[int]:
        """Find audio input device"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if self.device_name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
                    logger.info(f"[BackgroundListener] Found device: {device['name']} (index {i})")
                    return i
            logger.warning(f"[BackgroundListener] Device '{self.device_name}' not found, using default")
            return None
        except Exception as e:
            logger.error(f"[BackgroundListener] Error finding device: {e}")
            return None
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            # Open audio stream
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                dtype=np.float32,
                callback=self._audio_callback
            )
            
            self.stream.start()
            logger.info("[BackgroundListener] ✅ Audio stream started - listening continuously")
            
            # Log heartbeat every 30 seconds to confirm it's running
            last_heartbeat = time.time()
            
            # Process buffer periodically
            while self.running:
                time.sleep(0.1)  # Check every 100ms
                self._process_buffer()
                
                # Heartbeat log every 30 seconds
                if time.time() - last_heartbeat > 30.0:
                    logger.info("[BackgroundListener] 💓 Background listener active - continuously listening and transcribing")
                    last_heartbeat = time.time()
                
        except Exception as e:
            logger.error(f"[BackgroundListener] Error in listen loop: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback - called for each audio frame"""
        if status:
            logger.warning(f"[BackgroundListener] Audio status: {status}")
        
        # Extract audio data
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        
        # Simple VAD: check RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        with self.buffer_lock:
            if rms > self.vad_threshold:
                # Speech detected
                self.audio_buffer.append(audio_data)
                self.speech_frames += 1
                self.silence_frames = 0
                # Log speech detection periodically (every 50 frames to avoid spam)
                if not hasattr(self, '_last_speech_log') or self.speech_frames % 50 == 0:
                    logger.debug(f"[BackgroundListener] 🎤 Speech detected (RMS: {rms:.4f}, threshold: {self.vad_threshold:.4f}, frames: {self.speech_frames})")
                    self._last_speech_log = self.speech_frames
            else:
                # Silence
                self.silence_frames += 1
                # Still add to buffer if we're in a speech segment
                if len(self.audio_buffer) > 0:
                    self.audio_buffer.append(audio_data)
    
    def _process_buffer(self):
        """Process accumulated audio buffer"""
        with self.buffer_lock:
            # Check if we should transcribe
            should_transcribe = False
            # Log periodic status (every 5 seconds of processing)
            if hasattr(self, '_last_status_log'):
                if time.time() - self._last_status_log > 5.0:
                    logger.debug(f"[BackgroundListener] 📊 Status: buffer={len(self.audio_buffer)} chunks, speech_frames={self.speech_frames}, silence_frames={self.silence_frames}")
                    self._last_status_log = time.time()
            else:
                self._last_status_log = time.time()
            
            if len(self.audio_buffer) > 0:
                # If we have enough speech and hit silence threshold, transcribe
                if self.speech_frames >= self.min_speech_frames and self.silence_frames >= self.max_silence_frames:
                    should_transcribe = True
                    buffer_duration = len(self.audio_buffer) * self.frame_size / self.sample_rate
                    logger.info(f"[BackgroundListener] 🎙️ Ready to transcribe: {buffer_duration:.2f}s audio ({self.speech_frames} speech frames, {self.silence_frames} silence frames)")
                # Or if buffer is getting very large (max 10 seconds)
                elif len(self.audio_buffer) * self.frame_size >= 10 * self.sample_rate:
                    should_transcribe = True
                    logger.info(f"[BackgroundListener] 🎙️ Buffer full, transcribing: {len(self.audio_buffer)} chunks")
            
            if not should_transcribe:
                return
            
            # Extract audio for transcription
            audio_data = np.concatenate(self.audio_buffer)
            buffer_duration = len(audio_data) / self.sample_rate
            
            # Clear buffer
            self.audio_buffer = []
            self.speech_frames = 0
            self.silence_frames = 0
        
        # Transcribe in separate thread to avoid blocking
        logger.info(f"[BackgroundListener] 🎤 Sending {buffer_duration:.2f}s of audio to Whisper for transcription...")
        threading.Thread(target=self._transcribe_audio, args=(audio_data,), daemon=True).start()
    
    def _transcribe_audio(self, audio_data: np.ndarray):
        """Transcribe audio using Whisper service"""
        try:
            # Normalize audio
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.001:  # Too quiet, skip
                return
            
            # Normalize to target RMS
            target_rms = 0.12
            if rms > 0:
                gain = target_rms / rms
                gain = min(gain, 10.0)  # Max gain
                audio_data = audio_data * gain
                audio_data = np.clip(audio_data, -0.95, 0.95)
            
            # Convert to WAV
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_data, self.sample_rate, format="WAV")
            wav_io.seek(0)
            
            # Call Whisper service
            response = requests.post(
                f"{self.whisper_service_url}/transcribe",
                files={"audio": ("speech.wav", wav_io, "audio/wav")},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                
                if isinstance(text, dict):
                    text = text.get("text", "").strip()
                
                if text and len(text) > 3:  # Minimum meaningful text
                    logger.info(f"[BackgroundListener] ✅ Transcribed: '{text[:80]}...'")
                    # Store in memory
                    self.memory_manager.store_conversation(
                        text=text,
                        source="background",
                        metadata={"duration": len(audio_data) / self.sample_rate}
                    )
                    logger.info(f"[BackgroundListener] 💾 Stored in memory (source: background)")
                    
                    # Call callback if provided
                    if self.on_transcription:
                        self.on_transcription(text)
                else:
                    logger.debug(f"[BackgroundListener] ⚠️ Transcription too short or empty: '{text}'")
            else:
                logger.warning(f"[BackgroundListener] Whisper returned status {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning("[BackgroundListener] Whisper request timeout")
        except Exception as e:
            logger.error(f"[BackgroundListener] Transcription error: {e}")

