"""
core.gpu -- GPU mutual exclusion for Jetson unified memory.

Single lock guards all CUDA-touching operations (Whisper transcribe, LLM generate).
Piper TTS runs on CPU (ONNX) and does NOT need this lock.
"""

import threading

gpu_lock = threading.Lock()
