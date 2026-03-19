#!/usr/bin/env python3
"""
Real-time transcription on puck — captures audio from XVF3800 and
streams to local Whisper container as fast as possible.

Usage (on puck):
    python3 realtime_transcribe.py

Press Ctrl+C to stop.
"""

import io
import json
import struct
import time
import wave
import http.client
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIC_DEVICE = "hw:1,0"       # XVF3800
SAMPLE_RATE = 16000
CHANNELS = 2                 # XVF3800 USB has 2 channels
CHUNK_DURATION = 0.5         # seconds per read
VAD_THRESHOLD = 0.005        # RMS threshold to detect speech
SILENCE_TIMEOUT = 1.5        # seconds of silence to trigger transcription
MIN_SPEECH_DURATION = 0.3    # minimum speech duration to bother transcribing
WHISPER_HOST = "127.0.0.1"
WHISPER_PORT = 5000

# ---------------------------------------------------------------------------
# Audio capture via ALSA (sounddevice)
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    exit(1)


def rms(audio):
    """Calculate RMS of audio buffer."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)) / 32768.0)


def audio_to_wav_bytes(frames, sample_rate=SAMPLE_RATE):
    """Convert raw int16 frames to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames.tobytes())
    return buf.getvalue()


def whisper_transcribe(wav_bytes):
    """POST WAV bytes to Whisper container, return transcript."""
    boundary = "----RealtimeBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

    try:
        conn = http.client.HTTPConnection(WHISPER_HOST, WHISPER_PORT, timeout=15)
        conn.request(
            "POST", "/transcribe",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        return data.get("transcription", data.get("text", "")).strip()
    except Exception as e:
        return f"[whisper error: {e}]"


def main():
    print("=" * 60)
    print("REAL-TIME TRANSCRIPTION — XVF3800 → Whisper")
    print(f"Mic: {MIC_DEVICE}  Rate: {SAMPLE_RATE}  VAD: {VAD_THRESHOLD}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks = int(SILENCE_TIMEOUT / CHUNK_DURATION)

    # Open ALSA stream
    stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=chunk_samples,
    )
    stream.start()

    speech_buffer = []
    silent_count = 0
    is_speaking = False
    speech_start = None

    try:
        while True:
            data, overflowed = stream.read(chunk_samples)
            # Use channel 0 only (mono)
            mono = data[:, 0] if data.ndim > 1 else data
            level = rms(mono)

            if level >= VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    speech_start = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] Speech detected (RMS={level:.4f})", flush=True)
                speech_buffer.append(mono.copy())
                silent_count = 0
            else:
                if is_speaking:
                    silent_count += 1
                    speech_buffer.append(mono.copy())  # keep trailing silence

                    if silent_count >= silence_chunks:
                        # End of utterance — transcribe
                        duration = time.time() - speech_start
                        if duration >= MIN_SPEECH_DURATION:
                            all_audio = np.concatenate(speech_buffer)
                            wav_bytes = audio_to_wav_bytes(all_audio)
                            t0 = time.time()
                            transcript = whisper_transcribe(wav_bytes)
                            whisper_ms = (time.time() - t0) * 1000

                            if transcript and transcript not in ("[whisper error]", ""):
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] "
                                    f"({duration:.1f}s, whisper {whisper_ms:.0f}ms) "
                                    f'"{transcript}"',
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] "
                                    f"({duration:.1f}s) [no transcript]",
                                    flush=True,
                                )
                        else:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                f"(too short: {duration:.2f}s, skipped)",
                                flush=True,
                            )

                        speech_buffer.clear()
                        silent_count = 0
                        is_speaking = False

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
