import numpy as np

# === Configuration ===
CHANNEL_PLAYBACK = 5
FINGERPRINT_MATCH_THRESHOLD = 0.995
_last_fingerprint = None

# === Normalize and compute fingerprint ===
def compute_fingerprint(audio: np.ndarray) -> np.ndarray:
    audio = audio - np.mean(audio)
    norm = np.linalg.norm(audio)
    if norm == 0 or np.isnan(norm):
        print("[Fingerprint] ⚠️ Invalid audio (zero or NaN norm)")
        return np.zeros_like(audio)
    return audio / norm

# === Register current TTS fingerprint for suppression
def register_playback_fingerprint(audio: np.ndarray):
    global _last_fingerprint
    if len(audio) == 0:
        print("[Fingerprint] ⚠️ Skipped empty audio")
        return

    sample = audio[:2048]
    rms = np.sqrt(np.mean(sample**2))
    print(f"[Fingerprint] 🔊 Registering playback fingerprint: RMS={rms:.4f}")

    _last_fingerprint = compute_fingerprint(sample)
    print("[Fingerprint] 🎙️ Stored TTS playback fingerprint")

# === Match current mic input to TTS fingerprint
def fingerprint_match(mic_audio: np.ndarray) -> bool:
    global _last_fingerprint
    if _last_fingerprint is None:
        print("[Fingerprint] ⚠️ No stored fingerprint to compare")
        return False

    if len(mic_audio) < len(_last_fingerprint):
        print("[Fingerprint] ⚠️ Mic buffer too short for comparison")
        return False

    sample = mic_audio[:2048]
    mic_fp = compute_fingerprint(sample)
    sim = np.dot(mic_fp, _last_fingerprint)
    sim /= (np.linalg.norm(mic_fp) * np.linalg.norm(_last_fingerprint) + 1e-8)

    print(f"[Fingerprint] 🔍 Cosine similarity: {sim:.4f}")
    return sim >= FINGERPRINT_MATCH_THRESHOLD

# === Stub for main.py orchestration ===
def start_fingerprint_monitor():
    print("[Fingerprint] ✅ Fingerprint monitor initialized (stub)")
