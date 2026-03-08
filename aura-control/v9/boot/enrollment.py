"""
boot.enrollment -- CPU-only speaker embedding using resemblyzer.

Extracts 256-dim voice prints, stores/loads profiles, and matches
against known users via cosine similarity.

Storage layout in data/voice_profiles/:
    profiles.json           — index: {user_id: {name, created, embedding_file}}
    <user_id>_embedding.npy — numpy array (256,)
    <user_id>_samples/      — raw WAV recordings
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core.config import VOICE_PROFILES_DIR, EMBEDDING_DIM, EMBEDDING_MATCH_THRESHOLD


class VoiceEnrollment:
    """CPU-only speaker recognition via resemblyzer."""

    def __init__(self) -> None:
        self._dir = Path(VOICE_PROFILES_DIR)
        self._profiles_path = self._dir / "profiles.json"
        self._encoder = None  # lazy-loaded
        self._profiles: dict = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_profiles()

    # ------------------------------------------------------------------
    # Lazy-load resemblyzer (ImportError is caught by orchestrator)
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
        return self._encoder

    # ------------------------------------------------------------------
    # Profile persistence
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        if not self._profiles_path.exists():
            self._profiles = {}
            return
        try:
            with open(self._profiles_path) as f:
                self._profiles = json.load(f)
        except Exception as e:
            print(f"[enrollment] Failed to load profiles: {e}")
            self._profiles = {}

        # Pre-load embeddings
        for uid, meta in self._profiles.items():
            emb_file = self._dir / meta.get("embedding_file", "")
            if emb_file.exists():
                try:
                    self._embeddings[uid] = np.load(str(emb_file))
                except Exception:
                    pass

    def _save_profiles(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._profiles_path, "w") as f:
            json.dump(self._profiles, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_first_boot(self) -> bool:
        """True if no voice profiles exist."""
        return len(self._profiles) == 0

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract a 256-dim speaker embedding from audio (CPU, <1s).

        Args:
            audio: float32 mono audio, any sample rate (will be resampled to 16kHz
                   internally by resemblyzer if needed).
            sr: sample rate of the input audio.

        Returns:
            numpy array of shape (256,).
        """
        from resemblyzer import preprocess_wav
        encoder = self._get_encoder()
        wav = preprocess_wav(audio, source_sr=sr)
        embedding = encoder.embed_utterance(wav)
        return embedding

    def identify(self, audio: np.ndarray, sr: int = 16000) -> Tuple[Optional[str], float]:
        """Match audio against stored profiles via cosine similarity.

        Returns:
            (user_id, score) if score >= threshold, else (None, best_score).
        """
        if not self._embeddings:
            return None, 0.0

        emb = self.extract_embedding(audio, sr)
        best_uid = None
        best_score = 0.0

        for uid, stored_emb in self._embeddings.items():
            score = float(np.dot(emb, stored_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8
            ))
            if score > best_score:
                best_score = score
                best_uid = uid

        if best_score >= EMBEDDING_MATCH_THRESHOLD and best_uid is not None:
            return best_uid, best_score
        return None, best_score

    def enroll(self, name: str, audio: np.ndarray, sr: int = 16000) -> str:
        """Create a permanent voice profile.

        Args:
            name: user's name (can be updated later via retroactive transcription).
            audio: float32 mono audio for embedding extraction.
            sr: sample rate.

        Returns:
            The new user_id.
        """
        user_id = uuid.uuid4().hex[:12]
        emb = self.extract_embedding(audio, sr)

        # Save embedding
        self._dir.mkdir(parents=True, exist_ok=True)
        emb_filename = f"{user_id}_embedding.npy"
        np.save(str(self._dir / emb_filename), emb)

        # Save sample WAV
        samples_dir = self._dir / f"{user_id}_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        wav_path = samples_dir / f"enroll_{int(time.time())}.npy"
        np.save(str(wav_path), audio)

        # Update index
        self._profiles[user_id] = {
            "name": name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedding_file": emb_filename,
        }
        self._embeddings[user_id] = emb
        self._save_profiles()

        print(f"[enrollment] Enrolled user '{name}' as {user_id}")
        return user_id

    def update_name(self, user_id: str, name: str) -> bool:
        """Update a user's name (e.g. after retroactive transcription)."""
        if user_id not in self._profiles:
            return False
        self._profiles[user_id]["name"] = name
        self._save_profiles()
        print(f"[enrollment] Updated name for {user_id}: '{name}'")
        return True

    def deepen_profile(self, user_id: str, audio_samples: list[np.ndarray],
                       sr: int = 16000) -> bool:
        """Strengthen a voice profile by averaging in additional embeddings.

        Takes a list of audio clips, extracts embeddings from each, and
        averages them with the stored embedding for a more robust profile.
        """
        if user_id not in self._embeddings or not audio_samples:
            return False

        new_embs = []
        for audio in audio_samples:
            if audio is not None and len(audio) > sr * 0.5:  # at least 0.5s
                try:
                    emb = self.extract_embedding(audio, sr)
                    new_embs.append(emb)
                except Exception:
                    continue

        if not new_embs:
            return False

        # Average: old embedding + all new ones (old counts as 2x for stability)
        old_emb = self._embeddings[user_id]
        all_embs = [old_emb, old_emb] + new_embs  # weight original 2x
        avg_emb = np.mean(all_embs, axis=0).astype(np.float32)
        # Re-normalize
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-8:
            avg_emb = avg_emb / norm

        # Save updated embedding
        emb_filename = self._profiles[user_id]["embedding_file"]
        np.save(str(self._dir / emb_filename), avg_emb)
        self._embeddings[user_id] = avg_emb

        print(f"[enrollment] Deepened profile {user_id} with {len(new_embs)} samples")
        return True

    def get_name(self, user_id: str) -> Optional[str]:
        """Get the stored name for a user_id."""
        meta = self._profiles.get(user_id)
        return meta["name"] if meta else None
