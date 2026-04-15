"""Global speaker registry with voice embeddings for cross-meeting recognition."""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EmbeddingRecord:
    vector: list[float]
    source_meeting: str
    created_at: str


@dataclass
class SpeakerProfile:
    id: str
    name: str
    embeddings: list[EmbeddingRecord] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


MAX_EMBEDDINGS_PER_SPEAKER = 10


def _generate_id() -> str:
    return f"sp_{secrets.token_hex(3)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SpeakerRegistry:
    """Manages speakers.json — the global speaker profile store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._speakers: list[SpeakerProfile] = []
        self._match_threshold: float = 0.65
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._match_threshold = data.get("match_threshold", 0.65)
            for s in data.get("speakers", []):
                embeddings = [
                    EmbeddingRecord(
                        vector=e["vector"],
                        source_meeting=e["source_meeting"],
                        created_at=e["created_at"],
                    )
                    for e in s.get("embeddings", [])
                ]
                self._speakers.append(SpeakerProfile(
                    id=s["id"],
                    name=s["name"],
                    embeddings=embeddings,
                    created_at=s.get("created_at", ""),
                    updated_at=s.get("updated_at", ""),
                ))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "match_threshold": self._match_threshold,
            "speakers": [
                {
                    "id": s.id,
                    "name": s.name,
                    "embeddings": [
                        {
                            "vector": e.vector,
                            "source_meeting": e.source_meeting,
                            "created_at": e.created_at,
                        }
                        for e in s.embeddings
                    ],
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                }
                for s in self._speakers
            ],
        }
        self._path.write_text(json.dumps(data, indent=2))

    @property
    def match_threshold(self) -> float:
        return self._match_threshold

    def list_speakers(self) -> list[SpeakerProfile]:
        return list(self._speakers)

    def get_speaker(self, speaker_id: str) -> SpeakerProfile | None:
        for s in self._speakers:
            if s.id == speaker_id:
                return s
        return None

    def create_speaker(self, name: str) -> SpeakerProfile:
        now = _now_iso()
        profile = SpeakerProfile(
            id=_generate_id(),
            name=name,
            created_at=now,
            updated_at=now,
        )
        self._speakers.append(profile)
        self._save()
        return profile

    def rename_speaker(self, speaker_id: str, new_name: str) -> None:
        speaker = self.get_speaker(speaker_id)
        if speaker:
            speaker.name = new_name
            speaker.updated_at = _now_iso()
            self._save()

    def add_embedding(
        self, speaker_id: str, vector: list[float], source_meeting: str
    ) -> None:
        speaker = self.get_speaker(speaker_id)
        if not speaker:
            return
        record = EmbeddingRecord(
            vector=vector,
            source_meeting=source_meeting,
            created_at=_now_iso(),
        )
        speaker.embeddings.append(record)
        if len(speaker.embeddings) > MAX_EMBEDDINGS_PER_SPEAKER:
            speaker.embeddings = speaker.embeddings[-MAX_EMBEDDINGS_PER_SPEAKER:]
        speaker.updated_at = _now_iso()
        self._save()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _speaker_centroid(speaker: SpeakerProfile) -> np.ndarray | None:
    """Compute the centroid (mean) of a speaker's stored embeddings."""
    if not speaker.embeddings:
        return None
    vectors = np.array([e.vector for e in speaker.embeddings])
    return vectors.mean(axis=0)


def match_speakers(
    cluster_embeddings: dict[str, np.ndarray],
    registry: SpeakerRegistry,
) -> dict[str, SpeakerProfile]:
    """Match cluster embeddings against known speaker profiles.

    Args:
        cluster_embeddings: Maps cluster label ("Speaker 1") to its average embedding.
        registry: The global speaker registry.

    Returns:
        Dict mapping cluster label to matched SpeakerProfile. Only includes
        matches above the registry's threshold. If two clusters match the same
        speaker, only the highest-similarity match is kept.
    """
    speakers = registry.list_speakers()
    threshold = registry.match_threshold

    # Precompute centroids for all known speakers
    centroids: dict[str, np.ndarray] = {}
    for s in speakers:
        c = _speaker_centroid(s)
        if c is not None:
            centroids[s.id] = c

    if not centroids:
        return {}

    # For each cluster, find the best matching speaker
    all_matches: list[tuple[str, str, float]] = []
    for label, emb in cluster_embeddings.items():
        best_id = ""
        best_sim = -1.0
        for sid, centroid in centroids.items():
            sim = _cosine_similarity(np.array(emb), centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = sid
        if best_sim >= threshold and best_id:
            all_matches.append((label, best_id, best_sim))

    # Resolve conflicts: if two clusters match the same speaker, keep highest similarity
    best_per_speaker: dict[str, tuple[str, float]] = {}
    for label, sid, sim in all_matches:
        if sid not in best_per_speaker or sim > best_per_speaker[sid][1]:
            best_per_speaker[sid] = (label, sim)

    winning_labels = {label for label, _ in best_per_speaker.values()}

    result: dict[str, SpeakerProfile] = {}
    for label, sid, sim in all_matches:
        if label in winning_labels:
            speaker = registry.get_speaker(sid)
            if speaker:
                result[label] = speaker

    return result


def rewrite_transcript(transcript: str, speaker_map: dict[str, str]) -> str:
    """Rewrite speaker labels in a markdown transcript.

    Args:
        transcript: The markdown transcript text.
        speaker_map: Maps current name in transcript to new name.
                     e.g., {"Speaker 1": "Alice"} or {"Alice": "Alice Smith"}

    Returns:
        The transcript with speaker labels replaced.
    """
    result = transcript
    for old_name, new_name in speaker_map.items():
        result = result.replace(f"**{old_name}:**", f"**{new_name}:**")
    return result
