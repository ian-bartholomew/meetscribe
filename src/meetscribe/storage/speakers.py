"""Global speaker registry with voice embeddings for cross-meeting recognition."""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
