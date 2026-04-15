# Speaker Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add post-transcription speaker mapping with voice-embedding-based auto-recognition across meetings.

**Architecture:** New `speakers.py` module handles the global speaker registry (`speakers.json`) with CRUD, embedding storage, and cosine-similarity matching. Diarization returns per-cluster embeddings alongside segments. The meeting screen gains an inline collapsible speaker mapping UI that auto-suggests names from known profiles and rewrites transcripts on apply.

**Tech Stack:** Python, Textual (TUI), NumPy (cosine similarity), SpeechBrain ECAPA-TDNN (embeddings already in use)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/meetscribe/storage/speakers.py` (new) | Speaker registry CRUD, embedding matching, `speakers.json` I/O |
| `src/meetscribe/transcription/diarize.py` (modify) | Return per-cluster embeddings alongside `SpeakerSegment` list |
| `src/meetscribe/transcription/whisper.py` (modify) | Pass cluster embeddings through from diarize to caller |
| `src/meetscribe/tui/screens/meeting.py` (modify) | Inline speaker mapping UI, apply names, auto-expand after transcription |
| `src/meetscribe/storage/vault.py` (modify) | No schema changes needed — `save_metadata`/`load_metadata` already handle arbitrary dicts |
| `tests/test_speakers.py` (new) | Unit tests for speaker registry and matching |
| `tests/test_transcript_rewrite.py` (new) | Unit tests for transcript rewriting logic |
| `tests/test_diarize_embeddings.py` (new) | Unit tests for embedding return from diarization |

---

### Task 1: Speaker Registry — Data Model and CRUD

**Files:**

- Create: `src/meetscribe/storage/speakers.py`
- Test: `tests/test_speakers.py`

- [ ] **Step 1: Write failing tests for SpeakerRegistry**

```python
# tests/test_speakers.py
import json
from pathlib import Path

import pytest

from meetscribe.storage.speakers import SpeakerRegistry, SpeakerProfile


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "speakers.json"


@pytest.fixture
def registry(registry_path):
    return SpeakerRegistry(registry_path)


class TestSpeakerRegistryCreate:
    def test_create_speaker(self, registry):
        profile = registry.create_speaker("Alice")
        assert profile.name == "Alice"
        assert profile.id.startswith("sp_")
        assert len(profile.id) == 9  # sp_ + 6 hex
        assert profile.embeddings == []

    def test_create_speaker_persists(self, registry, registry_path):
        registry.create_speaker("Alice")
        data = json.loads(registry_path.read_text())
        assert len(data["speakers"]) == 1
        assert data["speakers"][0]["name"] == "Alice"

    def test_create_multiple_speakers(self, registry):
        alice = registry.create_speaker("Alice")
        bob = registry.create_speaker("Bob")
        assert alice.id != bob.id
        assert len(registry.list_speakers()) == 2


class TestSpeakerRegistryGet:
    def test_get_speaker_by_id(self, registry):
        alice = registry.create_speaker("Alice")
        found = registry.get_speaker(alice.id)
        assert found is not None
        assert found.name == "Alice"

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get_speaker("sp_000000") is None


class TestSpeakerRegistryUpdate:
    def test_rename_speaker(self, registry):
        alice = registry.create_speaker("Alice")
        registry.rename_speaker(alice.id, "Alice Smith")
        found = registry.get_speaker(alice.id)
        assert found.name == "Alice Smith"


class TestSpeakerRegistryLoadExisting:
    def test_loads_existing_file(self, registry_path):
        data = {
            "speakers": [
                {
                    "id": "sp_aaaaaa",
                    "name": "Alice",
                    "embeddings": [],
                    "created_at": "2026-04-15T10:00:00Z",
                    "updated_at": "2026-04-15T10:00:00Z",
                }
            ],
            "match_threshold": 0.65,
        }
        registry_path.write_text(json.dumps(data))
        reg = SpeakerRegistry(registry_path)
        assert len(reg.list_speakers()) == 1
        assert reg.list_speakers()[0].name == "Alice"

    def test_empty_file_creates_default(self, registry_path):
        reg = SpeakerRegistry(registry_path)
        assert reg.list_speakers() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_speakers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meetscribe.storage.speakers'`

- [ ] **Step 3: Implement SpeakerRegistry**

```python
# src/meetscribe/storage/speakers.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_speakers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/storage/speakers.py tests/test_speakers.py
git commit -m "feat: add SpeakerRegistry with CRUD and persistence"
```

---

### Task 2: Embedding Matching Logic

**Files:**

- Modify: `src/meetscribe/storage/speakers.py`
- Test: `tests/test_speakers.py`

- [ ] **Step 1: Write failing tests for matching**

Add to `tests/test_speakers.py`:

```python
import numpy as np


class TestSpeakerRegistryAddEmbedding:
    def test_add_embedding(self, registry):
        alice = registry.create_speaker("Alice")
        registry.add_embedding(alice.id, [0.1] * 192, "2026-04-15/standup")
        found = registry.get_speaker(alice.id)
        assert len(found.embeddings) == 1
        assert found.embeddings[0].source_meeting == "2026-04-15/standup"

    def test_embedding_cap(self, registry):
        alice = registry.create_speaker("Alice")
        for i in range(12):
            registry.add_embedding(alice.id, [float(i)] * 192, f"meeting-{i}")
        found = registry.get_speaker(alice.id)
        assert len(found.embeddings) == 10
        # Oldest dropped — first remaining should be meeting-2
        assert found.embeddings[0].source_meeting == "meeting-2"

    def test_add_embedding_nonexistent_speaker(self, registry):
        # Should not raise
        registry.add_embedding("sp_000000", [0.1] * 192, "meeting")


class TestSpeakerMatching:
    def test_match_known_speaker(self, registry):
        alice = registry.create_speaker("Alice")
        emb = np.random.randn(192).tolist()
        registry.add_embedding(alice.id, emb, "meeting-1")

        # Same embedding should match with high similarity
        from meetscribe.storage.speakers import match_speakers
        matches = match_speakers({"Speaker 1": np.array(emb)}, registry)
        assert "Speaker 1" in matches
        assert matches["Speaker 1"].name == "Alice"

    def test_no_match_below_threshold(self, registry):
        alice = registry.create_speaker("Alice")
        registry.add_embedding(alice.id, [1.0] * 192, "meeting-1")

        from meetscribe.storage.speakers import match_speakers
        # Orthogonal vector should not match
        candidate = np.zeros(192)
        candidate[0] = 1.0
        matches = match_speakers({"Speaker 1": candidate}, registry)
        assert "Speaker 1" not in matches

    def test_no_speakers_returns_empty(self, registry):
        from meetscribe.storage.speakers import match_speakers
        emb = np.random.randn(192)
        matches = match_speakers({"Speaker 1": emb}, registry)
        assert matches == {}

    def test_no_embeddings_returns_empty(self, registry):
        registry.create_speaker("Alice")  # No embeddings added
        from meetscribe.storage.speakers import match_speakers
        emb = np.random.randn(192)
        matches = match_speakers({"Speaker 1": emb}, registry)
        assert matches == {}

    def test_two_clusters_same_best_match(self, registry):
        """When two clusters match the same speaker, highest similarity wins."""
        alice = registry.create_speaker("Alice")
        emb = np.random.randn(192)
        emb_normalized = emb / np.linalg.norm(emb)
        registry.add_embedding(alice.id, emb_normalized.tolist(), "meeting-1")

        from meetscribe.storage.speakers import match_speakers
        # Cluster 1: very close to Alice
        close = emb_normalized + np.random.randn(192) * 0.01
        # Cluster 2: somewhat close to Alice
        farther = emb_normalized + np.random.randn(192) * 0.1
        candidates = {"Speaker 1": close, "Speaker 2": farther}
        matches = match_speakers(candidates, registry)
        # Only one should match Alice
        alice_matches = [k for k, v in matches.items() if v.id == alice.id]
        assert len(alice_matches) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_speakers.py::TestSpeakerMatching -v`
Expected: FAIL — `ImportError: cannot import name 'match_speakers'`

- [ ] **Step 3: Implement match_speakers**

Add to `src/meetscribe/storage/speakers.py`:

```python
import numpy as np


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
    # Store (cluster_label, speaker_id, similarity)
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
```

Note: add `import numpy as np` to the top of the file (after the existing imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_speakers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/storage/speakers.py tests/test_speakers.py
git commit -m "feat: add speaker embedding matching with cosine similarity"
```

---

### Task 3: Diarization Returns Per-Cluster Embeddings

**Files:**

- Modify: `src/meetscribe/transcription/diarize.py:100-165`
- Test: `tests/test_diarize_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_diarize_embeddings.py
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from meetscribe.transcription.diarize import (
    _cluster_speakers,
    DiarizationResult,
)


class TestClusterSpeakersEmbeddings:
    def test_returns_diarization_result(self):
        """_cluster_speakers returns a DiarizationResult with segments and embeddings."""
        # Create 4 segments, 2 per speaker (using clearly separable embeddings)
        emb_a = np.array([1.0] * 192)
        emb_b = np.array([-1.0] * 192)
        segments = [
            (0.0, 5.0, emb_a),
            (5.0, 10.0, emb_a),
            (10.0, 15.0, emb_b),
            (15.0, 20.0, emb_b),
        ]
        result = _cluster_speakers(segments, num_speakers=2)
        assert isinstance(result, DiarizationResult)
        assert len(result.segments) == 4
        assert len(result.cluster_embeddings) == 2
        # Each cluster embedding should be 192-dim
        for label, emb in result.cluster_embeddings.items():
            assert emb.shape == (192,)
            assert label.startswith("Speaker ")

    def test_single_segment(self):
        emb = np.array([1.0] * 192)
        segments = [(0.0, 5.0, emb)]
        result = _cluster_speakers(segments)
        assert len(result.segments) == 1
        assert len(result.cluster_embeddings) == 1
        assert "Speaker 1" in result.cluster_embeddings

    def test_empty_segments(self):
        result = _cluster_speakers([])
        assert result.segments == []
        assert result.cluster_embeddings == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diarize_embeddings.py -v`
Expected: FAIL — `ImportError: cannot import name 'DiarizationResult'`

- [ ] **Step 3: Modify diarize.py**

Add `DiarizationResult` dataclass after `SpeakerSegment`:

```python
@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    cluster_embeddings: dict[str, np.ndarray]
```

Modify `_cluster_speakers` (lines 100-132) to return `DiarizationResult` instead of `list[SpeakerSegment]`:

```python
def _cluster_speakers(
    segments: list[tuple[float, float, np.ndarray]],
    num_speakers: int | None = None,
    threshold: float = 1.2,
) -> DiarizationResult:
    """Cluster embeddings to identify speakers.

    Uses cosine distance with average linkage for better speaker separation.
    If num_speakers is provided, forces exactly that many clusters.
    Otherwise, uses the distance threshold to determine cluster count.
    """
    if not segments:
        return DiarizationResult(segments=[], cluster_embeddings={})

    embeddings = np.array([s[2] for s in segments])

    if len(embeddings) == 1:
        return DiarizationResult(
            segments=[SpeakerSegment(segments[0][0], segments[0][1], "Speaker 1")],
            cluster_embeddings={"Speaker 1": embeddings[0]},
        )

    # Compute cosine distances and cluster with average linkage
    distances = pdist(embeddings, metric="cosine")
    linkage_matrix = linkage(distances, method="average")

    if num_speakers and num_speakers > 0:
        labels = fcluster(linkage_matrix, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # Build speaker segments
    result_segments: list[SpeakerSegment] = []
    for (start, end, _), label in zip(segments, labels):
        result_segments.append(SpeakerSegment(start, end, f"Speaker {label}"))

    # Compute average embedding per cluster
    cluster_embeddings: dict[str, np.ndarray] = {}
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        cluster_embs = embeddings[mask]
        cluster_embeddings[f"Speaker {label}"] = cluster_embs.mean(axis=0)

    return DiarizationResult(segments=result_segments, cluster_embeddings=cluster_embeddings)
```

Modify `diarize` function (lines 135-165) to return `DiarizationResult`:

```python
def diarize(
    audio_path: Path,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on an audio file.

    Returns a DiarizationResult with speaker segments and per-cluster embeddings.
    """
    log.info("Loading audio: %s", audio_path)
    waveform = _load_audio(audio_path)
    duration = waveform.shape[1] / SAMPLE_RATE
    log.info("Audio duration: %.1fs", duration)

    log.info("Loading speaker embedding model...")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache" / "meetscribe" / "spkrec-ecapa"),
    )

    log.info("Extracting speaker embeddings...")
    segments = _extract_embeddings(waveform, model)
    log.info("Extracted %d segment embeddings (silent segments skipped)", len(segments))

    log.info("Clustering speakers...")
    result = _cluster_speakers(segments, num_speakers=num_speakers)

    speakers = {s.speaker for s in result.segments}
    log.info("Identified %d speakers", len(speakers))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diarize_embeddings.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/transcription/diarize.py tests/test_diarize_embeddings.py
git commit -m "feat: return per-cluster embeddings from diarization"
```

---

### Task 4: Update whisper.py to Pass Embeddings Through

**Files:**

- Modify: `src/meetscribe/transcription/whisper.py:112-177`
- Test: `tests/test_whisper.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_whisper.py`:

```python
class TestTranscribeAudioDiarization:
    @patch("meetscribe.transcription.whisper._load_model")
    @patch("meetscribe.transcription.whisper.diarize")
    @patch("meetscribe.transcription.whisper.assign_speakers_to_transcript")
    def test_returns_cluster_embeddings(self, mock_assign, mock_diarize, mock_load):
        from meetscribe.transcription.diarize import DiarizationResult, SpeakerSegment

        mock_model = MagicMock()
        mock_load.return_value = mock_model
        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.")
        mock_info = MagicMock(duration=5.0)
        mock_model.transcribe.return_value = ([seg1], mock_info)

        import numpy as np
        cluster_embs = {"Speaker 1": np.array([0.1] * 192)}
        mock_diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(0.0, 5.0, "Speaker 1")],
            cluster_embeddings=cluster_embs,
        )
        mock_assign.return_value = [("Speaker 1", "Hello.")]

        from meetscribe.transcription.whisper import transcribe_audio
        result = transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
            enable_diarization=True,
        )

        assert isinstance(result, tuple)
        transcript_text, embeddings = result
        assert "Speaker 1" in transcript_text
        assert "Speaker 1" in embeddings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_whisper.py::TestTranscribeAudioDiarization -v`
Expected: FAIL — result is a string, not a tuple

- [ ] **Step 3: Modify transcribe_audio return type**

In `src/meetscribe/transcription/whisper.py`, change `transcribe_audio` (lines 112-177) to return `tuple[str, dict | None]`:

```python
def transcribe_audio(
    audio_path: Path,
    model_name: str,
    meeting_name: str,
    meeting_date: str,
    enable_diarization: bool = False,
    num_speakers: int | None = None,
    on_segment: callable | None = None,
    custom_vocabulary: list[str] | None = None,
) -> tuple[str, dict | None]:
    """Transcribe an audio file and return formatted markdown transcript.

    Args:
        on_segment: Optional callback called with (segment_index, timestamp, text)
                    after each segment is transcribed. Use for live UI updates.
        custom_vocabulary: Optional list of words/phrases to prime the model with.

    Returns:
        Tuple of (formatted_transcript, cluster_embeddings).
        cluster_embeddings is a dict mapping speaker labels to numpy arrays,
        or None if diarization was not enabled.
    """
    import time as _time
    t0 = _time.monotonic()

    log.info("Loading model: %s", model_name)
    model = _load_model(model_name)
    log.info("Model loaded in %.1fs", _time.monotonic() - t0)

    log.info("Starting transcription of %s", audio_path)
    t1 = _time.monotonic()
    initial_prompt = ", ".join(custom_vocabulary) if custom_vocabulary else None
    segments, info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True, language="en",
        initial_prompt=initial_prompt,
    )
    log.info("Transcribe call returned (generator ready) in %.1fs", _time.monotonic() - t1)

    # Stream segments — call on_segment as each one arrives
    log.info("Iterating segments...")
    segment_list: list = []
    for segment in segments:
        segment_list.append(segment)
        if on_segment:
            timestamp = format_timestamp(segment.start)
            text = segment.text.strip()
            on_segment(len(segment_list) - 1, timestamp, text)
        if len(segment_list) % 50 == 0:
            log.info("Transcribed %d segments so far (at %.1fs in audio)",
                     len(segment_list), segment.end)

    duration = _format_duration(info.duration)
    log.info("Transcription complete: %d segments, %s duration, took %.1fs",
             len(segment_list), duration, _time.monotonic() - t0)

    speaker_labels = None
    cluster_embeddings = None
    if enable_diarization:
        log.info("Running speaker diarization...")
        from meetscribe.transcription.diarize import diarize, assign_speakers_to_transcript
        diarization_result = diarize(audio_path, num_speakers=num_speakers)
        speaker_labels = assign_speakers_to_transcript(segment_list, diarization_result.segments)
        cluster_embeddings = {
            label: emb.tolist() for label, emb in diarization_result.cluster_embeddings.items()
        }
        log.info("Diarization complete")

    transcript = format_transcript(
        segments=segment_list,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        model=model_name,
        duration=duration,
        speaker_labels=speaker_labels,
    )

    return transcript, cluster_embeddings
```

- [ ] **Step 4: Fix existing test that expects a string return**

Update `TestTranscribeAudio.test_returns_formatted_transcript` in `tests/test_whisper.py`:

```python
class TestTranscribeAudio:
    @patch("meetscribe.transcription.whisper.WhisperModel")
    def test_returns_formatted_transcript(self, mock_model_cls):
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.")
        seg2 = MagicMock(start=5.0, end=10.0, text=" World.")
        mock_info = MagicMock(duration=10.0)
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        result, embeddings = transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
        )

        assert "[00:00:00] Hello." in result
        assert "[00:00:05] World." in result
        assert embeddings is None
        mock_model_cls.assert_called_once()
        call_args = mock_model_cls.call_args
        assert call_args[1] == {"device": "cpu", "compute_type": "int8"}
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_whisper.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/meetscribe/transcription/whisper.py tests/test_whisper.py
git commit -m "feat: pass cluster embeddings through from diarization"
```

---

### Task 5: Transcript Rewriting Logic

**Files:**

- Modify: `src/meetscribe/storage/speakers.py`
- Test: `tests/test_transcript_rewrite.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcript_rewrite.py
import pytest

from meetscribe.storage.speakers import rewrite_transcript


class TestRewriteTranscript:
    def test_replaces_speaker_labels(self):
        transcript = (
            "---\nmeeting: Standup\n---\n\n"
            "**Speaker 1:**\n[00:00:05] Hello\n\n"
            "**Speaker 2:**\n[00:00:10] Hi there\n\n"
        )
        speaker_map = {
            "Speaker 1": "Alice",
            "Speaker 2": "Bob",
        }
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice:**" in result
        assert "**Bob:**" in result
        assert "**Speaker 1:**" not in result
        assert "**Speaker 2:**" not in result

    def test_partial_map(self):
        transcript = (
            "**Speaker 1:**\n[00:00:05] Hello\n\n"
            "**Speaker 2:**\n[00:00:10] Hi\n\n"
        )
        speaker_map = {"Speaker 1": "Alice"}
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice:**" in result
        assert "**Speaker 2:**" in result

    def test_remap_already_named(self):
        """When re-mapping, find the current name and replace it."""
        transcript = "**Alice:**\n[00:00:05] Hello\n\n"
        speaker_map = {"Alice": "Alice Smith"}
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice Smith:**" in result
        assert "**Alice:**" not in result

    def test_preserves_frontmatter(self):
        transcript = "---\nmeeting: Standup\ndate: 2026-04-15\n---\n\n**Speaker 1:**\n[00:00:05] Hello\n"
        speaker_map = {"Speaker 1": "Alice"}
        result = rewrite_transcript(transcript, speaker_map)
        assert result.startswith("---\nmeeting: Standup")
        assert "**Alice:**" in result

    def test_empty_map_returns_unchanged(self):
        transcript = "**Speaker 1:**\n[00:00:05] Hello\n"
        result = rewrite_transcript(transcript, {})
        assert result == transcript
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transcript_rewrite.py -v`
Expected: FAIL — `ImportError: cannot import name 'rewrite_transcript'`

- [ ] **Step 3: Implement rewrite_transcript**

Add to `src/meetscribe/storage/speakers.py`:

```python
import re as _re


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcript_rewrite.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/storage/speakers.py tests/test_transcript_rewrite.py
git commit -m "feat: add transcript rewriting for speaker name replacement"
```

---

### Task 6: Update Meeting Screen Caller to Handle New Return Type

**Files:**

- Modify: `src/meetscribe/tui/screens/meeting.py:266-323`

This task updates `_run_transcription` to unpack the new `(transcript, cluster_embeddings)` tuple from `transcribe_audio` and store the embeddings for the speaker mapping UI (Task 7).

- [ ] **Step 1: Add instance variable for pending embeddings**

In `MeetingScreen.__init__` (line 84-86), add storage for pending cluster embeddings:

```python
def __init__(self, meeting: MeetingInfo) -> None:
    super().__init__()
    self.meeting = meeting
    self._pending_cluster_embeddings: dict | None = None
```

- [ ] **Step 2: Update _run_transcription to unpack tuple**

In `_run_transcription` (line 298-310), change:

```python
            transcript = transcribe_audio(
```

to:

```python
            transcript, cluster_embeddings = transcribe_audio(
```

After line 310 (`transcript_path.write_text(transcript)`), add:

```python
            self._pending_cluster_embeddings = cluster_embeddings
```

- [ ] **Step 3: Run existing tests to ensure nothing broke**

Run: `pytest tests/ -v`
Expected: All PASS (TUI code isn't covered by existing unit tests, but transcript/whisper tests should still pass)

- [ ] **Step 4: Commit**

```bash
git add src/meetscribe/tui/screens/meeting.py
git commit -m "feat: store cluster embeddings from transcription for speaker mapping"
```

---

### Task 7: Inline Speaker Mapping UI

**Files:**

- Modify: `src/meetscribe/tui/screens/meeting.py`

This is the largest task — adds the collapsible speaker mapping section to the transcript tab.

- [ ] **Step 1: Add imports**

At the top of `meeting.py`, add to the existing imports:

```python
from textual.widgets import Collapsible
```

And add a new import for the speaker module:

```python
from meetscribe.storage.speakers import SpeakerRegistry, match_speakers, rewrite_transcript
```

- [ ] **Step 2: Add CSS for speaker mapping section**

Add to the `CSS` string in `MeetingScreen` (after the existing `#num-speakers` rule):

```css
    #speaker-mapping {
        display: none;
        height: auto;
        max-height: 15;
    }

    #speaker-mapping.visible {
        display: block;
    }

    .speaker-row {
        height: 3;
        layout: horizontal;
    }

    .speaker-label {
        width: 20;
        content-align-vertical: middle;
    }

    .speaker-input {
        width: 1fr;
    }

    .match-indicator {
        width: 12;
        content-align-vertical: middle;
    }
```

- [ ] **Step 3: Add speaker mapping section to _compose_transcript_tab**

Modify `_compose_transcript_tab` (lines 131-147). Insert the `Collapsible` between the controls bar and the loading indicator:

```python
    def _compose_transcript_tab(self) -> Vertical:
        model_options = [(m, m) for m in AVAILABLE_MODELS]
        config = self.app.config if hasattr(self, "app") and self.app else None
        default_model = config.transcription.default_model if config else "base"

        return Vertical(
            Horizontal(
                Select(model_options, value=default_model, id="whisper-model"),
                Checkbox("Identify speakers", id="diarize-checkbox"),
                Input(placeholder="# speakers", id="num-speakers", max_length=2),
                Button("Transcribe", id="transcribe-btn", variant="primary"),
                Button("Regenerate", id="regenerate-transcript-btn"),
                classes="controls-bar",
            ),
            Collapsible(
                Static("No speakers detected yet.", id="speaker-mapping-content"),
                Button("Apply Names", id="apply-speakers-btn", variant="primary"),
                title="Speaker Mapping",
                id="speaker-mapping",
                collapsed=True,
            ),
            LoadingIndicator(id="transcript-loading"),
            Markdown("*No transcript yet. Select a model and click Transcribe.*", id="transcript-view"),
        )
```

- [ ] **Step 4: Add method to populate speaker mapping inputs**

Add a method that dynamically populates the speaker mapping section with `Input` widgets:

```python
    def _populate_speaker_mapping(
        self,
        speaker_labels: list[str],
        suggestions: dict[str, str] | None = None,
    ) -> None:
        """Populate the speaker mapping section with input fields.

        Args:
            speaker_labels: Detected speaker labels, e.g. ["Speaker 1", "Speaker 2"]
            suggestions: Optional dict mapping label to suggested name from auto-matching
        """
        suggestions = suggestions or {}
        collapsible = self.query_one("#speaker-mapping", Collapsible)

        # Remove old content (keep the Apply button)
        for widget in list(collapsible.query(".speaker-row")):
            widget.remove()
        try:
            collapsible.query_one("#speaker-mapping-content").remove()
        except Exception:
            pass

        # Add a row per speaker
        apply_btn = collapsible.query_one("#apply-speakers-btn", Button)
        for label in speaker_labels:
            suggested = suggestions.get(label, "")
            indicator = "(matched)" if suggested else ""
            row = Horizontal(
                Static(f"{label} →", classes="speaker-label"),
                Input(
                    value=suggested,
                    placeholder="Enter name",
                    id=f"speaker-input-{label.replace(' ', '-').lower()}",
                    classes="speaker-input",
                ),
                Static(indicator, classes="match-indicator"),
                classes="speaker-row",
            )
            collapsible.mount(row, before=apply_btn)

        # Store the label list for apply handler
        self._speaker_labels = speaker_labels

        # Show and expand
        collapsible.add_class("visible")
        collapsible.collapsed = False
```

- [ ] **Step 5: Add auto-matching after transcription**

In `_run_transcription`, after storing `self._pending_cluster_embeddings`, add auto-matching and UI population. Replace the final success block (after `transcript_path.write_text(transcript)`) with:

```python
            self._pending_cluster_embeddings = cluster_embeddings

            # Final update with full formatted transcript
            self.app.call_from_thread(self.query_one("#transcript-view", Markdown).update, transcript)

            # If diarization was used, set up speaker mapping
            if cluster_embeddings:
                config = self.app.config
                speakers_path = Path(config.config_dir) / "speakers.json"
                registry = SpeakerRegistry(speakers_path)

                # Auto-match against known speakers
                import numpy as np
                np_embeddings = {
                    label: np.array(emb) for label, emb in cluster_embeddings.items()
                }
                matches = match_speakers(np_embeddings, registry)
                suggestions = {label: profile.name for label, profile in matches.items()}
                labels = sorted(cluster_embeddings.keys())

                self.app.call_from_thread(self._populate_speaker_mapping, labels, suggestions)

            elapsed = _time.monotonic() - t_start
            m, s = divmod(int(elapsed), 60)
            self.app.call_from_thread(self.notify, f"Transcription complete! ({m}m {s}s)")
```

- [ ] **Step 6: Add Apply Names handler**

```python
    @on(Button.Pressed, "#apply-speakers-btn")
    def do_apply_speaker_names(self) -> None:
        """Apply speaker name mappings to the transcript."""
        if not hasattr(self, "_speaker_labels"):
            self.notify("No speakers to map.", severity="error")
            return

        collapsible = self.query_one("#speaker-mapping", Collapsible)
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        speakers_path = Path(config.config_dir) / "speakers.json"
        registry = SpeakerRegistry(speakers_path)

        # Collect name inputs
        speaker_map: dict[str, dict] = {}
        rewrite_map: dict[str, str] = {}
        metadata = load_metadata(self.meeting.path)
        existing_map = metadata.get("speaker_map", {})

        for label in self._speaker_labels:
            input_id = f"speaker-input-{label.replace(' ', '-').lower()}"
            try:
                name = self.query_one(f"#{input_id}", Input).value.strip()
            except Exception:
                continue
            if not name:
                continue

            # Find or create speaker profile
            import numpy as np
            matched_profile = None
            if self._pending_cluster_embeddings and label in self._pending_cluster_embeddings:
                np_emb = np.array(self._pending_cluster_embeddings[label])
                matches = match_speakers({label: np_emb}, registry)
                if label in matches and matches[label].name == name:
                    matched_profile = matches[label]

            if not matched_profile:
                # Check if there's an existing profile with this name
                for s in registry.list_speakers():
                    if s.name == name:
                        matched_profile = s
                        break

            if not matched_profile:
                matched_profile = registry.create_speaker(name)

            # Add embedding from this meeting
            if self._pending_cluster_embeddings and label in self._pending_cluster_embeddings:
                meeting_ref = f"{self.meeting.date}/{self.meeting.name}"
                registry.add_embedding(
                    matched_profile.id,
                    self._pending_cluster_embeddings[label],
                    meeting_ref,
                )

            # Build the rewrite map: find what's currently in the transcript
            current_label = label
            if label in existing_map:
                current_label = existing_map[label].get("name", label)
            rewrite_map[current_label] = name

            speaker_map[label] = {
                "speaker_id": matched_profile.id,
                "name": name,
                "original_label": label,
            }

        if not rewrite_map:
            self.notify("No names entered.", severity="warning")
            return

        # Rewrite transcript file
        transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
        if transcripts:
            transcript_text = transcripts[0].read_text()
            updated = rewrite_transcript(transcript_text, rewrite_map)
            transcripts[0].write_text(updated)
            self.query_one("#transcript-view", Markdown).update(updated)

        # Save metadata
        save_metadata(self.meeting.path, {"speaker_map": speaker_map})

        # Clear match indicators
        for widget in collapsible.query(".match-indicator"):
            widget.update("")

        self.notify("Speaker names applied!")
```

- [ ] **Step 7: Load existing speaker map on mount**

Add to `_load_metadata` (lines 181-185):

```python
    def _load_metadata(self) -> None:
        meta = load_metadata(self.meeting.path)
        num_speakers = meta.get("num_speakers")
        if num_speakers is not None:
            self.query_one("#num-speakers", Input).value = str(num_speakers)

        # Load existing speaker map for display
        speaker_map = meta.get("speaker_map")
        if speaker_map:
            labels = sorted(speaker_map.keys())
            suggestions = {label: info["name"] for label, info in speaker_map.items()}
            self._populate_speaker_mapping(labels, suggestions)
            # Keep it collapsed since it's existing data
            self.query_one("#speaker-mapping", Collapsible).collapsed = True
```

- [ ] **Step 8: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/meetscribe/tui/screens/meeting.py
git commit -m "feat: add inline speaker mapping UI with auto-matching"
```

---

### Task 8: Config Directory Access

**Files:**

- Modify: `src/meetscribe/tui/screens/meeting.py`

The speaker mapping UI references `config.config_dir` to find `speakers.json`. Verify this exists on the config object.

- [ ] **Step 1: Check how config_dir is accessed**

Read `src/meetscribe/config.py` to verify the config object exposes a `config_dir` property or attribute. If it doesn't exist, we need to add it or use a hardcoded path.

- [ ] **Step 2: If config_dir doesn't exist, use the standard path**

Replace `Path(config.config_dir) / "speakers.json"` in both locations in `meeting.py` with:

```python
speakers_path = Path.home() / ".config" / "meetscribe" / "speakers.json"
```

This matches the existing config path convention documented in CLAUDE.md.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit (if changes were needed)**

```bash
git add src/meetscribe/tui/screens/meeting.py
git commit -m "fix: use standard config path for speakers.json"
```

---

### Task 9: End-to-End Manual Test

**Files:** None (manual verification)

- [ ] **Step 1: Install in dev mode**

Run: `pip install -e .`

- [ ] **Step 2: Launch and test with diarization**

Run: `meetscribe`

1. Open an existing meeting with a recording
2. Check "Identify speakers" and set # speakers
3. Click Transcribe
4. Verify: Speaker Mapping section appears and auto-expands after transcription
5. Verify: Any previously-seen speakers show auto-suggested names
6. Enter names for unnamed speakers
7. Click "Apply Names"
8. Verify: Transcript markdown updates with real names
9. Verify: `_metadata.json` has `speaker_map` with `original_label` backup
10. Verify: `~/.config/meetscribe/speakers.json` has speaker profiles with embeddings

- [ ] **Step 3: Test re-opening a meeting**

1. Go back and re-open the same meeting
2. Verify: Speaker Mapping section shows (collapsed) with previously applied names
3. Expand it, change a name, click Apply
4. Verify: Transcript updates with new name

- [ ] **Step 4: Test cross-meeting recognition**

1. Record/transcribe a new meeting with some of the same speakers
2. Verify: Auto-suggested names appear for recognized speakers

- [ ] **Step 5: Final commit**

Run: `pytest tests/ -v` — ensure all tests pass.

```bash
git add -A
git commit -m "feat: speaker mapping with voice-embedding auto-recognition"
```
