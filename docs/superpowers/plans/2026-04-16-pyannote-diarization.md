# Pyannote Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SpeechBrain diarization with pyannote-audio's pretrained pipeline for significantly better speaker turn detection.

**Architecture:** Drop-in replacement of `diarize()` internals. pyannote handles segmentation, embedding, and clustering via a single pipeline call. The `DiarizationResult` return type stays the same so all downstream code (whisper.py, meeting.py, speakers.py) is unaffected. Speaker profiles get reset due to embedding dimension change (192 → 512).

**Tech Stack:** pyannote.audio 3.3+, PyTorch (MPS/CPU), HuggingFace Hub

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/meetscribe/transcription/diarize.py` (rewrite) | Replace SpeechBrain internals with pyannote pipeline |
| `src/meetscribe/config.py` (modify) | Add `huggingface_token` field |
| `src/meetscribe/storage/speakers.py` (modify) | Add embedding dimension validation |
| `src/meetscribe/tui/app.py` (modify) | Update log suppression list |
| `pyproject.toml` (modify) | Add pyannote.audio dependency |
| `tests/test_diarize_embeddings.py` (rewrite) | Replace cluster tests with pyannote mock tests |
| `tests/test_speakers.py` (modify) | Add embedding reset test |
| `tests/test_config.py` (modify) | Add huggingface_token test |

---

### Task 1: Add HuggingFace Token to Config

**Files:**

- Modify: `src/meetscribe/config.py:49-54,61-85,88-105`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
class TestHuggingFaceToken:
    def test_default_is_empty(self):
        cfg = default_config()
        assert cfg.huggingface_token == ""

    def test_roundtrip(self, config_file):
        cfg = default_config()
        cfg.huggingface_token = "hf_test123"
        save_config(cfg, config_file)
        loaded = load_config(config_file)
        assert loaded.huggingface_token == "hf_test123"

    def test_load_old_config_without_token(self, config_file):
        """Config files from before this field should load with empty default."""
        cfg = default_config()
        save_config(cfg, config_file)
        # Manually remove the token line to simulate old config
        content = config_file.read_text()
        content = "\n".join(
            line for line in content.splitlines()
            if "huggingface_token" not in line
        )
        config_file.write_text(content)
        loaded = load_config(config_file)
        assert loaded.huggingface_token == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_config.py::TestHuggingFaceToken -v`
Expected: FAIL — `AttributeError: 'MeetscribeConfig' has no attribute 'huggingface_token'`

- [ ] **Step 3: Add huggingface_token to config**

In `src/meetscribe/config.py`:

Add field to `MeetscribeConfig` dataclass (line 54):

```python
@dataclass
class MeetscribeConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)
    log_level: str = "INFO"
    huggingface_token: str = ""
```

Add to `_config_to_dict` — IMPORTANT: must be a scalar before sections (TOML gotcha). Add it right after `"log_level"`:

```python
def _config_to_dict(cfg: MeetscribeConfig) -> dict[str, Any]:
    return {
        "log_level": cfg.log_level,
        "huggingface_token": cfg.huggingface_token,
        "vault": {
            ...
```

Add to `_dict_to_config`:

```python
    if "huggingface_token" in data:
        cfg.huggingface_token = data["huggingface_token"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/config.py tests/test_config.py
git commit -m "feat: add huggingface_token to config"
```

---

### Task 2: Embedding Dimension Validation in SpeakerRegistry

**Files:**

- Modify: `src/meetscribe/storage/speakers.py:1-70`
- Test: `tests/test_speakers.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_speakers.py`:

```python
class TestEmbeddingDimensionReset:
    def test_resets_profiles_with_wrong_dimensions(self, registry_path):
        """Profiles with old 192-dim embeddings should be cleared on load."""
        data = {
            "speakers": [
                {
                    "id": "sp_aaaaaa",
                    "name": "Alice",
                    "embeddings": [
                        {
                            "vector": [0.1] * 192,
                            "source_meeting": "old-meeting",
                            "created_at": "2026-04-15T10:00:00Z",
                        }
                    ],
                    "created_at": "2026-04-15T10:00:00Z",
                    "updated_at": "2026-04-15T10:00:00Z",
                }
            ],
            "match_threshold": 0.65,
        }
        registry_path.write_text(json.dumps(data))
        reg = SpeakerRegistry(registry_path)
        # Profiles should be cleared due to dimension mismatch
        assert reg.list_speakers() == []

    def test_keeps_profiles_with_correct_dimensions(self, registry_path):
        """Profiles with 512-dim embeddings should be kept."""
        from meetscribe.storage.speakers import EMBEDDING_DIM
        data = {
            "speakers": [
                {
                    "id": "sp_aaaaaa",
                    "name": "Alice",
                    "embeddings": [
                        {
                            "vector": [0.1] * EMBEDDING_DIM,
                            "source_meeting": "new-meeting",
                            "created_at": "2026-04-15T10:00:00Z",
                        }
                    ],
                    "created_at": "2026-04-15T10:00:00Z",
                    "updated_at": "2026-04-15T10:00:00Z",
                }
            ],
            "match_threshold": 0.65,
        }
        registry_path.write_text(json.dumps(data))
        reg = SpeakerRegistry(registry_path)
        assert len(reg.list_speakers()) == 1

    def test_keeps_profiles_with_no_embeddings(self, registry_path):
        """Profiles without embeddings should not be cleared."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_speakers.py::TestEmbeddingDimensionReset -v`
Expected: FAIL — `ImportError: cannot import name 'EMBEDDING_DIM'`

- [ ] **Step 3: Add dimension validation**

In `src/meetscribe/storage/speakers.py`, add constant after `MAX_EMBEDDINGS_PER_SPEAKER`:

```python
MAX_EMBEDDINGS_PER_SPEAKER = 10
EMBEDDING_DIM = 512
```

Add validation at the end of `_load()`, after the for loop that loads speakers:

```python
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

            # Validate embedding dimensions — reset if model changed
            for speaker in self._speakers:
                for emb in speaker.embeddings:
                    if len(emb.vector) != EMBEDDING_DIM:
                        log.warning("Speaker profiles reset — embedding model changed.")
                        self._speakers = []
                        self._save()
                        return
```

Also add `import logging` and `log = logging.getLogger("meetscribe.speakers")` at the top of the file:

```python
import json
import logging
import secrets
...

log = logging.getLogger("meetscribe.speakers")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_speakers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/storage/speakers.py tests/test_speakers.py
git commit -m "feat: validate embedding dimensions and reset on model change"
```

---

### Task 3: Replace Diarization with Pyannote

**Files:**

- Modify: `pyproject.toml`
- Rewrite: `src/meetscribe/transcription/diarize.py`
- Rewrite: `tests/test_diarize_embeddings.py`
- Modify: `src/meetscribe/tui/app.py:29-32`

- [ ] **Step 1: Add pyannote.audio dependency**

In `pyproject.toml`, add to dependencies:

```toml
dependencies = [
    "textual>=0.79.0",
    "faster-whisper>=1.0.0",
    "sounddevice>=0.5.0",
    "soundfile>=0.12.0",
    "openai>=1.0.0",
    "jinja2>=3.1.0",
    "tomli>=2.0.0;python_version<'3.11'",
    "tomli-w>=1.0.0",
    "numpy>=1.24.0",
    "pyannote.audio>=3.3.0",
]
```

- [ ] **Step 2: Install the new dependency**

Run: `pipx install -e /Users/ian.bartholomew/Dev/meetscribe --python python3.13 --force`

- [ ] **Step 3: Update log suppression in app.py**

In `src/meetscribe/tui/app.py`, line 30-32, replace `"speechbrain"` with `"pyannote"`:

```python
    noisy_loggers = [
        "markdown_it", "httpcore", "httpx", "asyncio",
        "urllib3", "pyannote", "torch",
    ]
```

- [ ] **Step 4: Write failing tests for pyannote diarize**

Replace `tests/test_diarize_embeddings.py` entirely:

```python
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from meetscribe.transcription.diarize import (
    SpeakerSegment,
    DiarizationResult,
    assign_speakers_to_words,
    diarize,
    _get_hf_token,
)


def _mock_word(start, end, word):
    """Create a mock faster-whisper Word object."""
    w = MagicMock()
    w.start = start
    w.end = end
    w.word = word
    return w


class TestAssignSpeakersToWords:
    def test_single_speaker(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(0.5, 1.0, " world"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 5.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 1
        assert result[0] == ("Speaker 1", 0.0, "hello world")

    def test_speaker_change_mid_segment(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(0.5, 1.0, " from"),
            _mock_word(1.0, 1.5, " me"),
            _mock_word(2.0, 2.5, " hi"),
            _mock_word(2.5, 3.0, " back"),
        ]
        speaker_segments = [
            SpeakerSegment(0.0, 1.5, "Speaker 1"),
            SpeakerSegment(1.5, 5.0, "Speaker 2"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 2
        assert result[0] == ("Speaker 1", 0.0, "hello from me")
        assert result[1] == ("Speaker 2", 2.0, "hi back")

    def test_alternating_speakers(self):
        words = [
            _mock_word(0.0, 0.5, " yes"),
            _mock_word(1.0, 1.5, " no"),
            _mock_word(2.0, 2.5, " maybe"),
        ]
        speaker_segments = [
            SpeakerSegment(0.0, 0.8, "Speaker 1"),
            SpeakerSegment(0.8, 1.8, "Speaker 2"),
            SpeakerSegment(1.8, 3.0, "Speaker 1"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 3
        assert result[0] == ("Speaker 1", 0.0, "yes")
        assert result[1] == ("Speaker 2", 1.0, "no")
        assert result[2] == ("Speaker 1", 2.0, "maybe")

    def test_consecutive_same_speaker_grouped(self):
        words = [
            _mock_word(0.0, 0.5, " one"),
            _mock_word(0.5, 1.0, " two"),
            _mock_word(1.0, 1.5, " three"),
            _mock_word(1.5, 2.0, " four"),
        ]
        speaker_segments = [
            SpeakerSegment(0.0, 1.0, "Speaker 1"),
            SpeakerSegment(1.0, 3.0, "Speaker 1"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 1
        assert result[0] == ("Speaker 1", 0.0, "one two three four")

    def test_word_with_no_overlap(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(10.0, 10.5, " orphan"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 1.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 2
        assert result[0] == ("Speaker 1", 0.0, "hello")
        assert result[1] == ("Unknown", 10.0, "orphan")

    def test_empty_words(self):
        result = assign_speakers_to_words([], [SpeakerSegment(0.0, 5.0, "Speaker 1")])
        assert result == []

    def test_timestamp_is_first_word(self):
        words = [
            _mock_word(5.0, 5.5, " late"),
            _mock_word(5.5, 6.0, " start"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 10.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert result[0][1] == 5.0


class TestGetHfToken:
    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        assert _get_hf_token("") == "hf_from_env"

    def test_falls_back_to_config(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _get_hf_token("hf_from_config") == "hf_from_config"

    def test_env_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        assert _get_hf_token("hf_from_config") == "hf_from_env"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="HuggingFace token required"):
            _get_hf_token("")


class TestDiarize:
    @patch("meetscribe.transcription.diarize._get_embedding_model")
    @patch("meetscribe.transcription.diarize._get_pipeline")
    @patch("meetscribe.transcription.diarize._get_hf_token")
    def test_returns_diarization_result(self, mock_token, mock_pipeline_fn, mock_emb_fn):
        from pathlib import Path

        mock_token.return_value = "hf_fake"

        # Mock pyannote pipeline output
        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        # Create mock annotation with itertracks
        mock_segment_1 = MagicMock()
        mock_segment_1.start = 0.0
        mock_segment_1.end = 5.0
        mock_segment_2 = MagicMock()
        mock_segment_2.start = 5.0
        mock_segment_2.end = 10.0

        mock_output = MagicMock()
        mock_exclusive = MagicMock()
        mock_exclusive.itertracks.return_value = [
            (mock_segment_1, None, "SPEAKER_00"),
            (mock_segment_2, None, "SPEAKER_01"),
        ]
        mock_output.exclusive_speaker_diarization = mock_exclusive
        mock_pipeline.return_value = mock_output

        # Mock embedding model
        mock_emb_model = MagicMock()
        mock_emb_fn.return_value = mock_emb_model
        mock_emb_model.return_value = np.random.randn(1, 512)

        result = diarize(Path("/fake/audio.flac"), num_speakers=2)

        assert isinstance(result, DiarizationResult)
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Speaker 1"
        assert result.segments[1].speaker == "Speaker 2"
        assert result.segments[0].start == 0.0
        assert result.segments[1].start == 5.0
        assert len(result.cluster_embeddings) == 2
        mock_pipeline.assert_called_once()

    @patch("meetscribe.transcription.diarize._get_embedding_model")
    @patch("meetscribe.transcription.diarize._get_pipeline")
    @patch("meetscribe.transcription.diarize._get_hf_token")
    def test_passes_num_speakers(self, mock_token, mock_pipeline_fn, mock_emb_fn):
        from pathlib import Path

        mock_token.return_value = "hf_fake"
        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_exclusive = MagicMock()
        mock_exclusive.itertracks.return_value = []
        mock_output.exclusive_speaker_diarization = mock_exclusive
        mock_pipeline.return_value = mock_output

        mock_emb_fn.return_value = MagicMock()

        diarize(Path("/fake/audio.flac"), num_speakers=3)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["num_speakers"] == 3
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_diarize_embeddings.py -v`
Expected: FAIL — `_get_hf_token` and `_get_pipeline` don't exist yet

- [ ] **Step 6: Rewrite diarize.py**

Replace the entire content of `src/meetscribe/transcription/diarize.py`:

```python
"""Speaker diarization using pyannote-audio pretrained pipeline."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("meetscribe.diarize")


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    cluster_embeddings: dict[str, np.ndarray]


_pipeline = None
_embedding_model = None


def _get_hf_token(config_token: str = "") -> str:
    """Get HuggingFace token from env var or config."""
    token = os.environ.get("HF_TOKEN", "") or config_token
    if not token:
        raise RuntimeError(
            "HuggingFace token required for speaker diarization. "
            "Set HF_TOKEN env var or huggingface_token in config.toml. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )
    return token


def _get_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_pipeline(hf_token: str):
    """Load and cache the pyannote diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        log.info("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        device = _get_device()
        log.info("Using device: %s", device)
        _pipeline.to(device)
    return _pipeline


def _get_embedding_model(hf_token: str):
    """Load and cache the pyannote speaker embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from pyannote.audio.pipelines import SpeakerEmbedding
        log.info("Loading pyannote embedding model...")
        _embedding_model = SpeakerEmbedding(
            embedding="pyannote/embedding", token=hf_token
        )
    return _embedding_model


def diarize(
    audio_path: Path,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on an audio file.

    Returns a DiarizationResult with speaker segments and per-cluster embeddings.
    """
    from meetscribe.config import load_config
    config = load_config()
    hf_token = _get_hf_token(config.huggingface_token)

    pipeline = _get_pipeline(hf_token)

    log.info("Running diarization on %s", audio_path)
    kwargs = {}
    if num_speakers is not None and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(str(audio_path), **kwargs)

    # Use exclusive diarization (non-overlapping) for clean transcript assignment
    exclusive = output.exclusive_speaker_diarization

    # Map pyannote labels (SPEAKER_00) to our format (Speaker 1)
    label_map: dict[str, str] = {}
    segments: list[SpeakerSegment] = []
    for segment, _, speaker in exclusive.itertracks(yield_label=True):
        if speaker not in label_map:
            label_map[speaker] = f"Speaker {len(label_map) + 1}"
        segments.append(SpeakerSegment(
            start=segment.start,
            end=segment.end,
            speaker=label_map[speaker],
        ))

    speakers = set(label_map.values())
    log.info("Identified %d speakers", len(speakers))

    # Extract per-speaker embeddings from longest segment
    cluster_embeddings: dict[str, np.ndarray] = {}
    embedding_model = _get_embedding_model(hf_token)
    for pyannote_label, our_label in label_map.items():
        # Find longest segment for this speaker
        speaker_segs = [s for s in segments if s.speaker == our_label]
        if not speaker_segs:
            continue
        longest = max(speaker_segs, key=lambda s: s.end - s.start)
        try:
            from pyannote.core import Segment
            crop = Segment(longest.start, longest.end)
            embedding = embedding_model({"audio": str(audio_path), "start": crop.start, "end": crop.end})
            if hasattr(embedding, 'numpy'):
                emb_np = embedding.squeeze().numpy()
            elif isinstance(embedding, np.ndarray):
                emb_np = embedding.squeeze()
            else:
                emb_np = np.array(embedding).squeeze()
            cluster_embeddings[our_label] = emb_np
        except Exception:
            log.warning("Failed to extract embedding for %s", our_label)

    log.info("Diarization complete")
    return DiarizationResult(segments=segments, cluster_embeddings=cluster_embeddings)


def assign_speakers_to_transcript(
    transcript_segments: list,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, str]]:
    """Match transcript segments to speakers based on time overlap.

    Args:
        transcript_segments: faster-whisper segments with .start, .end, .text
        speaker_segments: diarization results

    Returns:
        List of (speaker_label, text) tuples.
    """
    results: list[tuple[str, str]] = []

    for tseg in transcript_segments:
        t_start = tseg.start
        t_end = tseg.end

        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(t_start, sseg.start)
            overlap_end = min(t_end, sseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        results.append((best_speaker, tseg.text.strip()))

    return results


def assign_speakers_to_words(
    words: list,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, float, str]]:
    """Assign speakers to individual words and group consecutive same-speaker words.

    Args:
        words: faster-whisper Word objects with .start, .end, .word
        speaker_segments: diarization results

    Returns:
        List of (speaker_label, start_time, text) tuples. Each tuple is a group
        of consecutive words from the same speaker.
    """
    if not words:
        return []

    word_speakers: list[tuple[str, float, str]] = []
    for w in words:
        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(w.start, sseg.start)
            overlap_end = min(w.end, sseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        word_speakers.append((best_speaker, w.start, w.word.strip()))

    groups: list[tuple[str, float, str]] = []
    current_speaker = word_speakers[0][0]
    current_start = word_speakers[0][1]
    current_words: list[str] = [word_speakers[0][2]]

    for speaker, start, word in word_speakers[1:]:
        if speaker == current_speaker:
            current_words.append(word)
        else:
            groups.append((current_speaker, current_start, " ".join(current_words)))
            current_speaker = speaker
            current_start = start
            current_words = [word]

    groups.append((current_speaker, current_start, " ".join(current_words)))
    return groups
```

- [ ] **Step 7: Run all tests**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/meetscribe/transcription/diarize.py src/meetscribe/tui/app.py tests/test_diarize_embeddings.py
git commit -m "feat: replace SpeechBrain diarization with pyannote-audio pipeline"
```

---

### Task 4: Update whisper.py Import for Config Token

**Files:**

- Modify: `src/meetscribe/transcription/whisper.py`

The `diarize()` function now loads the config internally to get the HF token, so no changes are needed in whisper.py for that. However, we should verify the import still works since diarize.py was rewritten.

- [ ] **Step 1: Run all tests to verify nothing broke**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Manual smoke test**

Run: `meetscribe`

1. Open a meeting with a recording
2. Check "Identify speakers", set # speakers
3. Click Transcribe
4. Verify diarization runs with pyannote (check log for "Loading pyannote diarization pipeline")
5. Verify speaker labels appear in transcript
6. Verify speaker mapping UI works

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve integration issues from pyannote migration"
```
