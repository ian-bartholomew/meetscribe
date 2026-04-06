# Meetscribe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python TUI app that records system audio via BlackHole, transcribes with faster-whisper, and generates meeting summaries using local LLMs, storing everything in an Obsidian vault.

**Architecture:** Monolithic single-package Python app (`meetscribe`) with internal modules for audio, transcription, summarization, templates, storage, and TUI. The Textual framework provides the interactive terminal interface. All state is file-based — no database.

**Tech Stack:** Python 3.11+, Textual (TUI), faster-whisper (transcription), sounddevice + soundfile (audio capture), openai client (LLM providers), Jinja2 (templates), TOML (config)

**Spec:** `docs/superpowers/specs/2026-04-06-meetscribe-design.md`

---

### Task 1: Project Scaffolding & Config Module

**Files:**

- Create: `pyproject.toml`
- Create: `src/meetscribe/__init__.py`
- Create: `src/meetscribe/__main__.py`
- Create: `src/meetscribe/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "meetscribe"
version = "0.1.0"
description = "TUI app for recording, transcribing, and summarizing meetings"
requires-python = ">=3.11"
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
]

[project.scripts]
meetscribe = "meetscribe.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["src/meetscribe"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init and entry point**

`src/meetscribe/__init__.py`:

```python
"""Meetscribe — record, transcribe, and summarize meetings."""

__version__ = "0.1.0"
```

`src/meetscribe/__main__.py`:

```python
from meetscribe.tui.app import MeetscribeApp


def main() -> None:
    app = MeetscribeApp()
    app.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write failing test for config**

`tests/__init__.py`: empty file.

`tests/test_config.py`:

```python
import os
from pathlib import Path

import pytest

from meetscribe.config import MeetscribeConfig, load_config, save_config, default_config


@pytest.fixture
def config_dir(tmp_path):
    return tmp_path / "meetscribe"


@pytest.fixture
def config_file(config_dir):
    return config_dir / "config.toml"


class TestDefaultConfig:
    def test_returns_config_with_defaults(self):
        cfg = default_config()
        assert cfg.audio.device_name == "BlackHole 2ch"
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.channels == 2
        assert cfg.transcription.default_model == "base"
        assert cfg.summarization.default_provider == "ollama"
        assert cfg.summarization.default_model == "llama3"
        assert cfg.summarization.endpoints["ollama"] == "http://localhost:11434/v1"
        assert cfg.summarization.endpoints["lmstudio"] == "http://localhost:1234/v1"
        assert cfg.vault.root == ""
        assert cfg.vault.meetings_folder == "Meetings"


class TestSaveAndLoadConfig:
    def test_roundtrip(self, config_file):
        cfg = default_config()
        cfg.vault.root = "/tmp/test-vault"
        save_config(cfg, config_file)

        loaded = load_config(config_file)
        assert loaded.vault.root == "/tmp/test-vault"
        assert loaded.audio.device_name == "BlackHole 2ch"
        assert loaded.summarization.endpoints["ollama"] == "http://localhost:11434/v1"

    def test_load_missing_file_returns_default(self, config_file):
        cfg = load_config(config_file)
        assert cfg.vault.root == ""
        assert cfg.audio.device_name == "BlackHole 2ch"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/ian.bartholomew/Dev/transription && pip install -e ".[dev]" 2>/dev/null; pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meetscribe.config'`

- [ ] **Step 5: Implement config module**

`src/meetscribe/config.py`:

```python
from __future__ import annotations

import tomli_w
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


CONFIG_DIR = Path.home() / ".config" / "meetscribe"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class AudioConfig:
    device_name: str = "BlackHole 2ch"
    sample_rate: int = 44100
    channels: int = 2


@dataclass
class TranscriptionConfig:
    default_model: str = "base"


@dataclass
class SummarizationConfig:
    default_provider: str = "ollama"
    default_model: str = "llama3"
    endpoints: dict[str, str] = field(default_factory=lambda: {
        "ollama": "http://localhost:11434/v1",
        "lmstudio": "http://localhost:1234/v1",
    })


@dataclass
class VaultConfig:
    root: str = ""
    meetings_folder: str = "Meetings"


@dataclass
class MeetscribeConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)


def default_config() -> MeetscribeConfig:
    return MeetscribeConfig()


def _config_to_dict(cfg: MeetscribeConfig) -> dict[str, Any]:
    return {
        "vault": {
            "root": cfg.vault.root,
            "meetings_folder": cfg.vault.meetings_folder,
        },
        "audio": {
            "device_name": cfg.audio.device_name,
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.audio.channels,
        },
        "transcription": {
            "default_model": cfg.transcription.default_model,
        },
        "summarization": {
            "default_provider": cfg.summarization.default_provider,
            "default_model": cfg.summarization.default_model,
            "endpoints": cfg.summarization.endpoints,
        },
    }


def _dict_to_config(data: dict[str, Any]) -> MeetscribeConfig:
    cfg = MeetscribeConfig()
    if "vault" in data:
        cfg.vault = VaultConfig(**data["vault"])
    if "audio" in data:
        cfg.audio = AudioConfig(**data["audio"])
    if "transcription" in data:
        cfg.transcription = TranscriptionConfig(**data["transcription"])
    if "summarization" in data:
        s = data["summarization"]
        cfg.summarization = SummarizationConfig(
            default_provider=s.get("default_provider", cfg.summarization.default_provider),
            default_model=s.get("default_model", cfg.summarization.default_model),
            endpoints=s.get("endpoints", cfg.summarization.endpoints),
        )
    return cfg


def save_config(cfg: MeetscribeConfig, path: Path | None = None) -> None:
    path = path or CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(_config_to_dict(cfg), f)


def load_config(path: Path | None = None) -> MeetscribeConfig:
    path = path or CONFIG_FILE
    if not path.exists():
        return default_config()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _dict_to_config(data)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding and config module"
```

---

### Task 2: Storage / Vault Module

**Files:**

- Create: `src/meetscribe/storage/__init__.py`
- Create: `src/meetscribe/storage/vault.py`
- Create: `tests/test_vault.py`

- [ ] **Step 1: Write failing test**

`tests/test_vault.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from meetscribe.storage.vault import MeetingStorage


@pytest.fixture
def vault(tmp_path):
    return MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")


class TestMeetingPath:
    def test_creates_correct_path(self, vault, tmp_path):
        path = vault.meeting_dir("Weekly Standup", date(2026, 4, 6))
        assert path == tmp_path / "Meetings" / "2026" / "04" / "06" / "weekly-standup"

    def test_slugifies_name(self, vault, tmp_path):
        path = vault.meeting_dir("Q2 Planning — Session #1", date(2026, 4, 6))
        assert path == tmp_path / "Meetings" / "2026" / "04" / "06" / "q2-planning-session-1"


class TestEnsureMeetingDir:
    def test_creates_directories(self, vault):
        path = vault.ensure_meeting_dir("Standup", date(2026, 4, 6))
        assert path.exists()
        assert path.is_dir()


class TestRecordingPath:
    def test_returns_flac_path(self, vault):
        path = vault.recording_path("Standup", date(2026, 4, 6))
        assert path.name == "recording.flac"


class TestTranscriptPath:
    def test_includes_model_name(self, vault):
        path = vault.transcript_path("Standup", date(2026, 4, 6), "large-v3")
        assert path.name == "transcript-large-v3.md"


class TestSummaryPath:
    def test_includes_template_name(self, vault):
        path = vault.summary_path("Standup", date(2026, 4, 6), "standup")
        assert path.name == "summary-standup.md"


class TestMemosPath:
    def test_returns_memos_path(self, vault):
        path = vault.memos_path("Standup", date(2026, 4, 6))
        assert path.name == "memos.md"


class TestListMeetings:
    def test_lists_meetings_by_date(self, vault):
        vault.ensure_meeting_dir("Standup", date(2026, 4, 6))
        vault.ensure_meeting_dir("Retro", date(2026, 4, 6))
        vault.ensure_meeting_dir("Planning", date(2026, 3, 15))

        meetings = vault.list_meetings()
        assert len(meetings) == 3
        # Most recent first
        assert meetings[0].date == date(2026, 4, 6)
        assert meetings[-1].date == date(2026, 3, 15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vault.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement vault module**

`src/meetscribe/storage/__init__.py`: empty file.

`src/meetscribe/storage/vault.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class MeetingInfo:
    name: str
    date: date
    path: Path
    has_recording: bool = False
    has_transcript: bool = False
    has_summary: bool = False
    has_memos: bool = False


def slugify(name: str) -> str:
    """Convert a meeting name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug


class MeetingStorage:
    def __init__(self, vault_root: str | Path, meetings_folder: str = "Meetings") -> None:
        self.vault_root = Path(vault_root)
        self.meetings_folder = meetings_folder

    @property
    def meetings_root(self) -> Path:
        return self.vault_root / self.meetings_folder

    def meeting_dir(self, name: str, meeting_date: date) -> Path:
        return (
            self.meetings_root
            / f"{meeting_date.year}"
            / f"{meeting_date.month:02d}"
            / f"{meeting_date.day:02d}"
            / slugify(name)
        )

    def ensure_meeting_dir(self, name: str, meeting_date: date) -> Path:
        path = self.meeting_dir(name, meeting_date)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def recording_path(self, name: str, meeting_date: date) -> Path:
        return self.meeting_dir(name, meeting_date) / "recording.flac"

    def transcript_path(self, name: str, meeting_date: date, model: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"transcript-{model}.md"

    def summary_path(self, name: str, meeting_date: date, template_name: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"summary-{template_name}.md"

    def memos_path(self, name: str, meeting_date: date) -> Path:
        return self.meeting_dir(name, meeting_date) / "memos.md"

    def list_meetings(self) -> list[MeetingInfo]:
        """Scan the meetings folder and return all meetings, newest first."""
        meetings: list[MeetingInfo] = []
        root = self.meetings_root
        if not root.exists():
            return meetings

        for year_dir in sorted(root.iterdir(), reverse=True):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                    for meeting_dir in sorted(day_dir.iterdir(), reverse=True):
                        if not meeting_dir.is_dir():
                            continue
                        meeting_date = date(
                            int(year_dir.name),
                            int(month_dir.name),
                            int(day_dir.name),
                        )
                        files = {f.name for f in meeting_dir.iterdir()}
                        meetings.append(MeetingInfo(
                            name=meeting_dir.name,
                            date=meeting_date,
                            path=meeting_dir,
                            has_recording="recording.flac" in files,
                            has_transcript=any(f.startswith("transcript-") for f in files),
                            has_summary=any(f.startswith("summary-") for f in files),
                            has_memos="memos.md" in files,
                        ))
        return meetings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vault.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/storage/ tests/test_vault.py
git commit -m "feat: vault storage module with meeting directory management"
```

---

### Task 3: Audio Recorder Module

**Files:**

- Create: `src/meetscribe/audio/__init__.py`
- Create: `src/meetscribe/audio/recorder.py`
- Create: `tests/test_recorder.py`

- [ ] **Step 1: Write failing test**

`tests/test_recorder.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from meetscribe.audio.recorder import AudioRecorder, find_device


class TestFindDevice:
    @patch("meetscribe.audio.recorder.sd.query_devices")
    def test_finds_blackhole_device(self, mock_query):
        mock_query.return_value = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 0},
            {"name": "BlackHole 2ch", "max_input_channels": 2, "index": 1},
        ]
        device = find_device("BlackHole 2ch")
        assert device == 1

    @patch("meetscribe.audio.recorder.sd.query_devices")
    def test_raises_if_not_found(self, mock_query):
        mock_query.return_value = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 0},
        ]
        with pytest.raises(ValueError, match="BlackHole 2ch"):
            find_device("BlackHole 2ch")


class TestAudioRecorder:
    def test_init_sets_output_path(self, tmp_path):
        out = tmp_path / "test.flac"
        recorder = AudioRecorder(
            output_path=out,
            device_name="BlackHole 2ch",
            sample_rate=44100,
            channels=2,
        )
        assert recorder.output_path == out

    def test_is_recording_starts_false(self, tmp_path):
        out = tmp_path / "test.flac"
        recorder = AudioRecorder(
            output_path=out,
            device_name="BlackHole 2ch",
            sample_rate=44100,
            channels=2,
        )
        assert recorder.is_recording is False

    def test_peak_level_starts_at_zero(self, tmp_path):
        out = tmp_path / "test.flac"
        recorder = AudioRecorder(
            output_path=out,
            device_name="BlackHole 2ch",
            sample_rate=44100,
            channels=2,
        )
        assert recorder.peak_level == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement audio recorder**

`src/meetscribe/audio/__init__.py`: empty file.

`src/meetscribe/audio/recorder.py`:

```python
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def find_device(name: str) -> int:
    """Find an audio device index by name. Raises ValueError if not found."""
    devices = sd.query_devices()
    for dev in devices:
        if dev["name"] == name and dev["max_input_channels"] > 0:
            return dev["index"]
    raise ValueError(
        f"Audio device '{name}' not found. "
        f"Available input devices: {[d['name'] for d in devices if d['max_input_channels'] > 0]}"
    )


class AudioRecorder:
    """Records system audio to a FLAC file using sounddevice and soundfile."""

    def __init__(
        self,
        output_path: Path,
        device_name: str = "BlackHole 2ch",
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> None:
        self.output_path = output_path
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.peak_level = 0.0
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called from the audio thread for each block of audio data."""
        self.peak_level = float(np.abs(indata).max())
        self._queue.put(indata.copy())

    def _writer_loop(self) -> None:
        """Background thread that drains the queue and writes to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            str(self.output_path),
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            format="FLAC",
        ) as f:
            while True:
                data = self._queue.get()
                if data is None:
                    break
                f.write(data)

    def start(self) -> None:
        """Begin recording."""
        if self.is_recording:
            return
        device_index = find_device(self.device_name)
        self.is_recording = True
        self.peak_level = 0.0

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=device_index,
            channels=self.channels,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop recording and finalize the file."""
        if not self.is_recording:
            return
        self.is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._queue.put(None)  # Signal writer thread to finish
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.peak_level = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recorder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/audio/ tests/test_recorder.py
git commit -m "feat: audio recorder with BlackHole capture to FLAC"
```

---

### Task 4: Transcription Module

**Files:**

- Create: `src/meetscribe/transcription/__init__.py`
- Create: `src/meetscribe/transcription/whisper.py`
- Create: `tests/test_whisper.py`

- [ ] **Step 1: Write failing test**

`tests/test_whisper.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meetscribe.transcription.whisper import (
    transcribe_audio,
    format_transcript,
    format_timestamp,
    AVAILABLE_MODELS,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00"

    def test_minutes_and_seconds(self):
        assert format_timestamp(125.5) == "00:02:05"

    def test_hours(self):
        assert format_timestamp(3661.0) == "01:01:01"


class TestFormatTranscript:
    def test_formats_segments_with_frontmatter(self):
        segments = [
            MagicMock(start=0.0, end=5.0, text=" Hello world."),
            MagicMock(start=5.0, end=10.0, text=" Second segment."),
        ]
        result = format_transcript(
            segments=segments,
            meeting_name="Standup",
            meeting_date="2026-04-06",
            model="base",
            duration="00:10:00",
        )
        assert "meeting: Standup" in result
        assert "model: base" in result
        assert "[00:00:00] Hello world." in result
        assert "[00:00:05] Second segment." in result


class TestAvailableModels:
    def test_contains_expected_models(self):
        assert "tiny" in AVAILABLE_MODELS
        assert "base" in AVAILABLE_MODELS
        assert "small" in AVAILABLE_MODELS
        assert "medium" in AVAILABLE_MODELS
        assert "large-v3" in AVAILABLE_MODELS


class TestTranscribeAudio:
    @patch("meetscribe.transcription.whisper.WhisperModel")
    def test_returns_formatted_transcript(self, mock_model_cls):
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.")
        seg2 = MagicMock(start=5.0, end=10.0, text=" World.")
        mock_info = MagicMock(duration=10.0)
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        result = transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
        )

        assert "[00:00:00] Hello." in result
        assert "[00:00:05] World." in result
        mock_model_cls.assert_called_once_with("base", device="cpu", compute_type="int8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_whisper.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement transcription module**

`src/meetscribe/transcription/__init__.py`: empty file.

`src/meetscribe/transcription/whisper.py`:

```python
from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(
    segments: list,
    meeting_name: str,
    meeting_date: str,
    model: str,
    duration: str,
) -> str:
    """Format transcription segments into a markdown document with frontmatter."""
    lines = [
        "---",
        f"meeting: {meeting_name}",
        f"date: {meeting_date}",
        f"model: {model}",
        f'duration: "{duration}"',
        "---",
        "",
    ]
    for segment in segments:
        timestamp = format_timestamp(segment.start)
        text = segment.text.strip()
        lines.append(f"[{timestamp}] {text}")
        lines.append("")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format total seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    meeting_name: str,
    meeting_date: str,
) -> str:
    """Transcribe an audio file and return formatted markdown transcript."""
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    segment_list = list(segments)

    duration = _format_duration(info.duration)

    return format_transcript(
        segments=segment_list,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        model=model_name,
        duration=duration,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_whisper.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/transcription/ tests/test_whisper.py
git commit -m "feat: whisper transcription module with model selection"
```

---

### Task 5: Summarization Module

**Files:**

- Create: `src/meetscribe/summarization/__init__.py`
- Create: `src/meetscribe/summarization/provider.py`
- Create: `tests/test_summarization.py`

- [ ] **Step 1: Write failing test**

`tests/test_summarization.py`:

```python
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from meetscribe.summarization.provider import SummarizationProvider


class TestSummarizationProvider:
    def test_init_sets_endpoint(self):
        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        assert provider.model == "llama3"
        assert provider.base_url == "http://localhost:11434/v1"

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_list_models(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        model1 = MagicMock()
        model1.id = "llama3"
        model2 = MagicMock()
        model2.id = "mistral"
        mock_client.models.list.return_value = MagicMock(data=[model1, model2])

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        models = provider.list_models()
        assert models == ["llama3", "mistral"]

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_summarize(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary here."))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        result = provider.summarize("You are a summarizer.", "Summarize this meeting.")
        assert result == "Summary here."
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": "Summarize this meeting."},
            ],
        )

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_list_models_returns_empty_on_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = Exception("Connection refused")

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        models = provider.list_models()
        assert models == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summarization.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement summarization provider**

`src/meetscribe/summarization/__init__.py`: empty file.

`src/meetscribe/summarization/provider.py`:

```python
from __future__ import annotations

from openai import OpenAI


class SummarizationProvider:
    """Wraps an OpenAI-compatible API (Ollama or LM Studio) for summarization."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key="not-needed")

    def list_models(self) -> list[str]:
        """Query the provider for available models."""
        try:
            response = self._client.models.list()
            return [m.id for m in response.data]
        except Exception:
            return []

    def summarize(self, system_prompt: str, user_prompt: str) -> str:
        """Send a summarization request and return the response text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_summarization.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/summarization/ tests/test_summarization.py
git commit -m "feat: summarization provider with OpenAI-compatible API client"
```

---

### Task 6: Template Engine

**Files:**

- Create: `src/meetscribe/templates/__init__.py`
- Create: `src/meetscribe/templates/engine.py`
- Create: `templates/default.md`
- Create: `templates/standup.md`
- Create: `templates/retrospective.md`
- Create: `tests/test_templates.py`

- [ ] **Step 1: Write failing test**

`tests/test_templates.py`:

```python
from pathlib import Path

import pytest

from meetscribe.templates.engine import TemplateEngine


@pytest.fixture
def templates_dir(tmp_path):
    default = tmp_path / "default.md"
    default.write_text(
        "Summarize:\n{{ transcript }}\n{% if memos %}Notes: {{ memos }}{% endif %}"
    )
    standup = tmp_path / "standup.md"
    standup.write_text("Standup for {{ meeting_name }} on {{ date }}:\n{{ transcript }}")
    return tmp_path


@pytest.fixture
def engine(templates_dir):
    return TemplateEngine(templates_dir)


class TestListTemplates:
    def test_returns_template_names(self, engine):
        names = engine.list_templates()
        assert "default" in names
        assert "standup" in names

    def test_empty_dir_returns_empty(self, tmp_path):
        engine = TemplateEngine(tmp_path)
        assert engine.list_templates() == []


class TestRender:
    def test_renders_with_transcript(self, engine):
        result = engine.render(
            template_name="default",
            transcript="Hello world.",
            memos="",
            meeting_name="Standup",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Summarize:" in result
        assert "Hello world." in result
        assert "Notes:" not in result  # memos empty, should be skipped

    def test_renders_with_memos(self, engine):
        result = engine.render(
            template_name="default",
            transcript="Hello.",
            memos="Remember to follow up.",
            meeting_name="Standup",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Notes: Remember to follow up." in result

    def test_renders_meeting_name_and_date(self, engine):
        result = engine.render(
            template_name="standup",
            transcript="Content.",
            memos="",
            meeting_name="Weekly",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Standup for Weekly on 2026-04-06:" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_templates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement template engine**

`src/meetscribe/templates/__init__.py`: empty file.

`src/meetscribe/templates/engine.py`:

```python
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TemplateEngine:
    """Loads and renders Jinja2 meeting summary templates."""

    def __init__(self, templates_dir: Path) -> None:
        self.templates_dir = templates_dir
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            keep_trailing_newline=True,
        )

    def list_templates(self) -> list[str]:
        """Return sorted list of available template names (without .md extension)."""
        if not self.templates_dir.exists():
            return []
        return sorted(
            p.stem for p in self.templates_dir.glob("*.md")
        )

    def render(
        self,
        template_name: str,
        transcript: str,
        memos: str,
        meeting_name: str,
        date: str,
        duration: str,
    ) -> str:
        """Render a template with the given variables."""
        template = self._env.get_template(f"{template_name}.md")
        return template.render(
            transcript=transcript,
            memos=memos,
            meeting_name=meeting_name,
            date=date,
            duration=duration,
        )
```

- [ ] **Step 4: Create default templates**

`templates/default.md`:

```
You are a meeting summarizer. Create a clear, concise summary of the following meeting.

## Meeting: {{ meeting_name }}
**Date:** {{ date }}
**Duration:** {{ duration }}

## Transcript
{{ transcript }}

{% if memos %}
## Additional Notes from Attendee
{{ memos }}
{% endif %}

Please provide:
1. A brief overview (2-3 sentences)
2. Key discussion points
3. Action items (if any)
4. Decisions made (if any)
```

`templates/standup.md`:

```
Summarize the following meeting transcript as a standup summary.

## Meeting: {{ meeting_name }}
**Date:** {{ date }}

## Format
- **Yesterday:** What was discussed about past work
- **Today:** What was planned
- **Blockers:** Any blockers mentioned

## Transcript
{{ transcript }}

{% if memos %}
## Additional Notes
{{ memos }}
{% endif %}
```

`templates/retrospective.md`:

```
Summarize the following meeting transcript as a retrospective summary.

## Meeting: {{ meeting_name }}
**Date:** {{ date }}

## Format
- **What went well:** Positive outcomes and successes
- **What didn't go well:** Challenges and issues
- **Action items:** Improvements to implement

## Transcript
{{ transcript }}

{% if memos %}
## Additional Notes
{{ memos }}
{% endif %}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_templates.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/meetscribe/templates/ templates/ tests/test_templates.py
git commit -m "feat: Jinja2 template engine with default meeting templates"
```

---

### Task 7: TUI — Main App Shell & Home Screen

**Files:**

- Create: `src/meetscribe/tui/__init__.py`
- Create: `src/meetscribe/tui/app.py`
- Create: `src/meetscribe/tui/screens/__init__.py`
- Create: `src/meetscribe/tui/screens/home.py`
- Create: `src/meetscribe/tui/widgets/__init__.py`

- [ ] **Step 1: Create TUI package structure**

`src/meetscribe/tui/__init__.py`: empty file.
`src/meetscribe/tui/screens/__init__.py`: empty file.
`src/meetscribe/tui/widgets/__init__.py`: empty file.

- [ ] **Step 2: Implement main app**

`src/meetscribe/tui/app.py`:

```python
from textual.app import App

from meetscribe.config import load_config, MeetscribeConfig


class MeetscribeApp(App):
    """Meetscribe TUI application."""

    TITLE = "Meetscribe"
    CSS = """
    Screen {
        align: center middle;
    }

    #meeting-list {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }

    .title {
        text-style: bold;
        color: $text;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "push_screen('settings')", "Settings"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()

    def on_mount(self) -> None:
        from meetscribe.tui.screens.home import HomeScreen
        self.push_screen(HomeScreen())
```

- [ ] **Step 3: Implement home screen**

`src/meetscribe/tui/screens/home.py`:

```python
from __future__ import annotations

from datetime import date

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Static, ListView, ListItem, Label, Input

from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage, MeetingInfo


class MeetingListItem(ListItem):
    """A single meeting entry in the list."""

    def __init__(self, meeting: MeetingInfo) -> None:
        super().__init__()
        self.meeting = meeting

    def compose(self) -> ComposeResult:
        icons = ""
        if self.meeting.has_recording:
            icons += "[R]"
        if self.meeting.has_transcript:
            icons += "[T]"
        if self.meeting.has_summary:
            icons += "[S]"
        if self.meeting.has_memos:
            icons += "[M]"
        yield Label(
            f"{self.meeting.date}  {self.meeting.name}  {icons}"
        )


class HomeScreen(Screen):
    """Landing screen — start recording or browse past meetings."""

    BINDINGS = [
        ("n", "new_recording", "New Recording"),
        ("s", "app.push_screen('settings')", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Meetscribe", classes="title"),
            Button("New Recording", id="new-recording", variant="primary"),
            Static("Past Meetings"),
            ListView(id="meeting-list"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_meetings()

    def _refresh_meetings(self) -> None:
        config = self.app.config
        if not config.vault.root:
            return
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        meetings = storage.list_meetings()
        list_view = self.query_one("#meeting-list", ListView)
        list_view.clear()
        for meeting in meetings:
            list_view.append(MeetingListItem(meeting))

    @on(Button.Pressed, "#new-recording")
    def action_new_recording(self) -> None:
        from meetscribe.tui.screens.recording import RecordingScreen
        self.app.push_screen(RecordingScreen())

    @on(ListView.Selected, "#meeting-list")
    def on_meeting_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, MeetingListItem):
            from meetscribe.tui.screens.meeting import MeetingScreen
            self.app.push_screen(MeetingScreen(event.item.meeting))
```

- [ ] **Step 4: Verify the app launches**

Run: `cd /Users/ian.bartholomew/Dev/transription && python -c "from meetscribe.tui.app import MeetscribeApp; print('App imports OK')"`
Expected: `App imports OK`

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/tui/
git commit -m "feat: TUI app shell and home screen with meeting list"
```

---

### Task 8: TUI — Recording Screen

**Files:**

- Create: `src/meetscribe/tui/screens/recording.py`

- [ ] **Step 1: Implement recording screen**

`src/meetscribe/tui/screens/recording.py`:

```python
from __future__ import annotations

import time
from datetime import date

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Static, Input, Label

from meetscribe.audio.recorder import AudioRecorder
from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage


class RecordingScreen(Screen):
    """Screen for actively recording a meeting."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._recorder: AudioRecorder | None = None
        self._start_time: float = 0.0
        self._meeting_name: str = ""
        self._recording_active = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("New Recording", classes="title"),
            Input(placeholder="Meeting name...", id="meeting-name"),
            Button("Start Recording", id="start-btn", variant="primary"),
            Label("", id="timer"),
            Label("", id="level"),
            Button("Stop Recording", id="stop-btn", variant="error", disabled=True),
        )
        yield Footer()

    @on(Button.Pressed, "#start-btn")
    def start_recording(self) -> None:
        name_input = self.query_one("#meeting-name", Input)
        self._meeting_name = name_input.value.strip()
        if not self._meeting_name:
            self.notify("Please enter a meeting name.", severity="error")
            return

        config = self.app.config
        if not config.vault.root:
            self.notify("Please configure vault root in settings first.", severity="error")
            return

        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        output_path = storage.recording_path(self._meeting_name, date.today())
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._recorder = AudioRecorder(
            output_path=output_path,
            device_name=config.audio.device_name,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )

        try:
            self._recorder.start()
        except ValueError as e:
            self.notify(str(e), severity="error")
            return

        self._start_time = time.monotonic()
        self._recording_active = True

        name_input.disabled = True
        self.query_one("#start-btn", Button).disabled = True
        self.query_one("#stop-btn", Button).disabled = False

        self._update_timer()

    @work(exclusive=True)
    async def _update_timer(self) -> None:
        """Periodically update the timer and level display."""
        import asyncio
        while self._recording_active:
            elapsed = time.monotonic() - self._start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.query_one("#timer", Label).update(f"Recording: {h:02d}:{m:02d}:{s:02d}")

            if self._recorder:
                level = self._recorder.peak_level
                bar_len = int(level * 40)
                bar = "|" * bar_len + "." * (40 - bar_len)
                self.query_one("#level", Label).update(f"Level: [{bar}]")

            await asyncio.sleep(0.25)

    @on(Button.Pressed, "#stop-btn")
    def stop_recording(self) -> None:
        self._recording_active = False
        if self._recorder:
            self._recorder.stop()

        from meetscribe.storage.vault import MeetingStorage, MeetingInfo
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        meeting_dir = storage.meeting_dir(self._meeting_name, date.today())

        meeting = MeetingInfo(
            name=self._meeting_name,
            date=date.today(),
            path=meeting_dir,
            has_recording=True,
        )

        from meetscribe.tui.screens.meeting import MeetingScreen
        self.app.switch_screen(MeetingScreen(meeting))

    def action_cancel(self) -> None:
        self._recording_active = False
        if self._recorder and self._recorder.is_recording:
            self._recorder.stop()
        self.app.pop_screen()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from meetscribe.tui.screens.recording import RecordingScreen; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/meetscribe/tui/screens/recording.py
git commit -m "feat: recording screen with timer and level display"
```

---

### Task 9: TUI — Meeting Screen (Transcript, Summary, Memos Tabs)

**Files:**

- Create: `src/meetscribe/tui/screens/meeting.py`

- [ ] **Step 1: Implement meeting screen**

`src/meetscribe/tui/screens/meeting.py`:

```python
from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Markdown,
    Select,
    Static,
    TabPane,
    TabbedContent,
    TextArea,
)

from meetscribe.storage.vault import MeetingInfo, MeetingStorage
from meetscribe.transcription.whisper import AVAILABLE_MODELS


class MeetingScreen(Screen):
    """View and manage meeting artifacts: recording, transcript, summary, memos."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def __init__(self, meeting: MeetingInfo) -> None:
        super().__init__()
        self.meeting = meeting

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Meeting: {self.meeting.name} ({self.meeting.date})", classes="title")

        with TabbedContent("Recording", "Transcript", "Summary", "Memos"):
            with TabPane("Recording", id="recording-tab"):
                yield self._compose_recording_tab()
            with TabPane("Transcript", id="transcript-tab"):
                yield self._compose_transcript_tab()
            with TabPane("Summary", id="summary-tab"):
                yield self._compose_summary_tab()
            with TabPane("Memos", id="memos-tab"):
                yield self._compose_memos_tab()

        yield Footer()

    def _compose_recording_tab(self) -> Vertical:
        recording_path = self.meeting.path / "recording.flac"
        if recording_path.exists():
            size_mb = recording_path.stat().st_size / (1024 * 1024)
            info_text = f"Recording: {recording_path.name}\nSize: {size_mb:.1f} MB"
        else:
            info_text = "No recording found."
        return Vertical(Label(info_text, id="recording-info"))

    def _compose_transcript_tab(self) -> Vertical:
        model_options = [(m, m) for m in AVAILABLE_MODELS]
        config = self.app.config if hasattr(self, "app") and self.app else None
        default_model = config.transcription.default_model if config else "base"

        return Vertical(
            Horizontal(
                Select(model_options, value=default_model, id="whisper-model"),
                Button("Transcribe", id="transcribe-btn", variant="primary"),
                Button("Regenerate", id="regenerate-transcript-btn"),
            ),
            Markdown("*No transcript yet. Select a model and click Transcribe.*", id="transcript-view"),
        )

    def _compose_summary_tab(self) -> Vertical:
        return Vertical(
            Horizontal(
                Select([], id="template-select", prompt="Select template"),
                Select([], id="provider-select", prompt="Select provider"),
                Select([], id="llm-model-select", prompt="Select model"),
            ),
            Horizontal(
                Button("Summarize", id="summarize-btn", variant="primary"),
                Button("Regenerate", id="regenerate-summary-btn"),
                Button("Refresh Models", id="refresh-models-btn"),
            ),
            Markdown("*No summary yet. Select a template and model, then click Summarize.*", id="summary-view"),
        )

    def _compose_memos_tab(self) -> Vertical:
        return Vertical(
            TextArea(id="memos-editor"),
            Button("Save Memos", id="save-memos-btn", variant="primary"),
        )

    def on_mount(self) -> None:
        self._load_existing_transcript()
        self._load_existing_summary()
        self._load_memos()
        self._populate_templates()
        self._populate_providers()

    def _load_existing_transcript(self) -> None:
        """Load the most recent transcript if one exists."""
        for f in sorted(self.meeting.path.glob("transcript-*.md"), reverse=True):
            content = f.read_text()
            self.query_one("#transcript-view", Markdown).update(content)
            break

    def _load_existing_summary(self) -> None:
        """Load the most recent summary if one exists."""
        for f in sorted(self.meeting.path.glob("summary-*.md"), reverse=True):
            content = f.read_text()
            self.query_one("#summary-view", Markdown).update(content)
            break

    def _load_memos(self) -> None:
        memos_path = self.meeting.path / "memos.md"
        if memos_path.exists():
            self.query_one("#memos-editor", TextArea).load_text(memos_path.read_text())

    def _populate_templates(self) -> None:
        from meetscribe.templates.engine import TemplateEngine
        # Look for templates in the package templates dir
        templates_dir = Path(__file__).parent.parent.parent.parent / "templates"
        if not templates_dir.exists():
            templates_dir = Path(__file__).parent.parent.parent / "templates"
        engine = TemplateEngine(templates_dir)
        names = engine.list_templates()
        if names:
            options = [(n, n) for n in names]
            self.query_one("#template-select", Select).set_options(options)

    def _populate_providers(self) -> None:
        config = self.app.config
        provider_options = [(k, k) for k in config.summarization.endpoints]
        self.query_one("#provider-select", Select).set_options(provider_options)

    @on(Select.Changed, "#provider-select")
    def on_provider_changed(self, event: Select.Changed) -> None:
        """When provider changes, fetch available models."""
        if event.value and event.value != Select.BLANK:
            self._fetch_models(str(event.value))

    @work(thread=True)
    def _fetch_models(self, provider: str) -> None:
        config = self.app.config
        endpoint = config.summarization.endpoints.get(provider, "")
        if not endpoint:
            return
        from meetscribe.summarization.provider import SummarizationProvider
        p = SummarizationProvider(base_url=endpoint, model="")
        models = p.list_models()
        if models:
            options = [(m, m) for m in models]
            self.app.call_from_thread(
                self.query_one("#llm-model-select", Select).set_options, options
            )

    @on(Button.Pressed, "#refresh-models-btn")
    def refresh_models(self) -> None:
        provider_select = self.query_one("#provider-select", Select)
        if provider_select.value and provider_select.value != Select.BLANK:
            self._fetch_models(str(provider_select.value))

    @on(Button.Pressed, "#transcribe-btn")
    @on(Button.Pressed, "#regenerate-transcript-btn")
    def do_transcribe(self) -> None:
        model_select = self.query_one("#whisper-model", Select)
        model_name = str(model_select.value) if model_select.value != Select.BLANK else "base"
        self._run_transcription(model_name)

    @work(thread=True)
    def _run_transcription(self, model_name: str) -> None:
        self.app.call_from_thread(self.notify, f"Transcribing with {model_name}...")
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        recording_path = self.meeting.path / "recording.flac"

        if not recording_path.exists():
            self.app.call_from_thread(self.notify, "No recording found.", severity="error")
            return

        from meetscribe.transcription.whisper import transcribe_audio
        transcript = transcribe_audio(
            audio_path=recording_path,
            model_name=model_name,
            meeting_name=self.meeting.name,
            meeting_date=str(self.meeting.date),
        )

        transcript_path = storage.transcript_path(self.meeting.name, self.meeting.date, model_name)
        transcript_path.write_text(transcript)

        self.app.call_from_thread(self.query_one("#transcript-view", Markdown).update, transcript)
        self.app.call_from_thread(self.notify, "Transcription complete!")

    @on(Button.Pressed, "#summarize-btn")
    @on(Button.Pressed, "#regenerate-summary-btn")
    def do_summarize(self) -> None:
        template_select = self.query_one("#template-select", Select)
        provider_select = self.query_one("#provider-select", Select)
        model_select = self.query_one("#llm-model-select", Select)

        if template_select.value == Select.BLANK:
            self.notify("Please select a template.", severity="error")
            return
        if provider_select.value == Select.BLANK:
            self.notify("Please select a provider.", severity="error")
            return
        if model_select.value == Select.BLANK:
            self.notify("Please select a model.", severity="error")
            return

        self._run_summarization(
            template_name=str(template_select.value),
            provider=str(provider_select.value),
            model=str(model_select.value),
        )

    @work(thread=True)
    def _run_summarization(self, template_name: str, provider: str, model: str) -> None:
        self.app.call_from_thread(self.notify, f"Summarizing with {provider}/{model}...")
        config = self.app.config

        # Find the latest transcript
        transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
        if not transcripts:
            self.app.call_from_thread(self.notify, "No transcript found. Transcribe first.", severity="error")
            return
        transcript_text = transcripts[0].read_text()

        # Load memos
        memos_path = self.meeting.path / "memos.md"
        memos_text = memos_path.read_text() if memos_path.exists() else ""

        # Render template
        from meetscribe.templates.engine import TemplateEngine
        templates_dir = Path(__file__).parent.parent.parent.parent / "templates"
        if not templates_dir.exists():
            templates_dir = Path(__file__).parent.parent.parent / "templates"
        engine = TemplateEngine(templates_dir)

        rendered = engine.render(
            template_name=template_name,
            transcript=transcript_text,
            memos=memos_text,
            meeting_name=self.meeting.name,
            date=str(self.meeting.date),
            duration="",
        )

        # Send to LLM
        from meetscribe.summarization.provider import SummarizationProvider
        endpoint = config.summarization.endpoints.get(provider, "")
        llm = SummarizationProvider(base_url=endpoint, model=model)
        summary = llm.summarize(
            system_prompt="You are a meeting summarizer. Produce a clear, well-structured summary.",
            user_prompt=rendered,
        )

        # Save
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        summary_path = storage.summary_path(self.meeting.name, self.meeting.date, template_name)

        # Add frontmatter
        full_summary = (
            f"---\n"
            f"meeting: {self.meeting.name}\n"
            f"date: {self.meeting.date}\n"
            f"template: {template_name}\n"
            f"provider: {provider}\n"
            f"model: {model}\n"
            f"---\n\n"
            f"{summary}"
        )
        summary_path.write_text(full_summary)

        self.app.call_from_thread(self.query_one("#summary-view", Markdown).update, full_summary)
        self.app.call_from_thread(self.notify, "Summary complete!")

    @on(Button.Pressed, "#save-memos-btn")
    def save_memos(self) -> None:
        memos_text = self.query_one("#memos-editor", TextArea).text
        memos_path = self.meeting.path / "memos.md"
        memos_path.parent.mkdir(parents=True, exist_ok=True)
        memos_path.write_text(memos_text)
        self.notify("Memos saved.")

    def action_go_back(self) -> None:
        self.app.pop_screen()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from meetscribe.tui.screens.meeting import MeetingScreen; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/meetscribe/tui/screens/meeting.py
git commit -m "feat: meeting screen with transcript, summary, and memos tabs"
```

---

### Task 10: TUI — Settings Screen

**Files:**

- Create: `src/meetscribe/tui/screens/settings.py`

- [ ] **Step 1: Implement settings screen**

`src/meetscribe/tui/screens/settings.py`:

```python
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from meetscribe.config import save_config, MeetscribeConfig
from meetscribe.transcription.whisper import AVAILABLE_MODELS


class SettingsScreen(Screen):
    """Configure app defaults."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        config: MeetscribeConfig = self.app.config
        yield Header()
        yield Vertical(
            Static("Settings", classes="title"),

            Label("Vault Root Path:"),
            Input(value=config.vault.root, id="vault-root"),

            Label("Meetings Folder (relative to vault):"),
            Input(value=config.vault.meetings_folder, id="meetings-folder"),

            Label("Audio Device Name:"),
            Input(value=config.audio.device_name, id="device-name"),

            Label("Sample Rate:"),
            Input(value=str(config.audio.sample_rate), id="sample-rate"),

            Label("Channels:"),
            Input(value=str(config.audio.channels), id="channels"),

            Label("Default Whisper Model:"),
            Select(
                [(m, m) for m in AVAILABLE_MODELS],
                value=config.transcription.default_model,
                id="default-whisper-model",
            ),

            Label("Default LLM Provider:"),
            Select(
                [(k, k) for k in config.summarization.endpoints],
                value=config.summarization.default_provider,
                id="default-provider",
            ),

            Label("Default LLM Model:"),
            Input(value=config.summarization.default_model, id="default-llm-model"),

            Label("Ollama Endpoint:"),
            Input(value=config.summarization.endpoints.get("ollama", ""), id="ollama-endpoint"),

            Label("LM Studio Endpoint:"),
            Input(value=config.summarization.endpoints.get("lmstudio", ""), id="lmstudio-endpoint"),

            Button("Save Settings", id="save-btn", variant="primary"),
        )
        yield Footer()

    @on(Button.Pressed, "#save-btn")
    def save_settings(self) -> None:
        config: MeetscribeConfig = self.app.config

        config.vault.root = self.query_one("#vault-root", Input).value
        config.vault.meetings_folder = self.query_one("#meetings-folder", Input).value
        config.audio.device_name = self.query_one("#device-name", Input).value
        config.audio.sample_rate = int(self.query_one("#sample-rate", Input).value)
        config.audio.channels = int(self.query_one("#channels", Input).value)

        model_select = self.query_one("#default-whisper-model", Select)
        if model_select.value != Select.BLANK:
            config.transcription.default_model = str(model_select.value)

        provider_select = self.query_one("#default-provider", Select)
        if provider_select.value != Select.BLANK:
            config.summarization.default_provider = str(provider_select.value)

        config.summarization.default_model = self.query_one("#default-llm-model", Input).value
        config.summarization.endpoints["ollama"] = self.query_one("#ollama-endpoint", Input).value
        config.summarization.endpoints["lmstudio"] = self.query_one("#lmstudio-endpoint", Input).value

        save_config(config)
        self.notify("Settings saved.")

    def action_go_back(self) -> None:
        self.app.pop_screen()
```

- [ ] **Step 2: Register settings screen in app**

Update `src/meetscribe/tui/app.py` — add the settings screen to the `SCREENS` mapping:

Replace the `BINDINGS` and add `SCREENS`:

```python
from textual.app import App

from meetscribe.config import load_config, MeetscribeConfig


class MeetscribeApp(App):
    """Meetscribe TUI application."""

    TITLE = "Meetscribe"
    CSS = """
    Screen {
        align: center middle;
    }

    #meeting-list {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }

    .title {
        text-style: bold;
        color: $text;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "open_settings", "Settings"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()

    def on_mount(self) -> None:
        from meetscribe.tui.screens.home import HomeScreen
        self.push_screen(HomeScreen())

    def action_open_settings(self) -> None:
        from meetscribe.tui.screens.settings import SettingsScreen
        self.push_screen(SettingsScreen())
```

- [ ] **Step 3: Verify import**

Run: `python -c "from meetscribe.tui.screens.settings import SettingsScreen; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/meetscribe/tui/screens/settings.py src/meetscribe/tui/app.py
git commit -m "feat: settings screen with all configuration options"
```

---

### Task 11: First-Run Setup Flow

**Files:**

- Create: `src/meetscribe/tui/screens/setup.py`
- Modify: `src/meetscribe/tui/app.py`

- [ ] **Step 1: Implement first-run setup screen**

`src/meetscribe/tui/screens/setup.py`:

```python
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Static

from meetscribe.config import save_config


class SetupScreen(Screen):
    """First-run setup — collect minimum required configuration."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Welcome to Meetscribe!", classes="title"),
            Static("Let's configure the basics to get started."),
            Label("Obsidian Vault Root Path:"),
            Input(placeholder="/path/to/your/vault", id="vault-root"),
            Label("Meetings Folder (relative to vault root):"),
            Input(value="Meetings", id="meetings-folder"),
            Button("Save & Start", id="save-btn", variant="primary"),
        )
        yield Footer()

    @on(Button.Pressed, "#save-btn")
    def save_and_start(self) -> None:
        vault_root = self.query_one("#vault-root", Input).value.strip()
        if not vault_root:
            self.notify("Please enter a vault root path.", severity="error")
            return

        config = self.app.config
        config.vault.root = vault_root
        config.vault.meetings_folder = self.query_one("#meetings-folder", Input).value.strip()
        save_config(config)

        self.notify("Configuration saved!")
        from meetscribe.tui.screens.home import HomeScreen
        self.app.switch_screen(HomeScreen())
```

- [ ] **Step 2: Update app to check for first-run**

Replace `on_mount` in `src/meetscribe/tui/app.py`:

```python
    def on_mount(self) -> None:
        if not self.config.vault.root:
            from meetscribe.tui.screens.setup import SetupScreen
            self.push_screen(SetupScreen())
        else:
            from meetscribe.tui.screens.home import HomeScreen
            self.push_screen(HomeScreen())
```

- [ ] **Step 3: Verify import**

Run: `python -c "from meetscribe.tui.screens.setup import SetupScreen; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/meetscribe/tui/screens/setup.py src/meetscribe/tui/app.py
git commit -m "feat: first-run setup screen for initial configuration"
```

---

### Task 12: Integration Test — End-to-End Smoke Test

**Files:**

- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

`tests/test_integration.py`:

```python
"""Smoke tests that verify the modules wire together correctly."""
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meetscribe.config import default_config, save_config, load_config
from meetscribe.storage.vault import MeetingStorage
from meetscribe.templates.engine import TemplateEngine
from meetscribe.transcription.whisper import format_transcript, format_timestamp
from meetscribe.summarization.provider import SummarizationProvider


class TestEndToEndWorkflow:
    """Verify that the full workflow from recording to summary works."""

    def test_vault_stores_and_retrieves_transcript(self, tmp_path):
        storage = MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")
        meeting_date = date(2026, 4, 6)
        meeting_name = "Integration Test"

        # Create meeting dir
        storage.ensure_meeting_dir(meeting_name, meeting_date)

        # Write a transcript
        transcript_path = storage.transcript_path(meeting_name, meeting_date, "base")
        segments = [
            MagicMock(start=0.0, end=5.0, text=" Hello from integration test."),
        ]
        transcript = format_transcript(
            segments=segments,
            meeting_name=meeting_name,
            meeting_date=str(meeting_date),
            model="base",
            duration="00:00:05",
        )
        transcript_path.write_text(transcript)

        assert transcript_path.exists()
        content = transcript_path.read_text()
        assert "Hello from integration test." in content
        assert "model: base" in content

    def test_template_renders_with_transcript_and_memos(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "test.md").write_text(
            "Meeting: {{ meeting_name }}\n{{ transcript }}\n{% if memos %}Memos: {{ memos }}{% endif %}"
        )

        engine = TemplateEngine(tpl_dir)
        result = engine.render(
            template_name="test",
            transcript="[00:00:00] Discussion about X.",
            memos="Follow up on X.",
            meeting_name="Integration",
            date="2026-04-06",
            duration="00:05:00",
        )

        assert "Meeting: Integration" in result
        assert "[00:00:00] Discussion about X." in result
        assert "Memos: Follow up on X." in result

    def test_config_roundtrip(self, tmp_path):
        config_file = tmp_path / "config.toml"
        cfg = default_config()
        cfg.vault.root = "/test/vault"
        cfg.vault.meetings_folder = "MyMeetings"
        cfg.transcription.default_model = "large-v3"

        save_config(cfg, config_file)
        loaded = load_config(config_file)

        assert loaded.vault.root == "/test/vault"
        assert loaded.vault.meetings_folder == "MyMeetings"
        assert loaded.transcription.default_model == "large-v3"

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_summarization_sends_rendered_template(self, mock_openai_cls, tmp_path):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary: discussed X."))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = SummarizationProvider(base_url="http://localhost:11434/v1", model="llama3")
        result = provider.summarize("You summarize.", "Here is a transcript about X.")

        assert result == "Summary: discussed X."

    def test_meeting_listing(self, tmp_path):
        storage = MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")

        storage.ensure_meeting_dir("Standup", date(2026, 4, 6))
        # Write a fake recording file to test has_recording detection
        rec_path = storage.recording_path("Standup", date(2026, 4, 6))
        rec_path.write_bytes(b"fake audio data")

        meetings = storage.list_meetings()
        assert len(meetings) == 1
        assert meetings[0].name == "standup"
        assert meetings[0].has_recording is True
        assert meetings[0].has_transcript is False
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke tests for end-to-end workflow"
```

---

### Task 13: Final Wiring & README

**Files:**

- Create: `README.md`
- Verify: `src/meetscribe/__main__.py` wiring

- [ ] **Step 1: Verify full app launches**

Run: `python -c "from meetscribe.tui.app import MeetscribeApp; print('All modules wire up correctly')"`
Expected: `All modules wire up correctly`

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Create README**

`README.md`:

```markdown
# Meetscribe

A TUI app for recording, transcribing, and summarizing meetings.

## Features

- Record system audio via BlackHole
- Transcribe recordings locally with faster-whisper (multiple model sizes)
- Summarize meetings using local LLMs (Ollama or LM Studio)
- Customizable summary templates (Jinja2)
- Free-form memos to supplement transcripts
- All artifacts stored in your Obsidian vault

## Prerequisites

- Python 3.11+
- [BlackHole](https://existential.audio/blackhole/) audio driver installed
- [Ollama](https://ollama.ai/) and/or [LM Studio](https://lmstudio.ai/) for summarization

## Install

```bash
pip install -e .
```

## Usage

```bash
meetscribe
```

On first launch, you'll be prompted to configure your Obsidian vault path.

## Configuration

Config is stored at `~/.config/meetscribe/config.toml`. You can also edit settings from within the TUI by pressing `s`.

## Vault Structure

```
<vault>/<meetings_folder>/<year>/<month>/<day>/<meeting-name>/
├── recording.flac
├── transcript-<model>.md
├── summary-<template>.md
└── memos.md
```

## Templates

Meeting summary templates are Jinja2 `.md` files in the `templates/` directory. Available variables:

- `{{ transcript }}` — full transcript
- `{{ memos }}` — user notes
- `{{ meeting_name }}` — meeting name
- `{{ date }}` — meeting date
- `{{ duration }}` — recording duration

```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

- [ ] **Step 5: Run final full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS
