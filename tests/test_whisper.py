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
