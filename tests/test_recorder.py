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
