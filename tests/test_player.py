# tests/test_player.py
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from meetscribe.audio.player import AudioPlayer


class TestAudioPlayerInit:
    def test_initial_state(self):
        player = AudioPlayer(Path("/fake/audio.flac"))
        assert player.is_playing is False
        assert player.current_position == 0.0
        assert player.file_path == Path("/fake/audio.flac")


class TestAudioPlayerPlay:
    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_seeks_to_offset(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 16000
        mock_sf.channels = 1
        mock_sf_cls.return_value = mock_sf

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play(offset_seconds=30.0)

        # Should seek to frame 30 * 16000 = 480000
        mock_sf.seek.assert_called_once_with(480000)
        mock_stream.start.assert_called_once()
        assert player.is_playing is True

    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_from_zero(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 48000
        mock_sf.channels = 2
        mock_sf_cls.return_value = mock_sf
        mock_stream_cls.return_value = MagicMock()

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play()

        mock_sf.seek.assert_called_once_with(0)

    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_handles_bad_file(self, mock_sf_cls):
        mock_sf_cls.side_effect = RuntimeError("bad file")

        player = AudioPlayer(Path("/fake/bad.flac"))
        player.play()

        assert player.is_playing is False


class TestAudioPlayerStop:
    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_stop_when_playing(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 16000
        mock_sf.channels = 1
        mock_sf_cls.return_value = mock_sf

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play()
        player.stop()

        assert player.is_playing is False
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()
        mock_sf.close.assert_called()

    def test_stop_when_not_playing(self):
        player = AudioPlayer(Path("/fake/audio.flac"))
        player.stop()  # Should not raise
        assert player.is_playing is False
