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

    def test_word_with_no_overlap_uses_nearest(self):
        """Words in gaps get assigned to the nearest speaker segment."""
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(10.0, 10.5, " orphan"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 1.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 1  # Both assigned to Speaker 1, grouped
        assert result[0] == ("Speaker 1", 0.0, "hello orphan")

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
    @patch("meetscribe.transcription.diarize._load_audio_for_pyannote")
    @patch("meetscribe.transcription.diarize._get_embedding_model")
    @patch("meetscribe.transcription.diarize._get_pipeline")
    @patch("meetscribe.transcription.diarize._get_hf_token")
    def test_returns_diarization_result(self, mock_token, mock_pipeline_fn, mock_emb_fn, mock_load_audio):
        from pathlib import Path
        import torch

        mock_token.return_value = "hf_fake"
        mock_load_audio.return_value = {"waveform": torch.randn(1, 160000), "sample_rate": 16000}

        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

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

    @patch("meetscribe.transcription.diarize._load_audio_for_pyannote")
    @patch("meetscribe.transcription.diarize._get_embedding_model")
    @patch("meetscribe.transcription.diarize._get_pipeline")
    @patch("meetscribe.transcription.diarize._get_hf_token")
    def test_passes_num_speakers(self, mock_token, mock_pipeline_fn, mock_emb_fn, mock_load_audio):
        from pathlib import Path
        import torch

        mock_token.return_value = "hf_fake"
        mock_load_audio.return_value = {"waveform": torch.randn(1, 160000), "sample_rate": 16000}
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
