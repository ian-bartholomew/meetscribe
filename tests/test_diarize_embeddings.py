from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from meetscribe.transcription.diarize import (
    _cluster_speakers,
    DiarizationResult,
    SpeakerSegment,
    assign_speakers_to_words,
)


class TestClusterSpeakersEmbeddings:
    def test_returns_diarization_result(self):
        """_cluster_speakers returns a DiarizationResult with segments and embeddings."""
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


def _mock_word(start, end, word):
    """Create a mock faster-whisper Word object."""
    from unittest.mock import MagicMock
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
