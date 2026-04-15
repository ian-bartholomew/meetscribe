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
