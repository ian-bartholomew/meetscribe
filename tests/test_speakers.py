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
        assert reg.list_speakers() == []

    def test_keeps_profiles_with_correct_dimensions(self, registry_path):
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
