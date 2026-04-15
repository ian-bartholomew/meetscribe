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
