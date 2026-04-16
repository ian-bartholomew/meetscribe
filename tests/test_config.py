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
        assert cfg.audio.sample_rate == 48000
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
        content = config_file.read_text()
        content = "\n".join(
            line for line in content.splitlines()
            if "huggingface_token" not in line
        )
        config_file.write_text(content)
        loaded = load_config(config_file)
        assert loaded.huggingface_token == ""
