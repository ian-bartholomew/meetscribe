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
    mic_device_name: str = ""
    sample_rate: int = 48000
    channels: int = 2


@dataclass
class TranscriptionConfig:
    default_model: str = "base"
    custom_vocabulary: list[str] = field(default_factory=list)


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
    log_level: str = "INFO"


def default_config() -> MeetscribeConfig:
    return MeetscribeConfig()


def _config_to_dict(cfg: MeetscribeConfig) -> dict[str, Any]:
    # log_level must come before sections with sub-tables (like summarization.endpoints)
    # to avoid TOML parsing it as part of that section
    return {
        "log_level": cfg.log_level,
        "vault": {
            "root": cfg.vault.root,
            "meetings_folder": cfg.vault.meetings_folder,
        },
        "audio": {
            "device_name": cfg.audio.device_name,
            "mic_device_name": cfg.audio.mic_device_name,
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.audio.channels,
        },
        "transcription": {
            "default_model": cfg.transcription.default_model,
            "custom_vocabulary": cfg.transcription.custom_vocabulary,
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
    if "log_level" in data:
        cfg.log_level = data["log_level"]
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
