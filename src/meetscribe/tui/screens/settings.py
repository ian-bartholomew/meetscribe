from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from meetscribe.config import save_config, MeetscribeConfig
from meetscribe.transcription.whisper import AVAILABLE_MODELS


class SettingsScreen(Screen):
    """Configure app defaults."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        config: MeetscribeConfig = self.app.config
        yield Header()
        yield Vertical(
            Static("Settings", classes="title"),

            Label("Vault Root Path:"),
            Input(value=config.vault.root, id="vault-root"),

            Label("Meetings Folder (relative to vault):"),
            Input(value=config.vault.meetings_folder, id="meetings-folder"),

            Label("System Audio Device (e.g. BlackHole 2ch):"),
            Input(value=config.audio.device_name, id="device-name"),

            Label("Microphone Device (blank to disable):"),
            Input(value=config.audio.mic_device_name, id="mic-device-name"),

            Label("Sample Rate:"),
            Input(value=str(config.audio.sample_rate), id="sample-rate"),

            Label("Channels:"),
            Input(value=str(config.audio.channels), id="channels"),

            Label("Default Whisper Model:"),
            Select(
                [(m, m) for m in AVAILABLE_MODELS],
                value=config.transcription.default_model,
                id="default-whisper-model",
            ),

            Label("Default LLM Provider:"),
            Select(
                [(k, k) for k in config.summarization.endpoints],
                value=config.summarization.default_provider,
                id="default-provider",
            ),

            Label("Default LLM Model:"),
            Input(value=config.summarization.default_model, id="default-llm-model"),

            Label("Ollama Endpoint:"),
            Input(value=config.summarization.endpoints.get("ollama", ""), id="ollama-endpoint"),

            Label("LM Studio Endpoint:"),
            Input(value=config.summarization.endpoints.get("lmstudio", ""), id="lmstudio-endpoint"),

            Button("Save Settings", id="save-btn", variant="primary"),
        )
        yield Footer()

    @on(Button.Pressed, "#save-btn")
    def save_settings(self) -> None:
        config: MeetscribeConfig = self.app.config

        config.vault.root = self.query_one("#vault-root", Input).value
        config.vault.meetings_folder = self.query_one("#meetings-folder", Input).value
        config.audio.device_name = self.query_one("#device-name", Input).value
        config.audio.mic_device_name = self.query_one("#mic-device-name", Input).value
        config.audio.sample_rate = int(self.query_one("#sample-rate", Input).value)
        config.audio.channels = int(self.query_one("#channels", Input).value)

        model_select = self.query_one("#default-whisper-model", Select)
        if model_select.value != Select.BLANK:
            config.transcription.default_model = str(model_select.value)

        provider_select = self.query_one("#default-provider", Select)
        if provider_select.value != Select.BLANK:
            config.summarization.default_provider = str(provider_select.value)

        config.summarization.default_model = self.query_one("#default-llm-model", Input).value
        config.summarization.endpoints["ollama"] = self.query_one("#ollama-endpoint", Input).value
        config.summarization.endpoints["lmstudio"] = self.query_one("#lmstudio-endpoint", Input).value

        save_config(config)
        self.notify("Settings saved.")

    def action_go_back(self) -> None:
        self.app.pop_screen()
