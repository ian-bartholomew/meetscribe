"""Bulk process configuration dialog."""
from __future__ import annotations

from dataclasses import dataclass

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, Select, Static

from meetscribe.transcription.whisper import AVAILABLE_MODELS


@dataclass
class BulkProcessConfig:
    whisper_model: str
    template: str
    provider: str
    llm_model: str
    enable_diarization: bool


class BulkProcessDialog(ModalScreen[BulkProcessConfig | None]):
    """Modal dialog to configure bulk processing options."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    CSS = """
    BulkProcessDialog {
        align: center middle;
    }
    #bulk-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        overflow-y: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #bulk-buttons {
        height: auto;
        margin-top: 1;
    }
    #bulk-buttons Button {
        margin: 0 1;
    }
    .bulk-section {
        margin-top: 1;
    }
    """

    def __init__(
        self,
        templates: list[str],
        providers: list[str],
        default_whisper_model: str,
        default_template: str,
        default_provider: str,
        default_llm_model: str,
        num_transcriptions: int,
        num_summaries: int,
    ) -> None:
        super().__init__()
        self._templates = templates
        self._providers = providers
        self._default_whisper_model = default_whisper_model
        self._default_template = default_template
        self._default_provider = default_provider
        self._default_llm_model = default_llm_model
        self._num_transcriptions = num_transcriptions
        self._num_summaries = num_summaries

    def compose(self) -> ComposeResult:
        with Vertical(id="bulk-dialog"):
            yield Static("Bulk Process Missing Meetings")
            yield Label(
                f"{self._num_transcriptions} to transcribe, "
                f"{self._num_summaries}+ to summarize"
            )

            yield Static("Transcription", classes="bulk-section")
            yield Select(
                [(m, m) for m in AVAILABLE_MODELS],
                value=self._default_whisper_model,
                id="bulk-whisper-model",
            )
            yield Checkbox("Identify speakers (where # speakers is set)", id="bulk-diarize")

            yield Static("Summarization", classes="bulk-section")
            yield Select(
                [(t, t) for t in self._templates],
                value=self._default_template if self._default_template in self._templates else (self._templates[0] if self._templates else Select.BLANK),
                id="bulk-template",
            )
            yield Select(
                [(p, p) for p in self._providers],
                value=self._default_provider if self._default_provider in self._providers else (self._providers[0] if self._providers else Select.BLANK),
                id="bulk-provider",
            )
            yield Select(
                [],
                id="bulk-llm-model",
                prompt="Select model (pick provider first)",
            )

            with Horizontal(id="bulk-buttons"):
                yield Button("Start", id="bulk-start", variant="primary")
                yield Button("Cancel", id="bulk-cancel")

    def on_mount(self) -> None:
        # Trigger model fetch for the default provider
        provider_select = self.query_one("#bulk-provider", Select)
        if provider_select.value and provider_select.value != Select.BLANK:
            self._fetch_models(str(provider_select.value))

    @on(Select.Changed, "#bulk-provider")
    def on_provider_changed(self, event: Select.Changed) -> None:
        if event.value and event.value != Select.BLANK:
            self._fetch_models(str(event.value))

    @work(thread=True)
    def _fetch_models(self, provider: str) -> None:
        from meetscribe.summarization.provider import SummarizationProvider
        config = self.app.config
        endpoint = config.summarization.endpoints.get(provider, "")
        if not endpoint:
            return
        p = SummarizationProvider(base_url=endpoint, model="")
        models = p.list_models()
        if models:
            options = [(m, m) for m in models]
            default = self._default_llm_model

            def _update() -> None:
                model_select = self.query_one("#bulk-llm-model", Select)
                model_select.set_options(options)
                if default in [m for m, _ in options]:
                    model_select.value = default

            self.app.call_from_thread(_update)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#bulk-start")
    def on_start(self) -> None:
        whisper_select = self.query_one("#bulk-whisper-model", Select)
        template_select = self.query_one("#bulk-template", Select)
        provider_select = self.query_one("#bulk-provider", Select)
        llm_select = self.query_one("#bulk-llm-model", Select)

        whisper_model = str(whisper_select.value) if whisper_select.value != Select.BLANK else "base"

        if not template_select.value or template_select.value == Select.BLANK:
            self.notify("Please select a template.", severity="error")
            return
        if not provider_select.value or provider_select.value == Select.BLANK:
            self.notify("Please select a provider.", severity="error")
            return
        if not llm_select.value or not isinstance(llm_select.value, str):
            self.notify("Please select an LLM model.", severity="error")
            return

        self.dismiss(BulkProcessConfig(
            whisper_model=whisper_model,
            template=str(template_select.value),
            provider=str(provider_select.value),
            llm_model=str(llm_select.value),
            enable_diarization=self.query_one("#bulk-diarize", Checkbox).value,
        ))

    @on(Button.Pressed, "#bulk-cancel")
    def on_cancel(self) -> None:
        self.dismiss(None)
