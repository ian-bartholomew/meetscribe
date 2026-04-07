from __future__ import annotations

import logging
import traceback
from pathlib import Path

from textual import on, work

log = logging.getLogger("meetscribe.meeting")


def _find_templates_dir() -> Path:
    """Find the templates directory by walking up from this file."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = d / "templates"
        if candidate.exists() and any(candidate.glob("*.md")):
            return candidate
        d = d.parent
    # Fallback
    return Path(__file__).parent.parent.parent.parent / "templates"


from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    Static,
    TabPane,
    TabbedContent,
    TextArea,
)

from meetscribe.storage.vault import MeetingInfo, MeetingStorage
from meetscribe.transcription.whisper import AVAILABLE_MODELS


class MeetingScreen(Screen):
    """View and manage meeting artifacts: recording, transcript, summary, memos."""

    CSS = """
    #transcript-loading, #summary-loading {
        display: none;
        height: 3;
    }
    #transcript-loading.visible, #summary-loading.visible {
        display: block;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def __init__(self, meeting: MeetingInfo) -> None:
        super().__init__()
        self.meeting = meeting

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Meeting: {self.meeting.name} ({self.meeting.date})", classes="title")

        with TabbedContent("Recording", "Transcript", "Summary", "Memos"):
            with TabPane("Recording", id="recording-tab"):
                yield self._compose_recording_tab()
            with TabPane("Transcript", id="transcript-tab"):
                yield self._compose_transcript_tab()
            with TabPane("Summary", id="summary-tab"):
                yield self._compose_summary_tab()
            with TabPane("Memos", id="memos-tab"):
                yield self._compose_memos_tab()

        yield Footer()

    def _find_recording(self) -> Path | None:
        """Find recording file in any supported format."""
        for ext in ("flac", "mp3", "wav", "m4a", "ogg"):
            path = self.meeting.path / f"recording.{ext}"
            if path.exists():
                return path
        return None

    def _compose_recording_tab(self) -> Vertical:
        recording_path = self._find_recording()
        if recording_path:
            import soundfile as sf
            size_mb = recording_path.stat().st_size / (1024 * 1024)
            try:
                info = sf.info(str(recording_path))
                duration_s = info.duration
                h = int(duration_s // 3600)
                m = int((duration_s % 3600) // 60)
                s = int(duration_s % 60)
                duration_str = f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:
                duration_str = "unknown"
            info_text = f"Recording: {recording_path.name}\nSize: {size_mb:.1f} MB\nDuration: {duration_str}"
        else:
            info_text = "No recording found."
        return Vertical(Label(info_text, id="recording-info"))

    def _compose_transcript_tab(self) -> Vertical:
        model_options = [(m, m) for m in AVAILABLE_MODELS]
        config = self.app.config if hasattr(self, "app") and self.app else None
        default_model = config.transcription.default_model if config else "base"

        return Vertical(
            Horizontal(
                Select(model_options, value=default_model, id="whisper-model"),
                Checkbox("Identify speakers", id="diarize-checkbox"),
                Button("Transcribe", id="transcribe-btn", variant="primary"),
                Button("Regenerate", id="regenerate-transcript-btn"),
            ),
            LoadingIndicator(id="transcript-loading"),
            Markdown("*No transcript yet. Select a model and click Transcribe.*", id="transcript-view"),
        )

    def _compose_summary_tab(self) -> Vertical:
        return Vertical(
            Horizontal(
                Select([], id="template-select", prompt="Select template"),
                Select([], id="provider-select", prompt="Select provider"),
                Select([], id="llm-model-select", prompt="Select model"),
            ),
            Horizontal(
                Button("Summarize", id="summarize-btn", variant="primary"),
                Button("Regenerate", id="regenerate-summary-btn"),
                Button("Refresh Models", id="refresh-models-btn"),
            ),
            LoadingIndicator(id="summary-loading"),
            Markdown("*No summary yet. Select a template and model, then click Summarize.*", id="summary-view"),
        )

    def _compose_memos_tab(self) -> Vertical:
        return Vertical(
            TextArea(id="memos-editor"),
            Button("Save Memos", id="save-memos-btn", variant="primary"),
        )

    def on_mount(self) -> None:
        self._load_existing_transcript()
        self._load_existing_summary()
        self._load_memos()
        self._populate_templates()
        self._populate_providers()

    def _load_existing_transcript(self) -> None:
        """Load the most recent transcript if one exists."""
        for f in sorted(self.meeting.path.glob("transcript-*.md"), reverse=True):
            content = f.read_text()
            self.query_one("#transcript-view", Markdown).update(content)
            break

    def _load_existing_summary(self) -> None:
        """Load the most recent summary if one exists."""
        for f in sorted(self.meeting.path.glob("summary-*.md"), reverse=True):
            content = f.read_text()
            self.query_one("#summary-view", Markdown).update(content)
            break

    def _load_memos(self) -> None:
        memos_path = self.meeting.path / "memos.md"
        if memos_path.exists():
            self.query_one("#memos-editor", TextArea).load_text(memos_path.read_text())

    def _populate_templates(self) -> None:
        from meetscribe.templates.engine import TemplateEngine
        templates_dir = _find_templates_dir()
        engine = TemplateEngine(templates_dir)
        names = engine.list_templates()
        if names:
            options = [(n, n) for n in names]
            self.query_one("#template-select", Select).set_options(options)

    def _populate_providers(self) -> None:
        config = self.app.config
        provider_options = [(k, k) for k in config.summarization.endpoints]
        self.query_one("#provider-select", Select).set_options(provider_options)

    @on(Select.Changed, "#provider-select")
    def on_provider_changed(self, event: Select.Changed) -> None:
        """When provider changes, fetch available models."""
        if event.value and event.value != Select.BLANK:
            self._fetch_models(str(event.value))

    @work(thread=True)
    def _fetch_models(self, provider: str) -> None:
        config = self.app.config
        endpoint = config.summarization.endpoints.get(provider, "")
        if not endpoint:
            return
        from meetscribe.summarization.provider import SummarizationProvider
        p = SummarizationProvider(base_url=endpoint, model="")
        models = p.list_models()
        if models:
            options = [(m, m) for m in models]
            self.app.call_from_thread(
                self.query_one("#llm-model-select", Select).set_options, options
            )

    @on(Button.Pressed, "#refresh-models-btn")
    def refresh_models(self) -> None:
        provider_select = self.query_one("#provider-select", Select)
        if provider_select.value and provider_select.value != Select.BLANK:
            self._fetch_models(str(provider_select.value))

    @on(Button.Pressed, "#transcribe-btn")
    @on(Button.Pressed, "#regenerate-transcript-btn")
    def do_transcribe(self) -> None:
        model_select = self.query_one("#whisper-model", Select)
        model_name = str(model_select.value) if model_select.value != Select.BLANK else "base"
        diarize = self.query_one("#diarize-checkbox", Checkbox).value
        self._run_transcription(model_name, diarize)

    def _show_loading(self, widget_id: str) -> None:
        self.query_one(f"#{widget_id}").add_class("visible")

    def _hide_loading(self, widget_id: str) -> None:
        self.query_one(f"#{widget_id}").remove_class("visible")

    @work(thread=True)
    def _run_transcription(self, model_name: str, enable_diarization: bool = False) -> None:
        self.app.call_from_thread(self._show_loading, "transcript-loading")
        label = f"Transcribing with {model_name}"
        if enable_diarization:
            label += " + speaker identification"
        self.app.call_from_thread(self.notify, f"{label}...")
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        recording_path = self._find_recording()

        if not recording_path:
            self.app.call_from_thread(self._hide_loading, "transcript-loading")
            self.app.call_from_thread(self.notify, "No recording found.", severity="error")
            return

        try:
            log.info("Starting transcription: %s with model %s, diarize=%s", recording_path, model_name, enable_diarization)
            from meetscribe.transcription.whisper import transcribe_audio

            # Stream segments to the UI as they arrive
            live_lines: list[str] = []

            def on_segment(idx: int, timestamp: str, text: str) -> None:
                live_lines.append(f"[{timestamp}] {text}\n")
                preview = "\n".join(live_lines)
                self.app.call_from_thread(
                    self.query_one("#transcript-view", Markdown).update, preview
                )

            transcript = transcribe_audio(
                audio_path=recording_path,
                model_name=model_name,
                meeting_name=self.meeting.name,
                meeting_date=str(self.meeting.date),
                enable_diarization=enable_diarization,
                on_segment=on_segment,
            )

            transcript_path = storage.transcript_path(self.meeting.name, self.meeting.date, model_name)
            transcript_path.write_text(transcript)
            log.info("Transcription saved to %s", transcript_path)

            # Final update with full formatted transcript (includes frontmatter + diarization)
            self.app.call_from_thread(self.query_one("#transcript-view", Markdown).update, transcript)
            self.app.call_from_thread(self.notify, "Transcription complete!")
        except Exception:
            log.exception("Transcription failed")
            msg = f"Transcription failed. See log: {self.app.log_file}"
            self.app.call_from_thread(self.notify, msg, severity="error")
        finally:
            self.app.call_from_thread(self._hide_loading, "transcript-loading")

    def _select_value(self, select: Select) -> str | None:
        """Get a Select widget's value as a string, or None if unselected."""
        val = select.value
        if val is Select.BLANK or not isinstance(val, str):
            return None
        return val

    @on(Button.Pressed, "#summarize-btn")
    @on(Button.Pressed, "#regenerate-summary-btn")
    def do_summarize(self) -> None:
        template = self._select_value(self.query_one("#template-select", Select))
        provider = self._select_value(self.query_one("#provider-select", Select))
        model = self._select_value(self.query_one("#llm-model-select", Select))

        if not template:
            self.notify("Please select a template.", severity="error")
            return
        if not provider:
            self.notify("Please select a provider.", severity="error")
            return
        if not model:
            self.notify("Please select a model.", severity="error")
            return

        self._run_summarization(
            template_name=template,
            provider=provider,
            model=model,
        )

    @work(thread=True)
    def _run_summarization(self, template_name: str, provider: str, model: str) -> None:
        self.app.call_from_thread(self._show_loading, "summary-loading")
        self.app.call_from_thread(self.notify, f"Summarizing with {provider}/{model}...")
        config = self.app.config

        try:
            log.info("Starting summarization: template=%s provider=%s model=%s", template_name, provider, model)

            # Find the latest transcript
            transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
            if not transcripts:
                self.app.call_from_thread(self.notify, "No transcript found. Transcribe first.", severity="error")
                return
            transcript_text = transcripts[0].read_text()

            # Load memos
            memos_path = self.meeting.path / "memos.md"
            memos_text = memos_path.read_text() if memos_path.exists() else ""

            # Render template
            from meetscribe.templates.engine import TemplateEngine
            templates_dir = _find_templates_dir()
            engine = TemplateEngine(templates_dir)

            rendered = engine.render(
                template_name=template_name,
                transcript=transcript_text,
                memos=memos_text,
                meeting_name=self.meeting.name,
                date=str(self.meeting.date),
                duration="",
            )

            # Send to LLM
            from meetscribe.summarization.provider import SummarizationProvider
            endpoint = config.summarization.endpoints.get(provider, "")
            log.info("Provider=%r, endpoint=%r, endpoints=%r", provider, endpoint, config.summarization.endpoints)
            if not endpoint:
                self.app.call_from_thread(self.notify, f"No endpoint configured for provider '{provider}'", severity="error")
                return
            llm = SummarizationProvider(base_url=endpoint, model=model)
            summary = llm.summarize(
                system_prompt="You are a meeting summarizer. Produce a clear, well-structured summary.",
                user_prompt=rendered,
            )

            # Save
            storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
            summary_path = storage.summary_path(self.meeting.name, self.meeting.date, template_name)

            # Add frontmatter
            full_summary = (
                f"---\n"
                f"meeting: {self.meeting.name}\n"
                f"date: {self.meeting.date}\n"
                f"template: {template_name}\n"
                f"provider: {provider}\n"
                f"model: {model}\n"
                f"---\n\n"
                f"{summary}"
            )
            summary_path.write_text(full_summary)
            log.info("Summary saved to %s", summary_path)

            self.app.call_from_thread(self.query_one("#summary-view", Markdown).update, full_summary)
            self.app.call_from_thread(self.notify, "Summary complete!")
        except Exception:
            log.exception("Summarization failed")
            msg = f"Summarization failed. See log: {self.app.log_file}"
            self.app.call_from_thread(self.notify, msg, severity="error")
        finally:
            self.app.call_from_thread(self._hide_loading, "summary-loading")

    @on(Button.Pressed, "#save-memos-btn")
    def save_memos(self) -> None:
        memos_text = self.query_one("#memos-editor", TextArea).text
        memos_path = self.meeting.path / "memos.md"
        memos_path.parent.mkdir(parents=True, exist_ok=True)
        memos_path.write_text(memos_text)
        self.notify("Memos saved.")

    def action_go_back(self) -> None:
        self.app.pop_screen()
