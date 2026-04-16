from __future__ import annotations

import logging
import re
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


from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    RichLog,
    Select,
    Static,
    TabPane,
    TabbedContent,
    TextArea,
)

from meetscribe.storage.vault import MeetingInfo, MeetingStorage, load_metadata, save_metadata
from meetscribe.storage.speakers import SpeakerRegistry, match_speakers, rewrite_transcript
from meetscribe.config import CONFIG_DIR
from meetscribe.transcription.whisper import AVAILABLE_MODELS

# Dark background colors that work well with white text
SPEAKER_COLORS = [
    "#1a3a5c",  # blue
    "#1a4a2c",  # green
    "#4a1a4a",  # purple
    "#5c2a1a",  # red
    "#1a4a4a",  # teal
    "#5c3a1a",  # orange
    "#2a1a5c",  # indigo
    "#3a4a1a",  # olive
    "#1a2a4a",  # navy
    "#4a3a1a",  # brown
]


def _write_transcript_to_richlog(richlog: RichLog, content: str) -> None:
    """Parse transcript markdown and write speaker-colored lines to a RichLog."""
    richlog.clear()

    # Build speaker → color mapping from order of appearance
    speaker_color_map: dict[str, str] = {}
    current_speaker: str | None = None

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Detect speaker header: **Name:**
        if stripped.startswith("**") and stripped.endswith(":**"):
            speaker_name = stripped[2:-3]
            if speaker_name not in speaker_color_map:
                idx = len(speaker_color_map) % len(SPEAKER_COLORS)
                speaker_color_map[speaker_name] = SPEAKER_COLORS[idx]
            current_speaker = speaker_name
            text = Text(f"\n{speaker_name}", style=f"bold white on {speaker_color_map[speaker_name]}")
            richlog.write(text)
        elif stripped.startswith("---") or stripped.startswith("meeting:") or stripped.startswith("date:") or stripped.startswith("model:") or stripped.startswith("duration:"):
            # Skip frontmatter
            continue
        elif current_speaker and stripped.startswith("["):
            # Transcript line with timestamp
            bg = speaker_color_map[current_speaker]
            text = Text(f" {stripped}", style=f"white on {bg}")
            richlog.write(text)
        elif not current_speaker:
            # Non-diarized transcript or other content
            richlog.write(Text(stripped))


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

    .controls-bar {
        height: auto;
        max-height: 5;
    }

    #transcript-view, #summary-view {
        height: 1fr;
        overflow-y: auto;
    }

    #memos-editor {
        height: 1fr;
    }

    #num-speakers {
        width: 14;
    }
    #speaker-mapping {
        height: auto;
        max-height: 50%;
        overflow-y: auto;
    }
    .speaker-row {
        height: 3;
        layout: horizontal;
    }
    .speaker-label {
        width: 20;
        content-align-vertical: middle;
    }
    .speaker-input {
        width: 1fr;
    }
    .match-indicator {
        width: 12;
        content-align-vertical: middle;
    }
    #transcript-editor {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("r", "rename_meeting", "Rename"),
        ("d", "delete_meeting", "Delete"),
    ]

    def __init__(self, meeting: MeetingInfo) -> None:
        super().__init__()
        self.meeting = meeting
        self._pending_cluster_embeddings: dict[str, list[float]] | None = None
        self._speaker_labels: list[str] = []

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
                Input(placeholder="# speakers", id="num-speakers", max_length=2),
                Button("Transcribe", id="transcribe-btn", variant="primary"),
                Button("Regenerate", id="regenerate-transcript-btn"),
                Button("Edit", id="edit-transcript-btn"),
                classes="controls-bar",
            ),
            Collapsible(
                Static("No speakers detected yet.", id="speaker-mapping-content"),
                Button("Apply Names", id="apply-speakers-btn", variant="primary"),
                title="Speaker Mapping",
                id="speaker-mapping",
                collapsed=True,
            ),
            LoadingIndicator(id="transcript-loading"),
            RichLog(id="transcript-view", highlight=False, markup=False),
            TextArea(id="transcript-editor"),
        )

    def _compose_summary_tab(self) -> Vertical:
        return Vertical(
            Horizontal(
                Select([], id="template-select", prompt="Select template"),
                Select([], id="provider-select", prompt="Select provider"),
                Select([], id="llm-model-select", prompt="Select model"),
                classes="controls-bar",
            ),
            Horizontal(
                Button("Summarize", id="summarize-btn", variant="primary"),
                Button("Regenerate", id="regenerate-summary-btn"),
                Button("Refresh Models", id="refresh-models-btn"),
                classes="controls-bar",
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
        self.query_one("#speaker-mapping", Collapsible).display = False
        self.query_one("#transcript-editor", TextArea).display = False
        self._load_existing_transcript()
        self._load_existing_summary()
        self._load_memos()
        self._populate_templates()
        self._populate_providers()
        self._load_metadata()

    def _load_metadata(self) -> None:
        meta = load_metadata(self.meeting.path)
        num_speakers = meta.get("num_speakers")
        if num_speakers is not None:
            self.query_one("#num-speakers", Input).value = str(num_speakers)
        speaker_map = meta.get("speaker_map")
        if speaker_map:
            labels = sorted(speaker_map.keys())
            suggestions = {label: info["name"] for label, info in speaker_map.items()}
            self._populate_speaker_mapping(labels, suggestions)
            self.query_one("#speaker-mapping", Collapsible).collapsed = True
        else:
            self._detect_speakers_from_transcript()

    def _detect_speakers_from_transcript(self) -> None:
        """Detect speaker labels from an existing diarized transcript."""
        for f in sorted(self.meeting.path.glob("transcript-*.md"), reverse=True):
            content = f.read_text()
            labels = sorted(set(re.findall(r"\*\*(.+?):\*\*", content)))
            if labels:
                self._populate_speaker_mapping(labels)
                self.query_one("#speaker-mapping", Collapsible).collapsed = True
            break

    def _load_existing_transcript(self) -> None:
        """Load the most recent transcript if one exists."""
        for f in sorted(self.meeting.path.glob("transcript-*.md"), reverse=True):
            content = f.read_text()
            _write_transcript_to_richlog(self.query_one("#transcript-view", RichLog), content)
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

    @on(Button.Pressed, "#edit-transcript-btn")
    def do_toggle_edit_transcript(self) -> None:
        """Toggle between colored view and editable text editor."""
        btn = self.query_one("#edit-transcript-btn", Button)
        richlog = self.query_one("#transcript-view", RichLog)
        editor = self.query_one("#transcript-editor", TextArea)

        if editor.display:
            # Save and switch back to colored view
            updated_text = editor.text
            transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
            if transcripts:
                transcripts[0].write_text(updated_text)
            _write_transcript_to_richlog(richlog, updated_text)
            editor.display = False
            richlog.display = True
            btn.label = "Edit"
            self.notify("Transcript saved.")
            # Re-detect speakers in case labels changed
            self._detect_speakers_from_transcript()
        else:
            # Load transcript into editor
            transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
            if not transcripts:
                self.notify("No transcript to edit.", severity="error")
                return
            content = transcripts[0].read_text()
            editor.load_text(content)
            richlog.display = False
            editor.display = True
            btn.label = "Save"

    @on(Button.Pressed, "#transcribe-btn")
    @on(Button.Pressed, "#regenerate-transcript-btn")
    def do_transcribe(self) -> None:
        model_select = self.query_one("#whisper-model", Select)
        model_name = str(model_select.value) if model_select.value != Select.BLANK else "base"
        num_speakers_str = self.query_one("#num-speakers", Input).value.strip()
        num_speakers = int(num_speakers_str) if num_speakers_str.isdigit() else None
        if num_speakers:
            save_metadata(self.meeting.path, {"num_speakers": num_speakers})
        self._run_transcription(model_name, True, num_speakers)

    def _show_loading(self, widget_id: str) -> None:
        self.query_one(f"#{widget_id}").add_class("visible")

    def _hide_loading(self, widget_id: str) -> None:
        self.query_one(f"#{widget_id}").remove_class("visible")

    @work(thread=True)
    def _run_transcription(self, model_name: str, enable_diarization: bool = False, num_speakers: int | None = None) -> None:
        import time as _time
        t_start = _time.monotonic()
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
                live_lines.append(f"[{timestamp}] {text}")
                self.app.call_from_thread(
                    self.query_one("#transcript-view", RichLog).write, f"[{timestamp}] {text}"
                )

            transcript, cluster_embeddings = transcribe_audio(
                audio_path=recording_path,
                model_name=model_name,
                meeting_name=self.meeting.name,
                meeting_date=str(self.meeting.date),
                enable_diarization=enable_diarization,
                num_speakers=num_speakers,
                on_segment=on_segment,
                custom_vocabulary=config.transcription.custom_vocabulary or None,
            )

            transcript_path = storage.transcript_path(self.meeting.name, self.meeting.date, model_name)
            transcript_path.write_text(transcript)
            log.info("Transcription saved to %s", transcript_path)

            self._pending_cluster_embeddings = cluster_embeddings
            self.app.call_from_thread(
                _write_transcript_to_richlog, self.query_one("#transcript-view", RichLog), transcript
            )

            if not cluster_embeddings:
                # Clear stale speaker mapping if re-transcribing without diarization
                self.app.call_from_thread(self._clear_speaker_mapping)

            if cluster_embeddings:
                speakers_path = CONFIG_DIR / "speakers.json"
                registry = SpeakerRegistry(speakers_path)
                import numpy as np
                np_embeddings = {
                    label: np.array(emb) for label, emb in cluster_embeddings.items()
                }
                matches = match_speakers(np_embeddings, registry)
                suggestions = {label: profile.name for label, profile in matches.items()}
                labels = sorted(cluster_embeddings.keys())
                self.app.call_from_thread(self._populate_speaker_mapping, labels, suggestions)

            elapsed = _time.monotonic() - t_start
            m, s = divmod(int(elapsed), 60)
            self.app.call_from_thread(self.notify, f"Transcription complete! ({m}m {s}s)")
        except Exception:
            log.exception("Transcription failed")
            msg = f"Transcription failed. See log: {self.app.log_file}"
            self.app.call_from_thread(self.notify, msg, severity="error")
        finally:
            self.app.call_from_thread(self._hide_loading, "transcript-loading")

    def _clear_speaker_mapping(self) -> None:
        """Hide and reset the speaker mapping section."""
        collapsible = self.query_one("#speaker-mapping", Collapsible)
        for widget in list(collapsible.query(".speaker-row")):
            widget.remove()
        collapsible.display = False
        self._speaker_labels = []
        self._pending_cluster_embeddings = None
        # Clear stale speaker_map from metadata
        save_metadata(self.meeting.path, {"speaker_map": {}})

    def _populate_speaker_mapping(self, speaker_labels: list[str], suggestions: dict[str, str] | None = None) -> None:
        suggestions = suggestions or {}
        collapsible = self.query_one("#speaker-mapping", Collapsible)
        for widget in list(collapsible.query(".speaker-row")):
            widget.remove()
        try:
            collapsible.query_one("#speaker-mapping-content").remove()
        except Exception:
            pass
        apply_btn = collapsible.query_one("#apply-speakers-btn", Button)
        for label in speaker_labels:
            suggested = suggestions.get(label, "")
            indicator = "(matched)" if suggested else ""
            row = Horizontal(
                Static(f"{label} →", classes="speaker-label"),
                Input(value=suggested, placeholder="Enter name",
                      id=f"speaker-input-{label.replace(' ', '-').lower()}", classes="speaker-input"),
                Static(indicator, classes="match-indicator"),
                classes="speaker-row",
            )
            collapsible.mount(row, before=apply_btn)
        self._speaker_labels = speaker_labels
        collapsible.display = True
        collapsible.collapsed = False

    @on(Button.Pressed, "#apply-speakers-btn")
    def do_apply_speaker_names(self) -> None:
        if not self._speaker_labels:
            self.notify("No speakers to map.", severity="error")
            return
        collapsible = self.query_one("#speaker-mapping", Collapsible)
        speakers_path = CONFIG_DIR / "speakers.json"
        registry = SpeakerRegistry(speakers_path)
        speaker_map: dict[str, dict] = {}
        rewrite_map: dict[str, str] = {}
        metadata = load_metadata(self.meeting.path)
        existing_map = metadata.get("speaker_map", {})

        for label in self._speaker_labels:
            input_id = f"speaker-input-{label.replace(' ', '-').lower()}"
            try:
                name = self.query_one(f"#{input_id}", Input).value.strip()
            except Exception:
                continue
            if not name:
                continue
            import numpy as np
            matched_profile = None
            if self._pending_cluster_embeddings and label in self._pending_cluster_embeddings:
                np_emb = np.array(self._pending_cluster_embeddings[label])
                matches = match_speakers({label: np_emb}, registry)
                if label in matches and matches[label].name == name:
                    matched_profile = matches[label]
            if not matched_profile:
                for s in registry.list_speakers():
                    if s.name == name:
                        matched_profile = s
                        break
            if not matched_profile:
                matched_profile = registry.create_speaker(name)
            if self._pending_cluster_embeddings and label in self._pending_cluster_embeddings:
                meeting_ref = f"{self.meeting.date}/{self.meeting.name}"
                registry.add_embedding(matched_profile.id, self._pending_cluster_embeddings[label], meeting_ref)
            current_label = label
            if label in existing_map:
                current_label = existing_map[label].get("name", label)
            rewrite_map[current_label] = name
            speaker_map[label] = {"speaker_id": matched_profile.id, "name": name, "original_label": label}

        if not rewrite_map:
            self.notify("No names entered.", severity="warning")
            return
        transcripts = sorted(self.meeting.path.glob("transcript-*.md"), reverse=True)
        if transcripts:
            transcript_text = transcripts[0].read_text()
            updated = rewrite_transcript(transcript_text, rewrite_map)
            transcripts[0].write_text(updated)
            _write_transcript_to_richlog(self.query_one("#transcript-view", RichLog), updated)
        save_metadata(self.meeting.path, {"speaker_map": speaker_map})
        for widget in collapsible.query(".match-indicator"):
            widget.update("")
        self.notify("Speaker names applied!")

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
        import time as _time
        t_start = _time.monotonic()
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
            elapsed = _time.monotonic() - t_start
            m, s = divmod(int(elapsed), 60)
            self.app.call_from_thread(self.notify, f"Summary complete! ({m}m {s}s)")
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

    def action_rename_meeting(self) -> None:
        from meetscribe.tui.screens.dialogs import RenameDialog
        self.app.push_screen(
            RenameDialog(self.meeting.name),
            callback=self._do_rename,
        )

    def _do_rename(self, new_name: str | None) -> None:
        if not new_name:
            return
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        try:
            self.meeting = storage.rename_meeting(self.meeting, new_name)
            self.query_one(".title", Static).update(
                f"Meeting: {self.meeting.name} ({self.meeting.date})"
            )
            self.notify(f"Renamed to: {new_name}")
        except ValueError as e:
            self.notify(str(e), severity="error")

    def action_delete_meeting(self) -> None:
        from meetscribe.tui.screens.dialogs import ConfirmDialog
        self.app.push_screen(
            ConfirmDialog(f"Delete '{self.meeting.name}' ({self.meeting.date})?\nThis cannot be undone."),
            callback=self._do_delete,
        )

    def _do_delete(self, confirmed: bool) -> None:
        if not confirmed:
            return
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        storage.delete_meeting(self.meeting)
        self.notify(f"Deleted: {self.meeting.name}")
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()
