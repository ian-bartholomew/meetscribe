from __future__ import annotations

from datetime import date

import logging

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Footer, Static, Input

from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage, MeetingInfo

log = logging.getLogger("meetscribe.home")


class HomeScreen(Screen):
    """Landing screen — start recording or browse past meetings."""

    BINDINGS = [
        ("n", "new_recording", "New Recording"),
        ("r", "rename_meeting", "Rename"),
        ("d", "delete_meeting", "Delete"),
        ("b", "bulk_process", "Bulk Process"),
        ("s", "app.push_screen('settings')", "Settings"),
    ]

    CSS = """
    #meeting-table {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._meetings: list[MeetingInfo] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Meetscribe", classes="title"),
            Horizontal(
                Button("New Recording", id="new-recording", variant="primary"),
                Button("Bulk Process Missing", id="bulk-process"),
            ),
            DataTable(id="meeting-table", cursor_type="row"),
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#meeting-table", DataTable)
        table.add_columns("Date", "Name", "Duration", "Transcript", "Summary")
        self._refresh_meetings()

    def _refresh_meetings(self) -> None:
        config = self.app.config
        if not config.vault.root:
            return
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        self._meetings = storage.list_meetings()
        table = self.query_one("#meeting-table", DataTable)
        table.clear()
        for meeting in self._meetings:
            table.add_row(
                str(meeting.date),
                meeting.name,
                meeting.duration or "-",
                "yes" if meeting.has_transcript else "-",
                "yes" if meeting.has_summary else "-",
                key=meeting.name,
            )

    @on(Button.Pressed, "#new-recording")
    def action_new_recording(self) -> None:
        from meetscribe.tui.screens.recording import RecordingScreen
        self.app.push_screen(RecordingScreen())

    def _get_selected_meeting(self) -> MeetingInfo | None:
        table = self.query_one("#meeting-table", DataTable)
        if table.cursor_row is not None and 0 <= table.cursor_row < len(self._meetings):
            return self._meetings[table.cursor_row]
        return None

    def action_delete_meeting(self) -> None:
        meeting = self._get_selected_meeting()
        if not meeting:
            self.notify("No meeting selected.", severity="error")
            return
        from meetscribe.tui.screens.dialogs import ConfirmDialog
        self.app.push_screen(
            ConfirmDialog(f"Delete '{meeting.name}' ({meeting.date})?\nThis cannot be undone."),
            callback=lambda confirmed: self._do_delete(meeting) if confirmed else None,
        )

    def _do_delete(self, meeting: MeetingInfo) -> None:
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        storage.delete_meeting(meeting)
        self.notify(f"Deleted: {meeting.name}")
        self._refresh_meetings()

    def action_rename_meeting(self) -> None:
        meeting = self._get_selected_meeting()
        if not meeting:
            self.notify("No meeting selected.", severity="error")
            return
        from meetscribe.tui.screens.dialogs import RenameDialog
        self.app.push_screen(
            RenameDialog(meeting.name),
            callback=lambda new_name: self._do_rename(meeting, new_name) if new_name else None,
        )

    def _do_rename(self, meeting: MeetingInfo, new_name: str) -> None:
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        try:
            storage.rename_meeting(meeting, new_name)
            self.notify(f"Renamed to: {new_name}")
            self._refresh_meetings()
        except ValueError as e:
            self.notify(str(e), severity="error")

    @on(Button.Pressed, "#bulk-process")
    def action_bulk_process(self) -> None:
        need_transcript = [m for m in self._meetings if m.has_recording and not m.has_transcript]
        need_summary = [m for m in self._meetings if m.has_transcript and not m.has_summary]
        total = len(need_transcript) + len(need_summary)
        if total == 0:
            self.notify("All meetings are already processed.")
            return

        from meetscribe.tui.screens.meeting import _find_templates_dir
        from meetscribe.templates.engine import TemplateEngine
        templates = TemplateEngine(_find_templates_dir()).list_templates()
        config = self.app.config
        providers = list(config.summarization.endpoints.keys())

        from meetscribe.tui.screens.bulk_dialog import BulkProcessDialog
        self.app.push_screen(
            BulkProcessDialog(
                templates=templates,
                providers=providers,
                default_whisper_model=config.transcription.default_model,
                default_template="default",
                default_provider=config.summarization.default_provider,
                default_llm_model=config.summarization.default_model,
                num_transcriptions=len(need_transcript),
                num_summaries=len(need_summary),
            ),
            callback=self._on_bulk_config,
        )

    def _on_bulk_config(self, config_result) -> None:
        if config_result is None:
            return
        self._do_bulk_process(config_result)

    @work(thread=True, exclusive=True)
    def _do_bulk_process(self, bulk_config) -> None:
        from meetscribe.storage.vault import load_metadata
        from meetscribe.transcription.whisper import transcribe_audio
        from meetscribe.summarization.provider import SummarizationProvider
        from meetscribe.templates.engine import TemplateEngine
        from meetscribe.tui.screens.meeting import _find_templates_dir

        log.info("Bulk process starting: whisper=%s, template=%s, provider=%s, model=%s, diarize=%s",
                 bulk_config.whisper_model, bulk_config.template,
                 bulk_config.provider, bulk_config.llm_model,
                 bulk_config.enable_diarization)

        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        model_name = bulk_config.whisper_model

        # Re-scan meetings fresh (dialog may have changed state)
        all_meetings = storage.list_meetings()

        # Phase 1: Transcribe meetings missing transcripts
        need_transcript = [m for m in all_meetings if m.has_recording and not m.has_transcript]
        log.info("Bulk: %d meetings need transcription", len(need_transcript))

        for i, meeting in enumerate(need_transcript):
            log.info("Bulk transcribing %d/%d: %s", i + 1, len(need_transcript), meeting.name)

            # Get duration for progress display
            duration_str = meeting.duration or "unknown length"
            self.app.call_from_thread(
                self.notify, f"Transcribing {i+1}/{len(need_transcript)}: {meeting.name} ({duration_str})..."
            )
            try:
                # Find recording
                recording_path = None
                for ext in ("flac", "mp3", "wav", "m4a", "ogg"):
                    p = meeting.path / f"recording.{ext}"
                    if p.exists():
                        recording_path = p
                        break
                if not recording_path:
                    log.info("Bulk: no recording found for %s, skipping", meeting.name)
                    continue

                # Check metadata for num_speakers
                meta = load_metadata(meeting.path)
                num_speakers = meta.get("num_speakers")

                # Progress callback — show segment count as transcription streams
                segment_count = [0]
                def on_segment(idx, timestamp, text):
                    segment_count[0] = idx + 1
                    if segment_count[0] % 10 == 0:
                        self.app.call_from_thread(
                            self.notify, f"Transcribing {i+1}/{len(need_transcript)}: {meeting.name} — {segment_count[0]} segments..."
                        )

                transcript = transcribe_audio(
                    audio_path=recording_path,
                    model_name=model_name,
                    meeting_name=meeting.name,
                    meeting_date=str(meeting.date),
                    enable_diarization=bulk_config.enable_diarization and num_speakers is not None and num_speakers > 1,
                    num_speakers=num_speakers,
                    on_segment=on_segment,
                )
                transcript_path = storage.transcript_path(meeting.name, meeting.date, model_name)
                transcript_path.write_text(transcript)
                log.info("Bulk transcribed: %s", meeting.name)
            except Exception:
                log.exception("Bulk transcription failed for %s", meeting.name)

        # Phase 2: Summarize meetings missing summaries (re-scan to pick up new transcripts)
        all_meetings = storage.list_meetings()
        need_summary = [m for m in all_meetings if m.has_transcript and not m.has_summary]
        log.info("Bulk: %d meetings need summarization", len(need_summary))

        if need_summary:
            templates_dir = _find_templates_dir()
            engine = TemplateEngine(templates_dir)
            template_name = bulk_config.template
            provider_name = bulk_config.provider
            endpoint = config.summarization.endpoints.get(provider_name, "")
            llm_model = bulk_config.llm_model

            if not endpoint:
                log.error("Bulk: no endpoint for provider '%s'", provider_name)
                self.app.call_from_thread(
                    self.notify, f"No endpoint for provider '{provider_name}'. Skipping summaries.", severity="error"
                )
            else:
                for i, meeting in enumerate(need_summary):
                    log.info("Bulk summarizing %d/%d: %s", i + 1, len(need_summary), meeting.name)
                    self.app.call_from_thread(
                        self.notify, f"Summarizing {i+1}/{len(need_summary)}: {meeting.name}..."
                    )
                    try:
                        # Find latest transcript
                        transcripts = sorted(meeting.path.glob("transcript-*.md"), reverse=True)
                        if not transcripts:
                            continue
                        transcript_text = transcripts[0].read_text()

                        memos_path = meeting.path / "memos.md"
                        memos_text = memos_path.read_text() if memos_path.exists() else ""

                        rendered = engine.render(
                            template_name=template_name,
                            transcript=transcript_text,
                            memos=memos_text,
                            meeting_name=meeting.name,
                            date=str(meeting.date),
                            duration="",
                        )

                        llm = SummarizationProvider(base_url=endpoint, model=llm_model)
                        summary = llm.summarize(
                            system_prompt="You are a meeting summarizer. Produce a clear, well-structured summary.",
                            user_prompt=rendered,
                        )

                        summary_path = storage.summary_path(meeting.name, meeting.date, template_name)
                        full_summary = (
                            f"---\n"
                            f"meeting: {meeting.name}\n"
                            f"date: {meeting.date}\n"
                            f"template: {template_name}\n"
                            f"provider: {provider_name}\n"
                            f"model: {llm_model}\n"
                            f"---\n\n"
                            f"{summary}"
                        )
                        summary_path.write_text(full_summary)
                        log.info("Bulk summarized: %s", meeting.name)
                    except Exception:
                        log.exception("Bulk summarization failed for %s", meeting.name)

        self.app.call_from_thread(self._refresh_meetings)
        log.info("Bulk processing complete!")
        self.app.call_from_thread(self.notify, "Bulk processing complete!")

    @on(DataTable.RowSelected, "#meeting-table")
    def on_meeting_selected(self, event: DataTable.RowSelected) -> None:
        meeting = self._get_selected_meeting()
        if meeting:
            from meetscribe.tui.screens.meeting import MeetingScreen
            self.app.push_screen(MeetingScreen(meeting))
