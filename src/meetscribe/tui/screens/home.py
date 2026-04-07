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
        ("i", "import_hyprnote", "Import Hyprnote"),
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
                Button("Import from Hyprnote", id="import-hyprnote"),
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

    @on(Button.Pressed, "#import-hyprnote")
    def action_import_hyprnote(self) -> None:
        self._do_import()

    @work(thread=True)
    def _do_import(self) -> None:
        try:
            from meetscribe.importer import import_all_hyprnote
            config = self.app.config
            storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
            imported = import_all_hyprnote(storage)
            msg = f"Imported {len(imported)} session(s) from Hyprnote"
            log.info(msg)
            self.app.call_from_thread(self.notify, msg)
            self.app.call_from_thread(self._refresh_meetings)
        except Exception:
            log.exception("Import failed")
            self.app.call_from_thread(self.notify, "Import failed. Check log.", severity="error")

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
        # Count what needs processing
        need_transcript = [m for m in self._meetings if m.has_recording and not m.has_transcript]
        need_summary = [m for m in self._meetings if m.has_transcript and not m.has_summary]
        total = len(need_transcript) + len(need_summary)
        if total == 0:
            self.notify("All meetings are already processed.")
            return
        self.notify(f"Starting bulk process: {len(need_transcript)} transcriptions, {len(need_summary)} summaries...")
        self._do_bulk_process()

    @work(thread=True)
    def _do_bulk_process(self) -> None:
        import time as _time
        from pathlib import Path
        from meetscribe.storage.vault import load_metadata
        from meetscribe.transcription.whisper import transcribe_audio
        from meetscribe.summarization.provider import SummarizationProvider
        from meetscribe.templates.engine import TemplateEngine
        from meetscribe.tui.screens.meeting import _find_templates_dir

        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        model_name = config.transcription.default_model

        # Phase 1: Transcribe meetings missing transcripts
        need_transcript = [m for m in self._meetings if m.has_recording and not m.has_transcript]
        for i, meeting in enumerate(need_transcript):
            self.app.call_from_thread(
                self.notify, f"Transcribing {i+1}/{len(need_transcript)}: {meeting.name}..."
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
                    continue

                # Check metadata for num_speakers
                meta = load_metadata(meeting.path)
                num_speakers = meta.get("num_speakers")

                transcript = transcribe_audio(
                    audio_path=recording_path,
                    model_name=model_name,
                    meeting_name=meeting.name,
                    meeting_date=str(meeting.date),
                    enable_diarization=num_speakers is not None and num_speakers > 1,
                    num_speakers=num_speakers,
                )
                transcript_path = storage.transcript_path(meeting.name, meeting.date, model_name)
                transcript_path.write_text(transcript)
                log.info("Bulk transcribed: %s", meeting.name)
            except Exception:
                log.exception("Bulk transcription failed for %s", meeting.name)

        # Phase 2: Summarize meetings missing summaries (re-scan to pick up new transcripts)
        self._meetings = storage.list_meetings()
        need_summary = [m for m in self._meetings if m.has_transcript and not m.has_summary]

        if need_summary:
            templates_dir = _find_templates_dir()
            engine = TemplateEngine(templates_dir)
            template_name = "default"
            provider_name = config.summarization.default_provider
            endpoint = config.summarization.endpoints.get(provider_name, "")
            llm_model = config.summarization.default_model

            if not endpoint:
                self.app.call_from_thread(
                    self.notify, f"No endpoint for provider '{provider_name}'. Skipping summaries.", severity="error"
                )
            else:
                for i, meeting in enumerate(need_summary):
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
        self.app.call_from_thread(self.notify, "Bulk processing complete!")

    @on(DataTable.RowSelected, "#meeting-table")
    def on_meeting_selected(self, event: DataTable.RowSelected) -> None:
        meeting = self._get_selected_meeting()
        if meeting:
            from meetscribe.tui.screens.meeting import MeetingScreen
            self.app.push_screen(MeetingScreen(meeting))
