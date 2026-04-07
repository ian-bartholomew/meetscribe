from __future__ import annotations

from datetime import date

import logging

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Static, ListView, ListItem, Label, Input

from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage, MeetingInfo

log = logging.getLogger("meetscribe.home")


class MeetingListItem(ListItem):
    """A single meeting entry in the list."""

    def __init__(self, meeting: MeetingInfo) -> None:
        super().__init__()
        self.meeting = meeting

    def compose(self) -> ComposeResult:
        icons = ""
        if self.meeting.has_recording:
            icons += "[R]"
        if self.meeting.has_transcript:
            icons += "[T]"
        if self.meeting.has_summary:
            icons += "[S]"
        if self.meeting.has_memos:
            icons += "[M]"
        duration = f"  ({self.meeting.duration})" if self.meeting.duration else ""
        yield Label(
            f"{self.meeting.date}  {self.meeting.name}{duration}  {icons}"
        )


class HomeScreen(Screen):
    """Landing screen — start recording or browse past meetings."""

    BINDINGS = [
        ("n", "new_recording", "New Recording"),
        ("i", "import_hyprnote", "Import Hyprnote"),
        ("r", "rename_meeting", "Rename"),
        ("d", "delete_meeting", "Delete"),
        ("s", "app.push_screen('settings')", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Meetscribe", classes="title"),
            Horizontal(
                Button("New Recording", id="new-recording", variant="primary"),
                Button("Import from Hyprnote", id="import-hyprnote"),
            ),
            Static("Past Meetings"),
            ListView(id="meeting-list"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_meetings()

    def _refresh_meetings(self) -> None:
        config = self.app.config
        if not config.vault.root:
            return
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        meetings = storage.list_meetings()
        list_view = self.query_one("#meeting-list", ListView)
        list_view.clear()
        for meeting in meetings:
            list_view.append(MeetingListItem(meeting))

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
        list_view = self.query_one("#meeting-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, MeetingListItem):
            return list_view.highlighted_child.meeting
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

    @on(ListView.Selected, "#meeting-list")
    def on_meeting_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, MeetingListItem):
            from meetscribe.tui.screens.meeting import MeetingScreen
            self.app.push_screen(MeetingScreen(event.item.meeting))
