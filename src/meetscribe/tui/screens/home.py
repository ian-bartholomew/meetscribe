from __future__ import annotations

from datetime import date

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Static, ListView, ListItem, Label, Input

from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage, MeetingInfo


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
        yield Label(
            f"{self.meeting.date}  {self.meeting.name}  {icons}"
        )


class HomeScreen(Screen):
    """Landing screen — start recording or browse past meetings."""

    BINDINGS = [
        ("n", "new_recording", "New Recording"),
        ("s", "app.push_screen('settings')", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Meetscribe", classes="title"),
            Button("New Recording", id="new-recording", variant="primary"),
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

    @on(ListView.Selected, "#meeting-list")
    def on_meeting_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, MeetingListItem):
            from meetscribe.tui.screens.meeting import MeetingScreen
            self.app.push_screen(MeetingScreen(event.item.meeting))
