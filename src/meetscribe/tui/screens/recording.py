from __future__ import annotations

import time
from datetime import date

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Static, Input, Label

from meetscribe.audio.recorder import AudioRecorder
from meetscribe.config import load_config
from meetscribe.storage.vault import MeetingStorage


class RecordingScreen(Screen):
    """Screen for actively recording a meeting."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._recorder: AudioRecorder | None = None
        self._start_time: float = 0.0
        self._meeting_name: str = ""
        self._recording_active = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("New Recording", classes="title"),
            Input(placeholder="Meeting name...", id="meeting-name"),
            Button("Start Recording", id="start-btn", variant="primary"),
            Label("", id="timer"),
            Label("", id="level"),
            Button("Stop Recording", id="stop-btn", variant="error", disabled=True),
        )
        yield Footer()

    @on(Button.Pressed, "#start-btn")
    def start_recording(self) -> None:
        name_input = self.query_one("#meeting-name", Input)
        self._meeting_name = name_input.value.strip()
        if not self._meeting_name:
            self.notify("Please enter a meeting name.", severity="error")
            return

        config = self.app.config
        if not config.vault.root:
            self.notify("Please configure vault root in settings first.", severity="error")
            return

        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        output_path = storage.recording_path(self._meeting_name, date.today())
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._recorder = AudioRecorder(
            output_path=output_path,
            device_name=config.audio.device_name,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            mic_device_name=config.audio.mic_device_name or None,
        )

        try:
            self._recorder.start()
        except ValueError as e:
            self.notify(str(e), severity="error")
            return

        self._start_time = time.monotonic()
        self._recording_active = True

        name_input.disabled = True
        self.query_one("#start-btn", Button).disabled = True
        self.query_one("#stop-btn", Button).disabled = False

        self._refresh_recording_display()

    @work(exclusive=True)
    async def _refresh_recording_display(self) -> None:
        """Periodically update the timer and level display."""
        import asyncio
        while self._recording_active:
            elapsed = time.monotonic() - self._start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.query_one("#timer", Label).update(f"Recording: {h:02d}:{m:02d}:{s:02d}")

            if self._recorder:
                level = self._recorder.peak_level
                bar_len = int(level * 40)
                bar = "|" * bar_len + "." * (40 - bar_len)
                self.query_one("#level", Label).update(f"Level: [{bar}]")

            await asyncio.sleep(0.25)

    @on(Button.Pressed, "#stop-btn")
    def stop_recording(self) -> None:
        self._recording_active = False
        if self._recorder:
            self._recorder.stop()

        from meetscribe.storage.vault import MeetingStorage, MeetingInfo
        config = self.app.config
        storage = MeetingStorage(config.vault.root, config.vault.meetings_folder)
        meeting_dir = storage.meeting_dir(self._meeting_name, date.today())

        meeting = MeetingInfo(
            name=self._meeting_name,
            date=date.today(),
            path=meeting_dir,
            has_recording=True,
        )

        from meetscribe.tui.screens.meeting import MeetingScreen
        self.app.switch_screen(MeetingScreen(meeting))

    def action_cancel(self) -> None:
        self._recording_active = False
        if self._recorder and self._recorder.is_recording:
            self._recorder.stop()
        self.app.pop_screen()
