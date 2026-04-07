"""Simple modal dialog screens for confirmation and text input."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class ConfirmDialog(ModalScreen[bool]):
    """Modal confirmation dialog. Returns True if confirmed, False otherwise."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    ConfirmDialog {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        max-height: 12;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #confirm-buttons {
        height: auto;
        margin-top: 1;
    }
    #confirm-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self._message)
            with Horizontal(id="confirm-buttons"):
                yield Button("Delete", id="confirm-yes", variant="error")
                yield Button("Cancel", id="confirm-no")

    @on(Button.Pressed, "#confirm-yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class RenameDialog(ModalScreen[str | None]):
    """Modal rename dialog. Returns the new name or None if cancelled."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    RenameDialog {
        align: center middle;
    }
    #rename-dialog {
        width: 60;
        height: auto;
        max-height: 12;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #rename-buttons {
        height: auto;
        margin-top: 1;
    }
    #rename-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, current_name: str) -> None:
        super().__init__()
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-dialog"):
            yield Label("Rename meeting:")
            yield Input(value=self._current_name, id="rename-input")
            with Horizontal(id="rename-buttons"):
                yield Button("Rename", id="rename-ok", variant="primary")
                yield Button("Cancel", id="rename-cancel")

    @on(Button.Pressed, "#rename-ok")
    def on_ok(self) -> None:
        value = self.query_one("#rename-input", Input).value.strip()
        self.dismiss(value if value else None)

    @on(Button.Pressed, "#rename-cancel")
    def on_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#rename-input")
    def on_submit(self) -> None:
        self.on_ok()

    def action_cancel(self) -> None:
        self.dismiss(None)
