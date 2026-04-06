from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Static

from meetscribe.config import save_config


class SetupScreen(Screen):
    """First-run setup — collect minimum required configuration."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Welcome to Meetscribe!", classes="title"),
            Static("Let's configure the basics to get started."),
            Label("Obsidian Vault Root Path:"),
            Input(placeholder="/path/to/your/vault", id="vault-root"),
            Label("Meetings Folder (relative to vault root):"),
            Input(value="Meetings", id="meetings-folder"),
            Button("Save & Start", id="save-btn", variant="primary"),
        )
        yield Footer()

    @on(Button.Pressed, "#save-btn")
    def save_and_start(self) -> None:
        vault_root = self.query_one("#vault-root", Input).value.strip()
        if not vault_root:
            self.notify("Please enter a vault root path.", severity="error")
            return

        config = self.app.config
        config.vault.root = vault_root
        config.vault.meetings_folder = self.query_one("#meetings-folder", Input).value.strip()
        save_config(config)

        self.notify("Configuration saved!")
        from meetscribe.tui.screens.home import HomeScreen
        self.app.switch_screen(HomeScreen())
