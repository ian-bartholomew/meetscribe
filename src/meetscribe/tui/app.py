from textual.app import App

from meetscribe.config import load_config, MeetscribeConfig


class MeetscribeApp(App):
    """Meetscribe TUI application."""

    TITLE = "Meetscribe"
    CSS = """
    Screen {
        align: center middle;
    }

    #meeting-list {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }

    .title {
        text-style: bold;
        color: $text;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "open_settings", "Settings"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()

    def on_mount(self) -> None:
        if not self.config.vault.root:
            from meetscribe.tui.screens.setup import SetupScreen
            self.push_screen(SetupScreen())
        else:
            from meetscribe.tui.screens.home import HomeScreen
            self.push_screen(HomeScreen())

    def action_open_settings(self) -> None:
        from meetscribe.tui.screens.settings import SettingsScreen
        self.push_screen(SettingsScreen())
