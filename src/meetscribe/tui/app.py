import logging
from pathlib import Path

from textual.app import App

from meetscribe.config import load_config, MeetscribeConfig, CONFIG_DIR

LOG_FILE = CONFIG_DIR / "meetscribe.log"


def setup_logging(level: str = "INFO") -> None:
    """Configure file logging for the app.

    The meetscribe logger and root logger use the configured level.
    Third-party loggers (markdown_it, httpcore, httpx, etc.) stay at
    WARNING to avoid noise, unless the configured level is DEBUG.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Keep third-party loggers quiet unless explicitly set to DEBUG
    noisy_loggers = [
        "markdown_it", "httpcore", "httpx", "asyncio",
        "urllib3", "speechbrain", "torch",
    ]
    third_party_level = logging.DEBUG if log_level == logging.DEBUG else logging.WARNING
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(third_party_level)


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
        setup_logging(self.config.log_level)
        self.log_file = LOG_FILE
        logging.getLogger("meetscribe").info("App started")

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
