"""RichLog subclass that emits a message when a line with a timestamp is clicked."""
from __future__ import annotations

from typing import Self

from textual.events import Click
from textual.message import Message
from textual.widgets import RichLog


class ClickableRichLog(RichLog):
    """A RichLog that tracks line-to-timestamp mappings and emits LineClicked on click."""

    class LineClicked(Message):
        """Posted when a line associated with a timestamp is clicked."""

        def __init__(self, timestamp_seconds: float) -> None:
            self.timestamp_seconds = timestamp_seconds
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._line_timestamps: dict[int, float] = {}

    def write_with_timestamp(self, content, timestamp_seconds: float, **kwargs) -> Self:
        """Write content and associate the resulting line(s) with a timestamp."""
        before = len(self.lines)
        result = self.write(content, **kwargs)
        after = len(self.lines)
        for i in range(before, after):
            self._line_timestamps[i] = timestamp_seconds
        return result

    def clear(self) -> Self:
        """Clear content and timestamp mappings."""
        self._line_timestamps.clear()
        return super().clear()

    def on_click(self, event: Click) -> None:
        """Map click position to a line index and post LineClicked if it has a timestamp."""
        line_idx = event.y + self.scroll_offset.y
        if line_idx in self._line_timestamps:
            self.post_message(self.LineClicked(self._line_timestamps[line_idx]))
