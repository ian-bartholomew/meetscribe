from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class MeetingInfo:
    name: str
    date: date
    path: Path
    has_recording: bool = False
    has_transcript: bool = False
    has_summary: bool = False
    has_memos: bool = False


def slugify(name: str) -> str:
    """Convert a meeting name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug


class MeetingStorage:
    def __init__(self, vault_root: str | Path, meetings_folder: str = "Meetings") -> None:
        self.vault_root = Path(vault_root)
        self.meetings_folder = meetings_folder

    @property
    def meetings_root(self) -> Path:
        return self.vault_root / self.meetings_folder

    def meeting_dir(self, name: str, meeting_date: date) -> Path:
        return (
            self.meetings_root
            / f"{meeting_date.year}"
            / f"{meeting_date.month:02d}"
            / f"{meeting_date.day:02d}"
            / slugify(name)
        )

    def ensure_meeting_dir(self, name: str, meeting_date: date) -> Path:
        path = self.meeting_dir(name, meeting_date)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def recording_path(self, name: str, meeting_date: date) -> Path:
        return self.meeting_dir(name, meeting_date) / "recording.flac"

    def transcript_path(self, name: str, meeting_date: date, model: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"transcript-{model}.md"

    def summary_path(self, name: str, meeting_date: date, template_name: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"summary-{template_name}.md"

    def memos_path(self, name: str, meeting_date: date) -> Path:
        return self.meeting_dir(name, meeting_date) / "memos.md"

    def list_meetings(self) -> list[MeetingInfo]:
        """Scan the meetings folder and return all meetings, newest first."""
        meetings: list[MeetingInfo] = []
        root = self.meetings_root
        if not root.exists():
            return meetings

        for year_dir in sorted(root.iterdir(), reverse=True):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                    for meeting_dir in sorted(day_dir.iterdir(), reverse=True):
                        if not meeting_dir.is_dir():
                            continue
                        meeting_date = date(
                            int(year_dir.name),
                            int(month_dir.name),
                            int(day_dir.name),
                        )
                        files = {f.name for f in meeting_dir.iterdir()}
                        meetings.append(MeetingInfo(
                            name=meeting_dir.name,
                            date=meeting_date,
                            path=meeting_dir,
                            has_recording="recording.flac" in files,
                            has_transcript=any(f.startswith("transcript-") for f in files),
                            has_summary=any(f.startswith("summary-") for f in files),
                            has_memos="memos.md" in files,
                        ))
        return meetings
