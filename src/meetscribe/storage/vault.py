from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class MeetingInfo:
    name: str
    date: date
    path: Path
    has_recording: bool = False
    has_transcript: bool = False
    has_summary: bool = False
    has_memos: bool = False
    duration: str = ""


METADATA_FILE = "_metadata.json"


def load_metadata(meeting_path: Path) -> dict[str, Any]:
    """Load meeting metadata from _metadata.json."""
    meta_path = meeting_path / METADATA_FILE
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def save_metadata(meeting_path: Path, data: dict[str, Any]) -> None:
    """Save meeting metadata to _metadata.json. Merges with existing data."""
    existing = load_metadata(meeting_path)
    existing.update(data)
    meta_path = meeting_path / METADATA_FILE
    meta_path.write_text(json.dumps(existing, indent=2))


def slugify(name: str) -> str:
    """Convert a meeting name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug


def _get_recording_duration(meeting_dir: Path, files: set[str]) -> str:
    """Get the duration of the recording file, if any."""
    for ext in ("flac", "mp3", "wav", "m4a", "ogg"):
        fname = f"recording.{ext}"
        if fname in files:
            try:
                import soundfile as sf
                info = sf.info(str(meeting_dir / fname))
                total = int(info.duration)
                h, remainder = divmod(total, 3600)
                m, s = divmod(remainder, 60)
                if h > 0:
                    return f"{h}h {m:02d}m"
                return f"{m}m {s:02d}s"
            except Exception:
                return ""
    return ""


class MeetingStorage:
    def __init__(self, vault_root: str | Path, meetings_folder: str = "Meetings") -> None:
        self.vault_root = Path(vault_root).expanduser()
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

    def find_recording(self, name: str, meeting_date: date) -> Path | None:
        """Find the recording file in any supported format."""
        meeting = self.meeting_dir(name, meeting_date)
        for ext in ("flac", "mp3", "wav", "m4a", "ogg"):
            path = meeting / f"recording.{ext}"
            if path.exists():
                return path
        return None

    def transcript_path(self, name: str, meeting_date: date, model: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"transcript-{model}.md"

    def summary_path(self, name: str, meeting_date: date, template_name: str) -> Path:
        return self.meeting_dir(name, meeting_date) / f"summary-{template_name}.md"

    def memos_path(self, name: str, meeting_date: date) -> Path:
        return self.meeting_dir(name, meeting_date) / "memos.md"

    def delete_meeting(self, meeting: MeetingInfo) -> None:
        """Delete a meeting directory and all its contents."""
        if meeting.path.exists():
            shutil.rmtree(meeting.path)

    def rename_meeting(self, meeting: MeetingInfo, new_name: str) -> MeetingInfo:
        """Rename a meeting by moving its directory. Returns updated MeetingInfo."""
        new_dir = meeting.path.parent / slugify(new_name)
        if new_dir.exists():
            raise ValueError(f"A meeting named '{new_name}' already exists on {meeting.date}")
        meeting.path.rename(new_dir)
        return MeetingInfo(
            name=slugify(new_name),
            date=meeting.date,
            path=new_dir,
            has_recording=meeting.has_recording,
            has_transcript=meeting.has_transcript,
            has_summary=meeting.has_summary,
            has_memos=meeting.has_memos,
        )

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
                        duration = _get_recording_duration(meeting_dir, files)
                        meetings.append(MeetingInfo(
                            name=meeting_dir.name,
                            date=meeting_date,
                            path=meeting_dir,
                            has_recording=any(f.startswith("recording.") for f in files),
                            has_transcript=any(f.startswith("transcript-") for f in files),
                            has_summary=any(f.startswith("summary-") for f in files),
                            has_memos="memos.md" in files,
                            duration=duration,
                        ))
        return meetings
