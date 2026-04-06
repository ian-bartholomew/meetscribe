from datetime import date
from pathlib import Path

import pytest

from meetscribe.storage.vault import MeetingStorage


@pytest.fixture
def vault(tmp_path):
    return MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")


class TestMeetingPath:
    def test_creates_correct_path(self, vault, tmp_path):
        path = vault.meeting_dir("Weekly Standup", date(2026, 4, 6))
        assert path == tmp_path / "Meetings" / "2026" / "04" / "06" / "weekly-standup"

    def test_slugifies_name(self, vault, tmp_path):
        path = vault.meeting_dir("Q2 Planning — Session #1", date(2026, 4, 6))
        assert path == tmp_path / "Meetings" / "2026" / "04" / "06" / "q2-planning-session-1"


class TestEnsureMeetingDir:
    def test_creates_directories(self, vault):
        path = vault.ensure_meeting_dir("Standup", date(2026, 4, 6))
        assert path.exists()
        assert path.is_dir()


class TestRecordingPath:
    def test_returns_flac_path(self, vault):
        path = vault.recording_path("Standup", date(2026, 4, 6))
        assert path.name == "recording.flac"


class TestTranscriptPath:
    def test_includes_model_name(self, vault):
        path = vault.transcript_path("Standup", date(2026, 4, 6), "large-v3")
        assert path.name == "transcript-large-v3.md"


class TestSummaryPath:
    def test_includes_template_name(self, vault):
        path = vault.summary_path("Standup", date(2026, 4, 6), "standup")
        assert path.name == "summary-standup.md"


class TestMemosPath:
    def test_returns_memos_path(self, vault):
        path = vault.memos_path("Standup", date(2026, 4, 6))
        assert path.name == "memos.md"


class TestListMeetings:
    def test_lists_meetings_by_date(self, vault):
        vault.ensure_meeting_dir("Standup", date(2026, 4, 6))
        vault.ensure_meeting_dir("Retro", date(2026, 4, 6))
        vault.ensure_meeting_dir("Planning", date(2026, 3, 15))

        meetings = vault.list_meetings()
        assert len(meetings) == 3
        # Most recent first
        assert meetings[0].date == date(2026, 4, 6)
        assert meetings[-1].date == date(2026, 3, 15)
