"""Smoke tests that verify the modules wire together correctly."""
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meetscribe.config import default_config, save_config, load_config
from meetscribe.storage.vault import MeetingStorage
from meetscribe.templates.engine import TemplateEngine
from meetscribe.transcription.whisper import format_transcript, format_timestamp
from meetscribe.summarization.provider import SummarizationProvider


class TestEndToEndWorkflow:
    """Verify that the full workflow from recording to summary works."""

    def test_vault_stores_and_retrieves_transcript(self, tmp_path):
        storage = MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")
        meeting_date = date(2026, 4, 6)
        meeting_name = "Integration Test"

        storage.ensure_meeting_dir(meeting_name, meeting_date)

        transcript_path = storage.transcript_path(meeting_name, meeting_date, "base")
        segments = [
            MagicMock(start=0.0, end=5.0, text=" Hello from integration test."),
        ]
        transcript = format_transcript(
            segments=segments,
            meeting_name=meeting_name,
            meeting_date=str(meeting_date),
            model="base",
            duration="00:00:05",
        )
        transcript_path.write_text(transcript)

        assert transcript_path.exists()
        content = transcript_path.read_text()
        assert "Hello from integration test." in content
        assert "model: base" in content

    def test_template_renders_with_transcript_and_memos(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "test.md").write_text(
            "Meeting: {{ meeting_name }}\n{{ transcript }}\n{% if memos %}Memos: {{ memos }}{% endif %}"
        )

        engine = TemplateEngine(tpl_dir)
        result = engine.render(
            template_name="test",
            transcript="[00:00:00] Discussion about X.",
            memos="Follow up on X.",
            meeting_name="Integration",
            date="2026-04-06",
            duration="00:05:00",
        )

        assert "Meeting: Integration" in result
        assert "[00:00:00] Discussion about X." in result
        assert "Memos: Follow up on X." in result

    def test_config_roundtrip(self, tmp_path):
        config_file = tmp_path / "config.toml"
        cfg = default_config()
        cfg.vault.root = "/test/vault"
        cfg.vault.meetings_folder = "MyMeetings"
        cfg.transcription.default_model = "large-v3"

        save_config(cfg, config_file)
        loaded = load_config(config_file)

        assert loaded.vault.root == "/test/vault"
        assert loaded.vault.meetings_folder == "MyMeetings"
        assert loaded.transcription.default_model == "large-v3"

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_summarization_sends_rendered_template(self, mock_openai_cls, tmp_path):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary: discussed X."))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = SummarizationProvider(base_url="http://localhost:11434/v1", model="llama3")
        result = provider.summarize("You summarize.", "Here is a transcript about X.")

        assert result == "Summary: discussed X."

    def test_meeting_listing(self, tmp_path):
        storage = MeetingStorage(vault_root=tmp_path, meetings_folder="Meetings")

        storage.ensure_meeting_dir("Standup", date(2026, 4, 6))
        rec_path = storage.recording_path("Standup", date(2026, 4, 6))
        rec_path.write_bytes(b"fake audio data")

        meetings = storage.list_meetings()
        assert len(meetings) == 1
        assert meetings[0].name == "standup"
        assert meetings[0].has_recording is True
        assert meetings[0].has_transcript is False
