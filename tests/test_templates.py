from pathlib import Path

import pytest

from meetscribe.templates.engine import TemplateEngine


@pytest.fixture
def templates_dir(tmp_path):
    default = tmp_path / "default.md"
    default.write_text(
        "Summarize:\n{{ transcript }}\n{% if memos %}Notes: {{ memos }}{% endif %}"
    )
    standup = tmp_path / "standup.md"
    standup.write_text("Standup for {{ meeting_name }} on {{ date }}:\n{{ transcript }}")
    return tmp_path


@pytest.fixture
def engine(templates_dir):
    return TemplateEngine(templates_dir)


class TestListTemplates:
    def test_returns_template_names(self, engine):
        names = engine.list_templates()
        assert "default" in names
        assert "standup" in names

    def test_empty_dir_returns_empty(self, tmp_path):
        engine = TemplateEngine(tmp_path)
        assert engine.list_templates() == []


class TestRender:
    def test_renders_with_transcript(self, engine):
        result = engine.render(
            template_name="default",
            transcript="Hello world.",
            memos="",
            meeting_name="Standup",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Summarize:" in result
        assert "Hello world." in result
        assert "Notes:" not in result  # memos empty, should be skipped

    def test_renders_with_memos(self, engine):
        result = engine.render(
            template_name="default",
            transcript="Hello.",
            memos="Remember to follow up.",
            meeting_name="Standup",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Notes: Remember to follow up." in result

    def test_renders_meeting_name_and_date(self, engine):
        result = engine.render(
            template_name="standup",
            transcript="Content.",
            memos="",
            meeting_name="Weekly",
            date="2026-04-06",
            duration="00:10:00",
        )
        assert "Standup for Weekly on 2026-04-06:" in result
