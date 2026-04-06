from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TemplateEngine:
    """Loads and renders Jinja2 meeting summary templates."""

    def __init__(self, templates_dir: Path) -> None:
        self.templates_dir = templates_dir
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            keep_trailing_newline=True,
        )

    def list_templates(self) -> list[str]:
        """Return sorted list of available template names (without .md extension)."""
        if not self.templates_dir.exists():
            return []
        return sorted(
            p.stem for p in self.templates_dir.glob("*.md")
        )

    def render(
        self,
        template_name: str,
        transcript: str,
        memos: str,
        meeting_name: str,
        date: str,
        duration: str,
    ) -> str:
        """Render a template with the given variables."""
        template = self._env.get_template(f"{template_name}.md")
        return template.render(
            transcript=transcript,
            memos=memos,
            meeting_name=meeting_name,
            date=date,
            duration=duration,
        )
