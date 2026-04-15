import pytest

from meetscribe.storage.speakers import rewrite_transcript


class TestRewriteTranscript:
    def test_replaces_speaker_labels(self):
        transcript = (
            "---\nmeeting: Standup\n---\n\n"
            "**Speaker 1:**\n[00:00:05] Hello\n\n"
            "**Speaker 2:**\n[00:00:10] Hi there\n\n"
        )
        speaker_map = {
            "Speaker 1": "Alice",
            "Speaker 2": "Bob",
        }
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice:**" in result
        assert "**Bob:**" in result
        assert "**Speaker 1:**" not in result
        assert "**Speaker 2:**" not in result

    def test_partial_map(self):
        transcript = (
            "**Speaker 1:**\n[00:00:05] Hello\n\n"
            "**Speaker 2:**\n[00:00:10] Hi\n\n"
        )
        speaker_map = {"Speaker 1": "Alice"}
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice:**" in result
        assert "**Speaker 2:**" in result

    def test_remap_already_named(self):
        """When re-mapping, find the current name and replace it."""
        transcript = "**Alice:**\n[00:00:05] Hello\n\n"
        speaker_map = {"Alice": "Alice Smith"}
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice Smith:**" in result
        assert "**Alice:**" not in result

    def test_preserves_frontmatter(self):
        transcript = "---\nmeeting: Standup\ndate: 2026-04-15\n---\n\n**Speaker 1:**\n[00:00:05] Hello\n"
        speaker_map = {"Speaker 1": "Alice"}
        result = rewrite_transcript(transcript, speaker_map)
        assert result.startswith("---\nmeeting: Standup")
        assert "**Alice:**" in result

    def test_empty_map_returns_unchanged(self):
        transcript = "**Speaker 1:**\n[00:00:05] Hello\n"
        result = rewrite_transcript(transcript, {})
        assert result == transcript

    def test_does_not_replace_name_in_body_text(self):
        """Speaker name in transcript body should not be modified."""
        transcript = (
            "**Speaker 1:**\n"
            "[00:00:05] Speaker 1 said something about the project\n\n"
            "**Speaker 2:**\n"
            "[00:00:10] I agree with Speaker 1\n\n"
        )
        speaker_map = {"Speaker 1": "Alice", "Speaker 2": "Bob"}
        result = rewrite_transcript(transcript, speaker_map)
        assert "**Alice:**" in result
        assert "**Bob:**" in result
        # Body text references should be unchanged
        assert "Speaker 1 said something" in result
        assert "I agree with Speaker 1" in result
