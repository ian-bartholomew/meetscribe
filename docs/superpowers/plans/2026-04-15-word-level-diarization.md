# Word-Level Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve speaker attribution accuracy by assigning individual words to speakers using faster-whisper's word-level timestamps.

**Architecture:** Enable `word_timestamps=True` when diarizing. Add `assign_speakers_to_words` in diarize.py that assigns each word to a speaker and groups consecutive same-speaker words into lines. Update `format_transcript` to handle the new 3-tuple format with embedded timestamps.

**Tech Stack:** Python, faster-whisper (word_timestamps), existing SpeechBrain diarization

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/meetscribe/transcription/diarize.py` (modify) | Add `assign_speakers_to_words` function |
| `src/meetscribe/transcription/whisper.py` (modify) | Enable word timestamps, collect words, update `format_transcript` |
| `tests/test_diarize_embeddings.py` (modify) | Tests for `assign_speakers_to_words` |
| `tests/test_whisper.py` (modify) | Tests for word timestamp flow and updated `format_transcript` |

---

### Task 1: Word-Level Speaker Assignment Function

**Files:**

- Modify: `src/meetscribe/transcription/diarize.py:187-221`
- Test: `tests/test_diarize_embeddings.py`

- [ ] **Step 1: Write failing tests for `assign_speakers_to_words`**

Add to `tests/test_diarize_embeddings.py`:

```python
from meetscribe.transcription.diarize import (
    _cluster_speakers,
    DiarizationResult,
    SpeakerSegment,
    assign_speakers_to_words,
)


def _mock_word(start, end, word):
    """Create a mock faster-whisper Word object."""
    from unittest.mock import MagicMock
    w = MagicMock()
    w.start = start
    w.end = end
    w.word = word
    return w


class TestAssignSpeakersToWords:
    def test_single_speaker(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(0.5, 1.0, " world"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 5.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 1
        assert result[0] == ("Speaker 1", 0.0, "hello world")

    def test_speaker_change_mid_segment(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(0.5, 1.0, " from"),
            _mock_word(1.0, 1.5, " me"),
            _mock_word(2.0, 2.5, " hi"),
            _mock_word(2.5, 3.0, " back"),
        ]
        speaker_segments = [
            SpeakerSegment(0.0, 1.5, "Speaker 1"),
            SpeakerSegment(1.5, 5.0, "Speaker 2"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 2
        assert result[0] == ("Speaker 1", 0.0, "hello from me")
        assert result[1] == ("Speaker 2", 2.0, "hi back")

    def test_alternating_speakers(self):
        words = [
            _mock_word(0.0, 0.5, " yes"),
            _mock_word(1.0, 1.5, " no"),
            _mock_word(2.0, 2.5, " maybe"),
        ]
        speaker_segments = [
            SpeakerSegment(0.0, 0.8, "Speaker 1"),
            SpeakerSegment(0.8, 1.8, "Speaker 2"),
            SpeakerSegment(1.8, 3.0, "Speaker 1"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 3
        assert result[0] == ("Speaker 1", 0.0, "yes")
        assert result[1] == ("Speaker 2", 1.0, "no")
        assert result[2] == ("Speaker 1", 2.0, "maybe")

    def test_consecutive_same_speaker_grouped(self):
        """Words from same speaker across multiple speaker segments get grouped."""
        words = [
            _mock_word(0.0, 0.5, " one"),
            _mock_word(0.5, 1.0, " two"),
            _mock_word(1.0, 1.5, " three"),
            _mock_word(1.5, 2.0, " four"),
        ]
        # Two adjacent speaker segments for same speaker
        speaker_segments = [
            SpeakerSegment(0.0, 1.0, "Speaker 1"),
            SpeakerSegment(1.0, 3.0, "Speaker 1"),
        ]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 1
        assert result[0] == ("Speaker 1", 0.0, "one two three four")

    def test_word_with_no_overlap(self):
        words = [
            _mock_word(0.0, 0.5, " hello"),
            _mock_word(10.0, 10.5, " orphan"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 1.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert len(result) == 2
        assert result[0] == ("Speaker 1", 0.0, "hello")
        assert result[1] == ("Unknown", 10.0, "orphan")

    def test_empty_words(self):
        result = assign_speakers_to_words([], [SpeakerSegment(0.0, 5.0, "Speaker 1")])
        assert result == []

    def test_timestamp_is_first_word(self):
        """Each group's timestamp should be the start time of its first word."""
        words = [
            _mock_word(5.0, 5.5, " late"),
            _mock_word(5.5, 6.0, " start"),
        ]
        speaker_segments = [SpeakerSegment(0.0, 10.0, "Speaker 1")]
        result = assign_speakers_to_words(words, speaker_segments)
        assert result[0][1] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_diarize_embeddings.py::TestAssignSpeakersToWords -v`
Expected: FAIL — `ImportError: cannot import name 'assign_speakers_to_words'`

- [ ] **Step 3: Implement `assign_speakers_to_words`**

Add to `src/meetscribe/transcription/diarize.py` after `assign_speakers_to_transcript`:

```python
def assign_speakers_to_words(
    words: list,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, float, str]]:
    """Assign speakers to individual words and group consecutive same-speaker words.

    Args:
        words: faster-whisper Word objects with .start, .end, .word
        speaker_segments: diarization results

    Returns:
        List of (speaker_label, start_time, text) tuples. Each tuple is a group
        of consecutive words from the same speaker.
    """
    if not words:
        return []

    # Assign each word to a speaker
    word_speakers: list[tuple[str, float, str]] = []
    for w in words:
        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(w.start, sseg.start)
            overlap_end = min(w.end, sseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        word_speakers.append((best_speaker, w.start, w.word.strip()))

    # Group consecutive same-speaker words into lines
    groups: list[tuple[str, float, str]] = []
    current_speaker = word_speakers[0][0]
    current_start = word_speakers[0][1]
    current_words: list[str] = [word_speakers[0][2]]

    for speaker, start, word in word_speakers[1:]:
        if speaker == current_speaker:
            current_words.append(word)
        else:
            groups.append((current_speaker, current_start, " ".join(current_words)))
            current_speaker = speaker
            current_start = start
            current_words = [word]

    groups.append((current_speaker, current_start, " ".join(current_words)))
    return groups
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_diarize_embeddings.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/transcription/diarize.py tests/test_diarize_embeddings.py
git commit -m "feat: add word-level speaker assignment function"
```

---

### Task 2: Update `format_transcript` for 3-Tuple Speaker Labels

**Files:**

- Modify: `src/meetscribe/transcription/whisper.py:52-91`
- Test: `tests/test_whisper.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_whisper.py`:

```python
class TestFormatTranscriptWordLevel:
    def test_formats_with_word_level_speaker_labels(self):
        """3-tuple speaker_labels (speaker, timestamp, text) render correctly."""
        segments = []  # Not used when word-level labels provided
        speaker_labels = [
            ("Speaker 1", 0.0, "hello from me"),
            ("Speaker 2", 2.0, "hi back"),
            ("Speaker 1", 4.0, "great"),
        ]
        result = format_transcript(
            segments=segments,
            meeting_name="Standup",
            meeting_date="2026-04-06",
            model="base",
            duration="00:10:00",
            speaker_labels=speaker_labels,
        )
        assert "**Speaker 1:**" in result
        assert "[00:00:00] hello from me" in result
        assert "**Speaker 2:**" in result
        assert "[00:00:02] hi back" in result
        assert "[00:00:04] great" in result

    def test_speaker_grouping_in_output(self):
        """Consecutive lines from same speaker don't repeat the header."""
        segments = []
        speaker_labels = [
            ("Speaker 1", 0.0, "first line"),
            ("Speaker 1", 1.0, "second line"),
            ("Speaker 2", 3.0, "other speaker"),
        ]
        result = format_transcript(
            segments=segments,
            meeting_name="Test",
            meeting_date="2026-04-06",
            model="base",
            duration="00:05:00",
            speaker_labels=speaker_labels,
        )
        # Speaker 1 header should appear only once
        assert result.count("**Speaker 1:**") == 1
        assert "[00:00:00] first line" in result
        assert "[00:00:01] second line" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_whisper.py::TestFormatTranscriptWordLevel -v`
Expected: FAIL — the current code tries to index `segments[i]` which is empty

- [ ] **Step 3: Update `format_transcript`**

In `src/meetscribe/transcription/whisper.py`, replace the `format_transcript` function:

```python
def format_transcript(
    segments: list,
    meeting_name: str,
    meeting_date: str,
    model: str,
    duration: str,
    speaker_labels: list | None = None,
) -> str:
    """Format transcription segments into a markdown document with frontmatter.

    speaker_labels can be:
      - list[tuple[str, str]]: (speaker, text) — segment-level, uses segments for timestamps
      - list[tuple[str, float, str]]: (speaker, start_time, text) — word-level, timestamps embedded
      - None: no diarization, uses segments directly
    """
    lines = [
        "---",
        f"meeting: {meeting_name}",
        f"date: {meeting_date}",
        f"model: {model}",
        f'duration: "{duration}"',
        "---",
        "",
    ]

    if speaker_labels:
        prev_speaker = None
        for i, entry in enumerate(speaker_labels):
            if len(entry) == 3:
                speaker, start_time, text = entry
                timestamp = format_timestamp(start_time)
            else:
                speaker, text = entry
                timestamp = format_timestamp(segments[i].start)
            if speaker != prev_speaker:
                lines.append(f"**{speaker}:**")
                prev_speaker = speaker
            lines.append(f"[{timestamp}] {text}")
            lines.append("")
    else:
        for segment in segments:
            timestamp = format_timestamp(segment.start)
            text = segment.text.strip()
            lines.append(f"[{timestamp}] {text}")
            lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_whisper.py -v`
Expected: All PASS (new tests + existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/meetscribe/transcription/whisper.py tests/test_whisper.py
git commit -m "feat: update format_transcript for word-level speaker labels"
```

---

### Task 3: Enable Word Timestamps and Wire Up

**Files:**

- Modify: `src/meetscribe/transcription/whisper.py:112-182`
- Test: `tests/test_whisper.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_whisper.py`:

```python
class TestTranscribeAudioWordLevel:
    @patch("meetscribe.transcription.whisper._load_model")
    @patch("meetscribe.transcription.diarize.diarize")
    @patch("meetscribe.transcription.diarize.assign_speakers_to_words")
    def test_uses_word_level_assignment(self, mock_assign_words, mock_diarize, mock_load):
        """When diarization is enabled, word_timestamps=True and assign_speakers_to_words is used."""
        from meetscribe.transcription.diarize import DiarizationResult, SpeakerSegment

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        word1 = MagicMock(start=0.0, end=0.5, word=" Hello")
        word2 = MagicMock(start=0.5, end=1.0, word=" world")
        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello world", words=[word1, word2])
        mock_info = MagicMock(duration=5.0)
        mock_model.transcribe.return_value = ([seg1], mock_info)

        import numpy as np
        cluster_embs = {"Speaker 1": np.array([0.1] * 192)}
        mock_diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(0.0, 5.0, "Speaker 1")],
            cluster_embeddings=cluster_embs,
        )
        mock_assign_words.return_value = [("Speaker 1", 0.0, "Hello world")]

        result, embeddings = transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
            enable_diarization=True,
        )

        # Verify word_timestamps was enabled
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("word_timestamps") is True

        # Verify word-level assignment was used
        mock_assign_words.assert_called_once()
        assert "Speaker 1" in result
        assert "[00:00:00] Hello world" in result

    @patch("meetscribe.transcription.whisper._load_model")
    def test_no_word_timestamps_without_diarization(self, mock_load):
        """word_timestamps should not be enabled when diarization is off."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.")
        mock_info = MagicMock(duration=5.0)
        mock_model.transcribe.return_value = ([seg1], mock_info)

        transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
            enable_diarization=False,
        )

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("word_timestamps") is not True

    @patch("meetscribe.transcription.whisper._load_model")
    @patch("meetscribe.transcription.diarize.diarize")
    @patch("meetscribe.transcription.diarize.assign_speakers_to_transcript")
    def test_falls_back_without_words(self, mock_assign_seg, mock_diarize, mock_load):
        """Falls back to segment-level assignment when words are missing."""
        from meetscribe.transcription.diarize import DiarizationResult, SpeakerSegment

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Segment with no words attribute (words=None)
        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.", words=None)
        mock_info = MagicMock(duration=5.0)
        mock_model.transcribe.return_value = ([seg1], mock_info)

        import numpy as np
        cluster_embs = {"Speaker 1": np.array([0.1] * 192)}
        mock_diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(0.0, 5.0, "Speaker 1")],
            cluster_embeddings=cluster_embs,
        )
        mock_assign_seg.return_value = [("Speaker 1", "Hello.")]

        transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
            enable_diarization=True,
        )

        # Should fall back to segment-level
        mock_assign_seg.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_whisper.py::TestTranscribeAudioWordLevel -v`
Expected: FAIL

- [ ] **Step 3: Update `transcribe_audio`**

In `src/meetscribe/transcription/whisper.py`, modify the `transcribe_audio` function:

Change the `model.transcribe` call (around line 139):

```python
    segments, info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True, language="en",
        initial_prompt=initial_prompt,
        word_timestamps=enable_diarization,
    )
```

Replace the diarization block (around lines 162-172):

```python
    speaker_labels = None
    cluster_embeddings = None
    if enable_diarization:
        log.info("Running speaker diarization...")
        from meetscribe.transcription.diarize import (
            diarize, assign_speakers_to_transcript, assign_speakers_to_words,
        )
        diarization_result = diarize(audio_path, num_speakers=num_speakers)

        # Collect word-level timestamps if available
        all_words = []
        has_words = True
        for seg in segment_list:
            if seg.words:
                all_words.extend(seg.words)
            else:
                has_words = False
                break

        if has_words and all_words:
            speaker_labels = assign_speakers_to_words(all_words, diarization_result.segments)
            log.info("Word-level diarization: %d word groups", len(speaker_labels))
        else:
            speaker_labels = assign_speakers_to_transcript(segment_list, diarization_result.segments)
            log.info("Segment-level diarization fallback: %d segments", len(speaker_labels))

        cluster_embeddings = {
            label: emb.tolist() for label, emb in diarization_result.cluster_embeddings.items()
        }
        log.info("Diarization complete")
```

- [ ] **Step 4: Update existing diarization test**

The existing `TestTranscribeAudioDiarization.test_returns_cluster_embeddings` patches `assign_speakers_to_transcript` but the new code will try `assign_speakers_to_words` first. Update the test to provide segments with words so the word-level path is used:

In `tests/test_whisper.py`, replace `TestTranscribeAudioDiarization`:

```python
class TestTranscribeAudioDiarization:
    @patch("meetscribe.transcription.whisper._load_model")
    @patch("meetscribe.transcription.diarize.diarize")
    @patch("meetscribe.transcription.diarize.assign_speakers_to_words")
    def test_returns_cluster_embeddings(self, mock_assign_words, mock_diarize, mock_load):
        from meetscribe.transcription.diarize import DiarizationResult, SpeakerSegment

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        word1 = MagicMock(start=0.0, end=0.5, word=" Hello.")
        seg1 = MagicMock(start=0.0, end=5.0, text=" Hello.", words=[word1])
        mock_info = MagicMock(duration=5.0)
        mock_model.transcribe.return_value = ([seg1], mock_info)

        import numpy as np
        cluster_embs = {"Speaker 1": np.array([0.1] * 192)}
        mock_diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(0.0, 5.0, "Speaker 1")],
            cluster_embeddings=cluster_embs,
        )
        mock_assign_words.return_value = [("Speaker 1", 0.0, "Hello.")]

        result = transcribe_audio(
            audio_path=Path("/fake/recording.flac"),
            model_name="base",
            meeting_name="Standup",
            meeting_date="2026-04-06",
            enable_diarization=True,
        )

        assert isinstance(result, tuple)
        transcript_text, embeddings = result
        assert "Speaker 1" in transcript_text
        assert "Speaker 1" in embeddings
```

- [ ] **Step 5: Run all tests**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/meetscribe/transcription/whisper.py tests/test_whisper.py
git commit -m "feat: enable word-level timestamps for diarization with fallback"
```
