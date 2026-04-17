# Audio Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Click any transcript line to start audio playback from that timestamp, with stop controls.

**Architecture:** Transcript rendered as a `ListView` with `ListItem` per line (replacing the failed `ClickableRichLog`). Each timestamp line stores `timestamp_seconds` on the `ListItem`. `ListView.Selected` triggers `AudioPlayer.play(offset_seconds)`. Playback controls (stop + position) dock at the bottom and auto-hide.

**Tech Stack:** Textual (ListView, ListItem), sounddevice (OutputStream), soundfile (SoundFile seeking)

---

## Current State

Most of the implementation already exists from a prior commit (`7153544`). The code compiles and tests pass (86 tests). What's needed is verification that the ListView click → playback flow actually works end-to-end, and cleanup of the deleted `ClickableRichLog` file.

## File Structure

| File | Status | Responsibility |
|------|--------|---------------|
| `src/meetscribe/audio/player.py` | Done (commit `a4f7ac7`) | AudioPlayer with sounddevice + soundfile seeking |
| `src/meetscribe/tui/screens/meeting.py` | Done (commit `7153544`) | ListView transcript, playback handlers, controls |
| `src/meetscribe/tui/widgets/clickable_richlog.py` | Deleted (commit `7153544`) | No longer needed |
| `tests/test_player.py` | **Needed** | Unit tests for AudioPlayer |

---

### Task 1: AudioPlayer Unit Tests

**Files:**

- Create: `tests/test_player.py`

- [ ] **Step 1: Write tests for AudioPlayer**

```python
# tests/test_player.py
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from meetscribe.audio.player import AudioPlayer


class TestAudioPlayerInit:
    def test_initial_state(self):
        player = AudioPlayer(Path("/fake/audio.flac"))
        assert player.is_playing is False
        assert player.current_position == 0.0
        assert player.file_path == Path("/fake/audio.flac")


class TestAudioPlayerPlay:
    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_seeks_to_offset(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 16000
        mock_sf.channels = 1
        mock_sf_cls.return_value = mock_sf

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play(offset_seconds=30.0)

        # Should seek to frame 30 * 16000 = 480000
        mock_sf.seek.assert_called_once_with(480000)
        mock_stream.start.assert_called_once()
        assert player.is_playing is True

    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_from_zero(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 48000
        mock_sf.channels = 2
        mock_sf_cls.return_value = mock_sf
        mock_stream_cls.return_value = MagicMock()

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play()

        mock_sf.seek.assert_called_once_with(0)

    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_play_handles_bad_file(self, mock_sf_cls):
        mock_sf_cls.side_effect = RuntimeError("bad file")

        player = AudioPlayer(Path("/fake/bad.flac"))
        player.play()

        assert player.is_playing is False


class TestAudioPlayerStop:
    @patch("meetscribe.audio.player.sd.OutputStream")
    @patch("meetscribe.audio.player.sf.SoundFile")
    def test_stop_when_playing(self, mock_sf_cls, mock_stream_cls):
        mock_sf = MagicMock()
        mock_sf.samplerate = 16000
        mock_sf.channels = 1
        mock_sf_cls.return_value = mock_sf

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        player = AudioPlayer(Path("/fake/audio.flac"))
        player.play()
        player.stop()

        assert player.is_playing is False
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()
        mock_sf.close.assert_called()

    def test_stop_when_not_playing(self):
        player = AudioPlayer(Path("/fake/audio.flac"))
        player.stop()  # Should not raise
        assert player.is_playing is False
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/test_player.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `/Users/ian.bartholomew/.local/pipx/venvs/meetscribe/bin/python -m pytest tests/ -v`
Expected: All PASS (86 existing + new player tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_player.py
git commit -m "test: add unit tests for AudioPlayer"
```

---

### Task 2: Manual End-to-End Verification

**Files:** None (manual testing)

- [ ] **Step 1: Reinstall the app**

Run: `pipx install -e /Users/ian.bartholomew/Dev/meetscribe --python python3.13 --force`

- [ ] **Step 2: Test click-to-play**

Run: `meetscribe`

1. Open a meeting with a recording and transcript
2. Click on a transcript line with a timestamp
3. Verify: audio starts playing from that timestamp
4. Verify: playback controls (Stop button + position) appear at the bottom
5. Verify: position counter updates every ~250ms

- [ ] **Step 3: Test stop**

1. Click the Stop button
2. Verify: audio stops
3. Verify: playback controls disappear

- [ ] **Step 4: Test switching lines**

1. Click a transcript line to start playback
2. While playing, click a different line
3. Verify: audio switches to the new timestamp

- [ ] **Step 5: Test auto-hide on end**

1. Click a transcript line near the end of the recording
2. Wait for audio to finish
3. Verify: playback controls auto-hide

- [ ] **Step 6: Test screen exit cleanup**

1. Start playback
2. Press Escape to go back
3. Verify: audio stops (no orphaned playback)

- [ ] **Step 7: If issues found, fix and commit**

```bash
git add -A
git commit -m "fix: resolve audio playback integration issues"
```
