# Audio Playback from Transcript Timestamps Design

## Overview

Click any transcript line to start audio playback from that line's timestamp. Playback controls (stop button, position display) appear during playback and auto-hide when audio ends.

## Problem

There's no way to listen to specific parts of a meeting recording while reviewing the transcript. Users must switch to an external audio player and manually seek.

## Solution

### Transcript Viewer: ListView

Replace the `ClickableRichLog` (which doesn't receive click events reliably) with a `ListView`. Each transcript line becomes a `ListItem` containing a styled `Static` widget. Speaker headers are non-clickable ListItems. Timestamp lines store their seconds value on the ListItem.

`ListView.Selected` fires on click — this is battle-tested in Textual's own test suite, unlike RichLog's `on_click` which proved unreliable.

### AudioPlayer

Already implemented in `src/meetscribe/audio/player.py`. Uses `sounddevice.OutputStream` with callback + `soundfile.SoundFile` for seeking. `threading.Event` for clean stop from any thread.

### Playback Controls

A `Horizontal` bar at the bottom of the transcript tab:

- Stop button
- Position label (`HH:MM:SS`, refreshed every 250ms)

Hidden by default. Shows on playback start, hides on stop or end-of-file.

### Behavior

- Click transcript line → show controls, start playback from timestamp
- Click Stop → stop playback, hide controls  
- Audio ends → controls auto-hide
- Click different line while playing → stop current, start from new timestamp
- Leave screen (Escape) → stop any active playback

## Existing Code to Address

The previous implementation attempt (commit `a4f7ac7`) left code that needs updating:

- **`src/meetscribe/audio/player.py`** — Keep as-is, AudioPlayer works correctly
- **`src/meetscribe/tui/widgets/clickable_richlog.py`** — Delete, replaced by ListView approach
- **`src/meetscribe/tui/screens/meeting.py`** — Already has playback handlers, controls CSS, and cleanup. Update to use `ListView.Selected` instead of `ClickableRichLog.LineClicked`, and replace transcript rendering

## Files to Modify

| File | Change |
|------|--------|
| `src/meetscribe/tui/screens/meeting.py` | Replace ClickableRichLog with ListView, update transcript rendering, update playback handlers |
| `src/meetscribe/tui/widgets/clickable_richlog.py` | Delete — no longer needed |
| `src/meetscribe/audio/player.py` | No changes — already works |

## Key Changes in meeting.py

1. Replace `ClickableRichLog` import with `ListView`, `ListItem` from textual.widgets
2. Replace `_write_transcript_to_richlog` with `_build_transcript_items` that returns `list[ListItem]`, and `_populate_transcript_view` that clears and mounts them
3. Swap `ClickableRichLog(...)` for `ListView(id="transcript-view")` in compose
4. Change `handle_line_clicked` to handle `ListView.Selected` — read `timestamp_seconds` attribute from `event.item`
5. Update edit toggle to swap ListView/TextArea instead of RichLog/TextArea
6. Update live transcription streaming to append ListItems
7. Playback controls, stop handler, position refresh worker, cleanup on exit — already partially implemented, update to use ListView
8. Each `ListItem` stores `timestamp_seconds` as an attribute for timestamp lines. Speaker headers have no timestamp attribute (clicks on them are ignored).

## Testing

- Unit tests for AudioPlayer (mock sounddevice/soundfile)
- Manual: open meeting, click transcript line, verify playback starts at correct position
- Manual: click Stop, verify audio stops and controls hide
- Manual: click different line while playing, verify switch
- Manual: let audio play to end, verify controls auto-hide
