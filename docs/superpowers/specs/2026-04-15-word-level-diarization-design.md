# Word-Level Speaker Diarization Design

## Overview

Improve speaker attribution accuracy by using faster-whisper's word-level timestamps to assign individual words to speakers, rather than assigning entire segments. This eliminates the common problem of speaker bleed at segment boundaries.

## Problem

The current `assign_speakers_to_transcript` assigns each whisper segment (5-10 seconds) to a single speaker based on best time overlap. When a speaker change happens mid-segment, the entire segment gets attributed to one speaker, causing:

- Lines attributed to the wrong speaker near turn boundaries
- Two speakers' words merged into one speaker's line

## Solution

Enable `word_timestamps=True` in faster-whisper when diarization is active. Replace segment-level speaker assignment with word-level assignment that groups consecutive same-speaker words into lines.

## Changes

### 1. New Function: `assign_speakers_to_words`

In `src/meetscribe/transcription/diarize.py`.

**Input:**

- `words: list` — faster-whisper Word objects with `.start`, `.end`, `.word`
- `speaker_segments: list[SpeakerSegment]` — diarization results

**Algorithm:**

1. For each word, find the speaker segment with best time overlap
2. Group consecutive same-speaker words into lines
3. Each line gets the timestamp of its first word

**Returns:** `list[tuple[str, float, str]]` — `(speaker_label, start_time, text)`

This is a new return format (3-tuple with timestamp) because the re-grouped lines don't correspond to original segment indices.

The existing `assign_speakers_to_transcript` remains unchanged as a fallback.

### 2. Enable Word Timestamps in `transcribe_audio`

In `src/meetscribe/transcription/whisper.py`.

Pass `word_timestamps=True` to `model.transcribe()` when `enable_diarization=True`:

```python
segments, info = model.transcribe(
    str(audio_path), beam_size=5, vad_filter=True, language="en",
    initial_prompt=initial_prompt,
    word_timestamps=enable_diarization,
)
```

After collecting segments, extract all words and pass to the new function. If any segment has no words (word timestamps failed), fall back to `assign_speakers_to_transcript` for the entire transcript:

```python
if enable_diarization:
    all_words = []
    has_words = True
    for seg in segment_list:
        if seg.words:
            all_words.extend(seg.words)
        else:
            has_words = False
            break
    
    diarization_result = diarize(audio_path, num_speakers=num_speakers)
    if has_words and all_words:
        speaker_labels = assign_speakers_to_words(all_words, diarization_result.segments)
    else:
        speaker_labels = assign_speakers_to_transcript(segment_list, diarization_result.segments)
```

### 3. Update `format_transcript`

In `src/meetscribe/transcription/whisper.py`.

When `speaker_labels` contains 3-tuples `(speaker, start_time, text)`, use the embedded timestamp instead of indexing into the segments list. Detect the format by checking tuple length.

Output markdown is identical in structure — just with more accurate speaker breaks:

```markdown
**Speaker 1:**
[00:00:42] what's happening you tell me

**Speaker 2:**
[00:00:45] i had a one-on-one so i'm trying to order lunch
```

## Files to Modify

| File | Change |
|------|--------|
| `src/meetscribe/transcription/diarize.py` | Add `assign_speakers_to_words` function |
| `src/meetscribe/transcription/whisper.py` | Enable word timestamps, collect words, update `format_transcript` |
| `tests/test_diarize_embeddings.py` | Add tests for `assign_speakers_to_words` |
| `tests/test_whisper.py` | Add test for word timestamp enabling, update format test |

## Testing

### Unit tests for `assign_speakers_to_words`

- Words assigned to correct speaker based on overlap
- Consecutive same-speaker words grouped into one line
- Speaker change mid-segment produces two lines with correct timestamps
- Single word per speaker produces single-word lines
- Words with no overlapping speaker get "Unknown"

### Unit tests for `transcribe_audio`

- `word_timestamps=True` passed when `enable_diarization=True`
- `word_timestamps` not passed when `enable_diarization=False`
- Words collected from all segments and passed to new function

### Unit tests for `format_transcript`

- 3-tuple speaker_labels render correctly with embedded timestamps

### Existing tests unchanged

- `assign_speakers_to_transcript` tests remain (function not removed)
