# Speaker Mapping Feature Design

## Overview

Add the ability to map generic speaker labels ("Speaker 1", "Speaker 2") to real names after transcription, with automatic speaker recognition across meetings using voice embeddings.

## Goals

1. **Manual renaming** — After diarized transcription, provide an inline UI to assign real names to detected speakers, rewriting the markdown transcript.
2. **Auto-recognition** — Store speaker voice embeddings so returning speakers are automatically suggested in future meetings.
3. **Non-destructive** — Original speaker labels are preserved in metadata, enabling re-mapping without re-transcription.

## Data Model

### Global Speaker Registry: `~/.config/meetscribe/speakers.json`

Central registry of known speakers with voice embeddings.

```json
{
  "speakers": [
    {
      "id": "sp_a1b2c3",
      "name": "Alice",
      "embeddings": [
        {
          "vector": [0.12, -0.34, "...192 floats"],
          "source_meeting": "2026-04-15/team-sync",
          "created_at": "2026-04-15T10:30:00Z"
        }
      ],
      "created_at": "2026-04-15T10:30:00Z",
      "updated_at": "2026-04-15T10:30:00Z"
    }
  ],
  "match_threshold": 0.65
}
```

- **Multiple embeddings per speaker** — Each meeting adds an embedding. More data improves recognition.
- **192-dim ECAPA-TDNN vectors** — Same model already used by `diarize.py` (SpeechBrain).
- **IDs** — `sp_` prefix + 6 random hex chars for stable cross-meeting references.
- **`match_threshold`** — Cosine similarity threshold for auto-matching. Default 0.65 (conservative). Stored alongside speaker data. No UI for tuning in this feature.
- **Embedding cap** — Max 10 embeddings per speaker. When exceeded, the oldest is dropped. Keeps profiles fresh and bounded (~1.5KB per embedding, ~150KB for 20 speakers with 5 each).

### Per-Meeting Metadata: `_metadata.json`

Existing metadata file gets a `speaker_map` field.

```json
{
  "meeting_name": "Team Sync",
  "date": "2026-04-15",
  "num_speakers": 3,
  "speaker_map": {
    "Speaker 1": {
      "speaker_id": "sp_a1b2c3",
      "name": "Alice",
      "original_label": "Speaker 1"
    },
    "Speaker 2": {
      "speaker_id": "sp_d4e5f6",
      "name": "Bob",
      "original_label": "Speaker 2"
    }
  }
}
```

- **`original_label`** — Backup of what the transcript originally contained. Enables re-mapping.
- **`speaker_id`** — Links to the global speaker profile.
- Unnamed speakers are omitted from `speaker_map`.

## Auto-Matching Flow

After diarization completes, before the mapping UI is shown:

1. **Extract per-cluster embeddings** — `_cluster_speakers()` already groups audio chunks by speaker. Compute a representative embedding per cluster by averaging the chunk embeddings in that cluster.

2. **Compare against known speakers** — For each cluster embedding, compute cosine similarity against every speaker in `speakers.json`, using the centroid (mean) of their stored embeddings as the comparison vector.

3. **Apply match threshold** — Best match above 0.65 is auto-suggested. If multiple clusters match the same known speaker, the highest similarity wins; others fall back to manual entry.

4. **Present results** — Mapping inputs appear pre-populated with matched names (indicated as auto-detected) or empty for unmatched speakers.

5. **On apply:**
   - Matched speakers: add new embedding to existing profile.
   - New speakers (manually named): create new profile in `speakers.json` with their embedding.
   - Update `_metadata.json` with `speaker_map`.
   - Rewrite transcript markdown with real names.

### Edge Cases

- **No known speakers yet** — All inputs empty. First meeting bootstraps the library.
- **Diarization disabled** — No mapping UI shown.
- **User skips naming** — Transcript keeps generic labels. No embeddings saved. User can return later.
- **Re-transcription** — Old `speaker_map` is cleared. Matching runs again against the new diarization.

## Diarization Code Changes

`diarize()` currently returns `list[SpeakerSegment]`. It must also return per-cluster embeddings.

New return: a result containing both the speaker segments and a `dict[str, np.ndarray]` mapping cluster labels (e.g., `"Speaker 1"`) to their average 192-dim embeddings. This can be a named tuple or dataclass.

The `_cluster_speakers()` function already has access to the per-chunk embeddings and cluster assignments — it just needs to aggregate them per cluster before returning.

## TUI: Inline Speaker Mapping

A collapsible section on the meeting screen's transcript tab, between the transcription controls and the markdown viewer.

```
┌─ Transcript ──────────────────────────────────┐
│ Model: [base ▾]  ☑ Identify speakers  # [3]   │
│ [Transcribe]                                   │
│                                                │
│ ┌─ Speaker Mapping ─────────────────────────┐  │
│ │ Speaker 1 → [Alice________] (matched)     │  │
│ │ Speaker 2 → [Bob__________] (matched)     │  │
│ │ Speaker 3 → [_____________]               │  │
│ │                     [Apply Names]          │  │
│ └────────────────────────────────────────────┘  │
│                                                │
│ **Alice:**                                     │
│ [00:00:05] Good morning everyone               │
│ **Bob:**                                       │
│ [00:00:12] Thanks for joining                  │
└────────────────────────────────────────────────┘
```

### Behavior

- **After diarized transcription** — Section auto-expands. Inputs pre-populated from auto-matching. Focus on first empty input.
- **Existing transcript** — Section visible but collapsed. Expanding shows current mapping from `_metadata.json`, editable.
- **No diarization** — Section hidden entirely.
- **"Apply Names"** — Rewrites markdown, saves `speaker_map` to metadata, updates/creates speaker profiles in `speakers.json`, refreshes the markdown viewer.
- **Auto-match indicator** — "(matched)" label next to auto-suggested names. Disappears after applying (all names are confirmed at that point).

### Textual Components

- `Collapsible` widget for the section
- `Input` widgets for each speaker name
- `Button` for apply
- `Static` labels for original speaker label and match indicator

## Transcript Rewriting

When "Apply Names" is pressed:

1. Read the current transcript markdown.
2. For each entry in the speaker map, replace all occurrences of `**{original_label}:**` with `**{name}:**`.
3. Write the updated markdown back to the file.
4. Store the `speaker_map` (with `original_label` backup) in `_metadata.json`.

If the user changes names later, the rewrite first tries to find the current mapped name in the file (from the existing `speaker_map`), falling back to `original_label`. This handles cases where the user has previously applied names or edited the markdown externally.

## Embedding Management

- **Created** when a speaker is named after diarization. The cluster's average embedding is saved to their profile.
- **Accumulated** over meetings. Each identification adds an embedding, improving the centroid.
- **Capped at 10** per speaker. Oldest dropped when exceeded.
- **Never auto-deleted** — Removing profiles is out of scope for this feature.
- **Centroid computation** — Mean of all stored embeddings for a speaker. Used as the comparison vector during matching.

## Testing Strategy

### Unit Tests (offline, no GPU)

- **Speaker matching logic** — Candidate embedding vs. known profiles at various similarity levels. Edge cases: empty profiles, single speaker, all below threshold, two clusters matching same profile.
- **Transcript rewriting** — Markdown with speaker map applied. Partial maps, re-mapping, edge cases (speaker label text in transcript body).
- **Metadata round-trip** — Save/load `speaker_map`. Migration from old metadata without the field.
- **`speakers.json` CRUD** — Create, read, update profiles. Embedding cap enforcement, centroid computation, ID generation.
- **Embedding extraction** — Mock SpeechBrain, verify `diarize()` returns segments + per-cluster embeddings.

### Integration Tests (TUI)

- Speaker mapping section shows/hides based on diarization state.
- "Apply Names" triggers rewrite and metadata save.
- Auto-populated names appear in inputs from matching.

## Key Files to Modify

| File | Change |
|------|--------|
| `src/meetscribe/transcription/diarize.py` | Return per-cluster embeddings alongside segments |
| `src/meetscribe/transcription/whisper.py` | Pass embeddings through to caller |
| `src/meetscribe/tui/screens/meeting.py` | Add speaker mapping UI section |
| `src/meetscribe/storage/vault.py` | Speaker map metadata helpers |
| `src/meetscribe/storage/speakers.py` (new) | `speakers.json` CRUD, matching logic, embedding management |
| `tests/test_speakers.py` (new) | Unit tests for matching, rewriting, CRUD |
