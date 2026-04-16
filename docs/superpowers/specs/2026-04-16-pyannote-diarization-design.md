# Pyannote Speaker Diarization Design

## Overview

Replace the SpeechBrain ECAPA-TDNN + manual clustering diarization with pyannote-audio's pretrained speaker diarization pipeline. pyannote uses a neural segmentation model trained specifically for speaker boundaries, producing significantly more accurate turn detection for fast-paced conversations.

## Problem

The current approach (SpeechBrain embeddings + hierarchical clustering on fixed-length audio chunks) cannot detect speaker changes within a chunk. Tuning chunk size is a losing trade-off: longer chunks miss fast turn-taking, shorter chunks produce unstable embeddings. pyannote solves this with an end-to-end neural pipeline that models speaker turns directly.

## Changes

### 1. Rewrite `diarize()` Internals

Replace the body of `src/meetscribe/transcription/diarize.py`. The public interface stays the same:

```python
def diarize(audio_path: Path, num_speakers: int | None = None) -> DiarizationResult
```

**Remove:** `_load_audio`, `_extract_embeddings`, `_cluster_speakers`, all SpeechBrain/scipy imports, `SEGMENT_LENGTH`, `SEGMENT_STEP`, `ENERGY_THRESHOLD` constants.

**Keep:** `SpeakerSegment`, `DiarizationResult`, `assign_speakers_to_transcript`, `assign_speakers_to_words`.

**New internals:**

1. Load pyannote pipeline (cached in module-level variable after first load)
2. Determine device: `mps` if `torch.backends.mps.is_available()`, else `cpu`
3. Get HF token via `_get_hf_token()`
4. Run `pipeline(audio_path, num_speakers=num_speakers)`
5. Use `output.exclusive_speaker_diarization` for non-overlapping segments
6. Convert pyannote segments to `list[SpeakerSegment]` with labels `"Speaker 1"`, `"Speaker 2"`, etc. (map pyannote's `SPEAKER_00` labels to our format)
7. Extract per-speaker embeddings using `pyannote.audio.pipelines.SpeakerEmbedding` on the longest segment per speaker
8. Return `DiarizationResult`

### 2. HuggingFace Token Configuration

**Resolution order:**

1. Environment variable `HF_TOKEN`
2. `config.toml` field `huggingface_token` (top-level scalar, before sections)

**Config model:** Add `huggingface_token: str = ""` to `MeetscribeConfig`.

**`_config_to_dict`:** Add `"huggingface_token": cfg.huggingface_token` as the first key (scalar before sections, per TOML gotcha in CLAUDE.md).

**`_dict_to_config`:** Read `data.get("huggingface_token", "")`.

**Error:** If neither source has a token, raise with message: `"HuggingFace token required for speaker diarization. Set HF_TOKEN env var or huggingface_token in config.toml. Get a free token at https://huggingface.co/settings/tokens"`

### 3. Speaker Embedding Migration

pyannote produces 512-dim embeddings (vs SpeechBrain's 192-dim). Existing `speakers.json` profiles are incompatible.

**`speakers.py` changes:**

- Add `EMBEDDING_DIM = 512` constant
- In `SpeakerRegistry._load()`, after loading profiles, check if any profile has embeddings with length != `EMBEDDING_DIM`. If so, clear all profiles and re-save. Log warning: `"Speaker profiles reset — embedding model changed."`

### 4. Model Caching and Device Selection

```python
_pipeline = None
_embedding_model = None

def _get_pipeline(hf_token: str) -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        _pipeline.to(device)
    return _pipeline
```

Same pattern for the embedding model. Both cached for the process lifetime.

### 5. Dependency Changes

**`pyproject.toml`:**

- Add: `"pyannote.audio>=3.3.0"`
- Remove: `"speechbrain"` is not in pyproject.toml (it was likely installed as a transitive dependency or manually). No change needed there.

**`src/meetscribe/tui/app.py:31`:** Update the logging suppression list — replace `"speechbrain"` with `"pyannote"`.

## Files to Modify

| File | Change |
|------|--------|
| `src/meetscribe/transcription/diarize.py` | Replace internals with pyannote pipeline |
| `src/meetscribe/config.py` | Add `huggingface_token` to config model |
| `src/meetscribe/storage/speakers.py` | Add `EMBEDDING_DIM`, validation on load |
| `pyproject.toml` | Add `pyannote.audio` dependency |
| `src/meetscribe/tui/app.py` | Update log suppression list |
| `tests/test_diarize_embeddings.py` | Remove `_cluster_speakers` tests, add pyannote mock tests |
| `tests/test_speakers.py` | Add embedding dimension reset test |
| `tests/test_config.py` | Add `huggingface_token` test |

## Testing

### New tests

- **`diarize()` with pyannote** — Mock pipeline and verify `DiarizationResult` output, `num_speakers` passthrough, device selection
- **HF token resolution** — env var, config.toml, missing token error
- **Speaker profile reset** — Load registry with 192-dim embeddings, verify profiles cleared

### Removed tests

- `TestClusterSpeakersEmbeddings` — tests `_cluster_speakers` which is removed

### Unchanged tests

- `TestAssignSpeakersToWords` — function unchanged
- All speaker matching, transcript rewriting, metadata tests — unchanged
- All whisper.py tests — unchanged
