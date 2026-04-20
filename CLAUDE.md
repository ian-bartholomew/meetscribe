# Meetscribe

## Commands

- `pip install -e .` — install in dev mode
- `pipx install -e . --python python3.13` — install globally
- `meetscribe` — run the app
- `pytest tests/` — run all tests (94 tests, no network/GPU needed)

## Architecture

- Python TUI app using Textual, faster-whisper, sounddevice, pyannote-audio, openai client, Jinja2
- Single package at `src/meetscribe/` with modules: audio, transcription, summarization, templates, storage, tui
- Config at `~/.config/meetscribe/config.toml`, logs at `~/.config/meetscribe/meetscribe.log`
- Speaker profiles at `~/.config/meetscribe/speakers.json` — cross-meeting voice recognition
- All state is file-based — no database
- Meeting directories: flat `YYYY-MM-DD-meeting-name/` structure under the configured meetings folder
- Templates are Jinja2 `.md` files in `templates/` — drop a new file there to add a template

## Key Files

- `src/meetscribe/__main__.py` — entry point, contains critical multiprocessing.RLock patch
- `src/meetscribe/transcription/whisper.py` — transcription with cached model loading, word-level timestamps
- `src/meetscribe/transcription/diarize.py` — speaker diarization via pyannote-audio, word-level speaker assignment
- `src/meetscribe/audio/player.py` — audio playback with seeking (sounddevice + soundfile)
- `src/meetscribe/audio/recorder.py` — audio recording to FLAC
- `src/meetscribe/storage/vault.py` — filesystem operations, metadata, meeting CRUD
- `src/meetscribe/storage/speakers.py` — speaker registry, embedding matching, transcript rewriting
- `src/meetscribe/config.py` — config model, TOML load/save, `CONFIG_DIR`
- `src/meetscribe/tui/screens/meeting.py` — main meeting detail screen (largest file)
- `src/meetscribe/importer.py` — Hyprnote session import

## Critical Gotchas

- **tqdm crash**: tqdm's multiprocessing.RLock crashes in Textual worker threads on Python 3.13. Patched in `__main__.py` — do NOT remove.
- **Textual name collisions**: Never use method names starting with `_update_` on Screen subclasses — conflicts with Textual internals.
- **Select.BLANK**: `str(Select.BLANK)` returns `"Select.NULL"` (truthy). Always check `isinstance(val, str)` before using Select values.
- **TOML key ordering**: In `config.py`, scalar keys (like `log_level`, `huggingface_token`) must be serialized BEFORE sections with sub-tables (`summarization.endpoints`) or TOML parses them as part of that section.
- **Path("~")**: Python's `Path("~/foo")` does NOT expand `~`. Always call `.expanduser()`. See `vault.py`.
- **faster-whisper model loading**: Load from cached HuggingFace path directly (`_find_cached_model()` in `whisper.py`) to avoid tqdm download code path crashing in worker threads.
- **pyannote torchcodec**: pyannote 4.x uses torchcodec which requires FFmpeg. We bypass this by loading audio with soundfile and passing waveform dicts directly to the pipeline.
- **Mono audio playback**: `soundfile.read()` returns 1D arrays for mono files but `sounddevice.OutputStream` expects 2D. Always use `always_2d=True`.
- **Editable install staleness**: `pipx install -e .` doesn't always pick up code changes. Reinstall with `--force` after modifications.
