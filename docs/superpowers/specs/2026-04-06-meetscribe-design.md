# Meetscribe — Design Spec

## Overview

Meetscribe is a Python TUI CLI app for recording system audio from Zoom meetings (via BlackHole), transcribing recordings locally using faster-whisper, and generating summaries using local LLMs (Ollama or LM Studio). All artifacts are stored in an Obsidian vault with a configurable folder structure.

## Architecture

**Approach:** Monolithic single-package Python app with clean internal module boundaries.

**Entry point:** `meetscribe` CLI command launches a Textual TUI.

### Project Structure

```
meetscribe/
├── pyproject.toml
├── src/
│   └── meetscribe/
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── config.py            # Config loading/saving
│       ├── audio/
│       │   ├── __init__.py
│       │   └── recorder.py      # BlackHole system audio capture -> FLAC
│       ├── transcription/
│       │   ├── __init__.py
│       │   └── whisper.py       # faster-whisper integration, model selection
│       ├── summarization/
│       │   ├── __init__.py
│       │   ├── provider.py      # Base interface for LLM providers
│       │   ├── ollama.py        # Ollama OpenAI-compatible client
│       │   └── lmstudio.py      # LM Studio OpenAI-compatible client
│       ├── templates/
│       │   ├── __init__.py
│       │   └── engine.py        # Jinja2 template loading & rendering
│       ├── storage/
│       │   ├── __init__.py
│       │   └── vault.py         # Obsidian vault file management
│       └── tui/
│           ├── __init__.py
│           ├── app.py           # Main Textual app
│           ├── screens/
│           │   ├── __init__.py
│           │   ├── home.py      # Landing screen
│           │   ├── recording.py # Active recording screen
│           │   ├── meeting.py   # View/manage meeting artifacts
│           │   └── settings.py  # Configure defaults
│           └── widgets/
│               ├── __init__.py
│               └── ...          # Reusable TUI components
├── templates/                   # Default meeting templates (Jinja2 .md)
│   ├── default.md
│   ├── standup.md
│   └── retrospective.md
└── tests/
```

### Dependencies

- `textual` — TUI framework
- `faster-whisper` — local Whisper transcription
- `sounddevice` + `soundfile` — audio capture & FLAC encoding
- `openai` — client for Ollama and LM Studio (OpenAI-compatible APIs)
- `jinja2` — template rendering
- `tomli` / `tomli-w` — config file reading/writing

## Audio Recording

The recorder captures system audio via BlackHole, a virtual audio device on macOS.

**Flow:**

1. User starts recording from the TUI, providing a meeting name
2. `sounddevice` opens an input stream using the BlackHole device (identified by name via `sounddevice.query_devices()`)
3. Audio is captured in chunks and written to a FLAC file via `soundfile`
4. The TUI displays a recording timer and volume level indicator
5. User stops recording — stream closes, FLAC file is finalized

**File path:** `<vault.root>/<vault.meetings_folder>/<year>/<month>/<day>/<meeting-name>/recording.flac`

**Config options:**

- `audio.device_name` — defaults to `"BlackHole 2ch"`
- `audio.sample_rate` — defaults to `44100`
- `audio.channels` — defaults to `2`

## Transcription

Uses faster-whisper to transcribe FLAC recordings locally.

**Flow:**

1. User hits "Transcribe" or "Regenerate Transcript" on the Meeting screen
2. User selects a Whisper model (or uses default)
3. faster-whisper loads the model and processes the FLAC file
4. Transcript is saved as `transcript-<model>.md` (e.g., `transcript-base.md`)
5. TUI shows progress indicator and displays result when done

**Model selection:**

- `transcription.default_model` in config — defaults to `base`
- Available models: `tiny`, `base`, `small`, `medium`, `large-v3` (static list; TUI indicates which are downloaded locally)
- User can override model from the TUI when transcribing or regenerating

**Transcript format:**

```markdown
---
meeting: Weekly Standup
date: 2026-04-06
model: base
duration: "00:45:12"
---

[00:00:00] First segment of transcribed text...

[00:00:05] Next segment...
```

Timestamps from faster-whisper segment output. YAML frontmatter for Obsidian queryability.

**Regeneration:** Regenerating with the same model overwrites the file. Using a different model creates a new file (e.g., `transcript-large-v3.md`), preserving both versions.

## Summarization

Uses local LLMs via Ollama or LM Studio to generate meeting summaries from templates.

**Flow:**

1. User selects a meeting template and hits "Summarize" or "Regenerate Summary"
2. User selects provider (ollama/lmstudio) and model from the TUI (defaults from config)
3. The app loads the Jinja2 template, renders it with transcript, memos, and metadata
4. The rendered prompt is sent to the LLM via OpenAI-compatible chat completion API
5. Response is saved as `summary-<template-name>.md` (e.g., `summary-standup.md`)

**Provider interface:**
Both providers expose OpenAI-compatible APIs. A single `openai` client handles both — just swap the `base_url`:

- Ollama: `http://localhost:11434/v1`
- LM Studio: `http://localhost:1234/v1`

**Model discovery:**
The app queries each provider's `/v1/models` endpoint to populate model lists dynamically in the TUI. No hardcoded model lists.

**Template variables:**

- `{{ transcript }}` — full transcript text
- `{{ memos }}` — user's notes from the memos tab
- `{{ meeting_name }}` — name of the meeting
- `{{ date }}` — meeting date
- `{{ duration }}` — recording duration

**Example template (`templates/standup.md`):**

```markdown
Summarize the following meeting transcript as a standup summary.

## Format
- **Yesterday:** What was discussed about past work
- **Today:** What was planned
- **Blockers:** Any blockers mentioned

## Transcript
{{ transcript }}

{% if memos %}
## Additional Notes
{{ memos }}
{% endif %}
```

**Regeneration:** Each template produces its own file. Regenerating with the same template overwrites; using a different template creates a new file.

## Memos

Free-form notes that supplement the transcript for summary generation.

- Fourth tab on the Meeting screen — simple text editor widget
- Saved as `memos.md` in the meeting folder
- Auto-saves or saves on keybinding
- Passed to the LLM as `{{ memos }}` alongside the transcript when generating summaries

## TUI Screens & Navigation

### Home Screen

- "New Recording" button — prompts for meeting name, pushes to Recording screen
- Browsable list of past meetings by date
- Each entry shows meeting name, date, and icons for existing artifacts (recording/transcript/summary)
- `s` keybinding for Settings, `q` to quit

### Recording Screen

- Displays: meeting name, elapsed timer, volume level indicator
- "Stop Recording" button
- On stop: transitions to Meeting screen for that recording

### Meeting Screen

- Four tabbed panels: **Recording**, **Transcript**, **Summary**, **Memos**
- **Recording tab:** playback info, file size
- **Transcript tab:** view transcript, select model (static list with download status), "Transcribe" / "Regenerate" buttons
- **Summary tab:** select template, select provider + model (fetched from `/v1/models`), "Summarize" / "Regenerate" buttons
- **Memos tab:** text editor for free-form notes, auto-saves
- Progress indicators during transcription/summarization
- `escape` to go back to Home

### Settings Screen

- Vault root path and meetings folder
- Default Whisper model
- Default LLM provider, model, and endpoint URLs
- Audio device name
- Saves to `~/.config/meetscribe/config.toml`

## Storage & Vault Structure

**Meeting folder layout:**

```
<vault.root>/<vault.meetings_folder>/<year>/<month>/<day>/<meeting-name>/
├── recording.flac
├── transcript-base.md
├── transcript-large-v3.md      # if regenerated with different model
├── summary-standup.md
├── summary-default.md           # if generated with different template
└── memos.md
```

**Conventions:**

- Date components are zero-padded: `2026/04/06`
- Meeting name is slugified: lowercase, hyphens for spaces, special chars stripped
- Directories created on demand when recording starts
- All markdown files include YAML frontmatter for Obsidian queryability
- No database — the filesystem is the source of truth
- Home screen scans the meetings folder recursively, parsing the date structure to build the list

## Configuration

**Config file:** `~/.config/meetscribe/config.toml`

```toml
[vault]
root = "/Users/ian/my-vault"
meetings_folder = "Meetings"

[audio]
device_name = "BlackHole 2ch"
sample_rate = 44100
channels = 2

[transcription]
default_model = "base"

[summarization]
default_provider = "ollama"
default_model = "llama3"

[summarization.endpoints]
ollama = "http://localhost:11434/v1"
lmstudio = "http://localhost:1234/v1"
```

**First run:** If no config file exists, the app launches a first-run setup in the TUI — prompts for vault root and meetings folder at minimum, uses sensible defaults for everything else. Writes the config file.

**Settings screen:** Edits this same file. Changes take effect immediately.
