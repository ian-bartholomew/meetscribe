# Meetscribe

A TUI app for recording, transcribing, and summarizing meetings.

## Features

- Record system audio via BlackHole
- Transcribe recordings locally with faster-whisper (multiple model sizes)
- Summarize meetings using local LLMs (Ollama or LM Studio)
- Customizable summary templates (Jinja2)
- Free-form memos to supplement transcripts
- All artifacts stored in your Obsidian vault

## Prerequisites

- Python 3.11+
- [BlackHole](https://existential.audio/blackhole/) audio driver installed
- [Ollama](https://ollama.ai/) and/or [LM Studio](https://lmstudio.ai/) for summarization

## Install

```bash
pip install -e .
```

## Usage

```bash
meetscribe
```

On first launch, you'll be prompted to configure your Obsidian vault path.

## Configuration

Config is stored at `~/.config/meetscribe/config.toml`. You can also edit settings from within the TUI by pressing `s`.

## Vault Structure

```
<vault>/<meetings_folder>/<year>/<month>/<day>/<meeting-name>/
├── recording.flac
├── transcript-<model>.md
├── summary-<template>.md
└── memos.md
```

## Templates

Meeting summary templates are Jinja2 `.md` files in the `templates/` directory. Available variables:

- `{{ transcript }}` — full transcript
- `{{ memos }}` — user notes
- `{{ meeting_name }}` — meeting name
- `{{ date }}` — meeting date
- `{{ duration }}` — recording duration
