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

### BlackHole Setup

BlackHole is a virtual audio driver that lets Meetscribe capture system audio (e.g., Zoom calls).

1. Install BlackHole:

   ```bash
   brew install blackhole-2ch
   ```

2. Open **Audio MIDI Setup** (search Spotlight or find it in `/Applications/Utilities/`)

3. Click the **+** button at the bottom-left and select **Create Multi-Output Device**

4. Check both your normal output (speakers/headphones) **and** **BlackHole 2ch**

5. In **System Settings > Sound > Output**, select the new **Multi-Output Device** as your output

This routes audio to both your speakers and BlackHole simultaneously, so Meetscribe can record while you still hear the meeting.

## Install

```bash
pip install -e .
```

Or install globally with pipx:

```bash
pipx install -e /path/to/meetscribe --python python3.13
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
