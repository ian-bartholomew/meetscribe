from __future__ import annotations

import os
from pathlib import Path

# Disable tqdm's multiprocessing lock — it crashes in Textual worker threads
# on Python 3.13 due to fork_exec issues. Must be set before any tqdm import.
os.environ.setdefault("TQDM_DISABLE", "1")

from faster_whisper import WhisperModel

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(
    segments: list,
    meeting_name: str,
    meeting_date: str,
    model: str,
    duration: str,
) -> str:
    """Format transcription segments into a markdown document with frontmatter."""
    lines = [
        "---",
        f"meeting: {meeting_name}",
        f"date: {meeting_date}",
        f"model: {model}",
        f'duration: "{duration}"',
        "---",
        "",
    ]
    for segment in segments:
        timestamp = format_timestamp(segment.start)
        text = segment.text.strip()
        lines.append(f"[{timestamp}] {text}")
        lines.append("")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format total seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    meeting_name: str,
    meeting_date: str,
) -> str:
    """Transcribe an audio file and return formatted markdown transcript."""
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    segment_list = list(segments)

    duration = _format_duration(info.duration)

    return format_transcript(
        segments=segment_list,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        model=model_name,
        duration=duration,
    )
