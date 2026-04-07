from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

log = logging.getLogger("meetscribe.whisper")

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# Map model names to HuggingFace repo IDs
_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def _find_cached_model(model_name: str) -> str | None:
    """Find a cached model path without triggering tqdm/download code."""
    repo_id = _MODEL_REPOS.get(model_name)
    if not repo_id:
        return None

    # HuggingFace cache structure: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"

    if not repo_dir.exists():
        return None

    # Get the most recent snapshot
    snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if (snapshot / "model.bin").exists():
            return str(snapshot)

    return None


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
    # Try cached path first to avoid tqdm crash in Textual worker threads
    cached_path = _find_cached_model(model_name)
    if cached_path:
        log.info("Using cached model at %s", cached_path)
        model = WhisperModel(cached_path, device="cpu", compute_type="int8")
    else:
        log.info("Model not cached, downloading %s (this may take a minute)...", model_name)
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
