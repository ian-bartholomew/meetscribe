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
    speaker_labels: list[tuple[str, str]] | None = None,
) -> str:
    """Format transcription segments into a markdown document with frontmatter.

    If speaker_labels is provided, uses (speaker, text) tuples instead of
    raw segments for the body.
    """
    lines = [
        "---",
        f"meeting: {meeting_name}",
        f"date: {meeting_date}",
        f"model: {model}",
        f'duration: "{duration}"',
        "---",
        "",
    ]

    if speaker_labels:
        prev_speaker = None
        for i, (speaker, text) in enumerate(speaker_labels):
            timestamp = format_timestamp(segments[i].start)
            if speaker != prev_speaker:
                lines.append(f"**{speaker}:**")
                prev_speaker = speaker
            lines.append(f"[{timestamp}] {text}")
            lines.append("")
    else:
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


def _load_model(model_name: str) -> WhisperModel:
    """Load a WhisperModel, preferring the cached path."""
    cached_path = _find_cached_model(model_name)
    if cached_path:
        log.info("Using cached model at %s", cached_path)
        return WhisperModel(cached_path, device="cpu", compute_type="int8")
    log.info("Model not cached, downloading %s (this may take a minute)...", model_name)
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    meeting_name: str,
    meeting_date: str,
    enable_diarization: bool = False,
    num_speakers: int | None = None,
    on_segment: callable | None = None,
    custom_vocabulary: list[str] | None = None,
) -> tuple[str, dict | None]:
    """Transcribe an audio file and return formatted markdown transcript.

    Args:
        on_segment: Optional callback called with (segment_index, timestamp, text)
                    after each segment is transcribed. Use for live UI updates.
        custom_vocabulary: Optional list of words/phrases to prime the model with.
    """
    import time as _time
    t0 = _time.monotonic()

    log.info("Loading model: %s", model_name)
    model = _load_model(model_name)
    log.info("Model loaded in %.1fs", _time.monotonic() - t0)

    log.info("Starting transcription of %s", audio_path)
    t1 = _time.monotonic()
    initial_prompt = ", ".join(custom_vocabulary) if custom_vocabulary else None
    segments, info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True, language="en",
        initial_prompt=initial_prompt,
    )
    log.info("Transcribe call returned (generator ready) in %.1fs", _time.monotonic() - t1)

    # Stream segments — call on_segment as each one arrives
    log.info("Iterating segments...")
    segment_list: list = []
    for segment in segments:
        segment_list.append(segment)
        if on_segment:
            timestamp = format_timestamp(segment.start)
            text = segment.text.strip()
            on_segment(len(segment_list) - 1, timestamp, text)
        if len(segment_list) % 50 == 0:
            log.info("Transcribed %d segments so far (at %.1fs in audio)",
                     len(segment_list), segment.end)

    duration = _format_duration(info.duration)
    log.info("Transcription complete: %d segments, %s duration, took %.1fs",
             len(segment_list), duration, _time.monotonic() - t0)

    speaker_labels = None
    cluster_embeddings = None
    if enable_diarization:
        log.info("Running speaker diarization...")
        from meetscribe.transcription.diarize import diarize, assign_speakers_to_transcript
        diarization_result = diarize(audio_path, num_speakers=num_speakers)
        speaker_labels = assign_speakers_to_transcript(segment_list, diarization_result.segments)
        cluster_embeddings = {
            label: emb.tolist() for label, emb in diarization_result.cluster_embeddings.items()
        }
        log.info("Diarization complete")

    transcript = format_transcript(
        segments=segment_list,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        model=model_name,
        duration=duration,
        speaker_labels=speaker_labels,
    )
    return transcript, cluster_embeddings
