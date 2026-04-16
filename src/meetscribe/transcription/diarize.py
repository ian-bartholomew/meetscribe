"""Speaker diarization using pyannote-audio pretrained pipeline."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

log = logging.getLogger("meetscribe.diarize")


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    cluster_embeddings: dict[str, np.ndarray]


_pipeline = None
_embedding_model = None


def _get_hf_token(config_token: str = "") -> str:
    """Get HuggingFace token from env var or config."""
    token = os.environ.get("HF_TOKEN", "") or config_token
    if not token:
        raise RuntimeError(
            "HuggingFace token required for speaker diarization. "
            "Set HF_TOKEN env var or huggingface_token in config.toml. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )
    return token


def _get_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_pipeline(hf_token: str):
    """Load and cache the pyannote diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        log.info("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        device = _get_device()
        log.info("Using device: %s", device)
        _pipeline.to(device)
    return _pipeline


def _get_embedding_model(hf_token: str):
    """Load and cache the pyannote speaker embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from pyannote.audio import Model, Inference
        log.info("Loading pyannote embedding model...")
        model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        _embedding_model = Inference(model, window="whole")
    return _embedding_model


def _load_audio_for_pyannote(audio_path: Path) -> dict:
    """Load audio as a waveform dict for pyannote (bypasses torchcodec)."""
    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(
    audio_path: Path,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on an audio file.

    Returns a DiarizationResult with speaker segments and per-cluster embeddings.
    """
    from meetscribe.config import load_config
    config = load_config()
    hf_token = _get_hf_token(config.huggingface_token)

    pipeline = _get_pipeline(hf_token)

    log.info("Running diarization on %s", audio_path)
    audio_input = _load_audio_for_pyannote(audio_path)
    kwargs = {}
    if num_speakers is not None and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(audio_input, **kwargs)

    # Use exclusive diarization (non-overlapping) for clean transcript assignment
    exclusive = output.exclusive_speaker_diarization

    # Map pyannote labels (SPEAKER_00) to our format (Speaker 1)
    label_map: dict[str, str] = {}
    segments: list[SpeakerSegment] = []
    for segment, _, speaker in exclusive.itertracks(yield_label=True):
        if speaker not in label_map:
            label_map[speaker] = f"Speaker {len(label_map) + 1}"
        segments.append(SpeakerSegment(
            start=segment.start,
            end=segment.end,
            speaker=label_map[speaker],
        ))

    speakers = set(label_map.values())
    log.info("Identified %d speakers", len(speakers))

    # Extract per-speaker embeddings from longest segment
    cluster_embeddings: dict[str, np.ndarray] = {}
    try:
        embedding_model = _get_embedding_model(hf_token)
    except Exception:
        log.warning("Could not load embedding model — speaker profiles will not be updated")
        log.info("Accept model terms at https://hf.co/pyannote/embedding to enable speaker recognition")
        return DiarizationResult(segments=segments, cluster_embeddings=cluster_embeddings)
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]
    for pyannote_label, our_label in label_map.items():
        speaker_segs = [s for s in segments if s.speaker == our_label]
        if not speaker_segs:
            continue
        longest = max(speaker_segs, key=lambda s: s.end - s.start)
        try:
            start_sample = int(longest.start * sample_rate)
            end_sample = int(longest.end * sample_rate)
            crop_waveform = waveform[:, start_sample:end_sample]
            emb_input = {"waveform": crop_waveform, "sample_rate": sample_rate}
            embedding = embedding_model(emb_input)
            if hasattr(embedding, 'numpy'):
                emb_np = embedding.squeeze().numpy()
            elif isinstance(embedding, np.ndarray):
                emb_np = embedding.squeeze()
            else:
                emb_np = np.array(embedding).squeeze()
            cluster_embeddings[our_label] = emb_np
        except Exception:
            log.warning("Failed to extract embedding for %s", our_label, exc_info=True)

    log.info("Diarization complete")
    return DiarizationResult(segments=segments, cluster_embeddings=cluster_embeddings)


def assign_speakers_to_transcript(
    transcript_segments: list,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, str]]:
    """Match transcript segments to speakers based on time overlap.

    Args:
        transcript_segments: faster-whisper segments with .start, .end, .text
        speaker_segments: diarization results

    Returns:
        List of (speaker_label, text) tuples.
    """
    results: list[tuple[str, str]] = []

    for tseg in transcript_segments:
        t_start = tseg.start
        t_end = tseg.end

        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(t_start, sseg.start)
            overlap_end = min(t_end, sseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        results.append((best_speaker, tseg.text.strip()))

    return results


def assign_speakers_to_words(
    words: list,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, float, str]]:
    """Assign speakers to individual words and group consecutive same-speaker words.

    Args:
        words: faster-whisper Word objects with .start, .end, .word
        speaker_segments: diarization results

    Returns:
        List of (speaker_label, start_time, text) tuples. Each tuple is a group
        of consecutive words from the same speaker.
    """
    if not words:
        return []

    word_speakers: list[tuple[str, float, str]] = []
    for w in words:
        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(w.start, sseg.start)
            overlap_end = min(w.end, sseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        # If no overlap, assign to nearest speaker segment
        if best_speaker == "Unknown" and speaker_segments:
            word_mid = (w.start + w.end) / 2
            nearest = min(speaker_segments, key=lambda s: min(abs(s.start - word_mid), abs(s.end - word_mid)))
            best_speaker = nearest.speaker

        word_speakers.append((best_speaker, w.start, w.word.strip()))

    groups: list[tuple[str, float, str]] = []
    current_speaker = word_speakers[0][0]
    current_start = word_speakers[0][1]
    current_words: list[str] = [word_speakers[0][2]]

    for speaker, start, word in word_speakers[1:]:
        if speaker == current_speaker:
            current_words.append(word)
        else:
            groups.append((current_speaker, current_start, " ".join(current_words)))
            current_speaker = speaker
            current_start = start
            current_words = [word]

    groups.append((current_speaker, current_start, " ".join(current_words)))
    return groups
