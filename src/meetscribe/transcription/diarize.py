"""Speaker diarization using SpeechBrain ECAPA-TDNN embeddings + spectral clustering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from speechbrain.inference.speaker import SpeakerRecognition

log = logging.getLogger("meetscribe.diarize")

# Segment length in seconds for embedding extraction
SEGMENT_LENGTH = 3.0
SEGMENT_STEP = 1.5  # Overlap by half for better coverage
SAMPLE_RATE = 16000  # SpeechBrain models expect 16kHz


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


def _load_audio(audio_path: Path) -> torch.Tensor:
    """Load audio file and resample to 16kHz mono using soundfile + scipy."""
    data, sr = sf.read(str(audio_path), dtype="float32")

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        num_samples = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, num_samples).astype(np.float32)

    # Convert to torch tensor with shape (1, num_samples)
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform


def _extract_embeddings(
    waveform: torch.Tensor,
    model: SpeakerRecognition,
) -> list[tuple[float, float, np.ndarray]]:
    """Extract speaker embeddings from overlapping audio segments.

    Returns list of (start_time, end_time, embedding) tuples.
    """
    total_samples = waveform.shape[1]
    total_duration = total_samples / SAMPLE_RATE
    segment_samples = int(SEGMENT_LENGTH * SAMPLE_RATE)
    step_samples = int(SEGMENT_STEP * SAMPLE_RATE)

    segments: list[tuple[float, float, np.ndarray]] = []

    pos = 0
    while pos < total_samples:
        end = min(pos + segment_samples, total_samples)
        chunk = waveform[:, pos:end]

        # Skip very short segments (less than 0.5s)
        if chunk.shape[1] < SAMPLE_RATE * 0.5:
            break

        # Pad short segments
        if chunk.shape[1] < segment_samples:
            chunk = torch.nn.functional.pad(chunk, (0, segment_samples - chunk.shape[1]))

        with torch.no_grad():
            embedding = model.encode_batch(chunk)
            emb_np = embedding.squeeze().numpy()

        start_time = pos / SAMPLE_RATE
        end_time = min(end / SAMPLE_RATE, total_duration)
        segments.append((start_time, end_time, emb_np))

        pos += step_samples

    return segments


def _cluster_speakers(
    segments: list[tuple[float, float, np.ndarray]],
    num_speakers: int | None = None,
    threshold: float = 0.7,
) -> list[SpeakerSegment]:
    """Cluster embeddings to identify speakers.

    If num_speakers is None, automatically determines the number
    using the distance threshold.
    """
    if not segments:
        return []

    embeddings = np.array([s[2] for s in segments])

    if len(embeddings) == 1:
        return [SpeakerSegment(segments[0][0], segments[0][1], "Speaker 1")]

    # Hierarchical clustering with cosine distance
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    linkage_matrix = linkage(embeddings_norm, method="ward")

    if num_speakers:
        labels = fcluster(linkage_matrix, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    result: list[SpeakerSegment] = []
    for (start, end, _), label in zip(segments, labels):
        result.append(SpeakerSegment(start, end, f"Speaker {label}"))

    return result


def diarize(
    audio_path: Path,
    num_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Run speaker diarization on an audio file.

    Returns a list of SpeakerSegments with start/end times and speaker labels.
    """
    log.info("Loading audio: %s", audio_path)
    waveform = _load_audio(audio_path)
    duration = waveform.shape[1] / SAMPLE_RATE
    log.info("Audio duration: %.1fs", duration)

    log.info("Loading speaker embedding model...")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache" / "meetscribe" / "spkrec-ecapa"),
    )

    log.info("Extracting speaker embeddings...")
    segments = _extract_embeddings(waveform, model)
    log.info("Extracted %d segment embeddings", len(segments))

    log.info("Clustering speakers...")
    speaker_segments = _cluster_speakers(segments, num_speakers=num_speakers)

    # Count unique speakers
    speakers = {s.speaker for s in speaker_segments}
    log.info("Identified %d speakers", len(speakers))

    return speaker_segments


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
        t_mid = (t_start + t_end) / 2.0

        # Find the speaker segment that best overlaps with this transcript segment
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
