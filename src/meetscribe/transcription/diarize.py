"""Speaker diarization using SpeechBrain ECAPA-TDNN embeddings + clustering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from speechbrain.inference.speaker import SpeakerRecognition

log = logging.getLogger("meetscribe.diarize")

# Longer segments produce more stable speaker embeddings
SEGMENT_LENGTH = 5.0
SEGMENT_STEP = 2.5  # 50% overlap
SAMPLE_RATE = 16000  # SpeechBrain models expect 16kHz

# Minimum audio energy to consider a segment as speech (skip silence)
ENERGY_THRESHOLD = 0.005


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    cluster_embeddings: dict[str, np.ndarray]


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

    Skips silent segments to avoid polluting the clustering with noise.
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

        # Skip very short segments (less than 1s)
        if chunk.shape[1] < SAMPLE_RATE:
            break

        # Skip silent segments
        energy = float(torch.sqrt(torch.mean(chunk ** 2)))
        if energy < ENERGY_THRESHOLD:
            pos += step_samples
            continue

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
    threshold: float = 1.2,
) -> DiarizationResult:
    """Cluster embeddings to identify speakers.

    Uses cosine distance with average linkage for better speaker separation.
    If num_speakers is provided, forces exactly that many clusters.
    Otherwise, uses the distance threshold to determine cluster count.

    Returns a DiarizationResult with segments and per-cluster mean embeddings.
    """
    if not segments:
        return DiarizationResult(segments=[], cluster_embeddings={})

    embeddings = np.array([s[2] for s in segments])

    if len(embeddings) == 1:
        return DiarizationResult(
            segments=[SpeakerSegment(segments[0][0], segments[0][1], "Speaker 1")],
            cluster_embeddings={"Speaker 1": embeddings[0]},
        )

    # Compute cosine distances and cluster with average linkage
    distances = pdist(embeddings, metric="cosine")
    linkage_matrix = linkage(distances, method="average")

    if num_speakers and num_speakers > 0:
        labels = fcluster(linkage_matrix, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    result_segments: list[SpeakerSegment] = []
    for (start, end, _), label in zip(segments, labels):
        result_segments.append(SpeakerSegment(start, end, f"Speaker {label}"))

    # Compute mean embedding per cluster
    cluster_embeddings: dict[str, np.ndarray] = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        speaker_key = f"Speaker {label}"
        mask = labels == label
        cluster_embeddings[speaker_key] = embeddings[mask].mean(axis=0)

    return DiarizationResult(segments=result_segments, cluster_embeddings=cluster_embeddings)


def diarize(
    audio_path: Path,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on an audio file.

    Returns a DiarizationResult with speaker segments and per-cluster embeddings.
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
    log.info("Extracted %d segment embeddings (silent segments skipped)", len(segments))

    log.info("Clustering speakers...")
    result = _cluster_speakers(segments, num_speakers=num_speakers)

    # Count unique speakers
    speakers = {s.speaker for s in result.segments}
    log.info("Identified %d speakers", len(speakers))

    return result


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
