"""Import recordings from external sources (e.g., Hyprnote) into the Meetscribe vault."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from meetscribe.storage.vault import MeetingStorage, slugify

log = logging.getLogger("meetscribe.importer")


@dataclass
class HyprnoteSession:
    session_id: str
    title: str
    created_at: date
    audio_path: Path
    meta_path: Path
    session_dir: Path


def discover_hyprnote_sessions(
    hyprnote_dir: Path | None = None,
) -> list[HyprnoteSession]:
    """Find all Hyprnote sessions with audio files."""
    if hyprnote_dir is None:
        hyprnote_dir = Path.home() / "Library" / "Application Support" / "hyprnote" / "sessions"

    if not hyprnote_dir.exists():
        log.warning("Hyprnote sessions directory not found: %s", hyprnote_dir)
        return []

    sessions: list[HyprnoteSession] = []
    for session_dir in sorted(hyprnote_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        meta_path = session_dir / "_meta.json"
        audio_path = session_dir / "audio.mp3"

        if not meta_path.exists() or not audio_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())
            title = meta.get("title", "").strip()
            created_str = meta.get("created_at", "")
            created_date = datetime.fromisoformat(created_str.replace("Z", "+00:00")).date()

            if not title:
                title = f"untitled-{session_dir.name[:8]}"

            sessions.append(HyprnoteSession(
                session_id=meta.get("id", session_dir.name),
                title=title,
                created_at=created_date,
                audio_path=audio_path,
                meta_path=meta_path,
                session_dir=session_dir,
            ))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning("Skipping session %s: %s", session_dir.name, e)
            continue

    return sessions


def import_session(
    session: HyprnoteSession,
    storage: MeetingStorage,
) -> Path:
    """Import a Hyprnote session into the Meetscribe vault.

    Copies the audio file (as-is, MP3) into the meeting directory.
    Returns the meeting directory path.
    """
    meeting_dir = storage.ensure_meeting_dir(session.title, session.created_at)

    # Copy audio file — keep as MP3 (faster-whisper can transcribe it directly)
    dest_audio = meeting_dir / "recording.mp3"
    if not dest_audio.exists():
        shutil.copy2(session.audio_path, dest_audio)
        log.info("Imported audio: %s -> %s", session.audio_path, dest_audio)
    else:
        log.info("Audio already exists, skipping: %s", dest_audio)

    return meeting_dir


def import_all_hyprnote(storage: MeetingStorage) -> list[Path]:
    """Import all Hyprnote sessions into the vault. Returns list of meeting dirs."""
    sessions = discover_hyprnote_sessions()
    imported: list[Path] = []

    for session in sessions:
        try:
            meeting_dir = import_session(session, storage)
            imported.append(meeting_dir)
            log.info("Imported: %s (%s)", session.title, session.created_at)
        except Exception:
            log.exception("Failed to import session: %s", session.title)

    return imported
