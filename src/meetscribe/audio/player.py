"""Audio playback with seeking support using sounddevice + soundfile."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

log = logging.getLogger("meetscribe.player")


class AudioPlayer:
    """Play an audio file from any time offset, stoppable from another thread."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._is_playing = False
        self._stop_event = threading.Event()
        self._stream: sd.OutputStream | None = None
        self._sf: sf.SoundFile | None = None
        self._position: float = 0.0  # current position in seconds
        self._sample_rate: int = 0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_position(self) -> float:
        return self._position

    def play(self, offset_seconds: float = 0.0) -> None:
        """Start playback from a time offset in seconds."""
        if self._is_playing:
            self.stop()

        self._stop_event.clear()

        try:
            self._sf = sf.SoundFile(str(self.file_path))
        except Exception:
            log.error("Cannot open audio file: %s", self.file_path)
            return

        self._sample_rate = self._sf.samplerate
        channels = self._sf.channels
        start_frame = int(offset_seconds * self._sample_rate)
        self._sf.seek(start_frame)
        self._position = offset_seconds
        self._is_playing = True

        def callback(outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
            if self._stop_event.is_set():
                outdata[:] = 0
                raise sd.CallbackStop()

            data = self._sf.read(frames, dtype="float32")
            if len(data) == 0:
                outdata[:] = 0
                raise sd.CallbackStop()

            if len(data) < frames:
                outdata[: len(data)] = data
                outdata[len(data) :] = 0
                raise sd.CallbackStop()

            outdata[:] = data
            self._position = self._sf.tell() / self._sample_rate

        def finished() -> None:
            self._is_playing = False
            self._cleanup_stream()

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=channels,
                callback=callback,
                finished_callback=finished,
            )
            self._stream.start()
            log.info("Playback started at %.1fs", offset_seconds)
        except Exception:
            log.exception("Failed to start playback")
            self._is_playing = False
            self._cleanup_stream()

    def stop(self) -> None:
        """Stop playback. Safe to call from any thread."""
        if not self._is_playing:
            return
        self._stop_event.set()
        self._is_playing = False
        self._cleanup_stream()
        log.info("Playback stopped")

    def _cleanup_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
            self._sf = None
