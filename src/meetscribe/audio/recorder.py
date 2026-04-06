from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def find_device(name: str) -> int:
    """Find an audio device index by name. Raises ValueError if not found."""
    devices = sd.query_devices()
    for dev in devices:
        if dev["name"] == name and dev["max_input_channels"] > 0:
            return dev["index"]
    raise ValueError(
        f"Audio device '{name}' not found. "
        f"Available input devices: {[d['name'] for d in devices if d['max_input_channels'] > 0]}"
    )


class AudioRecorder:
    """Records system audio to a FLAC file using sounddevice and soundfile."""

    def __init__(
        self,
        output_path: Path,
        device_name: str = "BlackHole 2ch",
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> None:
        self.output_path = output_path
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.peak_level = 0.0
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called from the audio thread for each block of audio data."""
        self.peak_level = float(np.abs(indata).max())
        self._queue.put(indata.copy())

    def _writer_loop(self) -> None:
        """Background thread that drains the queue and writes to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            str(self.output_path),
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            format="FLAC",
        ) as f:
            while True:
                data = self._queue.get()
                if data is None:
                    break
                f.write(data)

    def start(self) -> None:
        """Begin recording."""
        if self.is_recording:
            return
        device_index = find_device(self.device_name)
        self.is_recording = True
        self.peak_level = 0.0

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=device_index,
            channels=self.channels,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop recording and finalize the file."""
        if not self.is_recording:
            return
        self.is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._queue.put(None)  # Signal writer thread to finish
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.peak_level = 0.0
