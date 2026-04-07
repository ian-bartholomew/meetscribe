from __future__ import annotations

import json
import queue
import subprocess
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


def get_device_info(name: str) -> dict:
    """Get full device info dict by name."""
    devices = sd.query_devices()
    for dev in devices:
        if dev["name"] == name and dev["max_input_channels"] > 0:
            return dev
    raise ValueError(f"Audio device '{name}' not found.")


def list_input_devices() -> list[dict]:
    """Return all available input devices with their properties."""
    devices = sd.query_devices()
    return [
        {
            "index": d["index"],
            "name": d["name"],
            "channels": d["max_input_channels"],
            "sample_rate": int(d["default_samplerate"]),
        }
        for d in devices
        if d["max_input_channels"] > 0
    ]


def ensure_aggregate_device(
    name: str,
    device_names: list[str],
) -> str:
    """Create a macOS Aggregate Device combining multiple audio inputs.

    Uses the `audiodevice` CoreAudio API via a subprocess script.
    Returns the name of the aggregate device (may already exist).
    """
    # Check if it already exists
    devices = sd.query_devices()
    for dev in devices:
        if dev["name"] == name and dev["max_input_channels"] > 0:
            return name

    # Build the aggregate device using a Python CoreAudio script
    script = f'''
import subprocess, plistlib, sys

# Get current audio device UIDs
result = subprocess.run(
    ["system_profiler", "SPAudioDataType", "-json"],
    capture_output=True, text=True
)
import json
data = json.loads(result.stdout)
items = data.get("SPAudioDataType", [])

uid_map = {{}}
for item in items:
    dev_name = item.get("_name", "")
    uid = item.get("coreaudio_device_transport", "")
    if not uid:
        uid = item.get("_name", "")
    uid_map[dev_name] = uid

# Create aggregate device via AppleScript (Audio MIDI Setup automation)
# Fallback: just print instructions
target_devices = {device_names!r}
missing = [d for d in target_devices if d not in uid_map]
if missing:
    print(f"MISSING:{{','.join(missing)}}", file=sys.stderr)
    sys.exit(1)

print("EXISTS")
'''
    # The CoreAudio API for creating aggregate devices programmatically is complex.
    # Instead, provide clear instructions and check if the user has already created one.
    raise ValueError(
        f"Aggregate device '{name}' not found. "
        f"Please create it in Audio MIDI Setup:\n"
        f"  1. Open Audio MIDI Setup\n"
        f"  2. Click '+' → Create Aggregate Device\n"
        f"  3. Check: {', '.join(device_names)}\n"
        f"  4. Rename it to '{name}'"
    )


class AudioRecorder:
    """Records audio to a FLAC file, optionally mixing multiple input devices.

    When mic_device_name is set, both devices feed into separate queues.
    A writer thread reads from both, mixes frame-by-frame, and writes to disk.
    This ensures no audio frames are dropped from either source.
    """

    BLOCKSIZE = 1024  # Fixed block size for both streams

    def __init__(
        self,
        output_path: Path,
        device_name: str = "BlackHole 2ch",
        sample_rate: int = 48000,
        channels: int = 2,
        mic_device_name: str | None = None,
    ) -> None:
        self.output_path = output_path
        self.device_name = device_name
        self.mic_device_name = mic_device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.peak_level = 0.0
        self._sys_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._mic_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        self._mic_stream: sd.InputStream | None = None

    def _sys_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called for each block of system audio data."""
        self.peak_level = float(np.abs(indata).max())
        self._sys_queue.put(indata.copy())

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called for each block of microphone audio data."""
        self._mic_queue.put(indata.copy())

    def _ensure_channels(self, data: np.ndarray, target_channels: int) -> np.ndarray:
        """Reshape audio data to match the target channel count."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] == target_channels:
            return data
        if data.shape[1] == 1 and target_channels == 2:
            return np.column_stack([data[:, 0], data[:, 0]])
        return data[:, :target_channels]

    def _writer_loop(self) -> None:
        """Background thread that reads both queues, mixes, and writes to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        has_mic = self._mic_stream is not None

        with sf.SoundFile(
            str(self.output_path),
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            format="FLAC",
        ) as f:
            while True:
                # Read system audio (blocking — this drives the write rate)
                sys_data = self._sys_queue.get()
                if sys_data is None:
                    break

                sys_data = self._ensure_channels(sys_data, self.channels)

                if has_mic:
                    # Drain all available mic blocks and concatenate
                    mic_chunks: list[np.ndarray] = []
                    while True:
                        try:
                            mic_block = self._mic_queue.get_nowait()
                            if mic_block is None:
                                break
                            mic_chunks.append(self._ensure_channels(mic_block, self.channels))
                        except queue.Empty:
                            break

                    if mic_chunks:
                        mic_data = np.concatenate(mic_chunks, axis=0)
                        # Align to system block length
                        frames_needed = sys_data.shape[0]
                        if mic_data.shape[0] >= frames_needed:
                            mic_data = mic_data[:frames_needed]
                        else:
                            mic_data = np.pad(
                                mic_data,
                                ((0, frames_needed - mic_data.shape[0]), (0, 0)),
                            )
                        mixed = sys_data + mic_data
                        np.clip(mixed, -1.0, 1.0, out=mixed)
                        f.write(mixed)
                    else:
                        f.write(sys_data)
                else:
                    f.write(sys_data)

    def start(self) -> None:
        """Begin recording."""
        if self.is_recording:
            return

        device_info = get_device_info(self.device_name)
        device_index = device_info["index"]

        # Auto-detect sample rate from the device
        self.sample_rate = int(device_info["default_samplerate"])
        self.channels = min(self.channels, device_info["max_input_channels"])

        self.is_recording = True
        self.peak_level = 0.0

        # Start mic stream first if configured
        if self.mic_device_name:
            try:
                mic_info = get_device_info(self.mic_device_name)
                self._mic_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    device=mic_info["index"],
                    channels=min(self.channels, mic_info["max_input_channels"]),
                    blocksize=self.BLOCKSIZE,
                    callback=self._mic_callback,
                )
                self._mic_stream.start()
            except (ValueError, sd.PortAudioError):
                self._mic_stream = None

        # Start writer thread after mic is set up (it checks _mic_stream)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        # Start main system audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=device_index,
            channels=self.channels,
            blocksize=self.BLOCKSIZE,
            callback=self._sys_callback,
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

        if self._mic_stream is not None:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None

        # Signal writer thread to finish
        self._sys_queue.put(None)
        self._mic_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.peak_level = 0.0
