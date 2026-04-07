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

    When mic_device_name is set, records from both the system audio device
    and the microphone, mixing them into a single stereo file.
    """

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
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        self._mic_stream: sd.InputStream | None = None
        self._mic_buffer: np.ndarray | None = None
        self._mic_lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called from the audio thread for each block of system audio data."""
        mixed = indata.copy()

        # Mix in microphone audio if available
        with self._mic_lock:
            if self._mic_buffer is not None:
                mic = self._mic_buffer
                self._mic_buffer = None
                # Ensure same shape — pad or truncate mic to match system frames
                if mic.shape[0] < mixed.shape[0]:
                    mic = np.pad(mic, ((0, mixed.shape[0] - mic.shape[0]), (0, 0)))
                elif mic.shape[0] > mixed.shape[0]:
                    mic = mic[:mixed.shape[0]]
                # Ensure same number of channels
                if mic.shape[1] < mixed.shape[1]:
                    mic = np.repeat(mic, mixed.shape[1], axis=1)
                elif mic.shape[1] > mixed.shape[1]:
                    mic = mic[:, :mixed.shape[1]]
                mixed = mixed + mic

        # Clip to prevent distortion
        np.clip(mixed, -1.0, 1.0, out=mixed)
        self.peak_level = float(np.abs(mixed).max())
        self._queue.put(mixed)

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called from the mic audio thread for each block."""
        with self._mic_lock:
            self._mic_buffer = indata.copy()

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

        device_info = get_device_info(self.device_name)
        device_index = device_info["index"]

        # Auto-detect sample rate from the device
        self.sample_rate = int(device_info["default_samplerate"])
        self.channels = min(self.channels, device_info["max_input_channels"])

        self.is_recording = True
        self.peak_level = 0.0

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        # Start mic stream first if configured
        if self.mic_device_name:
            try:
                mic_info = get_device_info(self.mic_device_name)
                self._mic_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    device=mic_info["index"],
                    channels=min(self.channels, mic_info["max_input_channels"]),
                    callback=self._mic_callback,
                )
                self._mic_stream.start()
            except (ValueError, sd.PortAudioError):
                self._mic_stream = None  # Proceed without mic

        # Start main system audio stream
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

        if self._mic_stream is not None:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None

        self._queue.put(None)  # Signal writer thread to finish
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.peak_level = 0.0
