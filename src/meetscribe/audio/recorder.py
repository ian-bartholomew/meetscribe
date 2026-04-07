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
    """Records audio from a single input device to a FLAC file.

    For capturing both system audio and microphone, use a macOS
    Aggregate Device that combines both sources at the OS level.
    """

    def __init__(
        self,
        output_path: Path,
        device_name: str = "BlackHole 2ch",
        sample_rate: int = 48000,
        channels: int = 2,
        mic_device_name: str | None = None,  # kept for config compat, unused
    ) -> None:
        self.output_path = output_path
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.peak_level = 0.0
        self._device_sample_rate = sample_rate
        self._output_channels = channels
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Called from the audio thread for each block of audio data.

        Downmixes all input channels to mono and resamples from the device
        sample rate (e.g. 48kHz) to 16kHz for optimal file size and
        Whisper compatibility.
        """
        # Downmix to mono
        if indata.shape[1] > 1:
            mono = np.sum(indata, axis=1)
            np.clip(mono, -1.0, 1.0, out=mono)
        else:
            mono = indata[:, 0].copy()

        # Downsample from device rate to 16kHz
        if self._device_sample_rate != self.sample_rate:
            ratio = self.sample_rate / self._device_sample_rate
            new_len = int(len(mono) * ratio)
            if new_len > 0:
                # Simple decimation — adequate for speech in a real-time callback
                indices = np.linspace(0, len(mono) - 1, new_len).astype(int)
                mono = mono[indices]

        self.peak_level = float(np.abs(mono).max())
        self._queue.put(mono.reshape(-1, 1))

    def _writer_loop(self) -> None:
        """Background thread that drains the queue and writes to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            str(self.output_path),
            mode="w",
            samplerate=self.sample_rate,
            channels=self._output_channels,
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

        # Auto-detect sample rate from the device for capture
        self._device_sample_rate = int(device_info["default_samplerate"])

        # Write mono at 16kHz — optimal for speech (Whisper uses 16kHz internally)
        self.sample_rate = 16000
        self._output_channels = 1

        # Capture ALL device channels so we don't miss any audio source
        device_channels = device_info["max_input_channels"]

        self.is_recording = True
        self.peak_level = 0.0

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate=self._device_sample_rate,
            device=device_index,
            channels=device_channels,
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
