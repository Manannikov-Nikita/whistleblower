import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


@dataclass(frozen=True)
class RecordingStats:
    frames: int
    overflows: int
    duration_sec: float


def list_input_devices() -> List[InputDevice]:
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append(
                InputDevice(
                    index=idx,
                    name=info.get("name", ""),
                    max_input_channels=info.get("max_input_channels", 0),
                    default_samplerate=float(info.get("default_samplerate", 0.0)),
                )
            )
    return devices


def resolve_device_index(device: Optional[str]) -> int:
    if device is None:
        default_device = sd.default.device[0]
        if default_device is None:
            raise ValueError("Default input device is not set.")
        return int(default_device)

    if device.isdigit():
        return int(device)

    matches = [
        d for d in list_input_devices() if device.lower() in d.name.lower()
    ]
    if not matches:
        raise ValueError(f"No input device matches '{device}'.")
    if len(matches) > 1:
        names = ", ".join(f"{d.index}:{d.name}" for d in matches)
        raise ValueError(
            f"Multiple devices match '{device}'. Be specific: {names}"
        )
    return matches[0].index


def _rms_level(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _format_meter(level: float, width: int = 30) -> str:
    level = max(0.0, min(1.0, level))
    filled = int(level * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {level:.3f}"


def probe_input_signal(
    device: Optional[str],
    seconds: float = 5.0,
    samplerate: int = 16000,
    channels: int = 1,
    block_seconds: float = 0.2,
    meter: bool = True,
) -> Tuple[List[float], List[np.ndarray]]:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if samplerate <= 0:
        raise ValueError("samplerate must be > 0")
    if channels <= 0:
        raise ValueError("channels must be > 0")

    device_index = resolve_device_index(device)
    blocksize = max(1, int(block_seconds * samplerate))

    levels: List[float] = []
    chunks: List[np.ndarray] = []

    with sd.InputStream(
        device=device_index,
        channels=channels,
        samplerate=samplerate,
        dtype="float32",
    ) as stream:
        start = time.monotonic()
        while (time.monotonic() - start) < seconds:
            data, overflowed = stream.read(blocksize)
            if overflowed:
                # Minor warning: meter continues, but we flag it in levels.
                pass
            level = _rms_level(data)
            levels.append(level)
            chunks.append(data.copy())
            if meter:
                print(_format_meter(level), flush=True)
    return levels, chunks


def summarize_levels(levels: Iterable[float]) -> Tuple[float, float]:
    levels_list = list(levels)
    if not levels_list:
        return 0.0, 0.0
    return float(np.mean(levels_list)), float(np.max(levels_list))


def record_to_wav(
    device: Optional[str],
    output_path: Path,
    samplerate: int = 16000,
    channels: int = 1,
    block_seconds: float = 0.2,
    stop_event: Optional[threading.Event] = None,
    duration: Optional[float] = None,
    meter: bool = False,
) -> RecordingStats:
    if samplerate <= 0:
        raise ValueError("samplerate must be > 0")
    if channels <= 0:
        raise ValueError("channels must be > 0")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be > 0")

    device_index = resolve_device_index(device)
    blocksize = max(1, int(block_seconds * samplerate))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stop_event is None:
        stop_event = threading.Event()

    frames_written = 0
    overflow_count = 0
    start = time.monotonic()

    try:
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)

            with sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
            ) as stream:
                while not stop_event.is_set():
                    if duration is not None:
                        elapsed = time.monotonic() - start
                        if elapsed >= duration:
                            break
                    data, overflowed = stream.read(blocksize)
                    if overflowed:
                        overflow_count += 1
                    level = _rms_level(data)
                    if meter:
                        print(_format_meter(level), flush=True)
                    clipped = np.clip(data, -1.0, 1.0)
                    pcm16 = (clipped * 32767).astype(np.int16)
                    wf.writeframes(pcm16.tobytes())
                    frames_written += pcm16.shape[0]
    except KeyboardInterrupt:
        pass

    duration_sec = frames_written / float(samplerate) if samplerate else 0.0
    return RecordingStats(
        frames=frames_written,
        overflows=overflow_count,
        duration_sec=duration_sec,
    )
