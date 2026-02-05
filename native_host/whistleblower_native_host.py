#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _resolve_inbox_dir() -> Optional[Path]:
    value = os.getenv("INBOX_DIR") or os.getenv("WHISTLEBLOWER_INBOX_DIR")
    if not value:
        return None
    return Path(value).expanduser()

def _output_dir() -> Path:
    return Path(os.getenv("OUTPUT_DIR", str(_repo_root() / "output"))).expanduser()

def _log_path() -> Path:
    return _output_dir() / "native_host.log"


def _log(message: str) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool
    asr_model: str
    asr_window_sec: float
    asr_stride_sec: float
    asr_overlap_sec: float
    diarize_window_sec: float
    diarize_stride_sec: float
    device: str
    poll_ms: int
    language: str


@dataclass
class LiveChunk:
    index: int
    path: Path
    wav_path: Path
    start_sec: float
    duration_sec: float

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


def _load_live_config() -> LiveConfig:
    return LiveConfig(
        enabled=_env_bool("LIVE_PROCESSING", True),
        asr_model=os.getenv("LIVE_ASR_MODEL", "small"),
        asr_window_sec=_env_float("LIVE_ASR_WINDOW_SEC", 30.0),
        asr_stride_sec=_env_float("LIVE_ASR_STRIDE_SEC", 10.0),
        asr_overlap_sec=_env_float("LIVE_ASR_OVERLAP_SEC", 5.0),
        diarize_window_sec=_env_float("LIVE_DIARIZE_WINDOW_SEC", 120.0),
        diarize_stride_sec=_env_float("LIVE_DIARIZE_STRIDE_SEC", 60.0),
        device=os.getenv("LIVE_DEVICE", "auto"),
        poll_ms=_env_int("LIVE_POLL_MS", 500),
        language=os.getenv("WHISPER_LANGUAGE", "ru"),
    )


def _default_session_name() -> str:
    return datetime.now().strftime("session-%Y%m%d-%H%M%S")


def _read_message() -> Optional[dict]:
    try:
        raw_len = sys.stdin.buffer.read(4)
        if not raw_len:
            return None
        message_len = struct.unpack("<I", raw_len)[0]
        raw = sys.stdin.buffer.read(message_len)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log(f"Failed to read message: {exc}")
        return None


def _send_message(payload: dict) -> None:
    try:
        raw = json.dumps(payload).encode("utf-8")
        sys.stdout.buffer.write(struct.pack("<I", len(raw)))
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
    except Exception as exc:  # noqa: BLE001
        _log(f"Failed to send message: {exc}")


def _check_audio_access(audio_path: Path) -> Optional[str]:
    try:
        with audio_path.open("rb") as handle:
            handle.read(1)
    except PermissionError:
        python_path = sys.executable
        return (
            "Нет доступа к файлу записи. Разрешите доступ в macOS → "
            "Privacy & Security → Full Disk Access для: "
            f"{python_path}."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Не удалось прочитать файл записи: {exc}"
    return None


def _ensure_ffmpeg_on_path() -> Optional[str]:
    ffmpeg = (
        os.getenv("FFMPEG_PATH")
        or shutil.which("ffmpeg")
        or ("/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None)
        or ("/usr/local/bin/ffmpeg" if Path("/usr/local/bin/ffmpeg").exists() else None)
    )
    if not ffmpeg:
        return None
    ffmpeg_dir = str(Path(ffmpeg).parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path.split(":"):
        os.environ["PATH"] = f"{ffmpeg_dir}:{current_path}"
    return ffmpeg


def _spawn_worker(audio_path: Path, session_dir: Optional[Path]) -> subprocess.Popen:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root()

    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "native_host.log"

    cmd = [sys.executable, str(script_path), "--worker", "--audio", str(audio_path)]
    if session_dir:
        cmd += ["--session-dir", str(session_dir)]

    log_handle = log_path.open("a", encoding="utf-8")
    _log(f"Spawning worker: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=log_handle,
        stderr=log_handle,
        env=os.environ.copy(),
    )
    log_handle.close()
    return process


def _spawn_live_worker(session_dir: Path, manifest_path: Path) -> subprocess.Popen:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root()

    live_dir = session_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    log_path = live_dir / "live.log"

    cmd = [
        sys.executable,
        str(script_path),
        "--live-worker",
        "--session-dir",
        str(session_dir),
        "--manifest",
        str(manifest_path),
    ]

    log_handle = log_path.open("a", encoding="utf-8")
    _log(f"Spawning live worker: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=log_handle,
        stderr=log_handle,
        env=os.environ.copy(),
    )
    log_handle.close()
    return process


def _init_live_state(session_dir: Path) -> dict:
    live_dir = session_dir / "live"
    chunks_dir = live_dir / "chunks"
    live_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks_dir / "manifest.jsonl"
    manifest_handle = manifest_path.open("a", encoding="utf-8")
    return {
        "enabled": True,
        "dir": live_dir,
        "chunks_dir": chunks_dir,
        "manifest_path": manifest_path,
        "manifest_handle": manifest_handle,
        "chunk_index": 0,
        "current_blob_id": None,
        "current_blob_timecode_ms": None,
        "current_blob_bytes": bytearray(),
        "session_dir": session_dir,
    }


def _write_manifest_entry(live_state: dict, entry: dict) -> None:
    handle = live_state.get("manifest_handle")
    if not handle:
        return
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    handle.flush()


def _finalize_live_blob(live_state: dict) -> None:
    if not live_state or not live_state.get("enabled"):
        return
    blob_bytes = live_state.get("current_blob_bytes")
    if not blob_bytes:
        return
    chunk_index = live_state.get("chunk_index", 0)
    live_state["chunk_index"] = chunk_index + 1

    chunks_dir = live_state["chunks_dir"]
    chunk_path = chunks_dir / f"chunk-{chunk_index:06d}.webm"
    chunk_path.write_bytes(blob_bytes)

    start_sec = None
    timecode_ms = live_state.get("current_blob_timecode_ms")
    if timecode_ms is not None:
        try:
            start_sec = float(timecode_ms) / 1000.0
        except (TypeError, ValueError):
            start_sec = None

    session_dir = live_state["session_dir"]
    entry = {
        "index": chunk_index,
        "path": chunk_path.relative_to(session_dir).as_posix(),
        "start_sec": start_sec,
        "end_sec": None,
    }
    _write_manifest_entry(live_state, entry)

    live_state["current_blob_id"] = None
    live_state["current_blob_timecode_ms"] = None
    live_state["current_blob_bytes"] = bytearray()


def _append_live_bytes(
    live_state: dict,
    blob_id: Optional[str],
    is_last: Optional[bool],
    timecode_ms: Optional[float],
    raw: bytes,
) -> None:
    if not live_state or not live_state.get("enabled"):
        return

    if not blob_id:
        live_state["current_blob_id"] = None
        live_state["current_blob_timecode_ms"] = timecode_ms
        live_state["current_blob_bytes"] = bytearray(raw)
        _finalize_live_blob(live_state)
        return

    current_id = live_state.get("current_blob_id")
    if current_id != blob_id:
        if live_state.get("current_blob_bytes"):
            _finalize_live_blob(live_state)
        live_state["current_blob_id"] = blob_id
        live_state["current_blob_timecode_ms"] = timecode_ms
        live_state["current_blob_bytes"] = bytearray()

    live_state["current_blob_bytes"].extend(raw)
    if is_last:
        _finalize_live_blob(live_state)


def run_host() -> None:
    _log("Native host started.")
    sessions: dict[str, dict] = {}
    while True:
        message = _read_message()
        if message is None:
            _log("No message received; exiting.")
            return

        msg_type = message.get("type")
        if msg_type == "ping":
            _send_message({"ok": True, "type": "pong"})
            continue

        if msg_type == "stream_start":
            session_id = message.get("session_id")
            if not session_id:
                _send_message({"ok": False, "error": "missing session_id"})
                continue
            if session_id in sessions:
                _send_message({"ok": False, "error": "session already exists"})
                continue

            mime_type = message.get("mime_type") or "audio/webm"
            extension = ".ogg" if "ogg" in mime_type else ".webm"
            output_dir = _output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            session_dir = output_dir / _default_session_name()
            session_dir.mkdir(parents=True, exist_ok=True)
            audio_path = session_dir / f"audio{extension}"
            handle = audio_path.open("ab")
            live_state = None
            live_config = _load_live_config()
            if live_config.enabled:
                try:
                    live_state = _init_live_state(session_dir)
                    live_state["process"] = _spawn_live_worker(
                        session_dir, live_state["manifest_path"]
                    )
                except Exception as exc:  # noqa: BLE001
                    _log(f"Failed to start live worker: {exc}")
                    live_state = None
            sessions[session_id] = {
                "path": audio_path,
                "handle": handle,
                "session_dir": session_dir,
                "bytes": 0,
                "live": live_state,
            }
            _send_message({"ok": True, "session_dir": str(session_dir)})
            continue

        if msg_type == "stream_chunk":
            session_id = message.get("session_id")
            data = message.get("data")
            if not session_id or not data:
                _send_message({"ok": False, "error": "missing session_id or data"})
                continue
            session = sessions.get(session_id)
            if not session:
                _send_message({"ok": False, "error": "unknown session"})
                continue
            try:
                raw = base64.b64decode(data)
            except Exception as exc:  # noqa: BLE001
                _send_message({"ok": False, "error": f"decode error: {exc}"})
                continue
            handle = session["handle"]
            handle.write(raw)
            session["bytes"] += len(raw)
            _append_live_bytes(
                session.get("live"),
                message.get("blob_id"),
                message.get("is_last"),
                message.get("timecode_ms"),
                raw,
            )
            _send_message({"ok": True})
            continue

        if msg_type == "stream_stop":
            session_id = message.get("session_id")
            if not session_id:
                _send_message({"ok": False, "error": "missing session_id"})
                continue
            session = sessions.pop(session_id, None)
            if not session:
                _send_message({"ok": False, "error": "unknown session"})
                continue
            live_state = session.get("live")
            if live_state:
                _finalize_live_blob(live_state)
                _write_manifest_entry(live_state, {"type": "eos"})
                manifest_handle = live_state.get("manifest_handle")
                if manifest_handle:
                    manifest_handle.close()
            handle = session["handle"]
            handle.close()
            audio_path = session["path"]
            session_dir = session["session_dir"]
            if session.get("bytes", 0) == 0:
                _send_message({"ok": False, "error": "empty recording"})
                continue
            try:
                process = _spawn_worker(audio_path, session_dir)
            except Exception as exc:  # noqa: BLE001
                _log(f"Failed to spawn worker: {exc}")
                _send_message({"ok": False, "error": str(exc)})
                continue
            _send_message({"ok": True, "pid": process.pid})
            continue

        if msg_type == "recording_saved":
            path_value = message.get("path")
            if not path_value:
                _send_message({"ok": False, "error": "missing path"})
                continue

            audio_path = Path(path_value).expanduser()
            if not audio_path.exists():
                _send_message({"ok": False, "error": "file not found"})
                continue

            access_error = _check_audio_access(audio_path)
            if access_error:
                _log(access_error)
                _send_message({"ok": False, "error": access_error})
                continue

            session_dir_value = message.get("session_dir")
            session_dir = Path(session_dir_value) if session_dir_value else None

            try:
                process = _spawn_worker(audio_path, session_dir)
            except Exception as exc:  # noqa: BLE001
                _log(f"Failed to spawn worker: {exc}")
                _send_message({"ok": False, "error": str(exc)})
                continue

            _send_message({"ok": True, "pid": process.pid})
            continue

        _send_message({"ok": False, "error": "unknown message type"})


def _convert_to_wav(source: Path, target: Path) -> None:
    ffmpeg = _ensure_ffmpeg_on_path()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it (brew install ffmpeg) or set FFMPEG_PATH."
        )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def _run_pipeline(audio_path: Path, session_dir: Path) -> None:
    from whistleblower.diarize import DiarizationError, run_diarization
    from whistleblower.merge_transcript import MergeError, merge_transcript
    from whistleblower.summarize import SummarizationError, summarize_text
    from whistleblower.transcribe import TranscriptionError, transcribe_audio

    transcript_path = session_dir / "transcript.txt"
    segments_path = session_dir / "transcript_segments.json"
    diarization_dir = session_dir / "diarization"
    speaker_path = session_dir / "speaker_transcript.md"
    summary_path = session_dir / "summary.md"

    transcribe_audio(
        audio_path=audio_path,
        model_name=os.getenv("WHISPER_MODEL", "medium"),
        language=os.getenv("WHISPER_LANGUAGE", "ru"),
        output_path=transcript_path,
        output_segments_path=segments_path,
    )
    print(f"Transcript saved to {transcript_path}")
    print(f"Transcript segments saved to {segments_path}")

    run_diarization(
        audio_path=audio_path,
        output_dir=diarization_dir,
        hf_token=None,
        min_speakers=None,
        max_speakers=None,
        model="pyannote/speaker-diarization-3.1",
        device=None,
    )
    print(f"Diarization saved to {diarization_dir}")

    merge_transcript(
        diarization_segments_path=diarization_dir / "segments.json",
        transcript_segments_path=segments_path,
        output_path=speaker_path,
    )
    print(f"Speaker transcript saved to {speaker_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        summarize_text(
            input_text=speaker_path.read_text(encoding="utf-8"),
            output_path=summary_path,
        )
        print(f"Summary saved to {summary_path}")
    else:
        print("OPENAI_API_KEY is not set; skipping summary.")


def run_worker(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    _load_dotenv(repo_root / ".env")
    _ensure_ffmpeg_on_path()
    from whistleblower.diarize import DiarizationError
    from whistleblower.merge_transcript import MergeError
    from whistleblower.summarize import SummarizationError
    from whistleblower.transcribe import TranscriptionError

    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dir = Path(args.session_dir) if args.session_dir else output_dir / _default_session_name()
    session_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.audio).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    dest = session_dir / f"audio{source.suffix.lower()}"
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    else:
        dest = source

    audio_for_processing = dest
    if dest.suffix.lower() != ".wav":
        wav_path = session_dir / "audio.wav"
        _convert_to_wav(dest, wav_path)
        audio_for_processing = wav_path

    print(f"Processing audio: {audio_for_processing}")
    try:
        _run_pipeline(audio_for_processing, session_dir)
    except (TranscriptionError, DiarizationError, MergeError, SummarizationError) as exc:
        print(str(exc))
        raise

    keep_raw = os.getenv("KEEP_RAW_AUDIO", "false").lower() in ("1", "true", "yes")
    if not keep_raw:
        for path in {dest, audio_for_processing}:
            if path.exists() and path.suffix.lower() in (".webm", ".ogg", ".wav"):
                try:
                    path.unlink()
                except OSError:
                    pass


def _live_log(handle, message: str) -> None:
    if not handle:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"[{timestamp}] {message}\n")
    handle.flush()


def _read_manifest_entries(handle, offset: int) -> tuple[list[dict], int]:
    handle.seek(offset)
    lines = handle.readlines()
    new_offset = handle.tell()
    entries: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries, new_offset


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if not rate:
        return 0.0
    return float(frames) / float(rate)


def _select_live_device(config: LiveConfig) -> str:
    if config.device != "auto":
        return config.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _apply_cpu_limits() -> None:
    cpu_count = os.cpu_count() or 4
    threads = max(1, min(4, cpu_count // 2))
    if os.getenv("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    if os.getenv("MKL_NUM_THREADS") is None:
        os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


def _init_live_asr_backend(config: LiveConfig, device: str, log_handle) -> Optional[tuple]:
    if device == "mps":
        try:
            import whisper

            model = whisper.load_model(config.asr_model, device=device)
            _live_log(log_handle, "Live ASR: using openai-whisper on mps.")
            return ("whisper", model, device)
        except Exception as exc:  # noqa: BLE001
            _live_log(log_handle, f"Live ASR disabled: {exc}")
            return None
    try:
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        model = WhisperModel(config.asr_model, device=device, compute_type=compute_type)
        _live_log(log_handle, "Live ASR: using faster-whisper.")
        return ("faster", model, device)
    except Exception as exc:  # noqa: BLE001
        _live_log(log_handle, f"Live ASR fallback to openai-whisper: {exc}")
        try:
            import whisper

            model = whisper.load_model(config.asr_model, device=device)
            _live_log(log_handle, "Live ASR: using openai-whisper.")
            return ("whisper", model, device)
        except Exception as exc2:  # noqa: BLE001
            _live_log(log_handle, f"Live ASR disabled: {exc2}")
            return None


def _transcribe_window(asr_backend: tuple, wav_path: Path, language: str) -> list[dict]:
    backend, model, device = asr_backend
    if backend == "faster":
        segments, _info = model.transcribe(
            str(wav_path),
            language=language,
            condition_on_previous_text=True,
        )
        output: list[dict] = []
        for segment in segments:
            text = (segment.text or "").strip()
            output.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                }
            )
        return output
    result = model.transcribe(
        str(wav_path),
        language=language,
        fp16=device != "cpu",
        verbose=False,
    )
    segments = result.get("segments", []) or []
    output = []
    for segment in segments:
        output.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": (segment.get("text", "") or "").strip(),
            }
        )
    return output


def _select_window_chunks(chunks: list[LiveChunk], window_start: float, window_end: float) -> list[LiveChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.end_sec > window_start and chunk.start_sec < window_end
    ]


def _render_window_wav(
    chunks: list[LiveChunk],
    window_start: float,
    window_end: float,
    target_path: Path,
    ffmpeg: str,
) -> None:
    if not chunks:
        raise RuntimeError("No chunks available to render window.")
    list_path = target_path.with_suffix(".list")
    lines = [f"file '{chunk.wav_path.as_posix()}'" for chunk in chunks]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    concat_start = min(chunk.start_sec for chunk in chunks)
    start_offset = max(0.0, window_start - concat_start)
    duration = max(0.0, window_end - window_start)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-af",
        f"atrim=start={start_offset}:duration={duration}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        list_path.unlink()
    except OSError:
        pass


def _write_live_transcript(live_dir: Path, segments: list[dict]) -> None:
    segments_path = live_dir / "live_transcript_segments.json"
    text_path = live_dir / "live_transcript.txt"
    payload = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in segments
    ]
    segments_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [seg["text"] for seg in segments if seg["text"]]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _init_live_diarization_backend(config: LiveConfig, log_handle):
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        _live_log(log_handle, "Live diarization disabled: missing Hugging Face token.")
        return None
    try:
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import (
            Problem,
            Resolution,
            Specifications,
            Task,
            TrainDataset,
            UnknownSpecificationsError,
            ValDataset,
        )
    except ModuleNotFoundError as exc:
        _live_log(log_handle, f"Live diarization disabled: {exc}")
        return None

    try:
        torch.serialization.add_safe_globals(
            [
                torch.torch_version.TorchVersion,
                Problem,
                Resolution,
                Specifications,
                Task,
                TrainDataset,
                UnknownSpecificationsError,
                ValDataset,
            ]
        )
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except Exception as exc:  # noqa: BLE001
        _live_log(log_handle, f"Live diarization disabled: {exc}")
        return None
    device = _select_live_device(config)
    if device == "mps":
        device = "cpu"
    if device:
        try:
            pipeline.to(device)
        except Exception:
            pass
    _live_log(log_handle, "Live diarization: pipeline ready.")
    return {"pipeline": pipeline}


def _map_speakers(
    window_segments: list[dict],
    global_segments: list[dict],
    overlap_start: float,
    overlap_end: float,
    next_speaker_id: int,
) -> tuple[dict, int]:
    local_labels = sorted({seg["speaker"] for seg in window_segments})
    if not global_segments or overlap_start >= overlap_end:
        mapping = {}
        for label in local_labels:
            mapping[label] = f"SPEAKER_{next_speaker_id:02d}"
            next_speaker_id += 1
        return mapping, next_speaker_id

    global_labels = sorted({seg["speaker"] for seg in global_segments})
    scores: list[tuple[float, str, str]] = []
    for local in local_labels:
        for global_label in global_labels:
            total = 0.0
            for seg in window_segments:
                if seg["speaker"] != local:
                    continue
                for gseg in global_segments:
                    if gseg["speaker"] != global_label:
                        continue
                    ov = _overlap(
                        max(seg["start"], overlap_start),
                        min(seg["end"], overlap_end),
                        max(gseg["start"], overlap_start),
                        min(gseg["end"], overlap_end),
                    )
                    total += ov
            if total > 0:
                scores.append((total, local, global_label))
    scores.sort(reverse=True)

    mapping: dict[str, str] = {}
    used_globals: set[str] = set()
    for _score, local, global_label in scores:
        if local in mapping or global_label in used_globals:
            continue
        mapping[local] = global_label
        used_globals.add(global_label)

    for label in local_labels:
        if label not in mapping:
            mapping[label] = f"SPEAKER_{next_speaker_id:02d}"
            next_speaker_id += 1
    return mapping, next_speaker_id


def _write_live_diarization(live_dir: Path, segments: list[dict], session_id: str) -> None:
    diar_dir = live_dir / "live_diarization"
    diar_dir.mkdir(parents=True, exist_ok=True)
    segments_path = diar_dir / "segments.json"
    payload = [
        {
            "speaker": seg["speaker"],
            "start": seg["start"],
            "duration": seg["duration"],
        }
        for seg in segments
    ]
    segments_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rttm_path = live_dir / "live_diarization.rttm"
    with rttm_path.open("w", encoding="utf-8") as handle:
        for seg in segments:
            handle.write(
                "SPEAKER "
                f"{session_id} 1 {seg['start']:.3f} {seg['duration']:.3f} "
                "<NA> <NA> "
                f"{seg['speaker']} <NA> <NA>\n"
            )


def _run_live_asr(
    config: LiveConfig,
    live_dir: Path,
    chunks: list[LiveChunk],
    asr_backend: tuple,
    asr_next_start: Optional[float],
    last_committed_time: float,
    segments: list[dict],
    log_handle,
    ffmpeg: str,
) -> tuple[float, float, list[dict]]:
    if not chunks:
        return asr_next_start or 0.0, last_committed_time, segments

    if asr_next_start is None:
        asr_next_start = chunks[0].start_sec

    latest_time = max(chunk.end_sec for chunk in chunks)
    window_sec = max(1.0, config.asr_window_sec)
    stride = max(1.0, config.asr_stride_sec)
    overlap = min(max(0.0, config.asr_overlap_sec), window_sec)
    updated = False

    while asr_next_start + window_sec <= latest_time:
        window_start = asr_next_start
        window_end = window_start + window_sec
        window_chunks = _select_window_chunks(chunks, window_start, window_end)
        if not window_chunks:
            asr_next_start += stride
            continue
        window_path = live_dir / "live_window.wav"
        try:
            _render_window_wav(window_chunks, window_start, window_end, window_path, ffmpeg)
            window_segments = _transcribe_window(asr_backend, window_path, config.language)
        except Exception as exc:  # noqa: BLE001
            _live_log(log_handle, f"Live ASR window failed: {exc}")
            asr_next_start += stride
            continue

        commit_until = max(window_start, window_end - overlap)
        for seg in window_segments:
            abs_start = seg["start"] + window_start
            abs_end = seg["end"] + window_start
            if abs_end <= last_committed_time or abs_end > commit_until:
                continue
            text = seg["text"].strip()
            if not text:
                continue
            segments.append({"start": abs_start, "end": abs_end, "text": text})
            last_committed_time = max(last_committed_time, abs_end)
            updated = True
        asr_next_start += stride

    if updated:
        segments.sort(key=lambda item: item["start"])
        _write_live_transcript(live_dir, segments)
    return asr_next_start, last_committed_time, segments


def _run_live_diarization(
    config: LiveConfig,
    live_dir: Path,
    chunks: list[LiveChunk],
    diar_backend: dict,
    diar_next_start: Optional[float],
    last_committed_time: float,
    global_segments: list[dict],
    next_speaker_id: int,
    log_handle,
    ffmpeg: str,
    session_id: str,
) -> tuple[float, float, list[dict], int]:
    if not chunks:
        return diar_next_start or 0.0, last_committed_time, global_segments, next_speaker_id

    if diar_next_start is None:
        diar_next_start = chunks[0].start_sec

    latest_time = max(chunk.end_sec for chunk in chunks)
    window_sec = max(5.0, config.diarize_window_sec)
    stride = max(1.0, config.diarize_stride_sec)
    overlap_sec = max(0.0, window_sec - stride)
    updated = False

    pipeline = diar_backend["pipeline"]

    while diar_next_start + window_sec <= latest_time:
        window_start = diar_next_start
        window_end = window_start + window_sec
        window_chunks = _select_window_chunks(chunks, window_start, window_end)
        if not window_chunks:
            diar_next_start += stride
            continue
        window_path = live_dir / "live_diar_window.wav"
        try:
            _render_window_wav(window_chunks, window_start, window_end, window_path, ffmpeg)
            diarization = pipeline(str(window_path))
        except Exception as exc:  # noqa: BLE001
            _live_log(log_handle, f"Live diarization window failed: {exc}")
            diar_next_start += stride
            continue

        window_segments: list[dict] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            window_segments.append(
                {
                    "speaker": speaker,
                    "start": float(segment.start) + window_start,
                    "end": float(segment.end) + window_start,
                }
            )

        overlap_start = max(window_start, window_end - overlap_sec)
        overlap_end = window_end
        mapping, next_speaker_id = _map_speakers(
            window_segments,
            global_segments,
            overlap_start,
            overlap_end,
            next_speaker_id,
        )

        commit_until = max(window_start, window_end - overlap_sec)
        for seg in window_segments:
            abs_start = seg["start"]
            abs_end = seg["end"]
            if abs_end <= last_committed_time or abs_end > commit_until:
                continue
            speaker = mapping.get(seg["speaker"], seg["speaker"])
            duration = max(0.0, abs_end - abs_start)
            if duration <= 0.0:
                continue
            global_segments.append(
                {
                    "speaker": speaker,
                    "start": abs_start,
                    "end": abs_end,
                    "duration": duration,
                }
            )
            last_committed_time = max(last_committed_time, abs_end)
            updated = True

        diar_next_start += stride

    if updated:
        global_segments.sort(key=lambda item: item["start"])
        _write_live_diarization(live_dir, global_segments, session_id)
    return diar_next_start, last_committed_time, global_segments, next_speaker_id


def _run_live_merge(live_dir: Path, log_handle) -> None:
    try:
        from whistleblower.merge_transcript import MergeError, merge_transcript
    except ModuleNotFoundError as exc:
        _live_log(log_handle, f"Live merge disabled: {exc}")
        return
    diar_segments = live_dir / "live_diarization" / "segments.json"
    transcript_segments = live_dir / "live_transcript_segments.json"
    output_path = live_dir / "speaker_transcript.live.md"
    if not diar_segments.exists() or not transcript_segments.exists():
        return
    try:
        merge_transcript(
            diarization_segments_path=diar_segments,
            transcript_segments_path=transcript_segments,
            output_path=output_path,
        )
    except MergeError as exc:
        _live_log(log_handle, f"Live merge failed: {exc}")


def run_live_worker(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    _load_dotenv(repo_root / ".env")
    config = _load_live_config()
    if not config.enabled:
        return

    session_dir = Path(args.session_dir).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    live_dir = session_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    chunks_wav_dir = live_dir / "chunks_wav"
    chunks_wav_dir.mkdir(parents=True, exist_ok=True)
    log_path = live_dir / "live.log"

    log_handle = log_path.open("a", encoding="utf-8")
    try:
        _live_log(log_handle, "Live worker started.")

        ffmpeg = _ensure_ffmpeg_on_path()
        if not ffmpeg:
            _live_log(log_handle, "ffmpeg not found; live processing disabled.")
            return

        if not manifest_path.exists():
            _live_log(log_handle, f"Manifest not found: {manifest_path}")
            return

        _apply_cpu_limits()
        device = _select_live_device(config)
        asr_backend = _init_live_asr_backend(config, device, log_handle)
        diar_backend = _init_live_diarization_backend(config, log_handle)

        chunks: list[LiveChunk] = []
        processed_indices: set[int] = set()
        manifest_offset = 0
        eos = False

        asr_next_start: Optional[float] = None
        diar_next_start: Optional[float] = None
        asr_segments: list[dict] = []
        last_asr_committed = 0.0
        diar_segments: list[dict] = []
        last_diar_committed = 0.0
        next_speaker_id = 0

        with manifest_path.open("r", encoding="utf-8") as manifest_handle:
            while True:
                entries, manifest_offset = _read_manifest_entries(
                    manifest_handle, manifest_offset
                )
                updated = False
                for entry in entries:
                    if entry.get("type") == "eos":
                        eos = True
                        continue
                    index = entry.get("index")
                    path_value = entry.get("path")
                    if index is None or not path_value or index in processed_indices:
                        continue
                    chunk_path = session_dir / path_value
                    if not chunk_path.exists():
                        continue
                    wav_path = chunks_wav_dir / f"chunk-{int(index):06d}.wav"
                    if not wav_path.exists():
                        try:
                            _convert_to_wav(chunk_path, wav_path)
                        except Exception as exc:  # noqa: BLE001
                            _live_log(log_handle, f"Failed to convert chunk: {exc}")
                            continue
                    try:
                        duration = _wav_duration(wav_path)
                    except Exception as exc:  # noqa: BLE001
                        _live_log(log_handle, f"Failed to read wav duration: {exc}")
                        continue
                    start_sec = entry.get("start_sec")
                    if start_sec is None:
                        start_sec = chunks[-1].end_sec if chunks else 0.0
                    chunk = LiveChunk(
                        index=int(index),
                        path=chunk_path,
                        wav_path=wav_path,
                        start_sec=float(start_sec),
                        duration_sec=float(duration),
                    )
                    chunks.append(chunk)
                    processed_indices.add(int(index))
                    updated = True

                if updated:
                    chunks.sort(key=lambda item: item.start_sec)
                    if asr_backend:
                        asr_next_start, last_asr_committed, asr_segments = _run_live_asr(
                            config=config,
                            live_dir=live_dir,
                            chunks=chunks,
                            asr_backend=asr_backend,
                            asr_next_start=asr_next_start,
                            last_committed_time=last_asr_committed,
                            segments=asr_segments,
                            log_handle=log_handle,
                            ffmpeg=ffmpeg,
                        )
                    if diar_backend:
                        (
                            diar_next_start,
                            last_diar_committed,
                            diar_segments,
                            next_speaker_id,
                        ) = _run_live_diarization(
                            config=config,
                            live_dir=live_dir,
                            chunks=chunks,
                            diar_backend=diar_backend,
                            diar_next_start=diar_next_start,
                            last_committed_time=last_diar_committed,
                            global_segments=diar_segments,
                            next_speaker_id=next_speaker_id,
                            log_handle=log_handle,
                            ffmpeg=ffmpeg,
                            session_id=session_dir.name,
                        )
                    _run_live_merge(live_dir, log_handle)

                if eos:
                    _live_log(log_handle, "End of stream received. Live worker exiting.")
                    break

                time.sleep(max(0.1, config.poll_ms / 1000.0))
    finally:
        log_handle.close()

def main() -> int:
    parser = argparse.ArgumentParser(description="Whistleblower native host")
    parser.add_argument("--worker", action="store_true", help="Run background worker")
    parser.add_argument("--live-worker", action="store_true", help="Run live worker")
    parser.add_argument("--audio", help="Audio file path")
    parser.add_argument("--session-dir", help="Optional session dir")
    parser.add_argument("--manifest", help="Path to live manifest JSONL")
    args = parser.parse_args()

    if args.live_worker:
        if not args.session_dir or not args.manifest:
            raise SystemExit("--session-dir and --manifest are required for live worker")
        run_live_worker(args)
        return 0

    if args.worker:
        if not args.audio:
            raise SystemExit("--audio is required for worker")
        run_worker(args)
        return 0

    run_host()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
