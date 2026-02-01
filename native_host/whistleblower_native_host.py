#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
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
            sessions[session_id] = {
                "path": audio_path,
                "handle": handle,
                "session_dir": session_dir,
                "bytes": 0,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Whistleblower native host")
    parser.add_argument("--worker", action="store_true", help="Run background worker")
    parser.add_argument("--audio", help="Audio file path")
    parser.add_argument("--session-dir", help="Optional session dir")
    args = parser.parse_args()

    if args.worker:
        if not args.audio:
            raise SystemExit("--audio is required for worker")
        run_worker(args)
        return 0

    run_host()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
