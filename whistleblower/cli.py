import argparse
import json
import os
import signal
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from whistleblower.audio_capture import (
    list_input_devices,
    probe_input_signal,
    record_to_wav,
    resolve_device_index,
    summarize_levels,
)
from whistleblower.diarize import DiarizationError, run_diarization
from whistleblower.merge_transcript import MergeError, merge_transcript
from whistleblower.summarize import SummarizationError, summarize_text
from whistleblower.transcribe import TranscriptionError, transcribe_audio


def _print_devices() -> None:
    devices = list_input_devices()
    if not devices:
        print("No input devices detected.")
        return
    for device in devices:
        print(
            f"{device.index}: {device.name} "
            f"(inputs={device.max_input_channels}, "
            f"default_sr={int(device.default_samplerate)})"
        )


def _save_wave(path: Path, samples: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm16.shape[1] if pcm16.ndim > 1 else 1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())


def _cmd_test_input(args: argparse.Namespace) -> int:
    try:
        device_index = resolve_device_index(args.device)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print("Available devices:", file=sys.stderr)
        _print_devices()
        return 2

    print(f"Using device index {device_index}")
    levels, chunks = probe_input_signal(
        device=args.device,
        seconds=args.seconds,
        samplerate=args.samplerate,
        channels=args.channels,
        block_seconds=args.block_seconds,
        meter=not args.no_meter,
    )
    mean_level, max_level = summarize_levels(levels)
    print(f"Mean RMS: {mean_level:.4f}")
    print(f"Max RMS:  {max_level:.4f}")

    if args.output:
        samples = np.concatenate(chunks, axis=0)
        _save_wave(Path(args.output), samples, args.samplerate)
        print(f"Saved sample to {args.output}")

    return 0


def _session_file(output_dir: Path) -> Path:
    return output_dir / ".whistleblower_session.json"


def _default_session_name() -> str:
    return datetime.now().strftime("session-%Y%m%d-%H%M%S")


def _write_session_info(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_session_info(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_dotenv(path: Path = Path(".env")) -> None:
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


def _cmd_start(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = _session_file(output_dir)
    if session_path.exists():
        print(
            f"Session file already exists: {session_path}.",
            file=sys.stderr,
        )
        print("Run stop or remove the file to continue.", file=sys.stderr)
        return 2

    session_name = args.session_name or _default_session_name()
    session_dir = output_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    audio_path = session_dir / "audio.wav"

    session_info = {
        "pid": os.getpid(),
        "session_dir": str(session_dir),
        "audio_path": str(audio_path),
        "device": args.device,
        "samplerate": args.samplerate,
        "channels": args.channels,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_session_info(session_path, session_info)

    stop_event = threading.Event()

    def _handle_sigterm(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print(f"Recording to {audio_path}")
    if args.duration:
        print("Press Ctrl+C to stop.")
    else:
        print("Press Enter to stop (or Ctrl+C).")

    stats_holder = {"stats": None, "error": None}

    def _run_record() -> None:
        try:
            stats_holder["stats"] = record_to_wav(
                device=args.device,
                output_path=audio_path,
                samplerate=args.samplerate,
                channels=args.channels,
                block_seconds=args.block_seconds,
                stop_event=stop_event,
                duration=args.duration,
                meter=args.meter,
            )
        except Exception as exc:  # noqa: BLE001
            stats_holder["error"] = exc

    record_thread = threading.Thread(target=_run_record, daemon=True)
    record_thread.start()

    if args.duration is None:
        def _wait_for_enter() -> None:
            try:
                input()
            except EOFError:
                return
            stop_event.set()

        input_thread = threading.Thread(target=_wait_for_enter, daemon=True)
        input_thread.start()

    try:
        while record_thread.is_alive():
            if stop_event.is_set():
                break
            time.sleep(0.2)
    finally:
        stop_event.set()
        record_thread.join()
        if session_path.exists():
            session_path.unlink()

    if stats_holder["error"] is not None:
        exc = stats_holder["error"]
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
            return 2
        raise exc

    stats = stats_holder["stats"]

    if stats is not None:
        print(
            f"Recorded {stats.duration_sec:.1f}s, "
            f"frames={stats.frames}, overflows={stats.overflows}"
        )

    if args.auto_summary:
        transcript_path = session_dir / "transcript.txt"
        segments_path = session_dir / "transcript_segments.json"
        diarization_dir = session_dir / "diarization"
        speaker_path = session_dir / "speaker_transcript.md"
        summary_path = session_dir / "summary.md"
        try:
            transcribe_audio(
                audio_path=audio_path,
                model_name=args.whisper_model,
                language=args.language,
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

            summarize_text(
                input_text=speaker_path.read_text(encoding="utf-8"),
                output_path=summary_path,
            )
            print(f"Summary saved to {summary_path}")
        except (TranscriptionError, DiarizationError, MergeError, SummarizationError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    if args.transcribe:
        transcript_path = session_dir / "transcript.txt"
        segments_path = session_dir / "transcript_segments.json"
        try:
            transcribe_audio(
                audio_path=audio_path,
                model_name=args.whisper_model,
                language=args.language,
                output_path=transcript_path,
                output_segments_path=segments_path,
            )
            print(f"Transcript saved to {transcript_path}")
            print(f"Transcript segments saved to {segments_path}")
        except TranscriptionError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    session_path = _session_file(output_dir)
    session_info = _read_session_info(session_path)
    if not session_info:
        print(f"No active session file found at {session_path}.")
        return 1

    pid = session_info.get("pid")
    if not pid:
        print("Session file is missing pid.", file=sys.stderr)
        return 2

    try:
        os.kill(int(pid), signal.SIGINT)
    except ProcessLookupError:
        print("Session process not found. Cleaning up session file.")
        session_path.unlink(missing_ok=True)
        return 1

    print(f"Sent stop signal to PID {pid}.")
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    audio_path: Optional[Path] = None
    if args.session_dir:
        audio_path = Path(args.session_dir) / "audio.wav"
        output_path = Path(args.session_dir) / "transcript.txt"
        segments_path = Path(args.session_dir) / "transcript_segments.json"
    else:
        if not args.audio:
            print("Provide --session-dir or --audio.", file=sys.stderr)
            return 2
        audio_path = Path(args.audio)
        output_path = (
            Path(args.output)
            if args.output
            else audio_path.with_suffix(".txt")
        )
        segments_path = (
            Path(args.segments_output)
            if args.segments_output
            else audio_path.with_suffix(".segments.json")
        )

    if audio_path is None:
        print("Provide --session-dir or --audio.", file=sys.stderr)
        return 2

    try:
        transcribe_audio(
            audio_path=audio_path,
            model_name=args.whisper_model,
            language=args.language,
            output_path=output_path,
            output_segments_path=segments_path,
        )
    except TranscriptionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Transcript saved to {output_path}")
    print(f"Transcript segments saved to {segments_path}")
    return 0


def _cmd_diarize(args: argparse.Namespace) -> int:
    if args.session_dir:
        audio_path = Path(args.session_dir) / "audio.wav"
        output_dir = Path(args.session_dir) / "diarization"
    else:
        if not args.audio:
            print("Provide --session-dir or --audio.", file=sys.stderr)
            return 2
        audio_path = Path(args.audio)
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else audio_path.parent / "diarization"
        )

    try:
        rttm_path = run_diarization(
            audio_path=audio_path,
            output_dir=output_dir,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            model=args.model,
            device=args.device,
        )
    except DiarizationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Diarization complete. RTTM: {rttm_path}")
    print(f"Segments JSON: {output_dir / 'segments.json'}")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    if args.session_dir:
        base_dir = Path(args.session_dir)
        diarization_segments = (
            base_dir / "diarization" / "segments.json"
        )
        transcript_segments = base_dir / "transcript_segments.json"
        output_path = base_dir / "speaker_transcript.md"
    else:
        if not args.diarization_segments or not args.transcript_segments:
            print(
                "Provide --session-dir or both --diarization-segments and "
                "--transcript-segments.",
                file=sys.stderr,
            )
            return 2
        diarization_segments = Path(args.diarization_segments)
        transcript_segments = Path(args.transcript_segments)
        output_path = (
            Path(args.output)
            if args.output
            else diarization_segments.parent / "speaker_transcript.md"
        )

    try:
        output = merge_transcript(
            diarization_segments_path=diarization_segments,
            transcript_segments_path=transcript_segments,
            output_path=output_path,
            max_gap=args.max_gap,
            unknown_label=args.unknown_label,
        )
    except MergeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Speaker transcript saved to {output}")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    if args.session_dir:
        base_dir = Path(args.session_dir)
        speaker_path = base_dir / "speaker_transcript.md"
        transcript_path = base_dir / "transcript.txt"
        input_path = speaker_path if speaker_path.exists() else transcript_path
        output_path = base_dir / "summary.md"
    else:
        if not args.input:
            print(
                "Provide --session-dir or --input.",
                file=sys.stderr,
            )
            return 2
        input_path = Path(args.input)
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_suffix(".summary.md")
        )

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        text = input_path.read_text(encoding="utf-8")
        summarize_text(
            input_text=text,
            output_path=output_path,
            model=args.model,
            language=args.language,
        )
    except (SummarizationError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Summary saved to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whistleblower",
        description="CLI for capturing audio from BlackHole and testing input.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "list-devices", help="List available input devices."
    )

    test_parser = subparsers.add_parser(
        "test-input", help="Probe input audio and show RMS meter."
    )
    test_parser.add_argument(
        "--device",
        help="Input device index or substring (e.g. 'BlackHole').",
    )
    test_parser.add_argument(
        "--seconds", type=float, default=5.0, help="Probe duration."
    )
    test_parser.add_argument(
        "--samplerate", type=int, default=16000, help="Sample rate."
    )
    test_parser.add_argument(
        "--channels", type=int, default=1, help="Number of input channels."
    )
    test_parser.add_argument(
        "--block-seconds",
        type=float,
        default=0.2,
        help="Meter update interval.",
    )
    test_parser.add_argument(
        "--no-meter",
        action="store_true",
        help="Disable live RMS meter output.",
    )
    test_parser.add_argument(
        "--output",
        help="Optional WAV path to save a short sample.",
    )

    start_parser = subparsers.add_parser(
        "start", help="Start recording from input device."
    )
    start_parser.add_argument(
        "--device",
        help="Input device index or substring (e.g. 'BlackHole').",
    )
    start_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for session output.",
    )
    start_parser.add_argument(
        "--session-name",
        help="Optional session folder name.",
    )
    start_parser.add_argument(
        "--samplerate", type=int, default=16000, help="Sample rate."
    )
    start_parser.add_argument(
        "--channels", type=int, default=1, help="Number of input channels."
    )
    start_parser.add_argument(
        "--block-seconds",
        type=float,
        default=0.2,
        help="Read block interval.",
    )
    start_parser.add_argument(
        "--duration",
        type=float,
        help="Optional max duration in seconds.",
    )
    start_parser.add_argument(
        "--meter",
        action="store_true",
        help="Print RMS meter while recording.",
    )
    start_parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Run Whisper transcription after recording (no diarization).",
    )
    start_parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model name.",
    )
    start_parser.add_argument(
        "--language",
        default="ru",
        help="Language for transcription.",
    )
    start_parser.add_argument(
        "--auto-summary",
        action="store_true",
        default=True,
        help="Run full pipeline (transcribe → diarize → merge → summarize).",
    )
    start_parser.add_argument(
        "--no-auto-summary",
        action="store_false",
        dest="auto_summary",
        help="Disable auto pipeline after stop.",
    )

    stop_parser = subparsers.add_parser(
        "stop", help="Stop a running recording session."
    )
    stop_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where session file is stored.",
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe a recorded audio file."
    )
    transcribe_parser.add_argument(
        "--session-dir",
        help="Session directory containing audio.wav.",
    )
    transcribe_parser.add_argument(
        "--audio",
        help="Path to audio file to transcribe.",
    )
    transcribe_parser.add_argument(
        "--output",
        help="Optional transcript output path.",
    )
    transcribe_parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model name.",
    )
    transcribe_parser.add_argument(
        "--language",
        default="ru",
        help="Language for transcription.",
    )
    transcribe_parser.add_argument(
        "--segments-output",
        help="Path to write transcript segments JSON.",
    )

    diarize_parser = subparsers.add_parser(
        "diarize", help="Run pyannote diarization on an audio file."
    )
    diarize_parser.add_argument(
        "--session-dir",
        help="Session directory containing audio.wav.",
    )
    diarize_parser.add_argument(
        "--audio",
        help="Path to audio file to diarize.",
    )
    diarize_parser.add_argument(
        "--output-dir",
        help="Output directory for diarization files.",
    )
    diarize_parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to consider.",
    )
    diarize_parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to consider.",
    )
    diarize_parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face model for diarization.",
    )
    diarize_parser.add_argument(
        "--hf-token",
        help="Hugging Face token (or set HUGGINGFACE_TOKEN).",
    )
    diarize_parser.add_argument(
        "--device",
        help="Torch device (e.g. cpu, mps).",
    )

    merge_parser = subparsers.add_parser(
        "merge-transcript",
        help="Merge diarization segments with transcript segments.",
    )
    merge_parser.add_argument(
        "--session-dir",
        help="Session directory containing diarization and transcript.",
    )
    merge_parser.add_argument(
        "--diarization-segments",
        help="Path to diarization segments JSON.",
    )
    merge_parser.add_argument(
        "--transcript-segments",
        help="Path to transcript segments JSON.",
    )
    merge_parser.add_argument(
        "--output",
        help="Path to save speaker transcript markdown.",
    )
    merge_parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="Max gap in seconds to merge adjacent speaker segments.",
    )
    merge_parser.add_argument(
        "--unknown-label",
        default="SPEAKER_UNKNOWN",
        help="Label to use when speaker cannot be assigned.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize transcript via OpenAI API."
    )
    summarize_parser.add_argument(
        "--session-dir",
        help="Session directory with speaker_transcript.md or transcript.txt.",
    )
    summarize_parser.add_argument(
        "--input",
        help="Input file to summarize.",
    )
    summarize_parser.add_argument(
        "--output",
        help="Path to save summary markdown.",
    )
    summarize_parser.add_argument(
        "--model",
        help="OpenAI model name (default from OPENAI_MODEL).",
    )
    summarize_parser.add_argument(
        "--language",
        default="Russian",
        help="Output language for summary.",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-devices":
        _print_devices()
        return 0
    if args.command == "test-input":
        return _cmd_test_input(args)
    if args.command == "start":
        return _cmd_start(args)
    if args.command == "stop":
        return _cmd_stop(args)
    if args.command == "transcribe":
        return _cmd_transcribe(args)
    if args.command == "diarize":
        return _cmd_diarize(args)
    if args.command == "merge-transcript":
        return _cmd_merge(args)
    if args.command == "summarize":
        return _cmd_summarize(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
