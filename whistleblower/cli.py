import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from whistleblower.diarize import DiarizationError, run_diarization
from whistleblower.merge_transcript import MergeError, merge_transcript
from whistleblower.summarize import SummarizationError, summarize_text
from whistleblower.transcribe import TranscriptionError, transcribe_audio


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


def _resolve_knowledge_base_dir(session_dir: Path) -> Path:
    configured = os.getenv("KNOWLEDGE_BASE_DIR")
    if configured:
        return Path(configured).expanduser()
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        return Path(output_dir).expanduser() / "knowledge_base"
    return session_dir.parent / "knowledge_base"


def _resolve_knowledge_session_dir(session_dir: Path) -> Path:
    return _resolve_knowledge_base_dir(session_dir) / session_dir.name


def _resolve_session_audio(session_dir: Path) -> Path:
    wav_path = session_dir / "audio.wav"
    if wav_path.exists():
        return wav_path
    candidates = sorted(session_dir.glob("audio.*"))
    if not candidates:
        raise FileNotFoundError(
            f"No audio file found in {session_dir}. Expected audio.wav or audio.*"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple audio files found in {session_dir}: {names}. Use --audio."
        )
    return candidates[0]


def _cmd_transcribe(args: argparse.Namespace) -> int:
    if args.session_dir:
        session_dir = Path(args.session_dir)
        try:
            audio_path = _resolve_session_audio(session_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        output_path = session_dir / "transcript.txt"
        segments_path = session_dir / "transcript_segments.json"
    else:
        if not args.audio:
            print("Provide --session-dir or --audio.", file=sys.stderr)
            return 2
        audio_path = Path(args.audio)
        output_path = Path(args.output) if args.output else audio_path.with_suffix(".txt")
        segments_path = (
            Path(args.segments_output)
            if args.segments_output
            else audio_path.with_suffix(".segments.json")
        )

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
        session_dir = Path(args.session_dir)
        try:
            audio_path = _resolve_session_audio(session_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        output_dir = session_dir / "diarization"
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
        diarization_segments = base_dir / "diarization" / "segments.json"
        transcript_segments = base_dir / "transcript_segments.json"
        output_path = _resolve_knowledge_session_dir(base_dir) / "speaker_transcript.md"
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
        knowledge_session_dir = _resolve_knowledge_session_dir(base_dir)
        speaker_path = knowledge_session_dir / "speaker_transcript.md"
        legacy_speaker_path = base_dir / "speaker_transcript.md"
        transcript_path = base_dir / "transcript.txt"
        if speaker_path.exists():
            input_path = speaker_path
        elif legacy_speaker_path.exists():
            input_path = legacy_speaker_path
        else:
            input_path = transcript_path
        output_path = knowledge_session_dir / "summary.md"
    else:
        if not args.input:
            print("Provide --session-dir or --input.", file=sys.stderr)
            return 2
        input_path = Path(args.input)
        output_path = (
            Path(args.output) if args.output else input_path.with_suffix(".summary.md")
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
        description=(
            "CLI pipeline for transcription, diarization, and summaries. "
            "Record audio separately (e.g., via the Chrome extension)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe a recorded audio file."
    )
    transcribe_parser.add_argument(
        "--session-dir",
        help="Session directory containing audio.wav or audio.*.",
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
        help="Session directory containing audio.wav or audio.*.",
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
