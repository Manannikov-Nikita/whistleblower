import json
from pathlib import Path
from typing import Optional


class TranscriptionError(RuntimeError):
    pass


def transcribe_audio(
    audio_path: Path,
    model_name: str = "medium",
    language: str = "ru",
    output_path: Optional[Path] = None,
    output_segments_path: Optional[Path] = None,
) -> str:
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise TranscriptionError(
            "openai-whisper is not installed. Run: uv sync"
        ) from exc

    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_path),
        language=language,
        fp16=False,
        verbose=False,
    )
    segments = result.get("segments", []) or []
    lines = [seg.get("text", "").strip() for seg in segments]
    text = "\n".join([line for line in lines if line]) or result.get(
        "text", ""
    ).strip()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if output_segments_path is not None:
        output_segments_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text", "") or "").strip(),
            }
            for seg in segments
        ]
        output_segments_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return text
