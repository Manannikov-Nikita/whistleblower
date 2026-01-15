import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class DiarizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiarizationSegment:
    speaker: str
    start: float
    duration: float


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    return token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


def _write_segments_json(path: Path, segments: List[DiarizationSegment]) -> None:
    payload = [
        {"speaker": s.speaker, "start": s.start, "duration": s.duration}
        for s in segments
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_diarization(
    audio_path: Path,
    output_dir: Path,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    model: str = "pyannote/speaker-diarization-3.1",
    device: Optional[str] = None,
) -> Path:
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
        raise DiarizationError(
            "pyannote.audio is not installed. Run: pip install -r "
            "requirements-diarization.txt"
        ) from exc

    if not audio_path.exists():
        raise DiarizationError(f"Audio file not found: {audio_path}")

    token = _resolve_hf_token(hf_token)
    if not token:
        raise DiarizationError(
            "Hugging Face token is required. Set HUGGINGFACE_TOKEN (or "
            "HF_TOKEN) or pass --hf-token."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Allow loading trusted pyannote checkpoints under torch >= 2.6.
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
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    except Exception as exc:
        raise DiarizationError(
            "Failed to load pyannote pipeline. Ensure your Hugging Face token "
            "is valid and you accepted the model terms at "
            f"https://hf.co/{model}."
        ) from exc
    if pipeline is None:
        raise DiarizationError(
            "Failed to load pyannote pipeline. Ensure your Hugging Face token "
            "is valid and you accepted the model terms at "
            f"https://hf.co/{model}."
        )
    if device:
        pipeline.to(device)

    diarization = pipeline(
        str(audio_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    rttm_path = output_dir / "diarization.rttm"
    with rttm_path.open("w", encoding="utf-8") as handle:
        diarization.write_rttm(handle)

    segments: List[DiarizationSegment] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker=speaker,
                start=float(segment.start),
                duration=float(segment.duration),
            )
        )

    segments_path = output_dir / "segments.json"
    _write_segments_json(segments_path, segments)

    return rttm_path
