import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class MergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DiarizationSegment:
    speaker: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class SpeakerUtterance:
    speaker: str
    start: float
    end: float
    text: str


def _load_transcript_segments(path: Path) -> List[TranscriptSegment]:
    if not path.exists():
        raise MergeError(f"Transcript segments file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: List[TranscriptSegment] = []
    for item in data:
        segments.append(
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=(item.get("text", "") or "").strip(),
            )
        )
    return segments


def _load_diarization_segments(path: Path) -> List[DiarizationSegment]:
    if not path.exists():
        raise MergeError(f"Diarization segments file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: List[DiarizationSegment] = []
    for item in data:
        segments.append(
            DiarizationSegment(
                speaker=str(item.get("speaker", "SPEAKER_UNKNOWN")),
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
            )
        )
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _pick_speaker(
    segment: TranscriptSegment,
    diarization_segments: List[DiarizationSegment],
) -> Optional[str]:
    best_speaker = None
    best_overlap = 0.0
    for diar in diarization_segments:
        ov = _overlap(segment.start, segment.end, diar.start, diar.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = diar.speaker
    return best_speaker


def merge_transcript(
    diarization_segments_path: Path,
    transcript_segments_path: Path,
    output_path: Path,
    max_gap: float = 1.0,
    unknown_label: str = "SPEAKER_UNKNOWN",
) -> Path:
    transcript_segments = _load_transcript_segments(transcript_segments_path)
    diarization_segments = _load_diarization_segments(diarization_segments_path)

    transcript_segments.sort(key=lambda seg: seg.start)
    diarization_segments.sort(key=lambda seg: seg.start)

    utterances: List[SpeakerUtterance] = []
    for segment in transcript_segments:
        if not segment.text:
            continue
        speaker = _pick_speaker(segment, diarization_segments) or unknown_label
        if utterances:
            last = utterances[-1]
            if (
                last.speaker == speaker
                and segment.start - last.end <= max_gap
            ):
                last.text = f"{last.text} {segment.text}".strip()
                last.end = segment.end
                continue
        utterances.append(
            SpeakerUtterance(
                speaker=speaker,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )

    lines = ["# Speaker Transcript", ""]
    for utterance in utterances:
        lines.append(f"- {utterance.speaker}: {utterance.text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
