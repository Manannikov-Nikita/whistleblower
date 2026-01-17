# Whistleblower

CLI pipeline for capturing meeting audio via BlackHole on macOS (Apple Silicon),
local Whisper transcription, speaker diarization (pyannote.audio), and
speaker-aware transcript output.

## Requirements
- macOS (Apple Silicon)
- Python 3.9–3.12 (3.11 recommended)
- Homebrew
- BlackHole audio device: https://existential.audio/blackhole/
- ffmpeg (for Whisper audio loading)
- uv (https://github.com/astral-sh/uv)

## Install
1) Install system dependencies:
```bash
brew install ffmpeg
```

2) Install uv:
```bash
brew install uv
```

3) Install Python deps:
```bash
uv sync
```

4) Create `.env` (already in repo) and fill keys:
```
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
HUGGINGFACE_TOKEN=
WHISPER_MODEL=medium
OUTPUT_DIR=./output
KEEP_RAW_AUDIO=false
CHUNK_SEC=20
```

5) Hugging Face access:
- Accept gated model terms:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0
- Make sure your token has read access to gated models.

## Quick Start
Main workflow (record until you press Enter, then auto-run pipeline to summary):
```bash
uv run whistleblower start --device BlackHole
# press Enter to stop
```

Output files will be in the session folder:
- `transcript.txt`
- `transcript_segments.json`
- `diarization/segments.json` + `diarization/diarization.rttm`
- `speaker_transcript.md`
- `summary.md`

Manual steps (if you want to control each stage):
```bash
uv run whistleblower start --device BlackHole --no-auto-summary
uv run whistleblower transcribe --session-dir output/session-YYYYmmdd-HHMMSS
uv run whistleblower diarize --session-dir output/session-YYYYmmdd-HHMMSS
uv run whistleblower merge-transcript --session-dir output/session-YYYYmmdd-HHMMSS
uv run whistleblower summarize --session-dir output/session-YYYYmmdd-HHMMSS
```

List input devices:
```bash
uv run whistleblower list-devices
```

Test input from BlackHole:
```bash
uv run whistleblower test-input --device BlackHole --seconds 5
```

Record a short sample:
```bash
uv run whistleblower test-input --device BlackHole --seconds 40 --output output/sample.wav
```

Transcribe (Whisper):
```bash
uv run whistleblower transcribe --audio output/sample.wav
```

Diarize (pyannote):
```bash
uv run whistleblower diarize --audio output/sample.wav
```

Merge transcript + diarization:
```bash
uv run whistleblower merge-transcript \
  --diarization-segments output/diarization/segments.json \
  --transcript-segments output/sample.segments.json \
  --output output/speaker_transcript.md
```

Summarize via OpenAI:
```bash
uv run whistleblower summarize --input output/speaker_transcript.md
```

## Notes
- The CLI auto-loads `.env` if present.
- pyannote models are gated and require a valid Hugging Face token.
- PyTorch 2.6+ uses safe checkpoint loading; code allowlists pyannote classes.
