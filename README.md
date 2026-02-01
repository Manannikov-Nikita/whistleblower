# Whistleblower

CLI pipeline for transcription, diarization, and summaries from recorded meeting audio.
Recording is handled by the Chrome extension in `chrome_audio/`.

## Requirements
- macOS (Apple Silicon)
- Python 3.9–3.12 (3.11 recommended)
- Google Chrome (for recording via extension)
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

## Record audio (Chrome extension)
Use the extension in `chrome_audio/` to record the current tab (optionally with mic).
It saves a `.webm` file to your Downloads folder. See `chrome_audio/README.md`.

## Quick Start
Example workflow using a session folder:
```bash
SESSION=output/session-YYYYmmdd-HHMMSS
mkdir -p "$SESSION"
# Pick the file you just recorded in Downloads.
cp ~/Downloads/your-recording.webm "$SESSION"/audio.webm

uv run whistleblower transcribe --session-dir "$SESSION"
uv run whistleblower diarize --session-dir "$SESSION"
uv run whistleblower merge-transcript --session-dir "$SESSION"
uv run whistleblower summarize --session-dir "$SESSION"
```

Output files will be in the session folder:
- `transcript.txt`
- `transcript_segments.json`
- `diarization/segments.json` + `diarization/diarization.rttm`
- `speaker_transcript.md`
- `summary.md`

Manual steps (if you want custom paths or separate files):
```bash
uv run whistleblower transcribe --audio ~/Downloads/recording.webm \
  --output output/transcript.txt \
  --segments-output output/transcript_segments.json

uv run whistleblower diarize --audio ~/Downloads/recording.webm \
  --output-dir output/diarization

uv run whistleblower merge-transcript \
  --diarization-segments output/diarization/segments.json \
  --transcript-segments output/transcript_segments.json \
  --output output/speaker_transcript.md

uv run whistleblower summarize --input output/speaker_transcript.md \
  --output output/summary.md
```

## Notes
- The CLI auto-loads `.env` if present.
- pyannote models are gated and require a valid Hugging Face token.
- PyTorch 2.6+ uses safe checkpoint loading; code allowlists pyannote classes.
