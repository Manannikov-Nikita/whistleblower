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
Quick install (recommended):
```bash
curl -fsSL https://raw.githubusercontent.com/Manannikov-Nikita/whistleblower/master/scripts/install.sh | bash
```
The installer will prompt for API keys and write `.env`.
Optional overrides:
```bash
curl -fsSL https://raw.githubusercontent.com/Manannikov-Nikita/whistleblower/master/scripts/install.sh | bash -s -- --dir ~/whistleblower --extension-id <EXTENSION_ID>
```

Manual install:
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

4) Create `.env` and fill keys:
```
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
HUGGINGFACE_TOKEN=
WHISPER_MODEL=medium
OUTPUT_DIR=./output
KEEP_RAW_AUDIO=false
CHUNK_SEC=20
LIVE_PROCESSING=true
LIVE_ASR_MODEL=small
LIVE_ASR_WINDOW_SEC=30
LIVE_ASR_STRIDE_SEC=10
LIVE_ASR_OVERLAP_SEC=5
LIVE_DIARIZE_WINDOW_SEC=120
LIVE_DIARIZE_STRIDE_SEC=60
LIVE_DEVICE=auto
LIVE_POLL_MS=500
FFMPEG_PATH=/opt/homebrew/bin/ffmpeg
```

5) Hugging Face access:
- Accept gated model terms:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0
- Make sure your token has read access to gated models.

## Record audio (Chrome extension)
Use the extension in `chrome_audio/` to record the current tab (optionally with mic).
Audio is streamed to the native host and processed immediately. See `chrome_audio/README.md`.

### Native Messaging (required for streaming)
Install the native host so the extension can stream audio and trigger the pipeline.
Use the same Python that has the project dependencies installed (via `uv sync`):
```bash
PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')" \\
  bash native_host/install_native_host.sh
```
Default extension ID is fixed: `kboppgghhbphgciaolnfakeldpphpikg`.
To override it:
```bash
bash native_host/install_native_host.sh --extension-id <EXTENSION_ID>
```

If you prefer a plain install (system Python in PATH):
```bash
bash native_host/install_native_host.sh
```
Logs go to `output/native_host.log`. Reload the extension after installing the host.

## Quick Start
Record via the Chrome extension, then check `output/session-*/` for results.

Output files will be in the session folder:
- `transcript.txt`
- `transcript_segments.json`
- `diarization/segments.json` + `diarization/diarization.rttm`
- `speaker_transcript.md`
- `summary.md`

Live (during recording) output files will be in `output/session-*/live/`:
- `chunks/` and `chunks_wav/`
- `live_transcript.txt` and `live_transcript_segments.json`
- `live_diarization/segments.json` and `live_diarization.rttm`
- `speaker_transcript.live.md`
- `live.log`

Manual steps (if you want custom paths or separate files):
```bash
uv run whistleblower transcribe --audio /path/to/recording.webm \
  --output output/transcript.txt \
  --segments-output output/transcript_segments.json

uv run whistleblower diarize --audio /path/to/recording.webm \
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
- For OpenAI-compatible endpoints (on-prem/OpenRouter), set `OPENAI_BASE_URL`.
- Live processing uses `faster-whisper` when available and falls back to `openai-whisper`.
