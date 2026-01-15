# Whistleblower

CLI pipeline for capturing meeting audio via BlackHole on macOS (Apple Silicon),
local Whisper transcription, speaker diarization (pyannote.audio), and
speaker-aware transcript output.

## Requirements
- macOS (Apple Silicon)
- Python 3.9+ (3.10+ recommended)
- Homebrew
- BlackHole audio device: https://existential.audio/blackhole/
- ffmpeg (for Whisper audio loading)

## Install
1) Install system dependencies:
```bash
brew install ffmpeg
```

2) Install Python deps:
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-diarization.txt
```

3) Create `.env` (already in repo) and fill keys:
```
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
HUGGINGFACE_TOKEN=
WHISPER_MODEL=medium
OUTPUT_DIR=./output
KEEP_RAW_AUDIO=false
CHUNK_SEC=20
```

4) Hugging Face access:
- Accept gated model terms:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0
- Make sure your token has read access to gated models.

## Quick Start
Main workflow (record until you press Enter, then auto-run pipeline to summary):
```bash
python3 -m whistleblower.cli start --device BlackHole
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
python3 -m whistleblower.cli start --device BlackHole --no-auto-summary
python3 -m whistleblower.cli transcribe --session-dir output/session-YYYYmmdd-HHMMSS
python3 -m whistleblower.cli diarize --session-dir output/session-YYYYmmdd-HHMMSS
python3 -m whistleblower.cli merge-transcript --session-dir output/session-YYYYmmdd-HHMMSS
python3 -m whistleblower.cli summarize --session-dir output/session-YYYYmmdd-HHMMSS
```

List input devices:
```bash
python3 -m whistleblower.cli list-devices
```

Test input from BlackHole:
```bash
python3 -m whistleblower.cli test-input --device BlackHole --seconds 5
```

Record a short sample:
```bash
python3 -m whistleblower.cli test-input --device BlackHole --seconds 40 --output output/sample.wav
```

Transcribe (Whisper):
```bash
python3 -m whistleblower.cli transcribe --audio output/sample.wav
```

Diarize (pyannote):
```bash
python3 -m whistleblower.cli diarize --audio output/sample.wav
```

Merge transcript + diarization:
```bash
python3 -m whistleblower.cli merge-transcript \
  --diarization-segments output/diarization/segments.json \
  --transcript-segments output/sample.segments.json \
  --output output/speaker_transcript.md
```

Summarize via OpenAI:
```bash
python3 -m whistleblower.cli summarize --input output/speaker_transcript.md
```

## Notes
- The CLI auto-loads `.env` if present.
- pyannote models are gated and require a valid Hugging Face token.
- PyTorch 2.6+ uses safe checkpoint loading; code allowlists pyannote classes.
