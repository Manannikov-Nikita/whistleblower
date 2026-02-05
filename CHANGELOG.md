# Start

## 2026-02-05
- Added live (during recording) processing via chunk manifests and a background live worker.
- Added live output artifacts under `output/session-*/live/`.
- Added `faster-whisper` dependency and live processing environment flags.

## 2026-02-03
- Installer default directory is now `~/whistleblower` (non-hidden).
- Installer prompts for `.env` values and writes the file with defaults for non-API settings.
- Added `OPENAI_BASE_URL` support for OpenAI-compatible endpoints and installer prompt.
