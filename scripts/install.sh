#!/usr/bin/env bash
set -euo pipefail

DEFAULT_INSTALL_DIR="$HOME/whistleblower"
DEFAULT_REPO_URL="https://github.com/Manannikov-Nikita/whistleblower.git"
DEFAULT_EXTENSION_ID="kboppgghhbphgciaolnfakeldpphpikg"

INSTALL_DIR="$DEFAULT_INSTALL_DIR"
EXTENSION_ID=""
REPO_URL="${WHISTLEBLOWER_REPO_URL:-$DEFAULT_REPO_URL}"
PROMPT_DEVICE=""

if [[ -t 0 || -t 1 || -t 2 ]]; then
  PROMPT_DEVICE="/dev/tty"
fi

prompt_with_default() {
  local label="$1"
  local default_value="$2"
  local input=""

  if [[ -n "$default_value" ]]; then
    if [[ -n "$PROMPT_DEVICE" ]]; then
      read -r -p "$label [$default_value]: " input < "$PROMPT_DEVICE" || true
    else
      read -r -p "$label [$default_value]: " input || true
    fi
  else
    if [[ -n "$PROMPT_DEVICE" ]]; then
      read -r -p "$label: " input < "$PROMPT_DEVICE" || true
    else
      read -r -p "$label: " input || true
    fi
  fi

  if [[ -z "$input" ]]; then
    input="$default_value"
  fi

  printf '%s' "$input"
}

prompt_secret() {
  local label="$1"
  local input=""

  if [[ -n "$PROMPT_DEVICE" ]]; then
    read -r -s -p "$label: " input < "$PROMPT_DEVICE" || true
    echo ""
  else
    read -r -p "$label: " input || true
  fi

  printf '%s' "$input"
}

create_env_file() {
  local env_path="$INSTALL_DIR/.env"

  if [[ -f "$env_path" ]]; then
    local overwrite=""
    if [[ -n "$PROMPT_DEVICE" ]]; then
      read -r -p ".env already exists at $env_path. Overwrite? [y/N]: " overwrite < "$PROMPT_DEVICE" || true
    else
      read -r -p ".env already exists at $env_path. Overwrite? [y/N]: " overwrite || true
    fi
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
      echo "Keeping existing .env"
      return
    fi
  fi

  echo "Configuring .env..."

  local openai_key
  local openai_model
  local openai_base_url
  local hf_token
  local whisper_model
  local output_dir
  local keep_raw
  local chunk_sec
  local ffmpeg_path
  local default_ffmpeg_path

  openai_key="$(prompt_secret "OpenAI API key (OPENAI_API_KEY)")"
  hf_token="$(prompt_secret "Hugging Face token (HUGGINGFACE_TOKEN, optional)")"
  openai_base_url="$(prompt_with_default "OpenAI base URL (OPENAI_BASE_URL)" "https://api.openai.com/v1")"
  openai_model="$(prompt_with_default "OpenAI model (OPENAI_MODEL)" "gpt-4o-mini")"
  whisper_model="$(prompt_with_default "Whisper model (WHISPER_MODEL)" "medium")"
  output_dir="$(prompt_with_default "Output dir (OUTPUT_DIR)" "./output")"
  keep_raw="$(prompt_with_default "Keep raw audio? (KEEP_RAW_AUDIO) [true/false]" "false")"
  chunk_sec="$(prompt_with_default "Chunk length in seconds (CHUNK_SEC)" "20")"
  default_ffmpeg_path="$(command -v ffmpeg || true)"
  ffmpeg_path="$(prompt_with_default "FFmpeg path (FFMPEG_PATH)" "$default_ffmpeg_path")"

  cat > "$env_path" <<EOF
OPENAI_API_KEY=$openai_key
OPENAI_BASE_URL=$openai_base_url
OPENAI_MODEL=$openai_model
HUGGINGFACE_TOKEN=$hf_token
WHISPER_MODEL=$whisper_model
OUTPUT_DIR=$output_dir
KEEP_RAW_AUDIO=$keep_raw
CHUNK_SEC=$chunk_sec
FFMPEG_PATH=$ffmpeg_path
EOF

  echo "Wrote .env to: $env_path"

  if [[ -z "$openai_key" ]]; then
    echo "Note: OPENAI_API_KEY is empty; summaries will be skipped."
  fi
  if [[ -z "$hf_token" ]]; then
    echo "Note: HUGGINGFACE_TOKEN is empty; diarization will require a token."
  fi
}

usage() {
  cat <<USAGE
Usage: install.sh [--dir <path>] [--extension-id <id>] [--help]

Options:
  --dir <path>           Install directory (default: $DEFAULT_INSTALL_DIR)
  --extension-id <id>    Override Chrome extension ID
  -h, --help             Show this help

Environment:
  WHISTLEBLOWER_REPO_URL         Override git repository URL
  WHISTLEBLOWER_EXTENSION_ID     Override extension ID
  WHISTLEBLOWER_MANIFEST_DIR     Override Native Messaging Hosts directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      INSTALL_DIR="${2:-}"
      shift 2
      ;;
    --extension-id)
      EXTENSION_ID="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$EXTENSION_ID" ]]; then
  EXTENSION_ID="${WHISTLEBLOWER_EXTENSION_ID:-$DEFAULT_EXTENSION_ID}"
fi

INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer supports macOS only."
  exit 1
fi

missing=()
for cmd in git python3 uv ffmpeg; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing+=("$cmd")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing dependencies: ${missing[*]}"
  echo "Install with Homebrew:"
  for cmd in "${missing[@]}"; do
    case "$cmd" in
      git) echo "  brew install git" ;;
      python3) echo "  brew install python" ;;
      uv) echo "  brew install uv" ;;
      ffmpeg) echo "  brew install ffmpeg" ;;
    esac
  done
  echo "If Homebrew is not installed: https://brew.sh/"
  exit 1
fi

if [[ -d "$INSTALL_DIR" ]]; then
  if git -C "$INSTALL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Using existing repo at: $INSTALL_DIR"
  else
    echo "Directory exists but is not a git repo: $INSTALL_DIR"
    exit 1
  fi
else
  echo "Cloning repo to: $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

create_env_file

echo "Installing Python dependencies with uv..."
uv sync

PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Failed to resolve Python via uv."
  exit 1
fi

INSTALL_ARGS=("--extension-id" "$EXTENSION_ID")
if [[ -n "${WHISTLEBLOWER_MANIFEST_DIR:-}" ]]; then
  INSTALL_ARGS+=("--manifest-dir" "$WHISTLEBLOWER_MANIFEST_DIR")
fi

PYTHON_BIN="$PYTHON_BIN" bash native_host/install_native_host.sh "${INSTALL_ARGS[@]}"

MANIFEST_DIR="${WHISTLEBLOWER_MANIFEST_DIR:-$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts}"
MANIFEST_PATH="$MANIFEST_DIR/com.whistleblower.native_host.json"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Warning: manifest not found at $MANIFEST_PATH"
else
  if ! grep -q "chrome-extension://$EXTENSION_ID/" "$MANIFEST_PATH"; then
    echo "Warning: manifest does not contain expected extension ID: $EXTENSION_ID"
  fi
fi

if [[ ! -x "native_host/whistleblower_native_host.sh" ]]; then
  echo "Warning: wrapper script is not executable: $INSTALL_DIR/native_host/whistleblower_native_host.sh"
fi

cat <<NEXT

Next steps:
1. Load the extension from: $INSTALL_DIR/chrome_audio
2. Restart Chrome.
3. Record a session and check logs in: $INSTALL_DIR/output/native_host.log

Extension ID: $EXTENSION_ID
NEXT
