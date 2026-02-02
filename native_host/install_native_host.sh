#!/usr/bin/env bash
set -euo pipefail

DEFAULT_EXTENSION_ID="kboppgghhbphgciaolnfakeldpphpikg"
EXTENSION_ID=""
MANIFEST_DIR=""

usage() {
  cat <<EOF
Usage: $0 [--extension-id <id>] [--manifest-dir <path>] [<chrome-extension-id>]

Options:
  --extension-id <id>  Override Chrome extension ID.
  --manifest-dir <path> Override Native Messaging Hosts directory.
  -h, --help            Show this help.

Defaults:
  extension-id: $DEFAULT_EXTENSION_ID (or WHISTLEBLOWER_EXTENSION_ID)
  manifest-dir: ~/Library/Application Support/Google/Chrome/NativeMessagingHosts
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extension-id)
      EXTENSION_ID="${2:-}"
      shift 2
      ;;
    --manifest-dir)
      MANIFEST_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$EXTENSION_ID" && "$1" != --* ]]; then
        EXTENSION_ID="$1"
        shift
      else
        echo "Unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_SCRIPT="$ROOT_DIR/native_host/whistleblower_native_host.py"
WRAPPER_SCRIPT="$ROOT_DIR/native_host/whistleblower_native_host.sh"

if [[ -z "$EXTENSION_ID" ]]; then
  EXTENSION_ID="${WHISTLEBLOWER_EXTENSION_ID:-$DEFAULT_EXTENSION_ID}"
fi
if [[ -z "$EXTENSION_ID" ]]; then
  echo "Extension ID is required. Use --extension-id <id>."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "$(command -v python 2>/dev/null)" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ -z "$PYTHON_BIN" && -x "$(command -v python3 2>/dev/null)" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python not found. Set PYTHON_BIN=/path/to/python and retry."
  exit 1
fi

if [[ ! -f "$HOST_SCRIPT" ]]; then
  echo "Host script not found: $HOST_SCRIPT"
  exit 1
fi

MANIFEST_DIR="${MANIFEST_DIR:-${WHISTLEBLOWER_MANIFEST_DIR:-$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts}}"
MANIFEST_PATH="$MANIFEST_DIR/com.whistleblower.native_host.json"

mkdir -p "$MANIFEST_DIR"
chmod +x "$HOST_SCRIPT"

cat > "$WRAPPER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$PYTHON_BIN" "$HOST_SCRIPT"
EOF
chmod +x "$WRAPPER_SCRIPT"

"$PYTHON_BIN" - <<PY > "$MANIFEST_PATH"
import json
from pathlib import Path

ext_id = "$EXTENSION_ID"
manifest = {
    "name": "com.whistleblower.native_host",
    "description": "Whistleblower native messaging host",
    "path": str(Path("$WRAPPER_SCRIPT")),
    "type": "stdio",
    "allowed_origins": [f"chrome-extension://{ext_id}/"],
}
print(json.dumps(manifest, indent=2))
PY

echo "Installed native host manifest to: $MANIFEST_PATH"
echo "Host script: $HOST_SCRIPT"
echo "Python: $PYTHON_BIN"
echo "Extension ID: $EXTENSION_ID"
echo "Reload the Chrome extension after installing the host."
