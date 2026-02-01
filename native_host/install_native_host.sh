#!/usr/bin/env bash
set -euo pipefail

EXTENSION_ID="${1:-}"
if [[ -z "$EXTENSION_ID" ]]; then
  echo "Usage: $0 <chrome-extension-id>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_SCRIPT="$ROOT_DIR/native_host/whistleblower_native_host.py"
WRAPPER_SCRIPT="$ROOT_DIR/native_host/whistleblower_native_host.sh"

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
MANIFEST_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
MANIFEST_PATH="$MANIFEST_DIR/com.whistleblower.native_host.json"

mkdir -p "$MANIFEST_DIR"
chmod +x "$HOST_SCRIPT"

cat > "$WRAPPER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$PYTHON_BIN" "$HOST_SCRIPT"
EOF
chmod +x "$WRAPPER_SCRIPT"

python3 - <<PY > "$MANIFEST_PATH"
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
