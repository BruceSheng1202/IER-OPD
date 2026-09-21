#!/usr/bin/env bash
# Run a saved launch plan; this entry point also requires explicit --execute.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/_runtime.py" "$@"
