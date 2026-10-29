#!/usr/bin/env bash
# Shared data-mode suite. The coordinator waits for each TPU child before the next.
set -euo pipefail

suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite_python="${DPCO_PYTHON:-$suite_root/../.venv/bin/python}"
if [[ ! -x "$suite_python" ]]; then
  echo "Project Python not found: $suite_python. Set DPCO_PYTHON to the project environment interpreter." >&2
  exit 1
fi
cd "$suite_root"
exec "$suite_python" -u "$suite_root/scripts/run_suite.py" "$@"
