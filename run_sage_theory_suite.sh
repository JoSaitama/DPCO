#!/usr/bin/env bash
# Independent finite CPU validation; never calls the production TPU launchers.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite_python="${SAGE_PYTHON:-$suite_root/../.venv/bin/python}"
if [[ ! -x "$suite_python" ]]; then
  echo "Project Python not found: $suite_python. Set SAGE_PYTHON explicitly." >&2
  exit 1
fi
cd "$suite_root"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
exec "$suite_python" -u scripts/validate_sage_theory.py "$@"
