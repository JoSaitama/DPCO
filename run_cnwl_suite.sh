#!/usr/bin/env bash
# Shared sequential coordinator; fixed author Red split for all three methods.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$suite_root/run_dpco_suite.sh" \
  --config configs/tpu/cnwl/dpco_adam.toml \
  --methods adam dpco_filter dpco_flip "$@"
