#!/usr/bin/env bash
# TPU-only qualification/short-grid entry; production updates remain unchanged.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$suite_root/run_dpco_suite.sh" \
  --config "$suite_root/configs/tpu/mechanisms/imagenet100_resnet50.toml" \
  --mode train --no-theorem-probes --mechanisms "$@"
