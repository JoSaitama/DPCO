#!/usr/bin/env bash
# Local iNaturalist defaults; reuse the common sequential TPU suite coordinator.
set -euo pipefail

suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$suite_root/run_sage_suite.sh" \
  --config configs/tpu/inaturalist2021_resnet50.toml \
  --methods sage_e sgd sage_l "$@"
