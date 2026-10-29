#!/usr/bin/env bash
# Same sequential worker0 TPU coordinator; all groups share official labels.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$suite_root/run_dpco_suite.sh" \
  --config configs/tpu/cifar100n/dpco_adam.toml \
  --methods adam dpco_filter dpco_flip "$@"
