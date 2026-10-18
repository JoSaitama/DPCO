# Development record

## 2026-10-18: real classification TPU entry and automatic data preparation

- CPU sanity is a foreground interface/recording check. Real smoke, theorem-validation
  and training are TPU-only nohup jobs on node-ziao1 worker0; worker1 remains unused.
  Renamed the earlier CPU smoke scripts to sanity_core.py and sanity_recording.py.
- Added isolated-local-runtime checks, startup timeout, per-host overlap protection,
  per-run bootstrap logging/reservation, explicit profiles, and a host-local GSPMD mesh.
- Added randomly initialized ResNet-50/101 V1.5 adapters from a pinned Flax source,
  retained source attribution/license, and configured 100/1000-class heads. No pretrained
  weights are requested or downloaded for this experiment.
- Added VJP/JVP full-reference alignment preserving raw scores, fixed B, frozen weights,
  empty-reference behavior and one model-state commit; no full training Jacobian is
  retained. Audit Jacobians remain available for numerical checks and sparse probes.
- Added original-resolution HF ImageNet streaming, the fixed CMC 100-class/index map,
  revision/limit-specific SQLite byte caches, disk preflight, resumable preparation,
  deterministic epoch shuffles/crops, and exact local-source recovery. Existing HF
  authentication is reused. Full preparation may incur large 1K-source network reads;
  smoke stops after a small selected subset and is explicitly not a complete benchmark.
- First stage is ImageNet-100/ResNet-50; subsequent ImageNet-1K/ResNet-101 verification
  remains later work. LR/budget settings are pilot placeholders, with vanilla SGD.
- Local tests cover global-batch parity on two CPU devices and an integrated encoded-image
  -> small Flax ResNet -> sharded trainer -> evaluation/recording -> exact resume path.
  No claim of actual TPU/libtpu execution or a measured convergence advantage is made
  before server validation. See tpu_training.md for execution and access details.
- Validation: 59 local CPU tests passed, including the two-device semantic check and
  real-adapter recovery integration. Both TPU presets and all eight original presets
  validated; the SAGE-E CPU sanity entry, launcher dry run, syntax/language/whitespace
  checks passed. Actual gated image access and TPU hardware execution remain server checks.

## 2026-10-17: agreed classification observation framework

- Retained the uncommitted SAGE-E presets, clarified README, result path helper, and
  analytic CPU smoke checks from the preceding change.
- Added independent named run directories, one metadata snapshot, persistent structured
  logs, buffered per-step CSV, full sample-weighted validation, and a shared recorder.
- Added configurable step/percentage logging, fractional-epoch/step evaluation, fixed
  milestone/best/latest checkpoint slots, atomic snapshot publication and source/cursor
  recovery. Checkpoint resumption restores model/optimizer/method/data state and rewinds
  CSV rows; seed/step alone is not claimed to restore a production data pipeline.
- Added optional read-only T1/T2/T3 classification probes, per-step reference-gain
  accumulation, explicit fixed audit objectives, window/event timing, and supported
  per-device memory readings. Probe switches default off and do not disable the actual
  SAGE algorithm.
- Added synthetic integration checks for state isolation, tail weighting, retention,
  threshold crossing, non-finite failures, and deterministic recovery without duplicate
  CSV rows. These checks do not certify ImageNet/ResNet/TPU training or convergence.
- ImageNet input, upstream Flax model adaptation, production-efficient scoring and
  single-host TPU parallel execution remain separate integration work.
- Source comments, configuration descriptions, and repository documentation remain English.
- Validation: 46 CPU unit/integration tests passed, including deterministic SAGE and
  random-flip recovery. Core analytic SAGE-E checks and recording smoke checks with
  probes enabled and disabled passed. All eight presets validated; Python syntax,
  repository language, ignored result paths, and whitespace checks passed.
