# Development record

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
