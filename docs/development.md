# Development record

## 2026-10-18: cross-run status journal and JSON field order

- Added results/logs/run_status.tsv as a shared append-only lifecycle/progress index,
  with run/seed/config identity, per-attempt IDs, timestamps, process identity, last
  observed step, budget and errors. Parent and child share an ID; resume keeps the
  original run name but receives a new attempt. File locking protects concurrent
  writers. No history rewrite or backfill of earlier server smokes is performed.
- Bootstrap, spawn and loop failures are covered; abrupt kills/host loss/hard watchdog
  exits remain distinguishable only through their missing terminal event and external
  process/log checks. Status progress reuses the existing logging cadence and host
  values without extra forward passes, device synchronization or resource polling.
- Ordered train JSON as step/event/train_ce/lr, followed by the remaining training
  fields. Step-bearing non-training events put step/event first and omit unavailable
  training fields. CSV/config/checkpoint learning_rate names are unchanged. Reworded
  the launch message to distinguish background startup from confirmed completion.
- Reviewed performance without changing the gradient backend, warmup, synchronization,
  metrics or checkpoint policies. elapsed_wall_s is cumulative and includes loop
  compilation/warmup. The SAGE source requests two VJP pullbacks and one JVP; the JVP
  requests another evaluation, with possible compiler reuse. SGD uses one ordinary
  value_and_grad. Additional algorithm work and compile cost are not assumed free.
- The user's server logs report successful SGD, SAGE-E and SAGE-L TPU smokes. These
  three-update tests establish pipeline completion, not throughput or theory claims.
  In the supplied full logs, step 2-to-3 timestamp gaps are about 0.551 s for both
  SGD and SAGE-E; these dispatch gaps include boundary overhead and are not isolated
  device timings. Existing timing.csv separates compile_warmup and train_window.
- Validation: 65 local CPU unit/integration tests passed, including concurrent status
  appends by three seed processes, bootstrap/spawn failures, shared attempt identity,
  resume history and console/file JSON ordering. No TPU timing benchmark, commit or
  push was performed by the assistant in this revision.

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
