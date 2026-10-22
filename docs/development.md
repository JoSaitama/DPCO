# Development record

## 2026-10-22: optional diagnostics on real training updates

- User authorized implementation of the three agreed early-reversal checks. Keep
  one selected method for the entire trajectory; do not introduce edit warm-up,
  alternating SGD steps, last-layer scoring or a new loss/regularizer. The earlier
  read-only restriction is superseded only for this diagnostic implementation.
- Added disabled-by-default diagnostics with independently controlled update,
  paired-source-loss and training-target-transfer checks. All switches, absolute
  state nodes, LR scales and audit-bank controls propagate through both shell
  suites, the nohup parent and child; dry-run prints effective settings. Legacy
  fixed-batch theorem probes remain independently controlled and unchanged.
- Preserve the original compiled training kernel. At selected states retain the
  real batch/before-state, run the normal update once, then inspect its real
  after-parameters. Independent scalar-loss gradients and method replay localize
  direction/update/BN-state discrepancies. Sequential whole-batch JIT stages
  bound simultaneously live activation workspaces without dense runtime Jacobians.
- Pair candidate losses at the frozen before-state. Scale1 uses the real candidate
  for the trajectory method; smaller scales use independent reconstructed directions.
  SAGE-E error masks stay frozen. Reference Delta and evaluated-objective projection
  have separate names; an implementation error can affect the scale1 residual.
- Prepare a reproducible training audit bank with a separate seed/fixed training
  augmentations; record membership once and source overlap per node. No training
  examples are withheld, iterator/RNG advanced, bank reselected, or sample energy
  computed. Target projections/loss differences are finite-bank diagnostics, not
  a global convergence or population guarantee.
- Register diagnostic_context/update/paired/transfer CSV streams in the shared
  recorder. step=t observes update_step=t+1, which matches the train CSV row;
  store source indices/epoch/augmentation context for replay. Flush at nodes,
  preserve CSV-cursor resume/legacy-disabled compatibility, and record diagnostic
  time separately. Disabled diagnostics create no bank, AD evaluations or output
  files. The small host-side enable check leaves the original training trace intact.
- English comments and docs describe switches, provenance, numerical interpretation
  and a 200-update ImageNet100/ResNet50/LR0.01/seed2 launch. LR*step is logged through
  cumulative step sizes and is not hard-coded as a universal reversal threshold.
- Validation: 135 CPU/offline tests passed with JAX0.8.1/Flax0.12.0, including dense
  oracle parity, sharded two-CPU global-BN parity, tiny-ResNet real-entry execution,
  injected wrong-LR/double-state faults, exact on/off trajectory equality, fixed
  masks/empty-reference/zero-LR handling, all individual switches, CLI round trips,
  and checkpoint resume without duplicate diagnostic records. Shell/Python syntax,
  whitespace and English-only content checks passed. No server TPU job, commit
  or push was performed; actual TPU memory/timing/residuals still require testing.

## 2026-10-21: training-objective-first theory validation protocol

- Professor/user priority: the current optimizer study evaluates training
  optimization and convergence speed first. Use unsigned train CE and training
  top-1 versus completed updates/sample exposures as primary outcome evidence.
  Training accuracy is a companion metric; the differentiable optimized objective
  remains CE. Validation/test/final held-out accuracy belong to a later model
  quality/generalization study and are excluded from this phase's conclusions.
- Retain the mechanism evidence required by each theorem: T1 checks reference
  projection/sign aggregation/Delta; T2 compares actual objectives after paired
  one-step updates from one frozen state; T3 needs its fixed-target and cumulative
  conditions. Training curves support optimization outcomes but do not replace
  those identities, conditions or expectation estimates. Do not silently change
  the professor's statements into an unconditional ordering of independent runs.
- First audit the existing ImageNet-100/ResNet-50, LR=0.01, seed-2 SGD/SAGE-E/
  SAGE-L training exports across early, middle and late stages. Check code,
  normalization, update/state/recording semantics and the earliest loss reversal
  before attributing the gap to backbone choice or changing the model/recipe.
- Use raw observations together with explicitly labeled fixed-window summaries.
  Recorded train metrics describe the current batch before its indexed update;
  preserve this convention. Do not interpret cross-batch CE differences as T2's
  paired same-objective gain, or fixed-budget loss gaps as a statistical expectation.
- This instruction authorizes protocol documentation and read-only analysis of
  existing data. It does not authorize training-code/config changes, new runtime
  metrics, disabling scheduled evaluation, server launches, commits or pushes.

## 2026-10-20: local iNaturalist training-suite integration

- Added an independent run_inaturalist_suite.sh with the existing sequential
  nohup TPU coordinator and all shared options. Pilot defaults: ResNet-50,
  sage_e/sgd/sage_l, training seed 5, 2,000 steps, B=256, LR=0.01 constant,
  L2/momentum/Nesterov off, full validation each epoch and probes off.
- Added dataset inaturalist2021 and --data-dir propagation through parent/child,
  class-head validation and explicit directory handling when switching sources.
  Read the completed proportional-full manifest/SQLite selection without HF,
  downloads, copies or reselection. Training seeds never change membership.
- Extracted the existing image reader/transforms/shuffle/evaluation into a shared
  module, retaining ImageNet import compatibility and cache identity. The local
  adapter checks manifest/completion/source markers/member identities/counts and
  byte lengths read-only; full stored-byte hashing remains an offline verify step.
  Smoke uses a distinct limited-read view without creating another image cache.
- Reused the actual training entry, references, methods, BatchNorm semantics,
  status TSV, checkpoint retention/recovery, metrics and optional probes. No new
  observation streams, theorem definitions, optimizer defaults for ImageNet or
  TPU topology rules were introduced. Git commit/push remain manual.
- Added offline dataset-integrity/read-only/view/cursor tests, suite override and
  child round trips, and tiny-backbone CPU runs of all three methods through the
  real shared entry and checkpoint resume. Actual ResNet-50 TPU execution and
  full selected-data throughput remain server checks.
- Validation: all 121 CPU/offline tests passed. After preserving the legacy
  ImageNet dataset-switch configuration exactly, all 16 affected suite/iNaturalist
  tests passed again. Shell syntax, executable permission, outside-repository
  dry-run, child option propagation, whitespace and English-only repository
  checks passed. No TPU job, commit or push was performed locally.

## 2026-10-19: reproducible proportional iNaturalist 2021 preparation

- Recorded the user-approved 140k full-train/20k validation/all-10k-class protocol
  in the outer local experiment record. Selection seed defaults to 42 and is
  independent of training seeds. No mini substitution or equal train quotas.
- Added exact largest-remainder quotas and canonical SHA-256 ranking, deterministic
  manifests and source checksums; selections do not depend on host paths, RNG
  library versions, input order, Python hash seed, or training state.
- Added a standalone plan/download/verify/all CLI and a small optional `data` extra
  for streaming annotation JSON. Stream official image tar.gz archives without
  saving full archives; keep selected original bytes and individual SHA-256 hashes
  in SQLite. Preserve partial progress, replay interrupted streams, lock writers,
  and publish completion only after source/count/content verification.
- Default limits: 15 decimal GB for preparation and 10 decimal GB free-space
  reserve. Capacity/network failures never silently truncate/reselect/resize.
  Network scans can still approach 232 GB; full-source download timing and actual
  selected storage remain server measurements, not guarantees.
- No training/model/optimizer/probe integration, new training metrics, TPU job,
  full data download, automatic commit or push is performed by this change.
  See inaturalist_subset.md for server preparation and cross-host verification.
- Validation: 115 CPU/offline unit and integration tests passed. New tests cover
  exact quotas/ties, all-class coverage, row/Python-hash independence, identical
  manifests in two host directories, distinct selection seeds with equal quotas,
  publication recovery, mid-split replay, selected-byte equality, corruption/source
  checksum/missing-member/capacity failures, concurrent lock rejection and a
  standalone CLI/module with no JAX imports. Existing two-device training/probe
  tests remain green. No full official dataset transfer or server preparation has
  been performed locally.

## 2026-10-18: bounded-memory theory probes after step-0 TPU OOM

- Diagnosed the user's real ResNet-50/B=256 failure at the initial probe's jacrev,
  after 5,000-image validation and before any update or initial checkpoint. The
  failing HBM allocation requested 6.12G with 4.63G remaining. Final Jacobian size
  alone had underestimated vectorized reverse-mode intermediate memory.
- Replaced runtime dense Jacobians with whole-batch compiled VJP/JVP references,
  scores and paired SGD/SAGE directions. Exact energy pulls back one loss at a
  time inside a compiled scalar-carry loop, avoiding B simultaneous backward
  activation trees and [B,P] gradient retention. No microbatching, sample-energy
  approximation or changes to batch/reference/BatchNorm semantics are introduced.
- JIT-compiled T2 objective forwards and the fixed audit value/gradient. Skip the
  energy loop when T3 is off. Sparse nodes, all existing CSV fields, checkpoint
  policy, optimizer updates and every TOML setting remain unchanged.
- Added dense-oracle comparisons for all statistics and recorded fields, padding,
  empty/all/mixed errors, small Flax ResNet/BN, read-only state, a scalar-loop/no-full-
  Jacobian structural check, and a runtime no-jacrev guard. The two-device CPU
  test also verifies exact probe energy and global BN against the dense oracle.
- Sequential exact energy has significant compute cost; no actual TPU memory or
  throughput result is claimed locally. Retain failed logs and reuse prepared data;
  this failed pre-checkpoint run must restart as a new run, not an assumed resume.
  Git commits/pushes remain manual user operations.
- Validation completed: 102 CPU unit/integration tests passed, including dense
  numeric oracles, recorded-field parity, real two-device BN/energy checks and
  the no-runtime-jacrev guard. Python/shell syntax, English source/docs and diff
  whitespace checks passed. The reported B=256 TPU OOM must still be retested
  by the user after pulling; no server-memory success is claimed from CPU tests.

## 2026-10-18: mutually exclusive shell validation cadence controls

- Added --eval-every-steps (positive integer) and --eval-every-epochs (positive
  finite float) to the common parent/child CLI as a mutually exclusive group.
  The resolver also rejects simultaneous values. Omission honors the config's
  evaluation mode/interval; all shipped profiles retain epochs/1.
- Forwarded cadence through every shell suite job and dry-run plan into the
  existing evaluation configuration. No new recording fields/streams, automatic
  subsampling, training updates, best-selection rule or checkpoint policy changes.
- Preserved initial/final validation and trigger deduplication. Tests exercise
  shell/child parity and precedence, invalid/conflicting arguments before launch,
  step/fractional-epoch triggers, full validation counts, parameter/data isolation
  and exact resume without duplicate evaluation rows.
- This revision accompanies the still-uncommitted checkpoint interval change.
  Commit/push remain manual user-only actions. No server job is changed locally.
- Validation completed: 97 CPU unit/integration tests passed, including both
  cadence controls, checkpoint failure safety, evaluation isolation/resume, and
  two-device CPU pipeline checks. Python/shell syntax, English source/docs and
  whitespace checks passed. No actual TPU job, commit or push was executed.

## 2026-10-18: shell control for periodic recovery checkpoints

- Added --checkpoint-every-steps as a positive-integer override of
  checkpoints.latest_every_steps. The shell suite, launcher dry-run/config and TPU
  child all share the same value; omitted CLI values honor the config (default 100).
- Preserved existing seven-slot retention and save ordering: complete/fsync and
  atomically publish the snapshot, atomically publish index.json, then remove old
  unreferenced snapshots. No optimizer, data, probe, metric or log cadence changes.
- Added tests for shell/child forwarding, config precedence and invalid interval
  fail-fast behavior, actual latest cadence independent of milestones, publication
  before cleanup, snapshot/index write failures, and cleanup failure recovery.
- Resume retains the existing full-configuration equality rule, including the
  chosen interval. No new per-step recording fields or files are introduced.
  Commits/pushes remain user-only; no actual server jobs are changed or launched.
- Validation completed: 93 CPU unit/integration tests passed, including injected
  checkpoint failure cases and six-job shell-to-child interval forwarding. Python
  and shell syntax, English source/docs and diff whitespace checks passed.

## 2026-10-18: optional momentum, cosine scheduling and formal batch 256

- Added zero-default momentum, an independent false-default Nesterov switch, and
  constant-default LR scheduling with an optional cosine floor. Suite options
  --momentum, --nesterov/--no-nesterov, --lr/--learning-rate, --lr-schedule and
  --min-lr reach each child, dry-run plan and effective configuration. No implicit
  warmup, LR scaling with batch, dampening or EMA normalization is introduced.
- Plain momentum accumulates beta*v+g; Nesterov uses g+beta*v_next. Optional coupled
  L2 is added to the method direction first. Zero momentum allocates no velocity;
  disabled L2/momentum with constant LR retains the original update JAXPR.
- Cosine uses update indices 0..T-1: peak at the first, floor at the last (T>=2).
  One-update runs use peak once; later schedule queries return floor. Existing LR
  columns record the applied step rate. Zero-rate probe ratios are blank/undefined
  and are not marked eligible for strict gain; no new metric fields are added.
- RunConfig and all 10 presets now default to global B=256 for train/validate.
  Smoke mode explicitly defaults to B=8, preserving the lightweight server check.
  --batch-size overrides either mode. The 100-step budget and 0.01 LR stay untuned.
- Momentum state is initialized before the initial checkpoint and warmup, saved
  with TrainState and restored without reset. Resume validates all update settings;
  missing legacy fields imply off/constant defaults without metadata rewriting.
  Older runs also need their original batch explicitly retained after preset changes.
- Existing theorem probes require zero momentum and L2; cosine remains supported.
  Mainline SAGE reference/flip construction, batch-mean normalization, data selection,
  device isolation, result paths, status TSV and checkpoint retention are unchanged.
- Verification includes independent multistep momentum/Nesterov/L2 recurrence,
  cosine endpoints and host/JIT agreement, suite option parity and fail-fast checks,
  exact resumed parameters/velocity/LR, legacy resume, and two-CPU global-batch
  audit-vs-JVP parity with the combined optimizer options. Commits/pushes remain
  user-only; no remote or actual TPU training is executed in this local revision.
- Validation completed: 87 CPU unit/integration tests passed, including the actual
  two-device CPU mesh check. Combined momentum/Nesterov/cosine-to-zero/L2 passed
  both CPU sanity scripts; English source/docs, Python/shell syntax and diff
  whitespace checks passed. Batch-256 TPU memory/throughput is not measured locally.

## 2026-10-18: opt-in coupled L2 with an unchanged disabled update

- Added training.l2_coefficient, default 0, and --l2-coefficient forwarding from
  run_sage_suite.sh through the coordinator to each TPU child. A CLI zero overrides
  a nonzero custom config; invalid/nonfinite/negative values fail before any job.
- A static Python branch excludes all regularization operations when the coefficient
  is zero. Enabled L2 adds lambda times parameter after sample aggregation only for
  leaves with ndim > 1, preserving reference construction, sign selection, raw CE
  and accuracy definitions. No new recording streams or metrics were added.
- Existing theorem probes reject nonzero L2 rather than reporting unregularized
  certificates for a regularized update. Legacy resume metadata missing the field
  compares as zero without any rewrite; a changed coefficient still fails resume.
- Momentum, cosine scheduling and default batch-size changes remain separate pending
  work. This change does not activate them or alter existing preset batches/budgets.
- Verification covers disabled JAXPR identity and exact numerical parity, enabled
  kernel-only updates, unchanged same-state SGD/SAGE metrics, shell/child forwarding,
  config validation and checkpoint resume. No local TPU timing or accuracy claim,
  Git commit or push is made.
- Validation completed: 80 CPU unit/integration tests passed. The L2-specific tests
  passed again after review, and opt-in L2 passed both CPU core and recording sanity
  scripts. Python/shell syntax, English source/docs and git diff whitespace checks
  passed. No TPU smoke or formal training was launched locally.

## 2026-10-18: compact lifecycle TSV and parameterized shell suites

- Replaced progress status events with exactly start/end per launch/resume attempt.
  Removed step/error columns; completed/failed are the terminal outcomes. Step and
  error detail remain in existing JSON/CSV. Child success is deferred until cleanup.
  Stable sidecar locking deduplicates parent/child writes and preserves two rows when
  the waiting coordinator finalizes a killed child. No per-step TSV file access occurs.
- No old-schema backup or migration is performed. The user will remove old smoke
  artifacts after their jobs finish. Incompatible TSV headers are rejected without
  rewriting data; a missing/empty TSV is initialized with the current header.
- Added --seed/--seeds and --method/--methods list aliases, dataset/model/data-mode
  overrides, parent/child config parity, preflight validation of the complete grid,
  seed-major serial execution, true child-exit waiting and failure-stop behavior.
  The default seed and all shipped presets now use 42; explicit custom config seeds
  remain effective when CLI seeds are omitted. Resume requires one matching job.
- Added executable run_sage_suite.sh and a small Python suite entry. It defaults
  to train mode with the existing ImageNet-100/ResNet-50/SAGE-E pilot, supports a smoke
  override and runs under outer nohup. SAGE_PYTHON can override the project interpreter.
  Mismatch modes are reserved and rejected before launch rather than relabeled clean.
  Each actual run receives its own launch-time timestamp and unchanged per-run outputs.
- Added tests for incompatible-schema rejection, two-row deduplication, cleanup failure, abrupt
  detached-child exit, seed aliases/defaults, method/dataset/model round trips, shell
  argument forwarding from another cwd, sequential execution and fail-fast behavior.
  Gradient calculations, learning-rate values, training metrics and checkpoint policies
  are unchanged. Git commits and pushes remain manual user-only operations.
- Validation: 74 CPU unit/integration tests passed, including actual detached stdlib
  child success/abrupt exit without TPU access. All 10 seed-42 presets, exact TSV
  header, shell syntax/executable flag, Python syntax, English content and whitespace
  checks passed. No actual server training, commit or push was performed locally.
- The generic suite name leaves room for future data modes; only clean is implemented
  today. The old untracked three-seed helper may be removed when its queue finishes.
  Environment tooling in outer .tools is separate from that helper and should be
  inspected before deletion. The new repository scripts/run_suite.py remains required.

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
