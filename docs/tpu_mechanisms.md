# Opt-in TPU T1/T2/fixed-target/LR mechanisms

This extends the REAL classification training path. It is not the synthetic CPU
theory suite and does not replace SGD/SAGE with an analytic optimizer. Existing
methods, strict-negative flips, valid-B means, Trainer/update/policy code and
default training computations are unchanged. Only optional observation, private
recording schemas, configuration forwarding and extra observer resume state are
connected to the existing experiment runner. No TPU run was performed during
development; the tests below are CPU/offline semantic checks.

## Existing versus new capabilities

File map (paths relative to the independent SAGE repository):

| Files | Reason |
|---|---|
| src/probes/mechanism_config.py | New dependency-light flags, validation, TOML/CLI conversion |
| src/probes/fixed_training_bank.py | New immutable fixed-target assets, fingerprints and streamed target gradients |
| src/probes/tpu_mechanisms.py | New read-only diagnostics, private schemas, compiled stages and window state |
| run_sage_mechanism_suite.sh; configs/tpu/mechanisms/imagenet100_resnet50.toml | New bounded qualification entry and isolated preset |
| src/config.py; src/training/launch.py; scripts/launch_tpu.py | Append optional config and forward it through existing parent/child/dry-run |
| scripts/train_classification.py | Construct observer/bank only under the new master switch |
| src/training/experiment.py | Guarded observation, private recorder schemas, target/prefix consistency and opt-in checkpoint state |
| tests/test_tpu_mechanisms.py; tests/mechanism_mesh_check.py | New offline and fresh multi-device acceptance tests |
| README.md; docs/development.md; docs/tpu_mechanisms.md | Usage, implementation record, fields/defaults/costs/commands |

The outer canonical research master and workspace memory also record the outcome;
these local records are not part of the code repository. No original method,
reference, trainer/update implementation, policy file or legacy CSV schema changes.

| Existing | New opt-in extension |
|---|---|
| Whole-global-batch scalar AD/JVP and method replay | Reuse their staged kernels, checking actual displacement and explicit numeric reliability |
| One trajectory reference per observer | Both E/L at each chosen frozen state, including on SGD trajectories |
| Relative LR scan | Separately labeled common absolute LR grid; relative grid retained |
| Real endpoint mixed into base-rate pair | Separate independent_pair and actual_pair rows, with matching-method/reference checks |
| Fixed training images but changing before-buffers | One genuinely fixed train-mode bank target with frozen initial input buffers |
| Per-audit-batch transfer | Ordinary-source transfer AND a separate uniform finite-bank enumeration |
| Sparse observations | Explicit consecutive windows, with actual-displacement telescoping decomposition |

The independent CPU suite remains `run_sage_theory_suite.sh`. This TPU entry is
`run_sage_mechanism_suite.sh`, which delegates to the existing single-host nohup
TPU coordinator. Both regular ImageNet and iNaturalist suites also accept the new
flags. Original policies can still use their existing diagnostics; the new
mechanism observer intentionally accepts only original sgd/sage_e/sage_l.

## Switches, defaults and cost

The production default is `[mechanism] enabled=false`. Below are dataclass
defaults; the separate qualification preset explicitly enables selected checks.
CLI values override TOML. Empty list options clear their configured lists.

Legacy `--diagnostics` and `--theorem-probes` remain independent, including their
original costs/relative LR meanings. The new wrapper disables old theorem probes;
use train mode and `--no-theorem-probes` when enabling mechanisms through a regular
suite. New `--mechanism-lr-scales` does not overwrite `--diagnostic-lr-scales`.

| CLI | TOML field / default | Cost when active |
|---|---|---|
| --mechanisms / --no-mechanisms | enabled=false | Master switch; off creates no bank, traces/compiles no new observer kernel, adds no private CSV/cursor |
| --mechanism-update-check | update_check=true | Actual method replay plus independent scalar AD/JVP at source nodes |
| --mechanism-paired-loss | paired_loss=true | E/L candidates on source objectives and F_A |
| --mechanism-bank-moments | bank_moments=true | Stream all K bank batches at moment nodes |
| --mechanism-norm-matched | norm_matched=false | Additional read-only equal-length candidates/forwards |
| --mechanism-steps | steps=[0,5,10] | PRE-update states for source T1/T2/update checks |
| --mechanism-moment-steps | moment_steps=[0] | Independent sparse schedule for finite-bank moments |
| --mechanism-oracle-steps | oracle_steps=[] | Explicit few nodes for exact-bank-reference candidates |
| --mechanism-absolute-lrs | absolute_lrs=[0,.001,.01,.05] | Same actual hypothetical LR values across training LR settings |
| --mechanism-lr-scales | lr_scales=[1,.5] | Current actual LR multiplied by each scale, labeled relative |
| --mechanism-windows 0:12 20:30 | windows=[] | EVERY actual update in each disjoint [start,end) window; expensive |
| --mechanism-bank-batches | bank_batches=2 | K whole logical batches; input memory scales with K, gradients stream |
| --mechanism-bank-seed | bank_seed=42 | Separate from training iterator/augmentation RNG |

Boolean check flags also accept their `--no-...` form. Update-only use can turn
off paired/moments, clear windows/oracle nodes and thereby avoid building a bank.
Pair and oracle comparisons require at least one LR list. Nodes at/after the
update budget are inactive; windows must fit entirely in that budget. A window
is not inferred from sparse nodes. Momentum, L2 and Nesterov must be off.

Source forward, Gd and scalar AD kernels are shared across E/L. The staged
observer also reuses scalar candidate losses for common SGD endpoints and
coincident absolute/relative rates, while keeping separately labeled rows. It
does not cache a large collection of parameter candidates. The staged
implementation synchronizes between kernels, never builds [B,P] gradients and
does not calculate B-backward sample energy. Extra bank gradients are streamed,
not stacked. Cost is still substantial: each target evaluation adds K forwards;
h_A adds K backwards; moments add E/L work for each bank batch; a window does
this on every included update. Norm matching and LR grids multiply forward cost.

## Fixed objective and recovery

F_A(theta) = (1/K) sum_k J_Ak(theta;s*), with each J a valid-sample mean over a
complete GLOBAL logical training batch. Freeze images, labels, augmentations,
grouping, valid masks, train-mode global BN and the INITIAL input buffers s*.
Each candidate's output buffers are discarded. Real training BN updates normally.
There is no eval-mode substitution and no microbatch/local-BN substitution.

The target ID hashes actual bank values/grouping/masks, sample identities,
buffer values/shapes/dtypes, model/task source signature, precision and dependency
versions. Metadata records these fingerprints and the buffer origin. s* is
reconstructed from the original model initialization (same seed/config/version),
not from current resumed buffers; no complete model snapshot is added for it.
Changing the target identity is rejected on resume, even with the same flags.
F_A is a fixed TRAINING PROXY, not full ImageNet, validation or population risk.

Observer counters/window sums are added to the normal checkpoint payload ONLY
when this extension is enabled. Private CSV cursors rewind with that checkpoint.
Legacy metadata without the new section means all extension defaults/off, without
rewriting old metadata. Exact config matching still applies. Disabled checkpoint
runs cannot recover this or the model; existing checkpoint save/retention rules
are unchanged. A window resumed midway needs its saved prefix; gaps/duplicates
are errors, never a silently complete cumulative result.

## Records and interpretations

Private streams are registered only in enabled runs; old global schemas do not
change. Each row uses step=t, update_step=t+1, trajectory_method/reference and real
source identity. Bank objectives name the fixed target ID; source loss pairs use
an explicit source-objective/state ID. Normal train CSV row t+1 describes the
update's pre-state as before. Source context retains exact indices/augmentation.

| Stream | Meaning |
|---|---|
| mechanism_context | Actual source context, LR and cumulative LR before the observed update |
| mechanism_update | Replay/independent/actual errors, actual displacement, BN-state errors, dtype/matmul policy and provisional numeric gate |
| mechanism_t1 | E/L or explicitly labeled oracle, reference validity/error/negative/flip counts, Delta/formula/residual, score margin and direction squared norms |
| mechanism_pairs | Explicit LR kind/value, objective, independent/actual candidate kind, normalization, H/gains/residuals and loss-resolution/contamination flags |
| mechanism_transfer | ORDINARY training source to F_A: Delta, Lambda=h_A^T C, mismatch=(r-h_A)^T C, identity residual; no source-unbiasedness assumption |
| mechanism_moments | exact_finite_bank: uniform enumeration of K BANK batch indices at s*, h-dot-directions, Delta/Lambda/mismatch and second moments; mean Gd minus h_A error |
| mechanism_windows | Every update of a configured continuous window; actual P/R/displacement energy and prefix/endpoint identity |
| mechanism_timing | Setup, individual first-use compilation, diagnostic execution and recording IO durations |

Finite-bank enumeration is exact only for that finite uniform batch-index
distribution. Population standard errors are not invented (standard_error is
blank). Neither it nor ordinary-source transfer is a whole-training expectation.
The exact-bank-reference oracle uses r=h_A on the SAME source, with an explicit
label. It never drives training or participates in actual method rankings.

For T2, source full CE, nonempty frozen-error CE and F_A are distinct objectives.
Error masks stay frozen even when candidate predictions change. All valid
contributions, including correctly classified ones, remain eligible to flip.
The objective gain is grad(J)^T(Gs-Gd), equal to Delta only for matching source
reference objectives. Empty error objectives are omitted; reference_valid=false
and no flips indicate the SGD fallback. Zero LR ratios/residuals are blank.

`independent_pair` constructs BOTH candidates from independent frozen directions.
`actual_pair` is a SEPARATE row only at the actual LR, for the actual matching
method/reference. Actual SGD can be compared with either counterfactual E/L;
actual SAGE-E cannot be used as a SAGE-L or oracle endpoint, or conversely.
Both grids remain labeled even if an absolute and relative entry coincide.

Norm matching uses eta_s=eta_d*||Gd||/||Gs|| ONLY for hypothetical candidates.
Zero SAGE norm gives an undefined row, not an epsilon denominator. Zero SGD norm
gives eta_s=0. Original Delta/gain is retained separately from the effective
norm-matched objective gain; the residual uses that effective gain. A norm-matched
candidate is never called a real update or an original-Delta guarantee.

### Numeric qualification is not a finite flag

Float32 thresholds are fixed in advance: absolute2e-6, relative1e-4, BN absolute
2e-6. They are provisional CPU-tested thresholds, NOT TPU certification. Relative
errors are undefined at zero denominator. Bfloat16 is explicitly unsupported by
this qualification gate; its results remain unqualified without independent
calibration, rather than passing via widened tolerances. Training dtype, BN,
batch size, matmul settings and optimizer are never changed by the observer.

Rows retain absolute errors even if the gate fails. Pair sign claims additionally
need H and eta*effective_gain above the explicit loss-resolution floor. An actual
endpoint's loss contamination must be less than10% of BOTH independent H and the
first-order signal. Otherwise signal_reliable=false; no tolerance is increased
to turn a result into a pass. Missing update checks do not imply qualification.
Finite-only rows and a completed process are not evidence of numerical correctness.

### Continuous-window identity, not a convergence certificate

For actual displacement u=theta_after-theta_before, record P=-h_A^T u and
R=F_A(after)-F_A(before)+P. For a complete window check
F_A(end)-F_A(start)=-sum P+sum R, with raw residual and explicit tolerance.
R is the measured nonlinear/numerical remainder. No unknown ResNet smoothness
constant is manufactured and R is not called an L-smooth upper bound. The
identity does not order independent SGD/SAGE paths or prove an acceleration rate.

Timing separates compile, setup, training, diagnostics and IO. Existing timing.csv
contains real training windows/compile warmup; mechanism_timing adds observer
costs. Setup includes fixed-bank construction and fingerprinting before the loop.
Diagnostic-enabled elapsed time is not optimizer-only throughput.

## Prepared-data TPU qualification and short-grid commands

The following are deliverable commands for a LATER authorized run. They were
only dry-run inspected during this development task. Use the existing worker0
project environment and already prepared data; no extra TPU resources are created.

```bash
cd ~/Project_SAGE/SAGE
source ../.venv/bin/activate
bash ./run_sage_mechanism_suite.sh \
  --config configs/tpu/mechanisms/imagenet100_resnet50.toml \
  --methods sgd sage_e sage_l --dry-run
```

Qualification:3 original methods, seed2, LR.01, B256,12 actual updates, K2,
float32 unchanged. The separate preset also enables a continuous [0,12) window,
norm matching, selected absolute/relative LR pairs and a single oracle node.

```bash
mkdir -p results/logs/mechanism_qualification
nohup bash ./run_sage_mechanism_suite.sh \
  --config configs/tpu/mechanisms/imagenet100_resnet50.toml \
  --methods sgd sage_e sage_l --seeds 2 \
  --steps 12 --batch-size 256 --lr 0.01 \
  --lr-schedule constant --momentum 0 --no-nesterov --l2-coefficient 0 \
  --no-checkpoints \
  >results/logs/mechanism_qualification/suite.log 2>&1 < /dev/null &
```

After inspecting real TPU devices, complete records, numerical errors/reliability,
compilation/diagnostic cost and peak memory, a user may authorize this nine-run
short constant-LR grid. It is NOT automatically launched by qualification.
Only the LR.01 three trajectories use the [0,80) continuous window.

```bash
mkdir -p results/logs/mechanism_short_grid
nohup bash -c '
set -euo pipefail
common=(
  --config configs/tpu/mechanisms/imagenet100_resnet50.toml
  --methods sgd sage_e sage_l --seeds 2 --steps 200 --batch-size 256
  --lr-schedule constant --momentum 0 --no-nesterov --l2-coefficient 0
  --no-checkpoints --eval-every-steps 200
  --mechanism-steps 0 1 5 10 20 25 30 50 75 100 150 199
  --mechanism-moment-steps 0 50 199 --mechanism-oracle-steps 0
)
for rate in 0.005 0.01 0.02; do
  if [[ "$rate" == "0.01" ]]; then
    windows=(--mechanism-windows 0:80)
  else
    windows=(--mechanism-windows)
  fi
  bash ./run_sage_mechanism_suite.sh "${common[@]}" --lr "$rate" "${windows[@]}"
done
' >results/logs/mechanism_short_grid/suite.log 2>&1 < /dev/null &
```

Loss signs or SAGE wins are never qualification criteria. Keep failed/unreliable
observations. Do not scale up based on CPU tests, process completion, or one
favorable short trace. Actual TPU precision, compilation, memory and runtime
remain unmeasured until these commands are separately authorized and executed.

## Offline acceptance (2026-10-24)

All173 CPU/offline regression tests pass, including13 new mechanism tests and a
fresh two-device CPU global-BN/dense-oracle process. Final targeted config/observer
checks also pass after preserving the original positional RunConfig arguments.
Tests cover disabled cursors/trajectories, true/counterfactual endpoint labels,
frozen/empty/all-error cases, zero LR/norms, numerical fault flags, fixed buffers,
finite-bank gradients, full window/resume identities, CLI/TOML forwarding and
tiny-ResNet real-entry wiring. Repeated LR labels reuse candidate forwards while
retaining distinct rows. Neither the3-job qualification nor the9-job grid was
executed on TPU; both passed configuration-only dry runs.
