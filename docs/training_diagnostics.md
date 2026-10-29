# Real-update training diagnostics

These optional observers investigate an early training-CE reversal without changing
SGD, DPCO-E or DPCO-L. Every real update still uses the selected method, original
batch denominator, full-parameter scores and the existing compiled training step.
No edit warm-up, intermittent SGD updates, new regularizer or score normalization
is introduced. The LR-times-step pattern is a descriptive observation whose apparent
onset depends on smoothing; it is not an assumed universal critical threshold.

## Launch and switches

On worker0, use the existing project environment and sequential nohup TPU suite:

```bash
cd ~/Project_DPCO/DPCO
source ../.venv/bin/activate
nohup ./run_dpco_suite.sh \
  --mode train --dataset imagenet100 --model resnet50 \
  --methods dpco_l --seeds 2 --steps 200 --batch-size 256 \
  --lr 0.01 --lr-schedule constant --momentum 0 --no-nesterov --l2-coefficient 0 \
  --diagnostics --no-theorem-probes \
  --diagnostic-steps 0 5 10 20 25 30 50 75 100 \
  --diagnostic-lr-scales 1 0.5 0.25 0.125 0.0625 \
  --diagnostic-audit-batches 2 --diagnostic-audit-seed 42 \
  --checkpoint-every-steps 100 --eval-every-epochs 1 \
  >/dev/null 2>&1 < /dev/null &
```

Use `--dry-run` without outer nohup to inspect resolved settings and every child
command before launching. Change `--methods` to `dpco_l dpco_e sgd` for sequential
runs. A diagnostic compares both directions at the current trajectory's frozen
state; it never switches the method that advances that trajectory. Normal initial,
final and configured periodic validation remains separate from these checks.

| Option | Default / meaning |
|---|---|
| `--diagnostics` / `--no-diagnostics` | Off by default; master observer switch |
| `--diagnostic-steps` | Pre-update state steps: `0 5 10 20 25 30 50 75 100` |
| `--diagnostic-update-check` / `--no-diagnostic-update-check` | Enabled within diagnostics; independent direction/real parameter/BN checks |
| `--diagnostic-paired-loss` / `--no-diagnostic-paired-loss` | Enabled within diagnostics; source-batch candidate losses |
| `--diagnostic-transfer` / `--no-diagnostic-transfer` | Enabled within diagnostics; other training-batch candidate losses and target projections |
| `--diagnostic-lr-scales` | `1 0.5 0.25 0.125 0.0625`; relative to the actual LR at this update |
| `--diagnostic-audit-batches` | 2 fixed logical training batches; only allocated when transfer is enabled |
| `--diagnostic-audit-seed` | 42; independent of the training seed/data iterator |
| `--theorem-probes` / `--no-theorem-probes` | Legacy fixed-batch T1-3 probes; independent of the new observers |

Settings are also available in a `[diagnostics]` TOML table using the dataclass
field names `enabled`, `steps`, `update_check`, `paired_loss`, `transfer`,
`lr_scales`, `audit_batches`, and `audit_seed`. CLI values override TOML values.
At least one check must be enabled under the master switch. Diagnostics require
momentum and L2 to be zero and cover only SGD/full common-reference DPCO. A cosine
schedule is supported numerically, including a zero floor, but the first reversal
audit should retain the original constant-LR protocol.

`--no-diagnostics` allocates no audit bank, traces no diagnostic kernels, performs
no diagnostic model/gradient evaluations or scalar transfers, and writes no new
diagnostic files. The training kernel and its normal outputs remain unchanged.
Legacy theorem probes can still incur their own costs if separately enabled.
Use mode `train` plus `--no-theorem-probes` for the lightweight early audit; mode
`validate` otherwise enables legacy probes, including their configured T3 energy.

## Step and input conventions

`step=t` in a new diagnostic record means the state has completed t updates; the
observer inspects update `update_step=t+1`. Its input is the exact batch and random
augmentation used by that update. The corresponding `train_steps.csv` row is t+1,
whose metrics describe the PRE-update parameters. The real after-state is theta(t+1).
Legacy probe step=t still denotes its frozen state at t completed updates.

There is no update after the final state. Configured diagnostic nodes t >= total
steps are inactive, shown in launcher dry-run output rather than silently moved.
To inspect the update starting at state100, train for at least101 updates; the
200-update example includes an interval after the suspected reversal.

`diagnostic_context.csv` records the source image indices, epoch/cursor, training
seed, dataset fingerprint and augmentation convention in `source_context_json`.
The fixed audit bank's indices/identities/seed are recorded once in
`results/logs/{run_name}/metadata.json` under `training_diagnostics`. No image bytes,
full parameter dumps or per-example gradient matrices are written by the observer.

## Check 1: independent directions and the actual update

The independent route freezes the source error mask, takes `grad` of the scalar
reference mean, obtains scores by JVP, freezes signs and differentiates the scalar
weighted CE with the original B. It does not call the method's VJP statistics helper.
Small offline tests additionally compare it against an explicit dense oracle.
All forwards retain whole-batch BN dependencies; no single-image BN substitution
or microbatch reference is used.
Runtime reference gradients, JVP scores and the two scalar-loss gradients use
separate sequential JIT stages, synchronized at stage boundaries. This bounds
simultaneously live activation workspaces; it does not reduce the logical batch.

The method is also replayed read-only at the frozen before-state to localize a
discrepancy. This replay is not mislabeled as a raw intermediate exported from the
original training kernel. The decisive parameter check compares the original
kernel's real after-parameters with theta_before - LR * independent_direction.
The expected direction is SGD for an SGD trajectory and DPCO for a DPCO trajectory.

`diagnostic_update.csv` contains:

- `direction_l2_error`, `direction_relative_error`: replayed method versus
  independent scalar-loss direction.
- `parameter_l2_error`, `parameter_relative_error`: real after-parameters versus
  the independent expected update; relative error uses the expected update norm.
- `replayed_parameter_l2_error`: real parameters versus the replayed direction's
  expected update, separating direction mismatch from update mismatch.
- `model_state_max_abs_error`, `replayed_model_state_max_abs_error`: real/replayed
  BN state versus one normal before-state forward. No candidate BN state is committed.
- reference validity, error/flip counts, reference Delta, identity residual,
  minimum absolute score, direction squared norms and a finite-status flag.

Raw absolute and relative residuals are retained. A nonzero floating-point residual
is not automatically called a bug, and a finite flag is not an accuracy certificate.
Norm reductions use float32; actual model dtype/default matmul policy is unchanged.
The metadata records JAX's matmul precision setting and x64 switch for provenance.

## Check 2: same-state source-batch losses

`diagnostic_paired.csv` records `full_batch_ce` and, for a nonempty DPCO-E error
reference, `frozen_error_ce`. Initial and both candidate losses use the identical
source images/augmentation and before-state BN semantics. The original error mask
is held fixed even if candidate predictions change.

At LR scale1, the candidate for the actual trajectory method is the REAL
after-parameters; the other candidate uses the independent direction. Other scales
use independent directions for both candidates. `actual_update_in_pair` distinguishes
these cases. Consequently an implementation mismatch is not hidden by reconstructing
both candidates. Until Check1 residuals are understood, the scale1 first-order
residual can contain implementation error as well as finite-step/numerical error.

- `difference` is H = candidate_DPCO_CE - candidate_SGD_CE.
- `reference_delta` always means r dot (Gs-Gd) for the independent directions.
- `objective_gain` is grad(evaluated objective) dot (Gs-Gd). These quantities differ
  for DPCO-E when evaluating full-batch CE; their names never silently swap meaning.
- `first_order_residual` is H/eta + objective_gain, not a certified smoothness bound.
- `j_initial`, `j_sgd`, `j_dpco`, `probe_lr`, scales, selected counts, norms and
  finiteness allow independent recalculation. At eta0 the ratio/residual are blank.

## Check 3: transfer to a fixed training audit bank

The bank is drawn once from training images without replacement, using a separate
selection seed and fixed TRAINING augmentations. Its whole-batch shape/mode matches
the source. Bank images remain ordinary training images. Overlap with the source
is counted per batch and recorded as `overlap_sample_count`; the bank is not
reselected at each node. This preserves target identity and avoids changing the
training stream. Interpret overlap when assessing transfer; this is not an iid or
held-out-generalization claim.

`diagnostic_transfer.csv` evaluates both candidates on each bank batch using frozen
before-state buffers. It records the same loss/step-size fields, the target batch
identity, target `objective_gain` and `transfer_difference = objective_gain -
reference_delta`. For a target F_A, objective_gain is grad(F_A) dot C, and the loss
difference should approach -eta*objective_gain in the corresponding local regime.
No source error mask is applied to different target images. A small training bank
is not the full training objective or the population objective, and these records
alone do not certify Theorem3 or order two independent training trajectories.

## Cost, persistence and interpretation

All four CSVs use the existing buffered recorder at
`results/files/{run_name}/diagnostic_*.csv`. Context/diagnostic rows are flushed at
nodes. Logs retain one diagnostic event per node and the status TSV retains its
existing start/end-only contract. `timing.csv` separates diagnostic duration,
including first-use compilation, from the normal training window. The fixed bank
is prepared before the loop clock, like other input/model preparation. Diagnostic
runs are not measurements of optimizer-only throughput.

Only selected before-states are temporarily retained. No extra checkpoint policy
is imposed. Existing checkpoint CSV cursors include these streams, so same-config
resume rewinds future rows before repeating updates. Older runs implicitly had
diagnostics disabled; enabling the observer on an old run requires a new experiment
or an explicitly reproduced initial-state run, not changing resume metadata.

The transfer observer does NOT compute sample energy or any [B,P] Jacobian. Each
target adds a target-gradient evaluation and candidate forward evaluations; the
update check adds independent AD plus a read-only method replay. These costs are
real and configurable. Actual ResNet-50 TPU memory, duration and residual scales
must be measured on the server; CPU tests establish semantics, not TPU performance.

Interpretation order: examine Check1 residuals first; then source H versus its
objective gain; then target H/projections/overlap. Retain unfavorable rows and all
scales. Repeated scales/checkpoints are not independent expectation samples. More
sampled source batches at one frozen state would be a separate sampling experiment.

## Opt-in policy trajectories

The continuous/alternating/half aliases use method=dpco_policy. Original SGD/DPCO
diagnostic records and default computation paths retain their existing semantics.
On policy trajectories, diagnostic_update compares the real update to the
independently reconstructed `Gd + alpha*(Gs-Gd)` direction. The recorded
reference_delta and norm_dpco_sq still describe FULL DPCO, not alpha-scaled DPCO.

`policy_steps.csv` stores step, policy_mode, policy_alpha, executed_method and
reference for each real update. Its step is the completed update number. The
policy counter in TrainState.method_state must equal TrainState.step and is saved
with the normal checkpoint; observation/replay and compile warmup never advance it.
`--alternating-first sgd|dpco` selects the initial phase (default sgd). The observer
uses the same configured offset at every state, not a new first step at each node.
Metadata records alternating_first; resume configuration must keep it unchanged.
Older metadata without this field means the original SGD-first convention.

The existing diagnostic_paired/transfer streams always compare full DPCO to SGD.
At scale1, the real endpoint is included only if alpha is exactly0 or1. For half,
both endpoints of that full comparison are hypothetical and actual_update_in_pair
is false. This avoids relabeling a half step as an original Theorem2 experiment.

The additional `diagnostic_policy.csv` compares the active policy to SGD on each
enabled source objective and training-audit target. It records policy mode/alpha,
target identity, LR scale, initial/SGD/policy CE, policy-minus-SGD difference,
alpha-scaled objective_gain, first_order_residual and finite/actual flags. Source
and audit objectives use the same frozen inputs/masks/train-BN convention as the
full pair. Scale1 includes the REAL after-parameters; smaller scales reconstruct
the scheduled alpha at the same before-state, not another alternation phase.
This stream is emitted only for policy jobs and follows the existing individual
paired/transfer switches and shared CSV buffering/recovery.

SGD policy rounds have zero theoretical correction gain; numerical differences
between the real fused SGD endpoint and independent reconstruction may remain.
Do not treat such differences as a measured nonzero DPCO correction. Policy
diagnostics are overhead and are disabled independently of the training policy.
