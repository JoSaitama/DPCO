# Experiment recording contract

This implements the agreed classification observation protocol. It does not add an
ImageNet input pipeline, a ResNet implementation, or TPU host isolation. Use the CPU
smoke scripts before integrating those adapters. Changes to future observation fields,
sample dumps, or new tasks must be discussed before implementation.

## Identity and configuration

One name is generated at startup and reused beneath results/logs, results/runs and
results/files. Its format is:

```text
{dataset}_{data_mode}_{model}_{method_label}_{seed}_{initial_lr}_{YYYYMMDD_HHMMSS}
```

Time uses the host timezone, recorded with an offset in metadata. Name collisions raise
an error; they never silently overwrite a run. Reference choice remains separate in
configuration, while directory labels use sage_e/sage_l (and explicit LOO suffixes).
SGD does not acquire an error/loss reference just because it shares the same trainer.

`metadata.json` stores the full configuration, original code commit, dirty status and source hash,
Python/dependency versions, local devices/process, dataset size, evaluation mode, and
caller-supplied experiment metadata. The production data/model adapter must supply
class manifest hash/mapping, split sizes/identities, tail policy, initialization source,
precision, actual model/data identity, augmentation and iterator conventions. These
cannot be inferred reliably by the generic loop. Keep the source tree corresponding
to a dirty run: a commit hash alone does not reproduce uncommitted code.

`training.total_steps` is the update budget. Fractional progress schedules are rounded
up to the first actual step crossing a threshold; several milestones can share one step.
`data.mode` currently labels clean data; mismatch modes require a real data adapter,
not just renaming a directory. Preset LR/budget values are integration placeholders.

## Records and meanings

All accuracies use fractions in files. Convert to percentages only in plots/tables.
`step` is the number of completed updates. The train row at step k describes the batch
forward at parameters theta_(k-1), before update k. Validation/probe rows at step k
evaluate theta_k. Losses are original, unsigned CE, not the signed weighted objective.

| Stream | Fields | Purpose |
| --- | --- | --- |
| train_steps.csv | step, samples_seen, epoch_progress, learning_rate, valid_batch_size, train_ce, train_top1, elapsed_wall_s | Per-batch loss/accuracy curves and diagnostics |
| evaluation.csv | step/progress, eval_sample_count, val_ce, val_top1, val_top5, elapsed_wall_s, eval_duration_s | Generalization curves, final results, and best selection |
| theorem1.csv | counts, reference validity, projections, Delta, eta*Delta; node rows also retain independent formula residual and direction norms | Aggregation identity and reference gain |
| theorem2.csv | objective/mode/batch identity, frozen subset count, LR/scale, initial and hypothetical SGD/SAGE losses, difference/LR, first-order residual, direction norms, validity flags | Same-origin finite-step probes |
| theorem3.csv | per-step cumulative reference gain and LR sums; node audit loss/gradient norm, target gain, transfer difference, sample energy, direction norms | Exact reference-gain accumulation and sparse target-transfer diagnostics |
| timing.csv | event, step, duration, window updates/examples, throughput, wall time, checkpoint bytes, device-specific peak memory when supported | Efficiency and resource accounting |
| summary.csv | completion/failure status, final validation CE/top-1/top-5, best validation top-1/step, steps/examples, total wall time | One-run table inputs; failures remain visible |

`samples_seen` excludes padding but includes repeated visits across epochs.
`epoch_progress = samples_seen / train_sample_count`; the data source must implement
the declared tail policy consistently. With drop-last it describes sample exposure,
not an implicit promise to cover the omitted examples.

Every step produces a train row. The recorder batches file writes, not sampling:
`csv_flush_steps=50` retains all 50 rows. Normal exit and checkpoint saves flush any
partial buffer. Crashes can lose an unwritten tail. Terminal and persistent train.log
receive the same structured JSON records, flushed independently. Log frequency does
not downsample the training CSV. CSV schemas are centralized in recording.py.

## Schedules

```toml
[logging]
mode = "steps"                # "percent" uses total update budget, not each epoch
interval = 5                  # steps=1/5/N; percent=1 gives up to 100 thresholds
csv_flush_steps = 50
timing_window_steps = 50

[evaluation]
mode = "epochs"               # alternatively "steps"
interval = 1.0                # epochs may be 0.25 or 0.5

[checkpoints]
latest_every_steps = 100
at_epoch_end = false

[probes]
enabled = false
theorem1 = true
theorem2 = true
theorem3 = true
fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
batches_per_node = 3
lr_scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
```

Always evaluate the full validation set at initialization and the final budget, plus
periodic triggers. Coincident triggers are deduplicated. Validation sums per-example
CE and correct counts, then divides by the actual total; unequal tail batches are not
averaged equally. Its adapter must disable training randomness/BatchNorm updates.
No candidate validation/probe model state is committed. Best is full validation top-1;
strict improvement replaces best, ties keep the earlier state. A separately held-out
test set, if used, must not select best or tune the algorithm.

Percent logging maps thresholds to actual updates. Tiny budgets cannot produce 100
distinct model states, and crossed thresholds on the same update produce one record.

## Read-only theorem measurements

The common-reference probes currently support full sage, not LOO. They are optional;
when disabled the runner performs no probe forward/gradient work and creates no theorem
CSV. SGD never computes extra reference gradients just to match SAGE diagnostics.

T1 lightweight scalars reuse the existing scores and direction each step when enabled.
No full per-sample gradient matrices are saved. At probe nodes, compute both
`Delta = r^T (Gs-Gd)` and `2/B * sum_negative |r^T gi|`, retaining their residual.
Empty references and zero gains remain recorded, not treated as strict wins.

T2 holds inputs, model state, common reference, and the selected current error mask
fixed. Evaluate hypothetical SGD and SAGE updates at every configured LR scale. These
updates do not alter real parameters, LR, training source, or training RNG. SAGE-E's
frozen error CE and complete batch CE are separate objectives; the latter uses its own
first-order target projection. An empty error subset has no error-CE row. All scales
are kept, including failures and steps without an advantage. A probe adapter must use
deterministic inputs and explicitly identify its evaluation/objective mode. A result
in eval mode does not verify a different training-mode BatchNorm objective.

T3 accumulates `sum eta_t Delta_t` exactly across every observed training update.
Audit loss/gradient, `grad F_audit^T (Gs-Gd)`, reference-transfer difference, sample
energy, and direction norms are measured only at configured nodes. The audit objective
must be a fixed differentiable training measurement with fixed processing/model-state
semantics. It is not validation data fed back into the update. Sparse audits do not
verify a full per-step convergence inequality. Changing batch/error losses cannot be
telescoped as though they were one fixed F. No unknown global smoothness constant is
estimated here and no convergence-rate or universal SAGE advantage is asserted.

For an exact fixed-reference theorem experiment, supply a separate controlled protocol;
current-batch error-reference training is a target-transfer experiment. Gradient variance
requires repeated batches at the same frozen parameters, a future Step 4 protocol.
Accuracy variability across seeds is not gradient variance.

## Time and resources

Per-step wall timestamps are host dispatch positions; async JAX completion can lag.
Training throughput synchronizes once per observation window. Windows can end early for
logging, validation, checkpointing, probes, or CSV flush. A 1-step log interval therefore
can cost more than 5-step/percentage logging. Do not infer pure device execution time
from differences between per-step dispatch timestamps.

Compile/warmup, active training windows (including input loading and all mandatory SAGE
work), evaluation, probes, and checkpoint events are recorded separately. Warmup uses
one batch and discards its candidate update, then restores the source state. Later new
shapes can cause JAX recompilation and will be included in the affected train window;
production adapters should pad tails to stable shapes. Event durations are observed,
not an exact additive decomposition under asynchronous I/O. Buffered recording overhead
still contributes to real total wall time. Throughput is not promised to be unchanged.

At initial/final nodes query each device's supported peak allocation counter. A blank
value means unavailable; CPU absence is not zero TPU HBM. No per-step memory polling or
full parameter transfers are used for logging. Device scalar reductions detect non-finite
parameters; errors flush a failed summary and persistent log. They do not silently drop
failed runs. Future detailed profiler measurements require a separate agreed protocol.

## Checkpoints and exact recovery

Seven logical slots: initial, p25, p50, p75, final, best, latest. The first five are fixed
milestones. Latest uses a configurable step interval and optionally epoch boundaries.
`index.json` maps slots to shared step snapshots. Publication is atomic; a temporary
snapshot can temporarily require additional space. Old unreferenced files are pruned.

Snapshots contain TrainState (parameters, model buffers, optimizer and method states,
step), seed/key convention, source snapshot (ordering/augmentation RNG and cursor),
cumulative sums, best-selection state, last evaluation, elapsed-time offset, and exact
CSV row cursors. Production source adapters must snapshot all state needed to replay
their iterator, including shuffle/prefetch behavior; seed/step alone is insufficient.
This core supports one process; no distributed checkpoint guarantee is implied.

Resume requires the existing run name and matching config. Load only trusted local
pickle snapshots. CSV rows after the chosen snapshot are removed before continuation,
preventing duplicated future measurements. Persistent logs remain append-only and mark
the resume event. Future checkpoint aliases are dropped when resuming an older state.
The directory is a single-writer run; concurrent processes must not share it.

## Offline plots and tables

Use the same budget and evaluation protocol for SGD/SAGE. Plot against step, samples,
and elapsed wall time. Integrate accuracy/loss curves offline over a declared common
horizon; this curve integral is not ROC-AUC. Final means the last fixed-budget state,
not best. Best is reported separately. Across independent seeds compute final mean/std
offline. Target accuracy/loss and interpolation rules must be declared before reporting
time-to-target; missing targets must be shown as not reached. No extra training forward
passes are required merely to draw these curves.
