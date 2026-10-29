# DPCO: historical reference and configurable negative contributions

This implementation covers the three agreed methods: ordinary Adam, DPCO with
filtering and Adam, and DPCO with flipping and Adam. Both interventions use the
same historical reference. `dpco_filter` / `dpco_flip` are launcher conveniences;
the saved method name for both is `dpco`.

## Algorithm

At completed step `t-1`, evaluate the original valid batch at the current
parameters. With valid sample count `B`:

```text
g[t,i] = gradient of ordinary observed-label CE for sample i
G[t]   = sum_i g[t,i] / B
H[0]   = 0
H[t]   = beta_ref * H[t-1] + (1-beta_ref) * G[t]
r[t]   = H[t] / (1-beta_ref**t)
q[t,i] = dot(r[t], g[t,i])
w[t,i] = negative_weight if q[t,i] < 0 else 1
d[t]   = sum_i w[t,i] * g[t,i] / B
```

The reference and contribution weights are held fixed when computing `d`.
`H` always consumes raw `G`, before editing, L2 or Adam preconditioning. The
reference includes the current batch; the first reference is its raw mean.
`beta_ref=0` reproduces batch-only scoring. Bias correction is positive and does
not change the sign decisions. Reference state is independent of Adam state.

`negative_weight` is a finite scalar:

| Value | Contribution of a negatively aligned sample |
|---|---|
| `0` | Zero contribution: filter/discard |
| `-1` | Reverse the whole sample gradient: flip |
| `-0.5` | Reverse with half magnitude |
| `0.5` | Retain with half magnitude |
| `1` | Retain unchanged; useful ordinary-Adam control |

Zero and positive alignments always receive weight 1. There is no clipping or
normalization of sample gradients. All aggregates use the original valid `B`,
including after filtering. Filtering acts on gradient contributions after the
forward pass; it does not remove examples from BatchNorm statistics.

Adam receives `d` and updates its standard first and second moments from `d`
and its coordinatewise square, with bias correction and epsilon outside the
square root. It does not take the mean of squared per-example gradients. The
implementation uses Optax `scale_by_adam` with `eps_root=0`. Optional existing
kernel-only coupled L2 is added after contribution editing and before Adam
moments; default L2 is zero. This is Adam, not AdamW. Existing constant, cosine
and inverse-square-root learning-rate schedules still count completed updates.

For filter/flip (more generally, `negative_weight <= 1`), editing does not decrease
the projection onto this reference. This is an algebraic property of `d`; it
does not guarantee better clean loss, generalization, or an improved actual Adam
step. Negative alignment is not a noise-label detector. In particular, an EMA of
noisy gradients still contains label bias. Values above 1 are permitted for
experiments but need not preserve even the reference-projection inequality.

## Configuration and commands

The separate preset is `configs/tpu/dpco/imagenet100_resnet50.toml`. It uses the
existing ImageNet-100/ResNet-50 adapters, `beta_ref=.9`, Adam `(.9,.999,1e-8)`,
LR `.001`, and a 100-update pilot budget. These are untuned starting settings,
not a completed benchmark recipe. Existing presets keep their original behavior.

Relevant TOML fields:

```toml
[method]
name = "dpco"
negative_weight = -1.0

[reference]
name = "ema"
beta = 0.9

[training]
optimizer = "adam"
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
learning_rate = 0.001
total_steps = 100
```

Inspect all three commands without loading JAX, accessing data or launching jobs:

```bash
python scripts/launch_tpu.py \
  --config configs/tpu/dpco/imagenet100_resnet50.toml \
  --mode train --methods adam dpco_filter dpco_flip --seed 42 --dry-run
```

Inspect a custom negative weight and reference decay:

```bash
python scripts/launch_tpu.py \
  --config configs/tpu/dpco/imagenet100_resnet50.toml \
  --mode train --method dpco --negative-weight -0.5 --reference-beta 0.95 --dry-run
```

`--method dpco --negative-weight 0` and `--method dpco --negative-weight -1`
are the canonical parameterized interface. An explicit weight conflicting with
`dpco_filter` / `dpco_flip` fails rather than silently changing the named preset.
In a mixed `adam dpco` suite, a shared negative weight applies only to DPCO.
`--adam-beta1`, `--adam-beta2`, `--adam-epsilon`, `--lr` and the existing shared
data/model/seed/budget controls reach both launcher and worker.

To inspect iNaturalist instead, append `--dataset inaturalist2021 --data-dir PATH`
with the prepared subset path. ImageNet-1K uses `--dataset imagenet1k`. Classifier
mapping and input adapters remain those of the selected dataset. Removing
`--dry-run` on a configured TPU server starts actual jobs; multi-method runs are
sequential and use the same initial seed, augmentation and batch order.

This change implements the optimizer/method comparison. It does not inject label
noise, create clean/corrupted manifests, or add the proposed group-specific CE
evaluation. The current real-data entry still accepts only `data.mode=clean`;
that label means no *injected* noise and does not certify perfect original
annotations. No formal dataset run has been executed for this implementation.

## State, numerical checks and recording

Reference state `(H, count)` and Adam state `(m, v, count)` are saved in the
existing complete checkpoint. Counts must match the completed training step.
Resume validates state types, counts, tree shapes and finite values before
rewinding any CSV; second moments must also be nonnegative. Configuration
changes, including weight and betas, reject resume. Start a new run to change them.
Legacy metadata missing these new fields is compared using the old defaults.

Compilation warmup computes a candidate and discards it. Validation and warmup
do not commit reference/Adam state or consume training data. The JVP backend
uses exact global-batch VJP/JVP operations without constructing a `[B,P]`
Jacobian. BatchNorm buffers are committed once per real update.

New DPCO runs add `dpco_steps.csv` alongside ordinary CE/accuracy CSVs. Its step
counts completed updates; the scores and gradients describe the pre-update
parameters. It records the negative count, edited count, negative weight,
reference update count, reference/raw/edited norms, reference/raw cosine and
`r dot (d-G)`. Cosine is recorded as zero if either vector is zero. The gain is
not an Adam loss-decrease claim. Runs check finite method/optimizer state as
well as model parameters and ordinary training metrics.

Run directory labels and lifecycle status distinguish EMA decay, weight and
Adam. Full metadata contains all optimizer hyperparameters. The old SGD report
script explicitly rejects Adam runs; its original plots should not be used to
interpret this new comparison. Legacy theorem/diagnostic/mechanism observers
also reject Adam/EMA configurations because their existing proofs and replay
assumptions use other update rules.

Local tests cover independent NumPy recurrences, arbitrary weights, zero scores,
padding, audit/JVP parity, BatchNorm state, whole-trajectory weight-one control,
complete resume, corrupt/missing state, parent/worker option parity, the real
entry with tiny CPU fixtures, and two-device CPU global-batch semantics. They
do not establish TPU memory/throughput or accuracy benefits.
