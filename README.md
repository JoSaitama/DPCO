# DPCO

Research code for SGD and sample-contribution correction with task/reference/method
separation. Repository code, comments and documentation use English.

## Current stage

The new [DPCO + Adam implementation](docs/dpco_adam.md) supports ordinary Adam
and a current-inclusive raw-gradient EMA reference with configurable negative
contributions: `--method dpco --negative-weight 0` filters, `-1` flips, and other
finite weights are accepted.

[CIFAR-100N preparation and training](docs/cifar100n.md) now connects the authors'
human annotations to the same three-method Adam flow, native 32x32 ResNets, and
fixed clean/corrupted ordinary-CE evaluation. Use
`python scripts/prepare_cifar100n.py --stage plan` and
`./run_cifar100n_suite.sh --dry-run` to inspect preparation and jobs. Its human
noise percentage is measured (40.2%), not an adjustable corruption generator.
The existing ImageNet preset remains `configs/tpu/dpco/imagenet100_resnet50.toml`;
synthetic noise injection into ImageNet is still separate planned work.

[Opt-in TPU mechanism checks](docs/tpu_mechanisms.md) extend the real classification
suite with both references, absolute LR candidates, a genuinely fixed train-mode
bank, finite-bank moments and continuous-window accounting. They are off by
default. `run_dpco_mechanism_suite.sh` is the separate TPU wrapper; the existing
synthetic theory suite remains CPU-only. Development tests do not certify TPU.

An independent [CPU theory/LR validation suite](docs/theory_validation.md) tests
the original SGD/DPCO-E/DPCO-L on smooth synthetic classification data, with
analytic counterexamples, conditional moments and full-target cumulative bounds.
Run `./run_dpco_theory_suite.sh --dry-run` to inspect its grid. It does not launch
TPU training, download data, save checkpoints or change the production/policy paths.

The first real-device pipeline targets ImageNet-100, ResNet-50, and two sequential
SGD/DPCO-E runs on node-ziao1 worker0. Worker1 remains idle and is not a peer. Later
ImageNet-1K/ResNet-101 checks use the same adapters after first-stage validation.

Implemented: a Flax ResNet V1.5 adapter, pinned Hugging Face data preparation, host-local
JAX data parallelism, exact common-reference VJP/JVP scoring, nohup launch, structured
recording, seven logical checkpoint slots, and optional T1-3 probes.

Local CPU checks validate mathematical parity, model shapes, input/recovery behavior
and launcher guards. They do not certify TPU/libtpu startup, isolation, throughput,
accuracy or convergence. The first server smoke must verify those hardware properties.
Pilot learning rate/budgets are not experimentally tuned classification recipes.

See [worker0 TPU protocol](docs/tpu_training.md) for setup, prerequisites, data costs,
real startup commands, and troubleshooting boundaries. See [recording contract](docs/recording.md)
for metric/checkpoint definitions.

[iNaturalist 2021 preparation and training](docs/inaturalist_subset.md) supports the
fixed 140k-train/20k-validation proportional selection from official full data.
`run_inaturalist_suite.sh` reads the completed local subset without HF access,
another download or an image copy. It shares the existing trainer, methods,
metrics, status TSV and checkpoint implementation.

[Real-update training diagnostics](docs/training_diagnostics.md) provides optional
early-step update checks, paired source-batch losses and transfer to other training
batches. Enable with `--diagnostics` through either shell suite; default is off.
The observer preserves the selected optimizer and original compiled training step.
Its switches are independent of the older fixed-batch theorem probes.

## Opt-in continuous, alternating and half DPCO

Existing `sgd`, `dpco_l` and `dpco_e` commands keep their original implementations.
New method aliases select a separate policy wrapper:

| Loss reference | Error reference | Correction strength alpha |
|---|---|---|
| `dpco_l_continuous` | `dpco_e_continuous` | 1 on every update |
| `dpco_l_alternating` | `dpco_e_alternating` | 0, 1, 0, 1, ... by default; first method is configurable |
| `dpco_l_half` | `dpco_e_half` | 0.5 on every update |

The direction is `Gd + alpha * (Gs - Gd)` before the common learning-rate update.
Half means filtering negative alignment with the original B denominator (the
existing DTV calculation), not half the learning rate or separate complete SGD/DPCO
computations. Half still needs reference/alignment work; half strength does not
promise half the wall time.
Alternating uses a compiled conditional: SGD rounds skip reference/alignment work.
Each round commits one model state and one parameter update. The checkpointed
policy counter preserves phase on resume, including resumes after an odd update.

For either alternating alias, `--alternating-first sgd` (default) selects
SGD -> DPCO -> SGD, while `--alternating-first dpco` selects DPCO -> SGD -> DPCO.
This suite option applies only to alternating jobs; baseline, continuous and half
jobs in the same grid are unchanged. It reaches both the nohup parent and child.
The effective config stores method.alternating_first; changing it on resume is
rejected. Runs using the nondefault DPCO-first order add `_first_dpco` to their
directory label and status method value. Per-step policy_alpha/executed_method
records show the actual order, and diagnostic metadata records its first method.

An example server smoke suite (real TPU through the existing nohup launcher):

```bash
nohup ./run_dpco_suite.sh --mode smoke --steps 5 --batch-size 16 \
  --methods dpco_l_continuous dpco_l_alternating dpco_l_half sgd --seed 5 \
  --no-theorem-probes --diagnostics --diagnostic-steps 0 1 2 3 4 \
  --no-diagnostic-transfer --diagnostic-lr-scales 1 0.5 \
  >/dev/null 2>&1 < /dev/null &
```

Both ImageNet and iNaturalist suites accept these aliases. For theory checks keep
momentum, Nesterov and L2 off, with the same constant LR/data/seed/budget for all
methods. Policy jobs use distinct run labels and write `policy_steps.csv`; normal
training JSON includes policy_mode, policy_alpha and executed_method. The shared
status TSV keeps its existing start/end-only schema and identifies policy mode
in its method value. Original jobs create no policy CSV.

Use `--diagnostics` for the new modes. Legacy `--theorem-probes` remains restricted
to the original full DPCO method and is not silently reinterpreted. Full-DPCO
counterfactuals remain in diagnostic_paired/transfer; diagnostic_policy records
the active policy versus SGD, with alpha-scaled objective gain and the real policy
endpoint at scale 1. See [diagnostic semantics](docs/training_diagnostics.md).

## Exported CSV guide and local training plots

See [the result CSV guide](docs/result_csv_guide.md) for every stream's fields,
T1/T2/T3 scope, active-policy versus full-DPCO candidates, and Mac instructions.
`scripts/plot_training_results.py` reads exported CSV/metadata without importing
JAX or training code. It compares initial LRs by line style, uses blue for SGD,
red for DPCO-E and orange for DPCO-L, and exports PNG/PDF/SVG plus provenance.
Warmup comparisons separate fixed-10% LR effects (`warmup10`) from fixed-LR
10%/30% SGD-prefix effects (`warmup_percent`); both include matching original
SGD baselines. See the guide for percentage/LR selection parameters.
Raw files are never edited; repeated comparable runs require explicit selection.

## Optional SGD prefix before DPCO

`--dpco-warmup-percent P` runs the first `ceil(total_steps * P / 100)`
updates with SGD, then switches to the original full DPCO-E/L rule. Default
`P=0` preserves the original method and compiled training path. `P=100` runs
SGD for the entire budget. This is a method warmup, not a learning-rate warmup:
the configured LR schedule continues on absolute completed updates throughout.
For example, `--steps 5000 --dpco-warmup-percent 10` uses SGD on updates 1–500
and DPCO from update 501. The SGD prefix is part of the total training budget.

```bash
./run_dpco_suite.sh --dry-run --methods dpco_e dpco_l sgd --seed 5 \
  --steps 5000 --batch-size 256 --lr 0.01 --dpco-warmup-percent 10 \
  --no-theorem-probes
```

The shared option applies to `dpco_e`, `dpco_l` and their `_continuous` aliases.
SGD, half and alternating suite jobs keep their original behavior. In TOML set
`method.dpco_warmup_percent = 10.0`; nonzero values on unsupported methods are
rejected. Percentages must be finite and between 0 and 100, including endpoints.

Warmup runs include `_warmup10pct` (or the chosen percentage) in their run names
and lifecycle method label. `policy_steps.csv` records `policy_mode=warmup`,
`policy_alpha=0/1` and `executed_method=sgd/dpco` for each completed update;
printed train JSON includes the same fields. The config in `metadata.json`
preserves the percentage, and suite dry-run reports the resolved update count.
Checkpoint recovery preserves the absolute phase; changing the percentage or
training budget on resume is rejected. Momentum, if enabled, is carried through
the switch without resetting; other update controls and batch averaging are
unchanged. Compilation warmup discards its candidate and consumes no prefix step.

Legacy `--theorem-probes` is incompatible with this phased method; use
`--no-theorem-probes`. Optional `--diagnostics` handles both phases and records
the active direction separately from full-DPCO counterfactuals. Extended
`--mechanisms` supports the original `dpco_e/l` warmup runs: `actual_endpoint`
is `sgd` during the prefix and `dpco` afterwards, while source T1/pairs and
conditional moments still describe hypothetical full-DPCO directions. The
extended observer retains its existing restriction on `_continuous` aliases.
This option enables a controlled experiment; it does not guarantee stability
or an advantage over SGD.

## Start on worker0

Use a separate Python 3.11+ environment. Do not reuse another project's dependencies:

    cd ~/Project_DPCO/DPCO
    python3 --version
    python3 -m venv ../.venv
    source ../.venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e '.[train,tpu,test]'
    python -m pip check

An existing HF login or HF_TOKEN is reused. GitHub SSH login is unrelated. If the
official ImageNet gate is not approved, follow the dataset URL shown by the error;
tokens never belong in configuration or Git. Optional interactive checks:

    hf auth whoami
    hf auth login

First run offline foreground CPU sanity:

    python scripts/sanity_check.py --config configs/tpu/imagenet100_resnet50_sgd.toml
    python scripts/sanity_check.py --config configs/tpu/imagenet100_resnet50_dpco_e.toml

CPU scripts were renamed from smoke_test.py/smoke_recording.py to
sanity_core.py/sanity_recording.py. Their synthetic linear classifier is not ResNet.

Then launch a real TPU smoke with the supplied nohup wrapper:

    python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sgd.toml

It prints the PID and exact per-run stdout.log path. Monitor that path with tail -f.
After SGD completes successfully, launch DPCO-E:

    python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_dpco_e.toml

For these single direct Python launches, no outer nohup is needed: the launcher uses detached input,
unbuffered output and a new session. Both jobs must run sequentially on worker0.
A host lock rejects overlapping jobs. No CPU/GPU fallback is permitted. A started
PID alone does not establish success; look for tpu_ready and completed updates.

`results/logs/run_status.tsv` is a shared, append-only status journal across runs/seeds.
Each attempt writes only start/running and end/completed-or-failed. It has no step/error
columns or progress rows. Detailed errors and steps stay in the isolated per-run logs.
Train JSON records start with step, event, train_ce and lr; CSV keeps learning_rate.
See the recording contract for fields and abrupt-termination limitations.

Smoke uses real TPU, real ResNet-50, and a small cached subset of original ImageNet
images, with three updates by default. It is not a complete dataset/theory benchmark.

## Shell suites and seed overrides

Use the executable root script for training suites. It defaults to train mode,
ImageNet-100, ResNet-50, DPCO-E, clean data and seed 42, using the TPU pilot config.
The global batch defaults to 256 for train/validate and 8 for smoke. An explicit
`--batch-size` overrides either mode; all four local TPU devices share this batch.
The Python interpreter defaults to outer Project_DPCO/.venv/bin/python; set DPCO_PYTHON
to another project-specific interpreter path if needed. The script works from any cwd.

Example: three sequential DPCO-E TPU smokes, with three updates each:

```bash
nohup ./run_dpco_suite.sh \
  --mode smoke --dataset imagenet100 --model resnet50 \
  --methods dpco_e --seeds 10 15 22 --data-mode clean \
  >/dev/null 2>&1 < /dev/null &
```

`--seed` and `--seeds` are aliases and accept one or more integers, for example
`--seed 42` or `--seeds 10 15 22 46`. ImageNet presets and the config default use
42; the iNaturalist pilot uses training seed 5. Explicit CLI seeds override the
config; an explicit seed in a custom config is
honored when the CLI omits seeds. To resume an older seed-0 run, pass `--seed 0` and
retain all its original settings. Resume permits just one seed/method.

`--methods dpco_e dpco_l sgd` runs each method for each listed seed, in seed-major
order. The method labels select dpco/error, dpco/loss, or unmodified SGD. Select
`--dataset imagenet100|imagenet1k|inaturalist2021`, `--model resnet50|resnet101`, and `--data-mode clean`
(`--datamode` is an alias). Mismatch modes are reserved and rejected before any job
starts; a directory label never silently simulates noisy data. All grid configurations
are validated before execution. Runs remain worker0-only and serial; failure stops
the remaining suite. The coordinator waits for child exit and successful status,
not just successful process creation. Individual jobs still use the nohup TPU launcher.

For formal training, use `--mode train` and configure `--steps`/the config budget;
the default 100-step pilot is not a tuned final training recipe. Optional `--batch-size`
overrides the logical batch. The wrapper always waits, so use the outer nohup above
to keep the whole queue alive after SSH disconnect. Preview without starting jobs:

```bash
./run_dpco_suite.sh --methods dpco_e dpco_l sgd --seeds 10 15 --dry-run
```

The independent `run_inaturalist_suite.sh` entry accepts the same options. Pilot
defaults: ResNet-50 from scratch, training seed 5, 2,000 steps/method, B=256,
constant LR=0.01, L2/momentum/Nesterov off, validation every epoch, probes off,
method order `dpco_e sgd dpco_l`. These are screening settings, not a tuned recipe.

```bash
./run_inaturalist_suite.sh --dry-run
mkdir -p results/logs/inaturalist2021_seed5_suite
nohup ./run_inaturalist_suite.sh --mode train --seeds 5 \
  --methods dpco_e sgd dpco_l --steps 2000 \
  --data-dir ../data/inaturalist2021_full_t140000_v20000_s42 \
  >results/logs/inaturalist2021_seed5_suite/launcher.log 2>&1 < /dev/null &
```

`--data-dir` is repository-relative or absolute and reaches every child. For
iNaturalist it is the directory containing the completed manifest and SQLite
database; for ImageNet it is the fingerprint-cache parent. Training seeds never
change subset membership. See the iNaturalist guide for smoke/probes/resume.

Set `--checkpoint-every-steps 10` (or 100, etc.) to control the rolling latest
recovery checkpoint. The positive integer overrides `[checkpoints]
latest_every_steps`; its default is 100 unless a custom config sets another value.
It is shared by all suite jobs and recorded in each effective configuration.
Use `--no-checkpoints` to disable **all** model snapshots and the slot index,
including initial, milestones, best, final and latest. Logs, metrics, diagnostics
and completion status are still saved. This avoids checkpoint serialization and
disk writes, but saves no trained model and provides no interruption recovery.
`--checkpoints` re-enables saving; omission honors `[checkpoints] enabled`, whose
default is `true`. An empty per-run `results/runs/` directory may still exist.
Resume requires checkpoints enabled and a matching original configuration.
This does not retain every periodic snapshot: initial/p25/p50/p75/final/best stay
protected, and only obsolete unreferenced snapshots are deleted. Initial/final
saves are mandatory while checkpointing is enabled; milestone and best saves remain independent of the interval.
The order is full snapshot write/flush/fsync, atomic snapshot publication, atomic
index publication, then deletion of old unreferenced snapshots. A write or index
publication failure preserves the previous indexed recovery point. Extra transient
disk space is required while both old and new snapshots exist. Logging and
validation intervals are independent; this option controls only periodic latest
saves. Resume currently requires the same original interval in the matching config.

Validation cadence is controlled independently with mutually exclusive options:
`--eval-every-steps 1000` or `--eval-every-epochs 1` (fractional values such as 0.25
or 0.5 are supported for epochs). Steps must be positive integers; epochs must be
positive finite values. Supplying both fails before any job starts. Omission honors
the config's evaluation mode/interval; the shipped default is every 1 epoch of
`samples_seen / train_sample_count` progress. These options reach every suite child
and are saved in the existing effective evaluation configuration. They do not
change which validation samples are evaluated: each trigger evaluates the full
configured validation split, which is intentionally a small subset in smoke mode.
New runs always evaluate at step 0 and all runs evaluate at the final step, even
when the periodic interval exceeds the budget. Coincident final/periodic triggers
evaluate only once. Checkpoint or log events do not force extra validation. Best
is selected only from actual full-validation results, and its selection can change
with cadence. Resume requires the original evaluation mode/interval and continues
the schedule from the restored step/sample count without duplicate evaluation rows.

L2 regularization is optional and off by default: `training.l2_coefficient = 0.0`.
The suite accepts `--l2-coefficient 0.0001` to enable it, or an explicit
`--l2-coefficient 0` to override a nonzero custom config. The same value applies to
all methods/seeds in that suite and is saved in each run's full configuration.
When zero, the updater follows the original computation: no L2 gradient, parameter
norm reduction, extra optimizer state, or device synchronization is added. This
does not promise identical measured wall time, which varies between runs.
When enabled, coupled L2 adds `lambda * parameter` only for leaves with `ndim > 1`,
after sample weighting/aggregation. References, scores, unsigned CE and accuracy are
unchanged at the same model state; subsequent training trajectories can differ.
Biases and BatchNorm scale/offset are excluded. This is not decoupled AdamW decay.
Current theorem probes require L2=0 and momentum=0 and reject nonzero values before launch, since
their objectives and hypothetical updates exclude regularization. Resume requires
the original coefficient; older runs without this field are treated as L2=0 without
rewriting their metadata. No benefit to validation accuracy is assumed in advance.

Optimizer and schedule options are shared by every method in a shell suite:

| Shell option | Config field under `[training]` | Default |
| --- | --- | --- |
| `--lr` / `--learning-rate` | `learning_rate` | 0.01 (untuned pilot) |
| `--momentum` | `momentum` | 0 (off) |
| `--nesterov` / `--no-nesterov` | `nesterov` | false |
| `--lr-schedule constant\|cosine\|inverse_sqrt` | `lr_schedule` | constant |
| `--min-lr` | `min_learning_rate` | 0 |
| `--l2-coefficient` | `l2_coefficient` | 0 (off) |

Momentum uses `v_next = momentum * v + g`, without EMA normalization/dampening.
Here `g` is the method's completed direction plus optional coupled L2. Plain
momentum updates with `v_next`; Nesterov updates with `g + momentum * v_next`.
Only a nonzero momentum allocates its velocity tree. Checkpoints preserve that
tree from the initial snapshot onward; warmup does not consume momentum history.
Nesterov requires nonzero momentum and is never implicitly enabled by momentum.

`--lr` is the actual constant rate, cosine peak or inverse-square-root initial rate, with no automatic batch-size
scaling. Cosine has no implicit warmup. For T>=2 updates indexed 0..T-1, it uses
`min_lr + (peak_lr-min_lr)*(1+cos(pi*step/(T-1)))/2`. Thus the first update uses the
peak and the last uses the floor, including zero. A one-update run uses the peak;
queries after the budget return the floor. A zero-rate final update leaves finite
parameters unchanged but still processes the batch and commits model/optimizer
state. At zero-rate probe nodes, T2 division-by-rate fields are blank and strict
gain eligibility is false; zero is never used as a denominator. A nonzero minimum
in constant or inverse_sqrt mode is rejected rather than silently ignored.

`inverse_sqrt` uses `learning_rate / sqrt(step + 1)`, where `step` is the absolute
number of completed updates BEFORE the next update, including checkpoint recovery.
The exponent is fixed at one half. It has no warmup, floor, switch point or total
budget dependence; it does not force the final rate to zero. Both the compiled
update and host-side CSV/probe rate use the same schedule implementation. Constant
remains the default. The bounded theory-motivated comparison and its limitations
are documented in [TPU mechanisms](docs/tpu_mechanisms.md#predetermined-inverse-square-root-comparison-2026-10-25).
New cosine/inverse_sqrt run directory names include the schedule immediately
before the timestamp, across logs/files/runs and lifecycle records. Constant
names retain their original format; resume continues to use the explicit saved name.

Preview a configured suite without launching or downloading:

```bash
./run_dpco_suite.sh --methods dpco_e dpco_l sgd --seeds 2 5 7 \
  --batch-size 256 --momentum 0.9 --lr 0.1 \
  --lr-schedule cosine --min-lr 0 --l2-coefficient 0 --dry-run
```

These values illustrate the interface, not a tuned accuracy recipe. Actual suites
still launch with outer nohup. For original theorem validation, retain zero momentum
and L2; a nonconstant rate does not turn the probes into a momentum theorem. Resume
requires the same rate/schedule/floor/momentum/Nesterov/batch/budget settings. Older
metadata missing the new optimizer fields implies off/constant defaults without
rewriting files; explicitly retain the old batch (for example `--batch-size 8` or
64) when resuming an old run whose preset batch has since changed to 256.

The lower-level launch_tpu.py also accepts these overrides; multiple jobs always wait
serially. A single direct launch remains detached unless `--wait` is specified.
The earlier untracked three-seed helper is obsolete and can be removed after its queue
finishes. Remove the old smoke-test TSV before updating; no backup, migration or
automatic result deletion is performed. Incompatible TSV headers are rejected.

## Data and model provisioning

The model architecture ships with the code, adapted from a pinned Google Flax source.
The selected optimizer experiment uses random initialization: no pretrained weight
download is necessary and no HF checkpoint is silently used.

Data is streamed from the official ILSVRC/imagenet-1k HF repository at a pinned commit.
The versioned CMC manifest fixes 100 synsets, original 1K labels and the 0..99 mapping.
Only selected original image bytes are cached in outer Project_DPCO/data/ using SQLite.
No full archive or duplicate Arrow dataset is deliberately materialized. Full 100-class
preparation may still read much of the ~154 GB original train/validation source over
the network; limited smoke preparation stops after a small number of selected images.
Existing completed caches need no further HF access. Images/tokens stay outside Git.

For full ImageNet-100, expected counts are 126,689 train and 5,000 validation images.
Smoke subsets may not cover all classes. The cache identity includes revision,
manifest mapping and sample limits, preventing smoke/full cache confusion.
Disk estimates include a 3 GB checkpoint reserve; full 1K needs a larger disk than
the previously discussed host free space.

Training shuffles a complete local epoch deterministically, then applies reproducible
random resized crops/flips. Incomplete training tails are dropped so BatchNorm never
sees padding; evaluation covers every valid example with a masked padded final batch.

## DPCO-E, DPCO-L and continuous DPCO-R

All three variants use ordinary per-image cross entropy and its current-parameter
gradient. Prediction feedback selects a reference; raw gradient alignment decides correction.

| Experiment label | Method | Reference | Definition |
| --- | --- | --- | --- |
| DPCO-E / dpco_e | dpco | error | Mean gradient of currently misclassified valid batch examples |
| DPCO-R / dpco_r | dpco | correct | Mean gradient of currently correctly classified valid batch examples |
| DPCO-L / dpco_l | dpco | loss | Mean gradient of all valid batch examples |

For error, freeze the current selection E = {i: argmax(logits_i) != label_i} when
differentiating its surrogate mean loss. Rebuild it each batch. No historical error
buffer or derivative of a 0/1 objective is used. Loss means the entire current batch,
not a high-loss subset or exact population gradient.

For correct, freeze R = {i: argmax(logits_i) == label_i} within the current
valid batch and average those examples' CE gradients. Recompute R on each real
batch. This is not the gradient of accuracy or a persistent correct-sample buffer.
An empty correct set falls back to SGD; an all-correct batch makes R/L identical.
The correct-reference denominator is |R|, while the final update denominator
remains the full valid B. Every valid example is scored and can be flipped,
including correct examples that conflict with their set's mean gradient.

The first R implementation supports only continuous `--methods dpco_r`, with
no method warmup, half, alternating or LOO variant. Existing E/L/SGD commands
and optimizer defaults are unchanged. No new preset is required; the suite
override composes method=dpco with reference=correct on the existing base config.
For example, run this manually on worker0 to request a real nohup TPU smoke:

```bash
nohup ./run_dpco_suite.sh --mode smoke --methods dpco_r --seeds 5 \
  --steps 5 --batch-size 16 --lr 0.01 --lr-schedule constant \
  --momentum 0 --no-nesterov --l2-coefficient 0 \
  --no-checkpoints --no-theorem-probes --diagnostics \
  --diagnostic-steps 0 1 2 3 4 --no-diagnostic-transfer \
  --diagnostic-lr-scales 1 0.5 >/dev/null 2>&1 < /dev/null &
```

New run directories include `_dpco_r_`; metadata/status store reference=correct.
Optional paired observers label the frozen reference objective `frozen_correct_ce`
(or `source_frozen_correct_ce` in mechanism streams), separately from full-batch
and fixed-bank targets. Empty correct sets do not emit a fictitious correct-set CE.
Existing `error_count` fields retain their literal error-count meaning.
Mechanism R runs add correct-reference comparisons and bank moments; old E/L/SGD
runs retain their original observer workload. Existing synthetic CPU theory and
E/L plotting suites are not extended to R by this training change.

Score every valid contribution with q_i = r^T g_i. DPCO uses weight -1 for a negative
score and +1 otherwise, including zero. Return sum(w_i*g_i)/B using the original valid
B; preserve gradient magnitude, do not cosine/unit-normalize, and apply LR once.
An erroneous example is not automatically flipped; a correct example may be flipped.
Empty error sets fall back to SGD; all-error batches make E/L references identical.
The error-reference denominator is the error count, distinct from the update mean B.

The TPU JVP backend obtains the same full-parameter scores without retaining a [B,P]
Jacobian. A dense audit Jacobian remains only for tiny offline correctness checks.
Runtime probes also use compiled whole-batch VJP/JVP for their paired directions.
Exact per-sample energy uses a scalar-carry loop that pulls back one loss at a time,
instead of materializing [B,P] gradients or B simultaneous backward activation
trees. This retains B, global BatchNorm, references and every existing CSV field;
it is not microbatch training or a stochastic energy estimate. The energy loop
runs only when T3 is enabled and still requires up to B backward evaluations at
each probe batch. Probe costs are recorded separately; no TPU fit/time guarantee
is inferred from the final gradient tensor size alone. Training BN
uses the same realized logical batch in all derivative passes, with state committed
once. These per-example loss derivatives can include BN cross-example coupling.

SGD skips reference construction. There is no sgd_e method. dpco_e/dpco_l/dpco_r are
experiment labels; method and reference remain separate configuration fields.

## Components and other methods

    task loss / feedback
      -> reference selection + alignment signal
      -> method weights and complete batch-mean direction
      -> trainer parameter update once
      -> shared host recorder / optional read-only probes

Tasks define objectives and model-state transitions. References build common/LOO means.
Signals compute feedback/scores; methods choose contributions and return directions.
Training orchestration owns schedules, updates, checkpoints and the single recorder.
Methods/references/signals never write CSV. Probes never consume training data/RNG or
commit hypothetical updates.

| Method | Nonnegative score | Negative score |
| --- | --- | --- |
| sgd | +1 | +1 |
| dpco | +1 | -1 |
| dtv | +1 | 0 |
| random_flip | Independent Bernoulli(p) flip | Independent Bernoulli(p) flip |

Random flipping covers the whole valid batch and ignores alignment. Its expected
fraction is p; no rounded fixed count is enforced. At p=0 it is SGD; p=1 negates all
contributions. Filtering keeps original B. LOO excludes a sample only when it belongs
to its reference subset. LOO uses the audit backend and is outside the initial run;
common-reference Delta theorems do not automatically apply to LOO.

## Configuration and execution modes

The eight root presets remain core integration configurations. The two runnable TPU
profiles are in configs/tpu/ and share data, initialization, LR, B and budget.

- smoke: real TPU/model, limited real data, probes off in supplied profiles, short budget.
- validate: full selected data, configured T1-3 probes on a DPCO trajectory.
- train: full selected data and configured budget; probes normally disabled.

Example controlled diagnostics after smoke succeeds:

    python scripts/launch_tpu.py --mode validate --steps 20 --config configs/tpu/imagenet100_resnet50_dpco_e.toml

This can require a long first data preparation. Sparse audit gradients can be costly.
Reference gains and target-transfer diagnostics do not by themselves prove the
original exact-reference global convergence theorem. All scanned learning rates,
including unfavorable outcomes, remain recorded. Initial/final fixed audit semantics
and training/evaluation BatchNorm modes are explicit.

For full-data pilot training:

    python scripts/launch_tpu.py --mode train --config configs/tpu/imagenet100_resnet50_sgd.toml

--steps and --batch-size are explicit overrides. Batch size must divide the number
of local TPU devices. Topology, visible chips and timeout are configuration values.
Startup checks global/local device ownership and never initializes a peer process.

## Results and recovery

    results/
      logs/{run_name}/     stdout.log, train.log, metadata.json, launch PID records
      runs/{run_name}/     checkpoint snapshots and slot index
      files/{run_name}/    train/evaluation/theorem/timing/summary CSV

The name is dataset_data-mode_model_method-label_seed_initial-lr_YYYYMMDD_HHMMSS.
All three directories share the same name; collisions cannot overwrite an old run.
Bootstrap reservations let nohup log failures before the main recorder starts.
The entire results root is ignored by Git.

Training CSV retains every update and flushes in batches, independently of log frequency.
Full validation uses sample-weighted CE/top-1/top-5 initially, periodically and finally.
Best uses validation top-1, with earlier ties retained; final means the fixed budget end.
Checkpoints share seven initial/25%/50%/75%/final/best/latest logical slots.
Restore includes model/optimizer/method state, input cursor/randomness, cumulative
statistics and CSV positions. Resume rewinds CSV rather than duplicating future rows.

Use the same original mode/config/overrides with --resume RUN_NAME --resume-slot latest.
The data source has no asynchronous prefetch; seed/epoch/cursor reproduce input order
and augmentation. No distributed checkpoint guarantee is implied.

## Local checks and source attribution

    python -m pip install -e '.[test]'
    JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m unittest discover -s tests -v
    python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sgd.toml --dry-run
    python main.py --check-config --config configs/tpu/imagenet100_resnet50_dpco_e.toml

Tests include independent analytic directions, JVP/Jacobian parity, BN state handling,
real 50/101 parameter shapes, two-CPU global batch/BN parity, deterministic data
recovery, recorder/checkpoint retention and fail-closed launch controls. Offline
fixtures do not establish successful HF image access or TPU hardware execution.

See THIRD_PARTY_NOTICES.md and licenses/ for model source revision/license and the
class-manifest/data provenance. Local research notes/protocols remain outside this
inner code repository. No data, tokens, model weights or training results belong in Git.
