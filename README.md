# SAGE

Research code for SGD and sample-contribution correction with task/reference/method
separation. Repository code, comments and documentation use English.

## Current stage

The first real-device pipeline targets ImageNet-100, ResNet-50, and two sequential
SGD/SAGE-E runs on node-ziao1 worker0. Worker1 remains idle and is not a peer. Later
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

## Start on worker0

Use a separate Python 3.11+ environment. Do not reuse another project's dependencies:

    cd ~/Project_SAGE/SAGE
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
    python scripts/sanity_check.py --config configs/tpu/imagenet100_resnet50_sage_e.toml

CPU scripts were renamed from smoke_test.py/smoke_recording.py to
sanity_core.py/sanity_recording.py. Their synthetic linear classifier is not ResNet.

Then launch a real TPU smoke with the supplied nohup wrapper:

    python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sgd.toml

It prints the PID and exact per-run stdout.log path. Monitor that path with tail -f.
After SGD completes successfully, launch SAGE-E:

    python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sage_e.toml

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
ImageNet-100, ResNet-50, SAGE-E, clean data and seed 42, using the TPU pilot config.
The global batch defaults to 256 for train/validate and 8 for smoke. An explicit
`--batch-size` overrides either mode; all four local TPU devices share this batch.
The Python interpreter defaults to outer Project_SAGE/.venv/bin/python; set SAGE_PYTHON
to another project-specific interpreter path if needed. The script works from any cwd.

Example: three sequential SAGE-E TPU smokes, with three updates each:

```bash
nohup ./run_sage_suite.sh \
  --mode smoke --dataset imagenet100 --model resnet50 \
  --methods sage_e --seeds 10 15 22 --data-mode clean \
  >/dev/null 2>&1 < /dev/null &
```

`--seed` and `--seeds` are aliases and accept one or more integers, for example
`--seed 42` or `--seeds 10 15 22 46`. All shipped presets and the config default use
42. Explicit CLI seeds override the config; an explicit seed in a custom config is
honored when the CLI omits seeds. To resume an older seed-0 run, pass `--seed 0` and
retain all its original settings. Resume permits just one seed/method.

`--methods sage_e sage_l sgd` runs each method for each listed seed, in seed-major
order. The method labels select sage/error, sage/loss, or unmodified SGD. Select
`--dataset imagenet100|imagenet1k`, `--model resnet50|resnet101`, and `--data-mode clean`
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
./run_sage_suite.sh --methods sage_e sage_l sgd --seeds 10 15 --dry-run
```

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
| `--lr-schedule constant\|cosine` | `lr_schedule` | constant |
| `--min-lr` | `min_learning_rate` | 0 |
| `--l2-coefficient` | `l2_coefficient` | 0 (off) |

Momentum uses `v_next = momentum * v + g`, without EMA normalization/dampening.
Here `g` is the method's completed direction plus optional coupled L2. Plain
momentum updates with `v_next`; Nesterov updates with `g + momentum * v_next`.
Only a nonzero momentum allocates its velocity tree. Checkpoints preserve that
tree from the initial snapshot onward; warmup does not consume momentum history.
Nesterov requires nonzero momentum and is never implicitly enabled by momentum.

`--lr` is the actual constant rate or cosine peak, with no automatic batch-size
scaling. Cosine has no implicit warmup. For T>=2 updates indexed 0..T-1, it uses
`min_lr + (peak_lr-min_lr)*(1+cos(pi*step/(T-1)))/2`. Thus the first update uses the
peak and the last uses the floor, including zero. A one-update run uses the peak;
queries after the budget return the floor. A zero-rate final update leaves finite
parameters unchanged but still processes the batch and commits model/optimizer
state. At zero-rate probe nodes, T2 division-by-rate fields are blank and strict
gain eligibility is false; zero is never used as a denominator. A nonzero minimum
in constant mode is rejected rather than silently ignored.

Preview a configured suite without launching or downloading:

```bash
./run_sage_suite.sh --methods sage_e sage_l sgd --seeds 2 5 7 \
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
Only selected original image bytes are cached in outer Project_SAGE/data/ using SQLite.
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

## SAGE-E and SAGE-L

Both variants use ordinary per-image cross entropy and its current-parameter gradient.
Prediction error selects a reference; raw gradient alignment decides correction.

| Experiment label | Method | Reference | Definition |
| --- | --- | --- | --- |
| SAGE-E / sage_e | sage | error | Mean gradient of currently misclassified valid batch examples |
| SAGE-L / sage_l | sage | loss | Mean gradient of all valid batch examples |

For error, freeze the current selection E = {i: argmax(logits_i) != label_i} when
differentiating its surrogate mean loss. Rebuild it each batch. No historical error
buffer or derivative of a 0/1 objective is used. Loss means the entire current batch,
not a high-loss subset or exact population gradient.

Score every valid contribution with q_i = r^T g_i. SAGE uses weight -1 for a negative
score and +1 otherwise, including zero. Return sum(w_i*g_i)/B using the original valid
B; preserve gradient magnitude, do not cosine/unit-normalize, and apply LR once.
An erroneous example is not automatically flipped; a correct example may be flipped.
Empty error sets fall back to SGD; all-error batches make E/L references identical.
The error-reference denominator is the error count, distinct from the update mean B.

The TPU JVP backend obtains the same full-parameter scores without retaining a [B,P]
Jacobian. An audit Jacobian remains for small checks and sparse probes. Training BN
uses the same realized logical batch in all derivative passes, with state committed
once. These per-example loss derivatives can include BN cross-example coupling.

SGD skips reference construction. There is no sgd_e method. sage_e/sage_l are
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
| sage | +1 | -1 |
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
- validate: full selected data, configured T1-3 probes on a SAGE trajectory.
- train: full selected data and configured budget; probes normally disabled.

Example controlled diagnostics after smoke succeeds:

    python scripts/launch_tpu.py --mode validate --steps 20 --config configs/tpu/imagenet100_resnet50_sage_e.toml

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
    python main.py --check-config --config configs/tpu/imagenet100_resnet50_sage_e.toml

Tests include independent analytic directions, JVP/Jacobian parity, BN state handling,
real 50/101 parameter shapes, two-CPU global batch/BN parity, deterministic data
recovery, recorder/checkpoint retention and fail-closed launch controls. Offline
fixtures do not establish successful HF image access or TPU hardware execution.

See THIRD_PARTY_NOTICES.md and licenses/ for model source revision/license and the
class-manifest/data provenance. Local research notes/protocols remain outside this
inner code repository. No data, tokens, model weights or training results belong in Git.
