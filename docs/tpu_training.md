# Worker0 TPU execution protocol

## Scope

The initial pipeline uses original ImageNet-100 images and a randomly initialized
ResNet-50. After the first-stage theory validation, use ImageNet-1K and ResNet-101 as
a later check. No pretrained weights are required: model architecture is supplied in
the repository. This is an optimizer comparison from scratch, not fine-tuning.

CPU sanity is allowed to check interfaces and file recording. Every real smoke,
theory-validation or training job must be started by `scripts/launch_tpu.py`, which
uses nohup and a TPU-only child. The child rejects foreground launch and CPU fallback.
The earlier CPU scripts were renamed to sanity_core.py and sanity_recording.py to
prevent accidentally presenting CPU sanity checks as TPU smoke tests.

Only node-ziao1 worker0 is in scope. Worker1 stays idle and is not contacted, started,
or used as a peer. One process uses the visible local devices through a data-parallel
GSPMD mesh. Model parameters/BN buffers are replicated and inputs are sharded along
the logical batch dimension. References, valid/error counts and update means are
global to that host's logical batch, rather than independently computed per chip.

## Install in a separate environment

Use Python 3.11 or later; 3.12+ is recommended for recent JAX/Flax packages. Do not
modify another project's environment. On worker0, from the cloned code repository:

```bash
python3 --version
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,tpu,test]'
python -m pip check
```

The packages were tested locally with CPU JAX 0.11.2 and Flax 0.12.10, not on a TPU.
The resolver can choose older compatible versions on older supported Python versions.
Record actual package/libtpu versions on the server; installation success is not a TPU
hardware test. TensorFlow/PyTorch are not required by this input/model path.

Hugging Face authentication is separate from GitHub SSH authentication. Existing
HF_TOKEN or the account's saved HF login is reused automatically; no token belongs in
TOML, logs, launch JSON, commands, or Git. If needed, run interactively:

```bash
hf auth whoami
hf auth login
```

The official dataset is gated: an authenticated account may still need to accept
https://huggingface.co/datasets/ILSVRC/imagenet-1k in its browser. Code cannot grant
that approval or reuse an unrelated account's dataset access. A completed local data
cache is reusable without a new network login.

## CPU sanity, then real TPU smoke

Foreground CPU sanity, without real data downloads:

```bash
python scripts/sanity_check.py --config configs/tpu/imagenet100_resnet50_sgd.toml
python scripts/sanity_check.py --config configs/tpu/imagenet100_resnet50_sage_e.toml
```

These synthetic checks are not ResNet/ImageNet/TPU results. Next launch SGD:

```bash
python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sgd.toml
```

The launcher prints a PID and the exact `results/logs/{run_name}/stdout.log` path.
Monitor that path with `tail -f`. Wait for the completed summary and successful updates
before launching SAGE-E on the same host:

```bash
python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sage_e.toml
```

For a single direct Python launch, no outer nohup is needed: it executes `nohup python -u ...`, with
stdin detached, a new session, and persistent per-run stdout/stderr. A host lock rejects
overlapping jobs rather than killing an existing job. PID creation does not prove
successful startup. Check for `tpu_ready`, real model initialization, finite training
rows and a `completed` event. Failure/traceback appears in the same stdout.log, even if
HF access or backend initialization fails before the main recorder is constructed.

The shared `results/logs/run_status.tsv` records only start and end per attempt,
with running/completed/failed status and no step/error columns. Each resume has a new
attempt ID. Normal bootstrap failures are covered even before metadata.json exists.
Step-bearing JSON logs put step/event first; train records then put train_ce/lr.
See recording.md for exact columns, schema checks and failure limitations.

For all new training suites, use the root shell entry under an outer nohup:

```bash
nohup ./run_sage_suite.sh \
  --mode smoke --dataset imagenet100 --model resnet50 \
  --methods sage_e sage_l sgd --seeds 10 15 22 --data-mode clean \
  >/dev/null 2>&1 < /dev/null &
```

This example creates nine sequential three-update smoke runs. Each successful run adds
two status rows. The wrapper defaults to train mode; pass smoke explicitly for tests.
It shares the same isolated TPU child entry, waits for process exit/TPU lock release,
checks terminal status and stops on the first failure. The waiting parent finalizes
an abrupt child failure even if the child could not write its own end row. Host loss
can still prevent terminal recording. No worker1 job is started.

Seeds default to 42 in shipped configs and may be overridden with --seed/--seeds.
Methods are sage_e/sage_l/sgd; dataset/model overrides update both the parent and child
config, including the classifier class count. Only clean data is currently supported;
mismatch modes fail before launch. Use --dry-run to inspect the full grid, and set the
formal training budget explicitly rather than treating the 100-step pilot as final.

Inspect a command without starting devices, creating run folders or downloading data:

```bash
python scripts/launch_tpu.py --mode smoke --config configs/tpu/imagenet100_resnet50_sgd.toml --dry-run
```

## Data preparation and interpretation

The default source is `ILSVRC/imagenet-1k` at commit
`49e2ee26f3810fb5a7536bbf732a7b07389a47b5`. The versioned manifest records the CMC
100 synsets, original 1K indices and their fixed 0..99 output mapping. This avoids
silently mixing a 160-pixel ImageNet-100 mirror with original-resolution data.

- Smoke streams until it obtains max(32, 2*batch_size) selected training images and
  the same number of validation images. These limited subsets are not necessarily
  balanced across all classes and are not a complete ImageNet-100 benchmark.
- Train/validate prepare the full selected split: 126,689 training and 5,000 validation
  images for this fixed 100-class subset. Full ImageNet-1K uses 1,281,167/50,000.
- Only selected original encoded image bytes are saved into a local SQLite database.
  No full 1K archive or duplicate decoded/Arrow dataset is deliberately materialized.
  However, filtering may still read a substantial part of the ~154 GB train/validation
  source over the network. Parquet label filters can reduce transfers when row groups
  permit it; do not assume that network bytes equal the selected cache size.
- Cache identity includes source revision, mapping hash and limits. Smoke and full
  caches are different. By default they live in outer `Project_SAGE/data/`, not Git.
  Override `data.prepared_dir` to use a larger disk. The preparer checks estimated
  remaining space plus a 3 GB reserve before writing; full 1K requires much more than
  the previously reported 43 GB free disk. Interrupted preparation can resume from
  committed rows, though replaying a source prefix can incur additional network reads.
- Training uses a deterministic epoch permutation and stateless per-example crop/flip
  randomness. It drops incomplete tails so training BatchNorm never includes padding.
  Full evaluation uses deterministic resize/center crop, pads its final batch, and
  weights metrics by valid counts. All source snapshots include epoch, cursor, seed
  and data identity; training recovery does not replay an HF iterator.

There is no synthetic fallback for failed HF access, no remote Python dataset script
execution, and no download of pretrained model weights. The prepared data and model
initialization are reused consistently between SGD and SAGE experiments.

## TPU topology and resource boundaries

The starting profile declares four visible chips and `2,2,1` chip bounds, matching the
previously discussed v5p host layout. These are configurable in `[execution]`, not
assumed for every future TPU type. It overrides process bounds/addresses to one local
process, never calls jax.distributed.initialize, and checks that all discovered devices
belong to process zero and that global/local counts match. The startup timeout aborts
a stalled runtime instead of waiting indefinitely for another worker. Environment
settings are a requested topology, not a substitute for verifying real libtpu behavior.

Batch size must be divisible by the number of local JAX TPU devices; device count is
not always chip count. Train/validate default to global B=256 (64 per local device
on the four-device host); smoke defaults to B=8. --batch-size overrides either.
Model precision is explicitly
float32 in the pilot; bfloat16 can be configured later and must be reported separately.
No claim of measured TPU throughput, peak memory or worker isolation is made until
the server run confirms those properties. The code never falls back to GPU/CPU.

The main elapsed_wall_s clock starts at the experiment loop, after data preparation
and random parameter initialization. This avoids comparing a first SGD download
against a warm-cache SAGE run as though the difference were optimizer speed. Bootstrap
stdout/launch metadata precede that clock; do not label it as full provisioning time.

The JVP backend computes exact full-parameter common-reference alignment through
VJP/JVP, then freezes weights and averages the weighted gradient over original B.
It does not normalize sample gradients or replace the reference by device-local means.
Compared with the audit Jacobian, it avoids retaining a [B, parameter_count] training
matrix. This is a memory representation change, not a different SAGE update rule.
Current JVP mode supports full references; LOO remains in the separate audit backend.

## Validation and training modes

For short T1-3 diagnostics after smoke succeeds:

```bash
python scripts/launch_tpu.py --mode validate --steps 20 --config configs/tpu/imagenet100_resnet50_sage_e.toml
```

This prepares the complete selected dataset, enables the configured theorem probes,
and runs real TPU updates. The original probe implementation can materialize gradients
at sparse audit nodes and is deliberately expensive. Its batches, candidate BN state
and hypothetical updates never replace the real training state. T2 uses the same
realized training-batch objective and frozen error mask; the T3 fixed audit uses initial
frozen BN buffers plus fixed training inputs. T3 target-transfer diagnostics are not an
automatic proof of the original exact-reference convergence theorem. ReLU/BN semantics
and all additional smoothness/target conditions remain part of interpreting a result.
The validation mode is for full SAGE; a normal SGD baseline skips extra reference work.

Full-data pilot training, normally with probes disabled:

```bash
python scripts/launch_tpu.py --mode train --config configs/tpu/imagenet100_resnet50_sgd.toml
python scripts/launch_tpu.py --mode train --config configs/tpu/imagenet100_resnet50_sage_e.toml
```

Run these sequentially. The preset 100-step budget and LR=0.01 remain untuned pilot
values, not a final accuracy recipe. Updates use SGD with optional momentum and
coupled L2, both off by default. No decoupled weight decay, clipping or norm matching
is added. Pass --momentum 0.9 to use plain momentum; --nesterov is separate and
requires a positive momentum. Its velocity is initialized once and saved/restored
with the training state. Both methods use the same configured update policy.
Pass --lr 0.1 --lr-schedule cosine --min-lr 0 to decay from peak to zero, or replace
zero with a positive floor. Constant is the default schedule; no implicit warmup
or batch-size LR scaling occurs. With T>=2, update 0 uses peak and update T-1 uses
floor. One-update runs use peak once; queries after the budget use floor.
Pass `--l2-coefficient 0.0001` through `run_sage_suite.sh` to enable it for all suite
jobs, or `--l2-coefficient 0` to force it off. Config files can use
`[training] l2_coefficient = 0.0` (put the key inside the existing training table).
Only parameter leaves with ndim > 1 receive lambda times parameter, after sample
aggregation. It never changes which gradients are flipped at the same state. Zero
uses the original compiled update operations; no regularization metrics are computed.
Current theorem probes require zero L2 and momentum; incompatible options fail
preflight. At zero LR, T2 division-by-LR fields are blank and strict gain eligibility
is false. Resume uses all original optimizer/schedule/budget/batch settings; absent
legacy optimizer fields imply off/constant defaults, without rewriting metadata.
Preserve an older run's batch explicitly if its preset has since changed to 256.
Logging/evaluation/checkpoint controls and
seven-slot retention follow recording.md. --steps and --batch-size are explicit
overrides saved in the effective run configuration. Probes are off in the supplied
smoke profiles; the configured global/per-theorem switches remain usable on SAGE.

Pass `--checkpoint-every-steps 10` through `run_sage_suite.sh` to replace the rolling
latest checkpoint every 10 completed updates. It defaults to 100 (or the custom
config's latest_every_steps) and accepts positive integers only. Fixed milestones,
best validation selection and mandatory initial/final saves keep their own rules.
New snapshots and index are published before obsolete snapshots are deleted; a
failed snapshot/index write keeps the previous indexed recovery point. This can
temporarily require space for both snapshots. Reuse the same interval on resume.

Validation cadence is separate: pass --eval-every-steps 1000 or
--eval-every-epochs 1 through the shell suite, never both. Epoch values can be
0.25/0.5/etc. and measure cumulative valid-sample progress. The default remains the
config's epochs/1. Initial and final validation remain mandatory; final/periodic
overlap runs only once. Validation always traverses the full configured split,
including only the small prepared validation subset in smoke mode. A checkpoint
save does not force validation. Keep the same evaluation settings on resume.

To resume a run, reuse its original mode/config/overrides and add:

```bash
python scripts/launch_tpu.py --mode train --config configs/tpu/imagenet100_resnet50_sage_e.toml --resume RUN_NAME --resume-slot latest
```

Existing run metadata/config must match. The new launch appends stdout and records
its PID in the same run directory; training CSV is rewound to the chosen checkpoint.

Later 1K/101 checks use `data.name="imagenet1k"`, `num_classes=1000`, no 100-class
filter, and `model.name="resnet101"`. The same architecture/data pipeline supports
that configuration, but no corresponding server experiment has been run yet.
