# SAGE

Modular research code for comparing SGD, SAGE, DTV-style filtering, and random-flip controls.

## Current status

The implementation separates tasks, signals, references, methods, and training orchestration.
It includes a classification task, loss/error references, standard and leave-one-out (LOO)
alignment, six method configurations, and executable single-step updates.

Directions are averaged over the original number of valid batch contributions, B.
Individual gradient magnitudes are preserved; the trainer does not divide by B again.
Alignment-based methods currently materialize per-example gradients for small-batch
correctness checks. This implementation does not establish large-model efficiency.

An injected single-process experiment loop now supports structured logs, buffered CSV,
full validation, seven logical checkpoint slots, exact source-state restoration, and
optional read-only classification theorem probes. Synthetic CPU checks exercise this path.
Flax ResNet integration, ImageNet readers, memory-efficient scoring, and host-local TPU
parallel training are not implemented yet. The repository cannot launch end-to-end
ImageNet training and contains no ImageNet training results.

## Current experiment scope

The immediate target is two independent single-host classification runs:

| Host | Experiment | Method configuration | Reference configuration |
| --- | --- | --- | --- |
| worker0 | SAGE-E (`sage_e`) | `method.name="sage"` | `reference.name="error"` |
| worker1 | Standard SGD | `method.name="sgd"` | Unused by SGD |

The two hosts run separate experiments, each using the complete training dataset.
They do not split one experiment's dataset or synchronize model updates with each other.
`global_batch_size` belongs to one experiment across its participating local devices;
it is not the sum of the two independent workers' batch sizes.
TPU runtime isolation and host-local multi-device execution still require integration;
the presence of two worker VMs does not itself configure independent TPU jobs.

Start with ImageNet-100 and ResNet-50, then extend to ResNet-101 and ImageNet-1K
according to the agreed experiment sequence. Paired SAGE-E and SGD presets use the same
seed, data/model settings, batch size, and learning rate. The first SAGE-E experiment
uses a full common reference: no leave-one-out exclusion and no historical error buffer.
DTV, random flipping, LOO, and cross-host collaborative training are deferred.

## SAGE-E and SAGE-L reference definitions

Both variants use the same per-example cross-entropy loss and current-parameter gradient,
`g_i = grad(loss_i)(params)`, over the B valid examples in the current logical batch.
Reference choice and contribution weighting are separate configuration decisions:

| Experiment label | Method | Reference | Common reference gradient |
| --- | --- | --- | --- |
| SAGE-E (`sage_e`) | `sage` | `error` | Mean of gradients of currently misclassified valid examples |
| SAGE-L (`sage_l`) | `sage` | `loss` | Mean of gradients of all valid examples in the current batch |

For SAGE-E, select `E = {i: argmax(logits_i) != label_i}` from the current forward pass.
If E is nonempty, `r_E = sum(g_i for i in E) / len(E)`. Freeze this discrete selection
while differentiating; do not differentiate a 0/1 classification-error objective.
Rebuild E for each batch at the current parameters. No past examples or gradients are cached.

For SAGE-L, `r_L = sum(g_i) / B`. This is the gradient of the current batch's mean loss,
not an exact full-dataset gradient and not a high-loss subset or loss-weighted mean.
Use the name SAGE-L (`sage_l`), not SAGE-Loss.

In both cases, score **every valid example**, including correctly classified examples,
using `q_i = inner_product(r, g_i)`. Set `w_i = -1` when `q_i < 0`, and `w_i = +1`
otherwise. Return `direction = sum(w_i * g_i) / B`, preserving raw gradient magnitudes.
Prediction error selects the reference; negative alignment selects the flipped contributions.
An error example is not automatically flipped, and a correct example may be flipped.

If E is empty, SAGE-E returns the standard SGD direction. If every valid example is
misclassified, SAGE-E and SAGE-L have identical references and directions. A zero
reference produces zero scores, so contributions remain unchanged. Padding is excluded
from prediction errors, reference counts, scoring, and the update denominator.

The code uses a single `sage` implementation; `sage_e` and `sage_l` are experiment
labels, not additional method names. Standard SGD neither constructs nor uses a reference,
so there is no `sgd_e` variant in this protocol. These reference definitions specify the
algorithm; they do not establish full-objective convergence or accuracy improvements.

## Component responsibilities

The trainer orchestrates an update without branching on method names:

```text
trainer.step
    -> method.compute_direction(task, params, batch, states, key)
         -> task losses / feedback
         -> signal + reference (only when needed)
         -> correction weights
         -> direction = sum_i(w_i * g_i) / valid_B
    -> updater: params - learning_rate * direction
```

- `tasks/`: define per-contribution losses, validity masks, task feedback, and explicit
  model state through a common interface across datasets.
- `signals/`: compute prediction errors or raw gradient inner products; they do not
  choose filtering or flipping rules.
- `references/`: construct common or per-example LOO reference gradients from the
  full batch or an error subset; they do not choose correction rules.
- `methods/`: use these components to choose weights and return a complete direction;
  they do not modify parameters or apply the learning rate.
- `training/trainer.py`: invoke the selected method and parameter updater, then advance
  training state without method-specific branches.
- `probes/`: read-only common-reference classification diagnostics. Optional probes
  return scalar records to the shared recorder, without updating training state or RNG.

SGD and `random_flip` skip reference construction and alignment scoring. The `error`
and `loss` implementations follow the definitions above; the selected source is passed
to the method during composition, without adding reference-specific trainer branches.

## Methods and references

Let `q_i` denote the raw gradient inner-product score for an example.

| `method.name` | Nonnegative score | Negative score |
| --- | --- | --- |
| `sgd` | +1 | +1 |
| `sage` | +1 | -1 |
| `dtv` | +1 | 0 |
| `random_flip` | -1 with probability p; otherwise +1 | -1 with probability p; otherwise +1 |

SAGE and DTV retain weight +1 at zero score. `random_flip` ignores alignment and
independently flips each valid example with probability p. All methods divide by the
original valid count B, not the weight sum or the number of retained examples.
At p=0, random flipping reduces to SGD; at p=1, it negates all contributions, not SAGE.

`sage_loo` and `dtv_loo` exclude the scored example from its reference. Reference source
remains independently configurable as `loss` or `error`. For an error reference, an
example is subtracted only if it belongs to the selected error subset; examples outside
that subset continue to use the complete error-subset reference.

An empty reference or singleton LOO reference is flagged through `has_reference`.
The fallback leaves that example's contribution uncorrected. Empty-error fallback is
part of the current SAGE-E protocol, not an additional convergence guarantee.
LOO remains outside the initial experiment and does not automatically inherit the common-reference
Delta theorem. The DTV implementation here is a filtering direction under the configured
reference, not a claim to reproduce every training detail of the original method.

## Experiment configurations

Each dataset/model pair has an SGD baseline and a matching full-reference SAGE-E preset:

| Dataset/model | SGD configuration | SAGE-E configuration |
| --- | --- | --- |
| ImageNet-100 / ResNet-50 | `configs/imagenet100_resnet50.toml` | `configs/imagenet100_resnet50_sage_e.toml` |
| ImageNet-100 / ResNet-101 | `configs/imagenet100_resnet101.toml` | `configs/imagenet100_resnet101_sage_e.toml` |
| ImageNet-1K / ResNet-50 | `configs/imagenet1k_resnet50.toml` | `configs/imagenet1k_resnet50_sage_e.toml` |
| ImageNet-1K / ResNet-101 | `configs/imagenet1k_resnet101.toml` | `configs/imagenet1k_resnet101_sage_e.toml` |

SAGE-E presets select `method.name="sage"` and `reference.name="error"`. SAGE-L can
use the same method with `reference.name="loss"`; it is not an initial scheduled run.
The learning rate 0.01 and 100-step budget are integration placeholders, not a tuned
ImageNet training recipe. Presets expose logging, evaluation, checkpoints, and probes.

## Result directory protocol

All experiment outputs belong under `results/` at the repository root by default:

```text
results/
  logs/{run_name}/       train.log and metadata.json
  runs/{run_name}/       Shared checkpoint snapshots and index.json
  files/{run_name}/      Training, validation, theorem, timing, summary CSV
```

Every preset declares `[results] root_dir = "results"`. Shared path construction lives
in `src/training/results.py`; relative roots resolve against an explicitly supplied
project directory. The smoke script uses the repository root, independent of the shell's
working directory. Configuration inspection alone creates no directories.

The shared run name is `{dataset}_{data_mode}_{model}_{method_label}_{seed}_{lr}_{YYYYMMDD_HHMMSS}`.
For example, `imagenet100_clean_resnet50_sage_e_0_0.01_20261017_143025`.
The LR component is the initial configured rate; JSON stores the full configuration,
timezone-aware timestamp, code/dependency/device information, and adapter metadata.
New runs reject existing names. Resuming requires an explicit existing name and snapshot.
`data.mode` is an identity label; this change does not implement data corruption.

`results/` and all its contents are ignored by Git. Every step retains raw pre-update
CE/top-1, valid count, LR, sample/epoch progress, and wall-time position. CSV writes are
buffered (50 steps by default); logs have an independent step/percentage schedule.
Validation is full-set and sample-weighted, initially, periodically, and finally.
Best uses full validation top-1, with the earlier step winning ties.
Checkpoints retain initial/25%/50%/75%/final/best/latest slots, sharing identical steps.
Theorem probes default off. See [recording.md](docs/recording.md) for fields, costs,
event semantics, recovery, and the limits of sparse theorem measurements.

## Repository layout

```text
SAGE/
  main.py                Explicit --check-config entry point; not a training launcher
  configs/               Paired SGD/SAGE-E ImageNet-100/1K and ResNet-50/101 presets
  scripts/smoke_test.py   CPU synthetic classification and trainer smoke checks
  scripts/smoke_recording.py  Synthetic end-to-end observation/checkpoint checks
  src/
    config.py
    cli.py
    data/                Common ClassificationBatch; ImageNet reader not integrated
    models/              Extension point for Flax model adapters
    tasks/               base.py, classification.py
    signals/             prediction_error.py, gradient_alignment.py
    references/          batch_mean.py, error_subset.py
    methods/             sgd.py, sage.py, dtv.py, random_flip.py, base.py
    training/            Trainer, experiment loop, schedules, metrics, recording,
                         evaluation, checkpoints, randomness, and result paths
    probes/              Read-only common-reference classification probes
  tests/                 Configuration, direction, reference, task, and trainer checks
  results/               Generated logs/, runs/, files/; entirely ignored by Git
```

## Local checks

Configuration inspection requires Python 3.11+. Numerical tests require CPU JAX,
available through the `test` extra. The core was checked with JAX 0.11.2 on a temporary
CPU environment; this does not certify the full TPU/Flax stack.

Training dependencies are not yet locked. A validated environment and the upstream model
revision must be pinned during integration. Configuration inspection does not check that
data files exist or validate the current device. The ImageNet-100 class list is not yet
selected; store it under version-controlled `configs/manifests/` once selected, while
keeping image data out of Git.

```bash
python main.py --check-config --config configs/imagenet100_resnet50.toml
python main.py --check-config --config configs/imagenet100_resnet50_sage_e.toml
python -m pip install -e '.[test]'
python scripts/smoke_test.py --config configs/imagenet100_resnet50_sage_e.toml
python scripts/smoke_test.py --config configs/imagenet100_resnet50.toml
python scripts/smoke_recording.py --probes
python -m unittest discover -s tests -v
```

The smoke script explicitly uses CPU, including on a TPU VM, and runs three compiled
updates by default (`--steps` changes this count). It exercises a small linear classifier
with the configured class count, five valid examples and one padding position. It does
not instantiate the configured ResNet, use the configured training batch size/image size,
read ImageNet, require a class manifest file, or initialize the TPU slice.

Checks compare SAGE-E/SAGE-L directions and compiled trainer updates against analytic
linear-softmax cross-entropy gradients. They cover full common references, reference
selection, padding, the original valid-B denominator, empty/all-error cases, and zero
scores. They do not certify TPU execution, training throughput, convergence, or theorem
advantages. A failed check exits with a nonzero status. No training artifacts are saved.

The separate `smoke_recording.py` executes four synthetic updates through the actual
experiment/recording path, with optional `--probes`. By default its files live in a
temporary directory and are removed after checking. To inspect retained artifacts:

```bash
python scripts/smoke_recording.py --probes --output-root results/smoke_checks
```

Its configured dataset/model names describe the preset being checked; metadata explicitly
identifies the actual synthetic data and linear-softmax model. These are infrastructure
checks, never ImageNet accuracy, ResNet throughput, or theorem-convergence experiments.

For future training integration on a configured TPU environment:

```bash
python -m pip install -e '.[train,tpu]'
```

This installation example has not been executed or validated on the project's TPU.
Initial experiments compare SGD and SAGE; providing DTV and random-flip implementations
does not mean that all ablations have been scheduled or run.

## Multi-device and state conventions

The global logical batch defines reference construction and contribution aggregation;
do not silently substitute device-local references. Reduce valid contribution sums and
counts globally before division rather than averaging unequal local means. Error-subset
references must likewise use the global error count.

Callers explicitly construct random keys from the seed, step, and device/example
convention; do not reuse an identical random key across devices. The current methods
assume one complete logical batch on a single device and do not implement distributed
reductions for production training.

`ClassificationTask.model_fn` is a pure function returning logits and candidate model
state; the trainer commits that state once. Training-mode BatchNorm will require explicit
handling of cross-example coupling. The linear-model tests do not resolve that integration.

Each logical training batch must contain at least one valid example. The experiment
loop requires a host-data source with exact `snapshot`/`restore`, a separate full-set
evaluation adapter, and explicit fixed probe/audit inputs when probes are enabled.
Training scalars are reused from the existing forward pass. Recording logic lives in
the host recorder, not in references, signals, or methods.

## Repository language

Use English for source code, identifiers, comments, docstrings, configuration descriptions,
repository documentation, and development notes. Discussions may be conducted in Chinese.
Local research records and collaboration protocols remain outside this code repository.
