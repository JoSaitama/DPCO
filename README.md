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

Flax ResNet integration, ImageNet readers, memory-efficient scoring, distributed TPU
training, and theorem instrumentation are not implemented yet. The repository cannot
launch end-to-end ImageNet training and contains no training results.

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
- `probes/`: reserve an extension point for instrumentation. Observation fields,
  frequencies, and storage formats must be agreed before adding recording code.

SGD and `random_flip` skip reference construction and alignment scoring.
The candidate `reference.name="error"` uses surrogate-loss gradients from misclassified
examples; it does not differentiate the discrete 0/1 error signal. Its availability does
not imply that every detail of SAGE-E has been finalized or experimentally validated.
The default `reference.name="loss"` uses the current valid batch mean gradient.

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
The current fallback leaves that example's contribution uncorrected. This is an
implementation convention, not a new convergence guarantee; the paper must specify its
empty-reference policy explicitly. LOO does not automatically inherit the common-reference
Delta theorem. The DTV implementation here is a filtering direction under the configured
reference, not a claim to reproduce every training detail of the original method.

## Repository layout

```text
SAGE/
  main.py                Explicit --check-config entry point; not a training launcher
  configs/               ImageNet-100/1K and ResNet-50/101 presets
  src/
    config.py
    cli.py
    data/                Common ClassificationBatch; ImageNet reader not integrated
    models/              Extension point for Flax model adapters
    tasks/               base.py, classification.py
    signals/             prediction_error.py, gradient_alignment.py
    references/          batch_mean.py, error_subset.py
    methods/             sgd.py, sage.py, dtv.py, random_flip.py, base.py
    training/            trainer.py, updates.py, randomness.py
    probes/              Instrumentation pending an agreed observation specification
  tests/                 Configuration, direction, reference, task, and trainer checks
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
python main.py --check-config --config configs/imagenet100_resnet50.toml --method sage_loo --reference loss
python main.py --check-config --config configs/imagenet100_resnet50.toml --method random_flip --flip-probability 0.25
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

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

Each logical training batch must contain at least one valid example. No unapproved
observation fields are collected, and no CSV/JSONL recording code has been added.

## Repository language

Use English for source code, identifiers, comments, docstrings, configuration descriptions,
repository documentation, and development notes. Discussions may be conducted in Chinese.
Local research records and collaboration protocols remain outside this code repository.
