# CIFAR-100N with DPCO and Adam

This adapter uses the authors' official human annotations with the existing
ImageNet-style separation: preparation, local data reader, pure classification
batch/task, model adapter, contribution method, Adam updater, experiment runner,
CSVs and complete checkpoints. The existing ImageNet/iNaturalist presets and
updates remain unchanged.

## Model selection

Pass `--model NAME` to `scripts/run_suite.py` (or the existing shell entry):

| Name | Architecture |
| --- | --- |
| `cifar_resnet18`, `cifar_resnet34` | Existing native32 3×3 stride1 stem, BasicBlocks; default remains34 |
| `cifar_resnet50` | Native32 3×3 stride1 stem, reuses the existing ResNet50 Bottleneck implementation |
| `resnet50`, `resnet101` | Directly reuse existing ImageNet-style 7×7 stride2/maxpool models, with a 100-class head |

Model selection follows the explicit name rather than silently dispatching all
CIFAR datasets to one backbone. `resnet50` and `cifar_resnet50` intentionally name
different stems. Both accept native32 CIFAR inputs; preprocessing stays unchanged.
The native50 blocks retain this project's existing bottleneck stride placement
and zero-initialized last BN scale; this is not a claim of bitwise reproduction of
the authors' PyTorch model. Changing model starts a new experiment; same-model
Adam/Filter/Flip comparisons are needed. The current34 beta sweep needs no changes.
For MNIST data/model/noise controls, see [MNIST](mnist.md).

## Official label semantics

Sources:

- [ICLR 2022 authors' release](https://github.com/UCSC-REAL/cifar-10-100n)
- [Official TFDS interface](https://www.tensorflow.org/datasets/catalog/cifar100_n)
- [Original CIFAR images](https://www.cs.toronto.edu/~kriz/cifar.html)

The pinned annotation revision is
`49df7d8a69e355470c77c1c2f2424916325a394b`. The original Python archive has the
publisher's MD5; both annotation/mapping arrays have pinned SHA-256 values in
`src/data/cifar100n_prepare.py`.

The authors provide two equivalent annotation interfaces with different ordering:

| Interface | Original label | Human label | Image order |
|---|---|---|---|
| PyTorch `.pt` | `clean_label` | `noisy_label` | CIFAR Python |
| Official ordered `.npy` / TFDS | `clean_label` / `label` | `noise_label` | TFDS |

This implementation downloads the official ordered NumPy file and its official
`image_order_c100.npy`, then converts to the CIFAR **Python** image order.
Mapping entry `order[i]` gives the Python index for TFDS index `i`.
Before training, every converted original label must equal the original CIFAR
image label. A mismatch, missing member, bad checksum or invalid permutation fails.
No Torch or TensorFlow dependency is required. This is an adapter for the official
release, not a claim to reproduce its complete PyTorch benchmark training recipe.

## Noise switches and percentage

| Option | Meaning |
|---|---|
| `--noise-type noisy100 --is-human` | Train on released human labels; the default |
| `--noise-type clean100` | Train on original labels as a control |
| `--noise-type noisy100 --no-is-human` | Generate class-dependent synthetic labels using the empirical original-to-human transition matrix |
| `--noise-seed 0` | Synthetic-label RNG only; authors' default is zero |
| `--noise-percent 40.2` | Assert the **actual loaded** mismatch percentage; does not create noise |

`--expected-noise-percent` is an alias for `--noise-percent`. Units are percentage
points from 0 to 100, not a fraction from 0 to 1.

The human version has **20,100 mismatches out of 50,000 (40.2%)** against the
original labels. Its percentage cannot be freely adjusted without changing the
benchmark. `--noise-percent 20` therefore fails rather than generating a new 20%
human dataset. The legacy `noise_rate` argument in the official CIFAR100 Python
constructor does not control its human-label loading or empirical-matrix sampling;
we do not expose it as a fake noise-strength control.

The synthetic control estimates
`T[a,b] = count(original=a, human=b) / count(original=a)` on the complete original
training set, then uses the authors' `RandomState(seed).multinomial` procedure in
canonical Python order. Its realized mismatch percentage is measured after
sampling; it is not forced to equal 40.2%. The default seed is independent of the
training seed. Human mode rejects a nonzero synthetic noise seed.

Changing label mode or the synthetic-label settings through the CLI clears the
preset's human-percentage assertion, unless a percentage is explicitly supplied;
`clean100` defaults to an assertion of zero. All actual counts, percentages and
label hashes are retained in metadata. The official-style `actual_noise_rate`
is a fraction in [0,1]; `actual_noise_percent` is 100 times that fraction, and
`noise_or_not` describes actual training-label mismatches. Smoke limits only the visible sample view;
the percentage assertion still checks the complete original training labels.

## Preparation on the server

From the repository directory:

```bash
cd ~/Project_DPCO/DPCO
source ../.venv/bin/activate
python -m pip install -e '.[train,data]'

# Inspect pinned URLs/checksums without downloading or creating a cache.
python scripts/prepare_cifar100n.py --stage plan

# Download original images plus official labels/mapping; verify and prepare.
python scripts/prepare_cifar100n.py --stage download

# Recheck immutable prepared files without networking.
python scripts/prepare_cifar100n.py --stage verify
```

Default destination is outside the code repository:

```text
Project_DPCO/data/cifar100n/
  raw/
    cifar-100-python.tar.gz
    CIFAR-100_human_ordered.npy
    image_order_c100.npy
  prepared/
    train_images.npy
    test_images.npy
    labels.npz
    manifest.json
    manifest.sha256
    complete.json
```

The image archive is approximately 161 MB compressed; preparation also stores
about 184 MB of decoded uint8 pixels. Allow additional space for partial downloads
and metadata. Downloads use TLS verification and publish only complete files with
the expected checksum. On failure, `.partial` files remain for diagnosis; a retry
currently restarts that file. Bad existing sources/prepared caches fail explicitly
instead of silently replacing research data. Preparation takes a directory lock,
reads only named archive members, and does not extract archive paths to disk.

Use `--output-dir /absolute/path` for another preparation directory. Training's
matching override is `--data-dir /absolute/path`. Normally training only reads
prepared data; the opt-in `--download` permits automatic preparation in the child.
Preparation imports NumPy but never JAX, Torch, TensorFlow or a TPU runtime.

## Shared training entry

New preset: `configs/tpu/cifar100n/dpco_adam.toml`.
It selects native 32x32 `cifar_resnet34`, random initialization, FP32, B=128,
Adam(.9,.999,1e-8), LR=.001, reference beta=.9 and a **100-update pilot**. These are
untuned settings and the pilot is too short to establish convergence or robustness.
`cifar_resnet18` is also available. The small-image models use a 3x3 stride-one
stem with no initial max-pool. They are separate from ImageNet's ResNet50/101.

Input preprocessing follows the official CIFAR-100 loader: zero-pad by four,
random 32x32 crop and horizontal flip for training, native unaugmented evaluation,
mean `(0.5071,0.4867,0.4408)`, std `(0.2675,0.2565,0.2761)`. The existing reproducible
seed/epoch/sample RNG contract is retained; this does not promise the same random
augmentation realization as PyTorch.

Inspect the three groups:

```bash
./run_cifar100n_suite.sh --dry-run
```

An actual short TPU server check, to run explicitly when ready:

```bash
./run_cifar100n_suite.sh --mode smoke --steps 3 --batch-size 8 --eval-every-steps 1
```

The suite uses the common sequential worker0/nohup launcher. It runs
`adam`, `dpco_filter`, `dpco_flip`; the last two both save method name `dpco` and
use negative weights 0 and -1. For a custom weight:

```bash
python scripts/launch_tpu.py \
  --config configs/tpu/cifar100n/dpco_adam.toml --mode train \
  --method dpco --negative-weight -0.5 --dry-run
```

Inspect original-label controls or the official-style synthetic control:

```bash
./run_cifar100n_suite.sh --noise-type clean100 --dry-run
./run_cifar100n_suite.sh --no-is-human --noise-seed 0 --dry-run
```

Remove `--dry-run` only to execute jobs on a configured TPU server. Select a
meaningful update budget separately: with B=128 and drop-last, a full training
loader pass contains 390 updates. The default pilot evaluates at start/end;
`--eval-every-steps` / `--eval-every-epochs` controls curve sampling. All three
methods should share data/label realization, seed, augmentation, model, Adam
settings and budget. Reference history always consumes raw observed-label batch
gradients; Adam moments consume the edited direction. No group labels/masks enter
`ClassificationBatch` or the weighting/reference code.

## Fixed clean/corrupted evaluation

The reference-noise labels define a fixed partition against the original labels:

- Human mode: use the released human annotations.
- Synthetic mode: use the one seeded empirical-matrix realization.
- A `clean100` training control keeps the same reference partition and audit label
  targets, even though its actual training mismatch rate is zero.

This makes clean-label and noisy-label training comparable on the same images.
The word clean is relative to the benchmark's original labels, not a guarantee
that all images have an unambiguous semantic ground truth.

Every evaluation writes `group_evaluation.csv`:

| Group | Fixed members | CE target |
|---|---|---|
| `clean` | Reference label equals original | Original label |
| `corrupted_noisy` | Reference label differs | Reference noisy label |
| `corrupted_original` | Same corrupted members | Original label |

Metrics are unweighted ordinary CE/top1/top5 in evaluation mode with running BN,
no training augmentation, correct valid-count weighting and no model/reference/
optimizer state commit. Empty groups record count zero and blank metrics, not NaN.
The three group CE curves are in addition to the usual train/evaluation CSVs.
Group metrics are evaluated over the whole configured view, not dynamically
selected by current loss or alignment. They do not by themselves report per-group
training flip rates. Raising noisy-target CE alone is not evidence of improvement.

CIFAR supplies a 10,000-image **test** set and no separate official validation set.
The common `evaluation.csv`/summary retains its legacy `val_*` column names for
compatibility; metadata explicitly identifies them as official-test monitoring.
These runs disable automatic `best` checkpoint selection based on that set.
Use fixed-budget/final results, and arrange a separate validation protocol before
hyperparameter selection; this implementation does not certify a blind test result
if users tune settings from the monitored test curve.

Complete checkpoints retain method/Adam state and data/augmentation cursor.
Fingerprints cover prepared data, observed/reference labels, noise settings and
smoke view; wrong data or label modes cannot silently resume an old trajectory.
Resuming an older checkpoint rewinds the group CSV as well. Run labels distinguish
human/synthetic, label mode and method weights. Legacy ImageNet metadata missing
new optional data fields is compared with the original defaults.

## Offline three-run plots

Use [the DPCO plotting entry point](../plots/README.md) to explicitly select the
Adam, EMA Flip and EMA Filter run directories. It reads `train_steps.csv` and
`timing.csv` without a training runtime, exports CE/accuracy against step and time,
and adds a window-weighted ms/step figure. `--smooth-window` controls the shared
trailing smoothing span; `--time-axis wall` includes training-loop overhead and
`--time-axis active` accumulates training windows only. Figures and source hashes
are saved under `results/figures/`, alongside `files/`, `runs/` and `logs/`.
Adam is blue, Flip red, and Filter orange (dashed). These train-label plots do not replace the
fixed-group evaluation or official-test curves.
