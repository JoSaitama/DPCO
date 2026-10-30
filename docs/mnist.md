# MNIST controlled verification

This is a new mechanism experiment, not a claim to replicate the professor's
unrecorded architecture, corruption process or hyperparameters. The user selected
the default **small CNN without BatchNorm and 20% synthetic wrong training labels**.
The task remains ordinary per-example cross entropy, with Adam, EMA reference and
optional filter/flip using the same shared implementation as CIFAR-100N.

## Data and noise semantics

- Original MNIST IDX order, 60,000 train and 10,000 test images, native 28×28×1.
- Pixels are float32 `uint8 / 255`; no random crop, horizontal flip or resizing.
- Four compressed IDX sources use the HTTPS mirror and MD5s listed in
  [torchvision's MNIST loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
  Parsing checks IDX magic, counts, dimensions, payload size and label range.
  Prepared NumPy files and manifests are additionally checked with SHA-256.
- The immutable prepared cache contains **clean source data only**. Fixed
  synthetic corruption is generated at loading and recorded in each run identity.
- `--label-noise-percent P` is the **MNIST generator**, in percentage points 0–100.
  A fixed `noise_seed` permutes all train indices, selects
  `floor(N*P/100 + 0.5)` samples without replacement, and replaces each selected
  label uniformly among the other nine classes. 20% means exactly 12,000 wrong
  labels on the full training split. Labels are not regenerated per batch/epoch.
- `--noise-seed` controls this label map independently of training `--seeds`.
  All three methods use the same map. Changing `--label-noise-percent` without
  `--data-mode` automatically selects clean for 0, otherwise noisy.
- `--noise-percent` / `--expected-noise-percent` remains an optional assertion
  about the **realized** mismatch percentage. It never generates labels. For
  percentages that round to a count, the realized rate may differ from the request.
  CIFAR-100N's existing official-label semantics are unchanged.
- Clean/test labels and corruption masks are excluded from training batches and
  do not enter reference construction or sample weights. Train batches contain
  only pixels, observed labels and valid masks.
- The fixed audit groups are `clean`, `corrupted_noisy` (observed labels) and
  `corrupted_original` (original labels). At 0% or 100%, empty groups record count
  zero and missing metrics. Original test labels are never corrupted; test is
  monitored, not used for best-checkpoint selection or parameter tuning.

Default cache: `Project_DPCO/data/mnist/{raw,prepared}` outside the code directory.
Downloads reuse verified completed files; an interrupted individual `.partial`
file is restarted on retry. `nohup` protects against terminal disconnection,
not machine restart. Corrupt existing source/prepared files fail validation and
are preserved for inspection.

## Choose the model

`--model` overrides `[model].name`; it never silently replaces architecture based
on the dataset. The native MNIST preset uses `mnist_cnn`:

| Name | Structure |
| --- | --- |
| `mnist_cnn` | Conv16/ReLU/pool, Conv32/ReLU/pool, Dense64/ReLU, Dense10; no BN/dropout |
| `mnist_mlp` | Flatten, Dense128/ReLU, Dense64/ReLU, Dense10; no BN/dropout |
| `resnet50`, `resnet101` | Existing ImageNet-style 7×7 stride2/maxpool stem; same residual implementation |
| `cifar_resnet18`, `cifar_resnet34` | Native 3×3 stride1 stem and BasicBlocks |
| `cifar_resnet50` | Native 3×3 stride1 stem and existing ResNet50 Bottleneck blocks |

All MNIST models receive one input channel and output ten classes. ResNets are
optional architecture studies, not the default small experiment. Existing
ImageNet/CIFAR model names and initialization remain unchanged; MNIST first-layer
shapes differ because its input is grayscale.

## Prepare on the server

The following assumes the established project/interpreter location. Use the
corresponding absolute paths on another server. There are no new runtime packages;
the existing NumPy/Pillow/JAX/Flax environment is sufficient.

```bash
cd /home/jason_chia925_gmail_com/Project_DPCO/DPCO
mkdir -p results/logs

# Short read-only plan, no download.
/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  scripts/prepare_mnist.py --stage plan

prepare_log="results/logs/mnist_prepare_$(date +%Y%m%d_%H%M%S).log"
nohup /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  -u scripts/prepare_mnist.py --stage download \
  > "$prepare_log" 2>&1 < /dev/null &
prepare_pid=$!
printf '%s\n' "$prepare_pid" > "$prepare_log.pid"
echo "PID=$prepare_pid LOG=$prepare_log"
tail -f "$prepare_log"
```

After the background preparation completes, short integrity verification:

```bash
/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  scripts/prepare_mnist.py --stage verify
```

## Preview and launch three methods

The preset uses seed42, B128, Adam(.9,.999,1e-8), lr.001/cosine to 1e-6,
reference beta .99, L2=0, 9,360 steps (20 drop-last traversals), full evaluation
every 468 steps and rolling checkpoint every 2,340 steps. These are exploratory
settings, not validated optimums or the professor's recipe. First preview or
run a short 100-step pipeline smoke before committing a full budget.

```bash
/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  scripts/run_suite.py --config configs/tpu/mnist/dpco_adam.toml \
  --methods adam dpco_filter dpco_flip \
  --model mnist_cnn --label-noise-percent 20 --noise-seed 0 --dry-run

train_log="results/logs/mnist_suite_$(date +%Y%m%d_%H%M%S).log"
nohup /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  -u scripts/run_suite.py --config configs/tpu/mnist/dpco_adam.toml \
  --methods adam dpco_filter dpco_flip --seeds 42 --wait \
  --model mnist_cnn --label-noise-percent 20 --noise-seed 0 \
  --reference-beta 0.99 \
  > "$train_log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$train_log.pid"
echo "PID=$train_pid LOG=$train_log"
tail -f "$train_log"
```

Keep the existing TPU launch architecture: one host, configured local chips, no
automatic CPU/GPU fallback. The preset assumes the same four local TPU chips as
the existing CIFAR preset; configure `[execution]` for a different machine layout.
Local unit/integration checks can explicitly use CPU, but do not establish TPU
performance or full MNIST convergence.

`dpco_filter` fixes negative weight to 0; `dpco_flip` fixes it to −1. For other
weights use `--methods dpco --negative-weight VALUE`. Reference remains the
bias-corrected EMA of raw observed-label batch gradients, current batch included;
it is independent of Adam moments. This addition does not tune/change that rule.

Outputs continue under `results/files/<run_name>`, `results/runs/<run_name>` and
`results/logs/<run_name>`. Names include dataset, model, method/reference beta,
weight, training seed, label corruption percentage and noise seed. Preserve run
directories for explicit recovery. Resume only with the original configuration,
dataset fingerprint and compatible model shapes; changing model/noise creates a
new run. Old theorem/diagnostic probes stay disabled on EMA/Adam/MNIST runs.
