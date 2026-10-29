# Proportional iNaturalist 2021 subset preparation

Preparation remains standalone without JAX/TPU initialization. The independent
`run_inaturalist_suite.sh` trains from its completed output using the common
classification infrastructure. ImageNet source/cache behavior is unchanged.

## Fixed protocol

- Source: official **2021 full train**, not `train_mini`, plus official validation.
- Default selection: 140,000 train and 20,000 validation images across all 10,000
  categories. Preserve original JPEG bytes; no resizing/recompression.
- Default **selection seed 42**, independent of training seed 5 or any later seed.
- Allocate quotas per split using `floor(M * N_c / N)`, then distribute remaining
  units by descending integer remainder and ascending original category ID for ties.
  Validation has 10 originals/category and therefore selects exactly two/category.
  Train follows the original full-train proportions, not equal class quotas.
- Select lowest SHA-256 ranks within each category. The canonical key includes the
  algorithm version, selection seed, split, original category ID, image ID and path.
  Neither host paths, runtime RNG, Python hash seed, row order nor training seed
  affects membership. All categories must have at least one selected sample.
- Stable original-category ordering determines classifier labels 0..9999; original
  image/category IDs and paths remain in the manifest. Train/validation are never
  mixed or resplit. No competition test images are downloaded.

The expected selected payload is roughly 13.35 decimal GB based on official archive
averages. This is not a measured capacity guarantee. The default preparation ceiling
is 15 decimal GB, including metadata, manifests, SQLite and journals; keep 10 decimal
GB free on the filesystem. Failures preserve partial output and do not silently
change the seed, counts, proportions, resolution or subset membership. If the
ceiling is exceeded, inspect measured usage and explicitly choose a new budget or
a new selection directory/counts before retrying.

## Install and plan

Use the project's isolated Python 3.11+ environment on worker0:

```bash
cd ~/Project_DPCO/DPCO
source ../.venv/bin/activate
python -m pip install -e '.[data]'
python scripts/prepare_inaturalist.py --stage plan --selection-seed 42
```

This downloads only official annotation archives (about 221 MB train and 9.4 MB val,
as advertised by the publisher) and streams their JSON records into a temporary
index database. It does not load the full annotation JSON into RAM. Metadata is
retained to make provenance and retries explicit. The full source index's pages
remain allocated after the smaller validation index replaces its rows.

Default destination, outside Git:

```text
Project_DPCO/data/inaturalist2021_full_t140000_v20000_s42/
  .prepare.lock
  selection.json
  manifest.json
  manifest.sha256
  metadata/
    train.json.tar.gz
    validation.json.tar.gz
    annotation_index.sqlite
  images.sqlite             # created by download
  complete.json             # published after successful verification
```

`manifest.json` is canonical, timestamp-free and host-independent. Its SHA-256 is
the selection identity. `selection.json` prevents accidentally reusing partial
output for different settings. No source images or generated data enter Git.

## Prepare selected images without retaining full archives

Preparation runs on the CPU and network, not the TPU. Keep its output in a named
preparation log directory; this is not a training run or a run-status TSV event.

```bash
mkdir -p results/logs/inat2021_full_t140000_v20000_s42_prepare
nohup ../.venv/bin/python -u scripts/prepare_inaturalist.py \
  --stage download --selection-seed 42 \
  --max-storage-gb 15 --reserve-gb 10 \
  >results/logs/inat2021_full_t140000_v20000_s42_prepare/prepare.log \
  2>&1 < /dev/null &
```

An `all` stage runs plan then download. `--train-samples`, `--validation-samples`
and `--selection-seed` configure the plan; download/verify honor the existing
manifest rather than regenerating it. `--output-dir` selects another destination;
relative explicit paths are relative to the current working directory. With
default paths, counts/seed also identify the output directory.

The image archives are read sequentially and discarded except for selected JPEG
bytes. Full source MD5 checks require reading the complete compressed archives,
potentially about 232 GB of network traffic despite only about 13-15 GB retained.
This does **not** promise selective remote download or a 13-15 GB transfer. No
private cloud bucket, Hugging Face account or live-observation query is required.
Official dataset terms still apply:
https://github.com/visipedia/inat_comp/blob/master/2021/README.md

Official annotation/image archives are pinned to their publisher's MD5 values;
selected images also receive individual SHA-256 hashes. Completion requires all
manifest members, source archive checksums and stored-image checks to pass. Images
are published only as a complete logical dataset after both splits verify.

## Retry and reproduce on another host

Rerun the same command after a network/process failure. Inserts are committed in
small batches. Completed verified splits are reused without network access;
interrupted splits replay from the start of the compressed archive, checking
existing image content and filling only missing entries. Archive checksums are
verified before a split is marked complete. An output-directory lock rejects
concurrent writers. A checksum mismatch is an explicit failure, not permission to
silently replace a corrupt cached image. Database files need not be byte-identical
across insertion histories; selected IDs, paths, original bytes and checksums must
match.

On node-ziao1, use the same code version, selection seed and sample counts. Compare:

```bash
cat ../data/inaturalist2021_full_t140000_v20000_s42/manifest.sha256
python scripts/prepare_inaturalist.py --stage verify --selection-seed 42
```

The manifest checksum must be identical to node-ziao2. `verify` is offline and
checks stored bytes and database integrity without changing the selected samples.
Alternatively, copy the prepared directory to another host and verify it there.
If interrupted exactly between manifest/checksum publication, rerun `plan` to
recover the checksum file. Retain manifests for reproducibility even if image
bytes are later removed manually.

## Train from the completed local selection

Use worker0's isolated Python environment with the existing TPU training
dependencies. Local training does not need Hugging Face authentication, additional
dataset libraries or a second download. The preparation `completed` event means
both splits have passed source/content verification.

The reader opens SQLite read-only. Startup checks manifest SHA-256, completion,
source-verification markers, 10,000-class labels, exact split/member identities
and stored byte lengths. It does not rehash 14+ GB of image bytes per run; use
`--stage verify` above for full offline content verification. Missing/incomplete
or mismatched data fails explicitly, without reselecting/downloading/replacing it.

The identifier `inaturalist2021` means this proportional subset of official **full
2021**, not official mini. All 10,000 classifier outputs remain present. Training
seeds control initialization, shuffle and augmentation, separately from manifest
selection seed 42. Existing metadata records the manifest identity; resume checks
the data fingerprint and cursor. No new CSV streams or observation fields are added.

Preview defaults without TPU initialization or local data access:

```bash
cd ~/Project_DPCO/DPCO
source ../.venv/bin/activate
./run_inaturalist_suite.sh --dry-run
```

Run the three real-TPU smokes sequentially first (default three updates, B=8):

```bash
mkdir -p results/logs/inaturalist2021_seed5_smoke_suite
nohup ./run_inaturalist_suite.sh --mode smoke --seeds 5 \
  --methods dpco_e sgd dpco_l \
  --data-dir ../data/inaturalist2021_full_t140000_v20000_s42 \
  >results/logs/inaturalist2021_seed5_smoke_suite/launcher.log 2>&1 < /dev/null &
```

Smoke checks full selection metadata, then limits reads to a prefix of at most
`max(32, 2*B)` examples per split. It does not write another dataset or change its
membership. This view may omit classes; the head still has 10,000 outputs. Its
fingerprint differs from full training, preventing accidental cross-view resume.

After smoke passes, launch the 2,000-step pilot:

```bash
mkdir -p results/logs/inaturalist2021_seed5_train_suite
nohup ./run_inaturalist_suite.sh \
  --mode train --dataset inaturalist2021 --model resnet50 --data-mode clean \
  --methods dpco_e sgd dpco_l --seeds 5 --steps 2000 --batch-size 256 \
  --data-dir ../data/inaturalist2021_full_t140000_v20000_s42 \
  --lr 0.01 --lr-schedule constant --min-lr 0 \
  --momentum 0 --no-nesterov --l2-coefficient 0 \
  --eval-every-epochs 1 --checkpoint-every-steps 100 \
  >results/logs/inaturalist2021_seed5_train_suite/launcher.log 2>&1 < /dev/null &
```

`configs/tpu/inaturalist2021_resnet50.toml` stores these defaults plus random
initialization, float32, image size 224, five-step log printing, 50-step CSV flushing
and probes off. The new shell reuses the existing suite coordinator and accepts
all common method/seed/model, batch, LR/cosine-floor, momentum/Nesterov/L2,
checkpoint and mutually exclusive validation-step/epoch overrides. `--data-dir`
points directly to the directory containing manifest.json, manifest.sha256,
images.sqlite and complete.json. Relative paths resolve against the repository.
Only clean mode is implemented; mismatch labels fail explicitly.

Preprocessing is shared with ImageNet: random resized crop/horizontal flip and
ImageNet mean/std for train; resize/center crop for evaluation. Train drops an
incomplete tail to avoid BatchNorm padding and preserves original valid-B
averaging. Each full validation covers all configured 20,000 images with
valid-count-weighted padding at the tail. Initial/final validation is retained.

Jobs run serially on worker0's four local TPU chips; worker1 is not contacted.
Each method/seed has isolated directories under results/logs, results/runs and
results/files, with inaturalist2021 in its run name. The existing start/end TSV,
CSV schemas, seven checkpoint slots and save-before-cleanup order are reused.
The first failure stops the queue; outer nohup protects the waiting coordinator.

For existing theorem probes, use `--mode validate --methods dpco_e` (or dpco_l)
in a separate queue. SGD remains `--mode train`; validate mode rejects it because
common-reference probes require DPCO. Fractions/LR scales stay in TOML, and T3
retains the current fixed-audit diagnostic scope. No theorem definitions change.

Resume one method/seed with its original settings, including any CLI overrides:

```bash
nohup ./run_inaturalist_suite.sh --mode train --methods dpco_e --seeds 5 \
  --resume YOUR_RUN_NAME --resume-slot latest \
  >results/logs/inaturalist2021_seed5_train_suite/resume.log 2>&1 < /dev/null &
```

This is a short proportional-subset screening run, not official mini/full
benchmark evaluation or a convergence claim. Sparse class-level examples limit
per-class accuracy and from-scratch generalization conclusions.
