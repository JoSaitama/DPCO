"""Native grayscale MNIST with fixed, exact-count synthetic label corruption."""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.cifar100n_prepare import canonical
from src.data.mnist_prepare import prepare, verify_prepared


def transform(image, image_size=28, *, rng=None):
    # No random crop/flip: digits retain their native orientation and geometry.
    if image_size != 28 or image.size != (28, 28) or image.mode != "L":
        raise ValueError("MNIST requires native 28x28 grayscale images")
    return np.asarray(image, np.float32)[..., None] / 255.


def corrupt_labels(clean, percent, seed):
    clean = np.asarray(clean)
    if clean.ndim != 1 or len(clean) == 0 or clean.dtype.kind not in "iu" or np.any((clean < 0) | (clean > 9)):
        raise ValueError("MNIST labels must be nonempty integers in [0,9]")
    if isinstance(percent, bool) or not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("label_noise_percent must be in [0,100]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("noise_seed must be a uint32 integer")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))
    offsets = rng.integers(1, 10, size=len(clean))
    count = int(math.floor(len(clean) * percent / 100 + .5))
    ids = order[:count]
    noisy = clean.astype(np.int32).copy()
    noisy[ids] = (clean[ids] + offsets[ids]) % 10
    return noisy


class MNISTImages:
    channels = 1
    transform = staticmethod(transform)

    def __init__(self, root, *, label_noise_percent=20., noise_seed=0, noise_percent=None,
                 download=False, limit=None, progress=None):
        if not isinstance(download, bool):
            raise ValueError("download must be boolean")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Smoke sample limit must be a positive integer")
        root = Path(root).expanduser().resolve()
        if download:
            prepare(root, download=True, progress=progress)
        if not (root / "prepared" / "complete.json").exists():
            raise FileNotFoundError(f"MNIST is not prepared at {root}; run scripts/prepare_mnist.py --stage download")
        manifest, digest = verify_prepared(root)
        self.path, self.manifest = root / "prepared", manifest
        self.full_counts = dict(manifest["counts"])
        if set(self.full_counts) != {"train", "validation"} or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.full_counts.values()):
            raise ValueError("Invalid MNIST split counts")
        self.pixels, self._closed = {}, False
        try:
            for split, name in (("train", "train_images.npy"), ("validation", "test_images.npy")):
                pixels = np.load(self.path / name, mmap_mode="r", allow_pickle=False)
                self.pixels[split] = pixels
                if pixels.dtype != np.uint8 or pixels.shape != (self.full_counts[split], 28, 28):
                    raise ValueError("MNIST image shape differs from manifest")
            with np.load(self.path / "labels.npz", allow_pickle=False) as data:
                self.original_labels = data["train_label"]
                self.test_labels = data["test_label"]
            for split, labels in (("train", self.original_labels), ("validation", self.test_labels)):
                if (labels.shape != (self.full_counts[split],) or labels.dtype.kind not in "iu"
                        or np.any((labels < 0) | (labels > 9))):
                    raise ValueError("MNIST label shape/range differs from manifest")
            self.observed_labels = corrupt_labels(self.original_labels, label_noise_percent, noise_seed)
            self.reference_labels = self.observed_labels
            self.corrupted = self.observed_labels != self.original_labels
            self.actual_noise_percent = 100 * float(np.mean(self.corrupted))
            if noise_percent is not None and (isinstance(noise_percent, bool) or not math.isfinite(noise_percent)
                    or not 0 <= noise_percent <= 100 or not math.isclose(
                        noise_percent, self.actual_noise_percent, rel_tol=0, abs_tol=1e-8)):
                raise ValueError(f"noise_percent checks actual rate: loaded {self.actual_noise_percent}%")
            self.counts = {split: min(n, limit) if limit is not None else n for split, n in self.full_counts.items()}
            self.group_indices = {"clean": np.flatnonzero(~self.corrupted[:self.counts["train"]]),
                                  "corrupted": np.flatnonzero(self.corrupted[:self.counts["train"]])}
            self.identity = {"dataset": "mnist", "manifest_sha256": digest,
                "noise_type": "symmetric_exclude_original_exact_count", "noise_seed": noise_seed,
                "label_noise_percent": label_noise_percent, "actual_noise_percent": self.actual_noise_percent,
                "active_mismatch_count": int(np.count_nonzero(self.corrupted)),
                "observed_labels_sha256": hashlib.sha256(self.observed_labels.tobytes()).hexdigest(),
                "corruption_algorithm": "numpy_default_rng_permutation_then_all_offsets_v1; round_half_up_count",
                "view_counts": self.counts, "full_counts": self.full_counts,
                "group_counts": {name: len(ids) for name, ids in self.group_indices.items()},
                "group_partition": "observed_labels_vs_original",
                "evaluation_split": "official_mnist_test_not_for_model_selection"}
            self.fingerprint = hashlib.sha256(canonical(self.identity)).hexdigest()
            for labels in (self.original_labels, self.observed_labels, self.test_labels, self.corrupted):
                labels.flags.writeable = False
        except BaseException:
            self.close()
            raise

    def get(self, split, index):
        if self._closed:
            raise ValueError("MNIST reader is closed")
        if split not in self.counts or not 0 <= int(index) < self.counts[split]:
            raise IndexError(f"MNIST image outside configured view: {split}/{index}")
        labels = self.observed_labels if split == "train" else self.test_labels
        return Image.fromarray(self.pixels[split][int(index)]), int(labels[int(index)])

    def label_view(self, kind):
        if kind not in ("original", "reference"):
            raise ValueError("Unknown MNIST label view")
        return _LabelView(self, kind)

    def close(self):
        self._closed = True
        for pixels in self.pixels.values():
            mmap = getattr(pixels, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()


class _LabelView:
    channels = 1

    def __init__(self, images, kind):
        self.images, self.kind = images, kind
        self.counts, self.transform = images.counts, images.transform

    def get(self, split, index):
        image, label = self.images.get(split, index)
        if split == "train":
            labels = self.images.original_labels if self.kind == "original" else self.images.reference_labels
            label = int(labels[index])
        return image, label
