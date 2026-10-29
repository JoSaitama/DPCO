"""CIFAR-100N official label modes in canonical CIFAR Python image order."""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.cifar100n_prepare import canonical, labels_vector, prepare, verify_prepared


# Match the CIFAR-100 normalization used by the official release's data loader.
MEAN = np.array([0.5071, 0.4867, 0.4408], np.float32)
STD = np.array([0.2675, 0.2565, 0.2761], np.float32)


def transform(image, image_size=32, *, rng=None):
    if image_size != 32 or image.size != (32, 32):
        raise ValueError("CIFAR-100N uses native 32x32 images")
    pixels = np.asarray(image, np.uint8)
    if rng is not None:
        # Official random crop with zero padding=4, then horizontal flip.
        padded = np.pad(pixels, ((4, 4), (4, 4), (0, 0)), mode="constant")
        top, left = rng.integers(0, 9, size=2)
        pixels = padded[top:top+32, left:left+32]
        if rng.random() < .5:
            pixels = pixels[:, ::-1]
    return (pixels.astype(np.float32) / 255. - MEAN) / STD


def synthetic_labels(clean, human, seed=0):
    """Official --no-is-human control: empirical class transition, RandomState.

    Canonical CIFAR-100 has exactly 500 examples per class. Normalize rows
    explicitly (equivalent to the official broadcast there); empty fixture rows
    use identity and cannot affect a sampled training label.
    """
    count = len(clean)
    clean = labels_vector(clean, count, name="clean")
    human = labels_vector(human, count, name="human")
    transition = np.zeros((100, 100), np.float64)
    np.add.at(transition, (clean, human), 1)
    sizes = transition.sum(axis=1)
    transition = transition / np.maximum(sizes[:, None], 1)
    transition[sizes == 0] = np.eye(100)[sizes == 0]
    flipper = np.random.RandomState(seed)
    # Same RNG, example order and multinomial calls as data/utils.py.
    labels = np.array([np.flatnonzero(flipper.multinomial(1, transition[y], 1)[0])[0]
                       for y in clean], np.int32)
    return labels, transition


class CIFAR100NImages:
    transform = staticmethod(transform)

    def __init__(self, root, *, noise_type="noisy100", is_human=True, noise_seed=0,
                 noise_percent=None, download=False, limit=None, progress=None):
        if noise_type not in ("clean100", "noisy100"):
            raise ValueError("noise_type must be clean100 or noisy100")
        if not isinstance(is_human, bool) or not isinstance(download, bool):
            raise ValueError("is_human/download must be boolean")
        if isinstance(noise_seed, bool) or not isinstance(noise_seed, int) or not 0 <= noise_seed < 2**32:
            raise ValueError("noise_seed must be a uint32 integer")
        if is_human and noise_seed != 0:
            raise ValueError("Human labels are fixed; noise_seed applies only to synthetic mode")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Smoke sample limit must be a positive integer")
        root = Path(root).expanduser().resolve()
        if download:
            prepare(root, download=True, progress=progress)
        if not (root / "prepared" / "complete.json").exists():
            raise FileNotFoundError(f"CIFAR-100N is not prepared at {root}; run scripts/prepare_cifar100n.py --stage download")
        manifest, digest = verify_prepared(root)
        self.path = root / "prepared"
        self.manifest = manifest
        self.full_counts = dict(manifest["counts"])
        if set(self.full_counts) != {"train", "validation"} or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.full_counts.values()):
            raise ValueError("Invalid CIFAR split counts")
        self.pixels = {}
        self._closed = False
        try:
            for split, filename in (("train", "train_images.npy"), ("validation", "test_images.npy")):
                self.pixels[split] = np.load(self.path / filename, mmap_mode="r", allow_pickle=False)
            for split, pixels in self.pixels.items():
                if pixels.dtype != np.uint8 or pixels.shape != (self.full_counts[split], 32, 32, 3):
                    self.close()
                    raise ValueError("CIFAR image shapes differ from manifest")
            with np.load(self.path / "labels.npz", allow_pickle=False) as labels:
                self.original_labels = labels_vector(labels["clean_label"], self.full_counts["train"], name="original")
                self.human_labels = labels_vector(labels["noise_label"], self.full_counts["train"], name="human")
                self.test_labels = labels_vector(labels["test_label"], self.full_counts["validation"], name="test")
            human_mismatch = int(np.count_nonzero(self.human_labels != self.original_labels))
            if (human_mismatch != manifest["human_mismatch_count"] or not math.isclose(
                    100 * human_mismatch / self.full_counts["train"], manifest["human_noise_percent"], abs_tol=1e-10)):
                self.close()
                raise ValueError("CIFAR human-label mismatch metadata differs from labels")
            self.reference_labels = self.human_labels if is_human else synthetic_labels(
                self.original_labels, self.human_labels, noise_seed)[0]
            self.observed_labels = self.original_labels if noise_type == "clean100" else self.reference_labels
            self.corrupted = self.reference_labels != self.original_labels
            self.noise_or_not = self.observed_labels != self.original_labels
            self.actual_noise_rate = float(np.mean(self.noise_or_not))
            self.actual_noise_percent = 100 * self.actual_noise_rate
            if noise_percent is not None:
                if (isinstance(noise_percent, bool) or not math.isfinite(noise_percent) or not 0 <= noise_percent <= 100
                        or not math.isclose(noise_percent, self.actual_noise_percent, rel_tol=0, abs_tol=1e-8)):
                    self.close()
                    raise ValueError(f"noise_percent checks the actual rate, not a generator: requested {noise_percent}, "
                                     f"loaded {self.actual_noise_percent:.12g}% ({noise_type}, is_human={is_human})")
            self.counts = {split: min(n, limit) if limit is not None else n for split, n in self.full_counts.items()}
            # All modes, including clean100 controls, use the same reference-label
            # partition. No original labels or masks enter ClassificationBatch.
            self.group_indices = {"clean": np.flatnonzero(~self.corrupted[:self.counts["train"]]),
                                  "corrupted": np.flatnonzero(self.corrupted[:self.counts["train"]])}
            self.identity = {"dataset": "cifar100n", "manifest_sha256": digest,
                "official_revision": manifest["provenance"].get("official_revision"),
                "noise_type": noise_type, "is_human": is_human, "noise_seed": noise_seed,
                "actual_noise_percent": self.actual_noise_percent,
                "actual_noise_rate": self.actual_noise_rate,
                "active_mismatch_count": int(np.count_nonzero(self.noise_or_not)),
                "human_noise_percent": manifest["human_noise_percent"],
                "reference_noise_percent": 100 * float(np.mean(self.corrupted)),
                "observed_labels_sha256": hashlib.sha256(self.observed_labels.tobytes()).hexdigest(),
                "reference_labels_sha256": hashlib.sha256(self.reference_labels.tobytes()).hexdigest(),
                "group_partition": "reference_labels_vs_original; unchanged for clean100 controls",
                "view_counts": self.counts, "full_counts": self.full_counts,
                "group_counts": {name: len(ids) for name, ids in self.group_indices.items()},
                "evaluation_split": "official_test_not_for_model_selection"}
            self.fingerprint = hashlib.sha256(canonical(self.identity)).hexdigest()
            for labels in (self.original_labels, self.human_labels, self.test_labels, self.reference_labels, self.observed_labels):
                labels.flags.writeable = False
            self.corrupted.flags.writeable = False
            self.noise_or_not.flags.writeable = False
        except BaseException:
            self.close()
            raise

    def get(self, split, index):
        if self._closed:
            raise ValueError("CIFAR image reader is closed")
        if split not in self.counts or not 0 <= int(index) < self.counts[split]:
            raise IndexError(f"CIFAR image outside configured view: {split}/{index}")
        label = self.observed_labels[int(index)] if split == "train" else self.test_labels[int(index)]
        return Image.fromarray(self.pixels[split][int(index)]), int(label)

    def example(self, index):
        """Official TFDS label meanings, in canonical Python order; audit only."""
        image, observed = self.get("train", index)
        return {"id": str(int(index)), "image": np.asarray(image),
                "label": int(self.original_labels[index]), "noise_label": int(self.human_labels[index]),
                "observed_label": observed}

    def label_view(self, kind):
        if kind not in ("original", "reference"):
            raise ValueError("Unknown CIFAR evaluation label view")
        return _LabelView(self, kind)

    def close(self):
        self._closed = True
        for pixels in getattr(self, "pixels", {}).values():
            mmap = getattr(pixels, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()


class _LabelView:
    def __init__(self, images, kind):
        self.images, self.kind = images, kind
        self.counts, self.transform = images.counts, images.transform

    def get(self, split, index):
        image, label = self.images.get(split, index)
        if split == "train":
            labels = self.images.original_labels if self.kind == "original" else self.images.reference_labels
            label = int(labels[index])
        return image, label
