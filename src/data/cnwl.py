"""Read-only CNWL Red cache; audit metadata never enters training batches."""

from io import BytesIO
import hashlib
import math
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from src.data.cifar100n_prepare import canonical
from src.data.cnwl_prepare import cache_dir, level, prepare, verify_prepared
from src.data.images import transform as imagenet_transform


def transform(image, image_size=84, *, rng=None):
    if image_size != 84 or image.size != (84, 84):
        raise ValueError("CNWL uses the released 84x84 image protocol")
    if rng is not None:
        return imagenet_transform(image, image_size, rng=rng)
    # No additional resize/center-crop in fixed train audits or clean validation.
    pixels = np.asarray(image.convert("RGB"), np.float32) / 255.
    return (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)


class CNWLImages:
    channels = 3
    transform = staticmethod(transform)

    def __init__(self, root, *, cnwl_noise_percent=20, noise_percent=None, download=False,
                 limit=None, progress=None):
        percent = level(cnwl_noise_percent)
        if not isinstance(download, bool):
            raise ValueError("download must be boolean")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Smoke sample limit must be positive")
        if download:
            prepare(root, percent=percent, download=True, progress=progress)
        self.manifest, digest = verify_prepared(root, percent)
        self.path = cache_dir(root, percent) / "images.sqlite"
        self.full_counts = dict(self.manifest["counts"])
        self.counts = {split: min(n, limit) if limit is not None else n for split, n in self.full_counts.items()}
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if dict(self.connection.execute("SELECT split,COUNT(*) FROM samples GROUP BY split")) != self.full_counts:
                raise ValueError("CNWL database split counts differ from its manifest")
            rows = self.connection.execute("SELECT id,is_clean FROM samples WHERE split='train' ORDER BY id").fetchall()
            if [i for i, _ in rows] != list(range(self.full_counts["train"])) or any(c not in (0, 1) for _, c in rows):
                raise ValueError("CNWL sample indices/clean indicators are invalid")
            self.corrupted = np.array([not clean for _, clean in rows], dtype=bool)
            self.actual_noise_percent = 100 * float(np.mean(self.corrupted))
            if noise_percent is not None and (isinstance(noise_percent, bool) or not math.isfinite(noise_percent)
                    or not math.isclose(noise_percent, self.actual_noise_percent, rel_tol=0, abs_tol=1e-8)):
                raise ValueError(f"noise_percent checks actual rate: loaded {self.actual_noise_percent}%; cnwl_noise_percent selects the nominal official split")
            if int(np.count_nonzero(self.corrupted)) != self.manifest["corrupted_count"]:
                raise ValueError("CNWL corruption counts differ from manifest")
            self.corrupted.flags.writeable = False
            active = self.corrupted[:self.counts["train"]]
            self.group_indices = {"clean": np.flatnonzero(~active), "corrupted": np.flatnonzero(active)}
            self.identity = {"dataset": "cnwl", "manifest_sha256": digest,
                "noise_type": "red", "cnwl_noise_percent": percent,
                "actual_noise_percent": self.actual_noise_percent, "view_counts": self.counts,
                "full_counts": self.full_counts, "group_counts": {n: len(i) for n, i in self.group_indices.items()},
                "group_partition": "official_clean_source_vs_incorrect_web_images",
                "original_labels_available": "clean_only; web_correct_class_not_supplied",
                "evaluation_split": "official_shared_clean_validation",
                "source_split": self.manifest["provenance"]["train_split"],
                "preprocessing": self.manifest["preprocessing"]}
            self.fingerprint = hashlib.sha256(canonical(self.identity)).hexdigest()
        except BaseException:
            self.close()
            raise

    def get(self, split, index):
        if split not in self.counts or not 0 <= int(index) < self.counts[split]:
            raise IndexError(f"CNWL sample outside configured view: {split}/{index}")
        row = self.connection.execute("SELECT image,label FROM samples WHERE split=? AND id=?", (split, int(index))).fetchone()
        if row is None:
            raise IndexError(f"Missing CNWL sample {split}/{index}")
        with Image.open(BytesIO(row[0])) as image:
            return image.convert("RGB"), int(row[1])

    def close(self):
        self.connection.close()
