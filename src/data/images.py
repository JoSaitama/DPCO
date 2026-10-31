"""Shared local image-byte reader, transforms and reproducible batch iteration."""

from io import BytesIO
import math
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image


class PreparedImages:
    def __init__(self, path, fingerprint):
        self.path, self.fingerprint = Path(path), fingerprint
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            self.counts = dict(self.connection.execute("SELECT split, COUNT(*) FROM samples GROUP BY split"))
        except BaseException:
            self.connection.close()
            raise

    def get(self, split, index):
        row = self.connection.execute("SELECT image,label FROM samples WHERE split=? AND id=?", (split, int(index))).fetchone()
        if row is None:
            raise IndexError(f"Missing {split} image {index}")
        with Image.open(BytesIO(row[0])) as image:
            decoded = image.convert("RGB")
        return decoded, row[1]

    def close(self):
        self.connection.close()


def transform(image, image_size, *, rng=None):
    """Random resized crop/flip for train; resize/center crop for evaluation."""
    width, height = image.size
    if rng is not None:
        for _ in range(10):
            area = width * height * rng.uniform(.08, 1.)
            ratio = math.exp(rng.uniform(math.log(.75), math.log(4 / 3)))
            crop_w, crop_h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left, top = int(rng.integers(width - crop_w + 1)), int(rng.integers(height - crop_h + 1))
                image = image.crop((left, top, left + crop_w, top + crop_h))
                break
        else:
            ratio = width / height
            crop_w, crop_h = (width, round(width / .75)) if ratio < .75 else ((round(height * 4 / 3), height) if ratio > 4 / 3 else (width, height))
            left, top = (width - crop_w) // 2, (height - crop_h) // 2
            image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        short = round(image_size * 256 / 224)
        scale = short / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left, top = (image.width - image_size) // 2, (image.height - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))
    pixels = np.asarray(image, dtype=np.float32) / 255.
    return (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)


class ImageBatchSource:
    def __init__(self, images, batch_size, image_size, seed):
        self.images, self.batch_size, self.image_size, self.seed = images, batch_size, image_size, seed
        self.transform = getattr(images, "transform", transform)
        self.count = images.counts["train"]
        if self.count < batch_size:
            raise ValueError("Training data must contain at least one complete batch")
        self.epoch, self.cursor = 0, 0
        self._shuffle()

    def _shuffle(self):
        self.order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 0])).permutation(self.count)

    def __next__(self):
        from src.data.batch import ClassificationBatch
        # Drop the incomplete training tail so BatchNorm never consumes padding.
        if self.cursor + self.batch_size > self.count:
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        pixels, labels = [], []
        for index in indices:
            image, label = self.images.get("train", index)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index), 1]))
            pixels.append(self.transform(image, self.image_size, rng=rng))
            labels.append(label)
        return ClassificationBatch(np.stack(pixels), np.array(labels, np.int32), np.ones(len(labels), bool))

    def snapshot(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed,
                "data_fingerprint": self.images.fingerprint}

    def diagnostic_context(self):
        """Describe the last emitted batch without advancing any RNG or cursor."""
        start = self.cursor - self.batch_size
        if start < 0:
            raise ValueError("A diagnostic context requires an already emitted batch")
        return {"batch_id": f"train_epoch_{self.epoch}_offset_{start}",
                "sample_indices": [int(i) for i in self.order[start:self.cursor]],
                "epoch": self.epoch, "cursor_start": start, "training_seed": self.seed,
                "data_fingerprint": self.images.fingerprint,
                "augmentation_stream": "training: seed/epoch/sample-index/1"}

    def restore(self, snapshot):
        if snapshot["seed"] != self.seed or snapshot["data_fingerprint"] != self.images.fingerprint:
            raise ValueError("Data snapshot does not match this experiment")
        self.epoch, self.cursor = snapshot["epoch"], snapshot["cursor"]
        self._shuffle()


def evaluation_batches(images, batch_size, image_size, *, split="validation", indices=None):
    from src.data.batch import ClassificationBatch
    indices = list(range(images.counts[split])) if indices is None else list(indices)
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        pixels = np.zeros((batch_size, image_size, image_size, getattr(images, "channels", 3)), np.float32)
        labels, mask = np.zeros(batch_size, np.int32), np.zeros(batch_size, bool)
        for offset, index in enumerate(ids):
            image, label = images.get(split, index)
            pixels[offset], labels[offset], mask[offset] = getattr(images, "transform", transform)(image, image_size), label, True
        yield ClassificationBatch(pixels, labels, mask)
