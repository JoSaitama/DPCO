"""A fixed training-image audit bank, independent of the training iterator."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from src.data.batch import ClassificationBatch
from src.data.images import transform


@dataclass(frozen=True)
class TrainingAuditBatch:
    identity: str
    batch: object
    indices: tuple[int, ...] = ()


def training_audit_batches(images, batch_size, image_size, count, seed):
    """Select once without replacement and freeze the audit augmentations.

    These images stay in ordinary training: this is not a held-out validation
    set. A source batch can overlap the bank. Observers report that overlap
    instead of silently changing the audit objective at each diagnostic node.
    """
    required = count * batch_size
    if images.counts["train"] < required:
        raise ValueError("The diagnostic audit bank needs audit_batches * batch_size training images")
    order = np.random.default_rng(np.random.SeedSequence([seed, 0x53414745])).permutation(images.counts["train"])
    batches = []
    for number in range(count):
        indices = tuple(int(i) for i in order[number * batch_size:(number + 1) * batch_size])
        pixels, labels = [], []
        for index in indices:
            image, label = images.get("train", index)
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0xA0D17]))
            pixels.append(transform(image, image_size, rng=rng))
            labels.append(label)
        identity = {"version": 1, "data": images.fingerprint, "seed": seed,
                    "indices": indices, "image_size": image_size, "augmentation_stream": "audit-v1"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        batch = ClassificationBatch(np.stack(pixels), np.asarray(labels, np.int32), np.ones(batch_size, bool))
        batches.append(TrainingAuditBatch(f"training_audit_{number}_{digest}", batch, indices))
    return tuple(batches)
