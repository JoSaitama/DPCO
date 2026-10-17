"""The classification interface does not depend on a dataset name."""

from typing import Any, NamedTuple

import jax.numpy as jnp


class ClassificationBatch(NamedTuple):
    images: Any
    labels: Any
    valid_mask: Any


def classification_batch(images, labels, valid_mask=None) -> ClassificationBatch:
    """Construct/validate a nonempty batch on the host before a compiled step."""
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)
    if images.ndim < 2 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError("images and integer labels must share leading batch axis B")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("Classification labels must be integer encoded")
    if valid_mask is None:
        valid_mask = jnp.ones(labels.shape, dtype=jnp.bool_)
    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask must have shape [B]")
    if not bool(jnp.any(valid_mask)):
        raise ValueError("A logical training batch must contain valid examples")
    return ClassificationBatch(images, labels, valid_mask)
