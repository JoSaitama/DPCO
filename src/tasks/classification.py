"""Per-image cross entropy and prediction feedback for any class count."""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from src.data.batch import ClassificationBatch
from src.signals.prediction_error import prediction_error
from src.tasks.base import TaskResult


@dataclass(frozen=True)
class ClassificationTask:
    # model_fn(params, images, model_state) -> (logits, next_model_state).
    model_fn: Callable
    num_classes: int

    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError("Classification requires at least two classes")

    def evaluate(self, params, batch: ClassificationBatch, model_state=None):
        mask = batch.valid_mask
        expanded_mask = mask.reshape((mask.shape[0],) + (1,) * (batch.images.ndim - 1))
        images = jnp.where(expanded_mask, batch.images, 0)
        labels = jnp.where(mask, batch.labels, 0)
        logits, next_state = self.model_fn(params, images, model_state)
        logits = jnp.asarray(logits, dtype=jnp.float32)
        if logits.shape != (labels.shape[0], self.num_classes):
            raise ValueError("Model logits must have shape [B, task.num_classes]")
        targets = jax.nn.one_hot(labels, self.num_classes)
        losses = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        in_range = (labels >= 0) & (labels < self.num_classes)
        losses = jnp.where(mask, jnp.where(in_range, losses, jnp.nan), 0.)
        errors = prediction_error(logits, labels, mask)
        return TaskResult(losses, mask, next_state, {"error_mask": errors})
