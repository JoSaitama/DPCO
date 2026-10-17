"""Analytic test tasks, never production training datasets."""

import jax.numpy as jnp

from src.tasks.base import TaskResult


class QuadraticTask:
    def evaluate(self, params, batch, model_state=None):
        targets, mask, errors = batch
        safe_targets = jnp.where(mask[:, None], targets, 0.)
        losses = 0.5 * jnp.sum((params["x"] - safe_targets) ** 2, axis=-1)
        losses = jnp.where(mask, losses, 0.)
        calls = 0 if model_state is None else model_state["calls"]
        return TaskResult(losses, mask, {"calls": calls + 1}, {"error_mask": errors})


def scalar_batch():
    # At theta=0: g_i=theta-y_i gives [1, 1, -1.5].
    return (jnp.array([[-1.], [-1.], [1.5]]),
            jnp.array([True, True, True]), jnp.array([False, True, False]))
