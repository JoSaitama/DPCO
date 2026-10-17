"""Uniform batch-mean gradient; no reference/alignment machinery."""

import jax.numpy as jnp

from src.methods.base import weighted_direction


class SGD:
    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        return weighted_direction(
            task, params, batch, model_state, method_state,
            lambda valid: jnp.ones(valid.shape, dtype=jnp.float32),
        )
