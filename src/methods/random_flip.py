"""Independent Bernoulli flips on the entire valid batch, not negative scores."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.methods.base import weighted_direction


@dataclass(frozen=True)
class RandomFlip:
    probability: float = 0.25

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")

    def weights(self, valid_mask, rng_key):
        if rng_key is None:
            raise ValueError("random_flip requires an explicit JAX random key")
        valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
        flipped = jax.random.bernoulli(rng_key, p=float(self.probability), shape=valid.shape)
        return jnp.where(valid, jnp.where(flipped, -1., 1.), 0.)

    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        if rng_key is None:
            raise ValueError("random_flip requires an explicit JAX random key")
        return weighted_direction(
            task, params, batch, model_state, method_state,
            lambda valid: self.weights(valid, rng_key),
        )
