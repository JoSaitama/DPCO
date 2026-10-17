"""Common/self-excluded reference filter, without retained-count rescaling."""

import jax.numpy as jnp

from src.methods.base import AlignmentMethod


class DTV(AlignmentMethod):
    def weights(self, scores, has_reference, valid_mask):
        drop = (scores < 0) & has_reference
        return jnp.where(valid_mask, jnp.where(drop, 0., 1.), 0.)
