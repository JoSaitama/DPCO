"""Flip negative raw alignment contributions, keeping the original B."""

import jax.numpy as jnp

from src.methods.base import AlignmentMethod


class DPCO(AlignmentMethod):
    def weights(self, scores, has_reference, valid_mask):
        flip = (scores < 0) & has_reference
        return jnp.where(valid_mask, jnp.where(flip, -1., 1.), 0.)
