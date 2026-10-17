"""Error selects a subset; reference uses that subset's surrogate gradients."""

import jax.numpy as jnp

from src.references.base import selected_mean


class ErrorSubsetReference:
    def build(self, gradients, valid_mask, feedback, *, exclude_self=False):
        if "error_mask" not in feedback:
            raise ValueError("ErrorSubsetReference requires task error_mask feedback")
        errors = jnp.asarray(feedback["error_mask"], dtype=jnp.bool_)
        valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
        if errors.shape != valid.shape:
            raise ValueError("error_mask and valid_mask must have the same shape [B]")
        return selected_mean(gradients, valid & errors, exclude_self=exclude_self)
