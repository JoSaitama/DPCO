"""Use current correctly classified samples' surrogate gradients as reference."""

import jax.numpy as jnp

from src.references.base import selected_mean


class CorrectSubsetReference:
    def selection_mask(self, valid_mask, feedback):
        if "error_mask" not in feedback:
            raise ValueError("Correct reference requires error_mask feedback")
        valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
        errors = jnp.asarray(feedback["error_mask"], dtype=jnp.bool_)
        if errors.shape != valid.shape:
            raise ValueError("error_mask and valid_mask must have the same shape [B]")
        # Complement correctness only within the valid batch; padding is never
        # a correct sample. Discrete prediction masks are frozen by the caller.
        return valid & ~errors

    def build(self, gradients, valid_mask, feedback, *, exclude_self=False):
        # The reference denominator is the correct count. The final update still
        # uses the original valid B in the existing DPCO aggregation path.
        return selected_mean(gradients, self.selection_mask(valid_mask, feedback),
                             exclude_self=exclude_self)
