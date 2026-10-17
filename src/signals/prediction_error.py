"""Prediction correctness is a signal, not a parameter update rule."""

import jax
import jax.numpy as jnp


def prediction_error(logits, labels, valid_mask):
    return jax.lax.stop_gradient(
        (jnp.argmax(logits, axis=-1) != labels) & jnp.asarray(valid_mask, dtype=jnp.bool_)
    )
