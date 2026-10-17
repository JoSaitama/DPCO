"""Explicit step keys; observers must use their own independent stream."""

import jax


def training_key(seed: int, step, *, stream: int = 0):
    """Reproduce keys for the same seed/step/stream and logical input layout.

    This is a single logical-batch convention, not a cross-sharding guarantee.
    Future multi-device code must define global sample identities explicitly.
    """
    return jax.random.fold_in(jax.random.fold_in(jax.random.key(seed), stream), step)
