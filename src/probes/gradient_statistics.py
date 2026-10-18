"""Exact common-reference probe statistics without a batch-by-parameter Jacobian."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class ProbeStatistics(NamedTuple):
    losses: Any
    valid: Any
    errors: Any
    has_reference: Any
    scores: Any
    reference: Any
    sgd: Any
    sage: Any
    sample_energy: Any


def common_reference_statistics(task, reference, params, batch, model_state, *, include_energy):
    """Keep the entire logical batch, including all BatchNorm dependencies.

    A VJP/JVP computes reference, scores and both aggregate directions. If exact
    sample energy is required, one cotangent row is pulled back at a time inside
    a compiled loop and immediately reduced to a scalar. This changes neither B
    nor the reference and does not split model evaluation into microbatches.
    """
    def losses_fn(p):
        result = task.evaluate(p, batch, model_state)
        return result.losses, (result.valid_mask, result.feedback)

    losses, pullback, (valid, feedback) = jax.vjp(losses_fn, params, has_aux=True)
    selected = jax.lax.stop_gradient(reference.selection_mask(valid, feedback))
    selected_count = jnp.sum(selected, dtype=jnp.float32)
    count = jnp.sum(valid, dtype=jnp.float32)
    direction = pullback(selected.astype(losses.dtype) / jnp.maximum(selected_count, 1.))[0]
    direction = jax.tree_util.tree_map(jax.lax.stop_gradient, direction)
    scores = jax.jvp(lambda p: task.evaluate(p, batch, model_state).losses,
                     (params,), (direction,))[1]
    weights = jax.lax.stop_gradient(jnp.where(valid,
        jnp.where((scores < 0) & (selected_count > 0), -1., 1.), 0.))
    gd = pullback(valid.astype(losses.dtype) / count)[0]
    gs = pullback(weights.astype(losses.dtype) / count)[0]

    energy = jnp.asarray(0., dtype=jnp.float32)
    if include_energy:
        def add_energy(index, total):
            cotangent = jax.nn.one_hot(index, losses.shape[0], dtype=losses.dtype)
            gradient = pullback(cotangent)[0]
            norm_sq = sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(gradient))
            return total + jnp.where(valid[index], norm_sq, 0.)

        # Only a scalar is carried across iterations; no [B, P] result or B
        # simultaneous backward activation trees are constructed or returned.
        energy = jax.lax.fori_loop(0, losses.shape[0], add_energy, energy) / count
    return ProbeStatistics(losses, valid, feedback["error_mask"], selected_count > 0,
                           scores, direction, gd, gs, energy)
