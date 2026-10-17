"""Methods return complete directions; trainer owns parameter updates."""

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from src.signals.gradient_alignment import gradient_alignment


class MethodResult(NamedTuple):
    direction: Any
    model_state: Any
    method_state: Any
    metrics: Mapping[str, Any]


class Method(Protocol):
    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None) -> MethodResult:
        ...


def mean_loss(losses, weights, valid_mask):
    """Freeze weights; divide by all valid contributions, not retained ones."""
    losses = jnp.asarray(losses)
    weights = jnp.asarray(weights)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if losses.ndim != 1 or weights.shape != losses.shape or valid.shape != losses.shape:
        raise ValueError("losses, weights and valid_mask must have shape [B]")
    frozen = jax.lax.stop_gradient(jnp.where(valid, weights, 0.))
    safe_losses = jnp.where(valid, losses, 0.)
    count = jnp.sum(valid, dtype=jnp.float32)
    return jnp.sum(frozen * safe_losses) / jnp.where(count > 0, count, jnp.nan)


def aggregate(gradients, weights, valid_mask):
    """Small-batch full-gradient implementation for correctness validation.

    This function expects one complete logical batch, not device-local shards.
    Future distributed code must combine global sums and counts before division.
    All-padding inputs produce NaN; callers must provide a valid logical batch.
    """
    weights = jnp.asarray(weights)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if weights.ndim != 1 or valid.shape != weights.shape:
        raise ValueError("weights and valid_mask must have shape [B]")
    frozen = jax.lax.stop_gradient(jnp.where(valid, weights, 0.))
    count = jnp.sum(valid, dtype=jnp.float32)

    def combine(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim < 1 or leaf.shape[0] != weights.shape[0]:
            raise ValueError("Gradient leaves must have leading batch axis B")
        shape = (valid.shape[0],) + (1,) * (leaf.ndim - 1)
        safe = jnp.where(valid.reshape(shape), leaf, 0.)
        return jnp.sum(safe * frozen.reshape(shape), axis=0) / jnp.where(count > 0, count, jnp.nan)

    return jax.tree_util.tree_map(combine, gradients)


def weighted_direction(task, params, batch, model_state, method_state, weights_fn):
    """One ordinary backward pass; SGD/random do not compute alignment."""
    def objective(p):
        result = task.evaluate(p, batch, model_state)
        weights = weights_fn(result.valid_mask)
        return mean_loss(result.losses, weights, result.valid_mask), result.model_state

    (_, next_model_state), direction = jax.value_and_grad(objective, has_aux=True)(params)
    # No new observation fields or persistent data collection are introduced.
    return MethodResult(direction, next_model_state, method_state, {})


@dataclass(frozen=True)
class AlignmentMethod:
    reference: Any
    exclude_self: bool = False

    def weights(self, scores, has_reference, valid_mask):
        raise NotImplementedError

    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        """Materialize per-example gradients for small-batch audit correctness.

        This is deliberately not advertised as the efficient ImageNet/TPU path.
        Empty reference positions retain the uncorrected sample contribution,
        preserving the old scaffold's explicit no-correction fallback.
        """
        def losses_fn(p):
            result = task.evaluate(p, batch, model_state)
            return result.losses, (result.model_state, result.feedback, result.valid_mask)

        gradients, (next_model_state, feedback, valid) = jax.jacrev(
            losses_fn, has_aux=True
        )(params)
        reference = self.reference.build(gradients, valid, feedback,
                                         exclude_self=self.exclude_self)
        scores = gradient_alignment(gradients, reference, valid)
        weights = self.weights(scores, reference.has_reference, valid)
        return MethodResult(aggregate(gradients, weights, valid),
                            next_model_state, method_state, {})
