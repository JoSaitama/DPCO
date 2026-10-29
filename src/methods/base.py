"""Methods return complete directions; trainer owns parameter updates."""

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from src.signals.gradient_alignment import gradient_alignment
from src.training.metrics import task_metrics, tree_dot


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
        return mean_loss(result.losses, weights, result.valid_mask), (result.model_state, task_metrics(result))

    (_, (next_model_state, metrics)), direction = jax.value_and_grad(objective, has_aux=True)(params)
    return MethodResult(direction, next_model_state, method_state, metrics)


@dataclass(frozen=True)
class AlignmentMethod:
    reference: Any
    exclude_self: bool = False
    record_theory: bool = False
    gradient_backend: str = "audit"

    def weights(self, scores, has_reference, valid_mask):
        raise NotImplementedError

    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        """Materialize per-example gradients for small-batch audit correctness.

        This is deliberately not advertised as the efficient ImageNet/TPU path.
        Empty reference positions retain the uncorrected sample contribution,
        preserving the old scaffold's explicit no-correction fallback.
        """
        if self.gradient_backend == "jvp":
            return self._directional(task, params, batch, model_state, method_state)
        if self.gradient_backend != "audit":
            raise ValueError("Unknown gradient backend")

        def losses_fn(p):
            result = task.evaluate(p, batch, model_state)
            return result.losses, (result.model_state, result.feedback, result.valid_mask,
                                   task_metrics(result))

        gradients, (next_model_state, feedback, valid, metrics) = jax.jacrev(
            losses_fn, has_aux=True
        )(params)
        reference = self.reference.build(gradients, valid, feedback,
                                         exclude_self=self.exclude_self)
        scores = gradient_alignment(gradients, reference, valid)
        weights = self.weights(scores, reference.has_reference, valid)
        direction = aggregate(gradients, weights, valid)
        if self.record_theory:
            if self.exclude_self:
                raise ValueError("Common-reference theory recording does not cover LOO")
            count = jnp.sum(valid)
            delta = jnp.sum(jnp.where(valid, (weights - 1.) * scores, 0.)) / count
            metrics.update(theory_delta=delta,
                           theory_negative_count=jnp.sum(valid & (scores < 0)),
                           theory_flip_count=jnp.sum(valid & (weights < 0)),
                           theory_error_count=jnp.sum(valid & feedback.get("error_mask", jnp.zeros_like(valid))),
                           theory_reference_valid=reference.has_reference,
                           theory_reference_projection_sgd=jnp.sum(jnp.where(valid, scores, 0.)) / count,
                           theory_reference_projection_dpco=tree_dot(reference.direction, direction))
        return MethodResult(direction, next_model_state, method_state, metrics)

    def _directional(self, task, params, batch, model_state, method_state):
        """Exact full-parameter scores via JVP, without a [B, P] Jacobian.

        The reference and weights are frozen. VJP returns both reference and
        weighted update gradients; JVP computes d(loss_vector)/dparams @ r.
        Batch-dependent layers keep the same realized batch objective throughout.
        """
        if self.exclude_self:
            raise ValueError("JVP currently supports full common references only")

        def losses_fn(p):
            result = task.evaluate(p, batch, model_state)
            return result.losses, (result.model_state, result.feedback,
                                   result.valid_mask, task_metrics(result))

        losses, pullback, (next_state, feedback, valid, metrics) = jax.vjp(losses_fn, params, has_aux=True)
        selected = self.reference.selection_mask(valid, feedback)
        selected = jax.lax.stop_gradient(selected)
        selected_count = jnp.sum(selected)
        reference = pullback(selected.astype(losses.dtype) / jnp.maximum(selected_count, 1))[0]
        reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
        scores = jax.jvp(lambda p: task.evaluate(p, batch, model_state).losses,
                         (params,), (reference,))[1]
        weights = jax.lax.stop_gradient(self.weights(scores, selected_count > 0, valid))
        count = jnp.sum(valid)
        direction = pullback(weights.astype(losses.dtype) / count)[0]
        if self.record_theory:
            delta = jnp.sum(jnp.where(valid, (weights - 1.) * scores, 0.)) / count
            metrics.update(theory_delta=delta,
                theory_negative_count=jnp.sum(valid & (scores < 0)),
                theory_flip_count=jnp.sum(valid & (weights < 0)),
                theory_error_count=jnp.sum(valid & feedback.get("error_mask", jnp.zeros_like(valid))),
                theory_reference_valid=selected_count > 0,
                theory_reference_projection_sgd=jnp.sum(jnp.where(valid, scores, 0.)) / count,
                theory_reference_projection_dpco=tree_dot(reference, direction))
        return MethodResult(direction, next_state, method_state, metrics)
