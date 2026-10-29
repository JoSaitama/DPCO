"""Current-inclusive raw-gradient EMA reference with configurable contributions."""

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from src.methods.base import aggregate
from src.training.metrics import task_metrics, tree_dot


class ReferenceState(NamedTuple):
    mean: Any
    updates: Any


@dataclass(frozen=True)
class HistoricalDPCO:
    beta: float = 0.9
    negative_weight: float = -1.0
    gradient_backend: str = "jvp"

    def __post_init__(self):
        if isinstance(self.beta, bool) or not math.isfinite(self.beta) or not 0 <= self.beta < 1:
            raise ValueError("reference beta must be finite and in [0, 1)")
        if isinstance(self.negative_weight, bool) or not math.isfinite(self.negative_weight):
            raise ValueError("negative_weight must be a finite number")
        if self.gradient_backend not in ("audit", "jvp"):
            raise ValueError("Unknown gradient backend")

    def initial_state(self, params):
        return ReferenceState(jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.result_type(p.dtype, jnp.float32)), params),
            jnp.array(0, jnp.int32))

    def validate_state(self, state, params, step):
        if (not isinstance(state, ReferenceState) or jnp.shape(state.updates) != ()
                or not jnp.issubdtype(jnp.asarray(state.updates).dtype, jnp.integer)
                or int(state.updates) < 0 or int(state.updates) != int(step)):
            raise ValueError("DPCO reference state/count does not match completed updates")
        if jax.tree_util.tree_structure(state.mean) != jax.tree_util.tree_structure(params):
            raise ValueError("DPCO reference tree differs from parameters")
        for mean, param in zip(jax.tree_util.tree_leaves(state.mean), jax.tree_util.tree_leaves(params)):
            if mean.shape != param.shape or not bool(jnp.all(jnp.isfinite(mean))):
                raise ValueError("DPCO reference has invalid shape or non-finite values")

    def compute_direction(self, task, params, batch, model_state=None, method_state=None, rng_key=None):
        if method_state is None:
            method_state = self.initial_state(params)

        def losses_fn(p):
            result = task.evaluate(p, batch, model_state)
            return result.losses, (result.model_state, result.valid_mask, task_metrics(result))

        if self.gradient_backend == "audit":
            gradients, (next_model_state, valid, metrics) = jax.jacrev(losses_fn, has_aux=True)(params)
            raw = aggregate(gradients, jnp.ones(valid.shape), valid)
        else:
            losses, pullback, (next_model_state, valid, metrics) = jax.vjp(losses_fn, params, has_aux=True)
            count = jnp.sum(valid)
            denominator = jnp.where(count > 0, count, jnp.nan)
            raw = pullback(valid.astype(losses.dtype) / denominator)[0]

        updates = method_state.updates + 1
        mean = jax.tree_util.tree_map(
            lambda old, g: self.beta * old + (1 - self.beta) * g, method_state.mean, raw)
        # expm1 avoids cancellation for beta very close to one; beta=0 is exact.
        correction = 1. if self.beta == 0 else -jnp.expm1(updates * math.log(self.beta))
        reference = jax.tree_util.tree_map(lambda m: jax.lax.stop_gradient(m / correction), mean)
        if self.gradient_backend == "audit":
            scores = jax.vmap(lambda g: tree_dot(reference, g))(gradients)
        else:
            scores = jax.jvp(lambda p: losses_fn(p)[0], (params,), (reference,))[1]
        negative = valid & (scores < 0)
        weights = jax.lax.stop_gradient(jnp.where(valid, jnp.where(negative, self.negative_weight, 1.), 0.))
        if self.gradient_backend == "audit":
            direction = aggregate(gradients, weights, valid)
        else:
            direction = pullback(weights.astype(losses.dtype) / denominator)[0]
        count = jnp.sum(valid)
        ref_sq, raw_sq = tree_dot(reference, reference), tree_dot(raw, raw)
        norm_product = jnp.sqrt(ref_sq * raw_sq)
        metrics.update(dpco_negative_count=jnp.sum(negative),
            dpco_edited_count=jnp.sum(negative) * (self.negative_weight != 1),
            dpco_negative_weight=jnp.asarray(self.negative_weight),
            dpco_reference_updates=updates,
            dpco_reference_norm=jnp.sqrt(ref_sq), dpco_raw_norm=jnp.sqrt(raw_sq),
            dpco_direction_norm=jnp.sqrt(tree_dot(direction, direction)),
            dpco_reference_cosine=jnp.where(norm_product > 0,
                tree_dot(reference, raw) / jnp.where(norm_product > 0, norm_product, 1.), 0.),
            dpco_reference_gain=jnp.sum(jnp.where(valid, (weights - 1) * scores, 0.)) / count)
        from src.methods.base import MethodResult
        return MethodResult(direction, next_model_state, ReferenceState(mean, updates), metrics)
