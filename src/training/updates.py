"""Parameter updates consume a complete direction; never divide by B again."""

from dataclasses import dataclass
import math
from typing import Any, Callable, NamedTuple, Protocol

import jax


class UpdateResult(NamedTuple):
    params: Any
    optimizer_state: Any


class MomentumState(NamedTuple):
    velocity: Any


class ParameterUpdater(Protocol):
    def apply(self, params, direction, step, optimizer_state=None) -> UpdateResult:
        ...


@dataclass(frozen=True)
class SGDUpdate:
    learning_rate: float | Callable
    l2_coefficient: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False

    def __post_init__(self):
        if (isinstance(self.momentum, bool) or not math.isfinite(self.momentum)
                or not 0 <= self.momentum < 1):
            raise ValueError("momentum must be finite and in [0, 1)")
        if not isinstance(self.nesterov, bool) or (self.nesterov and self.momentum == 0):
            raise ValueError("nesterov must be a boolean and requires nonzero momentum")
        if (isinstance(self.l2_coefficient, bool)
                or not math.isfinite(self.l2_coefficient) or self.l2_coefficient < 0):
            raise ValueError("l2_coefficient must be nonnegative and finite")
        if not callable(self.learning_rate):
            if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
                raise ValueError("learning_rate must be positive and finite")

    def init(self, params):
        if self.momentum == 0:
            return None
        return MomentumState(jax.tree_util.tree_map(jax.numpy.zeros_like, params))

    def apply(self, params, direction, step, optimizer_state=None):
        eta = self.learning_rate(step) if callable(self.learning_rate) else self.learning_rate
        # This is a static Python branch: disabled L2 adds no traced operations,
        # parameter norm reductions, optimizer buffers, or device synchronization.
        if self.l2_coefficient != 0:
            # Add coupled L2 only after sample weighting; never flip the penalty.
            # Match the Flax ImageNet mask: exclude biases and BN scale/offset.
            direction = jax.tree_util.tree_map(
                lambda p, d: d + self.l2_coefficient * p if p.ndim > 1 else d,
                params, direction)
        if self.momentum != 0:
            if optimizer_state is None:
                optimizer_state = self.init(params)
            if not isinstance(optimizer_state, MomentumState):
                raise ValueError("Expected MomentumState for a momentum update")
            velocity = jax.tree_util.tree_map(
                lambda v, d: self.momentum * v + d, optimizer_state.velocity, direction)
            if self.nesterov:
                direction = jax.tree_util.tree_map(
                    lambda d, v: d + self.momentum * v, direction, velocity)
            else:
                direction = velocity
            optimizer_state = MomentumState(velocity)
        next_params = jax.tree_util.tree_map(lambda p, d: p - eta * d, params, direction)
        return UpdateResult(next_params, optimizer_state)


def make_updater(training):
    from src.training.learning_rates import learning_rate_for
    if training.optimizer == "adam":
        return AdamUpdate(learning_rate_for(training), training.adam_beta1,
                          training.adam_beta2, training.adam_epsilon, training.l2_coefficient)
    return SGDUpdate(learning_rate_for(training), l2_coefficient=training.l2_coefficient,
                     momentum=training.momentum, nesterov=training.nesterov)


@dataclass(frozen=True)
class AdamUpdate:
    """Standard Adam (not AdamW); rate uses absolute completed-update count."""
    learning_rate: float | Callable
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    l2_coefficient: float = 0.0

    def __post_init__(self):
        SGDUpdate(self.learning_rate, self.l2_coefficient)
        for beta in (self.beta1, self.beta2):
            if isinstance(beta, bool) or not math.isfinite(beta) or not 0 <= beta < 1:
                raise ValueError("Adam betas must be finite and in [0, 1)")
        if isinstance(self.epsilon, bool) or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Adam epsilon must be finite and positive")

    def _transform(self):
        import optax
        return optax.scale_by_adam(b1=self.beta1, b2=self.beta2, eps=self.epsilon, eps_root=0.)

    def init(self, params):
        return self._transform().init(params)

    def validate_state(self, state, params, step):
        import optax
        if (not isinstance(state, optax.ScaleByAdamState) or jax.numpy.shape(state.count) != ()
                or not jax.numpy.issubdtype(jax.numpy.asarray(state.count).dtype, jax.numpy.integer)
                or int(state.count) < 0 or int(state.count) != int(step)):
            raise ValueError("Adam optimizer state/count does not match completed updates")
        for moments in (state.mu, state.nu):
            if jax.tree_util.tree_structure(moments) != jax.tree_util.tree_structure(params):
                raise ValueError("Adam moment tree differs from parameters")
            for m, p in zip(jax.tree_util.tree_leaves(moments), jax.tree_util.tree_leaves(params)):
                if m.shape != p.shape or not bool(jax.numpy.all(jax.numpy.isfinite(m))):
                    raise ValueError("Adam moments have invalid shape or non-finite values")
        if any(bool(jax.numpy.any(v < 0)) for v in jax.tree_util.tree_leaves(state.nu)):
            raise ValueError("Adam second moments must be nonnegative")

    def apply(self, params, direction, step, optimizer_state=None):
        if optimizer_state is None:
            optimizer_state = self.init(params)
        if self.l2_coefficient:
            direction = jax.tree_util.tree_map(
                lambda p, d: d + self.l2_coefficient * p if p.ndim > 1 else d, params, direction)
        scaled, next_state = self._transform().update(direction, optimizer_state, params)
        eta = self.learning_rate(step) if callable(self.learning_rate) else self.learning_rate
        next_params = jax.tree_util.tree_map(lambda p, d: p - eta * d, params, scaled)
        return UpdateResult(next_params, next_state)
