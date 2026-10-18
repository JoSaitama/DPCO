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
    return SGDUpdate(learning_rate_for(training), l2_coefficient=training.l2_coefficient,
                     momentum=training.momentum, nesterov=training.nesterov)
