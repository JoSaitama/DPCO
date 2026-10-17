"""Parameter updates consume a complete direction; never divide by B again."""

from dataclasses import dataclass
import math
from typing import Any, Callable, NamedTuple, Protocol

import jax


class UpdateResult(NamedTuple):
    params: Any
    optimizer_state: Any


class ParameterUpdater(Protocol):
    def apply(self, params, direction, step, optimizer_state=None) -> UpdateResult:
        ...


@dataclass(frozen=True)
class SGDUpdate:
    learning_rate: float | Callable

    def __post_init__(self):
        if not callable(self.learning_rate):
            if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
                raise ValueError("learning_rate must be positive and finite")

    def apply(self, params, direction, step, optimizer_state=None):
        eta = self.learning_rate(step) if callable(self.learning_rate) else self.learning_rate
        next_params = jax.tree_util.tree_map(lambda p, d: p - eta * d, params, direction)
        return UpdateResult(next_params, optimizer_state)
