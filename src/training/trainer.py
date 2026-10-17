"""Shared training orchestration, independent of dataset and method names."""

from typing import Any, NamedTuple


class TrainState(NamedTuple):
    params: Any
    model_state: Any = None
    method_state: Any = None
    optimizer_state: Any = None
    step: Any = 0


class Trainer:
    def __init__(self, task, method, updater):
        self.task = task
        self.method = method
        self.updater = updater

    def step(self, state: TrainState, batch, rng_key=None):
        result = self.method.compute_direction(
            self.task, state.params, batch, state.model_state,
            state.method_state, rng_key,
        )
        updated = self.updater.apply(
            state.params, result.direction, state.step, state.optimizer_state
        )
        next_state = TrainState(
            updated.params, result.model_state, result.method_state,
            updated.optimizer_state, state.step + 1,
        )
        return next_state, result.metrics
