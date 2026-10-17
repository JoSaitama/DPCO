"""Task objectives and feedback, independent of contribution correction."""

from typing import Any, Mapping, NamedTuple, Protocol


class TaskResult(NamedTuple):
    losses: Any
    valid_mask: Any
    model_state: Any
    feedback: Mapping[str, Any]


class Task(Protocol):
    def evaluate(self, params: Any, batch: Any, model_state: Any) -> TaskResult:
        """Return per-contribution losses [B], mask [B], state and feedback.

        Must be a pure JAX-differentiable computation. The caller commits the
        returned model state once. Invalid padded inputs must be made safe.
        Error feedback is a discrete signal, not an error-loss derivative.
        """
        ...
