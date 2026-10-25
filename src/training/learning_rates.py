"""Learning-rate schedules shared by compiled updates and host-side recording."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class CosineLearningRate:
    peak: float
    minimum: float
    total_steps: int

    def __post_init__(self):
        if not math.isfinite(self.peak) or self.peak <= 0:
            raise ValueError("peak learning rate must be positive and finite")
        if not math.isfinite(self.minimum) or not 0 <= self.minimum <= self.peak:
            raise ValueError("minimum learning rate must be finite and in [0, peak]")
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")

    def __call__(self, step):
        # step is the zero-based index of the update about to run. For T >= 2,
        # the first update uses peak and the last uses minimum. A one-step run
        # uses peak once. Values after the training budget stay at minimum.
        if isinstance(step, Real):
            progress = min(1., max(0., step / max(self.total_steps - 1, 1)))
            return self.minimum + .5 * (self.peak - self.minimum) * (1. + math.cos(math.pi * progress))
        import jax.numpy as jnp
        progress = jnp.clip(step / max(self.total_steps - 1, 1), 0., 1.)
        return self.minimum + .5 * (self.peak - self.minimum) * (1. + jnp.cos(jnp.pi * progress))


@dataclass(frozen=True)
class InverseSqrtLearningRate:
    initial: float

    def __post_init__(self):
        if not math.isfinite(self.initial) or self.initial <= 0:
            raise ValueError("initial learning rate must be positive and finite")

    def __call__(self, step):
        # step counts completed updates, including a restored checkpoint prefix.
        # This predetermined schedule has no warmup, floor or budget dependence.
        if isinstance(step, Real):
            return self.initial / math.sqrt(step + 1.)
        import jax.numpy as jnp
        return self.initial / jnp.sqrt(step + 1.)


def learning_rate_for(training):
    """Constant mode retains a scalar and the original compiled update path."""
    if training.lr_schedule == "constant":
        return training.learning_rate
    if training.lr_schedule == "cosine":
        return CosineLearningRate(training.learning_rate, training.min_learning_rate, training.total_steps)
    if training.lr_schedule == "inverse_sqrt":
        return InverseSqrtLearningRate(training.learning_rate)
    raise ValueError("lr_schedule must be constant, cosine or inverse_sqrt")
