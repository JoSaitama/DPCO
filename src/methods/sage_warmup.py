"""Optional SGD prefix followed by the unchanged full-reference SAGE method."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.methods.base import MethodResult
from src.methods.sage import SAGE
from src.methods.sage_policy import PolicyState
from src.methods.sgd import SGD


@dataclass(frozen=True)
class SAGEWarmup:
    reference: object
    warmup_steps: int
    gradient_backend: str = "audit"
    mode: str = "warmup"
    alternating_first: str = "sgd"

    def initial_state(self):
        return PolicyState(jnp.asarray(0, jnp.int32))

    def alpha(self, state):
        # This checkpointed counter counts real updates only. Compilation probes
        # discard their candidate state, and resume never restarts the prefix.
        return jnp.asarray(state.updates >= self.warmup_steps, jnp.float32)

    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        state = self.initial_state() if method_state is None else method_state
        if not isinstance(state, PolicyState):
            raise ValueError("SAGE warmup requires its checkpointed PolicyState")
        alpha = self.alpha(state)
        # Only the selected branch executes. SGD-prefix updates do not perform
        # reference construction or alignment scoring at runtime.
        result = jax.lax.cond(alpha == 1.,
            lambda: SAGE(self.reference, gradient_backend=self.gradient_backend).compute_direction(
                task, params, batch, model_state, None, rng_key),
            lambda: SGD().compute_direction(task, params, batch, model_state, None, rng_key))
        return MethodResult(result.direction, result.model_state, PolicyState(state.updates + 1),
                            {**result.metrics, "policy_alpha": alpha})
