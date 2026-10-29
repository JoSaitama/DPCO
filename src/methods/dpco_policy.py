"""Opt-in DPCO policies; legacy SGD/DPCO implementations remain untouched."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.methods.base import MethodResult
from src.methods.dtv import DTV
from src.methods.dpco import DPCO
from src.methods.sgd import SGD


POLICY_MODES = ("continuous", "alternating", "half")


class PolicyState(NamedTuple):
    # Number of completed real updates; warmup/observers must discard their copy.
    updates: object


def policy_alpha(mode, state, alternating_first="sgd"):
    if mode not in POLICY_MODES:
        raise ValueError(f"Unknown DPCO policy: {mode}")
    if alternating_first not in ("sgd", "dpco"):
        raise ValueError("alternating_first must be sgd or dpco")
    if mode == "continuous":
        return jnp.asarray(1., jnp.float32)
    if mode == "half":
        return jnp.asarray(.5, jnp.float32)
    # The checkpoint counter is absolute; selecting the first method is a fixed
    # phase offset, never a reset when training resumes or a probe is evaluated.
    return jnp.asarray((state.updates + int(alternating_first == "dpco")) % 2, jnp.float32)


@dataclass(frozen=True)
class DPCOPolicy:
    reference: object
    mode: str = "continuous"
    gradient_backend: str = "audit"
    alternating_first: str = "sgd"

    def __post_init__(self):
        if self.mode not in POLICY_MODES:
            raise ValueError(f"Unknown DPCO policy: {self.mode}")
        if self.alternating_first not in ("sgd", "dpco"):
            raise ValueError("alternating_first must be sgd or dpco")
        if self.mode != "alternating" and self.alternating_first != "sgd":
            raise ValueError("A nondefault first method requires mode=alternating")

    def initial_state(self):
        return PolicyState(jnp.asarray(0, jnp.int32))

    def compute_direction(self, task, params, batch, model_state=None,
                          method_state=None, rng_key=None):
        state = self.initial_state() if method_state is None else method_state
        if not isinstance(state, PolicyState):
            raise ValueError("DPCO policy requires its checkpointed PolicyState")
        dpco = DPCO(self.reference, gradient_backend=self.gradient_backend)
        if self.mode == "continuous":
            result = dpco.compute_direction(task, params, batch, model_state, None, rng_key)
        elif self.mode == "half":
            # (Gs + Gd)/2 is exactly a zero weight on negative alignment. Reuse
            # one weighted VJP with the original valid B, not two full methods.
            result = DTV(self.reference, gradient_backend=self.gradient_backend).compute_direction(
                task, params, batch, model_state, None, rng_key)
        else:
            # Both branches are traced once, but only the selected branch runs.
            # SGD rounds perform no reference/JVP computation at runtime.
            result = jax.lax.cond(policy_alpha(self.mode, state, self.alternating_first) == 1.,
                lambda: dpco.compute_direction(task, params, batch, model_state, None, rng_key),
                lambda: SGD().compute_direction(task, params, batch, model_state, None, rng_key))
        return MethodResult(result.direction, result.model_state,
            PolicyState(state.updates + 1),
            {**result.metrics, "policy_alpha": policy_alpha(self.mode, state, self.alternating_first)})
