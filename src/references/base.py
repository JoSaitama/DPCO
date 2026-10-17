"""Reference data and pure, method-independent mean construction."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ReferenceResult:
    direction: Any
    has_reference: Any
    selected_count: Any
    per_sample: bool = False


class Reference(Protocol):
    def build(self, gradients: Any, valid_mask: Any, feedback: Mapping[str, Any],
              *, exclude_self: bool = False) -> ReferenceResult:
        """Build one common reference or per-sample self-excluded references."""
        ...


def selected_mean(gradients, selected_mask, *, exclude_self=False):
    """Build means without unit-norm normalization; empty references are flagged."""
    selected = jnp.asarray(selected_mask, dtype=jnp.bool_)
    if selected.ndim != 1:
        raise ValueError("Reference selection mask must have shape [B]")
    count = jnp.sum(selected, dtype=jnp.float32)
    counts = count - selected.astype(jnp.float32) if exclude_self else count

    def build_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim < 1 or leaf.shape[0] != selected.shape[0]:
            raise ValueError("Each gradient leaf must have leading batch axis B")
        shape = (selected.shape[0],) + (1,) * (leaf.ndim - 1)
        selected_leaf = jnp.where(selected.reshape(shape), leaf, 0.)
        total = jnp.sum(selected_leaf, axis=0)
        if exclude_self:
            numerator = total[None, ...] - selected_leaf
            denominator = jnp.maximum(counts, 1.).reshape(shape)
        else:
            numerator = total
            denominator = jnp.maximum(counts, 1.)
        return jax.lax.stop_gradient(numerator / denominator)

    return ReferenceResult(
        jax.tree_util.tree_map(build_leaf, gradients), counts > 0,
        counts, per_sample=exclude_self,
    )
