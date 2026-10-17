"""Raw gradient inner products; common and LOO references are distinct."""

import jax
import jax.numpy as jnp


def gradient_alignment(gradients, reference, valid_mask):
    leaves, structure = jax.tree_util.tree_flatten(gradients)
    refs, ref_structure = jax.tree_util.tree_flatten(reference.direction)
    if structure != ref_structure or not leaves:
        raise ValueError("Gradient/reference PyTrees must match and be nonempty")
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    scores = jnp.zeros(valid.shape, dtype=jnp.float32)
    for leaf, ref in zip(leaves, refs):
        leaf = jnp.asarray(leaf)
        expected = leaf.shape if reference.per_sample else leaf.shape[1:]
        if leaf.ndim < 1 or leaf.shape[0] != valid.shape[0] or ref.shape != expected:
            raise ValueError("Invalid common or per-sample reference shape")
        mask_shape = (valid.shape[0],) + (1,) * (leaf.ndim - 1)
        leaf = jnp.where(valid.reshape(mask_shape), leaf, 0.)
        scores = scores + jnp.sum(leaf * ref, axis=tuple(range(1, leaf.ndim)))
    return jax.lax.stop_gradient(
        jnp.where(valid & reference.has_reference, scores, 0.)
    )
