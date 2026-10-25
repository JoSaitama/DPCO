"""A genuinely fixed train-mode, global-logical-batch proxy objective."""

import hashlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np


def tree_fingerprint(tree):
    """Hash values, shape, dtype and grouping; no parameter snapshot is saved."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


class FixedTrainingBank:
    """F_A(theta)=mean_k J_Ak(theta;s*); candidate output buffers are discarded.

    s* is reconstructed from the SAME initial model seed/version on resume, not
    taken from a resumed current model state. Fingerprint mismatches fail closed.
    Each A_k stays a complete global logical batch, even when arrays are sharded.
    """
    def __init__(self, batches, buffers, *, signature, buffer_origin="initial_model_state"):
        if not batches:
            raise ValueError("Fixed target requires at least one full training bank batch")
        self.batches = tuple(replace(item, batch=jax.tree_util.tree_map(jnp.asarray, item.batch)) for item in batches)
        self.buffers = jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), buffers)
        for item in batches:
            if not bool(np.any(jax.device_get(item.batch.valid_mask))):
                raise ValueError("A fixed target batch cannot be all padding")
        bank = [dict(batch_id=b.identity, sample_indices=list(b.indices),
            fingerprint=tree_fingerprint(b.batch)) for b in self.batches]
        buffers_hash = tree_fingerprint(self.buffers)
        identity = dict(version=1, bank=bank, buffer_fingerprint=buffers_hash,
            model_signature=signature, bn_mode="train_mode_global_logical_batch",
            reduction="uniform mean of valid-sample means of whole bank batches")
        self.identity = "fixed_train_bank_" + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.info = dict(target_id=self.identity, **identity, buffer_origin=buffer_origin,
            buffer_recovery="reconstruct initial model buffers from recorded model/data seed and verify fingerprints",
            output_buffers="discarded; never committed to training or this target",
            scope="fixed training proxy, neither full ImageNet nor validation")

    def evaluate(self, params, ad, call, *, gradient=True):
        value, accumulated = jnp.asarray(0., jnp.float32), None
        count = len(self.batches)
        for item in self.batches:
            mask = item.batch.valid_mask
            if gradient:
                current, direction = call("bank_value_grad", ad._objective_gradient,
                    params, item.batch, self.buffers, mask)
                scaled = jax.tree_util.tree_map(lambda g: g/count, direction)
                accumulated = scaled if accumulated is None else jax.tree_util.tree_map(
                    lambda a, g: a+g, accumulated, scaled)
                jax.block_until_ready(accumulated)
                del direction, scaled
            else:
                current = call("bank_value", ad._objective, params, item.batch, self.buffers, mask)
            value = value + current/count
        jax.block_until_ready(value)
        return value, accumulated
