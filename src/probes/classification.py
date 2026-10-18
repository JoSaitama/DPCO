"""Read-only common-reference probes at one frozen model/batch state.

Call with a deterministic probe task and explicit mode. Returned model state is
discarded, masks/inputs stay fixed, and no training RNG or iterator is consumed.
These diagnostics do not establish global smoothness or convergence rates.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.methods.base import aggregate
from src.signals.gradient_alignment import gradient_alignment
from src.training.metrics import tree_dot


@dataclass(frozen=True)
class ProbeBatch:
    identity: str
    batch: object


class ClassificationProbes:
    def __init__(self, task, reference, batches, *, reference_name, objective_mode,
                 audit_objective=None, audit_name=None):
        self.task, self.reference = task, reference
        self.batches = tuple(batches)
        self.reference_name, self.objective_mode = reference_name, objective_mode
        self.audit_objective, self.audit_name = audit_objective, audit_name
        if not objective_mode or not self.batches:
            raise ValueError("Probes require fixed named batches and an explicit objective mode")
        if audit_objective is not None and not audit_name:
            raise ValueError("A fixed audit objective requires a name")

    def run(self, state, *, step, learning_rate, config, cumulative):
        """Return small host-scalar rows only, never per-example gradient tensors."""
        rows = []
        if config.theorem3 and self.audit_objective is None:
            raise ValueError("Theorem 3 diagnostics require an explicit fixed audit objective")
        if config.theorem3:
            audit_loss, audit_gradient = jax.value_and_grad(self.audit_objective)(state.params)
        for probe_id, item in enumerate(self.batches[:config.batches_per_node]):
            batch, params, model_state = item.batch, state.params, state.model_state

            def losses_fn(p):
                result = self.task.evaluate(p, batch, model_state)
                return result.losses, (result.losses, result.valid_mask, result.feedback)

            gradients, (losses, valid, feedback) = jax.jacrev(losses_fn, has_aux=True)(params)
            reference = self.reference.build(gradients, valid, feedback, exclude_self=False)
            scores = gradient_alignment(gradients, reference, valid)
            weights = jnp.where(valid, jnp.where((scores < 0) & reference.has_reference, -1., 1.), 0.)
            gd = aggregate(gradients, jnp.ones_like(weights), valid)
            gs = aggregate(gradients, weights, valid)
            correction = jax.tree_util.tree_map(lambda s, d: s - d, gs, gd)
            count = jnp.sum(valid)
            delta = tree_dot(reference.direction, correction)
            delta_formula = 2 * jnp.sum(jnp.where(valid & (scores < 0), -scores, 0.)) / count
            d2, s2 = tree_dot(gd, gd), tree_dot(gs, gs)
            base = {"step": step, "phase": "node", "probe_id": item.identity}
            if config.theorem1:
                rows.append(("theorem1", {**base, "valid_batch_size": count,
                    "error_count": jnp.sum(valid & feedback["error_mask"]),
                    "negative_count": jnp.sum(valid & (scores < 0)),
                    "flip_count": jnp.sum(valid & (weights < 0)),
                    "reference_valid": reference.has_reference, "delta": delta,
                    "eta_delta": learning_rate * delta,
                    "reference_projection_sgd": tree_dot(reference.direction, gd),
                    "reference_projection_sage": tree_dot(reference.direction, gs),
                    "delta_formula": delta_formula, "identity_residual": delta - delta_formula,
                    "norm_sgd_sq": d2, "norm_sage_sq": s2}))
            if config.theorem2:
                errors = jax.lax.stop_gradient(valid & feedback["error_mask"])
                objectives = [("full_batch_ce", valid, tree_dot(gd, correction))]
                if self.reference_name == "error" and bool(jnp.any(errors)):
                    objectives.insert(0, ("frozen_error_ce", errors, delta))
                for objective_name, mask, objective_delta in objectives:
                    selected_count = jnp.sum(mask)

                    def objective(p):
                        result = self.task.evaluate(p, batch, model_state)
                        return jnp.sum(jnp.where(mask, result.losses, 0.)) / selected_count

                    initial = jnp.sum(jnp.where(mask, losses, 0.)) / selected_count
                    for scale in config.lr_scales:
                        eta = learning_rate * scale
                        pd = jax.tree_util.tree_map(lambda p, d: p - eta * d, params, gd)
                        ps = jax.tree_util.tree_map(lambda p, s: p - eta * s, params, gs)
                        jd, js = objective(pd), objective(ps)
                        difference = js - jd
                        # Cosine may reach zero. A zero-step comparison is valid,
                        # but division by its step size has no defined value.
                        ratio = difference / eta if eta > 0 else None
                        rows.append(("theorem2", {**base, "objective": objective_name,
                            "objective_mode": self.objective_mode, "batch_id": item.identity,
                            "reference_name": self.reference_name, "selected_count": selected_count,
                            "learning_rate": learning_rate, "lr_scale": scale, "probe_lr": eta,
                            "j_initial": initial, "j_sgd": jd, "j_sage": js,
                            "difference": difference, "difference_over_lr": ratio,
                            "delta": objective_delta, "first_order_residual": ratio + objective_delta if eta > 0 else None,
                            "norm_sgd_sq": d2, "norm_sage_sq": s2,
                            "finite": jnp.all(jnp.isfinite(jnp.array([initial, jd, js, objective_delta]))),
                            "strict_gain_eligible": (objective_delta > 0) & (eta > 0)}))
            if config.theorem3:
                target_gain = tree_dot(audit_gradient, correction)
                energy = sum(jnp.sum(jnp.where(valid.reshape((valid.shape[0],) + (1,) * (leaf.ndim - 1)), leaf ** 2, 0.))
                             for leaf in jax.tree_util.tree_leaves(gradients)) / count
                rows.append(("theorem3", {**base, "objective": self.audit_name,
                    "objective_mode": self.objective_mode, "audit_loss": audit_loss,
                    "audit_gradient_norm_sq": tree_dot(audit_gradient, audit_gradient),
                    "delta": delta, "target_gain": target_gain,
                    "transfer_difference": target_gain - delta, "sample_energy": energy,
                    "norm_sgd_sq": d2, "norm_sage_sq": s2, **cumulative}))
        # One tree transfer per node; do not synchronize once per field.
        values = jax.device_get([row for _, row in rows])
        return [(stream, {key: np.asarray(value).item() if isinstance(value, (np.ndarray, np.generic)) else value
                           for key, value in row.items()}) for (stream, _), row in zip(rows, values)]
