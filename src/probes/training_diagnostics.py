"""Read-only checks attached to REAL updates, with independent scalar-loss AD.

The original compiled training step is never replaced or instrumented. The
observer receives its before/after states and the exact realized input batch.
No dense [batch, parameter] Jacobian, data iteration, or state commit occurs here.
"""

import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.training.metrics import tree_dot


class IndependentDirections(NamedTuple):
    losses: object
    valid: object
    errors: object
    selected_count: object
    reference: object
    scores: object
    weights: object
    sgd: object
    dpco: object
    model_state: object


def independent_directions(task, reference_name, params, batch, model_state):
    """Differentiate scalar losses rather than reuse the method's VJP helper.

    Prediction selection, reference and signs are frozen at the input state. Every
    evaluation uses the entire logical batch, including BatchNorm coupling.
    Small offline tests additionally compare these results to a dense oracle;
    the scalar-loss route alone is not a proof of every AD backend's accuracy.
    """
    result = task.evaluate(params, batch, model_state)
    valid = result.valid_mask
    errors = jax.lax.stop_gradient(valid & result.feedback["error_mask"])
    selected = valid if reference_name == "loss" else errors
    if reference_name == "correct":
        selected = jax.lax.stop_gradient(valid & ~errors)
    elif reference_name not in ("loss", "error"):
        raise ValueError("Unknown diagnostic reference")
    count = jnp.sum(valid)
    selected_count = jnp.sum(selected)

    def mean_loss(p, mask, denominator):
        losses = task.evaluate(p, batch, model_state).losses
        return jnp.sum(jnp.where(mask, losses, 0.)) / jnp.maximum(denominator, 1)

    reference = jax.grad(lambda p: mean_loss(p, selected, selected_count))(params)
    reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
    scores = jax.jvp(lambda p: task.evaluate(p, batch, model_state).losses,
                     (params,), (reference,))[1]
    weights = jax.lax.stop_gradient(jnp.where(valid,
        jnp.where((scores < 0) & (selected_count > 0), -1., 1.), 0.))
    gd = jax.grad(lambda p: mean_loss(p, valid, count))(params)

    def weighted_loss(p):
        losses = task.evaluate(p, batch, model_state).losses
        return jnp.sum(jnp.where(valid, weights * losses, 0.)) / count

    gs = jax.grad(weighted_loss)(params)
    return IndependentDirections(result.losses, valid, errors, selected_count, reference, scores, weights,
                                 gd, gs, result.model_state)


def _difference(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.asarray(x, dtype=jnp.float32) ** 2)
                        for x in jax.tree_util.tree_leaves(tree)))


def _maximum_error(a, b):
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        return jnp.asarray(jnp.inf)
    leaves = list(zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    if not leaves:
        return jnp.asarray(0.)
    if any(x.shape != y.shape for x, y in leaves):
        return jnp.asarray(jnp.inf)
    return jnp.max(jnp.stack([jnp.max(jnp.abs(x - y)) for x, y in leaves]))


def _candidate(params, direction, eta):
    return jax.tree_util.tree_map(lambda p, g: p - eta * g, params, direction)


class TrainingDiagnostics:
    """Observe one trajectory while pairing SGD/DPCO at each frozen state."""

    def __init__(self, task, method, config, *, reference_name, trajectory_method,
                 audit_batches=(), objective_mode="train_batch_bn; frozen_before_buffers"):
        if reference_name not in ("loss", "error", "correct") or trajectory_method not in ("sgd", "dpco", "dpco_policy"):
            raise ValueError("Diagnostics require a loss/error/correct reference and an SGD/DPCO trajectory")
        if config.transfer and len(audit_batches) != config.audit_batches:
            raise ValueError("Transfer diagnostics require the configured fixed training audit bank")
        self.task, self.config = task, config
        self.method = method
        self.reference_name, self.trajectory_method = reference_name, trajectory_method
        self.warmup_steps = getattr(method, "warmup_steps", 0)
        self.audit_batches, self.objective_mode = tuple(audit_batches), objective_mode
        self._forward = jax.jit(task.evaluate)

        def mean_gradient(p, b, s, weights, denominator):
            def scalar_loss(p):
                losses = task.evaluate(p, b, s).losses
                return jnp.sum(jnp.where(b.valid_mask, weights * losses, 0.)) / jnp.maximum(denominator, 1.)
            return jax.grad(scalar_loss)(p)

        # The production observer stages scalar AD sequentially. Returning only
        # gradient trees between kernels prevents multiple full-network backward
        # activation workspaces from being live at once at B=256. The composite
        # independent_directions function above remains a small offline oracle.
        self._mean_gradient = jax.jit(mean_gradient)
        self._scores = jax.jit(lambda p, b, s, r: jax.jvp(
            lambda p: task.evaluate(p, b, s).losses, (p,), (r,))[1])

        def objective(p, b, s, mask):
            losses = task.evaluate(p, b, s).losses
            return jnp.sum(jnp.where(mask, losses, 0.)) / jnp.maximum(jnp.sum(mask), 1)

        self._objective = jax.jit(objective)
        self._objective_gradient = jax.jit(jax.value_and_grad(objective))
        # Replaying is read-only and localizes an independent-direction mismatch.
        # It is NOT described as the raw direction returned by the training kernel.
        self._replay = None if not config.update_check else jax.jit(
            lambda p, b, s, m, key: method.compute_direction(task, p, b, s, m, key))
        self._summary = jax.jit(self._summarize)
        self._update_errors = jax.jit(self._check_update)
        self._projection = jax.jit(tree_dot)
        self._candidate = jax.jit(_candidate)
        self._correction = jax.jit(_difference)

    def _independent(self, params, batch, model_state):
        result = self._forward(params, batch, model_state)
        valid = result.valid_mask
        errors = jax.lax.stop_gradient(valid & result.feedback["error_mask"])
        selected = valid if self.reference_name == "loss" else errors
        if self.reference_name == "correct":
            selected = jax.lax.stop_gradient(valid & ~errors)
        count = jnp.sum(valid, dtype=jnp.float32)
        selected_count = jnp.sum(selected, dtype=jnp.float32)
        reference = self._mean_gradient(params, batch, model_state,
            selected.astype(result.losses.dtype), selected_count)
        reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
        jax.block_until_ready(reference)
        scores = self._scores(params, batch, model_state, reference)
        jax.block_until_ready(scores)
        weights = jax.lax.stop_gradient(jnp.where(valid,
            jnp.where((scores < 0) & (selected_count > 0), -1., 1.), 0.))
        gd = self._mean_gradient(params, batch, model_state, valid.astype(result.losses.dtype), count)
        jax.block_until_ready(gd)
        gs = self._mean_gradient(params, batch, model_state, weights.astype(result.losses.dtype), count)
        jax.block_until_ready(gs)
        return IndependentDirections(result.losses, valid, errors, selected_count,
            reference, scores, weights, gd, gs, result.model_state)

    def metadata(self):
        metadata = {"schema_version": 1, "state_step": "completed updates BEFORE inspected update",
                "update_step": "state step + 1; matches the training CSV row",
                "checks": "independent scalar-loss AD, real candidate at scale 1, training-audit transfer",
                "candidate_policy": "scale 1 uses actual after.params for the trajectory method; others use independent directions",
                "objective_mode": self.objective_mode,
                "audit_seed": self.config.audit_seed,
                "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "audit_augmentation": "audit_seed/sample-index/0xA0D17; fixed training transform",
                "audit_bank": [{"batch_id": b.identity, "sample_indices": list(b.indices)} for b in self.audit_batches],
                "overlap_policy": "fixed bank, overlap counted, no target reselection or training exclusion",
                "scope": "finite training-audit targets; not a population or convergence certificate",
                "sample_energy": "not measured; no per-example energy loop"}
        if self.trajectory_method == "dpco_policy" or self.warmup_steps:
            metadata.update(policy_mode=self.method.mode,
                alternating_first=self.method.alternating_first,
                policy_comparison="diagnostic_policy compares the active policy to SGD; legacy paired/transfer compare FULL DPCO to SGD",
                candidate_policy="full pair uses actual endpoint only for alpha=0 or 1; policy pair uses actual endpoint at scale 1")
            if self.warmup_steps:
                metadata["sgd_prefix_updates"] = self.warmup_steps
        return metadata

    @staticmethod
    def _summarize(stats):
        correction = _difference(stats.dpco, stats.sgd)
        delta = tree_dot(stats.reference, correction)
        formula = jnp.sum(jnp.where(stats.valid, (stats.weights - 1.) * stats.scores, 0.)) / jnp.sum(stats.valid)
        return {"valid_batch_size": jnp.sum(stats.valid), "error_count": jnp.sum(stats.errors),
                "flip_count": jnp.sum(stats.valid & (stats.weights < 0)),
                "reference_valid": stats.selected_count > 0,
                "reference_delta": delta, "identity_residual": delta - formula,
                "score_min_abs": jnp.min(jnp.where(stats.valid, jnp.abs(stats.scores), jnp.inf)),
                "norm_sgd_sq": tree_dot(stats.sgd, stats.sgd),
                "norm_dpco_sq": tree_dot(stats.dpco, stats.dpco)}

    @staticmethod
    def _check_update(before, after, replay, expected_direction, expected_state, eta):
        direction_error = _norm(_difference(replay.direction, expected_direction))
        expected_params = _candidate(before.params, expected_direction, eta)
        replayed_params = _candidate(before.params, replay.direction, eta)
        parameter_error = _norm(_difference(after.params, expected_params))
        update_norm = _norm(_difference(expected_params, before.params))
        values = {"direction_l2_error": direction_error,
                  "direction_relative_error": direction_error / jnp.maximum(_norm(expected_direction), 1e-30),
                  "parameter_l2_error": parameter_error,
                  "parameter_relative_error": parameter_error / jnp.maximum(update_norm, 1e-30),
                  "replayed_parameter_l2_error": _norm(_difference(after.params, replayed_params)),
                  "model_state_max_abs_error": _maximum_error(after.model_state, expected_state),
                  "replayed_model_state_max_abs_error": _maximum_error(replay.model_state, expected_state)}
        return {**values, "finite": jnp.all(jnp.stack([jnp.isfinite(v) for v in values.values()]))}

    def run(self, before, after, batch, *, learning_rate, rng_key, context, cumulative_eta_before):
        """Return small scalar records; never publish model state or consume data."""
        step, update_step = int(before.step), int(after.step)
        if update_step != step + 1:
            raise ValueError("Diagnostics require exactly one actual completed update")
        alpha = None
        if self.trajectory_method == "dpco_policy" or self.warmup_steps:
            from src.methods.dpco_policy import PolicyState, policy_alpha
            if (not isinstance(before.method_state, PolicyState)
                    or not isinstance(after.method_state, PolicyState)
                    or int(before.method_state.updates) != step
                    or int(after.method_state.updates) != update_step):
                raise ValueError("Diagnostic policy phase does not match the real update")
            alpha = float(jax.device_get(self.method.alpha(before.method_state) if self.warmup_steps else
                policy_alpha(self.method.mode, before.method_state, self.method.alternating_first)))
        source_id = context.get("batch_id", f"training_update_{update_step}")
        base = {"step": step, "update_step": update_step, "source_batch_id": source_id,
                "trajectory_method": self.trajectory_method, "reference": self.reference_name}
        rows = [("diagnostic_context", {**base, "learning_rate": learning_rate,
                 "cumulative_eta_before": cumulative_eta_before, "objective_mode": self.objective_mode,
                 "source_context_json": json.dumps(context, sort_keys=True)})]
        stats = self._independent(before.params, batch, before.model_state)
        summary = self._summary(stats)
        correction = self._correction(stats.dpco, stats.sgd)
        common = {**base, "objective_mode": self.objective_mode, "learning_rate": learning_rate,
                  "reference_delta": summary["reference_delta"],
                  "norm_sgd_sq": summary["norm_sgd_sq"], "norm_dpco_sq": summary["norm_dpco_sq"]}
        if self.config.update_check:
            replay = self._replay(before.params, batch, before.model_state, before.method_state, rng_key)
            expected = stats.dpco if self.trajectory_method == "dpco" else stats.sgd
            if alpha is not None:
                expected = self._policy_direction(stats, alpha)
            errors = self._update_errors(before, after, replay, expected, stats.model_state, learning_rate)
            rows.append(("diagnostic_update", {**base, "learning_rate": learning_rate, **summary, **errors}))

        if self.config.paired_loss:
            objectives = [("full_batch_ce", stats.valid, self._projection(stats.sgd, correction))]
            if self.reference_name == "error" and bool(jax.device_get(jnp.any(stats.errors))):
                objectives.insert(0, ("frozen_error_ce", stats.errors, summary["reference_delta"]))
            if self.reference_name == "correct":
                correct = jax.lax.stop_gradient(stats.valid & ~stats.errors)
                if bool(jax.device_get(jnp.any(correct))):
                    objectives.insert(0, ("frozen_correct_ce", correct, summary["reference_delta"]))
            for name, mask, gain in objectives:
                initial = jnp.sum(jnp.where(mask, stats.losses, 0.)) / jnp.sum(mask)
                for scale in self.config.lr_scales:
                    row = self._paired_row(before, after, batch, mask, stats, scale, learning_rate, initial, gain,
                                           actual_alpha=alpha)
                    rows.append(("diagnostic_paired", {**common, "objective": name,
                                 "selected_count": jnp.sum(mask), **row}))
                    if alpha is not None:
                        rows.append(("diagnostic_policy", {**base, "policy_mode": self.method.mode,
                            "policy_alpha": alpha, "objective": name, "objective_mode": self.objective_mode,
                            "target_batch_id": source_id,
                            **self._policy_row(before, after, batch, mask, stats, scale,
                                               learning_rate, initial, gain, alpha)}))

        if self.config.transfer:
            source_indices = set(context.get("sample_indices", ()))
            for item in self.audit_batches:
                mask = item.batch.valid_mask
                initial, gradient = self._objective_gradient(before.params, item.batch, before.model_state, mask)
                gain = self._projection(gradient, correction)
                overlap = len(source_indices.intersection(item.indices)) if "sample_indices" in context else None
                for scale in self.config.lr_scales:
                    row = self._paired_row(before, after, item.batch, mask, stats, scale, learning_rate, initial, gain,
                                           actual_alpha=alpha)
                    rows.append(("diagnostic_transfer", {**common, "target_batch_id": item.identity,
                                 "overlap_sample_count": overlap, "selected_count": jnp.sum(mask),
                                 "transfer_difference": gain - summary["reference_delta"], **row}))
                    if alpha is not None:
                        rows.append(("diagnostic_policy", {**base, "policy_mode": self.method.mode,
                            "policy_alpha": alpha, "objective": "training_audit_ce",
                            "objective_mode": self.objective_mode, "target_batch_id": item.identity,
                            **self._policy_row(before, after, item.batch, mask, stats, scale,
                                               learning_rate, initial, gain, alpha)}))
        # This transfer contains scalars/strings only; parameter arrays stay on device.
        host = jax.device_get([row for _, row in rows])
        return [(stream, {k: np.asarray(v).item() if isinstance(v, (np.ndarray, np.generic)) else v
                          for k, v in row.items()}) for (stream, _), row in zip(rows, host)]

    @staticmethod
    def _policy_direction(stats, alpha):
        if alpha == 0.:
            return stats.sgd
        if alpha == 1.:
            return stats.dpco
        return jax.tree_util.tree_map(lambda d, s: .5 * (d + s), stats.sgd, stats.dpco)

    def _policy_row(self, before, after, batch, mask, stats, scale, learning_rate, initial, gain, alpha):
        """Actual-policy evidence is separate from the full-DPCO counterfactual."""
        eta = learning_rate * scale
        pd = self._candidate(before.params, stats.sgd, eta)
        pp = (after.params if scale == 1. else self._candidate(
            before.params, self._policy_direction(stats, alpha), eta))
        jd = self._objective(pd, batch, before.model_state, mask)
        jp = self._objective(pp, batch, before.model_state, mask)
        difference, actual_gain = jp - jd, alpha * gain
        return dict(lr_scale=scale, probe_lr=eta, j_initial=initial, j_sgd=jd, j_policy=jp,
            difference=difference, objective_gain=actual_gain,
            first_order_residual=difference / eta + actual_gain if eta > 0 else None,
            actual_update_in_pair=scale == 1.,
            finite=jnp.all(jnp.isfinite(jnp.stack([initial, jd, jp, actual_gain]))))

    def _paired_row(self, before, after, batch, mask, stats, scale, learning_rate, initial, gain,
                    *, actual_alpha=None):
        eta = learning_rate * scale
        pd = self._candidate(before.params, stats.sgd, eta)
        ps = self._candidate(before.params, stats.dpco, eta)
        # The base-rate comparison includes the ORIGINAL compiled step's result.
        # If implementation differs from independent AD, H and its residual expose
        # that mismatch instead of silently testing two reconstructed updates.
        actual = scale == 1. and (actual_alpha is None or actual_alpha in (0., 1.))
        if actual:
            if actual_alpha == 1. or (actual_alpha is None and self.trajectory_method == "dpco"):
                ps = after.params
            else:
                pd = after.params
        jd = self._objective(pd, batch, before.model_state, mask)
        js = self._objective(ps, batch, before.model_state, mask)
        difference = js - jd
        ratio = difference / eta if eta > 0 else None
        return {"lr_scale": scale, "probe_lr": eta, "j_initial": initial,
                "j_sgd": jd, "j_dpco": js, "difference": difference,
                "difference_over_lr": ratio, "objective_gain": gain,
                "first_order_residual": ratio + gain if eta > 0 else None,
                "actual_update_in_pair": actual,
                "finite": jnp.all(jnp.isfinite(jnp.stack([initial, jd, js, gain])))}
