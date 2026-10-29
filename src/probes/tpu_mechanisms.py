"""Opt-in mechanism observers, not a second optimizer or CPU replacement.

Reuse staged full-batch scalar AD/JVP from TrainingDiagnostics. No [B,P]
Jacobian, sample-energy loop, local/microbatch BN or training-state commit.
"""

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.config import DiagnosticConfig
from src.probes.fixed_training_bank import FixedTrainingBank
from src.probes.training_diagnostics import (TrainingDiagnostics, IndependentDirections,
    _difference, _norm, _maximum_error)
from src.training.metrics import tree_dot


BASE = ("step", "update_step", "trajectory_method", "trajectory_reference", "source_batch_id", "target_id")
SCHEMAS = {
    "mechanism_context": (*BASE, "learning_rate", "cumulative_eta_before", "source_context_json"),
    "mechanism_update": (*BASE, "direction_l2_error", "direction_relative_error", "direction_norm",
        "parameter_l2_error", "parameter_relative_error", "actual_displacement_norm", "replayed_parameter_l2_error",
        "model_state_max_abs_error", "replayed_model_state_max_abs_error", "parameter_dtype", "compute_dtype",
        "matmul_precision", "absolute_tolerance", "relative_tolerance", "tolerance_supported", "reliable", "finite"),
    "mechanism_t1": (*BASE, "reference", "reference_valid", "error_count", "flip_count", "negative_count",
        "reference_delta", "delta_formula", "identity_residual", "score_min_abs", "norm_sgd_sq", "norm_dpco_sq"),
    "mechanism_transfer": (*BASE, "reference", "scope", "reference_delta", "target_lambda", "mismatch",
        "transfer_residual", "h_norm_sq"),
    "mechanism_pairs": (*BASE, "reference", "objective", "buffer_semantics", "candidate_kind", "actual_endpoint",
        "lr_kind", "lr_scale", "probe_lr", "dpco_probe_lr", "normalization", "defined", "undefined_reason",
        "reference_valid", "j_initial", "j_sgd", "j_dpco", "difference", "objective_gain", "effective_objective_gain",
        "difference_over_lr", "first_order_residual", "endpoint_loss_error", "loss_resolution",
        "numeric_checked", "update_reliable", "signal_reliable", "finite"),
    "mechanism_moments": (*BASE, "reference", "estimator", "bank_batch_count", "probability_sum", "standard_error",
        "mean_h_dot_dpco", "mean_h_dot_sgd", "mean_delta", "mean_lambda", "mean_mismatch",
        "mean_norm_dpco_sq", "mean_norm_sgd_sq", "mean_gd_h_l2_error", "mean_gd_h_relative_error", "h_norm_sq"),
    "mechanism_windows": (*BASE, "window_start", "window_end", "window_count", "window_complete", "fa_before",
        "fa_after", "P", "R", "displacement_norm_sq", "cumulative_P", "cumulative_R", "endpoint_loss_difference",
        "identity_residual", "identity_tolerance", "identity_pass", "learning_rate", "cumulative_eta", "cumulative_eta_sq"),
    "mechanism_timing": ("step", "event", "kernel", "duration_s"),
}


def scalar(value):
    return float(jax.device_get(value))


def relative(error, norm):
    return error/norm if norm != 0 else None


class TPUMechanisms:
    """Read only the true before/after update, with frozen target asset identity."""
    ATOL = 2e-6
    RTOL = 1e-4

    def __init__(self, task, method, config, *, reference_name, trajectory_method,
                 bank_batches=(), fixed_buffers=None, signature="explicit_task_v1", compute_dtype="float32"):
        setup_start = time.perf_counter()
        if trajectory_method not in ("sgd", "dpco") or reference_name not in ("error", "loss", "correct"):
            raise ValueError("Extended observers cover original SGD/DPCO E/L/R, not policy updates")
        if getattr(method, "gradient_backend", "jvp") != "jvp":
            raise ValueError("Extended TPU observers require the JVP method backend; no dense replay")
        self.config, self.method = config, method
        self.reference_name, self.trajectory_method = reference_name, trajectory_method
        self.trajectory_label = "sgd" if trajectory_method == "sgd" else {
            "error": "dpco_e", "loss": "dpco_l", "correct": "dpco_r"}[reference_name]
        # Existing E/L runs retain exactly their two-reference observer workload.
        # Only an explicitly requested R trajectory adds correct-set candidates.
        self.reference_names = ("loss", "error", "correct") if reference_name == "correct" else ("loss", "error")
        self.compute_dtype = compute_dtype
        self.ad = TrainingDiagnostics(task, method, DiagnosticConfig(transfer=False, update_check=config.update_check),
            reference_name=reference_name, trajectory_method=trajectory_method)
        self.target = None
        if config.needs_bank:
            if len(bank_batches) != config.bank_batches:
                raise ValueError("Fixed bank size must match mechanism.bank_batches")
            self.target = FixedTrainingBank(bank_batches, fixed_buffers, signature=signature)
        self._cache, self._window_state, self._timings = {}, {}, []
        self._loss_cache = {}
        self._step = 0
        self._dot = jax.jit(tree_dot)
        self._norm = jax.jit(_norm)
        self._difference = jax.jit(_difference)
        self._state_error = jax.jit(_maximum_error)
        self._mean_add = jax.jit(lambda a, b, scale: jax.tree_util.tree_map(lambda x, y: x+y*scale, a, b))
        self.setup_duration_s = time.perf_counter()-setup_start
        self._setup_reported = False

    def _call(self, name, fn, *args):
        """Separate compilation from execution; synchronize stage workspaces."""
        leaves, structure = jax.tree_util.tree_flatten(args)
        signature = (name, str(structure), tuple((getattr(v, "shape", ()), str(getattr(v, "dtype", type(v))),
            str(getattr(v, "sharding", None))) for v in leaves))
        if signature not in self._cache:
            start = time.perf_counter()
            self._cache[signature] = fn.lower(*args).compile()
            self._timings.append(dict(step=self._step, event="compile", kernel=name, duration_s=time.perf_counter()-start))
        result = self._cache[signature](*args)
        jax.block_until_ready(result)
        return result

    def metadata(self):
        result = dict(schema_version=1, config_scope="mechanism", observer="read-only actual update; no training modification",
            trajectory_method=self.trajectory_label,
            state_step="completed updates BEFORE the inspected update", update_step="state step+1",
            target=None if self.target is None else self.target.info,
            source_semantics="actual source images/augmentation/mask, train-mode global BN, actual before buffers",
            bank_semantics="fixed whole batches/augmentation/masks/train-global-BN/input buffers; output buffers discarded",
            candidate_semantics="independent_pair and actual_pair are separate; actual endpoint only for matching executed method/reference/LR",
            norm_matching="diagnostic only; no epsilon; effective gain is separate from original Delta",
            tolerances=dict(absolute=self.ATOL, relative=self.RTOL, loss_resolution="ATOL*max(1,|J|)",
                supported="float32 provisional CPU-tested thresholds; bfloat16 not qualified by these tests",
                signal_rule="update reliable AND |independent H| above loss resolution AND endpoint contamination <10% of both H and eta*gain"),
            compute_dtype=self.compute_dtype, matmul_precision=str(jax.config.jax_default_matmul_precision),
            setup_duration_s=self.setup_duration_s,
            enable_x64=bool(jax.config.jax_enable_x64),
            sample_energy="not computed; no per-example backward loop", expectation="exact only over uniform finite bank batch index",
            windows="only complete consecutive [start,end) windows telescope; R is observed nonlinear remainder, not an L bound")
        if getattr(self.method, "warmup_steps", 0):
            result["sgd_prefix_updates"] = self.method.warmup_steps
            result["warmup_semantics"] = "actual_endpoint is SGD during prefix; T1/source/moments remain full-DPCO counterfactuals"
        return result

    def _executed_sgd(self, before):
        return self.trajectory_method == "sgd" or int(before.step) < getattr(self.method, "warmup_steps", 0)

    def snapshot(self):
        return dict(target_id=self.target.identity if self.target else None,
                    windows={key: dict(value) for key, value in self._window_state.items()})

    def restore(self, state, *, step=None):
        if state is None or state["target_id"] != (self.target.identity if self.target else None):
            raise ValueError("Mechanism checkpoint has no matching fixed target")
        self._window_state = {key: dict(value) for key, value in state["windows"].items()}
        if step is not None:
            for start, end in self.config.windows:
                if step <= start:
                    continue
                prefix = self._window_state.get(f"{start}:{end}")
                expected = min(step,end)
                if prefix is None or prefix["next_step"] != expected or prefix["count"] != expected-start:
                    raise ValueError("Mechanism checkpoint is missing a complete continuous-window prefix")

    def _sources(self, params, batch, buffers):
        result = self._call("forward", self.ad._forward, params, batch, buffers)
        valid = result.valid_mask
        errors = jax.lax.stop_gradient(valid & result.feedback["error_mask"])
        count = jnp.sum(valid, dtype=jnp.float32)
        if scalar(count) == 0:
            raise ValueError("Source batch must have a valid contribution")
        gd = self._call("mean_gradient", self.ad._mean_gradient, params, batch, buffers,
            valid.astype(result.losses.dtype), count)
        both = {}
        for name in self.reference_names:
            selected = valid if name == "loss" else errors
            if name == "correct":
                selected = jax.lax.stop_gradient(valid & ~errors)
            n = jnp.sum(selected, dtype=jnp.float32)
            r = gd if name == "loss" else (self._call("mean_gradient", self.ad._mean_gradient,
                params, batch, buffers, selected.astype(result.losses.dtype), n) if scalar(n) else
                jax.tree_util.tree_map(jnp.zeros_like, gd))
            both[name] = self._from_reference(params, batch, buffers, result, gd, errors, r, n)
        return both

    def _from_reference(self, params, batch, buffers, result, gd, errors, r, selected_count):
        r = jax.tree_util.tree_map(jax.lax.stop_gradient, r)
        scores = self._call("scores", self.ad._scores, params, batch, buffers, r)
        weights = jax.lax.stop_gradient(jnp.where(result.valid_mask,
            jnp.where((scores < 0) & (selected_count > 0), -1., 1.), 0.))
        gs = gd if not bool(jax.device_get(jnp.any(weights < 0))) else self._call(
            "mean_gradient", self.ad._mean_gradient, params, batch, buffers,
            weights.astype(result.losses.dtype), jnp.sum(result.valid_mask, dtype=jnp.float32))
        return IndependentDirections(result.losses, result.valid_mask, errors, selected_count, r, scores, weights,
            gd, gs, result.model_state)

    def _dot_value(self, a, b):
        return scalar(self._call("dot", self._dot, a, b))

    def _correction(self, stats):
        return self._call("difference", self._difference, stats.dpco, stats.sgd)

    def _summary(self, stats):
        info = jax.device_get(self._call("summary", self.ad._summary, stats))
        info = {key: np.asarray(value).item() for key, value in info.items()}
        count = int(np.count_nonzero(jax.device_get(stats.valid)))
        scores = np.asarray(jax.device_get(stats.scores))
        valid = np.asarray(jax.device_get(stats.valid))
        formula = float(-2 * np.sum(np.minimum(scores[valid], 0), dtype=np.float64)/count)
        info.update(negative_count=int(np.count_nonzero(scores[valid] < 0)), delta_formula=formula,
                    identity_residual=info["reference_delta"]-formula)
        return info

    def _check(self, before, after, batch, stats, eta, key):
        if getattr(self.method, "warmup_steps", 0):
            from src.methods.dpco_policy import PolicyState
            if (not isinstance(before.method_state, PolicyState) or not isinstance(after.method_state, PolicyState)
                    or int(before.method_state.updates) != int(before.step)
                    or int(after.method_state.updates) != int(after.step)):
                raise ValueError("Warmup checkpoint phase does not match completed updates")
        replay = self._call("method_replay", self.ad._replay,
            before.params, batch, before.model_state, before.method_state, key)
        expected = stats.sgd if self._executed_sgd(before) else stats.dpco
        difference = lambda a, b: self._call("difference", self._difference, a, b)
        norm = lambda a: scalar(self._call("norm", self._norm, a))
        pd = self._call("candidate", self.ad._candidate, before.params, expected, eta)
        pr = self._call("candidate", self.ad._candidate, before.params, replay.direction, eta)
        de, dn = norm(difference(replay.direction, expected)), norm(expected)
        pe, displacement = norm(difference(after.params, pd)), norm(difference(after.params, before.params))
        me = scalar(self._call("state_error", self._state_error, after.model_state, stats.model_state))
        rme = scalar(self._call("state_error", self._state_error, replay.model_state, stats.model_state))
        supported = self.compute_dtype == "float32"
        finite = bool(np.isfinite([de, dn, pe, displacement, me, rme]).all())
        return dict(direction_l2_error=de, direction_relative_error=relative(de, dn), direction_norm=dn,
            parameter_l2_error=pe, parameter_relative_error=relative(pe, displacement),
            actual_displacement_norm=displacement, replayed_parameter_l2_error=norm(difference(after.params, pr)),
            model_state_max_abs_error=me, replayed_model_state_max_abs_error=rme,
            parameter_dtype=",".join(sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(before.params)})),
            compute_dtype=self.compute_dtype, matmul_precision=str(jax.config.jax_default_matmul_precision),
            absolute_tolerance=self.ATOL, relative_tolerance=self.RTOL, tolerance_supported=supported,
            reliable=supported and finite and de <= self.ATOL+self.RTOL*dn
                and pe <= self.ATOL+self.RTOL*displacement and max(me, rme) <= self.ATOL,
            finite=finite)

    def _value(self, params, batch, buffers, mask):
        return scalar(self._call("objective", self.ad._objective, params, batch, buffers, mask))

    def _pair_rows(self, before, after, stats, reference, batch, eta_actual, bank_value, bank_h, check, base):
        c = self._correction(stats)
        ns, nd = self._dot_value(stats.dpco, stats.dpco), self._dot_value(stats.sgd, stats.sgd)
        objectives = [("source_full_batch_ce", stats.valid, self._dot_value(stats.sgd, c))]
        if reference == "error" and bool(jax.device_get(jnp.any(stats.errors))):
            objectives.append(("source_frozen_error_ce", stats.errors, self._dot_value(stats.reference, c)))
        if reference == "correct":
            correct = jax.lax.stop_gradient(stats.valid & ~stats.errors)
            if bool(jax.device_get(jnp.any(correct))):
                objectives.append(("source_frozen_correct_ce", correct, self._dot_value(stats.reference, c)))
        if self.target is not None:
            objectives.append(("fixed_training_bank_ce", None, self._dot_value(bank_h, c)))
        rates = [("absolute", None, float(v)) for v in self.config.absolute_lrs]
        rates.extend(("relative", float(scale), eta_actual*scale) for scale in self.config.lr_scales)
        for objective, mask, original_gain in objectives:
            is_bank = mask is None
            value = (lambda p: scalar(self.target.evaluate(p, self.ad, self._call, gradient=False)[0])) if is_bank else (
                lambda p: self._value(p, batch, before.model_state, mask))
            target_id = self.target.identity if is_bank else f"{base['source_batch_id']}:{objective}:state_{base['step']}"
            def cached(label, rate, direction=None):
                cache_key = (target_id, label, rate)
                if cache_key not in self._loss_cache:
                    candidate = (before.params if label == "initial" else after.params if label == "actual" else
                        self._call("candidate", self.ad._candidate, before.params, direction, rate))
                    self._loss_cache[cache_key] = value(candidate)
                return self._loss_cache[cache_key]
            initial = scalar(bank_value) if is_bank else cached("initial", 0.)
            if is_bank:
                gradient = bank_h
            elif objective == "source_frozen_error_ce":
                gradient = stats.reference
            else:
                gradient = stats.sgd
            projected_dpco = self._dot_value(gradient, stats.dpco)
            projected_sgd = self._dot_value(gradient, stats.sgd)
            for kind, scale, eta in rates:
                for normalization in (("original", "norm_matched") if self.config.norm_matched else ("original",)):
                    defined, reason = True, ""
                    eta_s = eta
                    if normalization == "norm_matched":
                        if ns == 0:
                            defined, reason, eta_s = False, "zero_dpco_norm", None
                        else:
                            eta_s = eta*math.sqrt(nd/ns)
                            reason = "zero_lr" if eta == 0 else ("zero_sgd_norm" if nd == 0 else "")
                    row_base = dict(base, target_id=target_id, reference=reference, objective=objective,
                        buffer_semantics="fixed_bank_input_buffers" if is_bank else "source_before_buffers",
                        lr_kind=kind, lr_scale=scale, probe_lr=eta, dpco_probe_lr=eta_s,
                        normalization=normalization, defined=defined, undefined_reason=reason,
                        reference_valid=bool(scalar(stats.selected_count)>0),
                        objective_gain=original_gain, numeric_checked=check is not None,
                        update_reliable=False if check is None else check["reliable"])
                    if not defined:
                        yield {**row_base, "candidate_kind": "independent_pair", "actual_endpoint": "none",
                            "j_initial": initial, "signal_reliable": False, "finite": True}
                        continue
                    # Cache SCALAR losses only, not full parameter candidates.
                    # Common SGD endpoints and duplicate absolute/relative rates
                    # need not repeat whole-bank forwards across E/L/oracle rows.
                    dpco_label = "sgd" if stats.dpco is stats.sgd else "dpco_"+reference
                    jd = cached("sgd", eta, stats.sgd)
                    js = cached(dpco_label, eta_s, stats.dpco)
                    h = js-jd
                    effective = (original_gain if normalization == "original" else
                        eta_s/eta*projected_dpco-projected_sgd) if eta else None
                    resolution = self.ATOL*max(1., abs(initial), abs(jd), abs(js))
                    data = dict(j_initial=initial, j_sgd=jd, j_dpco=js, difference=h,
                        effective_objective_gain=effective, difference_over_lr=h/eta if eta else None,
                        first_order_residual=h/eta+effective if eta else None,
                        loss_resolution=resolution, finite=bool(np.isfinite([initial, jd, js, original_gain]).all()))
                    signal = abs(eta*effective) if effective is not None else 0.
                    reliable = bool(check is not None and check["reliable"] and abs(h)>resolution and signal>resolution)
                    yield {**row_base, **data, "candidate_kind": "independent_pair", "actual_endpoint": "none",
                        "endpoint_loss_error": 0., "signal_reliable": reliable}
                    # Do not label a mismatched E/L or norm-matched direction as
                    # the real executed DPCO. The real SGD endpoint is common.
                    endpoint = ("sgd" if self._executed_sgd(before) else "dpco"
                        if reference == self.reference_name else None)
                    if normalization == "original" and eta == eta_actual and endpoint:
                        actual = cached("actual", eta)
                        error = abs(actual-(jd if endpoint == "sgd" else js))
                        ad, a_s = (actual, js) if endpoint == "sgd" else (jd, actual)
                        ah = a_s-ad
                        yield {**row_base, **data, "candidate_kind": "actual_pair", "actual_endpoint": endpoint,
                            "j_sgd": ad, "j_dpco": a_s, "difference": ah,
                            "difference_over_lr": ah/eta if eta else None,
                            "first_order_residual": ah/eta+effective if eta else None,
                            "endpoint_loss_error": error,
                            "finite": data["finite"] and bool(np.isfinite(actual)),
                            "signal_reliable": reliable and error < .1*min(abs(h), signal)}

    def _moments(self, before, bank_h, base):
        accum = {name: dict(hs=0., hd=0., delta=0., target=0., mismatch=0., ns=0., nd=0.) for name in self.reference_names}
        mean_gd = None
        k = len(self.target.batches)
        for item in self.target.batches:
            sources = self._sources(before.params, item.batch, self.target.buffers)
            gd = sources["loss"].sgd
            mean_gd = jax.tree_util.tree_map(lambda v: v/k, gd) if mean_gd is None else self._call(
                "mean_add", self._mean_add, mean_gd, gd, 1/k)
            for name, stats in sources.items():
                c = self._correction(stats)
                values = dict(hs=self._dot_value(bank_h, stats.dpco), hd=self._dot_value(bank_h, stats.sgd),
                    delta=self._dot_value(stats.reference, c), target=self._dot_value(bank_h, c),
                    mismatch=self._dot_value(self._call("difference", self._difference, stats.reference, bank_h), c),
                    ns=self._dot_value(stats.dpco, stats.dpco), nd=self._dot_value(stats.sgd, stats.sgd))
                for key, value in values.items():
                    accum[name][key] += value/k
            del sources, gd, stats, c
        error = scalar(self._call("norm", self._norm, self._call("difference", self._difference, mean_gd, bank_h)))
        hn = self._dot_value(bank_h, bank_h)
        for name, values in accum.items():
            yield dict(**base, reference=name, estimator="exact_finite_bank", bank_batch_count=k,
                probability_sum=1., standard_error=None, mean_h_dot_dpco=values["hs"], mean_h_dot_sgd=values["hd"],
                mean_delta=values["delta"], mean_lambda=values["target"], mean_mismatch=values["mismatch"],
                mean_norm_dpco_sq=values["ns"], mean_norm_sgd_sq=values["nd"], mean_gd_h_l2_error=error,
                mean_gd_h_relative_error=relative(error, math.sqrt(hn)), h_norm_sq=hn)

    def _windows(self, before, after, eta, fa_before, h, base):
        t = int(before.step)
        active = [(a,b) for a,b in self.config.windows if a <= t < b]
        if not active:
            return []
        fa_after = scalar(self.target.evaluate(after.params, self.ad, self._call, gradient=False)[0])
        displacement = self._call("difference", self._difference, after.params, before.params)
        p = -self._dot_value(h, displacement)
        r = fa_after-scalar(fa_before)+p
        rows = []
        for start, end in active:
            key = f"{start}:{end}"
            if key not in self._window_state:
                if t != start:
                    raise ValueError("Cannot start an incomplete cumulative window without checkpointed state")
                self._window_state[key] = dict(start_value=scalar(fa_before), count=0, P=0., R=0., eta=0., eta_sq=0., next_step=start,
                    target_id=self.target.identity)
            state = self._window_state[key]
            if state["target_id"] != self.target.identity:
                raise ValueError("Cannot accumulate a window across different fixed targets")
            if state["next_step"] != t:
                raise ValueError("Continuous mechanism window has a gap or duplicate update")
            state.update(P=state["P"]+p, R=state["R"]+r, eta=state["eta"]+eta,
                eta_sq=state["eta_sq"]+eta*eta, count=state["count"]+1, next_step=t+1)
            endpoint = fa_after-state["start_value"]
            residual = endpoint+state["P"]-state["R"]
            tolerance = self.ATOL*max(1., abs(state["start_value"]), abs(fa_after))
            rows.append(dict(**base, window_start=start, window_end=end, window_count=state["count"],
                window_complete=t+1 == end, fa_before=scalar(fa_before), fa_after=fa_after, P=p, R=r,
                displacement_norm_sq=self._dot_value(displacement, displacement), cumulative_P=state["P"],
                cumulative_R=state["R"], endpoint_loss_difference=endpoint, identity_residual=residual,
                identity_tolerance=tolerance, identity_pass=abs(residual)<=tolerance,
                learning_rate=eta, cumulative_eta=state["eta"], cumulative_eta_sq=state["eta_sq"]))
        return rows

    def run(self, before, after, batch, *, learning_rate, rng_key, context, cumulative_eta_before):
        if int(after.step) != int(before.step)+1:
            raise ValueError("Mechanisms require exactly one actual update")
        self._step, self._timings = int(before.step), []
        self._loss_cache = {}
        started = time.perf_counter()
        base = dict(step=self._step, update_step=int(after.step), trajectory_method=self.trajectory_label,
            trajectory_reference=self.reference_name, source_batch_id=context.get("batch_id", f"train_{self._step+1}"),
            target_id=self.target.identity if self.target else None)
        rows = [("mechanism_context", dict(**base, learning_rate=learning_rate,
            cumulative_eta_before=cumulative_eta_before, source_context_json=json.dumps(context, sort_keys=True)))]
        if not self._setup_reported:
            rows.append(("mechanism_timing", dict(step=self._step, event="setup", kernel="fixed_bank_and_observer",
                duration_s=self.setup_duration_s)))
            self._setup_reported = True
        bank_value, bank_h = (self.target.evaluate(before.params, self.ad, self._call)
                             if self.target is not None else (None, None))
        source_due = self._step in self.config.steps
        check = None
        sources = None
        if source_due and (self.config.update_check or self.config.paired_loss) or self._step in self.config.oracle_steps:
            sources = self._sources(before.params, batch, before.model_state)
        if source_due and self.config.update_check:
            check = self._check(before, after, batch, sources[self.reference_name], learning_rate, rng_key)
            rows.append(("mechanism_update", dict(**base, **check)))
        if sources is not None:
            if self._step in self.config.oracle_steps:
                result = self._call("forward", self.ad._forward, before.params, batch, before.model_state)
                gd = sources["loss"].sgd
                sources["exact_bank_reference_oracle"] = self._from_reference(before.params, batch, before.model_state,
                    result, gd, sources["loss"].errors, bank_h, jnp.sum(result.valid_mask, dtype=jnp.float32))
            for name, stats in sources.items():
                summary = self._summary(stats)
                rows.append(("mechanism_t1", dict(**base, reference=name,
                    **{k:v for k,v in summary.items() if k != "valid_batch_size"})))
                if self.target is not None:
                    c = self._correction(stats)
                    delta, target = summary["reference_delta"], self._dot_value(bank_h, c)
                    mismatch = self._dot_value(self._call("difference", self._difference, stats.reference, bank_h), c)
                    rows.append(("mechanism_transfer", dict(**base, reference=name, scope="ordinary_training_source_to_fixed_proxy",
                        reference_delta=delta, target_lambda=target, mismatch=mismatch,
                        transfer_residual=target-delta+mismatch, h_norm_sq=self._dot_value(bank_h, bank_h))))
                if (self.config.paired_loss and source_due) or name == "exact_bank_reference_oracle":
                    rows.extend(("mechanism_pairs", row) for row in self._pair_rows(before, after, stats, name,
                        batch, learning_rate, bank_value, bank_h, check, base))
            # Release ordinary-source gradient trees before bank enumeration.
            del sources, stats
            if self.target is not None:
                del c
            if self._step in self.config.oracle_steps:
                del gd, result
        if self.config.bank_moments and self._step in self.config.moment_steps:
            rows.extend(("mechanism_moments", row) for row in self._moments(before, bank_h, base))
        rows.extend(("mechanism_windows", row) for row in self._windows(before, after, learning_rate, bank_value, bank_h, base))
        total = time.perf_counter()-started
        rows.extend(("mechanism_timing", row) for row in self._timings)
        rows.append(("mechanism_timing", dict(step=self._step, event="diagnostic_compute", kernel="all_stages",
            duration_s=max(0., total-sum(r["duration_s"] for r in self._timings)))))
        return rows
