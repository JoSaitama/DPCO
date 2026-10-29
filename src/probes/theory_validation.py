"""Small CPU experiments; production methods update, NumPy oracles only observe.

The model has ONE parameter vector, no bias, and logits [0, x @ theta].
ClassificationTask intentionally retains its float32 loss path. Float64 below
means independent NumPy calculations, never a change to JAX's global precision.
"""

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.data.batch import classification_batch
from src.methods import create_method
from src.tasks.base import TaskResult
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.learning_rates import learning_rate_for
from src.config import TrainingConfig
from src.training.updates import SGDUpdate


BASE = ("step", "method", "seed", "schedule", "initial_lr")
SCHEMAS = {
    "theory_train": (*BASE, "state_before", "state_after", "samples_seen", "learning_rate",
        "cumulative_eta", "cumulative_eta_sq", "batch_id", "batch_indices_json",
        "full_train_ce_before", "full_train_ce_after", "full_train_ce_oracle_before",
        "full_train_ce_oracle_after", "train_top1_before", "train_top1_after",
        "h_norm_sq", "target_projection", "direction_norm_sq", "parameter_update_error",
        "cumulative_target_descent", "cumulative_second_order_cost", "smoothness_L",
        "cumulative_bound_rhs", "bound_slack", "bound_tolerance", "bound_pass", "train_duration_s",
        "diagnostic_duration_s", "elapsed_wall_s"),
    "theory_pairs": (*BASE, "source_id", "source_indices_json", "reference", "objective",
        "selected_count", "flip_count", "probe_lr", "j_initial", "j_sgd", "j_dpco",
        "difference", "objective_gain", "difference_over_lr", "first_order_residual",
        "reference_delta", "target_lambda", "mismatch", "identity_residual",
        "transfer_residual", "norm_sgd_sq", "norm_dpco_sq", "sample_energy",
        "energy_slack", "oracle_direction_error", "float32_tolerance", "finite"),
    "theory_moments": (*BASE, "reference", "estimator", "sample_count", "batch_size",
        "probability_sum", "statistic", "mean", "standard_error", "defined",
        "h_norm_sq", "alignment_ratio_threshold"),
}

FLOAT32_ATOL = 2e-6
RATIO_THRESHOLD = 1e-12


def random_stream(seed, stream):
    """Stable role-specific RNGs; probe counts cannot consume training randomness."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def synthetic_data(n=32, dimensions=3, seed=42):
    if n < 4 or n % 2 or dimensions < 1:
        raise ValueError("Use an even sample count >=4 and positive dimensions")
    features = random_stream(seed, 1).normal(size=(n // 2, dimensions)).astype(np.float32)
    # Identical features with opposite labels make linear separation impossible.
    x = np.repeat(features, 2, axis=0)
    y = np.tile(np.array([0, 1], np.int32), n // 2)
    return x, y


def fingerprint(x, y):
    return hashlib.sha256(np.asarray(x, np.float32).tobytes()
        + np.asarray(y, np.int32).tobytes()).hexdigest()


def linear_model(params, images, state):
    z = images @ params["theta"]
    return jnp.stack((jnp.zeros_like(z), z), axis=-1), state


def logistic_oracle(theta, x, y, valid=None):
    """Independent, overflow-safe binary logistic loss and individual gradients."""
    theta, x, y = np.asarray(theta, np.float64), np.asarray(x, np.float64), np.asarray(y)
    valid = np.ones(len(y), bool) if valid is None else np.asarray(valid, bool)
    x, y = x[valid], y[valid]
    if not len(y):
        raise ValueError("Oracle requires at least one valid sample")
    z = x @ theta
    p = np.exp(-np.logaddexp(0., -z))
    losses = np.logaddexp(0., z) - y * z
    gradients = (p - y)[:, None] * x
    # argmax([0,z]) selects class zero on ties, just like ClassificationTask.
    errors = (z > 0).astype(np.int32) != y
    return losses, gradients, errors


def oracle_directions(theta, x, y, reference, valid=None):
    losses, gradients, errors = logistic_oracle(theta, x, y, valid)
    selected = errors if reference == "error" else np.ones(len(losses), bool)
    r = gradients[selected].mean(0) if selected.any() else np.zeros(gradients.shape[1])
    scores = gradients @ r
    weights = np.where(scores < 0, -1., 1.)
    return gradients.mean(0), (weights[:, None] * gradients).mean(0), r, scores


class LogisticSystem:
    """Cache tiny kernels once; all returned directions are existing methods."""
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, np.float32), np.asarray(y, np.int32)
        self.task = ClassificationTask(linear_model, 2)
        self.full_batch = classification_batch(self.x, self.y)
        self.methods = {"sgd": create_method("sgd"),
            "dpco_e": create_method("dpco", reference_name="error", gradient_backend="jvp"),
            "dpco_l": create_method("dpco", reference_name="loss", gradient_backend="jvp")}
        self.kernels = {name: jax.jit(lambda p, b, method=method:
            method.compute_direction(self.task, p, b).direction["theta"])
            for name, method in self.methods.items()}
        self.many = {name: jax.jit(jax.vmap(kernel, in_axes=(None, 0)))
            for name, kernel in self.kernels.items()}
        self.evaluate = jax.jit(lambda p, b: self.task.evaluate(p, b))
        x64 = self.x.astype(np.float64)
        # Hessian = mean p_i(1-p_i) x_i x_i^T <= X^T X/(4N).
        self.smoothness = float(np.linalg.eigvalsh(x64.T @ x64 / (4 * len(x64)))[-1])

    def batch(self, indices):
        return classification_batch(self.x[indices], self.y[indices])

    def full(self, params):
        result = self.evaluate(params, self.full_batch)
        losses, grads, _ = logistic_oracle(params["theta"], self.x, self.y)
        return (float(np.mean(np.asarray(result.losses))),
            float(1 - np.mean(np.asarray(result.feedback["error_mask"]))),
            float(losses.mean()), grads.mean(0))

    def directions(self, params, batch, reference):
        gd = np.asarray(self.kernels["sgd"](params, batch), np.float64)
        gs = np.asarray(self.kernels["dpco_e" if reference == "error" else "dpco_l"](params, batch), np.float64)
        evaluation = self.evaluate(params, batch)
        valid = np.asarray(batch.valid_mask)
        errors = np.asarray(evaluation.feedback["error_mask"]) & valid
        selected = errors if reference == "error" else valid
        # This dense gradient is an independent SMALL-model observer only.
        losses, grads, _ = logistic_oracle(params["theta"], batch.images, batch.labels, valid)
        r = grads[selected[valid]].mean(0) if selected.any() else np.zeros_like(gd)
        scores = grads @ r
        correction = gs - gd
        h = logistic_oracle(params["theta"], self.x, self.y)[1].mean(0)
        delta = float(r @ correction)
        target = float(h @ correction)
        mismatch = float((r - h) @ correction)
        oracle_gd, oracle_gs, _, _ = oracle_directions(params["theta"], batch.images,
            batch.labels, reference, valid)
        energy = float(np.mean(np.sum(grads ** 2, axis=1)))
        return dict(gd=gd, gs=gs, reference=r, selected=selected, h=h,
            reference_delta=delta, target_lambda=target, mismatch=mismatch,
            identity_residual=delta + 2 * float(scores[scores < 0].sum()) / len(grads),
            transfer_residual=target - delta + mismatch,
            norm_sgd_sq=float(gd @ gd), norm_dpco_sq=float(gs @ gs),
            sample_energy=energy, energy_slack=energy - float(gs @ gs),
            flip_count=int(np.count_nonzero(scores < 0)), selected_count=int(selected.sum()),
            oracle_direction_error=float(max(np.linalg.norm(gd - oracle_gd), np.linalg.norm(gs - oracle_gs))))


def paired_rows(system, params, indices, reference, lrs, source_id):
    """Freeze source, mask and both directions, then vary candidate LR only."""
    batch = system.batch(indices)
    info = system.directions(params, batch, reference)
    all_valid = np.asarray(batch.valid_mask)
    objectives = [("source_full_batch_ce", batch, all_valid),
                  ("fixed_full_training_ce", system.full_batch, np.ones(len(system.y), bool))]
    if reference == "error" and info["selected_count"]:
        objectives.append(("source_frozen_error_ce", batch, info["selected"]))
    for objective, target, mask in objectives:
        grads = logistic_oracle(params["theta"], target.images, target.labels)[1]
        gain = float(grads[mask].mean(0) @ (info["gs"] - info["gd"]))
        initial = float(np.asarray(system.evaluate(params, target).losses)[mask].mean())
        for eta in lrs:
            candidates = [{"theta": params["theta"] - eta * jnp.asarray(info[key], dtype=jnp.float32)}
                          for key in ("gd", "gs")]
            jd, js = [float(np.asarray(system.evaluate(p, target).losses)[mask].mean()) for p in candidates]
            difference = js - jd
            public = {k: v for k, v in info.items() if k not in ("gd", "gs", "reference", "selected", "h")}
            yield dict(source_id=source_id, source_indices_json=json.dumps(np.asarray(indices).tolist()),
                reference=reference, objective=objective, probe_lr=eta, j_initial=initial,
                j_sgd=jd, j_dpco=js, difference=difference, objective_gain=gain,
                difference_over_lr=difference / eta if eta else None,
                first_order_residual=difference / eta + gain if eta else None,
                float32_tolerance=FLOAT32_ATOL,
                finite=bool(np.isfinite([initial, jd, js, gain]).all()), **public)


def moment_rows(system, params, batches, reference, *, exact=False):
    """Conditional moments at ONE state, uniform ordered IID batches.

    Exact enumeration means exhaustive finite sampling distribution evaluated by
    the real float32 kernels. It is not a claim of exact real arithmetic. MC SEs
    use independent batches, never the LR scan or different trajectory states.
    """
    batches = np.asarray(batches)
    b = batches.shape[1]
    from src.data.batch import ClassificationBatch
    batch = ClassificationBatch(jnp.asarray(system.x[batches]), jnp.asarray(system.y[batches]),
        jnp.ones(batches.shape, bool))
    gd = np.asarray(system.many["sgd"](params, batch), np.float64)
    gs = np.asarray(system.many["dpco_e" if reference == "error" else "dpco_l"](params, batch), np.float64)
    _, gradients, errors = logistic_oracle(params["theta"], system.x, system.y)
    sample_grads = gradients[batches]
    masks = errors[batches] if reference == "error" else np.ones(batches.shape, bool)
    counts = masks.sum(1)
    r = (sample_grads * masks[:, :, None]).sum(1) / np.maximum(counts, 1)[:, None]
    h = gradients.mean(0)
    c = gs - gd
    dot = lambda a, z: np.sum(a * z, axis=-1)
    delta, target = dot(r, c), c @ h
    scores = np.sum(sample_grads * r[:, None, :], axis=-1)
    values = {"h_dot_dpco": gs @ h, "h_dot_sgd": gd @ h,
        "delta": delta, "lambda": target, "mismatch": dot(r - h, c),
        "dpco_norm_sq": dot(gs, gs), "sgd_norm_sq": dot(gd, gd),
        "sample_energy": np.mean(dot(sample_grads, sample_grads), axis=1),
        "identity_residual": delta + 2 * np.minimum(scores, 0).sum(1) / b,
        "transfer_residual": target - delta + dot(r - h, c)}
    for i in range(len(h)):
        values[f"sgd_component_{i}"] = gd[:, i]
        values[f"dpco_component_{i}"] = gs[:, i]
        values[f"sgd_oracle_component_{i}"] = sample_grads[:, :, i].mean(1)
    norm_h = float(h @ h)
    values["alignment_ratio"] = gs @ h / norm_h if norm_h > RATIO_THRESHOLD else None
    for name, v in values.items():
        yield dict(reference=reference, estimator="exact_enumeration_float32" if exact else "monte_carlo",
            sample_count=len(batches), batch_size=b, probability_sum=1. if exact else None,
            statistic=name, mean=float(np.mean(v)) if v is not None else None,
            standard_error=(float(np.std(v, ddof=1) / math.sqrt(len(v)))
                if not exact and v is not None and len(v) > 1 else None),
            defined=v is not None, h_norm_sq=norm_h, alignment_ratio_threshold=RATIO_THRESHOLD)


class QuadraticPairTask:
    """Analytic case A only; not the logistic training experiment."""
    def evaluate(self, params, batch, model_state=None):
        targets, valid = batch
        return TaskResult(.5 * (params["theta"][0] - targets) ** 2,
            valid, model_state, {"error_mask": jnp.zeros_like(valid)})


def analytic_cases(mc_samples=2048):
    """Return all checks and adverse outcomes without filtering DPCO losses."""
    task = QuadraticPairTask()
    batch = (jnp.array([1., -1.]), jnp.array([True, True]))
    method = create_method("dpco", reference_name="loss", gradient_backend="jvp")
    trainers = {"dpco": Trainer(task, method, SGDUpdate(.1)),
                "sgd": Trainer(task, create_method("sgd"), SGDUpdate(.1))}
    trajectories = {}
    for name, trainer in trainers.items():
        step_fn = trainer.compile_step()
        state = TrainState({"theta": jnp.array([.25], dtype=jnp.float32)})
        trajectory = [.25]
        for _ in range(20):
            state, _ = step_fn(state, batch)
            trajectory.append(float(state.params["theta"][0]))
        trajectories[name] = trajectory
    first = next((t for t in range(21) if abs(trajectories["sgd"][t]) < abs(trajectories["dpco"][t])), None)
    theta = .05
    d = method.compute_direction(task, {"theta": jnp.array([theta])}, batch).direction["theta"]
    observed_h = .5 * ((theta - .1 * float(d[0])) ** 2 - (theta * .9) ** 2)
    formula_h = -.1 * (theta - theta ** 2) + .1 ** 2 * (1 - theta ** 2) / 2
    zero = method.compute_direction(task, {"theta": jnp.array([0.])}, batch).direction["theta"]
    a_checks = dict(first_crossover_is_16=first == 16,
        dpco_cycle=bool(np.allclose(trajectories["dpco"][:6], [.25,.15,.05,-.05,.05,-.05], atol=1e-6)),
        sgd_closed_form=bool(np.allclose(trajectories["sgd"], .25 * .9 ** np.arange(21), atol=1e-6)),
        zero_no_flip=bool(np.allclose(zero, 0)),
        h_matches_formula=abs(observed_h - formula_h) < 1e-8,
        h_expected=abs(observed_h - .0002375) < 1e-8)
    direction_checks = []
    for point in (.25, .15, .05, -.05, -.25, 0.):
        p = {"theta": jnp.array([point], dtype=jnp.float32)}
        gs = float(method.compute_direction(task, p, batch).direction["theta"][0])
        gd = float(create_method("sgd").compute_direction(task, p, batch).direction["theta"][0])
        delta = point * (gs-gd)
        direction_checks.append(dict(theta=point, Gd=gd, Gs=gs, delta=delta,
            pass_direction=bool(abs(gs-np.sign(point)) < 1e-6),
            pass_delta=abs(delta-(abs(point)-point**2)) < 1e-7))
    a_checks["direction_and_delta_formula"] = all(r["pass_direction"] and r["pass_delta"] for r in direction_checks)
    system = LogisticSystem(np.array([[-.01], [1.]], np.float32), np.array([1, 1]))
    params = {"theta": jnp.array([1.], dtype=jnp.float32)}
    info = system.directions(params, system.full_batch, "error")
    oracle_gd, oracle_gs, r, scores = oracle_directions([1.], system.x, system.y, "error")
    initial = float(logistic_oracle([1.], system.x, system.y)[0].mean())
    next_e, _ = Trainer(system.task, system.methods["dpco_e"], SGDUpdate(.01)).step(
        TrainState(params), system.full_batch)
    next_d, _ = Trainer(system.task, system.methods["sgd"], SGDUpdate(.01)).step(
        TrainState(params), system.full_batch)
    fe = float(logistic_oracle(next_e.params["theta"], system.x, system.y)[0].mean())
    fd = float(logistic_oracle(next_d.params["theta"], system.x, system.y)[0].mean())
    b_checks = dict(reference_positive=info["reference_delta"] > 0,
        target_alignment_negative=float(info["h"] @ info["gs"]) < 0,
        full_objective_worse=fe > initial > fd,
        oracle_matches=info["oracle_direction_error"] < FLOAT32_ATOL,
        expected_numbers=bool(np.allclose([oracle_gd[0], oracle_gs[0], r[0], initial, fe, fd],
            [-.13195821079, .13698321058, .00502499979, .50571068401, .50589153687, .50553663990], atol=2e-8)))
    tiny = LogisticSystem(np.array([[-.01], [1.], [-1.]], np.float32), np.ones(3, np.int32))
    ordered = np.array(list(product(range(3), repeat=2)))
    monte_carlo = random_stream(42, 5).integers(0, 3, size=(mc_samples, 2))
    conditional = []
    for reference in ("loss", "error"):
        conditional.extend(moment_rows(tiny, params, ordered, reference, exact=True))
        conditional.extend(moment_rows(tiny, params, monte_carlo, reference))
    h = tiny.full(params)[3][0]
    exact_gd = [r for r in conditional if r["estimator"].startswith("exact") and r["statistic"] == "sgd_component_0"]
    checks = dict(exact_unbiased_with_float32_tolerance=all(abs(r["mean"] - h) < FLOAT32_ATOL for r in exact_gd))
    # A finite MC estimate is not required to equal the full gradient. Its
    # independent-sample error is compared with the enumerated distribution.
    mc_comparisons = []
    for row in conditional:
        if row["estimator"] != "monte_carlo" or not row["defined"]:
            continue
        exact_row = next(r for r in conditional if r["reference"] == row["reference"]
            and r["statistic"] == row["statistic"] and r["estimator"].startswith("exact"))
        error = abs(row["mean"] - exact_row["mean"])
        se = row["standard_error"]
        mc_comparisons.append(dict(reference=row["reference"], statistic=row["statistic"],
            absolute_error=error, standard_error=se,
            within_six_se=error <= 6 * (se or 0.) + FLOAT32_ATOL))
    checks["mc_agrees_within_six_se"] = all(r["within_six_se"] for r in mc_comparisons)
    return {"A": {"checks": a_checks, "first_crossover": first, "trajectories": trajectories,
                  "same_state_H": observed_h, "formula_H": formula_h, "direction_checks": direction_checks,
                  "interpretation": "Exact reference; finite-step second-order reversal, not reference mismatch."},
        "B": {"checks": b_checks, "g1": float(r[0]), "g2": float(2*oracle_gd[0]-r[0]),
              "Gd": float(oracle_gd[0]), "Gs": float(info["gs"][0]),
              "h_dot_Gs": float(info["h"] @ info["gs"]),
              "delta": info["reference_delta"], "lambda": info["target_lambda"],
              "F_initial": initial, "F_dpco": fe, "F_sgd": fd},
        "C": {"checks": checks, "dataset_x": tiny.x.tolist(), "dataset_y": tiny.y.tolist(),
              "theta": [1.], "ordered_batch_count": len(ordered), "ordered_batch_probability": 1/9,
              "mc_seed": 42, "mc_comparisons": mc_comparisons, "moments": conditional},
        "all_checks_pass": all([*a_checks.values(), *b_checks.values(), *checks.values()])}


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int = 5
    method: str = "dpco_l"
    schedule: str = "constant"
    lr: float = .05
    minimum: float = 0.
    steps: int = 100
    batch_size: int = 8
    probe_steps: tuple = (0, 5, 20, 50, 99)
    probes: int = 32
    probe_seed: int = 42
    probe_lrs: tuple = (0., .001, .01, .05, .1)


def run_experiment(system, spec, recorder):
    """All real transitions are Trainer.step; observer computations are read-only."""
    if spec.method not in system.methods or spec.schedule not in ("constant", "cosine", "inverse_sqrt"):
        raise ValueError("Only original SGD/DPCO-E/L and constant/cosine/inverse_sqrt are supported")
    if spec.steps < 1 or spec.batch_size < 1 or spec.probes < 0:
        raise ValueError("Positive training budget/batch size and nonnegative probes required")
    training = TrainingConfig(total_steps=spec.steps, learning_rate=spec.lr,
        lr_schedule=spec.schedule, min_learning_rate=spec.minimum)
    lr = learning_rate_for(training)
    updater = SGDUpdate(lr)
    trainer = Trainer(system.task, system.methods[spec.method], updater)
    # This is the existing Trainer, not an analytic optimizer implementation.
    step_fn = trainer.compile_step()
    actual_direction = system.kernels[spec.method]
    state = TrainState({"theta": jnp.asarray(random_stream(spec.seed, 2).normal(
        scale=.4, size=system.x.shape[1]), dtype=jnp.float32)})
    indices = random_stream(spec.seed, 3).integers(0, len(system.y), size=(spec.steps, spec.batch_size))
    sequence_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    initial_ce, _, initial_f, _ = system.full(state.params)
    eta_sum = eta_sq = descent = second = train_s = diagnostic_s = 0.
    bound_ok = True
    started = time.perf_counter()
    for t, source_indices in enumerate(indices):
        observer_start = time.perf_counter()
        before_ce, before_acc, before_f, h = system.full(state.params)
        if not np.isfinite([before_ce, before_f, *h]).all():
            raise FloatingPointError(f"Nonfinite full-training evaluation at state {t}")
        eta = float(lr(t) if callable(lr) else lr)
        batch = system.batch(source_indices)
        direction = np.asarray(actual_direction(state.params, batch), np.float64)
        diag_duration = time.perf_counter() - observer_start
        duration = time.perf_counter()
        after, metrics = step_fn(state, batch)
        jax.block_until_ready(after)
        train_duration = time.perf_counter() - duration
        train_s += train_duration
        if not bool(metrics["parameters_finite"]):
            raise FloatingPointError(f"Nonfinite update at state {t}")
        observer_start = time.perf_counter()
        after_ce, after_acc, after_f, _ = system.full(after.params)
        if not np.isfinite([after_ce, after_f]).all():
            raise FloatingPointError(f"Nonfinite full-training CE after update {t+1}")
        error = float(np.linalg.norm(np.asarray(after.params["theta"], np.float64)
            - (np.asarray(state.params["theta"], np.float64) - eta * direction)))
        eta_sum += eta
        eta_sq += eta * eta
        projection, norm_sq = float(h @ direction), float(direction @ direction)
        descent += eta * projection
        second += system.smoothness * .5 * eta * eta * norm_sq
        rhs = initial_f - descent + second
        # Float32 parameter rounding is retained; the exact smooth inequality
        # applies to theta-eta*d. A tolerance flags, rather than hides, residuals.
        tolerance = FLOAT32_ATOL * (t + 1)
        passed = after_f <= rhs + tolerance
        bound_ok &= passed
        diag_duration += time.perf_counter() - observer_start
        if t in spec.probe_steps and spec.probes:
            diag_start = time.perf_counter()
            sampled = np.random.default_rng(np.random.SeedSequence([spec.probe_seed, spec.seed, 4, t])).integers(
                0, len(system.y), size=(spec.probes, spec.batch_size))
            for reference in ("loss", "error"):
                base = dict(step=t, method=spec.method, seed=spec.seed, schedule=spec.schedule, initial_lr=spec.lr)
                for row in moment_rows(system, state.params, sampled, reference):
                    recorder.record("theory_moments", **base, **row)
                # The REAL source batch is paired once; all probe batches enter
                # conditional moments. Extra probe counts do not alter this source.
                for row in paired_rows(system, state.params, source_indices, reference,
                        spec.probe_lrs, f"train_update_{t+1}"):
                    recorder.record("theory_pairs", **base, **row)
            diag_duration += time.perf_counter() - diag_start
        diagnostic_s += diag_duration
        row = dict(step=t+1, method=spec.method, seed=spec.seed, schedule=spec.schedule, initial_lr=spec.lr,
            state_before=t, state_after=t+1, samples_seen=(t+1)*spec.batch_size,
            learning_rate=eta, cumulative_eta=eta_sum, cumulative_eta_sq=eta_sq,
            batch_id=f"train_update_{t+1}", batch_indices_json=json.dumps(source_indices.tolist()),
            full_train_ce_before=before_ce, full_train_ce_after=after_ce,
            full_train_ce_oracle_before=before_f, full_train_ce_oracle_after=after_f,
            train_top1_before=before_acc, train_top1_after=after_acc,
            h_norm_sq=float(h @ h), target_projection=projection, direction_norm_sq=norm_sq,
            parameter_update_error=error, cumulative_target_descent=descent,
            cumulative_second_order_cost=second, smoothness_L=system.smoothness,
            cumulative_bound_rhs=rhs, bound_slack=rhs-after_f, bound_tolerance=tolerance, bound_pass=passed,
            train_duration_s=train_duration, diagnostic_duration_s=diag_duration,
            elapsed_wall_s=time.perf_counter()-started)
        recorder.record("theory_train", **row)
        if t % 25 == 0 or t+1 == spec.steps:
            recorder.log("theory_train", step=t+1, train_ce=after_ce, learning_rate=eta)
            recorder.flush()
        state = after
    return state, dict(status="completed", initial_full_train_ce=initial_ce,
        final_full_train_ce=after_ce, final_train_top1=after_acc, cumulative_bound_pass=bound_ok,
        final_bound_slack=rhs-after_f, cumulative_eta=eta_sum, cumulative_eta_sq=eta_sq,
        training_indices_sha256=sequence_hash, train_duration_s=train_s,
        diagnostic_duration_s=diagnostic_s, elapsed_wall_s=time.perf_counter()-started)
