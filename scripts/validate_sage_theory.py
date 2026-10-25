"""Independent, finite CPU theory/LR grid; never launches production training."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int, default=[5])
    parser.add_argument("--methods", nargs="+", choices=("sgd", "sage_e", "sage_l"),
                        default=["sgd", "sage_e", "sage_l"])
    parser.add_argument("--lrs", nargs="+", type=float, default=[.01, .1])
    parser.add_argument("--schedules", nargs="+", choices=("constant", "cosine", "inverse_sqrt"), default=["constant"])
    parser.add_argument("--min-lr", type=float, default=0.)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--probe-steps", nargs="+", type=int, default=[0, 5, 20, 50, 79])
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--probe-lrs", nargs="+", type=float, default=[0., .001, .01, .05, .1])
    parser.add_argument("--probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case-mc-samples", type=int, default=2048)
    parser.add_argument("--output", default="results")
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "dimensions", "case_mc_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.samples < 4 or args.samples % 2 or args.probe_count < 0:
        parser.error("samples must be even and >=4; probe-count must be nonnegative")
    if min([args.data_seed, args.probe_seed, *args.seeds, *args.probe_steps]) < 0:
        parser.error("Seeds and probe state steps must be nonnegative")
    import math
    if any(not math.isfinite(v) or v <= 0 for v in args.lrs):
        parser.error("Training LRs must be positive and finite")
    if any(not math.isfinite(v) or v < 0 for v in args.probe_lrs):
        parser.error("Probe LRs must be nonnegative and finite")
    if not math.isfinite(args.min_lr) or not 0 <= args.min_lr <= min(args.lrs):
        parser.error("min-lr must be between zero and every peak LR")
    if args.min_lr and any(schedule != "cosine" for schedule in args.schedules):
        parser.error("A nonzero min-lr requires a cosine-only grid")
    for values in (args.seeds, args.methods, args.lrs, args.schedules, args.probe_steps, args.probe_lrs):
        if len(set(values)) != len(values):
            parser.error("Grid values and probe lists must not contain duplicates")
    grid = [dict(seed=seed, method=method, schedule=schedule, lr=lr)
        for seed in args.seeds for schedule in args.schedules for lr in args.lrs for method in args.methods]
    if args.dry_run:
        print(json.dumps(dict(backend="cpu_only", options=vars(args), grid=grid,
            active_probe_state_steps=[t for t in args.probe_steps if t < args.steps],
            checkpoints=False), indent=2))
        return
    # This deliberately separate CPU experiment is an authorized exception to
    # the production nohup/TPU protocol. Do this before importing any JAX module.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    jax.config.update("jax_platforms", "cpu")
    from src.probes.theory_validation import (SCHEMAS, ExperimentSpec, LogisticSystem,
        analytic_cases, fingerprint, run_experiment, synthetic_data, FLOAT32_ATOL, RATIO_THRESHOLD)
    from src.training.recording import RunRecorder, atomic_json, environment_metadata
    from src.training.results import ResultsPaths

    output = ResultsPaths.resolve(args.output, base_dir=ROOT)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    suite = output.logs / f"theory_validation_{stamp}"
    suite.mkdir(parents=True, exist_ok=False)
    atomic_json(suite / "suite_config.json", vars(args))
    cases = analytic_cases(args.case_mc_samples)
    atomic_json(suite / "analytic_cases.json", cases)
    print(f"Analytic cases: {suite / 'analytic_cases.json'}", flush=True)
    failed = not cases["all_checks_pass"]
    if args.analytic_only:
        return 1 if failed else 0
    x, y = synthetic_data(args.samples, args.dimensions, args.data_seed)
    system = LogisticSystem(x, y)
    source = environment_metadata(ROOT)
    summaries = []
    for item in grid:
        spec = ExperimentSpec(**item, minimum=args.min_lr, steps=args.steps, batch_size=args.batch_size,
            probe_steps=tuple(args.probe_steps), probes=args.probe_count if args.probes else 0,
            probe_seed=args.probe_seed, probe_lrs=tuple(args.probe_lrs))
        # Keep LR/schedule/seed identities in all three per-run result folders.
        # Scientific notation may contain '+', which is not a safe path token.
        lr_label = format(spec.lr, ".12g").replace("+", "")
        name = (f"synthetic_clean_linear_{spec.method}_{spec.seed}_{lr_label}_"
                f"{spec.schedule}_{stamp}")
        paths = output.experiment(name)
        paths.create()
        metadata = dict(experiment="finite_cpu_theory_validation", spec=asdict(spec),
            environment=source, dataset=dict(generator="paired_identical_features_opposite_labels_v1",
                seed=args.data_seed, sample_count=len(y), dimensions=args.dimensions,
                fingerprint=fingerprint(x, y), features=x.tolist(), labels=y.tolist()),
            model="single vector theta; no bias; logits [0, x@theta]",
            actual_dtype="float32 ClassificationTask, existing JVP methods and Trainer",
            oracle_dtype="independent NumPy float64; no JAX precision changes",
            sampling="uniform ordered IID batches with replacement; common index sequence across methods/LRs",
            random_streams={"data": [args.data_seed, 1], "initialization": [spec.seed, 2],
                "training": [spec.seed, 3], "probes": [spec.probe_seed, spec.seed, 4, "state_step"]},
            full_target="unsigned CE over every fixed training example",
            smoothness=dict(L=system.smoothness, formula="lambda_max(X.T@X)/(4N); no bias"),
            conditional_moments="IID probe batches at one frozen state; sample SD/sqrt(n); no across-state pooling",
            tolerances=dict(float32_absolute=FLOAT32_ATOL, alignment_ratio_h_sq=RATIO_THRESHOLD,
                cumulative_bound="raw slack retained; tolerance=2e-6*completed_updates for float32 rounding"),
            timing="train_duration is Trainer step+sync including compilation; diagnostics include direction replay/full-target evaluation/probes; not production throughput",
            batch_sequence_fingerprint_algorithm="SHA256 of little-endian int64 ordered batch indices",
            active_probe_states=[t for t in spec.probe_steps if t < spec.steps],
            devices=[str(d) for d in jax.local_devices()], checkpoints=False)
        recorder = RunRecorder(paths, metadata=metadata, schemas=SCHEMAS)
        try:
            recorder.log("start", step=0, run_name=name)
            _, summary = run_experiment(system, spec, recorder)
            recorder.log("completed", step=spec.steps, run_name=name)
        except Exception as error:
            # Keep already observed rows, failure text and the rest of the grid.
            summary = dict(status="failed", error=f"{type(error).__name__}: {error}")
            (paths.logs / "failure.txt").write_text(traceback.format_exc())
            recorder.log("failed", run_name=name, error=summary["error"])
            failed = True
        finally:
            recorder.close()
        summary = dict(run_name=name, **item, **summary)
        atomic_json(paths.logs / "theory_summary.json", summary)
        summaries.append(summary)
        atomic_json(suite / "grid_results.json", summaries)
    best = []
    for seed in args.seeds:
        for schedule in args.schedules:
            for method in args.methods:
                group = [r for r in summaries if r["status"] == "completed" and
                         (r["seed"], r["schedule"], r["method"]) == (seed, schedule, method)]
                if group:
                    best.append(min(group, key=lambda r: r["final_full_train_ce"]))
    atomic_json(suite / "best_in_finite_grid.json", dict(selection="lowest final unsigned full-training CE; equal grid budget",
        limitation="finite-grid training diagnostic, not exact optimal LR or held-out generalization", results=best))
    print(f"Grid results: {suite / 'grid_results.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
