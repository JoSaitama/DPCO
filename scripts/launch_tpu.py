"""Start TPU-only smoke/training/validation through nohup on worker0.

No JAX import or HF access occurs in this parent process. CPU sanity checks have
their own foreground entry point. All bootstrap output belongs to the run log.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.training.launch import (absolute_path, add_overrides, diagnostic_cli_args,
                                 diagnostic_overrides, resolve_config)
from src.training.results import ResultsPaths, make_run_name
from src.training.runtime import configure_worker0
from src.training.status import RunStatus


def plan_job(args, path, config, method):
    """Build one child command without creating files or touching accelerators."""
    name = args.resume or make_run_name(config)
    paths = ResultsPaths.resolve(config.results.root_dir, base_dir=ROOT).experiment(name)
    command = ["nohup", sys.executable, "-u", str(ROOT / "scripts/train_classification.py"),
               "--config", str(path), "--mode", args.mode, "--run-name", name, "--reserved",
               "--seed", str(config.seed)]
    for flag, value in (("--steps", args.steps), ("--batch-size", args.batch_size),
                        ("--method", method), ("--dataset", args.dataset),
                        ("--alternating-first", args.alternating_first),
                        ("--model", args.model), ("--data-mode", args.data_mode),
                        ("--data-dir", args.data_dir),
                        ("--l2-coefficient", args.l2_coefficient), ("--momentum", args.momentum),
                        ("--lr", args.learning_rate), ("--lr-schedule", args.lr_schedule),
                        ("--min-lr", args.min_learning_rate),
                        ("--checkpoint-every-steps", args.checkpoint_every_steps),
                        ("--eval-every-steps", args.eval_every_steps),
                        ("--eval-every-epochs", args.eval_every_epochs)):
        if value is not None:
            command.extend([flag, str(value)])
    if args.nesterov is not None:
        command.append("--nesterov" if args.nesterov else "--no-nesterov")
    command.extend(diagnostic_cli_args(args))
    if args.resume:
        command.extend(["--resume", "--resume-slot", args.resume_slot])
    return paths, command


def launch_job(args, config, paths, command, *, wait):
    if shutil.which("nohup") is None:
        raise RuntimeError("nohup is required")
    env = os.environ.copy()
    env["SAGE_BACKGROUND_LAUNCH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    settings = configure_worker0(config.execution, environ=env)
    if args.resume:
        paths.create(resume=True)
    else:
        paths.reserve()
    status = RunStatus(paths, config, mode=args.mode, resume=bool(args.resume))
    command = [*command, "--attempt-id", status.attempt_id]
    with status.monitor():
        try:
            with (paths.logs / "stdout.log").open("a", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            with (paths.logs / "stdout.log").open("a", encoding="utf-8") as output:
                traceback.print_exc(file=output)
            raise
    # The child writes start with its real PID, before TPU/data initialization.
    # A failed spawn is finalized by the parent monitor with a start/end pair.
    launch = {"pid": process.pid, "hostname": socket.gethostname(), "command": command,
        "attempt_id": status.attempt_id,
        "runtime_settings": settings, "resume": bool(args.resume), "mode": args.mode}
    (paths.logs / f"launch_{process.pid}.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"Started nohup TPU job: pid={process.pid} run={paths.name}", flush=True)
    print(f"Log: {paths.logs / 'stdout.log'}", flush=True)
    if not wait:
        print("Launched in background; check stdout.log and results/logs/run_status.tsv for completion.", flush=True)
        return True
    # Wait for actual exit/TPU lock release, not just a completed log message.
    code = process.wait()
    if code != 0 or status.outcome() != "completed":
        status.record("failed", pid=process.pid)
        with (paths.logs / "stdout.log").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"event": "launcher_failed", "exit_code": code,
                "message": "Child exit was unsuccessful or lacked a completed status; suite stopped."}) + "\n")
        print(f"Job failed (exit={code}); stopping suite. See {paths.logs / 'stdout.log'}", flush=True)
        return False
    print(f"Completed run: {paths.name}", flush=True)
    return True


def main(*, default_mode="smoke", default_config=None, default_wait=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=default_config is None, default=default_config)
    parser.add_argument("--mode", choices=("smoke", "train", "validate"), default=default_mode)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", metavar="RUN_NAME")
    parser.add_argument("--resume-slot", default="latest",
                        choices=("initial", "p25", "p50", "p75", "final", "best", "latest"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true", default=default_wait,
                        help="Wait for completion; always enabled for multi-job launches and shell suites")
    add_overrides(parser, multiple=True)
    args = parser.parse_args()
    path = absolute_path(args.config, ROOT).resolve()
    seeds = args.seeds if args.seeds is not None else [load_config(path).seed]
    methods = args.methods or [None]
    if len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
        parser.error("Seeds and methods must not contain duplicates")
    if args.resume and len(seeds) * len(methods) != 1:
        parser.error("Resume accepts exactly one seed and one method")
    # Resolve the entire grid before starting any job, so invalid later choices
    # cannot leave a partially executed suite. CLI overrides reach both processes.
    jobs = []
    for seed in seeds:
        for method in methods:
            config = resolve_config(path, mode=args.mode, steps=args.steps, batch_size=args.batch_size,
                seed=seed, method=method, dataset=args.dataset, model=args.model, data_mode=args.data_mode,
                alternating_first=args.alternating_first,
                data_dir=args.data_dir,
                l2_coefficient=args.l2_coefficient, momentum=args.momentum, nesterov=args.nesterov,
                learning_rate=args.learning_rate, lr_schedule=args.lr_schedule,
                min_learning_rate=args.min_learning_rate, checkpoint_every_steps=args.checkpoint_every_steps,
                eval_every_steps=args.eval_every_steps, eval_every_epochs=args.eval_every_epochs,
                **diagnostic_overrides(args))
            jobs.append((config, method))
    wait = args.wait or len(jobs) > 1
    if args.dry_run:
        previews = [(config, *plan_job(args, path, config, method)) for config, method in jobs]
        plans = [{"command": command, "stdout": str(paths.logs / "stdout.log"),
            "mode": args.mode, "backend": "tpu_only", "worker": "worker0_only", "wait": wait,
            "seed": config.seed, "method": config.method.name, "reference": config.reference.name,
            "policy_mode": config.method.policy_mode if config.method.name == "sage_policy" else None,
            "alternating_first": config.method.alternating_first if config.method.name == "sage_policy"
                and config.method.policy_mode == "alternating" else None,
            "dataset": config.data.name, "num_classes": config.data.num_classes,
            "model": config.model.name, "data_mode": config.data.mode,
            "data_dir": str(absolute_path(config.data.prepared_dir, ROOT).resolve()),
            "total_steps": config.training.total_steps, "probes_enabled": config.probes.enabled,
            "batch_size": config.global_batch_size,
            "l2_coefficient": config.training.l2_coefficient,
            "momentum": config.training.momentum, "nesterov": config.training.nesterov,
            "learning_rate": config.training.learning_rate, "lr_schedule": config.training.lr_schedule,
            "min_learning_rate": config.training.min_learning_rate,
            "checkpoint_every_steps": config.checkpoints.latest_every_steps,
            "evaluation_mode": config.evaluation.mode,
            "evaluation_interval": config.evaluation.interval,
            "diagnostics": config.as_dict()["diagnostics"],
            "active_diagnostic_state_steps": [t for t in config.diagnostics.steps
                if config.diagnostics.enabled and t < config.training.total_steps]} for config, paths, command in previews]
        print(json.dumps(plans[0] if len(plans) == 1 else plans, indent=2))
        return
    for config, method in jobs:
        # Timestamp each actual launch, rather than the earlier grid planning time.
        paths, command = plan_job(args, path, config, method)
        if not launch_job(args, config, paths, command, wait=wait):
            raise SystemExit(1)
    if wait:
        print(f"Suite completed successfully: {len(jobs)} run(s).", flush=True)


if __name__ == "__main__":
    main()
