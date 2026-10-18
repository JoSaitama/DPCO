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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.launch import absolute_path, resolve_config
from src.training.results import ResultsPaths, make_run_name
from src.training.runtime import configure_worker0
from src.training.status import RunStatus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "train", "validate"), default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", metavar="RUN_NAME")
    parser.add_argument("--resume-slot", default="latest",
                        choices=("initial", "p25", "p50", "p75", "final", "best", "latest"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = absolute_path(args.config, ROOT).resolve()
    config = resolve_config(path, mode=args.mode, steps=args.steps, batch_size=args.batch_size)
    name = args.resume or make_run_name(config)
    paths = ResultsPaths.resolve(config.results.root_dir, base_dir=ROOT).experiment(name)
    command = ["nohup", sys.executable, "-u", str(ROOT / "scripts/train_classification.py"),
               "--config", str(path), "--mode", args.mode, "--run-name", name, "--reserved"]
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.resume:
        command.extend(["--resume", "--resume-slot", args.resume_slot])
    if args.dry_run:
        print(json.dumps({"command": command, "stdout": str(paths.logs / "stdout.log"),
            "mode": args.mode, "backend": "tpu_only", "worker": "worker0_only",
            "batch_size": config.global_batch_size}, indent=2))
        return
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
    command.extend(["--attempt-id", status.attempt_id])
    with status.monitor():
        # Publish before spawning, so a fast child cannot finish before its
        # launch row appears. The child publishes its own actual process ID.
        status.record("launching", pid="")
        with (paths.logs / "stdout.log").open("a", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    # The child owns status after a successful spawn. A later failure writing
    # launcher metadata must not mark a live training process as failed.
    # No environment dump: HF tokens and SSH credentials must never be serialized.
    launch = {"pid": process.pid, "hostname": socket.gethostname(), "command": command,
        "attempt_id": status.attempt_id,
        "runtime_settings": settings, "resume": bool(args.resume), "mode": args.mode}
    (paths.logs / f"launch_{process.pid}.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"Started nohup TPU job: pid={process.pid} run={name}")
    print(f"Log: {paths.logs / 'stdout.log'}")
    print("Launched in background; completion is not yet confirmed. Check stdout.log and results/logs/run_status.tsv.")


if __name__ == "__main__":
    main()
