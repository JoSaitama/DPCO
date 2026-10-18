"""Suite overrides, sequential execution, and finalization without real TPU use."""

from contextlib import redirect_stdout
import csv
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import launch_tpu
from src.config import ResultsConfig, load_config
from src.training.launch import resolve_config
from src.training.results import ResultsPaths


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml"


class SuiteTests(unittest.TestCase):
    def dry_run(self, *args):
        output = subprocess.check_output([sys.executable, str(ROOT / "scripts/launch_tpu.py"),
            "--config", str(CONFIG), "--dry-run", *args], cwd=gettempdir(), text=True)
        return json.loads(output)

    def test_defaults_and_single_or_multiple_seed_aliases(self):
        default = self.dry_run()
        self.assertEqual(default["seed"], 42)
        self.assertFalse(default["wait"])
        self.assertEqual(self.dry_run("--seed", "13")["seed"], 13)
        plans = self.dry_run("--seed", "10", "15", "22", "46")
        self.assertEqual([p["seed"] for p in plans], [10, 15, 22, 46])
        self.assertTrue(all(p["wait"] for p in plans))
        self.assertEqual([p["seed"] for p in self.dry_run("--seeds", "1", "2")], [1, 2])

    def test_method_grid_and_child_config_round_trip(self):
        plans = self.dry_run("--methods", "sage_e", "sage_l", "sgd", "--seeds", "10", "15",
                             "--dataset", "imagenet1k", "--model", "resnet101", "--datamode", "clean")
        self.assertEqual(len(plans), 6)
        self.assertEqual([(p["seed"], p["reference"]) for p in plans[:3]],
                         [(10, "error"), (10, "loss"), (10, "loss")])
        for plan in plans:
            command = plan["command"]
            value = lambda flag: command[command.index(flag) + 1]
            resolved = resolve_config(value("--config"), mode=value("--mode"),
                seed=int(value("--seed")), method=value("--method"),
                dataset=value("--dataset"), model=value("--model"), data_mode=value("--data-mode"))
            self.assertEqual(resolved.seed, plan["seed"])
            self.assertEqual(resolved.method.name, plan["method"])
            self.assertEqual(resolved.reference.name, plan["reference"])
            self.assertEqual(resolved.data.num_classes, 1000)
            self.assertEqual(resolved.model.name, "resnet101")
            self.assertIn(f"_{resolved.seed}_", plan["stdout"])

    def test_shell_entry_works_outside_repo_and_defaults_to_train_and_wait(self):
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run"],
                                         cwd=gettempdir(), env=env, text=True)
        plan = json.loads(output)
        self.assertEqual(plan["mode"], "train")
        self.assertEqual(plan["seed"], 42)
        self.assertTrue(plan["wait"])

    def test_invalid_grids_are_rejected_before_starting_any_job(self):
        for flags in (("--data-mode", "mismatch10"), ("--seed", "-1"),
                      ("--seeds", "1", "1"), ("--seed", "1", "2", "--resume", "old")):
            with self.subTest(flags=flags):
                argv = ["launch_tpu.py", "--config", str(CONFIG), *flags]
                with patch.object(sys, "argv", argv), patch.object(launch_tpu, "launch_job") as launch, \
                        redirect_stdout(StringIO()), patch("sys.stderr", new_callable=StringIO):
                    with self.assertRaises((ValueError, SystemExit)):
                        launch_tpu.main()
                    launch.assert_not_called()

    def test_multi_seed_suite_waits_and_stops_at_first_failure(self):
        argv = ["launch_tpu.py", "--config", str(CONFIG), "--seeds", "10", "15", "22"]
        with patch.object(sys, "argv", argv), \
                patch.object(launch_tpu, "launch_job", side_effect=[True, False]) as launch:
            with self.assertRaises(SystemExit) as error:
                launch_tpu.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual([call.args[1].seed for call in launch.call_args_list], [10, 15])
        self.assertTrue(all(call.kwargs["wait"] for call in launch.call_args_list))

    def test_waiting_launcher_observes_exit_and_finalizes_abrupt_child_failure(self):
        # Real detached stdlib children exercise wait(), exit and cross-process
        # status deduplication, without importing JAX or opening TPU devices.
        for abrupt in (False, True):
            with self.subTest(abrupt=abrupt), TemporaryDirectory() as tmp:
                config = replace(load_config(CONFIG), results=ResultsConfig(tmp))
                paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("child")
                marker = Path(tmp) / "exited"
                worker = f"""
import atexit, os, sys
from pathlib import Path
from src.config import load_config
from src.training.results import ResultsPaths
from src.training.status import RunStatus
config = load_config({str(CONFIG)!r})
paths = ResultsPaths.resolve({tmp!r}, base_dir=Path.cwd()).experiment('child')
status = RunStatus(paths, config, mode='smoke', attempt_id=sys.argv[-1])
atexit.register(lambda: Path({str(marker)!r}).write_text('exited'))
with status.monitor():
    status.record('started')
    if {abrupt!r}:
        os._exit(9)
    status.record('completed')
"""
                args = SimpleNamespace(mode="smoke", resume=False)
                command = ["nohup", sys.executable, "-u", "-c", worker]
                with patch.object(launch_tpu, "configure_worker0", return_value={}), redirect_stdout(StringIO()):
                    success = launch_tpu.launch_job(args, config, paths, command, wait=True)
                self.assertEqual(success, not abrupt)
                with (Path(tmp) / "logs/run_status.tsv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertEqual([r["event"] for r in rows], ["start", "end"])
                self.assertEqual(rows[-1]["status"], "failed" if abrupt else "completed")
                self.assertEqual(rows[0]["pid"], rows[-1]["pid"])
                if abrupt:
                    self.assertIn('"exit_code": 9', (paths.logs / "stdout.log").read_text())
                else:
                    self.assertTrue(marker.exists())
