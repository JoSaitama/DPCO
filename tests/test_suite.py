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
        self.assertEqual(plan["batch_size"], 256)
        self.assertEqual(plan["momentum"], 0.)
        self.assertEqual(plan["lr_schedule"], "constant")
        self.assertEqual(self.dry_run()["batch_size"], 8)
        self.assertEqual(self.dry_run("--batch-size", "128")["batch_size"], 128)

    def test_optimizer_options_round_trip_from_shell_to_child_and_config(self):
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run",
            "--methods", "sage_e", "sage_l", "sgd", "--seeds", "2", "5",
            "--momentum", ".9", "--nesterov", "--lr", ".1", "--lr-schedule", "cosine",
            "--min-lr", ".001", "--batch-size", "128"], cwd=gettempdir(), env=env, text=True)
        plans = json.loads(output)
        self.assertEqual(len(plans), 6)
        for plan in plans:
            command = plan["command"]
            value = lambda flag: command[command.index(flag) + 1]
            resolved = resolve_config(value("--config"), mode=value("--mode"),
                seed=int(value("--seed")), method=value("--method"),
                momentum=float(value("--momentum")), nesterov="--nesterov" in command,
                learning_rate=float(value("--lr")), lr_schedule=value("--lr-schedule"),
                min_learning_rate=float(value("--min-lr")), batch_size=int(value("--batch-size")))
            self.assertEqual(resolved.training.momentum, plan["momentum"])
            self.assertTrue(resolved.training.nesterov)
            self.assertEqual(resolved.training.learning_rate, .1)
            self.assertEqual(resolved.training.lr_schedule, "cosine")
            self.assertEqual(resolved.training.min_learning_rate, .001)
            self.assertEqual(resolved.global_batch_size, 128)
        with TemporaryDirectory() as tmp:
            custom = Path(tmp) / "optimizer.toml"
            custom.write_text(CONFIG.read_text().replace("[training]",
                '[training]\nmomentum = 0.9\nnesterov = true\nlr_schedule = "cosine"\nmin_learning_rate = 0.001'))
            plan = self.dry_run("--config", str(custom), "--mode", "train", "--momentum", "0",
                                "--no-nesterov", "--lr-schedule", "constant", "--min-lr", "0")
            self.assertFalse(plan["nesterov"])
            self.assertIn("--no-nesterov", plan["command"])
            self.assertEqual(plan["momentum"], 0.)
            self.assertEqual(plan["lr_schedule"], "constant")

    def test_optional_l2_reaches_every_child_and_explicit_zero_overrides_config(self):
        self.assertEqual(self.dry_run()["l2_coefficient"], 0.)
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run",
            "--methods", "sage_e", "sage_l", "sgd", "--l2-coefficient", "0.0001"],
            cwd=gettempdir(), env=env, text=True)
        for plan in json.loads(output):
            self.assertEqual(plan["l2_coefficient"], .0001)
            command = plan["command"]
            value = command[command.index("--l2-coefficient") + 1]
            resolved = resolve_config(CONFIG, mode="train", l2_coefficient=float(value))
            self.assertEqual(resolved.training.l2_coefficient, .0001)
        with TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.toml"
            custom.write_text(CONFIG.read_text().replace("[training]", "[training]\nl2_coefficient = 0.0001"))
            self.assertEqual(resolve_config(custom, mode="train").training.l2_coefficient, .0001)
            self.assertEqual(resolve_config(custom, mode="train", l2_coefficient=0.).training.l2_coefficient, 0.)

    def test_checkpoint_interval_round_trips_from_shell_to_every_child(self):
        self.assertEqual(self.dry_run()["checkpoint_every_steps"], 100)
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run",
            "--methods", "sage_e", "sage_l", "sgd", "--seeds", "2", "5",
            "--checkpoint-every-steps", "10"], cwd=gettempdir(), env=env, text=True)
        plans = json.loads(output)
        self.assertEqual(len(plans), 6)
        for plan in plans:
            command = plan["command"]
            self.assertEqual(plan["checkpoint_every_steps"], 10)
            value = command[command.index("--checkpoint-every-steps") + 1]
            child = resolve_config(CONFIG, mode="train", checkpoint_every_steps=int(value))
            self.assertEqual(child.checkpoints.latest_every_steps, 10)
            self.assertFalse(child.checkpoints.at_epoch_end)
        with TemporaryDirectory() as tmp:
            custom = Path(tmp) / "checkpoints.toml"
            custom.write_text(CONFIG.read_text().replace("latest_every_steps = 100", "latest_every_steps = 37"))
            self.assertEqual(resolve_config(custom, mode="train").checkpoints.latest_every_steps, 37)
            self.assertEqual(resolve_config(custom, mode="train", checkpoint_every_steps=10).checkpoints.latest_every_steps, 10)

    def test_invalid_grids_are_rejected_before_starting_any_job(self):
        for flags in (("--data-mode", "mismatch10"), ("--seed", "-1"),
                      ("--seeds", "1", "1"), ("--seed", "1", "2", "--resume", "old"),
                      ("--l2-coefficient", "-1"), ("--l2-coefficient", "nan"),
                      ("--mode", "validate", "--l2-coefficient", ".0001"),
                      ("--momentum", "1"), ("--nesterov",), ("--lr", "0"),
                      ("--min-lr", ".001"), ("--lr-schedule", "cosine", "--min-lr", "1"),
                      ("--mode", "validate", "--momentum", ".9"),
                      ("--checkpoint-every-steps", "0"), ("--checkpoint-every-steps", "-1"),
                      ("--checkpoint-every-steps", "2.5"),
                      ("--eval-every-steps", "2", "--eval-every-epochs", "1"),
                      ("--eval-every-steps", "0"), ("--eval-every-steps", "-1"),
                      ("--eval-every-steps", "2.5"), ("--eval-every-epochs", "0"),
                      ("--eval-every-epochs", "-1"), ("--eval-every-epochs", "nan"),
                      ("--eval-every-epochs", "inf")):
            with self.subTest(flags=flags):
                argv = ["launch_tpu.py", "--config", str(CONFIG), *flags]
                with patch.object(sys, "argv", argv), patch.object(launch_tpu, "launch_job") as launch, \
                        redirect_stdout(StringIO()), patch("sys.stderr", new_callable=StringIO):
                    with self.assertRaises((ValueError, SystemExit)):
                        launch_tpu.main()
                    launch.assert_not_called()

    def test_evaluation_overrides_round_trip_through_shell_and_child(self):
        default = self.dry_run()
        self.assertEqual((default["evaluation_mode"], default["evaluation_interval"]), ("epochs", 1.))
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        for flag, value, mode in (("--eval-every-steps", "10", "steps"),
                                  ("--eval-every-epochs", "0.25", "epochs")):
            with self.subTest(flag=flag):
                output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run",
                    "--methods", "sage_e", "sage_l", "sgd", "--seeds", "2", "5",
                    "--checkpoint-every-steps", "17", flag, value], cwd=gettempdir(), env=env, text=True)
                plans = json.loads(output)
                self.assertEqual(len(plans), 6)
                for plan in plans:
                    self.assertEqual((plan["evaluation_mode"], plan["evaluation_interval"]), (mode, float(value)))
                    self.assertEqual(plan["checkpoint_every_steps"], 17)
                    command = plan["command"]
                    child_value = command[command.index(flag) + 1]
                    options = ({"eval_every_steps": int(child_value)} if mode == "steps"
                               else {"eval_every_epochs": float(child_value)})
                    child = resolve_config(CONFIG, mode="train", checkpoint_every_steps=17, **options)
                    self.assertEqual((child.evaluation.mode, child.evaluation.interval), (mode, float(value)))
                    other = "--eval-every-epochs" if mode == "steps" else "--eval-every-steps"
                    self.assertNotIn(other, command)
        with TemporaryDirectory() as tmp:
            custom = Path(tmp) / "evaluation.toml"
            custom.write_text(CONFIG.read_text().replace('[evaluation]\nmode = "epochs"\ninterval = 1.0',
                '[evaluation]\nmode = "steps"\ninterval = 37'))
            self.assertEqual(resolve_config(custom, mode="train").evaluation.interval, 37)
            changed = resolve_config(custom, mode="train", eval_every_epochs=.5)
            self.assertEqual((changed.evaluation.mode, changed.evaluation.interval), ("epochs", .5))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            resolve_config(CONFIG, mode="train", eval_every_steps=2, eval_every_epochs=1.)

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
