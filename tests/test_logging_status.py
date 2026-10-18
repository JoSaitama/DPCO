"""Status lifecycle, shared-file concurrency, and readable JSON contracts."""

from contextlib import redirect_stdout
import csv
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.config import ResultsConfig, load_config
from src.training.recording import RunRecorder
from src.training.results import ResultsPaths
from src.training.status import RunStatus, STATUS_FIELDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml"


def read_status(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class LoggingStatusTests(unittest.TestCase):
    def test_json_order_and_lr_alias_match_console_and_persistent_log(self):
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("ordered")
            paths.create()
            recorder = RunRecorder(paths)
            output = StringIO()
            with redirect_stdout(output):
                recorder.log("train", elapsed_wall_s=1., learning_rate=.01,
                             train_ce=.5, step=1, train_top1=.25, samples_seen=8)
                recorder.log("validation", elapsed_wall_s=2., val_ce=.4, step=1)
                recorder.log("start", run_name=paths.name, step=0)
            recorder.close()
            self.assertEqual(output.getvalue(), (paths.logs / "train.log").read_text())
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(list(rows[0])[:4], ["step", "event", "train_ce", "lr"])
            self.assertEqual(rows[0]["lr"], .01)
            self.assertNotIn("learning_rate", rows[0])
            for row in rows:
                self.assertEqual(list(row)[:2], ["step", "event"])
            self.assertNotIn("train_ce", rows[1])
            self.assertNotIn("lr", rows[1])

    def test_launch_resume_and_bootstrap_failure_keep_attempt_history(self):
        with TemporaryDirectory() as tmp:
            config = load_config(CONFIG)
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("history")
            parent = RunStatus(paths, config, mode="smoke")
            parent.record("launching", pid="")
            child = RunStatus(paths, config, mode="smoke", attempt_id=parent.attempt_id)
            with child.monitor():
                child.record("started")
                child.record("start", step=0)
                child.record("train", step=3)
                child.record("completed", step=3)
            resumed = RunStatus(paths, config, mode="smoke", resume=True)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with resumed.monitor(), resumed.monitor():
                    resumed.record("started")
                    raise RuntimeError("fixture failure\nsecond line\tcolumn")
            rows = read_status(parent.path)
            self.assertEqual(len(rows), 7)
            self.assertEqual({row["attempt_id"] for row in rows[:5]}, {parent.attempt_id})
            self.assertNotEqual(parent.attempt_id, resumed.attempt_id)
            self.assertEqual(rows[4]["status"], "completed")
            self.assertEqual(rows[-1]["status"], "failed")
            self.assertEqual(rows[-1]["step"], "")
            self.assertEqual(rows[-1]["resume"], "True")
            self.assertEqual(rows[-1]["reference"], "")
            self.assertIn("fixture failure second line column", rows[-1]["error"])
            self.assertEqual(len(parent.path.read_text().splitlines()), 8)

    def test_independent_seed_processes_append_one_header_without_lost_rows(self):
        script = """
from dataclasses import replace
from pathlib import Path
import sys
from src.config import load_config
from src.training.results import ResultsPaths
from src.training.status import RunStatus
config = replace(load_config(sys.argv[1]), seed=int(sys.argv[3]))
paths = ResultsPaths.resolve(sys.argv[2], base_dir=Path.cwd()).experiment('seed' + sys.argv[3])
status = RunStatus(paths, config, mode='smoke')
status.record('started')
for step in range(10):
    status.record('train', step=step)
status.record('completed', step=10)
"""
        with TemporaryDirectory() as tmp:
            processes = [subprocess.Popen([sys.executable, "-c", script, str(CONFIG), tmp, str(seed)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for seed in range(3)]
            for process in processes:
                output, error = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, output + error)
            path = Path(tmp) / "logs/run_status.tsv"
            rows = read_status(path)
            self.assertEqual(len(rows), 36)
            self.assertEqual(list(rows[0]), list(STATUS_FIELDS))
            for seed in range(3):
                selected = [row for row in rows if row["seed"] == str(seed)]
                self.assertEqual(len(selected), 12)
                self.assertEqual(selected[-1]["status"], "completed")
                self.assertEqual(selected[-1]["step"], "10")

    def test_child_records_failures_before_the_main_recorder_exists(self):
        from scripts import train_classification
        for stage in ("runtime", "data"):
            with self.subTest(stage=stage), TemporaryDirectory() as tmp:
                config = replace(load_config(CONFIG), results=ResultsConfig(tmp))
                argv = ["train_classification.py", "--config", str(CONFIG),
                        "--mode", "smoke", "--run-name", "bootstrap", "--attempt-id", "fixture"]
                with patch.object(sys, "argv", argv), \
                        patch.object(train_classification, "resolve_config", return_value=config), \
                        patch.object(train_classification, "configure_worker0", return_value={}), \
                        patch.object(train_classification, "require_local_tpu",
                            side_effect=RuntimeError("fixture runtime") if stage == "runtime" else None,
                            return_value=[]), \
                        patch.object(train_classification, "run", side_effect=RuntimeError("fixture data")), \
                        redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "fixture"):
                    train_classification.main()
                rows = read_status(Path(tmp) / "logs/run_status.tsv")
                self.assertEqual([row["event"] for row in rows], ["started", "failed"])
                self.assertEqual(rows[-1]["attempt_id"], "fixture")
                self.assertIn("fixture " + stage, rows[-1]["error"])
                self.assertFalse((Path(tmp) / "logs/bootstrap/metadata.json").exists())

    def test_launcher_records_spawn_failure_without_starting_accelerators(self):
        from scripts import launch_tpu
        with TemporaryDirectory() as tmp:
            config = replace(load_config(CONFIG), results=ResultsConfig(tmp))
            argv = ["launch_tpu.py", "--config", str(CONFIG), "--mode", "smoke"]
            with patch.object(sys, "argv", argv), \
                    patch.object(launch_tpu, "resolve_config", return_value=config), \
                    patch.object(launch_tpu, "configure_worker0", return_value={}), \
                    patch.object(launch_tpu.shutil, "which", return_value="/usr/bin/nohup"), \
                    patch.object(launch_tpu.subprocess, "Popen", side_effect=OSError("fixture spawn")), \
                    self.assertRaisesRegex(OSError, "fixture spawn"):
                launch_tpu.main()
            rows = read_status(Path(tmp) / "logs/run_status.tsv")
            self.assertEqual([row["event"] for row in rows], ["launching", "failed"])
            self.assertEqual(rows[0]["attempt_id"], rows[-1]["attempt_id"])
            self.assertEqual(rows[0]["pid"], "")
