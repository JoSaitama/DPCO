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
            child = RunStatus(paths, config, mode="smoke", attempt_id=parent.attempt_id)
            with child.monitor():
                child.record("started")
                child.record("start")
                child.record("train")
                child.record("completed")
            resumed = RunStatus(paths, config, mode="smoke", resume=True)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with resumed.monitor(), resumed.monitor():
                    resumed.record("started")
                    raise RuntimeError("fixture failure\nsecond line\tcolumn")
            rows = read_status(parent.path)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["event"] for row in rows], ["start", "end", "start", "end"])
            self.assertEqual({row["attempt_id"] for row in rows[:2]}, {parent.attempt_id})
            self.assertNotEqual(parent.attempt_id, resumed.attempt_id)
            self.assertEqual(rows[1]["status"], "completed")
            self.assertEqual(rows[-1]["status"], "failed")
            self.assertNotIn("step", rows[-1])
            self.assertEqual(rows[-1]["resume"], "True")
            self.assertEqual(rows[-1]["reference"], "")
            self.assertNotIn("error", rows[-1])
            self.assertEqual(len(parent.path.read_text().splitlines()), 5)

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
    status.record('train')
status.record('completed')
"""
        with TemporaryDirectory() as tmp:
            processes = [subprocess.Popen([sys.executable, "-c", script, str(CONFIG), tmp, str(seed)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for seed in range(3)]
            for process in processes:
                output, error = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, output + error)
            path = Path(tmp) / "logs/run_status.tsv"
            rows = read_status(path)
            self.assertEqual(len(rows), 6)
            self.assertEqual(list(rows[0]), list(STATUS_FIELDS))
            for seed in range(3):
                selected = [row for row in rows if row["seed"] == str(seed)]
                self.assertEqual(len(selected), 2)
                self.assertEqual(selected[-1]["status"], "completed")
                self.assertEqual([row["event"] for row in selected], ["start", "end"])

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
                self.assertEqual([row["event"] for row in rows], ["start", "end"])
                self.assertEqual(rows[-1]["attempt_id"], "fixture")
                self.assertEqual(rows[-1]["status"], "failed")
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
            self.assertEqual([row["event"] for row in rows], ["start", "end"])
            self.assertEqual(rows[0]["attempt_id"], rows[-1]["attempt_id"])
            self.assertEqual(rows[-1]["status"], "failed")

            logs = list((Path(tmp) / "logs").glob("*/stdout.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("fixture spawn", logs[0].read_text())

    def test_success_waits_for_cleanup_and_cleanup_failure_has_only_two_rows(self):
        with TemporaryDirectory() as tmp:
            status = RunStatus(ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("cleanup"),
                               load_config(CONFIG), mode="smoke")
            with self.assertRaisesRegex(RuntimeError, "cleanup"), status.monitor():
                status.record("started")
                with status.monitor():
                    status.record("completed")
                self.assertIsNone(status.outcome())
                raise RuntimeError("cleanup failure")
            rows = read_status(status.path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["status"], "failed")

    def test_parent_can_finalize_dead_child_without_duplicating_its_start(self):
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("killed")
            config = load_config(CONFIG)
            child = RunStatus(paths, config, mode="smoke")
            child.record("started", pid=12345)
            parent = RunStatus(paths, config, mode="smoke", attempt_id=child.attempt_id)
            parent.record("failed", pid=12345)
            parent.record("failed", pid=12345)
            rows = read_status(child.path)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["pid"] == "12345" for row in rows))
            self.assertEqual(parent.outcome(), "failed")

    def test_incompatible_status_schema_is_rejected_without_backup_or_rewrite(self):
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("new")
            paths.parent.create()
            path = paths.parent.logs / "run_status.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(*STATUS_FIELDS, "step", "error"), delimiter="\t")
                writer.writeheader()
                writer.writerow(dict(attempt_id="old", run_name="old", seed=0,
                                     event="train", step=3, status="running", error="old detail"))
            original = path.read_bytes()
            status = RunStatus(paths, load_config(CONFIG), mode="smoke")
            with self.assertRaisesRegex(ValueError, "Remove the old smoke-test TSV"):
                status.record("started")
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(path.parent.glob("run_status.legacy-*.tsv")))
            path.unlink()
            status.record("started")
            status.record("completed")
            rows = read_status(path)
            self.assertEqual(list(rows[0]), list(STATUS_FIELDS))
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["event"] for r in rows], ["start", "end"])
