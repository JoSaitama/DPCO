"""Offline numerical and rendering checks for the explicit three-run plotter."""

import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dpco_comparison_plots", ROOT / "plots/plot_dpco_training.py")
plots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plots
SPEC.loader.exec_module(plots)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(directory, method=None):
    directory.mkdir(parents=True)
    rows = [dict(step=i + 1, train_ce=ce, train_top1=acc, valid_batch_size=count,
                 elapsed_wall_s=0.)  # Dispatch timestamps must not drive time plots.
            for i, (ce, acc, count) in enumerate(zip(
                [8., 4., 6., 2., 4., 0.], [0., .25, .5, .75, 1., 1.], [1, 3, 2, 2, 1, 1]))]
    write_csv(directory / "train_steps.csv", rows)
    timing = [dict(step=0, event="compile_warmup", duration_s=9., window_steps="", elapsed_wall_s=9.)]
    timing += [dict(step=end, event="train_window", duration_s=duration, window_steps=count, elapsed_wall_s=wall)
               for end, count, duration, wall in [(2, 2, .4, 10.), (3, 1, .5, 10.7), (6, 3, .6, 12.)]]
    write_csv(directory / "timing.csv", timing)
    if method is not None:
        config = {"training": {"optimizer": "adam", "learning_rate": .0003},
            "method": {"name": "adam" if method == "adam" else "dpco",
                       "negative_weight": 0 if method == "filter" else -1},
            "reference": {"name": "loss" if method == "adam" else "ema", "beta": .99}}
        path = directory.parent.parent / "logs" / directory.name
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({"config": config}))
    return rows, timing


class DPCOTrainingPlotTests(unittest.TestCase):
    def test_weighted_smoothing_complete_windows_and_time_units(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            fixture(directory)
            run = plots.load_run("adam", directory)
            steps, timing = plots.curves(run, 2)
            self.assertAlmostEqual(steps[1]["smooth_ce"], 5.)
            self.assertAlmostEqual(steps[2]["smooth_ce"], 4.8)
            # Window at step 3 needs two whole windows to cover >=2 updates.
            self.assertEqual(timing[1]["smoothing_steps"], 3)
            self.assertAlmostEqual(timing[1]["smooth_ce"], 32 / 6)
            self.assertAlmostEqual(timing[1]["smooth_ms_per_step"], 300.)
            self.assertAlmostEqual(timing[-1]["raw_ce"], 2.)
            self.assertAlmostEqual(timing[-1]["raw_accuracy_percent"], 87.5)
            self.assertAlmostEqual(timing[-1]["wall_minutes"], 12 / 60)
            self.assertAlmostEqual(timing[-1]["active_minutes"], 1.5 / 60)
            self.assertAlmostEqual(timing[-1]["raw_ms_per_step"], 200.)
            # Changing future batches cannot alter a trailing average's prefix.
            old = steps[:3]
            run.train[-1]["ce"] = 1000.
            self.assertEqual(plots.curves(run, 2)[0][:3], old)

    def test_window_one_leaves_raw_data_and_weighted_cost_unchanged(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            fixture(directory)
            run = plots.load_run("adam", directory)
            steps, timing = plots.curves(run, 1)
            for row in steps:
                self.assertAlmostEqual(row["raw_ce"], row["smooth_ce"])
                self.assertAlmostEqual(row["raw_accuracy_percent"], row["smooth_accuracy_percent"])
            for row in timing:
                self.assertAlmostEqual(row["raw_ms_per_step"], row["smooth_ms_per_step"])
                self.assertAlmostEqual(row["raw_ce"], row["smooth_ce"])
            # One long smoothing window must weight durations by update count.
            _, smoothed = plots.curves(run, 100)
            self.assertAlmostEqual(smoothed[-1]["smooth_ms_per_step"], 250.)
            self.assertNotAlmostEqual(smoothed[-1]["smooth_ms_per_step"], (200 + 500 + 200) / 3)

    def test_incomplete_overlapping_or_backward_timing_is_rejected(self):
        for alteration in ("gap", "overlap", "missing", "backward", "nonfinite"):
            with self.subTest(alteration=alteration), TemporaryDirectory() as tmp:
                directory = Path(tmp) / "run"
                _, timing = fixture(directory)
                if alteration == "gap":
                    timing[-1]["window_steps"] = 2
                elif alteration == "overlap":
                    timing[-1]["window_steps"] = 4
                elif alteration == "missing":
                    timing.pop()
                elif alteration == "backward":
                    timing[-1]["elapsed_wall_s"] = 10.
                else:
                    timing[-1]["duration_s"] = "nan"
                write_csv(directory / "timing.csv", timing)
                with self.assertRaises(ValueError):
                    plots.load_run("adam", directory)

    def test_percentage_units_step_gaps_and_method_mislabel_are_rejected(self):
        for alteration in ("percent", "gap", "method"):
            with self.subTest(alteration=alteration), TemporaryDirectory() as tmp:
                directory = Path(tmp) / "files/run"
                rows, _ = fixture(directory, "filter")
                if alteration == "percent":
                    rows[1]["train_top1"] = 25
                elif alteration == "gap":
                    rows[1]["step"] = 3
                write_csv(directory / "train_steps.csv", rows)
                with self.assertRaises(ValueError):
                    plots.load_run("adam", directory)

    def test_five_figures_manifest_source_integrity_and_output_protection(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = {method: root / "results/files" / method for method in plots.STYLES}
            for method, path in directories.items():
                fixture(path, method)
            inputs = {p: p.read_bytes() for p in root.rglob("*.csv")}
            args = [item for method, path in directories.items() for item in (f"--{method}-dir", str(path))]
            args += ["--smooth-window", "2", "--output-name", "comparison", "--dpi", "70"]
            with patch.object(plots, "ROOT", root):
                output = plots.main(args)
                self.assertEqual(output, root / "results/figures/comparison")
                self.assertFalse((root / "results/files/plots").exists())
                self.assertEqual(len(list(output.glob("*.png"))), 5)
                self.assertEqual(len(list(output.glob("*.pdf"))), 5)
                manifest = json.loads((output / "plot_manifest.json").read_text())
                self.assertFalse(manifest["comparison_notes"])
                self.assertAlmostEqual(manifest["runs"][0]["weighted_mean_ms_per_step"], 250.)
                for path in output.glob("*.png"):
                    self.assertEqual(plt.imread(path).shape[:2], (280, 350))
                for path in output.glob("*.pdf"):
                    self.assertTrue(path.read_bytes().startswith(b"%PDF"))
                with self.assertRaises(SystemExit), patch("sys.stderr", StringIO()):
                    plots.main(args)
            self.assertTrue(all(path.read_bytes() == original for path, original in inputs.items()))

    def test_axes_ratio_colors_accuracy_and_active_time(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            fixture(directory)
            run = plots.load_run("adam", directory)
            data = {method: plots.curves(run, 1) for method in plots.STYLES}
            fig = plots.make_figure(data, "time", "accuracy", None, "Train accuracy (%)",
                                    time_axis="active", show_raw=False, plt=plt)
            try:
                fig.canvas.draw()
                ax = fig.axes[0]
                box = ax.get_window_extent()
                self.assertAlmostEqual(box.height / box.width, .8)
                self.assertEqual([line.get_color() for line in ax.lines], ["#1f77b4", "#d62728", "#ff7f0e"])
                self.assertEqual([line.get_linestyle() for line in ax.lines], ["-", "-", "--"])
                self.assertAlmostEqual(ax.lines[0].get_xdata()[-1], 1.5 / 60)
                self.assertAlmostEqual(ax.lines[0].get_ydata()[-1], 87.5)
                self.assertEqual(tuple(ax.get_ylim()), (0, 100))
            finally:
                plt.close(fig)

    def test_help_needs_neither_training_runtime_nor_matplotlib(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "plots/plot_dpco_training.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--smooth-window", result.stdout)

    def test_comparison_discloses_mismatched_data_and_settings(self):
        with TemporaryDirectory() as tmp:
            runs = []
            for method in plots.STYLES:
                path = Path(tmp) / "results/files" / method
                fixture(path, method)
                run = plots.load_run(method, path)
                run.metadata["experiment"] = {"data_fingerprint": method}
                runs.append(run)
            runs[-1].metadata["config"]["training"]["learning_rate"] = .001
            notes = plots.comparison_notes(runs)
            self.assertTrue(any("fingerprints differ" in note for note in notes))
            self.assertTrue(any("training" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
