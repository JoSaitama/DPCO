"""Offline tests for plotting selection and units; no training dependencies."""

import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dpco_training_plots", ROOT/"scripts/plot_training_results.py")
plots = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plots
spec.loader.exec_module(plots)


def config(schedule="constant"):
    return dict(data={"name": "imagenet100"}, model={"name": "resnet50"}, seed=5, global_batch_size=256,
        method={"name": "dpco_policy", "policy_mode": "half"}, reference={"name": "error"},
        training={"learning_rate": .01, "total_steps": 4, "lr_schedule": schedule})


def rows(schedule="constant"):
    return [dict(step=t+1, samples_seen=(t+1)*256, epoch_progress=(t+1)/10,
        learning_rate=.01 if schedule == "constant" else .005*(1+math.cos(math.pi*t/3))
            if schedule == "cosine" else .01/math.sqrt(t+1),
        valid_batch_size=256, train_ce=4.-t/10, train_top1=t/10, elapsed_wall_s=t+1.) for t in range(4)]


class TrainingPlotTests(unittest.TestCase):
    def warmup_fixture(self, family, percent, rate=.01, name=None):
        c = config()
        c["method"] = {"name": "sgd"} if family == "SGD" else {
            "name": "dpco", "dpco_warmup_percent": percent}
        c["reference"]["name"] = "error" if family == "DPCO-E" else "loss"
        c["training"]["learning_rate"] = rate
        family, mode, first = plots.method_identity(c)
        return plots.Run(Path(name or f"{family}_{percent}_{rate}"), Path("meta"),
                         {"config": c}, rows(), family, mode, first, "constant", rate, "completed")

    def test_warmup_comparisons_keep_rate_and_percentage_separate(self):
        runs = [self.warmup_fixture(f, p, rate)
                for rate in (.01, .05) for f, p in
                (("SGD", 0), ("DPCO-E", 10), ("DPCO-L", 10))]
        runs += [self.warmup_fixture(f, 30) for f in ("DPCO-E", "DPCO-L")]
        selected, missing = plots.select_group(runs, "warmup10", [.01, .05])
        self.assertEqual(len(selected), 6)
        self.assertFalse(missing)
        self.assertFalse(any(r.mode == "warmup30" for r in selected))
        selected, missing = plots.select_group(runs, "warmup_percent", [.01, .05])
        self.assertEqual(len(selected), 5)
        self.assertFalse(missing)
        self.assertEqual({r.initial_lr for r in selected}, {.01})
        self.assertEqual(sum(r.family == "SGD" for r in selected), 1)
        _, missing = plots.select_group(runs, "warmup_percent", [.01, .05], warmup_comparison_lr=.05)
        self.assertEqual(missing, [("DPCO-E", .05, "warmup30"), ("DPCO-L", .05, "warmup30")])
        with self.assertRaisesRegex(ValueError, "Ambiguous repeated"):
            plots.select_group(runs + [copy.copy(runs[-1])], "warmup_percent", [.01, .05])

    def test_warmup_plot_styles_and_sgd_baseline(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        runs = [self.warmup_fixture(f, p) for f, p in
                (("SGD", 0), ("DPCO-E", 10), ("DPCO-E", 30), ("DPCO-L", 10), ("DPCO-L", 30))]
        args = SimpleNamespace(lrs=[.01, .05], smooth_window=1, x_axis="step", max_step=None,
            formats=["png"], dpi=70, panel_height_ratio=.8, line_width=1.05, dash_lighten=.2,
            warmup_percents=[10., 30.], warmup_comparison_lr=.01)
        with TemporaryDirectory() as tmp, patch.object(plt, "close"):
            item = plots.plot_group("warmup_percent", runs, [], args, Path(tmp),
                                    {"constant": ((0, 5), (0, 40))})
            fig = plt.gcf()
            self.assertEqual([line.get_linestyle() for line in fig.axes[0].lines], ["-", "--", "-", "--", "-"])
            self.assertEqual(fig.axes[0].lines[0].get_color(), plots.curve_color("SGD", False, .2))
            labels = [t.get_text() for t in fig.legends[0].texts]
            self.assertEqual(labels, ["SGD", "DPCO-E", "DPCO-L", "SGD prefix 10%", "SGD prefix 30%"])
            self.assertEqual(item["style_dimension"], "sgd_prefix_percent")
            self.assertEqual(item["fixed_initial_lr"], .01)
            self.assertEqual(len(item["run_ids"]), 5)
        plt.close(fig)

    def test_warmup_policy_switch_uses_recorded_percentage(self):
        for percent in (10, 30):
            with TemporaryDirectory() as tmp:
                run = self.warmup_fixture("DPCO-E", percent)
                run.directory = Path(tmp)
                prefix = math.ceil(len(run.rows)*percent/100)
                policy = [dict(step=t+1, policy_alpha=0 if t < prefix else 1,
                               executed_method="sgd" if t < prefix else "dpco") for t in range(4)]
                plots.write_csv(run.directory/"policy_steps.csv", policy, tuple(policy[0]))
                plots.validate_policy(run)
                policy[prefix]["executed_method"] = "sgd"
                plots.write_csv(run.directory/"policy_steps.csv", policy, tuple(policy[0]))
                with self.assertRaisesRegex(ValueError, "policy mismatch"):
                    plots.validate_policy(run)

    def test_clean_figure_geometry_sparse_dashes_and_no_explanatory_text(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        run = plots.Run(Path("one"), Path("meta"), {"config": config()}, rows(),
                        "DPCO-E", "half", None, "constant", .01, "completed")
        args = SimpleNamespace(lrs=[.01, .05], smooth_window=1, x_axis="step", max_step=None,
            formats=["png"], dpi=70, panel_height_ratio=.8, line_width=1.05, dash_lighten=.2)
        with TemporaryDirectory() as tmp, patch.object(plt, "close"):
            plots.plot_group("half", [run], [], args, Path(tmp),
                             {"constant": ((0, 5), (0, 40))})
            fig = plt.gcf()
            fig.canvas.draw()
            self.assertEqual(fig.texts, [])
            self.assertEqual(len(fig.axes), 2)
            for ax in fig.axes:
                box = ax.get_window_extent()
                self.assertAlmostEqual(box.height/box.width, .8, places=10)
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(ax.lines[0].get_linewidth(), 1.05)
                self.assertEqual(ax.lines[0].get_linestyle(), "--")
                self.assertEqual(ax.lines[0].get_color(), plots.curve_color("DPCO-E", True, .2))
            legend = fig.legends[0].get_window_extent()
            for ax in fig.axes:
                self.assertLess(legend.y1, ax.xaxis.label.get_window_extent().y0)
        plt.close(fig)
        self.assertEqual(plots.learning_rate_styles([.05, .01])[.01], (0, (8, 5)))

    def test_causal_smoothing_growing_prefix_and_raw_option(self):
        values = [1., 2., 3., 100.]
        self.assertEqual(plots.trailing_mean(values, 2), [1., 1.5, 2.5, 51.5])
        self.assertEqual(plots.trailing_mean(values, 1), values)
        self.assertEqual(plots.trailing_mean(values[:3], 2), plots.trailing_mean(values, 2)[:3])

    def test_metadata_method_identity_and_alternating_first(self):
        c = config()
        self.assertEqual(plots.method_identity(c), ("DPCO-E", "half", None))
        c["method"].update(policy_mode="alternating", alternating_first="dpco")
        self.assertEqual(plots.method_identity(c), ("DPCO-E", "alternating", "dpco"))
        c["method"]["name"] = "dpco"
        c["method"]["policy_mode"] = "continuous"
        c["reference"]["name"] = "loss"
        self.assertEqual(plots.method_identity(c), ("DPCO-L", "continuous", None))
        # A uniform-direction method with an Adam updater must never be labeled SGD.
        c["training"]["optimizer"] = "adam"
        c["method"]["name"] = "sgd"
        with self.assertRaisesRegex(ValueError, "SGD runs only"):
            plots.method_identity(c)

    def test_recorded_rates_and_accuracy_units_are_validated(self):
        for schedule in ("constant", "cosine", "inverse_sqrt"):
            plots.validate_training(rows(schedule), config(schedule), "fixture")
        bad = rows(); bad[1]["learning_rate"] = .02
        with self.assertRaisesRegex(ValueError, "recorded LR"):
            plots.validate_training(bad, config(), "fixture")
        bad = rows(); bad[1]["train_top1"] = 42
        with self.assertRaisesRegex(ValueError, "accuracy fraction"):
            plots.validate_training(bad, config(), "fixture")

    def test_duplicate_runs_fail_instead_of_selecting_best_or_latest(self):
        r = plots.Run(Path("one"), Path("meta"), {"config": config()}, rows(), "DPCO-E", "half", None, "constant", .01, "completed")
        other = copy.copy(r); other.directory = Path("two")
        with self.assertRaisesRegex(ValueError, "Ambiguous repeated"):
            plots.select_group([r, other], "half", [.01])
        selected, missing = plots.select_group([r], "half", [.01])
        self.assertEqual(selected, [r])
        self.assertEqual(missing, [("SGD", .01), ("DPCO-L", .01)])

    def test_cumulative_lr_uses_recorded_schedule_and_preupdate_position(self):
        r = plots.Run(Path("one"), Path("meta"), {"config": config()}, rows(), "DPCO-E", "half", None, "constant", .01, "completed")
        self.assertEqual(plots.x_values(r, "cumulative_eta_before"), [0., .01, .02, .03])

    def test_split_export_join_and_policy_validation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, logs = root/"files_new/run_a", root/"logs_new/run_a"
            data.mkdir(parents=True); logs.mkdir(parents=True)
            plots.write_csv(data/"train_steps.csv", rows(), plots.TRAIN_COLUMNS)
            plots.write_csv(data/"summary.csv", [{"status": "completed"}], ["status"])
            policy = [dict(step=i+1, policy_alpha=.5, executed_method="half_dpco") for i in range(4)]
            plots.write_csv(data/"policy_steps.csv", policy, tuple(policy[0]))
            (logs/"metadata.json").write_text(json.dumps({"config": config()}))
            args = SimpleNamespace(run_ids=None, dataset="imagenet100", model="resnet50", seed=5,
                                   batch_size=256, steps=4, lrs=[.01])
            found, excluded = plots.discover_runs(root, args)
            self.assertEqual(len(found), 1); self.assertFalse(excluded)
            policy[2]["policy_alpha"] = 1
            plots.write_csv(data/"policy_steps.csv", policy, tuple(policy[0]))
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                plots.discover_runs(root, args)


if __name__ == "__main__":
    unittest.main()
