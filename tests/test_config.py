from dataclasses import replace
from pathlib import Path
import unittest

from src.config import (METHODS, EvaluationConfig, LoggingConfig, MethodConfig,
                        ProbeConfig, ResultsConfig, load_config)


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_model_dataset_presets(self):
        paths = sorted((ROOT / "configs").glob("*.toml"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertIn(config.data.num_classes, (100, 1000))
                self.assertEqual(config.results.root_dir, "results")
                self.assertEqual(config.global_batch_size, 256)

    def test_dpco_e_presets_match_sgd_comparison_settings(self):
        for path in sorted((ROOT / "configs").glob("*_dpco_e.toml")):
            with self.subTest(path=path.name):
                dpco = load_config(path)
                sgd = load_config(path.with_name(path.name.replace("_dpco_e", "")))
                self.assertEqual(dpco.method.name, "dpco")
                self.assertEqual(dpco.reference.name, "error")
                self.assertEqual(sgd.method.name, "sgd")
                for field in ("data", "model", "training", "results", "seed", "global_batch_size"):
                    self.assertEqual(getattr(dpco, field), getattr(sgd, field))

    def test_results_root_cannot_be_empty(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        with self.assertRaises(ValueError):
            replace(config, results=ResultsConfig("  ")).validate()

    def test_recording_schedules_reject_invalid_intervals(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        for logging in (LoggingConfig("steps", .5), LoggingConfig("percent", 101),
                        LoggingConfig(csv_flush_steps=0)):
            with self.subTest(logging=logging), self.assertRaises(ValueError):
                replace(config, logging=logging).validate()
        with self.assertRaises(ValueError):
            replace(config, evaluation=EvaluationConfig("epochs", 0)).validate()

    def test_common_reference_probes_do_not_silently_cover_sgd_or_loo(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        for method in ("sgd", "dpco_loo"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                replace(config, method=MethodConfig(method), probes=ProbeConfig(enabled=True)).validate()

    def test_probe_parameters_are_validated_even_when_disabled(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        for probes in (ProbeConfig(fractions=(0., 1.1)), ProbeConfig(lr_scales=(0.,)),
                       ProbeConfig(batches_per_node=0)):
            with self.subTest(probes=probes), self.assertRaises(ValueError):
                replace(config, probes=probes).validate()

    def test_data_mode_is_safe_and_remains_a_label(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        replace(config, data=replace(config.data, mode="mismatch10")).validate()
        with self.assertRaises(ValueError):
            replace(config, data=replace(config.data, mode="../outside")).validate()

    def test_rejects_head_dataset_class_mismatch(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        config = replace(config, data=replace(config.data, num_classes=1000))
        with self.assertRaises(ValueError):
            config.validate()

    def test_ablation_probability_is_bounded(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        config = replace(config, method=MethodConfig("random_flip", 1.1))
        with self.assertRaises(ValueError):
            config.validate()

    def test_method_changes_without_changing_data_or_model(self):
        config = load_config(ROOT / "configs/imagenet100_resnet50.toml")
        for name in METHODS:
            candidate = replace(config, method=MethodConfig(name))
            if name == "adam":
                candidate = replace(candidate, training=replace(candidate.training, optimizer="adam"))
            candidate.validate()
            self.assertEqual(candidate.data, config.data)
            self.assertEqual(candidate.model, config.model)
