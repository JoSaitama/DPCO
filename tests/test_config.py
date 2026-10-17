from dataclasses import replace
from pathlib import Path
import unittest

from src.config import METHODS, MethodConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_model_dataset_presets(self):
        paths = sorted((ROOT / "configs").glob("*.toml"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertIn(config.data.num_classes, (100, 1000))

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
            candidate.validate()
            self.assertEqual(candidate.data, config.data)
            self.assertEqual(candidate.model, config.model)
