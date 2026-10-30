"""Keep model names faithful to stem/block choices and exercise CIFAR50 entry paths."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from src.models.classification import create_model
from src.models.cifar_resnet import CifarResNet
from src.models.resnet import ResNet
from src.training.checkpoints import CheckpointStore
from src.training.launch import resolve_config
from tests.test_cifar100n import PRESET, config, fixture
from tests.test_dpco_history import assert_tree


class ModelSelectionTests(unittest.TestCase):
    def test_existing_and_native_50_have_distinct_stems_and_bottleneck_heads(self):
        for name, kernel in (("resnet50", 7), ("cifar_resnet50", 3)):
            c = resolve_config(PRESET, mode="train", model=name)
            self.assertEqual(c.model.name, name)
            model = create_model(name, 100)
            shapes = jax.eval_shape(lambda k: model.init(k, jnp.zeros((1, 32, 32, 3)), train=True),
                                    jax.random.key(0))["params"]
            self.assertEqual(shapes["conv_init"]["kernel"].shape, (kernel, kernel, 3, 64))
            self.assertEqual(shapes["head"]["kernel"].shape, (2048, 100))
        with self.assertRaises(ValueError):
            create_model("unregistered", 100)

    def test_both_50_entry_paths_train_and_recover_adam_reference_and_bn(self):
        from scripts.train_classification import run
        with TemporaryDirectory() as tmp:
            fixture(Path(tmp) / "data")
            for name, target, model in (
                    ("resnet50", "src.models.resnet.create_resnet", ResNet((1,), 100, num_filters=2)),
                    ("cifar_resnet50", "src.models.cifar_resnet.create_cifar_resnet",
                     CifarResNet((1,), 100, num_filters=2, bottleneck=True))):
                c = config(tmp)
                c = replace(c, model=replace(c.model, name=name))
                c.validate()
                args = SimpleNamespace(mode="train", run_name=name, resume=False, reserved=False,
                                       resume_slot="latest")
                # Real entry/BN/gradient/EMA/Adam/checkpoints, reduced backbone cost only.
                with patch(target, return_value=model), redirect_stdout(StringIO()):
                    run(args, c, jax.local_devices(), status=None)
                    store = CheckpointStore(Path(tmp) / "results/runs" / name)
                    expected = store.load("final")["train_state"]
                    args.resume, args.resume_slot = True, "p50"
                    run(args, c, jax.local_devices(), status=None)
                    actual = CheckpointStore(store.directory).load("final")["train_state"]
                assert_tree(self, actual, expected, exact=True)
                self.assertEqual(int(actual.method_state.updates), 2)
                self.assertEqual(int(actual.optimizer_state.count), 2)
                self.assertIn("batch_stats", actual.model_state)


if __name__ == "__main__":
    unittest.main()
