"""Offline tests for the real model/data adapters and fail-closed TPU launcher."""

from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from src.config import EvaluationConfig, ExecutionConfig, ResultsConfig, load_config
from src.data.batch import classification_batch
from src.data.imagenet_hf import (DataPlan, ImageBatchSource, evaluation_batches,
                                  label_mapping, prepare)
from src.methods import create_method
from src.models.resnet import ResNet, create_resnet, initialize, model_adapter
from src.tasks.classification import ClassificationTask
from src.training import TrainState
from src.training.launch import resolve_config
from src.training.evaluation import EvaluationSpec
from src.training.experiment import ExperimentRunner
from src.training.local_mesh import LocalMeshTrainer, local_mesh
from src.training.results import ResultsPaths
from src.training.runtime import configure_worker0, require_local_tpu
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(value):
    stream = BytesIO()
    Image.fromarray(np.full((12, 14, 3), value, np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def fake_stream(plan, split):
    count = plan.train_limit if split == "train" else plan.validation_limit
    return ({"image": {"bytes": image_bytes(i % 255)}, "label": i % 2}
            for i in range(count)), ["class0", "class1"]


class TPUPipelineTests(unittest.TestCase):
    def test_manifest_and_paired_tpu_presets(self):
        paths = sorted((ROOT / "configs/tpu").glob("imagenet*.toml"))
        self.assertEqual(len(paths), 2)
        configs = [load_config(path) for path in paths]
        for config in configs:
            mapping, checksum = label_mapping(config.data, ROOT)
            self.assertEqual(len(mapping), 100)
            self.assertEqual(mapping[452], 0)
            self.assertEqual(mapping[169], 99)
            self.assertEqual(len(checksum), 64)
            self.assertEqual(config.execution.gradient_backend, "jvp")
            self.assertEqual(config.model.initialization, "random")
        self.assertEqual(configs[0].data, configs[1].data)
        self.assertEqual(configs[0].training, configs[1].training)
        identity, digest = label_mapping(replace(configs[0].data, name="imagenet1k"), ROOT)
        self.assertEqual(identity, {i: i for i in range(1000)})
        self.assertEqual(len(digest), 64)

    def test_real_adapters_record_and_resume_through_the_shared_loop(self):
        with TemporaryDirectory() as tmp:
            config = load_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml")
            config = replace(config, global_batch_size=2,
                training=replace(config.training, total_steps=2),
                data=replace(config.data, image_size=8),
                results=ResultsConfig(str(Path(tmp) / "results")),
                evaluation=EvaluationConfig("steps", 1))
            plan = DataPlan("fixture/imagenet", "a" * 40, {0: 0, 1: 1}, "fixture", 4, 3)
            images = prepare(plan, Path(tmp) / "data", stream_factory=fake_stream)
            try:
                model = ResNet((1,), 100, num_filters=2)
                params, buffers = initialize(model, 0, 8)
                train_task = ClassificationTask(model_adapter(model, train=True), 100)
                eval_task = ClassificationTask(model_adapter(model, train=False), 100)
                method = create_method("sage", reference_name="error", gradient_backend="jvp")
                trainer = LocalMeshTrainer(train_task, method, SGDUpdate(.01), local_mesh(jax.local_devices()))
                initial = jax.device_put(TrainState(params, buffers), trainer.replicated)
                evaluation = EvaluationSpec(eval_task, lambda: evaluation_batches(images, 2, 8), 3)
                source = ImageBatchSource(images, 2, 8, 0)
                paths = ResultsPaths.resolve(config.results.root_dir, base_dir=ROOT).experiment("real_adapter_fixture")
                paths.reserve()
                runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=4, metadata={"actual_model": "tiny_resnet_fixture", "actual_data": "synthetic_encoded_images"},
                    run_name=paths.name, reserved=True, console=False)
                final = runner.run(initial)
                self.assertEqual(int(final.step), 2)
                self.assertEqual(source.cursor, 4)
                source2 = ImageBatchSource(images, 2, 8, 0)
                resumed = ExperimentRunner(config, trainer, source2, evaluation, base_dir=ROOT,
                    train_sample_count=4, metadata={}, run_name=paths.name, resume=True, console=False)
                final2 = resumed.run(initial, resume_slot="p50")
                for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(final2)):
                    np.testing.assert_array_equal(a, b)
            finally:
                images.close()

    def test_jvp_matches_full_jacobian_for_full_reference_methods(self):
        for name in ("sage", "dtv"):
            for reference in ("error", "loss"):
                with self.subTest(name=name, reference=reference):
                    args = (QuadraticTask(), {"x": jnp.array([.1])}, scalar_batch())
                    full = create_method(name, reference_name=reference).compute_direction(*args)
                    directional = create_method(name, reference_name=reference,
                        gradient_backend="jvp").compute_direction(*args)
                    np.testing.assert_allclose(full.direction["x"], directional.direction["x"], atol=1e-6)

    def test_jvp_and_batchnorm_state_match_without_committing_twice(self):
        model = ResNet((1,), 3, num_filters=4)
        params, state = initialize(model, 3, 8)
        task = ClassificationTask(model_adapter(model, train=True), 3)
        batch = classification_batch(np.random.default_rng(3).normal(size=(3, 8, 8, 3)).astype(np.float32),
                                     np.array([0, 1, 2], np.int32))
        results = []
        for backend in ("audit", "jvp"):
            method = create_method("sage", reference_name="error", gradient_backend=backend)
            results.append(jax.jit(lambda p, s: method.compute_direction(task, p, batch, s))(params, state))
        for left, right in zip(jax.tree_util.tree_leaves(results[0].direction), jax.tree_util.tree_leaves(results[1].direction)):
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=2e-6)
        expected = task.evaluate(params, batch, state).model_state
        for left, right in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(results[1].model_state)):
            np.testing.assert_allclose(left, right, atol=1e-6)

    def test_resnet50_and_101_parameter_shapes_are_real_architectures(self):
        for name, expected in (("resnet50", 23712932), ("resnet101", 42705060)):
            model = create_resnet(name, 100)
            shapes = jax.eval_shape(lambda key: model.init(key, jnp.zeros((1, 8, 8, 3)), train=True),
                                    jax.random.key(0))
            self.assertEqual(sum(leaf.size for leaf in jax.tree_util.tree_leaves(shapes["params"])), expected)

    def test_sharded_step_uses_global_direction_and_preserves_mesh_when_recording(self):
        mesh = local_mesh(jax.local_devices())
        task = QuadraticTask()
        method = create_method("sage", reference_name="error", gradient_backend="jvp")
        trainer = LocalMeshTrainer(task, method, SGDUpdate(.1), mesh)
        # This numerical check uses a classification-shaped batch with the real
        # ClassificationTask; the earlier quadratic task checks reference math.
        def model(params, images, state):
            return images @ params["w"], state
        task = ClassificationTask(model, 2)
        trainer = LocalMeshTrainer(task, method, SGDUpdate(.1), mesh)
        state = jax.device_put(TrainState({"w": jnp.array([[0., 1.]])}), trainer.replicated)
        batch = classification_batch(np.array([[1.], [-2.]], np.float32), np.array([0, 0], np.int32))
        eager, _ = trainer.step(state, batch, jax.random.key(0))
        compiled, _ = trainer.compile_step()(state, batch, jax.random.key(0))
        np.testing.assert_allclose(eager.params["w"], compiled.params["w"], atol=1e-6)
        self.assertIs(trainer.with_method(replace(method, record_theory=True)).mesh, mesh)

    def test_prepare_once_reuses_pinned_bytes_without_authentication(self):
        plan = DataPlan("fixture/imagenet", "a" * 40, {0: 0, 1: 1}, "fixture", 5, 3)
        with TemporaryDirectory() as tmp:
            images = prepare(plan, tmp, stream_factory=fake_stream)
            self.assertEqual(images.counts, {"train": 5, "validation": 3})
            images.close()
            def forbidden(*args):
                raise AssertionError("Prepared data should not access HF")
            images = prepare(plan, tmp, stream_factory=forbidden)
            self.assertEqual(images.get("train", 1)[1], 1)
            images.close()

    def test_incomplete_data_preparation_resumes_without_duplicate_rows(self):
        plan = DataPlan("fixture/imagenet", "a" * 40, {0: 0, 1: 1}, "fixture", 130, 2)
        def interrupted(plan, split):
            def rows():
                for index in range(130):
                    if index == 129:
                        raise ConnectionError("fixture interruption")
                    yield {"image": {"bytes": image_bytes(index)}, "label": index % 2}
            return rows(), ["class0", "class1"]
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                prepare(plan, tmp, stream_factory=interrupted)
            images = prepare(plan, tmp, stream_factory=fake_stream)
            self.assertEqual(images.counts["train"], 130)
            images.close()

    def test_epoch_shuffle_crop_and_tail_evaluation_restore_exactly(self):
        plan = DataPlan("fixture/imagenet", "a" * 40, {0: 0, 1: 1}, "fixture", 5, 3)
        with TemporaryDirectory() as tmp:
            images = prepare(plan, tmp, stream_factory=fake_stream)
            source = ImageBatchSource(images, 2, 8, 4)
            snapshot = source.snapshot()
            first = next(source)
            next(source)
            next(source)
            self.assertEqual(source.epoch, 1)
            source.restore(snapshot)
            restored = next(source)
            for left, right in zip(first, restored):
                np.testing.assert_array_equal(left, right)
            self.assertTrue(first.valid_mask.all())
            validation = list(evaluation_batches(images, 2, 8))
            self.assertEqual(sum(int(batch.valid_mask.sum()) for batch in validation), 3)
            self.assertEqual(validation[-1].images.shape, (2, 8, 8, 3))
            images.close()

    def test_runtime_rejects_foreground_worker1_and_cpu_fallback(self):
        config = ExecutionConfig()
        with self.assertRaises(RuntimeError):
            configure_worker0(config, environ={}, hostname="node-w-0")
        with self.assertRaises(RuntimeError):
            configure_worker0(config, environ={"SAGE_BACKGROUND_LAUNCH": "1"}, hostname="node-w-1")
        env = {"SAGE_BACKGROUND_LAUNCH": "1", "JAX_PROCESS_COUNT": "2"}
        settings = configure_worker0(config, environ=env, hostname="node-w-0")
        self.assertEqual(settings["TPU_PROCESS_BOUNDS"], "1,1,1")
        self.assertNotIn("JAX_PROCESS_COUNT", env)
        fake = SimpleNamespace(config=SimpleNamespace(update=lambda *args: None),
            local_devices=lambda **kw: [SimpleNamespace(platform="cpu", process_index=0)],
            default_backend=lambda: "cpu", process_count=lambda: 1)
        with patch.dict(sys.modules, {"jax": fake}), self.assertRaises(RuntimeError):
            require_local_tpu(config)

    def test_nohup_reservation_claims_only_once(self):
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("reserved")
            paths.reserve()
            (paths.logs / "stdout.log").write_text("bootstrap\n")
            paths.claim_reserved()
            with self.assertRaises(FileNotFoundError):
                paths.claim_reserved()

    def test_launcher_dry_run_and_smoke_configuration(self):
        path = ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml"
        config = resolve_config(path, mode="smoke")
        self.assertEqual(config.training.total_steps, 3)
        self.assertFalse(config.probes.enabled)
        output = subprocess.check_output([sys.executable, str(ROOT / "scripts/launch_tpu.py"),
                                          "--config", str(path), "--dry-run"], cwd=gettempdir(), text=True)
        launch = json.loads(output)
        self.assertEqual(launch["command"][0], "nohup")
        self.assertEqual(launch["backend"], "tpu_only")
        self.assertEqual(launch["worker"], "worker0_only")

    def test_two_cpu_global_batch_semantics_in_fresh_process(self):
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
               "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
        output = subprocess.check_output([sys.executable, "-m", "tests.local_mesh_check"],
                                         cwd=ROOT, env=env, text=True, timeout=120)
        self.assertIn("PASS: two-device global BN", output)
