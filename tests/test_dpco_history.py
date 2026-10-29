"""Independent numerical checks for EMA scoring, weighted edits, and recovery."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import load_config
from src.data.batch import classification_batch
from src.methods import create_method
from src.methods.dpco_history import HistoricalDPCO, ReferenceState
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.experiment import ExperimentRunner
from src.training.launch import resolve_config
from src.training.results import make_run_name
from src.training.updates import AdamUpdate
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import ROOT, fixture_config, fixture_components, read_rows

PRESET = ROOT / "configs/tpu/dpco/imagenet100_resnet50.toml"


def assert_tree(test, left, right, *, exact=False, atol=1e-6):
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=atol)


def components(root, weight=-1., method="dpco"):
    c = fixture_config(root)
    c = replace(c, method=replace(c.method, name=method, negative_weight=weight),
                reference=replace(c.reference, name="ema" if method == "dpco" else "loss", beta=.8),
                training=replace(c.training, optimizer="adam", learning_rate=.01))
    trainer, source, evaluation, initial, _ = fixture_components(c)
    trainer.method = create_method(method, reference_name=c.reference.name,
        reference_beta=c.reference.beta, negative_weight=weight, gradient_backend="jvp")
    return c, trainer, source, evaluation, initial


class HistoricalDPCOTests(unittest.TestCase):
    def test_first_step_and_arbitrary_weights_original_denominator(self):
        # At x=0, gradients [1,1,-1.5], raw mean=1/6, negative only last.
        for backend in ("audit", "jvp"):
            for weight in (0., -1., -.5, .25, 1., 2.):
                with self.subTest(backend=backend, weight=weight):
                    method = HistoricalDPCO(.9, weight, backend)
                    result = jax.jit(lambda p: method.compute_direction(
                        QuadraticTask(), p, scalar_batch()))({"x": jnp.zeros(1)})
                    np.testing.assert_allclose(result.direction["x"], [(2 - 1.5 * weight) / 3], atol=1e-6)
                    np.testing.assert_allclose(result.method_state.mean["x"], [.1 / 6], atol=1e-7)
                    self.assertEqual(int(result.method_state.updates), 1)
                    self.assertEqual(int(result.metrics["dpco_negative_count"]), 1)
                    self.assertEqual(int(result.model_state["calls"]), 1)
                    self.assertAlmostEqual(float(result.metrics["dpco_reference_norm"]), 1/6, places=6)

    def test_history_uses_raw_not_edited_and_changes_selection(self):
        params = {"x": jnp.zeros(1)}
        task = QuadraticTask()
        # First mean=+1/6 but flip direction=+7/6. Second raw mean=-.1.
        second = (jnp.array([[1.], [-.8]]), jnp.ones(2, bool), jnp.zeros(2, bool))
        for backend in ("audit", "jvp"):
            method = HistoricalDPCO(.9, -1., backend)
            first = method.compute_direction(task, params, scalar_batch())
            result = method.compute_direction(task, params, second, first.model_state, first.method_state)
            expected_mean = .9 * (.1 / 6) + .1 * (-.1)
            np.testing.assert_allclose(result.method_state.mean["x"], [expected_mean], atol=1e-7)
            # Historical ref remains positive, so flip -1 and retain +.8 -> +.9.
            np.testing.assert_allclose(result.direction["x"], [.9], atol=1e-6)
            self.assertEqual(int(result.model_state["calls"]), 2)
            self.assertEqual(int(result.method_state.updates), 2)
            current_only = HistoricalDPCO(0., -1., backend).compute_direction(task, params, second)
            np.testing.assert_allclose(current_only.direction["x"], [-.9], atol=1e-6)

    def test_multidimensional_audit_jvp_match_explicit_numpy_recurrence(self):
        params = {"x": jnp.array([.2, -.3])}
        mask = np.array([True, True, True, False])
        raw_mean = np.zeros(2)
        states = [None, None]
        rng = np.random.default_rng(5)
        for t in range(1, 5):
            targets = rng.normal(size=(4, 2)).astype(np.float32)
            batch = (jnp.asarray(targets), jnp.asarray(mask), jnp.zeros(4, bool))
            gradients = np.asarray(params["x"]) - targets
            raw_mean = .8 * raw_mean + .2 * gradients[mask].mean(0)
            reference = raw_mean / (1 - .8**t)
            weights = np.where(gradients @ reference < 0, -.4, 1.)
            expected = (gradients[mask] * weights[mask, None]).mean(0)
            for k, backend in enumerate(("audit", "jvp")):
                out = HistoricalDPCO(.8, -.4, backend).compute_direction(
                    QuadraticTask(), params, batch, method_state=states[k])
                np.testing.assert_allclose(out.direction["x"], expected, atol=1e-6)
                np.testing.assert_allclose(out.method_state.mean["x"], raw_mean, atol=1e-6)
                states[k] = out.method_state

    def test_beta_zero_degenerates_to_legacy_batch_rules(self):
        for weight, legacy in ((-1., "dpco"), (0., "dtv")):
            for backend in ("audit", "jvp"):
                args = (QuadraticTask(), {"x": jnp.array([.1])}, scalar_batch())
                historical = HistoricalDPCO(0., weight, backend).compute_direction(*args)
                old = create_method(legacy, gradient_backend=backend).compute_direction(*args)
                assert_tree(self, historical.direction, old.direction)

    def test_zero_reference_and_zero_align_are_retained(self):
        # Use exactly representable gradients [2,0], mean=1, previous EMA=-1.
        # A decimal 1/6 cancellation would test rounding rather than exact zero.
        method = HistoricalDPCO(.5, 0.)
        state = ReferenceState({"x": jnp.array([-1.])}, jnp.array(1))
        batch = (jnp.array([[-2.], [0.]]), jnp.ones(2, bool), jnp.zeros(2, bool))
        result = method.compute_direction(QuadraticTask(), {"x": jnp.zeros(1)},
                                          batch, method_state=state)
        np.testing.assert_array_equal(result.direction["x"], [1.])
        self.assertEqual(int(result.metrics["dpco_negative_count"]), 0)
        self.assertEqual(float(result.metrics["dpco_reference_cosine"]), 0.)
        # Nonzero reference [1,0], with one exactly orthogonal sample [0,1].
        batch = (jnp.array([[0., -1.], [-1., 0.]]), jnp.ones(2, bool), jnp.zeros(2, bool))
        state = ReferenceState({"x": jnp.array([1.5, -.5])}, jnp.array(1))
        result = method.compute_direction(QuadraticTask(), {"x": jnp.zeros(2)}, batch, method_state=state)
        np.testing.assert_array_equal(result.direction["x"], [.5, .5])

    def test_padding_is_excluded_from_mean_scores_and_denominator(self):
        task = ClassificationTask(lambda p, x, s: (x @ p["w"], s), 2)
        params = {"w": jnp.array([[0., 1.]])}
        short = classification_batch(jnp.array([[1.], [-2.]]), jnp.array([0, 0]))
        padded = classification_batch(jnp.array([[1.], [-2.], [jnp.nan]]),
            jnp.array([0, 0, 999]), jnp.array([True, True, False]))
        for backend in ("audit", "jvp"):
            method = HistoricalDPCO(.9, 0., backend)
            a = method.compute_direction(task, params, short)
            b = method.compute_direction(task, params, padded)
            assert_tree(self, a.direction, b.direction)
            assert_tree(self, a.method_state, b.method_state)
            self.assertEqual(int(b.metrics["valid_batch_size"]), 2)

    def test_real_batchnorm_buffers_commit_one_forward(self):
        from src.models.resnet import ResNet, initialize, model_adapter
        model = ResNet((1,), 3, num_filters=2)
        params, buffers = initialize(model, 3, 8)
        task = ClassificationTask(model_adapter(model, train=True), 3)
        batch = classification_batch(np.random.default_rng(3).normal(size=(3, 8, 8, 3)).astype(np.float32),
                                     np.arange(3, dtype=np.int32))
        expected = task.evaluate(params, batch, buffers).model_state
        results = []
        for backend in ("audit", "jvp"):
            method = HistoricalDPCO(.9, -.5, backend)
            result = jax.jit(lambda p, s: method.compute_direction(task, p, batch, s))(params, buffers)
            assert_tree(self, result.model_state, expected)
            results.append(result)
        assert_tree(self, results[0].direction, results[1].direction, atol=2e-6)

    def test_beta_close_to_one_still_has_finite_corrected_first_reference(self):
        result = HistoricalDPCO(1 - 1e-12).compute_direction(
            QuadraticTask(), {"x": jnp.zeros(1)}, scalar_batch())
        self.assertAlmostEqual(float(result.metrics["dpco_reference_norm"]), 1/6, places=6)

    def test_invalid_parameters_and_state_fail_closed(self):
        for beta in (-.1, 1., float("nan"), True):
            with self.assertRaises(ValueError):
                HistoricalDPCO(beta)
        for weight in (float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                HistoricalDPCO(negative_weight=weight)
        method = HistoricalDPCO()
        params = {"x": jnp.zeros(1)}
        for state in (None, ReferenceState(params, jnp.array(1.5)),
                      ReferenceState(params, jnp.array(2)),
                      ReferenceState({"x": jnp.array([jnp.nan])}, jnp.array(1))):
            with self.assertRaises(ValueError):
                method.validate_state(state, params, 1)


class AdamAndRecoveryTests(unittest.TestCase):
    def test_standard_adam_matches_independent_numpy_with_schedule_and_l2(self):
        for l2 in (0., .1):
            updater = AdamUpdate(lambda t: .02/(t+1), .8, .95, 1e-7, l2)
            params = {"w": jnp.array([[.3, -.2]]), "b": jnp.zeros(2)}
            expected = jax.tree_util.tree_map(np.asarray, params)
            mu = jax.tree_util.tree_map(np.zeros_like, expected)
            nu = jax.tree_util.tree_map(np.zeros_like, expected)
            state = updater.init(params)
            for t, vector in enumerate(([2., -3.], [-1., 4.], [0., 0.])):
                direction = {"w": jnp.array([vector]), "b": jnp.array(vector)}
                for key in params:
                    g = np.asarray(direction[key]) + (l2 * expected[key] if expected[key].ndim > 1 else 0)
                    mu[key] = .8 * mu[key] + .2 * g
                    nu[key] = .95 * nu[key] + .05 * g*g
                    expected[key] = expected[key] - .02/(t+1) * (mu[key]/(1-.8**(t+1))) / (
                        np.sqrt(nu[key]/(1-.95**(t+1))) + 1e-7)
                out = jax.jit(updater.apply)(params, direction, t, state)
                params, state = out
                assert_tree(self, params, expected)
                updater.validate_state(state, params, t+1)

    def test_adam_moments_use_edited_direction_not_raw_gradient(self):
        method = HistoricalDPCO(.9, -1.)
        updater = AdamUpdate(.01, .8, .95)
        params = {"x": jnp.zeros(1)}
        state = TrainState(params, method_state=method.initial_state(params), optimizer_state=updater.init(params))
        result, _ = jax.jit(Trainer(QuadraticTask(), method, updater).step)(state, scalar_batch())
        np.testing.assert_allclose(result.method_state.mean["x"], [.1/6], atol=1e-7)
        np.testing.assert_allclose(result.optimizer_state.mu["x"], [.2*7/6], atol=1e-7)
        np.testing.assert_allclose(result.optimizer_state.nu["x"], [.05*(7/6)**2], atol=1e-7)

    def test_three_groups_and_weight_one_control_resume_exactly(self):
        for method, weight in (("adam", -1.), ("dpco", 0.), ("dpco", -1.), ("dpco", 1.)):
            with self.subTest(method=method, weight=weight), TemporaryDirectory() as tmp:
                c, trainer, source, evaluation, initial = components(tmp, weight, method)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="test")
                final = runner.run(initial)
                # Compile warmup and initial validation must not consume history or data.
                self.assertEqual(source.index, 4)
                self.assertEqual(int(final.optimizer_state.count), 4)
                if method == "dpco":
                    self.assertEqual(int(final.method_state.updates), 4)
                    rows = read_rows(runner.paths.files / "dpco_steps.csv")
                    self.assertEqual([int(row["dpco_reference_updates"]) for row in rows], [1, 2, 3, 4])
                _, _, source2, evaluation2, _ = components(tmp, weight, method)
                resumed = ExperimentRunner(c, trainer, source2, evaluation2, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="test", resume=True)
                final2 = resumed.run(initial, resume_slot="p50")
                assert_tree(self, final, final2, exact=True)
                if method == "dpco":
                    self.assertEqual(len(read_rows(runner.paths.files / "dpco_steps.csv")), 4)

    def test_weight_one_matches_plain_adam_whole_trajectory(self):
        with TemporaryDirectory() as tmp:
            _, trainer, source, _, initial = components(tmp, 1.)
            baseline = Trainer(trainer.task, create_method("adam"), trainer.updater)
            a = initial._replace(method_state=trainer.method.initial_state(initial.params),
                                 optimizer_state=trainer.updater.init(initial.params))
            b = initial._replace(optimizer_state=trainer.updater.init(initial.params))
            for _ in range(5):
                batch = next(source)
                a, _ = jax.jit(trainer.step)(a, batch)
                b, _ = jax.jit(baseline.step)(b, batch)
                assert_tree(self, a.params, b.params, exact=True)
                assert_tree(self, a.optimizer_state, b.optimizer_state, exact=True)

    def test_missing_reference_or_adam_checkpoint_rejected_before_csv_rewind(self):
        for field in ("method_state", "optimizer_state"):
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                c, trainer, source, evaluation, initial = components(tmp)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="test")
                runner.run(initial)
                before = (runner.paths.files / "dpco_steps.csv").read_bytes()
                path = runner.paths.runs / runner.store.index["p50"]
                saved = pickle.loads(path.read_bytes())
                saved["train_state"] = saved["train_state"]._replace(**{field: None})
                path.write_bytes(pickle.dumps(saved))
                resumed = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="test", resume=True)
                with self.assertRaisesRegex(ValueError, "state/count"):
                    resumed.run(initial, resume_slot="p50")
                self.assertEqual((runner.paths.files / "dpco_steps.csv").read_bytes(), before)
                self.assertTrue(resumed.recorder.log_file.closed)

    def test_real_training_entry_constructs_three_groups_and_resumes(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.train_classification import run
        from src.data.imagenet_hf import DataPlan, prepare
        from src.models.resnet import ResNet
        from src.training.checkpoints import CheckpointStore
        from tests.test_tpu_pipeline import fake_stream
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for alias in ("adam", "dpco_filter", "dpco_flip"):
                c = resolve_config(PRESET, mode="smoke", method=alias, steps=2, batch_size=2,
                                   reference_beta=.85, adam_beta1=.8, adam_beta2=.95)
                c = replace(c, data=replace(c.data, image_size=8),
                            results=replace(c.results, root_dir=str(root / "results")))
                args = SimpleNamespace(mode="smoke", run_name=alias, resume=False, reserved=False, resume_slot="latest")
                def images_factory(*args, **kwargs):
                    return prepare(DataPlan("fixture", "a"*40, {0: 0, 1: 1}, "fixture", 4, 3),
                                   root/"data", stream_factory=fake_stream), {}
                with patch("src.data.classification.open_training_images", side_effect=images_factory), \
                     patch("src.models.resnet.create_resnet", return_value=ResNet((1,), 100, num_filters=2)), \
                     redirect_stdout(StringIO()):
                    run(args, c, jax.local_devices(), status=None)
                    store = CheckpointStore(root / "results/runs" / alias)
                    final = store.load("final")["train_state"]
                    args.resume, args.resume_slot = True, "p50"
                    run(args, c, jax.local_devices(), status=None)
                    final2 = CheckpointStore(store.directory).load("final")["train_state"]
                assert_tree(self, final, final2, exact=True)
                self.assertEqual(int(final.optimizer_state.count), 2)
                if alias != "adam":
                    self.assertEqual(int(final.method_state.updates), 2)
                metadata = json.loads((root / "results/logs" / alias / "metadata.json").read_text())
                self.assertEqual(metadata["experiment"]["optimizer"], "adam")
                self.assertEqual(metadata["config"]["method"]["name"], c.method.name)

    def test_invalid_adam_options(self):
        for kwargs in ({"beta1": 1.}, {"beta2": -.1}, {"epsilon": 0.}, {"epsilon": float("nan")}):
            with self.assertRaises(ValueError):
                AdamUpdate(.01, **kwargs)


class DPCOConfigurationTests(unittest.TestCase):
    def test_plain_filter_flip_share_settings_and_have_distinct_names(self):
        configs = [resolve_config(PRESET, mode="train", method=m) for m in ("adam", "dpco_filter", "dpco_flip")]
        self.assertEqual([c.method.name for c in configs], ["adam", "dpco", "dpco"])
        self.assertEqual([c.method.negative_weight for c in configs[1:]], [0., -1.])
        self.assertEqual(len({make_run_name(c) for c in configs}), 3)
        for c in configs:
            self.assertEqual(c.training, configs[0].training)
            self.assertEqual(c.data, configs[0].data)
            self.assertEqual(c.training.optimizer, "adam")
        custom = resolve_config(PRESET, mode="train", method="dpco", negative_weight=-.25, reference_beta=.95)
        self.assertEqual(custom.method.negative_weight, -.25)
        self.assertEqual(custom.reference.beta, .95)

    def test_parent_child_round_trip_and_dry_run_has_no_runtime_imports(self):
        # This child only plans commands; deliberately block JAX imports.
        code = '''import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('jax', 'optax', 'flax'):
        raise AssertionError('accelerator imported during dry run')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['scripts/launch_tpu.py', '--config', 'configs/tpu/dpco/imagenet100_resnet50.toml',
 '--mode', 'train', '--methods', 'adam', 'dpco_filter', 'dpco_flip', '--dry-run',
 '--reference-beta', '.85', '--adam-beta1', '.88', '--adam-beta2', '.98', '--adam-epsilon', '1e-7']
runpy.run_path('scripts/launch_tpu.py', run_name='__main__')
'''
        plans = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True))
        self.assertEqual(len(plans), 3)
        from src.training.launch import add_overrides, diagnostic_overrides, dpco_overrides
        import argparse
        parser = argparse.ArgumentParser()
        add_overrides(parser)
        for plan in plans:
            command = plan["command"]
            start = command.index("--seed")
            args = parser.parse_args(command[start:])
            child = resolve_config(PRESET, mode="train", seed=args.seed, method=args.method,
                                   **dpco_overrides(args), **diagnostic_overrides(args))
            self.assertEqual(child.training.adam_beta1, .88)
            self.assertEqual(child.training.adam_beta2, .98)
            self.assertEqual(child.training.adam_epsilon, 1e-7)
            self.assertEqual(child.training.optimizer, plan["optimizer"])
            if child.method.name == "dpco":
                self.assertEqual(child.reference.beta, plan["reference_beta"])
                self.assertEqual(child.method.negative_weight, plan["negative_weight"])

    def test_invalid_combinations_and_nonfinite_weights_are_rejected(self):
        for kwargs in ({"negative_weight": float("inf")}, {"reference_beta": 1.},
                       {"adam_beta2": 1.}, {"momentum": .9}, {"theorem_probes": True},
                       {"diagnostics": True}, {"dpco_warmup_percent": 10.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_config(PRESET, mode="train", **kwargs)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_config(PRESET, mode="train", method="dpco_filter", negative_weight=-1.)


if __name__ == "__main__":
    unittest.main()
