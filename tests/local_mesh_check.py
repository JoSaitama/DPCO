"""Run in a fresh two-CPU process to validate global-batch GSPMD semantics."""

import jax
import jax.numpy as jnp
import numpy as np

from src.data.batch import classification_batch
from src.methods import create_method
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.local_mesh import LocalMeshTrainer, local_mesh
from src.config import TrainingConfig
from src.training.updates import make_updater
from src.references import create_reference
from src.probes.gradient_statistics import common_reference_statistics


def main():
    assert jax.default_backend() == "cpu"
    assert len(jax.local_devices()) == 2
    def model_fn(params, images, state):
        mean = jnp.mean(images, axis=0)
        variance = jnp.mean((images - mean) ** 2, axis=0)
        normalized = (images - mean) / jnp.sqrt(variance + 1e-5)
        return normalized @ params["w"] + params["b"], {"mean": .9 * state["mean"] + .1 * mean}
    task = ClassificationTask(model_fn, 2)
    initial = TrainState({"w": jnp.array([[0., 1.], [1., 0.]]), "b": jnp.zeros(2)},
                         {"mean": jnp.zeros(2)})
    batch = classification_batch(np.array([[5., -2.], [1., 4.], [-2., 1.], [-4., -3.]], np.float32),
                                 np.array([0, 1, 0, 0], np.int32))
    for method_name, reference in (("sgd", "loss"), ("dpco", "error"), ("dpco", "loss")):
        for training in (TrainingConfig(learning_rate=.1),
                         TrainingConfig(learning_rate=.1, total_steps=3, momentum=.9, nesterov=True,
                                        l2_coefficient=.0001, lr_schedule="cosine", min_learning_rate=.001)):
            updater = make_updater(training)
            audit = create_method(method_name, reference_name=reference, gradient_backend="audit")
            reference_step = jax.jit(Trainer(task, audit, updater).step)
            method = create_method(method_name, reference_name=reference, gradient_backend="jvp")
            distributed = LocalMeshTrainer(task, method, updater, local_mesh(jax.local_devices()))
            expected = initial._replace(optimizer_state=updater.init(initial.params))
            actual = jax.device_put(expected, distributed.replicated)
            compiled_step = distributed.compile_step()
            for step in range(3):
                expected, _ = reference_step(expected, batch, jax.random.key(step))
                actual, _ = compiled_step(actual, batch, jax.random.key(step))
                for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    # The failing production probe now uses whole-batch compiled scalar VJPs.
    # Validate its global BatchNorm and exact sample-energy semantics too.
    from tests.test_probe_memory import dense_statistics
    sharded = jax.device_put(initial, distributed.replicated)
    sharded_batch = jax.device_put(batch, distributed.batch_sharding)
    for name in ("loss", "error"):
        reference = create_reference(name)
        expected = dense_statistics(task, reference, initial.params, batch, initial.model_state, include_energy=True)
        actual = jax.jit(lambda p, b, s: common_reference_statistics(
            task, reference, p, b, s, include_energy=True))(sharded.params, sharded_batch, sharded.model_state)
        for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    # Real-update observers must retain the same whole-batch BN geometry on a
    # sharded trajectory, and must not replace the actual compiled update.
    from src.config import DiagnosticConfig
    from src.probes.training_diagnostics import TrainingDiagnostics
    method = create_method("dpco", reference_name="loss", gradient_backend="jvp")
    trainer = LocalMeshTrainer(task, method, make_updater(TrainingConfig(learning_rate=.01)), local_mesh(jax.local_devices()))
    before = jax.device_put(initial, trainer.replicated)
    after, _ = trainer.compile_step()(before, sharded_batch, jax.random.key(0))
    diagnostic = TrainingDiagnostics(task, method, DiagnosticConfig(enabled=True, transfer=False, lr_scales=(1.,)),
        reference_name="loss", trajectory_method="dpco")
    rows = diagnostic.run(before, after, sharded_batch, learning_rate=.01, rng_key=jax.random.key(0),
        context={"batch_id": "global_sharded_batch"}, cumulative_eta_before=0.)
    row = next(row for stream, row in rows if stream == "diagnostic_update")
    assert row["direction_relative_error"] < 2e-5, row
    assert row["model_state_max_abs_error"] < 2e-6, row
    # Opt-in policies must preserve global BN and the same original-B algebra.
    # Compare against independent dense SGD/DPCO/DTV rounds, including both
    # branches of the alternating conditional on a replicated policy counter.
    for mode, first in (("continuous", "sgd"), ("alternating", "sgd"),
                        ("alternating", "dpco"), ("half", "sgd")):
        for reference in ("loss", "error"):
            method = create_method("dpco_policy", policy_mode=mode,
                reference_name=reference, gradient_backend="jvp", alternating_first=first)
            trainer = LocalMeshTrainer(task, method, make_updater(TrainingConfig(learning_rate=.01)),
                                       local_mesh(jax.local_devices()))
            actual = jax.device_put(initial._replace(method_state=method.initial_state()), trainer.replicated)
            expected = initial
            compiled = trainer.compile_step()
            for t in range(3):
                selected = {"continuous": "dpco", "half": "dtv",
                            "alternating": "dpco" if (t + int(first == "dpco")) % 2 else "sgd"}[mode]
                baseline = Trainer(task, create_method(selected, reference_name=reference,
                    gradient_backend="audit"), trainer.updater)
                expected, _ = baseline.compile_step()(expected, batch, jax.random.key(t))
                actual, _ = compiled(actual, sharded_batch, jax.random.key(t))
                for a, b in zip(jax.tree_util.tree_leaves(expected),
                                jax.tree_util.tree_leaves(actual._replace(method_state=None))):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                assert int(actual.method_state.updates) == t + 1
    # EMA + Adam must share a single global raw-gradient history; compare with
    # dense per-example Jacobians across multiple updates and both edit rules.
    from src.methods.dpco_history import HistoricalDPCO
    from src.training.updates import AdamUpdate
    for weight in (0., -1., -.5):
        updater = AdamUpdate(.001)
        audit = HistoricalDPCO(.85, weight, "audit")
        method = HistoricalDPCO(.85, weight, "jvp")
        dense = Trainer(task, audit, updater).compile_step()
        mesh_trainer = LocalMeshTrainer(task, method, updater, local_mesh(jax.local_devices()))
        expected = initial._replace(method_state=audit.initial_state(initial.params),
                                    optimizer_state=updater.init(initial.params))
        actual = jax.device_put(expected, mesh_trainer.replicated)
        compiled = mesh_trainer.compile_step()
        for t in range(3):
            expected, expected_metrics = dense(expected, batch, jax.random.key(t))
            actual, metrics = compiled(actual, sharded_batch, jax.random.key(t))
            for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(metrics["dpco_negative_count"], expected_metrics["dpco_negative_count"])
            assert int(actual.method_state.updates) == t + 1
            assert int(actual.optimizer_state.count) == t + 1
    # New native-image backbone: global convolution/BatchNorm, EMA and Adam
    # must agree with a dense full-Jacobian oracle on the same logical batch.
    from src.models.cifar_resnet import CifarResNet
    from src.models.resnet import initialize, model_adapter
    native_model = CifarResNet((1,), 100, num_filters=2)
    native_params, native_buffers = initialize(native_model, 19, 32)
    native_task = ClassificationTask(model_adapter(native_model, train=True), 100)
    native_batch = classification_batch(np.random.default_rng(5).normal(size=(4, 32, 32, 3)).astype(np.float32),
                                         np.array([0, 1, 2, 3], np.int32))
    native_updater = AdamUpdate(.001)
    for weight in (0., -1.):
        oracle = HistoricalDPCO(.85, weight, "audit")
        method = HistoricalDPCO(.85, weight, "jvp")
        native_mesh = LocalMeshTrainer(native_task, method, native_updater, local_mesh(jax.local_devices()))
        expected = TrainState(native_params, native_buffers, oracle.initial_state(native_params),
                              native_updater.init(native_params))
        actual = jax.device_put(expected, native_mesh.replicated)
        dense_step = Trainer(native_task, oracle, native_updater).compile_step()
        mesh_step = native_mesh.compile_step()
        for t in range(2):
            expected, _ = dense_step(expected, native_batch, jax.random.key(t))
            actual, _ = mesh_step(actual, native_batch, jax.random.key(t))
            for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
                np.testing.assert_allclose(a, b, rtol=3e-4, atol=3e-6)
    print("PASS: two-device global BN, full-gradient oracle, legacy updates, DPCO policies and EMA/Adam weights")


if __name__ == "__main__":
    main()
