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
    for method_name, reference in (("sgd", "loss"), ("sage", "error"), ("sage", "loss")):
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
    print("PASS: two-device global BN, reference/error count, JVP scores, bounded-memory probes, momentum/cosine/L2 and weighted updates match full Jacobian")


if __name__ == "__main__":
    main()
