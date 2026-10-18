"""Run in a fresh two-CPU process to validate global-batch GSPMD semantics."""

import jax
import jax.numpy as jnp
import numpy as np

from src.data.batch import classification_batch
from src.methods import create_method
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.local_mesh import LocalMeshTrainer, local_mesh
from src.training.updates import SGDUpdate


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
        audit = create_method(method_name, reference_name=reference, gradient_backend="audit")
        expected, _ = jax.jit(Trainer(task, audit, SGDUpdate(.1)).step)(initial, batch, jax.random.key(0))
        method = create_method(method_name, reference_name=reference, gradient_backend="jvp")
        distributed = LocalMeshTrainer(task, method, SGDUpdate(.1), local_mesh(jax.local_devices()))
        state = jax.device_put(initial, distributed.replicated)
        actual, _ = distributed.compile_step()(state, batch, jax.random.key(0))
        for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    print("PASS: two-device global BN, reference/error count, JVP scores and weighted update match full Jacobian")


if __name__ == "__main__":
    main()
