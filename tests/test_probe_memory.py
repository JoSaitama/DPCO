"""Bounded-memory probe math must match the small-batch dense Jacobian oracle."""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ProbeConfig
from src.data.batch import classification_batch
from src.methods.base import aggregate
from src.models.resnet import ResNet, initialize, model_adapter
from src.probes.classification import ClassificationProbes, ProbeBatch
from src.probes.gradient_statistics import ProbeStatistics, common_reference_statistics
from src.references import create_reference
from src.signals.gradient_alignment import gradient_alignment
from src.tasks.classification import ClassificationTask
from src.training import TrainState


def dense_statistics(task, reference, params, batch, state, *, include_energy):
    """Pre-fix mathematical path, restricted to tiny offline test fixtures."""
    def losses_fn(p):
        result = task.evaluate(p, batch, state)
        return result.losses, (result.valid_mask, result.feedback)

    gradients, (valid, feedback) = jax.jacrev(losses_fn, has_aux=True)(params)
    losses = task.evaluate(params, batch, state).losses
    r = reference.build(gradients, valid, feedback)
    scores = gradient_alignment(gradients, r, valid)
    weights = jnp.where(valid, jnp.where((scores < 0) & r.has_reference, -1., 1.), 0.)
    gd = aggregate(gradients, jnp.ones_like(weights), valid)
    gs = aggregate(gradients, weights, valid)
    energy = (sum(jnp.sum(jnp.where(valid.reshape((valid.shape[0],) + (1,) * (g.ndim - 1)),
                                    g ** 2, 0.)) for g in jax.tree_util.tree_leaves(gradients))
              / jnp.sum(valid)) if include_energy else jnp.array(0.)
    return ProbeStatistics(losses, valid, feedback["error_mask"], r.has_reference,
                           scores, r.direction, gd, gs, energy)


def coupled_fixture(error_mode):
    def model_fn(params, images, state):
        mean = jnp.mean(images, axis=0)
        variance = jnp.mean((images - mean) ** 2, axis=0)
        normalized = (images - mean) / jnp.sqrt(variance + 1e-5)
        return normalized @ params["w"] + params["b"], {"mean": mean}

    task = ClassificationTask(model_fn, 2)
    state = TrainState({"w": jnp.array([[.3, -.7], [.8, -.2], [-.6, .4]]), "b": jnp.array([.1, -.1])},
                       {"mean": jnp.zeros(3)})
    images = jnp.array([[5., -2., 1.], [1., 4., -3.], [-2., 1., 2.], [-4., -3., 4.],
                        [2., 3., -1.], [-3., 2., 5.], [100., 100., 100.]])
    valid = jnp.array([True] * 6 + [False])
    batch = classification_batch(images, jnp.zeros(7, dtype=jnp.int32), valid)
    # Construct error sets explicitly, independently of reference selection.
    predictions = jnp.argmax(model_fn(state.params, jnp.where(valid[:, None], images, 0.), state.model_state)[0], axis=-1)
    labels = predictions if error_mode == "none" else 1 - predictions
    if error_mode == "mixed":
        labels = predictions.at[:2].set(1 - predictions[:2])
    return task, state, classification_batch(images, labels, valid)


class ProbeMemoryTests(unittest.TestCase):
    def test_bounded_statistics_match_dense_oracle_with_padding_and_global_bn(self):
        for error_mode in ("none", "all", "mixed"):
            task, state, batch = coupled_fixture(error_mode)
            before = jax.tree_util.tree_map(np.array, state)
            for name in ("error", "loss"):
                for energy in (False, True):
                    with self.subTest(errors=error_mode, reference=name, energy=energy):
                        ref = create_reference(name)
                        expected = dense_statistics(task, ref, state.params, batch, state.model_state,
                                                    include_energy=energy)
                        function = jax.jit(lambda p, b, s: common_reference_statistics(task, ref, p, b, s,
                                                                                    include_energy=energy))
                        actual = function(state.params, batch, state.model_state)
                        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
                            np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-6)
            for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(before)):
                np.testing.assert_array_equal(a, b)

    def test_all_recorded_fields_match_dense_probe_oracle(self):
        task, state, batch = coupled_fixture("mixed")
        for name in ("error", "loss"):
            with self.subTest(reference=name):
                audit = lambda p: jnp.mean(task.evaluate(p, batch, state.model_state).losses)
                kwargs = dict(reference_name=name, objective_mode="coupled_bn_test",
                              audit_objective=audit, audit_name="fixed_test")
                actual = ClassificationProbes(task, create_reference(name), [ProbeBatch("fixed", batch)], **kwargs)
                oracle = ClassificationProbes(task, create_reference(name), [ProbeBatch("fixed", batch)], **kwargs)
                oracle._statistics = {enabled: lambda p, b, s, enabled=enabled: dense_statistics(
                    task, oracle.reference, p, b, s, include_energy=enabled) for enabled in (False, True)}
                args = dict(step=0, learning_rate=.01, config=ProbeConfig(enabled=True), cumulative={})
                rows, expected = actual.run(state, **args), oracle.run(state, **args)
                self.assertEqual(len(rows), len(expected))
                for (stream, row), (expected_stream, other) in zip(rows, expected):
                    self.assertEqual(stream, expected_stream)
                    self.assertEqual(row.keys(), other.keys())
                    for field, value in row.items():
                        if isinstance(value, (str, bool)) or value is None:
                            self.assertEqual(value, other[field])
                        else:
                            np.testing.assert_allclose(value, other[field], rtol=5e-4, atol=2e-4,
                                                       err_msg=f"{stream}.{field}")

    def test_energy_is_a_scalar_loop_without_a_batch_parameter_jacobian(self):
        task, state, batch = coupled_fixture("mixed")
        ref = create_reference("loss")
        for energy in (False, True):
            function = lambda p, b, s: common_reference_statistics(task, ref, p, b, s, include_energy=energy)
            closed = jax.make_jaxpr(function)(state.params, batch, state.model_state)
            shapes, primitives = [], []
            visited = set()

            def visit(value):
                if id(value) in visited:
                    return
                visited.add(id(value))
                if hasattr(value, "eqns"):
                    for equation in value.eqns:
                        primitives.append(equation.primitive.name)
                        for var in equation.outvars:
                            shapes.append(getattr(var.aval, "shape", ()))
                        for parameter in equation.params.values():
                            visit(parameter)
                elif hasattr(value, "jaxpr"):
                    visit(value.jaxpr)
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        visit(item)

            visit(closed)
            self.assertNotIn((7, 3, 2), shapes)
            self.assertEqual(any(name in ("scan", "while") for name in primitives), energy)

    def test_small_flax_resnet_statistics_match_dense_oracle(self):
        model = ResNet((1,), 3, num_filters=2)
        params, buffers = initialize(model, 5, 8)
        task = ClassificationTask(model_adapter(model, train=True), 3)
        batch = classification_batch(jax.random.normal(jax.random.key(2), (4, 8, 8, 3)), jnp.array([0, 1, 2, 1]))
        ref = create_reference("error")
        expected = dense_statistics(task, ref, params, batch, buffers, include_energy=True)
        actual = jax.jit(lambda p, b, s: common_reference_statistics(task, ref, p, b, s, include_energy=True))(
            params, batch, buffers)
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=3e-6)

    def test_runtime_probes_need_no_jacrev_and_skip_energy_when_theorem3_is_off(self):
        task, state, batch = coupled_fixture("mixed")
        probes = ClassificationProbes(task, create_reference("error"), [ProbeBatch("fixed", batch)],
                                      reference_name="error", objective_mode="coupled_bn_test")
        with patch("jax.jacrev", side_effect=AssertionError("Dense runtime probe is forbidden")):
            rows = probes.run(state, step=0, learning_rate=.01,
                              config=ProbeConfig(enabled=True, theorem3=False), cumulative={})
        self.assertTrue(any(stream == "theorem1" for stream, _ in rows))
        self.assertTrue(any(stream == "theorem2" for stream, _ in rows))
        self.assertFalse(any(stream == "theorem3" for stream, _ in rows))


if __name__ == "__main__":
    unittest.main()
