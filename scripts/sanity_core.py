"""CPU sanity checks for the core pipeline, without ImageNet or ResNet training."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def run_sanity(config, steps):
    import jax

    # This process checks core correctness only, including when run on a TPU VM.
    # It must not initialize or join the VM's multi-host TPU slice.
    jax.config.update("jax_platforms", "cpu")

    import jax.numpy as jnp
    import numpy as np

    from src.data import classification_batch
    from src.methods import create_method
    from src.references import create_reference
    from src.tasks.classification import ClassificationTask
    from src.training import TrainState, Trainer
    from src.training.results import ResultsPaths
    from src.training.updates import make_updater

    def model_fn(params, images, state):
        return images @ params["w"] + params["b"], state

    def assert_direction(actual, expected):
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=2e-6)

    def analytic_batch(params, batch):
        """Compute linear-softmax CE gradients independently of JAX autodiff."""
        valid = np.asarray(batch.valid_mask)
        images = np.where(valid[:, None], np.asarray(batch.images), 0.)
        labels = np.where(valid, np.asarray(batch.labels), 0)
        logits = images @ np.asarray(params["w"]) + np.asarray(params["b"])
        exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
        residual = exponentials / exponentials.sum(axis=-1, keepdims=True)
        residual[np.arange(len(labels)), labels] -= 1.
        residual = np.where(valid[:, None], residual, 0.)
        gradients = {"w": images[:, :, None] * residual[:, None, :], "b": residual}
        errors = (logits.argmax(axis=-1) != labels) & valid
        return gradients, errors, valid

    def expected_direction(params, batch, method_name, reference_name):
        gradients, errors, valid = analytic_batch(params, batch)
        weights = np.ones(len(valid), dtype=np.float32)
        if method_name == "sage":
            selected = errors if reference_name == "error" else valid
            if selected.any():
                reference = {name: leaf[selected].mean(axis=0)
                             for name, leaf in gradients.items()}
                scores = sum((leaf * reference[name]).reshape(len(valid), -1).sum(axis=1)
                             for name, leaf in gradients.items())
                weights = np.where(scores < 0., -1., 1.)
        weights = np.where(valid, weights, 0.)
        return {
            name: (leaf * weights.reshape((len(valid),) + (1,) * (leaf.ndim - 1))).sum(axis=0)
                  / valid.sum()
            for name, leaf in gradients.items()
        }

    # Five valid examples and one padded position exercise the original-B rule.
    # Class 1 has positive logit on the first example; all true labels are class 0.
    classes = config.data.num_classes
    params = {
        "w": jnp.zeros((1, classes), dtype=jnp.float32).at[0, 1].set(1.),
        "b": jnp.full((classes,), -30., dtype=jnp.float32).at[:2].set(0.),
    }
    images = jnp.array([[1.], [-2.], [-2.], [-2.], [-2.], [100.]])
    valid = jnp.array([True, True, True, True, True, False])
    labels = jnp.array([0, 0, 0, 0, 0, classes - 1], dtype=jnp.int32)
    batch = classification_batch(images, labels, valid)
    task = ClassificationTask(model_fn, classes)
    gradients, errors, valid_np = analytic_batch(params, batch)
    evaluated = task.evaluate(params, batch)
    np.testing.assert_array_equal(evaluated.feedback["error_mask"], errors)
    np.testing.assert_array_equal(errors, [True, False, False, False, False, False])

    for reference_name, selected in (("error", errors), ("loss", valid_np)):
        reference = create_reference(reference_name).build(
            gradients, valid, evaluated.feedback
        )
        assert not reference.per_sample
        assert int(reference.selected_count) == int(selected.sum())
        assert_direction(reference.direction, {
            name: leaf[selected].mean(axis=0) for name, leaf in gradients.items()
        })
        method = create_method("sage", reference_name=reference_name)
        assert not method.exclude_self
        actual = method.compute_direction(task, params, batch).direction
        assert_direction(actual, expected_direction(params, batch, "sage", reference_name))
    print("PASS: SAGE-E/SAGE-L full references, raw alignment, flips, and valid-B aggregation")

    direction_e = create_method("sage", reference_name="error").compute_direction(task, params, batch).direction
    direction_l = create_method("sage", reference_name="loss").compute_direction(task, params, batch).direction
    if np.allclose(direction_e["w"], direction_l["w"]):
        raise AssertionError("The fixture must distinguish error and loss references")

    predictions = jnp.argmax(model_fn(params, jnp.where(valid[:, None], images, 0.), None)[0], axis=-1)
    all_correct = classification_batch(images, predictions, valid)
    all_wrong = classification_batch(images, (predictions + 1) % classes, valid)
    error_method = create_method("sage", reference_name="error")
    assert_direction(
        error_method.compute_direction(task, params, all_correct).direction,
        create_method("sgd").compute_direction(task, params, all_correct).direction,
    )
    assert_direction(
        error_method.compute_direction(task, params, all_wrong).direction,
        create_method("sage", reference_name="loss").compute_direction(task, params, all_wrong).direction,
    )
    np.testing.assert_array_equal(
        error_method.weights(jnp.array([-1., 0., 1.]), jnp.array(True), jnp.ones(3, dtype=bool)),
        [-1., 1., 1.],
    )
    print("PASS: empty-error SGD fallback, all-error E/L equivalence, and zero-score retention")

    method = create_method(config.method.name, reference_name=config.reference.name)
    updater = make_updater(config.training)
    trainer = Trainer(task, method, updater)
    compiled_step = jax.jit(trainer.step)
    state = TrainState(params, optimizer_state=updater.init(params))
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    for step in range(steps):
        direction = expected_direction(state.params, batch, config.method.name, config.reference.name)
        direction = {name: direction[name] + (
            config.training.l2_coefficient * np.asarray(value) if value.ndim > 1 else 0.)
            for name, value in state.params.items()}
        if config.training.momentum != 0:
            velocity = {name: config.training.momentum * velocity[name] + direction[name] for name in direction}
            direction = ({name: direction[name] + config.training.momentum * velocity[name] for name in direction}
                         if config.training.nesterov else velocity)
        rate = updater.learning_rate
        eta = rate(step) if callable(rate) else rate
        expected_params = {name: np.asarray(value) - eta * direction[name] for name, value in state.params.items()}
        state, _ = compiled_step(state, batch)
        assert_direction(state.params, expected_params)
        if int(state.step) != step + 1:
            raise AssertionError("Trainer step did not advance exactly once")
    print(f"PASS: {steps} compiled trainer steps match analytic updates")

    paths = ResultsPaths.resolve(config.results.root_dir, base_dir=ROOT)
    paths.create()
    print(f"Result directories: {paths.logs}, {paths.runs}, {paths.files}")
    print(f"PASS: CPU core sanity check ({config.method.name}, reference={config.reference.name})")
    print(f"Synthetic fixture: 5 valid examples, 1 padding position, {classes} classes")
    print("No ResNet, ImageNet, TPU-device, throughput, or theorem-convergence validation was performed.")
    print("No log files, checkpoints, training metrics, or analysis CSV files were written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet100_resnet50_sage_e.toml",
                        help="Configuration path, relative to the repository root unless absolute")
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of synthetic compiled updates; must be positive")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    if config.method.name not in ("sgd", "sage"):
        parser.error("This sanity check covers full-reference sage and standard sgd only")
    run_sanity(config, args.steps)


if __name__ == "__main__":
    main()
