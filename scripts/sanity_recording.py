"""CPU sanity: exercise recording/checkpoint/probe paths using synthetic batches.

No configured ResNet or ImageNet data is instantiated. Without --output-root,
artifacts are created in a temporary directory and removed after verification.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exercise(output_root, config_path, enable_probes):
    import jax
    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp
    import numpy as np

    from src.config import (CheckpointConfig, EvaluationConfig, LoggingConfig,
                            ProbeConfig, ResultsConfig, load_config)
    from src.data.batch import ClassificationBatch
    from src.methods import create_method
    from src.probes.classification import ClassificationProbes, ProbeBatch
    from src.references import create_reference
    from src.tasks.classification import ClassificationTask
    from src.training import TrainState, Trainer
    from src.training.evaluation import EvaluationSpec
    from src.training.experiment import ExperimentRunner
    from src.training.updates import make_updater
    from src.training.schedules import sage_warmup_steps

    config = load_config(config_path)
    config = replace(config, results=ResultsConfig(str(output_root)),
        training=replace(config.training, total_steps=4),
        logging=LoggingConfig("steps", 1, 3, 3), evaluation=EvaluationConfig("epochs", .5),
        checkpoints=CheckpointConfig(2, True),
        probes=ProbeConfig(enabled=enable_probes, batches_per_node=1))

    class Source:
        def __init__(self):
            self.index = 0
            self.batch = ClassificationBatch(np.array([[1.], [-2.], [-2.]], np.float32),
                np.zeros(3, np.int32), np.ones(3, bool))

        def __next__(self):
            self.index += 1
            return self.batch

        def snapshot(self):
            return {"index": self.index}

        def restore(self, state):
            self.index = state["index"]

    source = Source()
    def model_fn(params, images, state):
        return images @ params["w"] + params["b"], state
    classes = config.data.num_classes
    task = ClassificationTask(model_fn, classes)
    state = TrainState({"w": jnp.zeros((1, classes)).at[0, 1].set(1.),
        "b": jnp.full(classes, -30.).at[:2].set(0.)})
    trainer = Trainer(task, create_method(config.method.name, reference_name=config.reference.name,
                                         flip_probability=config.method.flip_probability,
                                         policy_mode=config.method.policy_mode,
                                         alternating_first=config.method.alternating_first,
                                         warmup_steps=sage_warmup_steps(config)),
                      make_updater(config.training))
    evaluation = EvaluationSpec(task, lambda: iter([source.batch]), 3, "deterministic_linear")
    def audit(p):
        return jnp.mean(task.evaluate(p, source.batch).losses)
    probes = ClassificationProbes(task, create_reference(config.reference.name),
        [ProbeBatch("synthetic_fixed_batch", source.batch)], reference_name=config.reference.name,
        objective_mode="deterministic_linear", audit_objective=audit, audit_name="synthetic_fixed_audit")
    runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
        train_sample_count=6, probes=probes,
        metadata={"execution_kind": "synthetic_cpu_sanity", "actual_model": "linear_softmax",
                  "actual_dataset": "three_synthetic_examples", "actual_valid_batch_size": 3,
                  "configured_model_and_dataset_are_not_loaded": True})
    final = runner.run(state)
    if int(final.step) != 4 or source.index != 4:
        raise AssertionError("Warmup/probes must not consume training examples")
    import csv
    with (runner.paths.files / "train_steps.csv").open() as handle:
        if len(list(csv.DictReader(handle))) != 4:
            raise AssertionError("Expected every step, including the final partial CSV buffer")
    if not (runner.paths.logs / "metadata.json").exists():
        raise AssertionError("Missing metadata")
    print(f"PASS: buffered CSV, full validation, seven-slot checkpoints, timing; probes={enable_probes}")
    print(f"Artifacts: {runner.paths.parent.root}; run={runner.paths.name}")
    print("Synthetic CPU only: no ImageNet, ResNet, TPU, speed, or convergence claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet100_resnet50_sage_e.toml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--probes", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if args.output_root:
        root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
        exercise(root, config_path, args.probes)
    else:
        with TemporaryDirectory(prefix="sage-recording-sanity-") as directory:
            exercise(Path(directory), config_path, args.probes)
        print("Temporary sanity artifacts removed; pass --output-root to retain them.")


if __name__ == "__main__":
    main()
