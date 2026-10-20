"""Dependency-light configuration resolution shared by nohup parent and child."""

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import DATASET_CLASSES, INATURALIST_PREPARED_DIR, MethodConfig, ReferenceConfig, load_config


EXPERIMENT_METHODS = {"sgd": ("sgd", "loss"), "sage_e": ("sage", "error"),
                      "sage_l": ("sage", "loss")}


def add_overrides(parser, *, multiple=False):
    """Use identical override names in the launcher and single-run child."""
    if multiple:
        parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int,
                            help="One or more seeds; omission honors the selected config")
        parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                            choices=tuple(EXPERIMENT_METHODS))
    else:
        parser.add_argument("--seed", type=int)
        parser.add_argument("--method", choices=tuple(EXPERIMENT_METHODS))
    parser.add_argument("--dataset", choices=tuple(DATASET_CLASSES))
    parser.add_argument("--data-dir", help="Prepared data location, relative to the repository or absolute; iNaturalist uses the subset directory itself")
    parser.add_argument("--model", choices=("resnet50", "resnet101"))
    parser.add_argument("--data-mode", "--datamode", dest="data_mode",
                        help="Only clean is implemented; mismatch modes are reserved")
    parser.add_argument("--l2-coefficient", type=float,
                        help="Coupled L2 on matrix/kernel weights after sample aggregation; default 0 (off)")
    parser.add_argument("--momentum", type=float, help="Momentum coefficient in [0, 1); default 0 (off)")
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=None,
                        help="Use Nesterov momentum; requires --momentum > 0")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float,
                        help="Constant learning rate or cosine peak; not automatically scaled by batch size")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"))
    parser.add_argument("--min-lr", dest="min_learning_rate", type=float,
                        help="Cosine floor, including zero; default 0")
    parser.add_argument("--checkpoint-every-steps", type=int,
                        help="Save the rolling latest checkpoint every N completed steps; default 100")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--eval-every-steps", type=int,
                            help="Run full configured validation every N completed updates")
    evaluation.add_argument("--eval-every-epochs", type=float,
                            help="Run full configured validation every N epochs of sample progress; default 1")


def resolve_config(path, *, mode, steps=None, batch_size=None, seed=None,
                   method=None, dataset=None, model=None, data_mode=None, data_dir=None, l2_coefficient=None,
                   momentum=None, nesterov=None, learning_rate=None, lr_schedule=None,
                   min_learning_rate=None, checkpoint_every_steps=None,
                   eval_every_steps=None, eval_every_epochs=None):
    if eval_every_steps is not None and eval_every_epochs is not None:
        raise ValueError("eval_every_steps and eval_every_epochs are mutually exclusive")
    config = load_config(path)
    if mode not in ("smoke", "train", "validate"):
        raise ValueError("Unknown execution mode")
    if seed is not None:
        config = replace(config, seed=seed)
    if method is not None:
        name, reference = EXPERIMENT_METHODS[method]
        config = replace(config, method=MethodConfig(name), reference=ReferenceConfig(reference))
    if dataset is not None:
        directory = config.data.prepared_dir
        if (dataset == "inaturalist2021") != (config.data.name == "inaturalist2021"):
            directory = INATURALIST_PREPARED_DIR if dataset == "inaturalist2021" else "../data"
        manifest = config.data.class_manifest
        if dataset == "inaturalist2021":
            manifest = None
        elif dataset == "imagenet100" and manifest is None:
            manifest = "configs/manifests/imagenet100_classes.txt"
        config = replace(config, data=replace(config.data, name=dataset,
            num_classes=DATASET_CLASSES[dataset], prepared_dir=directory,
            class_manifest=manifest))
    if data_dir is not None:
        config = replace(config, data=replace(config.data, prepared_dir=data_dir))
    if model is not None:
        config = replace(config, model=replace(config.model, name=model))
    if data_mode is not None:
        config = replace(config, data=replace(config.data, mode=data_mode))
    training_overrides = {name: value for name, value in (
        ("l2_coefficient", l2_coefficient), ("momentum", momentum), ("nesterov", nesterov),
        ("learning_rate", learning_rate), ("lr_schedule", lr_schedule),
        ("min_learning_rate", min_learning_rate)) if value is not None}
    config = replace(config, training=replace(config.training, **training_overrides))
    if checkpoint_every_steps is not None:
        config = replace(config, checkpoints=replace(config.checkpoints,
            latest_every_steps=checkpoint_every_steps))
    if eval_every_steps is not None:
        config = replace(config, evaluation=replace(config.evaluation,
            mode="steps", interval=eval_every_steps))
    elif eval_every_epochs is not None:
        config = replace(config, evaluation=replace(config.evaluation,
            mode="epochs", interval=eval_every_epochs))
    if config.data.mode != "clean":
        raise ValueError("Only data-mode=clean is implemented; mismatch modes are reserved, not silently simulated")
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        config = replace(config, training=replace(config.training, total_steps=steps))
    elif mode == "smoke":
        config = replace(config, training=replace(config.training, total_steps=3))
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config = replace(config, global_batch_size=batch_size)
    elif mode == "smoke":
        # Keep a smoke lightweight; use --batch-size to test a larger batch.
        config = replace(config, global_batch_size=8)
    if mode == "smoke":
        config = replace(config, logging=replace(config.logging, mode="steps", interval=1))
    if mode == "validate":
        config = replace(config, probes=replace(config.probes, enabled=True))
    config.validate()
    return config


def absolute_path(path, base_dir):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path
