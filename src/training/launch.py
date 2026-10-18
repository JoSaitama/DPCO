"""Dependency-light configuration resolution shared by nohup parent and child."""

from dataclasses import replace
from pathlib import Path

from src.config import MethodConfig, ReferenceConfig, load_config


EXPERIMENT_METHODS = {"sgd": ("sgd", "loss"), "sage_e": ("sage", "error"),
                      "sage_l": ("sage", "loss")}


def add_overrides(parser, *, multiple=False):
    """Use identical override names in the launcher and single-run child."""
    if multiple:
        parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int,
                            help="One or more seeds; shipped configs default to 42")
        parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                            choices=tuple(EXPERIMENT_METHODS))
    else:
        parser.add_argument("--seed", type=int)
        parser.add_argument("--method", choices=tuple(EXPERIMENT_METHODS))
    parser.add_argument("--dataset", choices=("imagenet100", "imagenet1k"))
    parser.add_argument("--model", choices=("resnet50", "resnet101"))
    parser.add_argument("--data-mode", "--datamode", dest="data_mode",
                        help="Only clean is implemented; mismatch modes are reserved")


def resolve_config(path, *, mode, steps=None, batch_size=None, seed=None,
                   method=None, dataset=None, model=None, data_mode=None):
    config = load_config(path)
    if mode not in ("smoke", "train", "validate"):
        raise ValueError("Unknown execution mode")
    if seed is not None:
        config = replace(config, seed=seed)
    if method is not None:
        name, reference = EXPERIMENT_METHODS[method]
        config = replace(config, method=MethodConfig(name), reference=ReferenceConfig(reference))
    if dataset is not None:
        config = replace(config, data=replace(config.data, name=dataset,
            num_classes={"imagenet100": 100, "imagenet1k": 1000}[dataset]))
    if model is not None:
        config = replace(config, model=replace(config.model, name=model))
    if data_mode is not None:
        config = replace(config, data=replace(config.data, mode=data_mode))
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
    if mode == "smoke":
        config = replace(config, logging=replace(config.logging, mode="steps", interval=1))
    if mode == "validate":
        config = replace(config, probes=replace(config.probes, enabled=True))
    config.validate()
    return config


def absolute_path(path, base_dir):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path
