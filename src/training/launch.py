"""Dependency-light configuration resolution shared by nohup parent and child."""

from dataclasses import replace
from pathlib import Path

from src.config import load_config


def resolve_config(path, *, mode, steps=None, batch_size=None):
    config = load_config(path)
    if mode not in ("smoke", "train", "validate"):
        raise ValueError("Unknown execution mode")
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
