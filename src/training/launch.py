"""Dependency-light configuration resolution shared by nohup parent and child."""

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import DATASET_CLASSES, INATURALIST_PREPARED_DIR, MethodConfig, ReferenceConfig, load_config


EXPERIMENT_METHODS = {"sgd": ("sgd", "loss"), "sage_e": ("sage", "error"),
                      "sage_l": ("sage", "loss")}
POLICY_METHODS = {f"sage_{reference}_{mode}": (signal, mode)
                  for reference, signal in (("l", "loss"), ("e", "error"))
                  for mode in ("continuous", "alternating", "half")}


def add_overrides(parser, *, multiple=False):
    """Use identical override names in the launcher and single-run child."""
    from src.probes.mechanism_config import add_arguments
    add_arguments(parser)
    if multiple:
        parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int,
                            help="One or more seeds; omission honors the selected config")
        parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                            choices=(*EXPERIMENT_METHODS, *POLICY_METHODS))
    else:
        parser.add_argument("--seed", type=int)
        parser.add_argument("--method", choices=(*EXPERIMENT_METHODS, *POLICY_METHODS))
    parser.add_argument("--dataset", choices=tuple(DATASET_CLASSES))
    parser.add_argument("--alternating-first", choices=("sgd", "sage"),
                        help="First update of alternating SAGE policies; default sgd. Other methods in a suite are unchanged.")
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
                        help="Constant rate, cosine peak or inverse-square-root initial rate; no automatic batch-size scaling")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine", "inverse_sqrt"))
    parser.add_argument("--min-lr", dest="min_learning_rate", type=float,
                        help="Cosine floor, including zero; default 0")
    parser.add_argument("--checkpoint-every-steps", type=int,
                        help="Save the rolling latest checkpoint every N completed steps; default 100")
    parser.add_argument("--checkpoints", action=argparse.BooleanOptionalAction, default=None,
                        help="Save model/recovery checkpoints; --no-checkpoints disables all slots without disabling CSV/logs")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--eval-every-steps", type=int,
                            help="Run full configured validation every N completed updates")
    evaluation.add_argument("--eval-every-epochs", type=float,
                            help="Run full configured validation every N epochs of sample progress; default 1")
    parser.add_argument("--theorem-probes", action=argparse.BooleanOptionalAction, default=None,
                        help="Control legacy fixed-batch theorem probes independently of real-update diagnostics")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=None,
                        help="Read-only real-update diagnostics; default off; no change to the optimizer")
    parser.add_argument("--diagnostic-steps", nargs="+", type=int,
                        help="Completed updates BEFORE the inspected update: state t observes update t+1")
    for flag in ("update-check", "paired-loss", "transfer"):
        parser.add_argument(f"--diagnostic-{flag}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostic-lr-scales", nargs="+", type=float)
    parser.add_argument("--diagnostic-audit-batches", type=int)
    parser.add_argument("--diagnostic-audit-seed", type=int)


DIAGNOSTIC_OPTIONS = ("theorem_probes", "diagnostics", "diagnostic_steps", "diagnostic_update_check",
                      "diagnostic_paired_loss", "diagnostic_transfer", "diagnostic_lr_scales",
                      "diagnostic_audit_batches", "diagnostic_audit_seed")


def diagnostic_overrides(args):
    """Keep parent, child and shell-suite diagnostic options identical."""
    return {name: getattr(args, name, None) for name in DIAGNOSTIC_OPTIONS}


def diagnostic_cli_args(args):
    command = []
    for name, value in diagnostic_overrides(args).items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, (list, tuple)):
            command.extend([flag, *(str(item) for item in value)])
        else:
            command.extend([flag, str(value)])
    return command


def resolve_config(path, *, mode, steps=None, batch_size=None, seed=None,
                   method=None, dataset=None, model=None, data_mode=None, data_dir=None, l2_coefficient=None,
                   alternating_first=None,
                   momentum=None, nesterov=None, learning_rate=None, lr_schedule=None,
                   min_learning_rate=None, checkpoint_every_steps=None, checkpoints=None,
                   eval_every_steps=None, eval_every_epochs=None, theorem_probes=None,
                   diagnostics=None, diagnostic_steps=None, diagnostic_update_check=None,
                   diagnostic_paired_loss=None, diagnostic_transfer=None, diagnostic_lr_scales=None,
                   diagnostic_audit_batches=None, diagnostic_audit_seed=None, **mechanism_options):
    if eval_every_steps is not None and eval_every_epochs is not None:
        raise ValueError("eval_every_steps and eval_every_epochs are mutually exclusive")
    config = load_config(path)
    if mode not in ("smoke", "train", "validate"):
        raise ValueError("Unknown execution mode")
    if seed is not None:
        config = replace(config, seed=seed)
    if method is not None:
        if method in POLICY_METHODS:
            reference, policy_mode = POLICY_METHODS[method]
            selected_method = MethodConfig("sage_policy", policy_mode=policy_mode)
        else:
            name, reference = EXPERIMENT_METHODS[method]
            selected_method = MethodConfig(name)
        config = replace(config, method=selected_method, reference=ReferenceConfig(reference))
    if alternating_first is not None:
        if alternating_first not in ("sgd", "sage"):
            raise ValueError("alternating_first must be sgd or sage")
        # A common suite option applies only to its alternating jobs. Preserve
        # baseline/continuous/half configs and their identities exactly.
        if config.method.name == "sage_policy" and config.method.policy_mode == "alternating":
            config = replace(config, method=replace(config.method, alternating_first=alternating_first))
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
    if checkpoints is not None:
        config = replace(config, checkpoints=replace(config.checkpoints, enabled=checkpoints))
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
    if theorem_probes is not None:
        config = replace(config, probes=replace(config.probes, enabled=theorem_probes))
    diagnostic_changes = {name: value for name, value in (
        ("enabled", diagnostics), ("steps", diagnostic_steps), ("update_check", diagnostic_update_check),
        ("paired_loss", diagnostic_paired_loss), ("transfer", diagnostic_transfer),
        ("lr_scales", diagnostic_lr_scales), ("audit_batches", diagnostic_audit_batches),
        ("audit_seed", diagnostic_audit_seed)) if value is not None}
    config = replace(config, diagnostics=replace(config.diagnostics, **diagnostic_changes))
    from src.probes.mechanism_config import changes, OPTIONS
    unknown = set(mechanism_options) - set(OPTIONS)
    if unknown:
        raise TypeError(f"Unknown configuration overrides: {sorted(unknown)}")
    config = replace(config, mechanism=replace(config.mechanism, **changes(mechanism_options)))
    config.validate()
    return config


def absolute_path(path, base_dir):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path
