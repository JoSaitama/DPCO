"""Dependency-light configuration resolution shared by nohup parent and child."""

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import DATASET_CLASSES, INATURALIST_PREPARED_DIR, MODELS, MethodConfig, ReferenceConfig, load_config


EXPERIMENT_METHODS = {"adam": ("adam", "loss"), "dpco": ("dpco", "ema"),
                      "dpco_filter": ("dpco", "ema"), "dpco_flip": ("dpco", "ema"), "sgd": ("sgd", "loss"), "dpco_e": ("dpco", "error"),
                      "dpco_l": ("dpco", "loss"), "dpco_r": ("dpco", "correct")}
POLICY_METHODS = {f"dpco_{reference}_{mode}": (signal, mode)
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
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--reference-beta", type=float, help="Raw batch-gradient EMA decay; default 0.9")
    parser.add_argument("--negative-weight", type=float,
                        help="DPCO contribution for alignment < 0: 0 filter, -1 flip; any finite scalar")
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--adam-epsilon", type=float)
    parser.add_argument("--noise-type", choices=("clean100", "noisy100", "symmetric", "red"),
                        help="CIFAR: clean100/noisy100; MNIST: symmetric; CNWL: red")
    parser.add_argument("--cnwl-noise-percent", type=int, choices=(0, 5, 10, 15, 20, 30, 40, 50, 60, 80),
                        help="CNWL only: select the author's Red split; never generates random labels")
    parser.add_argument("--is-human", action=argparse.BooleanOptionalAction, default=None,
                        help="Use released labels (CNWL: human-verified web noise); CIFAR --no-is-human samples the empirical transition matrix")
    parser.add_argument("--noise-seed", type=int, help="Synthetic labels only; official default 0")
    parser.add_argument("--noise-manifest", help="ImageNet only: fixed CLIP noise_manifest.json; sets noisy/clip_idn mode")
    parser.add_argument("--noise-percent", "--expected-noise-percent", dest="noise_percent", type=float,
                        help="Expected ACTUAL mismatch percentage (0..100), checked after loading; never generates noise")
    parser.add_argument("--label-noise-percent", type=float,
                        help="MNIST only: generate this percentage of fixed wrong labels (0..100); 0 is clean")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None,
                        help="Allow dataset source downloads; CNWL still needs original ImageNet sources in data-dir/raw")
    parser.add_argument("--dataset", choices=tuple(DATASET_CLASSES))
    parser.add_argument("--alternating-first", choices=("sgd", "dpco"),
                        help="First update of alternating DPCO policies; default sgd. Other methods in a suite are unchanged.")
    parser.add_argument("--dpco-warmup-percent", type=float,
                        help="Initial percentage of updates using SGD before original/continuous DPCO; 0 (default) disables it. Other suite methods are unchanged.")
    parser.add_argument("--data-dir", help="Prepared data location, relative to the repository or absolute; iNaturalist uses the subset directory itself")
    parser.add_argument("--model", choices=MODELS,
                        help="resnet50/101 keep existing ImageNet stem; cifar_resnet* use native stem; mnist_cnn/mlp have no BN")
    parser.add_argument("--data-mode", "--datamode", dest="data_mode",
                        help="ImageNet/iNaturalist: clean only; CIFAR/MNIST: must match noise settings")
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


CIFAR_OPTIONS = ("noise_type", "is_human", "noise_seed", "noise_percent", "download", "label_noise_percent", "cnwl_noise_percent", "noise_manifest")


def cifar_overrides(args):
    return {name: getattr(args, name, None) for name in CIFAR_OPTIONS}


def cifar_cli_args(args):
    command = []
    for name, value in cifar_overrides(args).items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + flag[2:])
        else:
            command.extend((flag, str(value)))
    return command


DPCO_OPTIONS = ("optimizer", "reference_beta", "negative_weight", "adam_beta1", "adam_beta2", "adam_epsilon")


def dpco_overrides(args):
    return {name: getattr(args, name, None) for name in DPCO_OPTIONS}


def dpco_cli_args(args):
    return [part for name, value in dpco_overrides(args).items() if value is not None
            for part in ("--" + name.replace("_", "-"), str(value))]


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
                   alternating_first=None, dpco_warmup_percent=None,
                   momentum=None, nesterov=None, learning_rate=None, lr_schedule=None,
                   min_learning_rate=None, optimizer=None, reference_beta=None, negative_weight=None,
                   adam_beta1=None, adam_beta2=None, adam_epsilon=None,
                   noise_type=None, is_human=None, noise_seed=None, noise_percent=None, download=None,
                   label_noise_percent=None,
                   cnwl_noise_percent=None,
                   noise_manifest=None,
                   checkpoint_every_steps=None, checkpoints=None,
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
            selected_method = MethodConfig("dpco_policy", policy_mode=policy_mode)
        else:
            name, reference = EXPERIMENT_METHODS[method]
            weight = {"dpco_filter": 0., "dpco_flip": -1.}.get(method,
                config.method.negative_weight if config.reference.name == "ema" else -1.)
            selected_method = MethodConfig(name, negative_weight=weight if reference == "ema" else -1.)
        config = replace(config, method=selected_method,
                         reference=replace(config.reference, name=reference))
        if method in ("adam", "dpco", "dpco_filter", "dpco_flip"):
            config = replace(config, training=replace(config.training, optimizer="adam", momentum=0., nesterov=False))
        elif method == "sgd":
            config = replace(config, training=replace(config.training, optimizer="sgd"))
    if reference_beta is not None:
        config = replace(config, reference=replace(config.reference, beta=reference_beta))
    if negative_weight is not None:
        import math
        if isinstance(negative_weight, bool) or not math.isfinite(negative_weight):
            raise ValueError("negative_weight must be a finite number")
    if negative_weight is not None and config.reference.name == "ema":
        if method in ("dpco_filter", "dpco_flip") and negative_weight != config.method.negative_weight:
            raise ValueError("Alias conflicts with negative_weight; use --method dpco for custom weights")
        config = replace(config, method=replace(config.method, negative_weight=negative_weight))
    elif negative_weight is not None and config.method.name != "adam":
        raise ValueError("negative_weight requires dpco with EMA reference")
    if alternating_first is not None:
        if alternating_first not in ("sgd", "dpco"):
            raise ValueError("alternating_first must be sgd or dpco")
        # A common suite option applies only to its alternating jobs. Preserve
        # baseline/continuous/half configs and their identities exactly.
        if config.method.name == "dpco_policy" and config.method.policy_mode == "alternating":
            config = replace(config, method=replace(config.method, alternating_first=alternating_first))
    if dpco_warmup_percent is not None:
        import math
        if (isinstance(dpco_warmup_percent, bool) or not math.isfinite(dpco_warmup_percent)
                or not 0 <= dpco_warmup_percent <= 100):
            raise ValueError("dpco_warmup_percent must be finite and in [0, 100]")
        # A mixed suite can share the option without changing SGD, alternating
        # or half policies. Direct TOML configurations reject invalid combinations.
        if config.method.name == "dpco" or (config.method.name == "dpco_policy"
                and config.method.policy_mode == "continuous"):
            config = replace(config, method=replace(config.method, dpco_warmup_percent=dpco_warmup_percent))
    if dataset is not None:
        directory = config.data.prepared_dir
        if dataset != config.data.name:
            if dataset in ("cifar100n", "mnist", "cnwl"):
                directory = "../data/" + dataset
            elif dataset == "inaturalist2021":
                directory = INATURALIST_PREPARED_DIR
            elif config.data.name in ("inaturalist2021", "cifar100n", "mnist", "cnwl"):
                directory = "../data"
        manifest = config.data.class_manifest
        if dataset in ("inaturalist2021", "cifar100n", "mnist", "cnwl"):
            manifest = None
        elif dataset == "imagenet100" and manifest is None:
            manifest = "configs/manifests/imagenet100_classes.txt"
        changes = dict(name=dataset, num_classes=DATASET_CLASSES[dataset],
                       prepared_dir=directory, class_manifest=manifest)
        if dataset != config.data.name:
            changes["cnwl_noise_percent"] = 20
            changes["noise_manifest"] = None
        if dataset == "cnwl" and config.data.name != "cnwl":
            changes.update(image_size=84, mode="noisy", noise_type="red", is_human=True,
                           noise_seed=0, noise_percent=None, label_noise_percent=0., download=False)
            if model is None:
                config = replace(config, model=replace(config.model, name="resnet50"))
        elif dataset == "mnist" and config.data.name != "mnist":
            changes.update(image_size=28, mode="noisy", noise_type="symmetric", is_human=False,
                           noise_seed=0, noise_percent=None, label_noise_percent=20., download=False)
            if model is None:
                config = replace(config, model=replace(config.model, name="mnist_cnn"))
        elif dataset == "cifar100n" and config.data.name != "cifar100n":
            changes.update(image_size=32, mode="noisy", noise_type="noisy100", is_human=True,
                           noise_seed=0, noise_percent=None, label_noise_percent=0., download=False)
            if model is None:
                config = replace(config, model=replace(config.model, name="cifar_resnet34"))
        elif dataset not in ("cifar100n", "mnist", "cnwl") and config.data.name in ("cifar100n", "mnist", "cnwl"):
            changes.update(image_size=224, mode="clean", noise_type=None, is_human=True,
                           noise_seed=0, noise_percent=None, download=False, label_noise_percent=0.)
            if model is None:
                config = replace(config, model=replace(config.model, name="resnet50"))
        config = replace(config, data=replace(config.data, **changes))
    noise_changes = {key: value for key, value in (("noise_type", noise_type), ("is_human", is_human),
        ("noise_seed", noise_seed), ("noise_percent", noise_percent), ("download", download),
        ("label_noise_percent", label_noise_percent)) if value is not None}
    if cnwl_noise_percent is not None:
        noise_changes["cnwl_noise_percent"] = cnwl_noise_percent
        if config.data.name == "cnwl":
            if data_mode is None:
                noise_changes["mode"] = "clean" if cnwl_noise_percent == 0 else "noisy"
            if noise_percent is None:
                noise_changes["noise_percent"] = None
    if config.data.name == "mnist":
        if label_noise_percent is not None:
            if data_mode is None:
                noise_changes["mode"] = "noisy" if label_noise_percent > 0 else "clean"
            if noise_percent is None:
                noise_changes["noise_percent"] = None
        elif data_mode == "clean":
            noise_changes["label_noise_percent"] = 0.
            if noise_percent is None:
                noise_changes["noise_percent"] = None
    if noise_type is not None and config.data.name == "cifar100n" and data_mode is None:
        noise_changes["mode"] = "clean" if noise_type == "clean100" else "noisy"
    if (config.data.name == "cifar100n" and noise_percent is None
            and any(value is not None for value in (noise_type, is_human, noise_seed))):
        active_type = noise_changes.get("noise_type", config.data.noise_type)
        noise_changes["noise_percent"] = 0. if active_type == "clean100" else None
    config = replace(config, data=replace(config.data, **noise_changes))
    if noise_manifest is not None:
        config = replace(config, data=replace(config.data, noise_manifest=str(noise_manifest),
            noise_type="clip_idn", is_human=False, mode="noisy"))
    if data_dir is not None:
        config = replace(config, data=replace(config.data, prepared_dir=data_dir))
    if model is not None:
        config = replace(config, model=replace(config.model, name=model))
    if data_mode is not None:
        config = replace(config, data=replace(config.data, mode=data_mode))
    training_overrides = {name: value for name, value in (
        ("l2_coefficient", l2_coefficient), ("momentum", momentum), ("nesterov", nesterov),
        ("learning_rate", learning_rate), ("lr_schedule", lr_schedule),
        ("min_learning_rate", min_learning_rate), ("optimizer", optimizer),
        ("adam_beta1", adam_beta1), ("adam_beta2", adam_beta2),
        ("adam_epsilon", adam_epsilon)) if value is not None}
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
    if config.data.mode != "clean" and config.data.name not in ("cifar100n", "mnist", "cnwl") and not config.data.noise_manifest:
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
