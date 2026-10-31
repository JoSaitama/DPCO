"""Shared result paths; recording schemas and checkpoint policies are separate."""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re


@dataclass(frozen=True)
class ResultsPaths:
    root: Path

    @classmethod
    def resolve(cls, root_dir: str, *, base_dir: Path):
        """Resolve relative output roots against an explicit project directory."""
        root = Path(root_dir).expanduser()
        if not root.is_absolute():
            root = Path(base_dir) / root
        return cls(root.resolve())

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def files(self) -> Path:
        return self.root / "files"

    def create(self) -> None:
        """Create directories only; never write logs, checkpoints, or CSV data."""
        for path in (self.logs, self.runs, self.files):
            path.mkdir(parents=True, exist_ok=True)

    def experiment(self, run_name):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in (".", ".."):
            raise ValueError("run_name must be one safe directory component")
        return ExperimentPaths(self, run_name)


def method_label(config):
    """Label optimizer and EMA weights without changing historical SGD labels."""
    label = config.method.name
    if label == "dpco_policy":
        label = "dpco_" + ("e" if config.reference.name == "error" else "l") + "_" + config.method.policy_mode
        if config.method.policy_mode == "alternating" and config.method.alternating_first == "dpco":
            label += "_first_dpco"
    if config.reference.name == "ema":
        label = "dpco_ema_b" + format(config.reference.beta, ".12g") + "_w" + format(config.method.negative_weight, ".12g").replace("+", "")
    elif label in ("dpco", "dpco_loo", "dtv", "dtv_loo"):
        suffix = {"error": "_e", "loss": "_l", "correct": "_r"}[config.reference.name]
        label = label.replace("_loo", "") + suffix + ("_loo" if label.endswith("_loo") else "")
    if config.method.dpco_warmup_percent:
        label += "_warmup" + format(config.method.dpco_warmup_percent, ".12g") + "pct"
    if config.training.optimizer == "adam" and config.method.name != "adam":
        label += "_adam"
    return label


def make_run_name(config, *, timestamp=None):
    """Use one timezone-aware timestamp across all three experiment directories."""
    timestamp = timestamp or datetime.now().astimezone()
    label = method_label(config)
    parts = (config.data.name, config.data.mode, config.model.name, label,
             str(config.seed), format(config.training.learning_rate, ".12g"))
    if config.data.name == "cifar100n":
        parts += (config.data.noise_type, "human" if config.data.is_human else f"synthetic_s{config.data.noise_seed}")
    elif config.data.name == "mnist":
        parts += ("symmetric" + format(config.data.label_noise_percent, ".12g"),
                  f"noise_s{config.data.noise_seed}")
    elif config.data.name == "cnwl":
        parts += ("red" + str(config.data.cnwl_noise_percent), "official_split")
    # Label newly created decaying-rate runs in logs/files/runs and lifecycle
    # records. Constant names and explicit historical resume names stay unchanged.
    if config.training.lr_schedule != "constant":
        parts += (config.training.lr_schedule,)
    parts += (timestamp.strftime("%Y%m%d_%H%M%S"),)
    name = "_".join(parts)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Run-name fields must be safe directory components")
    return name


@dataclass(frozen=True)
class ExperimentPaths:
    parent: ResultsPaths
    name: str

    @property
    def logs(self):
        return self.parent.logs / self.name

    @property
    def runs(self):
        return self.parent.runs / self.name

    @property
    def files(self):
        return self.parent.files / self.name

    def create(self, *, resume=False):
        paths = (self.logs, self.runs, self.files)
        if resume:
            if not all(path.is_dir() for path in paths):
                raise FileNotFoundError("Resume requires all existing experiment directories")
        else:
            if any(path.exists() for path in paths):
                raise FileExistsError(f"Experiment already exists: {self.name}")
            for path in paths:
                path.mkdir(parents=True, exist_ok=False)

    def reserve(self):
        """Reserve a new run before nohup opens its persistent stdout file."""
        self.create()
        (self.runs / ".launch_reserved").touch(exist_ok=False)

    def claim_reserved(self):
        if not all(path.is_dir() for path in (self.logs, self.runs, self.files)):
            raise FileNotFoundError("Missing launch reservation directories")
        if (self.logs / "metadata.json").exists():
            raise FileExistsError("A new launch must not overwrite an existing run")
        (self.runs / ".launch_reserved").unlink()
