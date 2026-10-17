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


def make_run_name(config, *, timestamp=None):
    """Use one timezone-aware timestamp across all three experiment directories."""
    timestamp = timestamp or datetime.now().astimezone()
    label = config.method.name
    if label in ("sage", "sage_loo", "dtv", "dtv_loo"):
        label = label.replace("_loo", "") + ("_e" if config.reference.name == "error" else "_l") + ("_loo" if label.endswith("_loo") else "")
    parts = (config.data.name, config.data.mode, config.model.name, label,
             str(config.seed), format(config.training.learning_rate, ".12g"),
             timestamp.strftime("%Y%m%d_%H%M%S"))
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
