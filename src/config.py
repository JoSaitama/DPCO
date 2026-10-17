"""Experiment configuration, importable without accelerator dependencies."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib


METHODS = ("sgd", "sage", "sage_loo", "dtv", "dtv_loo", "random_flip")
REFERENCES = ("loss", "error")


@dataclass(frozen=True)
class DataConfig:
    name: str
    num_classes: int
    image_size: int = 224
    class_manifest: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str


@dataclass(frozen=True)
class MethodConfig:
    name: str = "sgd"
    flip_probability: float = 0.25


@dataclass(frozen=True)
class ReferenceConfig:
    name: str = "loss"


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig
    method: MethodConfig = MethodConfig()
    reference: ReferenceConfig = ReferenceConfig()
    training: TrainingConfig = TrainingConfig()
    seed: int = 0
    global_batch_size: int = 64

    def validate(self) -> None:
        if self.method.name not in METHODS:
            raise ValueError(f"Unknown method: {self.method.name}")
        if self.reference.name not in REFERENCES:
            raise ValueError(f"Unknown reference: {self.reference.name}")
        if not 0 <= self.method.flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        classes = {"imagenet100": 100, "imagenet1k": 1000}
        if self.data.name not in classes:
            raise ValueError(f"Unsupported dataset: {self.data.name}")
        if self.data.num_classes != classes[self.data.name]:
            raise ValueError("num_classes must match the selected dataset")
        if self.data.name == "imagenet100" and not self.data.class_manifest:
            raise ValueError("imagenet100 requires a fixed class_manifest path")
        if self.model.name not in ("resnet50", "resnet101"):
            raise ValueError(f"Unsupported model: {self.model.name}")
        if self.global_batch_size <= 0 or self.data.image_size <= 0:
            raise ValueError("Batch size and image size must be positive")
        if not math.isfinite(self.training.learning_rate) or self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = RunConfig(
        data=DataConfig(**raw.pop("data")),
        model=ModelConfig(**raw.pop("model")),
        method=MethodConfig(**raw.pop("method", {})),
        reference=ReferenceConfig(**raw.pop("reference", {})),
        training=TrainingConfig(**raw.pop("training", {})),
        **raw,
    )
    config.validate()
    return config
