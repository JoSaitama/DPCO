"""Experiment configuration, importable without accelerator dependencies."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib


METHODS = ("sgd", "sage", "sage_loo", "dtv", "dtv_loo", "random_flip", "sage_policy")
REFERENCES = ("loss", "error")
DATASET_CLASSES = {"imagenet100": 100, "imagenet1k": 1000, "inaturalist2021": 10000}
INATURALIST_PREPARED_DIR = "../data/inaturalist2021_full_t140000_v20000_s42"


@dataclass(frozen=True)
class DataConfig:
    name: str
    num_classes: int
    image_size: int = 224
    class_manifest: str | None = None
    mode: str = "clean"
    hf_repo_id: str = "ILSVRC/imagenet-1k"
    hf_revision: str = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
    prepared_dir: str = "../data"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    initialization: str = "random"
    precision: str = "float32"


@dataclass(frozen=True)
class MethodConfig:
    name: str = "sgd"
    flip_probability: float = 0.25
    policy_mode: str = "continuous"
    alternating_first: str = "sgd"


@dataclass(frozen=True)
class ReferenceConfig:
    name: str = "loss"


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    total_steps: int = 100
    l2_coefficient: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False
    lr_schedule: str = "constant"
    min_learning_rate: float = 0.0


@dataclass(frozen=True)
class LoggingConfig:
    mode: str = "steps"
    interval: float = 5
    csv_flush_steps: int = 50
    timing_window_steps: int = 50


@dataclass(frozen=True)
class EvaluationConfig:
    mode: str = "epochs"
    interval: float = 1


@dataclass(frozen=True)
class CheckpointConfig:
    latest_every_steps: int = 100
    at_epoch_end: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class ProbeConfig:
    enabled: bool = False
    theorem1: bool = True
    theorem2: bool = True
    theorem3: bool = True
    fractions: tuple[float, ...] = (0., .25, .5, .75, 1.)
    batches_per_node: int = 3
    lr_scales: tuple[float, ...] = (1., .5, .25, .125, .0625)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Optional observers of real updates; steps count PRE-update states.

    State step t observes update t+1. Nodes beyond the update budget are inactive,
    so a short smoke can use the same configuration as a longer diagnostic run.
    """
    enabled: bool = False
    steps: tuple[int, ...] = (0, 5, 10, 20, 25, 30, 50, 75, 100)
    update_check: bool = True
    paired_loss: bool = True
    transfer: bool = True
    lr_scales: tuple[float, ...] = (1., .5, .25, .125, .0625)
    audit_batches: int = 2
    audit_seed: int = 42


@dataclass(frozen=True)
class ResultsConfig:
    root_dir: str = "results"


@dataclass(frozen=True)
class ExecutionConfig:
    visible_chips: str = "0,1,2,3"
    chips_per_process_bounds: str = "2,2,1"
    port: int = 8476
    startup_timeout_s: int = 180
    gradient_backend: str = "audit"


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig
    method: MethodConfig = MethodConfig()
    reference: ReferenceConfig = ReferenceConfig()
    training: TrainingConfig = TrainingConfig()
    results: ResultsConfig = ResultsConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    checkpoints: CheckpointConfig = CheckpointConfig()
    probes: ProbeConfig = ProbeConfig()
    diagnostics: DiagnosticConfig = DiagnosticConfig()
    execution: ExecutionConfig = ExecutionConfig()
    seed: int = 42
    global_batch_size: int = 256

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.method.name not in METHODS:
            raise ValueError(f"Unknown method: {self.method.name}")
        if self.method.policy_mode not in ("continuous", "alternating", "half"):
            raise ValueError("policy_mode must be continuous, alternating or half")
        if self.method.name != "sage_policy" and self.method.policy_mode != "continuous":
            raise ValueError("Nondefault policy_mode requires method=sage_policy")
        if self.method.alternating_first not in ("sgd", "sage"):
            raise ValueError("alternating_first must be sgd or sage")
        if self.method.alternating_first != "sgd" and not (
                self.method.name == "sage_policy" and self.method.policy_mode == "alternating"):
            raise ValueError("A nondefault alternating_first requires the alternating SAGE policy")
        if self.reference.name not in REFERENCES:
            raise ValueError(f"Unknown reference: {self.reference.name}")
        if not 0 <= self.method.flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        classes = DATASET_CLASSES
        if self.data.name not in classes:
            raise ValueError(f"Unsupported dataset: {self.data.name}")
        if self.data.num_classes != classes[self.data.name]:
            raise ValueError("num_classes must match the selected dataset")
        if self.data.name == "imagenet100" and not self.data.class_manifest:
            raise ValueError("imagenet100 requires a fixed class_manifest path")
        if self.model.name not in ("resnet50", "resnet101"):
            raise ValueError(f"Unsupported model: {self.model.name}")
        if self.model.initialization != "random":
            raise ValueError("The current optimizer comparison uses random initialization, not pretrained weights")
        if self.model.precision not in ("float32", "bfloat16"):
            raise ValueError("model.precision must be float32 or bfloat16")
        if self.execution.gradient_backend not in ("audit", "jvp"):
            raise ValueError("Unknown gradient backend")
        if self.execution.gradient_backend == "jvp" and self.method.name.endswith("_loo"):
            raise ValueError("The JVP backend currently supports full common references only")
        if (not isinstance(self.execution.port, int) or not isinstance(self.execution.startup_timeout_s, int)
                or not 1024 <= self.execution.port <= 65535 or self.execution.startup_timeout_s <= 0):
            raise ValueError("Invalid runtime port or startup timeout")
        for bounds in (self.execution.visible_chips, self.execution.chips_per_process_bounds):
            if not bounds or any(not part.isdigit() for part in bounds.split(",")):
                raise ValueError("TPU chip settings must be comma-separated nonnegative integers")
        dimensions = self.execution.chips_per_process_bounds.split(",")
        if len(dimensions) != 3 or any(int(part) <= 0 for part in dimensions):
            raise ValueError("TPU chip bounds require three positive dimensions")
        chips = self.execution.visible_chips.split(",")
        if len(set(chips)) != len(chips) or math.prod(int(part) for part in dimensions) != len(chips):
            raise ValueError("Visible TPU chips must be unique and match the chip-bounds product")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.global_batch_size, self.data.image_size)):
            raise ValueError("Batch size and image size must be positive integers")
        if not math.isfinite(self.training.learning_rate) or self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if (isinstance(self.training.l2_coefficient, bool)
                or not math.isfinite(self.training.l2_coefficient)
                or self.training.l2_coefficient < 0):
            raise ValueError("l2_coefficient must be nonnegative and finite")
        if self.probes.enabled and self.training.l2_coefficient != 0:
            raise ValueError("Current theorem probes require l2_coefficient=0; their objectives exclude regularization")
        if (isinstance(self.training.momentum, bool) or not math.isfinite(self.training.momentum)
                or not 0 <= self.training.momentum < 1):
            raise ValueError("momentum must be finite and in [0, 1)")
        if not isinstance(self.training.nesterov, bool):
            raise ValueError("nesterov must be a boolean")
        if self.training.nesterov and self.training.momentum == 0:
            raise ValueError("nesterov requires nonzero momentum")
        if self.probes.enabled and self.training.momentum != 0:
            raise ValueError("Current theorem probes require momentum=0; their updates exclude momentum")
        if self.training.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")
        if (isinstance(self.training.min_learning_rate, bool)
                or not math.isfinite(self.training.min_learning_rate)
                or not 0 <= self.training.min_learning_rate <= self.training.learning_rate):
            raise ValueError("min_learning_rate must be finite and in [0, learning_rate]")
        if self.training.lr_schedule == "constant" and self.training.min_learning_rate != 0:
            raise ValueError("min_learning_rate requires lr_schedule=cosine")
        if not isinstance(self.results.root_dir, str) or not self.results.root_dir.strip():
            raise ValueError("results.root_dir must be a nonempty path")
        from src.training.schedules import validate_interval
        if not isinstance(self.checkpoints.enabled, bool):
            raise ValueError("checkpoints.enabled must be a boolean")
        validate_interval(self.logging.mode, self.logging.interval, ("steps", "percent"))
        validate_interval(self.evaluation.mode, self.evaluation.interval, ("steps", "epochs"))
        for value in (self.training.total_steps, self.logging.csv_flush_steps,
                      self.logging.timing_window_steps, self.checkpoints.latest_every_steps,
                      self.probes.batches_per_node):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Step counts and probe batch counts must be positive integers")
        if not self.probes.fractions or any(not math.isfinite(p) or not 0 <= p <= 1
                                            for p in self.probes.fractions):
            raise ValueError("Probe fractions must be finite values in [0, 1]")
        if not self.probes.lr_scales or any(not math.isfinite(s) or s <= 0
                                            for s in self.probes.lr_scales):
            raise ValueError("Probe learning-rate scales must be positive and finite")
        if self.probes.enabled and self.method.name != "sage":
            raise ValueError("Common-reference theorem probes currently require full sage")
        d = self.diagnostics
        for flag in (d.enabled, d.update_check, d.paired_loss, d.transfer):
            if not isinstance(flag, bool):
                raise ValueError("Diagnostic switches must be booleans")
        if (not d.steps or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in d.steps)
                or len(set(d.steps)) != len(d.steps)):
            raise ValueError("Diagnostic steps must be distinct nonnegative completed-update counts")
        if not d.lr_scales or any(isinstance(s, bool) or not math.isfinite(s) or s <= 0 for s in d.lr_scales):
            raise ValueError("Diagnostic learning-rate scales must be positive and finite")
        if isinstance(d.audit_batches, bool) or not isinstance(d.audit_batches, int) or d.audit_batches <= 0:
            raise ValueError("Diagnostic audit_batches must be a positive integer")
        if isinstance(d.audit_seed, bool) or not isinstance(d.audit_seed, int) or not 0 <= d.audit_seed < 2**32:
            raise ValueError("Diagnostic audit_seed must be an integer in [0, 2**32)")
        if d.enabled:
            if not (d.update_check or d.paired_loss or d.transfer):
                raise ValueError("Enable at least one diagnostic check")
            if self.method.name not in ("sgd", "sage", "sage_policy"):
                raise ValueError("Real-update diagnostics cover SGD and full common-reference SAGE only")
            if self.training.momentum != 0 or self.training.l2_coefficient != 0:
                raise ValueError("Real-update diagnostics require momentum=0 and l2_coefficient=0")
        if not self.data.mode or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." for c in self.data.mode):
            raise ValueError("data.mode must be a safe name such as clean or mismatch10")

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
        results=ResultsConfig(**raw.pop("results", {})),
        logging=LoggingConfig(**raw.pop("logging", {})),
        evaluation=EvaluationConfig(**raw.pop("evaluation", {})),
        checkpoints=CheckpointConfig(**raw.pop("checkpoints", {})),
        probes=ProbeConfig(**raw.pop("probes", {})),
        diagnostics=DiagnosticConfig(**raw.pop("diagnostics", {})),
        execution=ExecutionConfig(**raw.pop("execution", {})),
        **raw,
    )
    config.validate()
    return config
