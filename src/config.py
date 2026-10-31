"""Experiment configuration, importable without accelerator dependencies."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib

from src.probes.mechanism_config import MechanismConfig


METHODS = ("sgd", "adam", "dpco", "dpco_loo", "dtv", "dtv_loo", "random_flip", "dpco_policy")
REFERENCES = ("loss", "error", "correct", "ema")
DATASET_CLASSES = {"imagenet100": 100, "imagenet1k": 1000, "inaturalist2021": 10000, "cifar100n": 100, "mnist": 10, "cnwl": 100}
IMAGENET_MODELS = ("resnet50", "resnet101")
CIFAR_MODELS = ("cifar_resnet18", "cifar_resnet34", "cifar_resnet50")
MNIST_MODELS = ("mnist_cnn", "mnist_mlp")
MODELS = (*IMAGENET_MODELS, *CIFAR_MODELS, *MNIST_MODELS)
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
    noise_type: str | None = None
    is_human: bool = True
    noise_seed: int = 0
    noise_percent: float | None = None
    download: bool = False
    # MNIST generator control; noise_percent remains an actual-rate assertion.
    label_noise_percent: float = 0.0
    # CNWL official Red split selector, NOT a label-noise generator.
    cnwl_noise_percent: int = 20
    # Immutable offline CLIP artifact; training never generates/changes these labels.
    noise_manifest: str | None = None


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
    dpco_warmup_percent: float = 0.0
    negative_weight: float = -1.0


@dataclass(frozen=True)
class ReferenceConfig:
    name: str = "loss"
    beta: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    total_steps: int = 100
    l2_coefficient: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False
    lr_schedule: str = "constant"
    min_learning_rate: float = 0.0
    optimizer: str = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


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
    mechanism: MechanismConfig = MechanismConfig()

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.method.name not in METHODS:
            raise ValueError(f"Unknown method: {self.method.name}")
        if self.method.policy_mode not in ("continuous", "alternating", "half"):
            raise ValueError("policy_mode must be continuous, alternating or half")
        if self.method.name != "dpco_policy" and self.method.policy_mode != "continuous":
            raise ValueError("Nondefault policy_mode requires method=dpco_policy")
        if self.method.alternating_first not in ("sgd", "dpco"):
            raise ValueError("alternating_first must be sgd or dpco")
        if self.method.alternating_first != "sgd" and not (
                self.method.name == "dpco_policy" and self.method.policy_mode == "alternating"):
            raise ValueError("A nondefault alternating_first requires the alternating DPCO policy")
        percent = self.method.dpco_warmup_percent
        if isinstance(percent, bool) or not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("dpco_warmup_percent must be finite and in [0, 100]")
        if percent and not (self.method.name == "dpco" or (
                self.method.name == "dpco_policy" and self.method.policy_mode == "continuous")):
            raise ValueError("DPCO warmup supports original dpco and continuous DPCO only")
        if percent and self.probes.enabled:
            raise ValueError("DPCO warmup requires legacy theorem probes off; use diagnostics/mechanisms for actual-phase checks")
        if self.reference.name not in REFERENCES:
            raise ValueError(f"Unknown reference: {self.reference.name}")
        if (isinstance(self.reference.beta, bool) or not math.isfinite(self.reference.beta)
                or not 0 <= self.reference.beta < 1):
            raise ValueError("reference beta must be finite and in [0, 1)")
        if (isinstance(self.method.negative_weight, bool) or not math.isfinite(self.method.negative_weight)):
            raise ValueError("negative_weight must be a finite number")
        if self.reference.name == "ema" and (
                self.method.name != "dpco" or self.method.policy_mode != "continuous" or percent):
            raise ValueError("EMA reference requires continuous dpco without method warmup")
        if self.method.negative_weight != -1 and self.reference.name != "ema":
            raise ValueError("negative_weight requires dpco with EMA reference")
        if self.training.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be sgd or adam")
        if self.method.name == "adam" and self.training.optimizer != "adam":
            raise ValueError("Plain adam method requires optimizer=adam")
        for beta in (self.training.adam_beta1, self.training.adam_beta2):
            if isinstance(beta, bool) or not math.isfinite(beta) or not 0 <= beta < 1:
                raise ValueError("Adam betas must be finite and in [0, 1)")
        epsilon = self.training.adam_epsilon
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Adam epsilon must be finite and positive")
        if self.training.optimizer == "adam" and (self.training.momentum or self.training.nesterov):
            raise ValueError("Adam uses its own betas; SGD momentum/Nesterov must be off")
        if (self.training.optimizer == "adam" or self.reference.name == "ema") and (
                self.probes.enabled or self.diagnostics.enabled or self.mechanism.enabled):
            raise ValueError("Adam/EMA runs require legacy theorem/diagnostic/mechanism observers off")
        if self.reference.name == "correct" and (
                self.method.name != "dpco" or self.method.dpco_warmup_percent != 0):
            raise ValueError("Correct reference currently supports continuous dpco_r only, without method warmup")
        if not 0 <= self.method.flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        classes = DATASET_CLASSES
        if self.data.name not in classes:
            raise ValueError(f"Unsupported dataset: {self.data.name}")
        if self.data.num_classes != classes[self.data.name]:
            raise ValueError("num_classes must match the selected dataset")
        if self.data.name == "imagenet100" and not self.data.class_manifest:
            raise ValueError("imagenet100 requires a fixed class_manifest path")
        if not isinstance(self.data.is_human, bool) or not isinstance(self.data.download, bool):
            raise ValueError("is_human and download must be boolean")
        if (isinstance(self.data.noise_seed, bool) or not isinstance(self.data.noise_seed, int)
                or not 0 <= self.data.noise_seed < 2**32):
            raise ValueError("noise_seed must be an integer in [0, 2**32)")
        expected_noise = self.data.noise_percent
        if expected_noise is not None and (isinstance(expected_noise, bool)
                or not math.isfinite(expected_noise) or not 0 <= expected_noise <= 100):
            raise ValueError("noise_percent must be finite and in [0, 100]; it checks the actual rate")
        generated_noise = self.data.label_noise_percent
        if (isinstance(generated_noise, bool) or not math.isfinite(generated_noise)
                or not 0 <= generated_noise <= 100):
            raise ValueError("label_noise_percent must be finite and in [0, 100]")
        if self.data.name != "mnist" and generated_noise != 0:
            raise ValueError("label_noise_percent is a MNIST-only generator control")
        if self.data.name != "cnwl" and self.data.cnwl_noise_percent != 20:
            raise ValueError("cnwl_noise_percent requires dataset=cnwl")
        cifar_models = CIFAR_MODELS
        if self.data.noise_manifest is not None and (
                self.data.name not in ("imagenet100", "imagenet1k")
                or not isinstance(self.data.noise_manifest, str) or not self.data.noise_manifest.strip()):
            raise ValueError("noise_manifest requires an ImageNet100/1K artifact path")
        if self.data.name == "cnwl":
            selected = self.data.cnwl_noise_percent
            if isinstance(selected, bool) or selected not in (0, 5, 10, 15, 20, 30, 40, 50, 60, 80):
                raise ValueError("cnwl_noise_percent must select an official Red split")
            if self.data.noise_type != "red" or not self.data.is_human or self.data.noise_seed != 0:
                raise ValueError("CNWL uses fixed released Red web labels; noise_type=red, is_human=true, noise_seed=0")
            if self.data.mode != ("clean" if selected == 0 else "noisy"):
                raise ValueError("CNWL data.mode must match the selected official noise split")
            if self.data.image_size != 84 or self.model.name not in (*IMAGENET_MODELS, *cifar_models):
                raise ValueError("CNWL requires image_size=84 and a supported ResNet")
            if self.probes.enabled or self.diagnostics.enabled or self.mechanism.enabled:
                raise ValueError("CNWL requires legacy observers off; correct labels for web noise are unavailable")
        elif self.data.name == "cifar100n":
            if self.data.noise_type not in ("clean100", "noisy100"):
                raise ValueError("CIFAR-100N noise_type must be clean100 or noisy100")
            expected_mode = "clean" if self.data.noise_type == "clean100" else "noisy"
            if self.data.mode != expected_mode:
                raise ValueError("CIFAR-100N data.mode must match noise_type")
            if self.data.image_size != 32 or self.model.name not in (*cifar_models, *IMAGENET_MODELS):
                raise ValueError("CIFAR-100N requires image_size=32 and a supported ResNet model")
            if self.data.is_human and self.data.noise_seed != 0:
                raise ValueError("Human CIFAR-100N labels are fixed; noise_seed applies only to synthetic labels")
            if self.data.noise_type == "clean100" and expected_noise not in (None, 0):
                raise ValueError("clean100 has 0 percent noise")
            if self.probes.enabled or self.diagnostics.enabled or self.mechanism.enabled:
                raise ValueError("CIFAR-100N uses grouped evaluation; legacy observers must be off")
        elif self.data.name == "mnist":
            if self.data.noise_type != "symmetric" or self.data.is_human:
                raise ValueError("MNIST uses synthetic symmetric labels; set noise_type=symmetric and is_human=false")
            if self.data.mode != ("noisy" if generated_noise > 0 else "clean"):
                raise ValueError("MNIST data.mode must match label_noise_percent (0=clean)")
            if self.data.image_size != 28 or self.model.name not in MODELS:
                raise ValueError("MNIST requires image_size=28 and a supported classification model")
            if self.probes.enabled or self.diagnostics.enabled or self.mechanism.enabled:
                raise ValueError("MNIST uses grouped evaluation; legacy observers must be off")
        elif self.data.noise_manifest is not None:
            if self.data.noise_type != "clip_idn" or self.data.is_human or self.data.mode != "noisy":
                raise ValueError("ImageNet noise_manifest requires noise_type=clip_idn, is_human=false, mode=noisy")
            if self.data.noise_seed != 0 or self.data.download:
                raise ValueError("ImageNet noise is fixed by its artifact; set noise_seed in the preparation script, not training")
            if self.model.name not in IMAGENET_MODELS:
                raise ValueError("ImageNet CLIP noise requires an ImageNet ResNet backbone")
            if self.probes.enabled or self.diagnostics.enabled or self.mechanism.enabled:
                raise ValueError("CLIP-noisy ImageNet uses grouped evaluation; legacy observers must be off")
        elif (self.data.noise_type is not None or expected_noise is not None or not self.data.is_human
                or self.data.noise_seed != 0 or self.data.download or self.model.name in cifar_models):
            raise ValueError("CIFAR noise controls/models require dataset=cifar100n")
        if self.model.name not in MODELS:
            raise ValueError(f"Unsupported model: {self.model.name}")
        if self.model.name in MNIST_MODELS and self.data.name != "mnist":
            raise ValueError("mnist_cnn/mnist_mlp require dataset=mnist")
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
        if self.training.lr_schedule not in ("constant", "cosine", "inverse_sqrt"):
            raise ValueError("lr_schedule must be constant, cosine or inverse_sqrt")
        if (isinstance(self.training.min_learning_rate, bool)
                or not math.isfinite(self.training.min_learning_rate)
                or not 0 <= self.training.min_learning_rate <= self.training.learning_rate):
            raise ValueError("min_learning_rate must be finite and in [0, learning_rate]")
        if self.training.lr_schedule != "cosine" and self.training.min_learning_rate != 0:
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
        if self.probes.enabled and self.method.name != "dpco":
            raise ValueError("Common-reference theorem probes currently require full dpco")
        d = self.diagnostics
        self.mechanism.validate(self.training, self.method)
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
            if self.method.name not in ("sgd", "dpco", "dpco_policy"):
                raise ValueError("Real-update diagnostics cover SGD and full common-reference DPCO only")
            if self.training.momentum != 0 or self.training.l2_coefficient != 0:
                raise ValueError("Real-update diagnostics require momentum=0 and l2_coefficient=0")
        if not self.data.mode or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." for c in self.data.mode):
            raise ValueError("data.mode must be a safe name such as clean or mismatch10")

    def as_dict(self) -> dict:
        result = asdict(self)
        if self.data.noise_manifest is None:
            result["data"].pop("noise_manifest")
        if self.data.name != "cnwl":
            # Preserve existing dataset metadata and checkpoint resume identities.
            result["data"].pop("cnwl_noise_percent")
        return result


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
        mechanism=MechanismConfig(**raw.pop("mechanism", {})),
        execution=ExecutionConfig(**raw.pop("execution", {})),
        **raw,
    )
    config.validate()
    return config
