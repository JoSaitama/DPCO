"""Child entry point; launch only through scripts/launch_tpu.py."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.launch import absolute_path, add_overrides, diagnostic_overrides, dpco_overrides, dpco_cli_args, cifar_overrides, cifar_cli_args, resolve_config
from src.training.runtime import configure_worker0, require_local_tpu
from src.training.results import ResultsPaths
from src.training.status import RunStatus
from src.probes.mechanism_config import overrides as mechanism_overrides


def run(args, config, devices, *, status):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from src.data.images import ImageBatchSource, evaluation_batches
    from src.data.classification import open_training_images
    from src.methods import create_method
    from src.models.resnet import create_resnet, initialize, model_adapter
    from src.probes.classification import ClassificationProbes, ProbeBatch
    from src.references import create_reference
    from src.tasks.classification import ClassificationTask
    from src.training import TrainState
    from src.training.evaluation import EvaluationSpec
    from src.training.experiment import ExperimentRunner
    from src.training.local_mesh import LocalMeshTrainer, local_mesh
    from src.training.metrics import task_metrics
    from src.training.updates import make_updater
    from src.training.schedules import dpco_warmup_steps

    batch_size, image_size = config.global_batch_size, config.data.image_size
    if batch_size % len(devices):
        raise ValueError(f"Global batch size {batch_size} must be divisible by {len(devices)} local TPU devices")
    if config.execution.gradient_backend != "jvp":
        raise ValueError("TPU entry requires execution.gradient_backend='jvp' to avoid full training Jacobians")
    if config.data.mode != "clean" and config.data.name != "cifar100n":
        raise ValueError("Data mismatch modes are not implemented; a label must not silently leave data clean")
    limit = max(32, 2 * batch_size) if args.mode == "smoke" else None
    images, data_identity = open_training_images(config.data, ROOT, limit=limit,
        progress=lambda **fields: print(json.dumps(fields), flush=True))
    try:
        mesh = local_mesh(devices)
        replicated = NamedSharding(mesh, P())
        if config.data.name == "cifar100n":
            from src.models.cifar_resnet import create_cifar_resnet
            model = create_cifar_resnet(config.model.name, config.data.num_classes, config.model.precision)
        else:
            model = create_resnet(config.model.name, config.data.num_classes, config.model.precision)
        print(json.dumps({"event": "model_initialize", "model": config.model.name,
                          "initialization": "random", "precision": config.model.precision}), flush=True)
        params, model_state = initialize(model, config.seed, image_size)
        updater = make_updater(config.training)
        state = jax.device_put(TrainState(params, model_state,
            optimizer_state=updater.init(params)), replicated)
        train_task = ClassificationTask(model_adapter(model, train=True), config.data.num_classes)
        eval_task = ClassificationTask(model_adapter(model, train=False), config.data.num_classes)
        method = create_method(config.method.name, reference_name=config.reference.name,
            flip_probability=config.method.flip_probability, gradient_backend="jvp",
            policy_mode=config.method.policy_mode, alternating_first=config.method.alternating_first,
            warmup_steps=dpco_warmup_steps(config), reference_beta=config.reference.beta,
            negative_weight=config.method.negative_weight)
        if config.reference.name == "ema":
            state = state._replace(method_state=jax.device_put(method.initial_state(params), replicated))
        elif config.method.name == "dpco_policy" or dpco_warmup_steps(config):
            state = state._replace(method_state=jax.device_put(method.initial_state(), replicated))
        trainer = LocalMeshTrainer(train_task, method, updater, mesh)
        source = ImageBatchSource(images, batch_size, image_size, config.seed)
        diagnostics = None
        mechanism = None
        if config.diagnostics.enabled:
            from dataclasses import replace
            from src.data.diagnostic_batches import training_audit_batches
            from src.probes.training_diagnostics import TrainingDiagnostics

            audits = ()
            if config.diagnostics.transfer:
                audits = tuple(replace(item, batch=jax.device_put(item.batch, trainer.batch_sharding))
                    for item in training_audit_batches(images, batch_size, image_size,
                        config.diagnostics.audit_batches, config.diagnostics.audit_seed))
            diagnostics = TrainingDiagnostics(train_task, method, config.diagnostics,
                reference_name=config.reference.name, trajectory_method=config.method.name, audit_batches=audits)
        if config.mechanism.enabled:
            import hashlib
            import importlib.metadata
            import time
            from dataclasses import replace
            from src.data.diagnostic_batches import training_audit_batches
            from src.probes.tpu_mechanisms import TPUMechanisms
            mechanism_setup = time.perf_counter()
            bank = ()
            if config.mechanism.needs_bank:
                bank = tuple(replace(item, batch=jax.device_put(item.batch, trainer.batch_sharding))
                    for item in training_audit_batches(images, batch_size, image_size,
                        config.mechanism.bank_batches, config.mechanism.bank_seed))
            signature = hashlib.sha256((ROOT / "src/models/resnet.py").read_bytes()
                + (ROOT / "src/tasks/classification.py").read_bytes()).hexdigest()
            signature += f":{config.model.name}:{config.model.precision}:{config.data.num_classes}"
            signature += f":jax={jax.__version__}:flax={importlib.metadata.version('flax')}:matmul={jax.config.jax_default_matmul_precision}"
            mechanism = TPUMechanisms(train_task, method, config.mechanism,
                reference_name=config.reference.name, trajectory_method=config.method.name,
                bank_batches=bank, fixed_buffers=state.model_state, signature=signature,
                compute_dtype=config.model.precision)
            mechanism.setup_duration_s = time.perf_counter()-mechanism_setup
        batch_metrics = jax.jit(lambda p, b, s: task_metrics(eval_task.evaluate(p, b, s), include_top5=True),
            in_shardings=(replicated, trainer.batch_sharding, replicated), out_shardings=replicated)
        evaluation = EvaluationSpec(eval_task,
            lambda: evaluation_batches(images, batch_size, image_size),
            images.counts["validation"], "running_batchnorm_eval", batch_metrics)
        if config.data.name == "cifar100n":
            from src.training.evaluation import cifar100n_evaluation
            evaluation = cifar100n_evaluation(images, eval_task, batch_size, batch_metrics)
        probes = None
        if config.probes.enabled:
            batches = []
            for index in range(config.probes.batches_per_node):
                ids = [(index * batch_size + offset) % images.counts["train"] for offset in range(batch_size)]
                batch = next(evaluation_batches(images, batch_size, image_size, split="train", indices=ids))
                batches.append(ProbeBatch(f"fixed_train_batch_{index}",
                                          jax.device_put(batch, trainer.batch_sharding)))
            # Freeze the initial BN buffers as well as inputs for a genuine fixed
            # differentiable audit objective. This is not the changing train loss.
            fixed_state = state.model_state
            audit_batch = batches[0].batch
            def audit_objective(p):
                return jnp.mean(eval_task.evaluate(p, audit_batch, fixed_state).losses)
            probes = ClassificationProbes(train_task, create_reference(config.reference.name), batches,
                reference_name=config.reference.name,
                objective_mode="train_batch_bn; audit=frozen_initial_bn_eval",
                audit_objective=audit_objective, audit_name="fixed_training_audit")
        metadata = {"execution_mode": args.mode, "backend_requirement": "tpu_only",
            "wall_time_scope": "experiment_loop_only; data preparation and random initialization precede the loop clock",
            "worker_scope": "worker0_host_only", "data_parallel_devices": len(devices),
            "data_plan": data_identity, "data_fingerprint": images.fingerprint,
            "prepared_database": str(images.path), "split_counts": images.counts,
            "smoke_is_not_full_dataset_validation": args.mode == "smoke",
            "normalization": "CIFAR100 mean/std; train zero-pad4/random-crop32/flip; eval native32" if config.data.name == "cifar100n" else "ImageNet mean/std; train random resized crop and horizontal flip; eval resize/center crop",
            "evaluation_split": "official_test_monitor_only_no_best_selection" if config.data.name == "cifar100n" else "configured_validation",
            "group_evaluation": "fixed reference-label clean/corrupted groups; truth metadata excluded from training batch" if config.data.name == "cifar100n" else None,
            "train_tail": "drop_last_to_exclude_padding_from_training_batchnorm",
            "evaluation_tail": "padded_with_valid_count_weighting",
            "optimizer": config.training.optimizer,
            "update_semantics": "Adam uses edited direction and its elementwise square; optional coupled kernel L2; no AdamW, clipping or norm matching" if config.training.optimizer == "adam" else "SGD with optional momentum/Nesterov and coupled kernel L2",
            "reference_semantics": "bias-corrected EMA of raw global batch gradients, current batch included; independent of Adam moments" if config.reference.name == "ema" else config.reference.name,
            "sample_weight_semantics": "negative alignment gets method.negative_weight; zero/positive gets 1; original valid B denominator; stop-gradient weights" if config.reference.name == "ema" else "legacy method",
            "model": config.model.name, "model_initialization": "random",
            "model_source_revision": "dpco_cifar_basicblock_v1" if config.data.name == "cifar100n" else "81b168d62f151feca9e8e7abc591df80168395b4",
            "parameter_count": sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
            "gradient_backend": "exact_common_reference_vjp_jvp",
            "data_rng": "epoch shuffle and per-example stateless crop RNG; no prefetch",
            "theorem_scope": "reference identities and fixed-target diagnostics, not an automatic convergence-rate proof"}
        runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
            train_sample_count=images.counts["train"], metadata=metadata, probes=probes,
            run_name=args.run_name, resume=args.resume, reserved=args.reserved, status=status,
            diagnostics=diagnostics, mechanism=mechanism)
        runner.run(state, resume_slot=args.resume_slot)
    finally:
        images.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "validate"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--reserved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-slot", default="latest")
    parser.add_argument("--attempt-id")
    add_overrides(parser)
    args = parser.parse_args()
    config = resolve_config(absolute_path(args.config, ROOT), mode=args.mode,
                            steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                            method=args.method, dataset=args.dataset, model=args.model,
                            alternating_first=args.alternating_first,
                            dpco_warmup_percent=args.dpco_warmup_percent,
                            data_mode=args.data_mode, data_dir=args.data_dir,
                            l2_coefficient=args.l2_coefficient,
                            momentum=args.momentum, nesterov=args.nesterov,
                            learning_rate=args.learning_rate, lr_schedule=args.lr_schedule,
                            min_learning_rate=args.min_learning_rate,
                            checkpoint_every_steps=args.checkpoint_every_steps,
                            checkpoints=args.checkpoints,
                            eval_every_steps=args.eval_every_steps, eval_every_epochs=args.eval_every_epochs,
                            **diagnostic_overrides(args), **dpco_overrides(args), **cifar_overrides(args), **mechanism_overrides(args))
    if args.resume and not config.checkpoints.enabled:
        parser.error("Resume requires checkpoints enabled; --no-checkpoints runs cannot be resumed")
    paths = ResultsPaths.resolve(config.results.root_dir, base_dir=ROOT).experiment(args.run_name)
    status = RunStatus(paths, config, mode=args.mode, resume=args.resume, attempt_id=args.attempt_id)
    with status.monitor():
        status.record("started")
        settings = configure_worker0(config.execution)
        # This host lock prevents concurrent SGD/DPCO jobs from claiming the same
        # chips. It does not start, contact, or place work on worker1.
        lock_path = Path(tempfile.gettempdir()) / f"dpco-tpu-worker0-{os.getuid()}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another DPCO TPU job is running on this host; run SGD and DPCO sequentially") from None
            devices = require_local_tpu(config.execution)
            print(json.dumps({"event": "tpu_ready", "devices": [str(d) for d in devices],
                              "local_process_only": True, "runtime_settings": settings}), flush=True)
            run(args, config, devices, status=status)


if __name__ == "__main__":
    main()
