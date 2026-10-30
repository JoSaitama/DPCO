"""Single-process experiment loop with injected data, model, and probe adapters.

This is not an ImageNet reader or a TPU launcher. A training source must implement
__next__, snapshot and restore, including its shuffle/augmentation RNG state.
"""

from dataclasses import asdict, replace
import json
import math
import time

import jax
import numpy as np

from src.config import DataConfig, DiagnosticConfig, TrainingConfig, MechanismConfig
from src.methods.base import AlignmentMethod
from src.training.checkpoints import CheckpointStore
from src.training.randomness import training_key
from src.training.recording import RunRecorder, environment_metadata
from src.training.results import ResultsPaths, make_run_name
from src.training.schedules import EventSchedule, milestone_steps
from src.training.status import RunStatus


class ExperimentRunner:
    def __init__(self, config, trainer, source, evaluation, *, base_dir,
                 train_sample_count, metadata, probes=None, console=True,
                 run_name=None, resume=False, reserved=False, status=None, diagnostics=None, mechanism=None):
        config.validate()
        if resume and not config.checkpoints.enabled:
            raise ValueError("Resume requires checkpoints enabled")
        if train_sample_count <= 0:
            raise ValueError("train_sample_count must be positive")
        if jax.process_count() != 1:
            raise ValueError("ExperimentRunner currently supports one isolated JAX process only")
        if not all(callable(getattr(source, name, None)) for name in ("__next__", "snapshot", "restore")):
            raise TypeError("Training source must support exact snapshot/restore")
        if config.probes.enabled and probes is None:
            raise ValueError("Enabled probes require an explicit read-only probe adapter")
        if config.diagnostics.enabled and diagnostics is None:
            raise ValueError("Enabled real-update diagnostics require an explicit read-only adapter")
        if config.mechanism.enabled and mechanism is None:
            raise ValueError("Enabled mechanisms require an explicit read-only adapter")
        if resume and run_name is None:
            raise ValueError("Resume requires an explicit existing run name")
        if config.probes.enabled and not isinstance(trainer.method, AlignmentMethod):
            raise ValueError("Theorem probes require an alignment method")
        self.config, self.trainer, self.source = config, trainer, source
        if isinstance(trainer.method, AlignmentMethod):
            record = config.probes.enabled and (config.probes.theorem1 or config.probes.theorem3)
            self.trainer = trainer.with_method(replace(trainer.method, record_theory=record))
        self.evaluation, self.probes = evaluation, probes
        self.diagnostics = diagnostics if config.diagnostics.enabled else None
        self.mechanism = mechanism if config.mechanism.enabled else None
        schemas = None
        if self.mechanism is not None:
            from src.training.recording import SCHEMAS
            from src.probes.tpu_mechanisms import SCHEMAS as MECHANISM_SCHEMAS
            schemas = {**SCHEMAS, **MECHANISM_SCHEMAS}
        if config.reference.name == "ema":
            from src.training.recording import SCHEMAS, DPCO_SCHEMA
            schemas = {**(SCHEMAS if schemas is None else schemas), "dpco_steps": DPCO_SCHEMA}
        if callable(getattr(evaluation, "evaluate_groups", None)):
            from src.training.recording import SCHEMAS
            schemas = {**(SCHEMAS if schemas is None else schemas), "group_evaluation":
                ("step", "samples_seen", "group", "label_source", "sample_count", "ce", "top1", "top5")}
        self.sample_count = train_sample_count
        self.paths = ResultsPaths.resolve(config.results.root_dir, base_dir=base_dir).experiment(run_name or make_run_name(config))
        if reserved and not resume:
            self.paths.claim_reserved()
        else:
            self.paths.create(resume=resume)
        self.store = CheckpointStore(self.paths.runs)
        status = status or RunStatus(self.paths, config,
            mode=metadata.get("execution_mode", "sanity"), resume=resume)
        self.recorder = RunRecorder(self.paths, metadata={
            "run_name": self.paths.name, "config": config.as_dict(),
            "environment": environment_metadata(base_dir), "experiment": metadata,
            "devices": [{"id": device.id, "platform": device.platform,
                         "kind": device.device_kind} for device in jax.local_devices()],
            "process_index": jax.process_index(), "process_count": jax.process_count(),
            "train_sample_count": train_sample_count,
            "evaluation_mode": evaluation.mode,
            "evaluation_selects_best": getattr(evaluation, "select_best", True),
            "state_convention": "step counts completed updates; train metrics describe parameters before that update",
            "accuracy_unit": "fraction", "rng_convention": "seed/step/stream; source owns data RNG",
            "train_timestamp_convention": "host dispatch time; window throughput synchronizes device completion",
            **({"training_diagnostics": self.diagnostics.metadata()} if self.diagnostics is not None else {}),
            **({"tpu_mechanisms": self.mechanism.metadata()} if self.mechanism is not None else {}),
        }, console=console, resume=resume, status=status, schemas=schemas)
        if resume:
            stored_metadata = json.loads((self.paths.logs / "metadata.json").read_text())
            stored_config = stored_metadata["config"]
            # Older checkpoints predate optional update controls and used the off path.
            # Compare its implicit default without rewriting original metadata.
            for key in ("l2_coefficient", "momentum", "nesterov", "lr_schedule", "min_learning_rate",
                        "optimizer", "adam_beta1", "adam_beta2", "adam_epsilon"):
                stored_config["training"].setdefault(key, getattr(TrainingConfig(), key))
            # Legacy runs implicitly had diagnostics disabled. Do not enable an
            # observer during resume or reinterpret their existing CSV records.
            stored_config.setdefault("diagnostics", json.loads(json.dumps(asdict(DiagnosticConfig()))))
            stored_config.setdefault("mechanism", json.loads(json.dumps(asdict(MechanismConfig()))))
            stored_config["method"].setdefault("policy_mode", "continuous")
            stored_config["method"].setdefault("alternating_first", "sgd")
            stored_config["method"].setdefault("dpco_warmup_percent", 0.0)
            stored_config["method"].setdefault("negative_weight", -1.0)
            stored_config["reference"].setdefault("beta", 0.9)
            for key in ("noise_type", "is_human", "noise_seed", "noise_percent", "download", "label_noise_percent"):
                stored_config["data"].setdefault(key, getattr(DataConfig("imagenet100", 100), key))
            # Runs created before the opt-out switch always saved checkpoints.
            stored_config["checkpoints"].setdefault("enabled", True)
            if stored_config != json.loads(json.dumps(config.as_dict())):
                self.recorder.close()
                raise ValueError("Resume configuration differs from the original run")
            if self.mechanism is not None and stored_metadata.get("tpu_mechanisms", {}).get("target") != self.mechanism.metadata()["target"]:
                self.recorder.close()
                raise ValueError("Resume fixed mechanism target differs from the original run")
        self.resume = resume
        self.log_schedule = EventSchedule(config.logging.mode, config.logging.interval,
            total_steps=config.training.total_steps, train_sample_count=train_sample_count)
        self.eval_schedule = EventSchedule(config.evaluation.mode, config.evaluation.interval,
            total_steps=config.training.total_steps, train_sample_count=train_sample_count)

    def run(self, initial_state, *, resume_slot="latest"):
        with self.recorder.status.monitor():
            try:
                return self._run(initial_state, resume_slot=resume_slot)
            finally:
                # Initialization/resume validation can fail before the loop's try.
                if not self.recorder.log_file.closed:
                    self.recorder.close()

    def _run(self, initial_state, *, resume_slot):
        c, recorder = self.config, self.recorder
        state, samples_seen, elapsed_offset = initial_state, 0, 0.
        stateful_method = c.method.name == "dpco_policy" or c.method.dpco_warmup_percent > 0
        if stateful_method and state.method_state is None:
            state = state._replace(method_state=self.trainer.method.initial_state())
        historical = c.reference.name == "ema"
        if historical and state.method_state is None:
            state = state._replace(method_state=self.trainer.method.initial_state(state.params))
        cumulative = dict(cumulative_reference_gain=0., cumulative_eta=0., cumulative_eta_sq=0.)
        best, best_step, last_eval = None, None, None
        if self.resume:
            saved = self.store.load(resume_slot)
            if self.mechanism is not None:
                # Validate observer prefixes before rewinding CSVs or pruning an
                # abandoned future. A corrupt observer state must fail closed.
                self.mechanism.restore(saved.get("mechanism_state"), step=int(saved["train_state"].step))
            state, samples_seen = saved["train_state"], saved["samples_seen"]
            if stateful_method:
                from src.methods.dpco_policy import PolicyState
                if (not isinstance(state.method_state, PolicyState)
                        or int(state.method_state.updates) != int(state.step)):
                    raise ValueError("Policy checkpoint phase does not match completed updates")
            if state.optimizer_state is None and getattr(self.trainer.updater, "momentum", 0) != 0:
                raise ValueError("Momentum checkpoint is missing optimizer state")
            if historical:
                self.trainer.method.validate_state(state.method_state, state.params, state.step)
            validate_optimizer = getattr(self.trainer.updater, "validate_state", None)
            if callable(validate_optimizer):
                validate_optimizer(state.optimizer_state, state.params, state.step)
            self.source.restore(saved["source_state"])
            cumulative, best, best_step = saved["cumulative"], saved["best"], saved["best_step"]
            last_eval, elapsed_offset = saved["last_eval"], saved["elapsed_wall_s"]
            recorder.rewind(saved["csv_cursors"])
            self.store.discard_future(int(state.step))
        initializer = getattr(self.trainer.updater, "init", None)
        if state.optimizer_state is None and callable(initializer):
            # Establish a stable state tree before warmup and the initial save.
            state = state._replace(optimizer_state=initializer(state.params))
        if historical:
            self.trainer.method.validate_state(state.method_state, state.params, state.step)
        validator = getattr(self.trainer.updater, "validate_state", None)
        if callable(validator):
            validator(state.optimizer_state, state.params, state.step)
        start = time.perf_counter()
        elapsed = lambda: elapsed_offset + time.perf_counter() - start
        step = int(state.step)
        self.log_schedule.seek(step, samples_seen)
        self.eval_schedule.seek(step, samples_seen)
        nodes = milestone_steps(c.training.total_steps, c.probes.fractions)
        diagnostic_nodes = set(c.diagnostics.steps) if self.diagnostics is not None else set()
        milestones = {}
        for fraction, slot in ((0., "initial"), (.25, "p25"), (.5, "p50"), (.75, "p75"), (1., "final")):
            milestones.setdefault(math.ceil(fraction * c.training.total_steps), []).append(slot)

        def timing(event, duration, **kwargs):
            recorder.record("timing", step=int(state.step), event=event, duration_s=duration,
                            elapsed_wall_s=elapsed(), **kwargs)

        def memory_snapshot():
            for device in jax.local_devices():
                try:
                    stats = device.memory_stats()
                except (NotImplementedError, RuntimeError):
                    stats = None
                peak = None if stats is None else stats.get("peak_bytes_in_use")
                timing("memory_snapshot", 0., device=str(device), peak_device_memory_bytes=peak)

        def evaluate():
            nonlocal best, best_step, last_eval
            before = time.perf_counter()
            values = self.evaluation.evaluate(state)
            if not all(math.isfinite(float(v)) for v in values.values()):
                raise FloatingPointError("Non-finite validation metrics")
            last_eval = {"step": int(state.step), **values}
            improved = getattr(self.evaluation, "select_best", True) and (best is None or values["val_top1"] > best)
            if improved:
                best, best_step = values["val_top1"], int(state.step)
            group_rows = (self.evaluation.evaluate_groups(state)
                          if callable(getattr(self.evaluation, "evaluate_groups", None)) else ())
            for row in group_rows:
                metrics = (row["ce"], row["top1"], row["top5"])
                if row["sample_count"] > 0 and not all(v is not None and math.isfinite(float(v)) for v in metrics):
                    raise FloatingPointError("Non-finite fixed-group metrics")
            recorder.record("evaluation", **last_eval, samples_seen=samples_seen,
                epoch_progress=samples_seen / self.sample_count, elapsed_wall_s=elapsed(),
                eval_duration_s=time.perf_counter() - before)
            for row in group_rows:
                recorder.record("group_evaluation", step=int(state.step), samples_seen=samples_seen, **row)
                recorder.log("group_evaluation", step=int(state.step), **row)
            recorder.log("validation", **last_eval, elapsed_wall_s=elapsed())
            timing("evaluation", time.perf_counter() - before)
            return improved

        def probe():
            if c.probes.enabled and int(state.step) in nodes:
                before = time.perf_counter()
                rate = self.trainer.updater.learning_rate
                eta = float(rate(int(state.step)) if callable(rate) else rate)
                for stream, row in self.probes.run(state, step=int(state.step), learning_rate=eta,
                                                   config=c.probes, cumulative=cumulative):
                    recorder.record(stream, **row)
                timing("probe", time.perf_counter() - before)

        def save(slots):
            # Return before cursor flush, source snapshot and device serialization.
            # Disabling checkpoints must not change training or probe recording.
            if not c.checkpoints.enabled or not slots:
                return
            before = time.perf_counter()
            cursors = recorder.cursors()
            payload = {
                "train_state": state, "source_state": self.source.snapshot(),
                "rng_seed": c.seed, "samples_seen": samples_seen, "cumulative": cumulative,
                "best": best, "best_step": best_step, "last_eval": last_eval,
                "elapsed_wall_s": elapsed(), "csv_cursors": cursors,
            }
            if self.mechanism is not None:
                payload["mechanism_state"] = self.mechanism.snapshot()
            size = self.store.save(int(state.step), payload, slots)
            timing("checkpoint", time.perf_counter() - before, checkpoint_bytes=size)

        compiled_step = self.trainer.compile_step()
        window_start = time.perf_counter()
        window_steps = window_samples = 0
        try:
            if not self.resume and step != 0:
                raise ValueError("A new run must start at step zero; use explicit resume otherwise")
            recorder.log("resume" if self.resume else "start", step=step, run_name=self.paths.name)
            if not self.resume:
                improved = evaluate()
                probe()
                save(["initial", "latest"] + (["best"] if improved else []))
                memory_snapshot()
            # Compile with a peeked batch and discard the candidate update. Restore
            # data state so warmup consumes neither examples nor augmentation RNG.
            if step < c.training.total_steps:
                before = time.perf_counter()
                cursor = self.source.snapshot()
                batch = next(self.source)
                key = training_key(c.seed, state.step)
                executable = compiled_step.lower(state, batch, key).compile()
                candidate, _ = executable(state, batch, key)
                jax.block_until_ready(candidate)
                self.source.restore(cursor)
                timing("compile_warmup", time.perf_counter() - before)
            window_start = time.perf_counter()
            pending = []

            def drain():
                nonlocal window_steps, window_samples, window_start
                if not pending:
                    return
                duration = time.perf_counter() - window_start
                # One tree transfer per window; asynchronous compute is included.
                host = jax.device_get([entry[1] for entry in pending])
                duration = time.perf_counter() - window_start
                for (row_step, _, eta, seen, stamp), values in zip(pending, host):
                    values = {key: np.asarray(value).item() for key, value in values.items()}
                    record = dict(step=row_step, samples_seen=seen,
                        epoch_progress=seen / self.sample_count, learning_rate=eta,
                        valid_batch_size=values["valid_batch_size"],
                        train_ce=values["train_ce"], train_top1=values["train_top1"], elapsed_wall_s=stamp)
                    recorder.record("train_steps", **record)
                    policy_fields = {}
                    if stateful_method:
                        alpha = float(values["policy_alpha"])
                        policy_fields = dict(policy_mode="warmup" if c.method.dpco_warmup_percent else c.method.policy_mode, policy_alpha=alpha,
                            executed_method={0.: "sgd", .5: "half_dpco", 1.: "dpco"}[alpha])
                        recorder.record("policy_steps", step=row_step, reference=c.reference.name, **policy_fields)
                    if historical:
                        fields = {key: value for key, value in values.items() if key.startswith("dpco_")}
                        if not all(math.isfinite(float(value)) for value in fields.values()):
                            raise FloatingPointError(f"Non-finite DPCO statistics at step {row_step}")
                        recorder.record("dpco_steps", step=row_step,
                            valid_batch_size=values["valid_batch_size"], **fields)
                    if not values.get("state_finite", True):
                        raise FloatingPointError(f"Non-finite method/optimizer state at step {row_step}")
                    if not values["parameters_finite"]:
                        raise FloatingPointError(f"Non-finite model parameters at step {row_step}")
                    if not all(math.isfinite(float(values[key])) for key in ("train_ce", "train_top1")):
                        raise FloatingPointError(f"Non-finite training metrics at step {row_step}")
                    if "theory_delta" in values:
                        delta = values["theory_delta"]
                        if not math.isfinite(delta):
                            raise FloatingPointError("Non-finite reference gain")
                        cumulative["cumulative_reference_gain"] += eta * delta
                        if c.probes.theorem1:
                            recorder.record("theorem1", step=row_step, phase="train", probe_id="",
                                valid_batch_size=values["valid_batch_size"], error_count=values["theory_error_count"],
                                negative_count=values["theory_negative_count"], flip_count=values["theory_flip_count"],
                                reference_valid=values["theory_reference_valid"], delta=delta, eta_delta=eta * delta,
                                reference_projection_sgd=values["theory_reference_projection_sgd"],
                                reference_projection_dpco=values["theory_reference_projection_dpco"])
                    cumulative["cumulative_eta"] += eta
                    cumulative["cumulative_eta_sq"] += eta ** 2
                    if c.probes.enabled and c.probes.theorem3:
                        recorder.record("theorem3", step=row_step, phase="train", probe_id="",
                                        delta=values.get("theory_delta"), **cumulative)
                    if self.log_schedule.due(row_step, seen):
                        recorder.log("train", **record, **policy_fields)
                pending.clear()
                timing("train_window", duration, window_steps=window_steps, window_samples=window_samples,
                    steps_per_s=window_steps / duration, samples_per_s=window_samples / duration)
                window_steps = window_samples = 0

            while step < c.training.total_steps:
                before_samples = samples_seen
                diagnostic_due = self.diagnostics is not None and step in diagnostic_nodes
                mechanism_due = self.mechanism is not None and c.mechanism.due(step)
                batch = next(self.source)
                if diagnostic_due or mechanism_due:
                    # Retain only selected states. The NORMAL compiled step below
                    # remains identical, with no extra gradient outputs or traces.
                    before_state = state
                    context = (self.source.diagnostic_context() if callable(getattr(self.source, "diagnostic_context", None))
                               else {"batch_id": f"training_update_{step + 1}", "source_state": self.source.snapshot()})
                # Source-side batches are host data; count without synchronizing TPU metrics.
                count = int(np.count_nonzero(np.asarray(batch.valid_mask)))
                rate = self.trainer.updater.learning_rate
                eta = float(rate(step) if callable(rate) else rate)
                state, metrics = compiled_step(state, batch, training_key(c.seed, step))
                step += 1
                samples_seen += count
                window_steps += 1
                window_samples += count
                pending.append((step, metrics, eta, samples_seen, elapsed()))
                eval_due = self.eval_schedule.due(step, samples_seen)
                epoch_end = samples_seen // self.sample_count > before_samples // self.sample_count
                save_slots = list(milestones.get(step, [])) if c.checkpoints.enabled else []
                if c.checkpoints.enabled and (step % c.checkpoints.latest_every_steps == 0 or (c.checkpoints.at_epoch_end and epoch_end) or step == c.training.total_steps):
                    save_slots.append("latest")
                node_due = c.probes.enabled and step in nodes
                log_due = self.log_schedule.would_trigger(step, samples_seen)
                boundary = (len(pending) >= c.logging.timing_window_steps or step % c.logging.csv_flush_steps == 0
                            or log_due or eval_due or node_due or save_slots or diagnostic_due or mechanism_due or step == c.training.total_steps)
                if boundary:
                    drain()
                    jax.block_until_ready(state)
                    if diagnostic_due:
                        before_diagnostic = time.perf_counter()
                        rows = self.diagnostics.run(before_state, state, batch, learning_rate=eta,
                            rng_key=training_key(c.seed, before_state.step), context=context,
                            cumulative_eta_before=cumulative["cumulative_eta"] - eta)
                        for stream, row in rows:
                            recorder.record(stream, **row)
                        duration = time.perf_counter() - before_diagnostic
                        timing("diagnostic", duration)
                        recorder.log("diagnostic", step=int(before_state.step), update_step=step,
                                     rows=len(rows), duration_s=duration)
                        # Synchronizing/writing happens only at diagnostic nodes;
                        # release the frozen parameter tree immediately afterwards.
                        recorder.flush()
                        del rows
                    if mechanism_due:
                        rows = self.mechanism.run(before_state, state, batch, learning_rate=eta,
                            rng_key=training_key(c.seed, before_state.step), context=context,
                            cumulative_eta_before=cumulative["cumulative_eta"]-eta)
                        io_start = time.perf_counter()
                        for stream, row in rows:
                            recorder.record(stream, **row)
                        recorder.flush()
                        recorder.record("mechanism_timing", step=int(before_state.step), event="recording_io",
                            kernel="recorder", duration_s=time.perf_counter()-io_start)
                        del rows
                    if diagnostic_due or mechanism_due:
                        del before_state, context
                    if eval_due or step == c.training.total_steps:
                        if evaluate():
                            save_slots.append("best")
                    probe()
                    save(save_slots)
                    if step % c.logging.csv_flush_steps == 0:
                        recorder.flush()
                    window_start = time.perf_counter()
            summary = dict(status="completed", step=step, samples_seen=samples_seen, elapsed_wall_s=elapsed(),
                final_val_ce=last_eval["val_ce"], final_val_top1=last_eval["val_top1"],
                final_val_top5=last_eval["val_top5"], best_val_top1=best, best_step=best_step)
            memory_snapshot()
            recorder.record("summary", **summary)
            recorder.log("completed", **summary)
            return state
        except Exception as error:
            recorder.log("failed", step=int(state.step), error=str(error), elapsed_wall_s=elapsed())
            recorder.record("summary", status="failed", step=int(state.step), samples_seen=samples_seen,
                            elapsed_wall_s=elapsed(), error=str(error))
            raise
        finally:
            recorder.close()
