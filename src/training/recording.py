"""One host-side outlet for buffered CSV, persistent logs, and run metadata."""

import csv
from datetime import datetime
import importlib.metadata
import hashlib
import json
import math
import os
import platform
import subprocess
import sys


SCHEMAS = {
    "train_steps": ("step", "samples_seen", "epoch_progress", "learning_rate", "valid_batch_size", "train_ce", "train_top1", "elapsed_wall_s"),
    "evaluation": ("step", "samples_seen", "epoch_progress", "eval_sample_count", "val_ce", "val_top1", "val_top5", "elapsed_wall_s", "eval_duration_s"),
    "theorem1": ("step", "phase", "probe_id", "valid_batch_size", "error_count", "negative_count", "flip_count", "reference_valid", "delta", "eta_delta", "reference_projection_sgd", "reference_projection_sage", "delta_formula", "identity_residual", "norm_sgd_sq", "norm_sage_sq"),
    "theorem2": ("step", "phase", "probe_id", "objective", "objective_mode", "batch_id", "reference_name", "selected_count", "learning_rate", "lr_scale", "probe_lr", "j_initial", "j_sgd", "j_sage", "difference", "difference_over_lr", "delta", "first_order_residual", "norm_sgd_sq", "norm_sage_sq", "finite", "strict_gain_eligible"),
    "theorem3": ("step", "phase", "probe_id", "objective", "objective_mode", "audit_loss", "audit_gradient_norm_sq", "delta", "target_gain", "transfer_difference", "sample_energy", "norm_sgd_sq", "norm_sage_sq", "cumulative_reference_gain", "cumulative_eta", "cumulative_eta_sq"),
    "timing": ("step", "event", "duration_s", "window_steps", "window_samples", "steps_per_s", "samples_per_s", "elapsed_wall_s", "checkpoint_bytes", "device", "peak_device_memory_bytes"),
    "summary": ("status", "step", "samples_seen", "elapsed_wall_s", "final_val_ce", "final_val_top1", "final_val_top5", "best_val_top1", "best_step", "error"),
}

# Real-update diagnostics deliberately have their own streams. Legacy theorem
# CSV meanings and headers stay unchanged, including their historical step axis.
DIAGNOSTIC_BASE = ("step", "update_step", "source_batch_id", "trajectory_method", "reference")
DIAGNOSTIC_PAIR = (*DIAGNOSTIC_BASE, "objective_mode", "learning_rate", "reference_delta",
    "norm_sgd_sq", "norm_sage_sq", "selected_count", "lr_scale", "probe_lr", "j_initial",
    "j_sgd", "j_sage", "difference", "difference_over_lr", "objective_gain",
    "first_order_residual", "actual_update_in_pair", "finite")
SCHEMAS.update({
    "diagnostic_context": (*DIAGNOSTIC_BASE, "learning_rate", "cumulative_eta_before",
        "objective_mode", "source_context_json"),
    "diagnostic_update": (*DIAGNOSTIC_BASE, "learning_rate", "valid_batch_size", "error_count",
        "flip_count", "reference_valid", "reference_delta", "identity_residual", "score_min_abs",
        "norm_sgd_sq", "norm_sage_sq", "direction_l2_error", "direction_relative_error",
        "parameter_l2_error", "parameter_relative_error", "replayed_parameter_l2_error",
        "model_state_max_abs_error", "replayed_model_state_max_abs_error", "finite"),
    "diagnostic_paired": (*DIAGNOSTIC_PAIR, "objective"),
    "diagnostic_transfer": (*DIAGNOSTIC_PAIR, "target_batch_id", "overlap_sample_count", "transfer_difference"),
})


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def environment_metadata(base_dir):
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=base_dir, text=True,
                                    capture_output=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    dependencies = {}
    for name in ("jax", "jaxlib", "flax", "optax", "tensorflow", "tensorflow-datasets", "numpy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    from pathlib import Path
    digest = hashlib.sha256()
    files = git("ls-files", "--cached", "--others", "--exclude-standard")
    if files:
        for name in sorted(set(files.splitlines())):
            path = Path(base_dir) / name
            if path.is_file():
                digest.update(name.encode("utf-8") + b"\0" + path.read_bytes())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": dependencies, "git_commit": git("rev-parse", "HEAD"),
            "git_worktree_status": git("status", "--porcelain"),
            "source_sha256": digest.hexdigest() if files else None,
            "created_at": datetime.now().astimezone().isoformat()}


def finite_or_none(value):
    return value if value is None or not isinstance(value, float) or math.isfinite(value) else None


def log_payload(event, fields):
    """Keep readable field order; CSV/config retain the learning_rate name."""
    payload = {"event": event, **{k: finite_or_none(v) for k, v in fields.items()}}
    if "learning_rate" in payload:
        payload["lr"] = payload.pop("learning_rate")
    order = ("step", "event", "train_ce", "lr", "train_top1", "valid_batch_size",
             "samples_seen", "epoch_progress", "elapsed_wall_s")
    return {**{key: payload[key] for key in order if key in payload},
            **{key: value for key, value in payload.items() if key not in order}}


class RunRecorder:
    """Numeric CSV rows are buffered; logs are independently flushed immediately.

    This object accepts host scalars only. Device transfer belongs to the runner,
    so recording never triggers hidden JAX synchronization per field.
    """

    def __init__(self, paths, *, metadata=None, console=True, resume=False, status=None):
        self.paths, self.console, self.status = paths, console, status
        self.buffers = {name: [] for name in SCHEMAS}
        self.row_counts = {name: 0 for name in SCHEMAS}
        if not resume:
            atomic_json(paths.logs / "metadata.json", metadata or {})
        self.log_file = (paths.logs / "train.log").open("a", encoding="utf-8")

    def record(self, stream, **row):
        fields = SCHEMAS[stream]
        extra = set(row) - set(fields)
        if extra:
            raise ValueError(f"Unknown fields in {stream}: {sorted(extra)}")
        self.buffers[stream].append(row)

    def log(self, event, **fields):
        payload = log_payload(event, fields)
        line = json.dumps(payload, allow_nan=False)
        self.log_file.write(line + "\n")
        self.log_file.flush()
        if self.console:
            print(line, file=sys.stdout, flush=True)
        if self.status is not None:
            self.status.record(event)

    def flush(self):
        for stream, rows in self.buffers.items():
            if not rows:
                continue
            path = self.paths.files / (stream + ".csv")
            new = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCHEMAS[stream])
                if new:
                    writer.writeheader()
                writer.writerows(rows)
            self.row_counts[stream] += len(rows)
            rows.clear()

    def rewind(self, cursors):
        """Keep exactly the flushed rows present when a checkpoint was written."""
        for rows in self.buffers.values():
            rows.clear()
        for stream in SCHEMAS:
            path = self.paths.files / (stream + ".csv")
            if not path.exists():
                if cursors.get(stream, 0):
                    raise ValueError(f"Missing recorded stream: {stream}")
                continue
            count = cursors.get(stream, 0)
            temporary = path.with_suffix(".csv.tmp")
            with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as destination:
                reader = csv.DictReader(source)
                writer = csv.DictWriter(destination, fieldnames=SCHEMAS[stream])
                writer.writeheader()
                for _ in range(count):
                    row = next(reader, None)
                    if row is None:
                        raise ValueError(f"Recorded stream is shorter than checkpoint: {stream}")
                    writer.writerow(row)
            os.replace(temporary, path)
            self.row_counts[stream] = count

    def cursors(self):
        self.flush()
        return dict(self.row_counts)

    def close(self):
        try:
            self.flush()
        finally:
            self.log_file.close()
