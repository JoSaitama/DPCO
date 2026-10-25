"""Two lifecycle rows per attempt; no training metrics or accelerator imports."""

from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import os
import socket
from uuid import uuid4


STATUS_FIELDS = (
    "timestamp", "run_name", "attempt_id", "dataset", "data_mode", "model",
    "method", "reference", "seed", "initial_lr", "mode", "resume",
    "event", "status", "total_steps", "pid", "hostname",
)
START_EVENTS = ("started", "start", "resume")
END_EVENTS = ("completed", "failed", "interrupted")


class RunStatus:
    """Write start/running and end/completed-or-failed once per attempt.

    A stable sidecar lock covers schema checks and cross-process deduplication. Child
    cleanup remains inside monitor(), so success is published only after cleanup.
    A waiting launcher can finalize a killed child; host loss cannot guarantee an end.
    """

    def __init__(self, paths, config, *, mode, resume=False, attempt_id=None):
        self.path = paths.parent.logs / "run_status.tsv"
        self.identity = dict(
            run_name=paths.name, attempt_id=attempt_id or uuid4().hex,
            dataset=config.data.name, data_mode=config.data.mode,
            model=config.model.name, method=(f"sage_policy_{config.method.policy_mode}"
                if config.method.name == "sage_policy" else config.method.name),
            reference=config.reference.name if config.method.name in
                ("sage", "sage_loo", "dtv", "dtv_loo", "sage_policy") else "",
            seed=config.seed, initial_lr=config.training.learning_rate,
            mode=mode, resume=resume, total_steps=config.training.total_steps,
        )
        if (config.method.name == "sage_policy" and config.method.policy_mode == "alternating"
                and config.method.alternating_first == "sage"):
            self.identity["method"] += "_first_sage"
        if config.method.sage_warmup_percent:
            self.identity["method"] += "_warmup" + format(config.method.sage_warmup_percent, ".12g") + "pct"
        self.terminal = None
        self._depth = 0
        self._pending_success = False

    @property
    def attempt_id(self):
        return self.identity["attempt_id"]

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._check_schema()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with self.path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _check_schema(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != STATUS_FIELDS:
                raise ValueError(f"Incompatible status TSV schema: {self.path}. "
                                 "Remove the old smoke-test TSV before relaunching; no backup or migration is performed.")

    def _append(self, event, status, *, pid=None):
        with self._locked():
            history = [row for row in self._read() if row["attempt_id"] == self.attempt_id]
            ended = next((row for row in history if row["event"] == "end"), None)
            if ended is not None:
                self.terminal = ended["status"]
                return
            if event == "start" and history:
                return
            base = dict(self.identity, timestamp=datetime.now().astimezone().isoformat(),
                        pid=os.getpid() if pid is None else pid, hostname=socket.gethostname())
            rows = []
            if not history:
                rows.append(dict(base, event="start", status="running"))
            if event == "end":
                rows.append(dict(base, event="end", status=status))
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS,
                                        delimiter="\t", lineterminator="\n")
                if new:
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
            if event == "end":
                self.terminal = status

    def record(self, event, *, pid=None):
        if event in START_EVENTS:
            self._append("start", "running", pid=pid)
        elif event in END_EVENTS:
            if event == "completed" and self._depth:
                self._pending_success = True
            else:
                self._append("end", "completed" if event == "completed" else "failed", pid=pid)
        # Train/validation events deliberately have no TSV path or file access.

    def outcome(self):
        with self._locked():
            ended = [row for row in self._read()
                     if row["attempt_id"] == self.attempt_id and row["event"] == "end"]
        return ended[-1]["status"] if ended else None

    @contextmanager
    def monitor(self):
        self._depth += 1
        try:
            yield self
        except BaseException:
            self.record("failed")
            raise
        else:
            if self._depth == 1 and self._pending_success:
                self._append("end", "completed")
        finally:
            self._depth -= 1
