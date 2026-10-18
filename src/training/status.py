"""Append-only, cross-run status journal; no accelerator imports or polling."""

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
    "event", "status", "step", "total_steps", "pid", "hostname", "error",
)
EVENT_STATUS = {
    "launching": "launching", "started": "running", "start": "running",
    "resume": "running", "train": "running", "completed": "completed",
    "failed": "failed", "interrupted": "interrupted",
}


class RunStatus:
    """Each launch/resume has one attempt ID shared by parent and child.

    The last row for an attempt is its latest observed status, not a liveness
    guarantee. SIGKILL, host loss, and os._exit cannot publish a terminal row.
    """

    def __init__(self, paths, config, *, mode, resume=False, attempt_id=None):
        self.path = paths.parent.logs / "run_status.tsv"
        self.identity = dict(
            run_name=paths.name, attempt_id=attempt_id or uuid4().hex,
            dataset=config.data.name, data_mode=config.data.mode,
            model=config.model.name, method=config.method.name,
            reference=config.reference.name if config.method.name in
                ("sage", "sage_loo", "dtv", "dtv_loo") else "",
            seed=config.seed, initial_lr=config.training.learning_rate,
            mode=mode, resume=resume, total_steps=config.training.total_steps,
        )
        self.last_step = None
        self.terminal = None

    @property
    def attempt_id(self):
        return self.identity["attempt_id"]

    def record(self, event, *, step=None, error="", pid=None):
        if event not in EVENT_STATUS:
            return
        if step is not None:
            self.last_step = int(step)
        row = dict(self.identity, timestamp=datetime.now().astimezone().isoformat(),
            event=event, status=EVENT_STATUS[event], step=self.last_step,
            pid=os.getpid() if pid is None else pid, hostname=socket.gethostname(),
            error=str(error).replace("\t", " ").replace("\r", " ").replace("\n", " "))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lock the shared append stream so concurrent seed jobs cannot interleave
        # rows or race to write multiple headers. No read/rewrite of old rows.
        with self.path.open("a+", newline="", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0, os.SEEK_END)
                writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS,
                                        delimiter="\t", lineterminator="\n")
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        self.terminal = event if event in ("completed", "failed", "interrupted") else None

    @contextmanager
    def monitor(self):
        """Include bootstrap failures and avoid duplicate nested failure rows."""
        try:
            yield self
        except BaseException as error:
            if self.terminal not in ("failed", "interrupted"):
                event = "failed" if isinstance(error, Exception) else "interrupted"
                self.record(event, error=f"{type(error).__name__}: {error}")
            raise
