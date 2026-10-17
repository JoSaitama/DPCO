"""Seven logical checkpoint slots sharing atomic, complete state snapshots."""

import json
import os
import pickle

from src.training.recording import atomic_json


class CheckpointStore:
    """Use only trusted local snapshots: pickle is not an exchange format.

    A snapshot contains caller-supplied train/data/RNG/cumulative state. The index
    is published only after its snapshot is durable; old unreferenced snapshots
    are then removed. Transient space for the new snapshot is required.
    """
    SLOTS = {"initial", "p25", "p50", "p75", "final", "best", "latest"}

    def __init__(self, directory):
        self.directory = directory
        self.index_path = directory / "index.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def save(self, step, payload, slots):
        if not set(slots) <= self.SLOTS or not slots:
            raise ValueError("Unknown or empty checkpoint slots")
        import jax
        path = self.directory / f"step_{step:09d}.pkl"
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(jax.device_get(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        new_index = {**self.index, **{slot: path.name for slot in slots}}
        atomic_json(self.index_path, new_index)
        self.index = new_index
        retained = set(self.index.values())
        for old in self.directory.glob("step_*.pkl"):
            if old.name not in retained:
                old.unlink()
        return path.stat().st_size

    def load(self, slot="latest"):
        with (self.directory / self.index[slot]).open("rb") as handle:
            return pickle.load(handle)

    def discard_future(self, step):
        """Drop slots from an abandoned future before resuming an older snapshot."""
        self.index = {key: value for key, value in self.index.items()
                      if int(value.removeprefix("step_").removesuffix(".pkl")) <= step}
        atomic_json(self.index_path, self.index)
