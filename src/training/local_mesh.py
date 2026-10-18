"""One-process, host-local data parallelism with global logical-batch semantics."""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.data.batch import ClassificationBatch
from src.training.trainer import Trainer


class LocalMeshTrainer(Trainer):
    def __init__(self, task, method, updater, mesh):
        super().__init__(task, method, updater)
        self.mesh = mesh
        self.replicated = NamedSharding(mesh, P())
        self.batch_sharding = ClassificationBatch(*(NamedSharding(mesh, P("data")) for _ in range(3)))

    def with_method(self, method):
        return LocalMeshTrainer(self.task, method, self.updater, self.mesh)

    def compile_step(self):
        # GSPMD differentiates/reduces the whole global batch, including BN and
        # error-reference counts; no device-local reference is substituted.
        return jax.jit(self.step,
            in_shardings=(self.replicated, self.batch_sharding, self.replicated),
            out_shardings=(self.replicated, self.replicated))


def local_mesh(devices):
    if not devices or any(device.process_index != jax.process_index() for device in devices):
        raise ValueError("Mesh must contain local devices only")
    return Mesh(np.asarray(devices), ("data",))
