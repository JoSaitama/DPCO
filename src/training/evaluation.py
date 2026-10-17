"""Full validation with sample-weighted totals and no model-state commits."""

from dataclasses import dataclass

import jax
import numpy as np

from src.training.metrics import task_metrics


@dataclass(frozen=True)
class EvaluationSpec:
    task: object
    batches: object
    expected_sample_count: int
    mode: str = "eval"

    def evaluate(self, state):
        """The caller supplies a deterministic evaluation-mode model adapter."""
        if self.expected_sample_count <= 0:
            raise ValueError("Validation sample count must be positive")
        totals = None
        for batch in self.batches():
            metrics = task_metrics(self.task.evaluate(state.params, batch, state.model_state), include_top5=True)
            values = {key: metrics[key] for key in ("valid_batch_size", "loss_sum", "correct_count", "top5_count")}
            totals = values if totals is None else {key: totals[key] + values[key] for key in values}
        if totals is None:
            raise ValueError("Validation iterator is empty")
        totals = {key: np.asarray(value).item() for key, value in jax.device_get(totals).items()}
        count = totals["valid_batch_size"]
        if count != self.expected_sample_count:
            raise ValueError(f"Expected {self.expected_sample_count} validation examples, received {count}")
        return {"eval_sample_count": count, "val_ce": totals["loss_sum"] / count,
                "val_top1": totals["correct_count"] / count,
                "val_top5": totals["top5_count"] / count}
