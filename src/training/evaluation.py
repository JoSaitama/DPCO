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
    batch_metrics: object = None

    def evaluate(self, state):
        """The caller supplies a deterministic evaluation-mode model adapter."""
        if self.expected_sample_count <= 0:
            raise ValueError("Validation sample count must be positive")
        totals = None
        for batch in self.batches():
            metrics = (self.batch_metrics(state.params, batch, state.model_state)
                       if self.batch_metrics is not None else
                       task_metrics(self.task.evaluate(state.params, batch, state.model_state), include_top5=True))
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


@dataclass(frozen=True)
class GroupedEvaluation:
    """Ordinary primary evaluation plus fixed, read-only label audit groups."""
    primary: EvaluationSpec
    groups: tuple
    select_best: bool = False

    @property
    def mode(self):
        return self.primary.mode + "; fixed_reference_label_groups"

    def evaluate(self, state):
        return self.primary.evaluate(state)

    def evaluate_groups(self, state):
        rows = []
        for name, label_source, spec in self.groups:
            if spec is None:
                rows.append(dict(group=name, label_source=label_source, sample_count=0,
                                 ce=None, top1=None, top5=None))
                continue
            values = spec.evaluate(state)
            rows.append(dict(group=name, label_source=label_source,
                sample_count=values["eval_sample_count"], ce=values["val_ce"],
                top1=values["val_top1"], top5=values["val_top5"]))
        return rows


def cifar100n_evaluation(images, task, batch_size, batch_metrics=None):
    return noisy_classification_evaluation(images, task, batch_size, 32,
        "official_cifar100_test_eval_no_model_selection", batch_metrics)


def noisy_classification_evaluation(images, task, batch_size, image_size, mode, batch_metrics=None):
    from src.data.images import evaluation_batches
    primary = EvaluationSpec(task, lambda: evaluation_batches(images, batch_size, image_size),
        images.counts["validation"], mode, batch_metrics)
    groups = []
    for name, partition, labels in (("clean", "clean", "original"),
            ("corrupted_noisy", "corrupted", "reference"),
            ("corrupted_original", "corrupted", "original")):
        ids = tuple(int(i) for i in images.group_indices[partition])
        view = images.label_view(labels)
        spec = EvaluationSpec(task,
            lambda ids=ids, view=view: evaluation_batches(view, batch_size, image_size, split="train", indices=ids),
            len(ids), "fixed_train_eval_no_augmentation", batch_metrics) if ids else None
        groups.append((name, labels, spec))
    return GroupedEvaluation(primary, tuple(groups))
