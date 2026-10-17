"""Pure metric reductions; these functions never synchronize or write files."""

import jax.numpy as jnp


def task_metrics(result, *, include_top5=False):
    valid = result.valid_mask
    count = jnp.sum(valid)
    loss_sum = jnp.sum(jnp.where(valid, result.losses, 0.))
    metrics = {"valid_batch_size": count, "loss_sum": loss_sum}
    if "top5_correct_mask" in result.feedback:
        correct = valid & ~result.feedback["error_mask"]
        metrics.update(train_ce=loss_sum / count,
                       train_top1=jnp.sum(correct) / count,
                       correct_count=jnp.sum(correct))
        if include_top5:
            metrics["top5_count"] = jnp.sum(valid & result.feedback["top5_correct_mask"])
    return metrics


def tree_dot(left, right):
    import jax
    return sum(jnp.vdot(a, b).real for a, b in
               zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))
