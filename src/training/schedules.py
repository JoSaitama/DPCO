"""Host-side event schedules based on threshold crossing, never float equality."""

import math


def validate_interval(mode, interval, choices):
    if mode not in choices or not math.isfinite(interval) or interval <= 0:
        raise ValueError(f"Expected a positive interval and mode in {choices}")
    if mode == "steps" and (isinstance(interval, bool) or int(interval) != interval):
        raise ValueError("Step intervals must be positive integers")
    if mode == "percent" and interval > 100:
        raise ValueError("Percentage intervals must not exceed 100")


def milestone_steps(total_steps, fractions=(0., .25, .5, .75, 1.)):
    return {max(0, min(total_steps, math.ceil(total_steps * p))) for p in fractions}


def sage_warmup_steps(config):
    """Number of real SGD-prefix updates; unrelated to learning-rate warmup."""
    return math.ceil(config.training.total_steps * config.method.sage_warmup_percent / 100)


class EventSchedule:
    def __init__(self, mode, interval, *, total_steps, train_sample_count):
        validate_interval(mode, interval, ("steps", "percent", "epochs"))
        if total_steps <= 0 or train_sample_count <= 0:
            raise ValueError("Training budget and sample count must be positive")
        self.mode, self.interval = mode, interval
        self.total_steps, self.train_sample_count = total_steps, train_sample_count
        self.bucket = 0

    def seek(self, step, samples_seen):
        """Position a schedule after a restored checkpoint without replaying events."""
        self.bucket = self._bucket(step, samples_seen)

    def _bucket(self, step, samples_seen):
        if self.mode == "steps":
            progress = step
        elif self.mode == "percent":
            progress = 100 * step / self.total_steps
        else:
            progress = samples_seen / self.train_sample_count
        return math.floor(progress / self.interval + 1e-10)

    def due(self, step, samples_seen):
        bucket = self._bucket(step, samples_seen)
        crossed = bucket > self.bucket
        self.bucket = max(self.bucket, bucket)
        return crossed

    def would_trigger(self, step, samples_seen):
        return self._bucket(step, samples_seen) > self.bucket
