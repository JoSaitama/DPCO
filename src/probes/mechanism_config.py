"""Dependency-light opt-in TPU mechanism configuration and CLI adapters."""

import argparse
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MechanismConfig:
    enabled: bool = False
    update_check: bool = True
    paired_loss: bool = True
    bank_moments: bool = True
    norm_matched: bool = False
    steps: tuple = (0, 5, 10)
    moment_steps: tuple = (0,)
    oracle_steps: tuple = ()
    absolute_lrs: tuple = (0., .001, .01, .05)
    lr_scales: tuple = (1., .5)
    windows: tuple = ()
    bank_batches: int = 2
    bank_seed: int = 42

    def validate(self, training, method):
        for name in ("enabled", "update_check", "paired_loss", "bank_moments", "norm_matched"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"mechanism.{name} must be boolean")
        for name in ("steps", "moment_steps", "oracle_steps"):
            values = getattr(self, name)
            if len(set(values)) != len(values) or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in values):
                raise ValueError(f"mechanism.{name} needs distinct nonnegative state steps")
        for name in ("absolute_lrs", "lr_scales"):
            values = getattr(self, name)
            if len(set(values)) != len(values) or any(isinstance(v, bool) or not math.isfinite(v)
                    or v < 0 or (name == "lr_scales" and v == 0) for v in values):
                raise ValueError(f"Invalid mechanism.{name}")
        previous_end = -1
        for window in self.windows:
            if (len(window) != 2 or any(isinstance(t, bool) or not isinstance(t, int) for t in window)
                    or not 0 <= window[0] < window[1] or window[0] < previous_end):
                raise ValueError("Mechanism windows must be sorted, disjoint [start,end) ranges")
            if self.enabled and window[1] > training.total_steps:
                raise ValueError("Mechanism windows must fit within the training update budget")
            previous_end = window[1]
        if isinstance(self.bank_batches, bool) or not isinstance(self.bank_batches, int) or self.bank_batches <= 0:
            raise ValueError("mechanism.bank_batches must be positive")
        if isinstance(self.bank_seed, bool) or not isinstance(self.bank_seed, int) or not 0 <= self.bank_seed < 2**32:
            raise ValueError("Invalid mechanism.bank_seed")
        if self.enabled:
            if method.name not in ("sgd", "dpco"):
                raise ValueError("Extended mechanisms currently cover original SGD/DPCO only; policy behavior is unchanged")
            if training.momentum != 0 or training.l2_coefficient != 0 or training.nesterov:
                raise ValueError("Extended mechanisms require momentum/L2/Nesterov off")
            if not (self.update_check or self.paired_loss or self.bank_moments or self.windows or self.oracle_steps):
                raise ValueError("Enable at least one extended mechanism check")
            if (self.paired_loss or self.oracle_steps) and not (self.absolute_lrs or self.lr_scales):
                raise ValueError("Paired checks require at least one LR list")

    @property
    def needs_bank(self):
        return bool(self.paired_loss or self.bank_moments or self.windows or self.oracle_steps)

    def due(self, step):
        return self.enabled and (step in self.steps and (self.update_check or self.paired_loss)
            or self.bank_moments and step in self.moment_steps
            or step in self.oracle_steps or any(start <= step < end for start, end in self.windows))


def parse_window(text):
    try:
        start, end = text.split(":")
        return int(start), int(end)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a window start:end, for example 0:12") from None


OPTIONS = ("mechanisms", "mechanism_update_check", "mechanism_paired_loss", "mechanism_bank_moments",
    "mechanism_norm_matched", "mechanism_steps", "mechanism_moment_steps", "mechanism_oracle_steps",
    "mechanism_absolute_lrs", "mechanism_lr_scales", "mechanism_windows", "mechanism_bank_batches", "mechanism_bank_seed")


def add_arguments(parser):
    parser.add_argument("--mechanisms", action=argparse.BooleanOptionalAction, default=None,
                        help="Opt-in TPU T1/T2/fixed-bank/window observers; default off")
    for name in ("update-check", "paired-loss", "bank-moments", "norm-matched"):
        parser.add_argument("--mechanism-"+name, action=argparse.BooleanOptionalAction, default=None)
    for name in ("steps", "moment-steps", "oracle-steps"):
        parser.add_argument("--mechanism-"+name, nargs="*", type=int)
    for name in ("absolute-lrs", "lr-scales"):
        parser.add_argument("--mechanism-"+name, nargs="*", type=float)
    parser.add_argument("--mechanism-windows", nargs="*", type=parse_window,
                        help="Disjoint continuous PRE-update windows, e.g. 0:12 20:30")
    parser.add_argument("--mechanism-bank-batches", type=int)
    parser.add_argument("--mechanism-bank-seed", type=int)


def overrides(args):
    return {name: getattr(args, name, None) for name in OPTIONS}


def changes(options):
    return {("enabled" if name == "mechanisms" else name.removeprefix("mechanism_")): value
            for name, value in options.items() if value is not None}


def cli_args(args):
    result = []
    for name, value in overrides(args).items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-"+flag[2:])
        elif name == "mechanism_windows":
            result.extend([flag, *(f"{a}:{b}" for a, b in value)])
        elif isinstance(value, (tuple, list)):
            result.extend([flag, *(str(v) for v in value)])
        else:
            result.extend([flag, str(value)])
    return result
