"""Compare three explicit runs offline; never import the training runtime."""

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "adam": {"label": "Adam", "color": "#1f77b4", "linestyle": "-"},
    "flip": {"label": "DPCO Flip", "color": "#d62728", "linestyle": "-"},
    "filter": {"label": "DPCO Filter", "color": "#1f77b4", "linestyle": "--"},
}
FIGURES = (
    ("train_ce_vs_step", "step", "ce", "Training step", "Train CE"),
    ("train_accuracy_vs_step", "step", "accuracy", "Training step", "Train accuracy (%)"),
    ("train_ce_vs_time", "time", "ce", None, "Train CE"),
    ("train_accuracy_vs_time", "time", "accuracy", None, "Train accuracy (%)"),
    ("ms_per_step_vs_step", "cost", "ms", "Training step", "Training time (ms/step)"),
)


def number(row, key, source, *, integer=False, minimum=0):
    try:
        value = float(row[key])
        if not math.isfinite(value) or value < minimum or (integer and not value.is_integer()):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{source}: invalid {key}={row.get(key)!r}") from None
    return int(value) if integer else value


def read_csv(path, required, sources):
    data = path.read_bytes()
    reader = csv.DictReader(StringIO(data.decode("utf-8-sig"), newline=""))
    if not required <= set(reader.fieldnames or ()):
        raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or ()))}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    sources[path] = hashlib.sha256(data).hexdigest()
    return rows


@dataclass
class Run:
    method: str
    directory: Path
    train: list
    windows: list
    sources: dict
    metadata: dict | None = None


def resolve_input(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / "results/files" / path if len(path.parts) == 1 else ROOT / path
    return path.resolve()


def load_run(method, directory):
    directory = Path(directory).resolve()
    sources = {}
    train = []
    for row in read_csv(directory / "train_steps.csv",
                        {"step", "train_ce", "train_top1", "valid_batch_size"}, sources):
        source = f"{directory}/train_steps.csv row {len(train) + 2}"
        step = number(row, "step", source, integer=True, minimum=1)
        if step != len(train) + 1:
            raise ValueError(f"{source}: steps must be a contiguous prefix starting at 1")
        accuracy = number(row, "train_top1", source)
        if accuracy > 1:
            raise ValueError(f"{source}: train_top1 must be a fraction in [0, 1]")
        train.append({"step": step, "ce": number(row, "train_ce", source),
                      "accuracy": 100 * accuracy,
                      "samples": number(row, "valid_batch_size", source, integer=True, minimum=1)})
    windows = []
    previous_step, previous_wall = 0, 0.
    for row in read_csv(directory / "timing.csv",
                        {"step", "event", "duration_s", "window_steps", "elapsed_wall_s"}, sources):
        if row["event"] != "train_window":
            continue
        source = f"{directory}/timing.csv train_window {len(windows) + 1}"
        end = number(row, "step", source, integer=True, minimum=1)
        count = number(row, "window_steps", source, integer=True, minimum=1)
        duration = number(row, "duration_s", source)
        wall = number(row, "elapsed_wall_s", source)
        if duration <= 0 or end - count != previous_step or end > len(train):
            raise ValueError(f"{source}: invalid duration, overlapping/gapped window or missing training rows")
        if wall < previous_wall or wall - previous_wall + 1e-6 < duration:
            raise ValueError(f"{source}: inconsistent elapsed wall time")
        windows.append({"step": end, "steps": count, "duration_s": duration, "wall_s": wall})
        previous_step, previous_wall = end, wall
    if not windows or previous_step != len(train):
        raise ValueError(f"{directory}: train_window coverage must match all training rows; use fully flushed CSVs")
    metadata = None
    # Metadata is optional for portable CSV exports, but validates labels when available.
    if directory.parent.name == "files":
        path = directory.parent.parent / "logs" / directory.name / "metadata.json"
        if path.is_file():
            data = path.read_bytes()
            sources[path] = hashlib.sha256(data).hexdigest()
            metadata = json.loads(data)
            config = metadata["config"]
            valid = config["training"].get("optimizer", "sgd") == "adam"
            if method == "adam":
                valid = valid and config["method"]["name"] == "adam"
            else:
                valid = (valid and config["method"]["name"] == "dpco"
                         and config["reference"]["name"] == "ema"
                         and config["method"].get("negative_weight", -1) == (0 if method == "filter" else -1))
            if not valid:
                raise ValueError(f"{path}: metadata does not match the requested {method} label")
    return Run(method, directory, train, windows, sources, metadata)


def prefixes(rows, key, *, weight=None):
    result = [0.]
    for row in rows:
        result.append(result[-1] + row[key] * (row[weight] if weight else 1))
    return result


def curves(run, smooth_window):
    """Step smoothing uses W updates; timing smoothing never splits recorded windows."""
    if smooth_window < 1:
        raise ValueError("smooth_window must be positive")
    samples = prefixes(run.train, "samples")
    ce = prefixes(run.train, "ce", weight="samples")
    accuracy = prefixes(run.train, "accuracy", weight="samples")

    def mean(values, start, end):
        return (values[end] - values[start]) / (samples[end] - samples[start])

    step_rows = []
    for row in run.train:
        end = row["step"]
        start = max(0, end - smooth_window)
        step_rows.append({"method": run.method, "step": end,
            "raw_ce": row["ce"], "raw_accuracy_percent": row["accuracy"],
            "smooth_ce": mean(ce, start, end),
            "smooth_accuracy_percent": mean(accuracy, start, end)})
    ends = [row["step"] for row in run.windows]
    durations = prefixes(run.windows, "duration_s")
    timing_rows = []
    for i, window in enumerate(run.windows):
        end, count = window["step"], window["steps"]
        first = bisect_right(ends, end - smooth_window, hi=i + 1)
        start = run.windows[first]["step"] - run.windows[first]["steps"]
        timing_rows.append({"method": run.method, "step": end,
            "window_steps": count, "duration_s": window["duration_s"],
            "wall_minutes": window["wall_s"] / 60, "active_minutes": durations[i + 1] / 60,
            "raw_ce": mean(ce, end - count, end),
            "raw_accuracy_percent": mean(accuracy, end - count, end),
            "smooth_ce": mean(ce, start, end),
            "smooth_accuracy_percent": mean(accuracy, start, end),
            "raw_ms_per_step": 1000 * window["duration_s"] / count,
            "smooth_ms_per_step": 1000 * (durations[i + 1] - durations[first]) / (end - start),
            "smoothing_steps": end - start})
    return step_rows, timing_rows


def comparison_notes(runs):
    notes = []
    if len({len(run.train) for run in runs}) > 1:
        notes.append("Runs have different recorded step budgets; curves end at their actual last points.")
    if len({tuple(sorted({row['samples'] for row in run.train})) for run in runs}) > 1:
        notes.append("Recorded batch sizes differ; equal steps do not mean equal samples seen.")
    if any(run.metadata is None for run in runs):
        notes.append("Some metadata.json files are unavailable; those method labels come from explicit CLI arguments.")
    available = [run.metadata["config"] for run in runs if run.metadata is not None]
    for key in ("data", "model", "training", "seed", "global_batch_size", "execution", "evaluation", "checkpoints"):
        if len({json.dumps(config.get(key), sort_keys=True) for config in available}) > 1:
            notes.append(f"Metadata differs in {key}; this is not a matched comparison on that setting.")
    references = [run.metadata["config"]["reference"] for run in runs
                  if run.method != "adam" and run.metadata is not None]
    if len({json.dumps(value, sort_keys=True) for value in references}) > 1:
        notes.append("Flip and Filter use different reference configurations.")
    fingerprints = [run.metadata.get("experiment", {}).get("data_fingerprint") for run in runs
                    if run.metadata is not None]
    if len({value for value in fingerprints if value is not None}) > 1:
        notes.append("Recorded data fingerprints differ; runs did not use the same data identity.")
    return notes


def make_figure(data, kind, metric, xlabel, ylabel, *, time_axis, show_raw, plt):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.17, right=.97, bottom=.19, top=.97)
    ax.set_box_aspect(.8)
    for method, (step_rows, timing_rows) in data.items():
        rows = step_rows if kind == "step" else timing_rows
        xkey = ("wall_minutes" if time_axis == "wall" else "active_minutes") if kind == "time" else "step"
        suffix = {"ce": "ce", "accuracy": "accuracy_percent", "ms": "ms_per_step"}[metric]
        x = [row[xkey] for row in rows]
        raw = [row["raw_" + suffix] for row in rows]
        smooth = [row["smooth_" + suffix] for row in rows]
        style = STYLES[method]
        if show_raw and raw != smooth:
            ax.plot(x, raw, color=style["color"], linestyle=style["linestyle"], alpha=.18, linewidth=.6)
        ax.plot(x, smooth, **style, linewidth=1.5)
    if xlabel is None:
        xlabel = "Training-loop wall time (min)" if time_axis == "wall" else "Active training time (min)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, **({"top": 100} if metric == "accuracy" else {}))
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return fig


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for method in STYLES:
        parser.add_argument(f"--{method}-dir", required=True,
                            help="Absolute directory, repository-relative path, or a run name beneath results/files")
    parser.add_argument("--smooth-window", type=int, default=100,
                        help="Trailing update count (default 100); 1 disables extra smoothing. Timing uses whole windows.")
    parser.add_argument("--time-axis", choices=("wall", "active"), default="wall")
    parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=True,
                        help="Show faint unsmoothed curves alongside smoothed ones")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output-name", default=None,
                        help="New directory name beneath results/files/plots; existing output is never overwritten")
    args = parser.parse_args(argv)
    if args.smooth_window < 1 or args.dpi < 1:
        parser.error("smooth-window and dpi must be positive")
    name = args.output_name or "dpco_training_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        parser.error("output-name must be one safe directory name")
    output = ROOT / "results/files/plots" / name
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output-name")
    try:
        directories = [resolve_input(getattr(args, method + "_dir")) for method in STYLES]
        if len(set(directories)) != 3:
            raise ValueError("Specify three distinct run directories")
        if any(directory == output.resolve() or directory in output.resolve().parents for directory in directories):
            raise ValueError("Output must not be inside an input run directory")
        runs = [load_run(method, directory) for method, directory in zip(STYLES, directories)]
        data = {run.method: curves(run, args.smooth_window) for run in runs}
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        parser.error(str(error))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("Matplotlib is required; see plots/README.md for the nohup installation command (project extra: plot)")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "figure.facecolor": "white"})
    notes = comparison_notes(runs)
    for note in notes:
        print("Note: " + note, file=sys.stderr)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for filename, kind, metric, xlabel, ylabel in FIGURES:
        fig = make_figure(data, kind, metric, xlabel, ylabel,
                          time_axis=args.time_axis, show_raw=args.raw, plt=plt)
        try:
            for extension in dict.fromkeys(args.formats):
                target = output / (filename + "." + extension)
                # Do not use bbox_inches='tight': exported canvas must remain 5 x 4 inches.
                fig.savefig(target, dpi=args.dpi)
                artifacts.append(target.name)
        finally:
            plt.close(fig)
    for filename, index in (("step_curves.csv", 0), ("timing_curves.csv", 1)):
        write_csv(output / filename, [row for pair in data.values() for row in pair[index]])
        artifacts.append(filename)
    for run in runs:
        for path, expected in run.sources.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Input changed while plotting: {path}; output has no completed manifest")
    manifest = {"format": "dpco-training-plots-v1", "created_at": datetime.now().astimezone().isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "matplotlib": matplotlib.__version__,
        "arguments": vars(args), "styles": STYLES, "figure_inches": [5, 4], "axes_height_width_ratio": .8,
        "smoothing": "Step plots: sample-weighted trailing W updates with a growing prefix. Time/cost: whole trailing train_window records covering at least W updates (or the available prefix). W=1 leaves each timing window intact.",
        "timing": {"axis": args.time_axis,
            "wall": "Completed train_window endpoints; training-loop elapsed time includes initial evaluation, JIT and prior evaluation/checkpoint overhead, excludes pre-loop data preparation/model initialization.",
            "active": "Sum of train_window durations, including input handling and device synchronization; not pure accelerator kernel time.",
            "metric_alignment": "Sample-weighted window metrics; time coordinates are completed-window endpoints, not per-step dispatch timestamps."},
        "metric_scope": "Ordinary CE and top-1 on observed training labels, pre-update batch statistics; noisy-label fit is not clean-label generalization.",
        "comparison_notes": notes, "artifacts": artifacts,
        "runs": [{"method": run.method, "directory": str(run.directory), "steps": len(run.train),
            "windows": len(run.windows), "sources_sha256": {str(p): digest for p, digest in run.sources.items()},
            "total_train_window_s": sum(w["duration_s"] for w in run.windows),
            "weighted_mean_ms_per_step": 1000 * sum(w["duration_s"] for w in run.windows) / len(run.train),
            "last_window_wall_s": run.windows[-1]["wall_s"]} for run in runs]}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Completed: 5 figures. Output: {output}")
    return output


if __name__ == "__main__":
    main()
