"""Compare explicit Adam/Flip/Filter runs offline, including repeated-seed group evaluations."""

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
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "adam": {"label": "Adam", "color": "#1f77b4", "linestyle": "-"},
    "flip": {"label": "DPCO Flip", "color": "#d62728", "linestyle": "-"},
    "filter": {"label": "DPCO Filter", "color": "#ff7f0e", "linestyle": "--"},
}
FIGURES = (
    ("train_ce_vs_step", "step", "ce", "Training step", "Train CE"),
    ("train_accuracy_vs_step", "step", "accuracy", "Training step", "Train accuracy (%)"),
    ("train_ce_vs_time", "time", "ce", None, "Train CE"),
    ("train_accuracy_vs_time", "time", "accuracy", None, "Train accuracy (%)"),
    ("ms_per_step_vs_step", "cost", "ms", "Training step", "Training time (ms/step)"),
)
EVALUATION_FIGURES = (
    ("clean_train_ce_vs_step", "step", "clean_ce", "Training step", "Clean training-subset CE"),
    ("clean_train_ce_vs_time", "time", "clean_ce", "Training-loop wall time (min)", "Clean training-subset CE"),
    ("original_train_accuracy_vs_step", "step", "original_accuracy_percent", "Training step", "Original-label train accuracy (%)"),
    ("original_train_accuracy_vs_time", "time", "original_accuracy_percent", "Training-loop wall time (min)", "Original-label train accuracy (%)"),
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


@dataclass
class EvaluationRun:
    method: str
    directory: Path
    points: list
    sources: dict
    metadata: dict | None
    seed: int | None
    seed_source: str | None


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
    metadata = load_metadata(method, directory, sources)
    return Run(method, directory, train, windows, sources, metadata)


def load_metadata(method, directory, sources):
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
    return metadata


def load_evaluation_run(method, directory, smooth_window=1):
    """Use fixed train groups, with primary-evaluation completion timestamps."""
    if smooth_window < 1:
        raise ValueError("eval-smooth-window must be positive")
    directory = Path(directory).resolve()
    sources, evaluations, groups = {}, {}, {}
    previous_step, previous_wall, previous_seen = -1, -1., -1
    for row in read_csv(directory / "evaluation.csv", {"step", "samples_seen", "elapsed_wall_s"}, sources):
        source = f"{directory}/evaluation.csv"
        step = number(row, "step", source, integer=True)
        wall = number(row, "elapsed_wall_s", source)
        seen = number(row, "samples_seen", source, integer=True)
        if step <= previous_step or wall <= previous_wall or seen <= previous_seen:
            raise ValueError(f"{source}: steps, samples_seen and wall timestamps must strictly increase")
        evaluations[step] = {"samples_seen": seen, "wall_minutes": wall / 60}
        previous_step, previous_wall, previous_seen = step, wall, seen
    required = {"step", "samples_seen", "group", "label_source", "sample_count", "ce", "top1"}
    for row in read_csv(directory / "group_evaluation.csv", required, sources):
        source = f"{directory}/group_evaluation.csv"
        step = number(row, "step", source, integer=True)
        key = (step, row["group"])
        if key in groups:
            raise ValueError(f"{source}: duplicate group/step {key}")
        if step not in evaluations or number(row, "samples_seen", source, integer=True) != evaluations[step]["samples_seen"]:
            raise ValueError(f"{source}: group step/samples_seen has no exact evaluation match: {key}")
        groups[key] = row
    points = []
    group_sizes = None
    for step, evaluation in evaluations.items():
        selected = []
        for group in ("clean", "corrupted_original"):
            row = groups.get((step, group))
            source = f"{directory}/group_evaluation.csv step {step} group {group}"
            if row is None or row["label_source"] != "original":
                raise ValueError(f"{source}: missing original-label group")
            count = number(row, "sample_count", source, integer=True)
            accuracy = number(row, "top1", source) if count else 0.
            if accuracy > 1:
                raise ValueError(f"{source}: top1 must be a fraction in [0, 1]")
            ce = number(row, "ce", source) if count else 0.
            selected.append((count, accuracy, ce))
        (clean_n, clean_acc, clean_ce), (corrupted_n, corrupted_acc, _) = selected
        if clean_n == 0:
            raise ValueError(f"{directory}: clean training subset is empty; its CE is undefined")
        sizes = (clean_n, corrupted_n)
        if group_sizes is not None and sizes != group_sizes:
            raise ValueError(f"{directory}: fixed training-group sizes change across evaluations")
        group_sizes = sizes
        points.append({"step": step, **evaluation, "clean_samples": clean_n,
            "corrupted_samples": corrupted_n, "raw_clean_ce": clean_ce,
            "raw_original_accuracy_percent": 100 * (clean_n * clean_acc + corrupted_n * corrupted_acc) / (clean_n + corrupted_n)})
    for i, point in enumerate(points):
        prefix = points[max(0, i + 1 - smooth_window):i + 1]
        for metric in ("clean_ce", "original_accuracy_percent"):
            point["smooth_" + metric] = mean(row["raw_" + metric] for row in prefix)
    metadata = load_metadata(method, directory, sources)
    name_seed = re.search(r"(?:^|_)adam_(\d+)(?:_|$)", directory.name)
    seed = int(name_seed[1]) if name_seed else None
    seed_source = "directory_name_unverified" if name_seed else None
    if metadata is not None and "seed" in metadata["config"]:
        config_seed = number(metadata["config"], "seed", f"{directory}: metadata config", integer=True)
        if seed is not None and seed != config_seed:
            raise ValueError(f"{directory}: directory seed disagrees with metadata")
        seed, seed_source = config_seed, "metadata"
    return EvaluationRun(method, directory, points, sources, metadata, seed, seed_source)


def interpolate(points, xkey, x, value_key):
    """Piecewise linear display alignment; never extrapolate measurements."""
    xs = [point[xkey] for point in points]
    if x < xs[0] or x > xs[-1]:
        raise ValueError("Time alignment cannot extrapolate")
    left = bisect_right(xs, x) - 1
    if xs[left] == x:
        return points[left][value_key]
    a, b = points[left], points[left + 1]
    weight = (x - a[xkey]) / (b[xkey] - a[xkey])
    return a[value_key] + weight * (b[value_key] - a[value_key])


def evaluation_aggregates(runs):
    """Equal weight per run; step nodes exact, wall nodes aligned within overlap."""
    aggregated = {"step": [], "time": []}
    for method in STYLES:
        selected = [run for run in runs if run.method == method]
        grids = [tuple((p["step"], p["samples_seen"]) for p in run.points) for run in selected]
        if len(set(grids)) != 1:
            raise ValueError(f"{method}: repeated runs must have identical evaluation step/samples_seen nodes")
        seeds = [run.seed for run in selected if run.seed is not None]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{method}: duplicate training seed")
        lower = max(run.points[0]["wall_minutes"] for run in selected)
        upper = min(run.points[-1]["wall_minutes"] for run in selected)
        if lower > upper:
            raise ValueError(f"{method}: repeated runs have no common wall-time coverage")
        time_grid = sorted({lower, upper} | {p["wall_minutes"] for run in selected for p in run.points
                                             if lower <= p["wall_minutes"] <= upper})
        for axis, grid in (("step", [p["step"] for p in selected[0].points]), ("time", time_grid)):
            for index, x in enumerate(grid):
                row = {"method": method, "step" if axis == "step" else "wall_minutes": x,
                       "run_count": len(selected), "std_ddof": 1 if len(selected) > 1 else 0}
                if axis == "step":
                    row["samples_seen"] = selected[0].points[index]["samples_seen"]
                for metric in ("clean_ce", "original_accuracy_percent"):
                    for version in ("raw", "smooth"):
                        key = version + "_" + metric
                        values = ([run.points[index][key] for run in selected] if axis == "step" else
                                  [interpolate(run.points, "wall_minutes", x, key) for run in selected])
                        row[key + "_mean"] = mean(values)
                        row[key + "_std"] = stdev(values) if len(values) > 1 else 0.
                aggregated[axis].append(row)
    return aggregated


def evaluation_notes(runs):
    notes = metadata_notes(runs, include_seed=False)
    counts = {method: sum(run.method == method for run in runs) for method in STYLES}
    if len(set(counts.values())) != 1:
        notes.append(f"Methods have different repeat counts: {counts}.")
    for method in STYLES:
        selected = [run for run in runs if run.method == method]
        if any(run.seed_source != "metadata" for run in selected):
            notes.append(f"{method}: training-seed identity is not fully authenticated by metadata; directory-derived seeds are unverified.")
    known_sets = [{run.seed for run in runs if run.method == method and run.seed is not None} for method in STYLES]
    if any(seeds != known_sets[0] for seeds in known_sets[1:]):
        notes.append("Methods have different identified seed sets; no paired-seed inference is made.")
    if len({tuple((p["step"], p["samples_seen"]) for p in run.points) for run in runs}) > 1:
        notes.append("Methods use different evaluation step/sample grids; compare their actual recorded nodes.")
    return notes


def make_evaluation_figure(runs, aggregated, kind, metric, xlabel, ylabel, *, show_raw, plt):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.18, right=.97, bottom=.19, top=.97)
    ax.set_box_aspect(.8)
    xkey = "step" if kind == "step" else "wall_minutes"
    for method, style in STYLES.items():
        if show_raw:
            for run in runs:
                if run.method == method:
                    ax.plot([p[xkey] for p in run.points], [p["raw_" + metric] for p in run.points],
                            color=style["color"], linestyle=style["linestyle"], alpha=.18, linewidth=.6)
        rows = [row for row in aggregated[kind] if row["method"] == method]
        x = [row[xkey] for row in rows]
        y = [row["smooth_" + metric + "_mean"] for row in rows]
        sd = [row["smooth_" + metric + "_std"] for row in rows]
        ax.fill_between(x, [m - s for m, s in zip(y, sd)], [m + s for m, s in zip(y, sd)],
                        color=style["color"], alpha=.15, linewidth=0)
        ax.plot(x, y, **{**style, "label": f"{style['label']} (n={rows[0]['run_count']})"}, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    # Preserve the complete mean +/- SD band, including values outside physical
    # bounds. Do not clip uncertainty or truncate the early-training CE range.
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return fig


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
    return notes + metadata_notes(runs)


def metadata_notes(runs, *, include_seed=True):
    notes = []
    if any(run.metadata is None for run in runs):
        notes.append("Some metadata.json files are unavailable; those method labels come from explicit CLI arguments.")
    available = [run.metadata["config"] for run in runs if run.metadata is not None]
    for key in ("data", "model", "training", "seed", "global_batch_size", "execution", "evaluation", "checkpoints"):
        if key == "seed" and not include_seed:
            continue
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
        parser.add_argument(f"--{method}-dir", required=True, action="append",
                            help="Explicit run directory; repeat for each seed in clean-original mode. Absolute, repository-relative, or a run name beneath results/files.")
    parser.add_argument("--plot-set", choices=("training", "clean-original"), default="training",
                        help="training: original five batch/cost plots; clean-original: four fixed-train-group evaluation plots")
    parser.add_argument("--eval-smooth-window", type=int, default=1,
                        help="Trailing evaluation-node count for clean-original plots (default 1, no smoothing); independent of smooth-window")
    parser.add_argument("--smooth-window", type=int, default=100,
                        help="Trailing update count (default 100); 1 disables extra smoothing. Timing uses whole windows.")
    parser.add_argument("--time-axis", choices=("wall", "active"), default="wall")
    parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=True,
                        help="Show faint unsmoothed curves alongside smoothed ones")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output-name", default=None,
                        help="New directory name beneath results/figures; existing output is never overwritten")
    args = parser.parse_args(argv)
    if args.smooth_window < 1 or args.eval_smooth_window < 1 or args.dpi < 1:
        parser.error("smooth-window, eval-smooth-window and dpi must be positive")
    evaluation_mode = args.plot_set == "clean-original"
    if evaluation_mode and args.time_axis != "wall":
        parser.error("clean-original plots require --time-axis wall; evaluation.csv records wall time")
    if not evaluation_mode and any(len(getattr(args, method + "_dir")) != 1 for method in STYLES):
        parser.error("Repeated directories require --plot-set clean-original; training mode keeps one run per method")
    prefix = "dpco_clean_original_" if evaluation_mode else "dpco_training_"
    name = args.output_name or prefix + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        parser.error("output-name must be one safe directory name")
    output = ROOT / "results/figures" / name
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output-name")
    try:
        inputs = [(method, resolve_input(value)) for method in STYLES for value in getattr(args, method + "_dir")]
        directories = [directory for _, directory in inputs]
        if len(set(directories)) != len(directories):
            raise ValueError("Every input run directory must be distinct; duplicate runs would bias repeat statistics")
        if any(directory == output.resolve() or directory in output.resolve().parents for directory in directories):
            raise ValueError("Output must not be inside an input run directory")
        if evaluation_mode:
            runs = [load_evaluation_run(method, directory, args.eval_smooth_window) for method, directory in inputs]
            if len({(run.points[0]["clean_samples"], run.points[0]["corrupted_samples"]) for run in runs}) != 1:
                raise ValueError("Compared runs must use the same fixed clean/corrupted group sizes")
            data = evaluation_aggregates(runs)
        else:
            runs = [load_run(method, directory) for method, directory in inputs]
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
    notes = evaluation_notes(runs) if evaluation_mode else comparison_notes(runs)
    for note in notes:
        print("Note: " + note, file=sys.stderr)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    figures = EVALUATION_FIGURES if evaluation_mode else FIGURES
    for filename, kind, metric, xlabel, ylabel in figures:
        if evaluation_mode:
            fig = make_evaluation_figure(runs, data, kind, metric, xlabel, ylabel, show_raw=args.raw, plt=plt)
        else:
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
    if evaluation_mode:
        raw_rows = [{"method": run.method, "run_directory": str(run.directory), "seed": run.seed,
                     "seed_source": run.seed_source, **point} for run in runs for point in run.points]
        exports = {"evaluation_runs.csv": raw_rows, "evaluation_step_curves.csv": data["step"],
                   "evaluation_time_curves.csv": data["time"]}
    else:
        exports = {filename: [row for pair in data.values() for row in pair[index]]
                   for filename, index in (("step_curves.csv", 0), ("timing_curves.csv", 1))}
    for filename, rows in exports.items():
        write_csv(output / filename, rows)
        artifacts.append(filename)
    for run in runs:
        for path, expected in run.sources.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Input changed while plotting: {path}; output has no completed manifest")
    manifest = {"format": "dpco-group-evaluation-plots-v1" if evaluation_mode else "dpco-training-plots-v1", "created_at": datetime.now().astimezone().isoformat(),
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
        "runs": [] if evaluation_mode else [{"method": run.method, "directory": str(run.directory), "steps": len(run.train),
            "windows": len(run.windows), "sources_sha256": {str(p): digest for p, digest in run.sources.items()},
            "total_train_window_s": sum(w["duration_s"] for w in run.windows),
            "weighted_mean_ms_per_step": 1000 * sum(w["duration_s"] for w in run.windows) / len(run.train),
            "last_window_wall_s": run.windows[-1]["wall_s"]} for run in runs]}
    if evaluation_mode:
        manifest.update({
            "smoothing": f"Each run uses an equally weighted trailing window of {args.eval_smooth_window} evaluation nodes with a growing prefix, then runs are aggregated equally. smooth-window is unused in this mode.",
            "metric_scope": "Fixed training-set audit, not held-out validation. Clean CE is ordinary unweighted CE on the unchanged-label subset. Full-train original-label accuracy is 100*(n_clean*top1_clean+n_corrupted*top1_corrupted_original)/(n_clean+n_corrupted); corrupted_noisy is excluded. Original labels are used only for evaluation.",
            "statistics": {"weighting": "equal weight per run", "band": "mean +/- one sample standard deviation (ddof=1), not a confidence interval; single-run std convention is 0 with no uncertainty estimate",
                "step_alignment": "Exact matching recorded step and samples_seen nodes within each method; no step interpolation",
                "time_alignment": "Per method: piecewise-linear interpolation on the union of recorded evaluation completion times restricted to the overlap of all repeats, including overlap endpoints; no extrapolation or duration normalization. Interpolated values are display alignment, not additional evaluations.",
                "time_support_minutes": {method: {"start": min(r["wall_minutes"] for r in data["time"] if r["method"] == method),
                                                  "end": max(r["wall_minutes"] for r in data["time"] if r["method"] == method)} for method in STYLES}},
            "timing": {"axis": "wall", "source": "evaluation.csv elapsed_wall_s/60 joined to group_evaluation.csv by exact step and samples_seen",
                "wall": "Metric availability after primary and group evaluations; includes in-loop training/JIT/evaluation/checkpoint overhead, excludes pre-loop data preparation and model initialization. Step-0 timestamps are retained.",
                "raw_curves": "Faint individual runs retain their own complete measured time ranges; aggregate bands use common coverage per method."},
            "runs": [{"method": run.method, "directory": str(run.directory), "seed": run.seed, "seed_source": run.seed_source,
                "evaluation_nodes": len(run.points), "first_step": run.points[0]["step"], "last_step": run.points[-1]["step"],
                "clean_samples": run.points[0]["clean_samples"], "corrupted_samples": run.points[0]["corrupted_samples"],
                "sources_sha256": {str(p): digest for p, digest in run.sources.items()}} for run in runs]})
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Completed: {len(figures)} figures. Output: {output}")
    return output


if __name__ == "__main__":
    main()
