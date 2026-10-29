#!/usr/bin/env python3
"""Plot exported DPCO training curves on macOS without JAX or training imports.

Dependencies: Python 3.11+ and matplotlib. Read source CSV/JSON files only.
Example, from the code repository:
  python scripts/plot_training_results.py --input ../result/probe \
      --output ../reports/training_curves_20261027 --smooth-window 50
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
COLORS = {"SGD": "#2563EB", "DPCO-E": "#D62728", "DPCO-L": "#FF7F0E"}
GROUPS = {
    "continuous": ("constant", "continuous", None, "Continuous DPCO vs SGD"),
    "half": ("constant", "half", None, "Half/filter DPCO vs SGD"),
    "alternating_sgd_first": ("constant", "alternating", "sgd", "Alternating DPCO vs SGD (SGD first)"),
    "alternating_dpco_first": ("constant", "alternating", "dpco", "Alternating DPCO vs SGD (DPCO first)"),
    "cosine": ("cosine", "continuous", None, "Cosine decay to zero: DPCO vs SGD"),
    "inverse_sqrt": ("inverse_sqrt", "continuous", None, "Inverse-square-root decay: DPCO vs SGD"),
    "warmup10": ("constant", "warmup10", None, "10% SGD prefix before DPCO vs SGD"),
    "warmup_percent": ("constant", "warmup_percent", None, "10% vs 30% SGD prefix at fixed LR"),
}
DEFAULT_GROUPS = tuple(GROUPS)
TRAIN_COLUMNS = ("step", "samples_seen", "epoch_progress", "learning_rate", "valid_batch_size",
                 "train_ce", "train_top1", "elapsed_wall_s")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def trailing_mean(values, window):
    """Causal equal-step mean; the first window-1 points use the growing prefix."""
    if window < 1:
        raise ValueError("smooth-window must be at least 1")
    queue, total, result = deque(), 0., []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def method_identity(config):
    if config["training"].get("optimizer", "sgd") != "sgd":
        raise ValueError("This legacy report supports SGD runs only; Adam runs have different update semantics")
    method = config["method"]
    if method["name"] == "sgd":
        return "SGD", "sgd", None
    if method["name"] not in ("dpco", "dpco_policy"):
        raise ValueError(f"Unsupported method: {method['name']}")
    reference = config["reference"]["name"]
    if reference not in ("loss", "error"):
        raise ValueError(f"Unsupported reference: {reference}")
    family = "DPCO-E" if reference == "error" else "DPCO-L"
    warmup = method.get("dpco_warmup_percent", 0.)
    if warmup:
        return family, "warmup10" if warmup == 10 else f"warmup{warmup:g}", None
    mode = method.get("policy_mode", "continuous")
    return family, mode, method.get("alternating_first", "sgd") if mode == "alternating" else None


@dataclass
class Run:
    directory: Path
    metadata_path: Path
    metadata: dict
    rows: list
    family: str
    mode: str
    first: str | None
    schedule: str
    initial_lr: float
    status: str

    @property
    def name(self):
        return self.directory.name

    @property
    def config(self):
        return self.metadata["config"]

    @property
    def commit(self):
        return self.metadata.get("environment", {}).get("git_commit")

    @property
    def key(self):
        return self.family, self.mode, self.first, self.schedule, self.initial_lr


def validate_training(rows, config, name):
    total = config["training"]["total_steps"]
    if len(rows) != total:
        raise ValueError(f"{name}: expected {total} training rows, found {len(rows)}")
    b = config["global_batch_size"]
    schedule = config["training"].get("lr_schedule", "constant")
    initial = config["training"]["learning_rate"]
    floor = config["training"].get("min_learning_rate", 0.)
    for index, row in enumerate(rows):
        if row["step"] != index + 1:
            raise ValueError(f"{name}: steps must be exactly 1..{total}, with no duplicates/gaps")
        if not all(math.isfinite(row[key]) for key in TRAIN_COLUMNS):
            raise ValueError(f"{name}: non-finite training metric at row {index+1}")
        if row["valid_batch_size"] != b or row["samples_seen"] != (index+1)*b:
            raise ValueError(f"{name}: batch size/sample exposure differs from the full-batch protocol")
        if not 0 <= row["train_top1"] <= 1 or row["train_ce"] < 0:
            raise ValueError(f"{name}: invalid CE or accuracy fraction at row {index+1}")
        expected = initial
        if schedule == "cosine":
            expected = floor + .5*(initial-floor)*(1+math.cos(math.pi*index/max(total-1, 1)))
        elif schedule == "inverse_sqrt":
            expected = initial/math.sqrt(index+1)
        elif schedule != "constant":
            raise ValueError(f"{name}: unsupported schedule {schedule}")
        if not math.isclose(row["learning_rate"], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{name}: recorded LR does not match metadata at update {index+1}")


def validate_policy(run):
    path = run.directory/"policy_steps.csv"
    required = run.config["method"]["name"] == "dpco_policy" or run.config["method"].get("dpco_warmup_percent", 0) > 0
    if not path.exists():
        if required:
            raise ValueError(f"{run.name}: missing policy_steps.csv")
        return
    _, rows = read_csv(path)
    if len(rows) != len(run.rows):
        raise ValueError(f"{run.name}: policy and training row counts differ")
    count = math.ceil(len(rows)*run.config["method"].get("dpco_warmup_percent", 0)/100)
    for t, row in enumerate(rows):
        alpha = 0. if t < count else (1. if run.mode.startswith("warmup") else
            .5 if run.mode == "half" else float((t+int(run.first == "dpco")) % 2)
            if run.mode == "alternating" else 1.)
        executed = {0.: "sgd", .5: "half_dpco", 1.: "dpco"}[alpha]
        if int(row["step"]) != t+1 or float(row["policy_alpha"]) != alpha or row["executed_method"] != executed:
            raise ValueError(f"{run.name}: policy mismatch at update {t+1}")


def discover_runs(root, args):
    """Join split files_*/logs_* exports by exact run_id, never filename guessing."""
    metadata = {}
    for path in sorted(root.rglob("metadata.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        if name in metadata and metadata[name][1] != record:
            raise ValueError(f"Conflicting metadata for {name}; supply a narrower --input directory")
        metadata[name] = path, record
    runs, exclusions = [], []
    for train_path in sorted(root.rglob("train_steps.csv")):
        name = train_path.parent.name
        if args.run_ids and name not in args.run_ids:
            exclusions.append({"run_id": name, "reason": "excluded by --run-ids"})
            continue
        if name not in metadata:
            raise ValueError(f"Missing metadata.json for {name}; export the matching logs directory")
        metadata_path, meta = metadata[name]
        config = meta["config"]
        filters = (config["data"]["name"] == args.dataset, config["model"]["name"] == args.model,
            config["seed"] == args.seed, config["global_batch_size"] == args.batch_size,
            config["training"]["total_steps"] == args.steps,
            any(math.isclose(config["training"]["learning_rate"], rate, rel_tol=1e-12) for rate in args.lrs))
        if not all(filters):
            exclusions.append({"run_id": name, "reason": "outside requested dataset/model/seed/B/T/initial-LR cohort"})
            continue
        family, mode, first = method_identity(config)
        columns, raw = read_csv(train_path)
        if not set(TRAIN_COLUMNS).issubset(columns):
            raise ValueError(f"{train_path}: missing required training columns")
        _, summaries = read_csv(train_path.parent/"summary.csv")
        status = summaries[-1]["status"] if summaries else "missing"
        if status != "completed":
            exclusions.append({"run_id": name, "reason": f"run status {status}; not plotted as completed"})
            continue
        rows = [{field: float(row[field]) for field in TRAIN_COLUMNS} for row in raw]
        validate_training(rows, config, name)
        schedule = config["training"].get("lr_schedule", "constant")
        run = Run(train_path.parent, metadata_path, meta, rows, family, mode, first,
                  schedule, float(config["training"]["learning_rate"]), status)
        validate_policy(run)
        runs.append(run)
    return runs, exclusions


def select_group(runs, group, rates, *, warmup_comparison_lr=.01, warmup_percents=(10., 30.)):
    schedule, mode, first, _ = GROUPS[group]
    if group == "warmup_percent":
        # Hold LR fixed: the line-style variable is the SGD-prefix percentage.
        # Never substitute a different LR or a different warmup percentage.
        modes = {f"warmup{percent:g}" for percent in warmup_percents}
        selected = [r for r in runs if r.schedule == schedule
                    and math.isclose(r.initial_lr, warmup_comparison_lr, rel_tol=1e-12)
                    and (r.family == "SGD" or r.mode in modes)]
        keys = [(r.family, r.initial_lr, r.mode) for r in selected]
        expected = [("SGD", warmup_comparison_lr, "sgd")]
        expected += [(family, warmup_comparison_lr, f"warmup{percent:g}")
                     for family in ("DPCO-E", "DPCO-L") for percent in warmup_percents]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            names = [r.name for r in selected if (r.family, r.initial_lr, r.mode) in duplicates]
            raise ValueError(f"Ambiguous repeated runs in {group}: {names}. Select explicitly with --run-ids.")
        return sorted(selected, key=lambda r: (list(COLORS).index(r.family), r.mode)), [key for key in expected if key not in keys]
    selected = [r for r in runs if r.schedule == schedule and (r.family == "SGD" or
        r.mode == mode and r.first == first)]
    keys = [(r.family, r.initial_lr) for r in selected]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        names = [r.name for r in selected if (r.family, r.initial_lr) in duplicates]
        raise ValueError(f"Ambiguous repeated runs in {group}: {names}. Select explicitly with --run-ids.")
    missing = [(family, rate) for family in COLORS for rate in rates if (family, rate) not in keys]
    return sorted(selected, key=lambda r: (list(COLORS).index(r.family), r.initial_lr)), missing


def x_values(run, axis):
    if axis == "cumulative_eta_before":
        total, values = 0., []
        for row in run.rows:
            values.append(total)
            total += row["learning_rate"]
        return values
    return [r[axis] for r in run.rows]


def comparability_notes(runs):
    notes = []
    if len({r.commit for r in runs}) > 1:
        notes.append("Different code revisions across curves; see plot_manifest.json.")
    if len({r.metadata.get("experiment", {}).get("data_fingerprint") for r in runs}) > 1:
        notes.append("Different or missing data fingerprints; check run metadata.")
    settings = {json.dumps({key: r.config["training"].get(key, default) for key, default in
        (("momentum", 0.), ("nesterov", False), ("l2_coefficient", 0.))}, sort_keys=True) for r in runs}
    if len(settings) > 1:
        notes.append("Optimizer extras differ across curves.")
    if len({r.config["model"].get("precision", "float32") for r in runs}) > 1:
        notes.append("Model compute precision differs across curves.")
    return notes


def shared_limits(plans, args):
    """Use common y limits for policy figures with the same LR schedule."""
    by_schedule = {}
    for group, runs, _ in plans:
        selected = by_schedule.setdefault(GROUPS[group][0], {})
        selected.update({run.name: run for run in runs})
    limits = {}
    for schedule, runs in by_schedule.items():
        ce_values, acc_values = [], []
        for run in runs.values():
            count = min(len(run.rows), args.max_step or len(run.rows))
            ce_values.extend(trailing_mean([r["train_ce"] for r in run.rows], args.smooth_window)[:count])
            acc_values.extend(trailing_mean([100*r["train_top1"] for r in run.rows], args.smooth_window)[:count])
        low, high = min(ce_values), max(ce_values)
        pad = max(.02, .05*(high-low))
        limits[schedule] = ((max(0., low-pad), high+pad), (0., min(100., max(5., 5*math.ceil((max(acc_values)+1)/5)))))
    return limits


def learning_rate_styles(rates):
    """Keep LR identity consistent while making dashed curves clearly sparse."""
    patterns = ((0, (8, 5)), "-", (0, (8, 4, 1.5, 4)), (0, (1.5, 4)))
    return dict(zip(sorted(rates), patterns))


def curve_color(family, dashed, lighten):
    """A pale version of the same method color is a redundant LR cue."""
    from matplotlib.colors import to_rgb
    base = to_rgb(COLORS[family])
    return tuple(channel + lighten*(1-channel) for channel in base) if dashed else base


def plot_group(group, runs, missing, args, output, limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "pdf.fonttype": 42, "svg.fonttype": "none"})
    styles = learning_rate_styles(args.lrs)
    compare_percent = group == "warmup_percent"
    percent_styles = learning_rate_styles(args.warmup_percents) if compare_percent else {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=.085, right=.98, top=.98, bottom=.22, wspace=.30)
    # Set the physical plotting-box ratio, not a ratio of data-unit scales.
    # Width:height = 1:0.8 by default, independently of the metric's range.
    for ax in axes:
        ax.set_box_aspect(args.panel_height_ratio)
    xlabel = {"step": "Update step", "samples_seen": "Samples seen",
              "elapsed_wall_s": "Elapsed time (s)",
              "cumulative_eta_before": "Cumulative learning rate"}[args.x_axis]
    for run in runs:
        style = ("-" if run.family == "SGD" else
                 percent_styles[run.config["method"]["dpco_warmup_percent"]]) if compare_percent else styles[run.initial_lr]
        x = x_values(run, args.x_axis)
        ce = trailing_mean([r["train_ce"] for r in run.rows], args.smooth_window)
        acc = trailing_mean([100*r["train_top1"] for r in run.rows], args.smooth_window)
        indices = [i for i, row in enumerate(run.rows) if args.max_step is None or row["step"] <= args.max_step]
        for ax, values in zip(axes, (ce, acc)):
            dashed = style != "-"
            ax.plot([x[i] for i in indices], [values[i] for i in indices],
                color=curve_color(run.family, dashed, args.dash_lighten),
                linestyle=style, linewidth=args.line_width,
                solid_capstyle="round", dash_capstyle="butt")
    for ax, ylabel in zip(axes, ("Train CE", "Train accuracy (%)")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#ECEFF3", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.margins(x=.01)
    for ax, interval in zip(axes, limits[GROUPS[group][0]]):
        ax.set_ylim(*interval)
    handles = [Line2D([0], [0], color=color, lw=args.line_width, label=family) for family, color in COLORS.items()]
    if compare_percent:
        handles += [Line2D([0], [0], color="#6B7280" if percent_styles[percent] != "-" else "#374151",
                          lw=args.line_width, linestyle=percent_styles[percent],
                          label=f"SGD prefix {percent:g}%") for percent in sorted(args.warmup_percents)]
    else:
        handles += [Line2D([0], [0], color="#6B7280" if styles[rate] != "-" else "#374151",
                      lw=args.line_width, linestyle=styles[rate],
                      label=f"Initial LR {rate:g}") for rate in sorted(args.lrs)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=len(handles),
               frameon=False, fontsize=10, handlelength=3.8, columnspacing=1.5)
    # Scientific context and comparability notes remain in the manifest/gallery,
    # leaving exported figures with only axes, ticks and the requested legend.
    notes = comparability_notes(runs)
    if compare_percent:
        notes.append(f"Constant LR {args.warmup_comparison_lr:g}; line styles encode SGD-prefix percentage, not LR. SGD has no prefix switch.")
    if missing:
        notes.append("Missing curves: " + "; ".join(str(item) for item in missing))
    paths = []
    for extension in args.formats:
        path = output/f"{group}.{extension}"
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    title = (f"SGD prefix {' vs '.join(f'{p:g}%' for p in args.warmup_percents)} at constant LR {args.warmup_comparison_lr:g}"
             if compare_percent else GROUPS[group][3])
    return {"group": group, "title": title, "files": paths,
            "style_dimension": "sgd_prefix_percent" if compare_percent else "initial_lr",
            "fixed_initial_lr": args.warmup_comparison_lr if compare_percent else None,
            "run_ids": [r.name for r in runs], "missing_curves": missing, "comparison_notes": notes}


def export_inventory(runs, output):
    inventory, summaries, csv_inventory, fingerprints = [], [], [], {}
    schemas = {}
    for run in runs:
        train = run.directory/"train_steps.csv"
        config = run.config
        inventory.append(dict(run_id=run.name, family=run.family, mode=run.mode,
            alternating_first=run.first, initial_lr=run.initial_lr, schedule=run.schedule,
            seed=config["seed"], batch_size=config["global_batch_size"], steps=len(run.rows), status=run.status,
            code_commit=run.commit, source_sha256=run.metadata.get("environment", {}).get("source_sha256"),
            data_fingerprint=run.metadata.get("experiment", {}).get("data_fingerprint"),
            train_csv=str(train), metadata_json=str(run.metadata_path)))
        for label, tail in (("all", run.rows), ("last100", run.rows[-100:]), ("last1000", run.rows[-1000:])):
            summaries.append(dict(run_id=run.name, family=run.family, mode=run.mode, first=run.first,
                schedule=run.schedule, initial_lr=run.initial_lr, interval=label,
                mean_train_ce=sum(r["train_ce"] for r in tail)/len(tail),
                mean_train_top1_percent=100*sum(r["train_top1"] for r in tail)/len(tail),
                row_count=len(tail), first_step=tail[0]["step"], last_step=tail[-1]["step"]))
        for path in sorted(run.directory.glob("*.csv")):
            fields, rows = read_csv(path)
            digest = sha256(path)
            fingerprints[str(path)] = digest
            csv_inventory.append(dict(run_id=run.name, file=path.name, rows=len(rows), path=str(path), sha256=digest))
            key = path.name + "|" + ",".join(fields)
            item = schemas.setdefault(key, {"filename": path.name, "columns": fields, "file_count": 0,
                                             "nonempty_file_count": 0, "total_rows": 0})
            item["file_count"] += 1
            item["nonempty_file_count"] += bool(rows)
            item["total_rows"] += len(rows)
        fingerprints[str(run.metadata_path)] = sha256(run.metadata_path)
    for name, rows in (("run_inventory.csv", inventory), ("training_summary.csv", summaries), ("csv_inventory.csv", csv_inventory)):
        if rows:
            write_csv(output/name, rows, tuple(rows[0]))
    (output/"csv_schemas.json").write_text(json.dumps(list(schemas.values()), indent=2)+"\n")
    return inventory, fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO.parent/"result/probe")
    parser.add_argument("--output", type=Path, default=REPO.parent/"reports"/f"training_curves_{datetime.now():%Y%m%d}")
    parser.add_argument("--dataset", default="imagenet100", choices=("imagenet100",))
    parser.add_argument("--model", default="resnet50", choices=("resnet50", "resnet101"))
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[.01, .05])
    parser.add_argument("--groups", choices=tuple(GROUPS), nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--warmup-comparison-lr", type=float, default=.01,
                        help="Fixed LR for warmup_percent (must also be included in --lrs)")
    parser.add_argument("--warmup-percents", type=float, nargs="+", default=[10., 30.],
                        help="SGD-prefix percentages compared by warmup_percent; not LR warmup")
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--max-step", type=int, help="Display an early prefix; do not change the required run budget")
    parser.add_argument("--x-axis", choices=("step", "samples_seen", "elapsed_wall_s", "cumulative_eta_before"), default="step")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--panel-height-ratio", type=float, default=.8,
                        help="Physical panel height divided by width; default .8 means width:height=1:.8")
    parser.add_argument("--line-width", type=float, default=1.05)
    parser.add_argument("--dash-lighten", type=float, default=.20,
                        help="Mix dashed method colors with this fraction of white; default .20")
    parser.add_argument("--run-ids", nargs="+", help="Explicitly choose run IDs when a cohort has repeated runs")
    args = parser.parse_args(argv)
    if (args.smooth_window < 1 or args.steps < 1 or args.batch_size < 1 or args.dpi < 1
            or args.max_step is not None and args.max_step < 1):
        parser.error("Window, steps, batch size, dpi and optional max-step must be positive")
    if not 1 <= len(set(args.lrs)) <= 4 or len(set(args.lrs)) != len(args.lrs) or any(not math.isfinite(r) or r <= 0 for r in args.lrs):
        parser.error("Provide 1..4 distinct positive finite initial learning rates")
    if (not 1 <= len(set(args.warmup_percents)) <= 4
            or len(set(args.warmup_percents)) != len(args.warmup_percents)
            or any(not math.isfinite(p) or not 0 < p < 100 for p in args.warmup_percents)):
        parser.error("Provide 1..4 distinct SGD-prefix percentages strictly between 0 and 100")
    if "warmup_percent" in args.groups and (not math.isfinite(args.warmup_comparison_lr)
            or not any(math.isclose(args.warmup_comparison_lr, r, rel_tol=1e-12) for r in args.lrs)):
        parser.error("--warmup-comparison-lr must be included in --lrs")
    if (not math.isfinite(args.panel_height_ratio) or args.panel_height_ratio <= 0
            or not math.isfinite(args.line_width) or args.line_width <= 0
            or not math.isfinite(args.dash_lighten) or not 0 <= args.dash_lighten <= 1):
        parser.error("Panel ratio and line width must be positive finite; dash-lighten must be in [0,1]")
    root, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    if root == output or root in output.parents:
        parser.error("Output must be outside the raw input tree")
    runs, exclusions = discover_runs(root, args)
    if not runs:
        parser.error("No completed runs matched the requested cohort")
    plans = [(group, *select_group(runs, group, args.lrs,
              warmup_comparison_lr=args.warmup_comparison_lr,
              warmup_percents=args.warmup_percents)) for group in args.groups]
    # Resolve duplicate/missing cohorts before creating figures. No latest/best
    # selection, synthetic zero curves, seed pooling or hidden run replacement.
    for group, selected, missing in plans:
        if missing:
            print(f"WARNING {group}: missing {missing}", file=sys.stderr)
        if not selected:
            parser.error(f"No runs matched group {group}")
    output.mkdir(parents=True, exist_ok=True)
    inventory, fingerprints = export_inventory(runs, output)
    limits = shared_limits(plans, args)
    plots = [plot_group(group, selected, missing, args, output, limits) for group, selected, missing in plans]
    changed = [name for name, digest in fingerprints.items() if sha256(name) != digest]
    if changed:
        raise RuntimeError(f"Source files changed while plotting: {changed}")
    manifest = {"input": str(root), "output": str(output), "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "plot_script_sha256": sha256(Path(__file__)), "python_version": sys.version,
        "created_at": datetime.now().astimezone().isoformat(), "run_count": len(runs), "plots": plots,
        "exclusions": exclusions, "run_inventory": inventory, "input_sha256": fingerprints,
        "smoothing": "trailing equal-step arithmetic mean; growing prefix; no downsampling or centered smoothing",
        "accuracy": "original train_top1 fractions multiplied by 100 for display only",
        "figure_style": {"panel_width_to_height": [1, args.panel_height_ratio],
            "line_width": args.line_width, "dash_pattern": [8, 5], "dash_lighten": args.dash_lighten,
            "text": "axis labels, ticks and legend only; explanatory context is retained in this manifest and gallery"},
        "step_semantics": "train row k reports current-batch metrics at theta_(k-1), before update k",
        "inference_scope": "descriptive one-seed training trajectories; not paired T2 tests or a T3 convergence proof",
        "same_seed_scope": "shared seed/protocol is not a verified bitwise initialization/input/augmentation identity"}
    (output/"plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    cards = []
    for plot in plots:
        png = next((name for name in plot["files"] if name.endswith(".png")), None)
        links = " ".join(f'<a href="{html.escape(name)}">{Path(name).suffix[1:].upper()}</a>' for name in plot["files"])
        image = f'<img src="{html.escape(png)}" alt="{html.escape(plot["title"])}">' if png else ""
        notes = " ".join(plot["comparison_notes"])
        cards.append(f'<section><h2>{html.escape(plot["title"])}</h2>{image}<p>{links}</p><p>{html.escape(notes)}</p></section>')
    page = '<!doctype html><meta charset="utf-8"><title>DPCO training curves</title><style>body{font:16px system-ui;margin:32px auto;max-width:1320px;padding:0 24px;color:#172033}img{width:100%;height:auto}section{margin:36px 0}a{margin-right:16px}p{line-height:1.6}</style>'
    page += f'<h1>DPCO training curves</h1><p>{len(runs)} completed runs. Seed {args.seed}, batch {args.batch_size}, {args.steps} updates. '
    page += f'Trailing mean: {args.smooth_window} steps, growing prefix. Blue: SGD; red: DPCO-E; orange: DPCO-L. '
    page += 'Initial LR 0.01 uses pale sparse dashes; 0.05 uses solid method colors with the default two-rate selection. '
    page += 'These are unsigned online training metrics. Different LR cohorts can have different code revisions; see the manifest. No convergence theorem is certified by these curves.</p>'
    page += '<p><a href="run_inventory.csv">Run inventory</a><a href="training_summary.csv">Training summaries</a><a href="csv_inventory.csv">CSV inventory</a><a href="plot_manifest.json">Manifest</a></p>' + "".join(cards)
    (output/"index.html").write_text(page, encoding="utf-8")
    print(f"Completed: {len(plots)} figures, {len(runs)} inspected runs. Output: {output}")
    print("Source CSV/JSON hashes rechecked unchanged. No training modules were imported.")
    return manifest


if __name__ == "__main__":
    main()
