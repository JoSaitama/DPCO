"""Prepare the official CNWL Red splits without synthesizing label noise."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../data/cnwl")
    parser.add_argument("--manual-dir", help="Official zip/extracted metadata, ImageNet archives and noisy_images directory")
    parser.add_argument("--cnwl-noise-percent", type=int, choices=(0, 5, 10, 15, 20, 30, 40, 50, 60, 80), default=20)
    parser.add_argument("--stage", choices=("plan", "prepare", "verify"), default="plan")
    parser.add_argument("--download", action="store_true", help="Allow official metadata and selected web URI downloads; never downloads gated ImageNet archives")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-dir", help="Defaults to results/files/cnwl_prepare_<timestamp>")
    args = parser.parse_args()
    from src.data.cnwl_prepare import HOMEPAGE, METADATA_URL, RED_COUNTS, prepare, verify_prepared
    from src.data.cifar100n_prepare import atomic_json
    from src.training.launch import absolute_path
    directory = absolute_path(args.data_dir, ROOT)
    manual = absolute_path(args.manual_dir, ROOT) if args.manual_dir else directory / "raw"
    if args.stage == "plan":
        print(json.dumps({"dataset": "cnwl", "noise_type": "red", "homepage": HOMEPAGE,
            "metadata_url": METADATA_URL, "data_dir": str(directory), "manual_dir": str(manual),
            "selected_split": f"red_noise_nl_{args.cnwl_noise_percent / 100}",
            "train_count": RED_COUNTS[args.cnwl_noise_percent], "validation_count": 5000,
            "image_size": 84, "natural_web_noise": True, "download_enabled": args.download,
            "missing_sources_policy": "fail_without_dropping_or_relabeling_samples"}, indent=2))
        return
    report_dir = absolute_path(args.report_dir, ROOT) if args.report_dir else ROOT / "results/files" / (
        "cnwl_prepare_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
    # Diagnostic artifacts follow the same results/files contract as training.
    if not report_dir.is_relative_to(ROOT / "results/files"):
        parser.error("CNWL reports must be below DPCO/results/files/")
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "verify":
            manifest, digest = verify_prepared(directory, args.cnwl_noise_percent)
        else:
            manifest, digest = prepare(directory, percent=args.cnwl_noise_percent, manual_dir=manual,
                download=args.download, workers=args.workers, report_path=report_dir / "report.json",
                progress=lambda **fields: print(json.dumps(fields), flush=True))
        atomic_json(report_dir / "verified.json", {"manifest_sha256": digest, **manifest})
        print(json.dumps({"event": "cnwl_ready", "manifest_sha256": digest,
            "counts": manifest["counts"], "actual_noise_percent": manifest["actual_noise_percent"],
            "report_dir": str(report_dir)}), flush=True)
    except BaseException as error:
        atomic_json(report_dir / "error.json", {"status": "failed", "error": str(error),
            "metadata_url": METADATA_URL, "no_samples_dropped": True})
        raise


if __name__ == "__main__":
    main()
