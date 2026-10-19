"""Plan and prepare a fixed proportional iNaturalist2021 FULL subset on local disk."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.inaturalist_subset import (StorageBudget, build_manifest, preparation_lock,
                                        prepare_images, verify_images)


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "download", "verify", "all"), default="plan")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Data selection only; independent of the training seed")
    parser.add_argument("--train-samples", type=positive_integer, default=140000)
    parser.add_argument("--validation-samples", type=positive_integer, default=20000)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ../data/inaturalist2021_full_t140000_v20000_s42 (relative to repo)")
    parser.add_argument("--max-storage-gb", type=positive_integer, default=15,
                        help="Preparation directory ceiling, decimal GB; failure never changes selection")
    parser.add_argument("--reserve-gb", type=int, default=10,
                        help="Keep this much filesystem space free, decimal GB")
    args = parser.parse_args(argv)
    if args.selection_seed < 0 or args.reserve_gb < 0:
        parser.error("Selection seed and reserve must be nonnegative")
    output = args.output_dir or ROOT.parent / "data" / (
        f"inaturalist2021_full_t{args.train_samples}_v{args.validation_samples}_s{args.selection_seed}")
    output = output.expanduser().resolve()
    budget = StorageBudget(output, args.max_storage_gb * 10**9, args.reserve_gb * 10**9)

    def progress(**fields):
        print(json.dumps(fields, sort_keys=True), flush=True)

    with preparation_lock(output):
        budget.check()
        if args.stage in ("plan", "all"):
            _, digest = build_manifest(output, seed=args.selection_seed, train_count=args.train_samples,
                validation_count=args.validation_samples, budget=budget, progress=progress)
            progress(event="manifest_ready", manifest_sha256=digest, output_dir=str(output))
        if args.stage in ("download", "all"):
            prepare_images(output, budget=budget, progress=progress)
        if args.stage == "verify":
            progress(event="verified", **verify_images(output))


if __name__ == "__main__":
    main()
