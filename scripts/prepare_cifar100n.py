"""Download/verify official CIFAR-100N sources and publish a local immutable cache."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "download", "verify"), default="plan")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "data/cifar100n")
    args = parser.parse_args(argv)
    from src.data.cifar100n_prepare import ASSETS, REVISION, prepare, verify_prepared
    output = args.output_dir.expanduser().resolve()
    if args.stage == "plan":
        print(json.dumps({"output_dir": str(output), "official_revision": REVISION,
            "sources": {name: {"url": item[0], item[1]: item[2]} for name, item in ASSETS.items()},
            "split_counts": {"train": 50000, "test": 10000},
            "noise_percent_semantics": "measured label mismatch; not a human noise generator",
            "runtime_dependencies": "numpy; certifi optional; no JAX/Torch/TensorFlow"}, indent=2))
        return
    manifest, digest = (prepare(output, download=True,
        progress=lambda **fields: print(json.dumps(fields), flush=True)) if args.stage == "download"
        else verify_prepared(output))
    print(json.dumps({"event": "verified", "output_dir": str(output),
                      "manifest_sha256": digest, **manifest}, indent=2))


if __name__ == "__main__":
    main()
