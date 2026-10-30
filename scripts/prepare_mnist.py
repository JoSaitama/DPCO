"""Plan/download/verify MNIST source IDX files and an immutable local cache."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "download", "verify"), default="plan")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "data/mnist")
    args = parser.parse_args()
    from src.data.mnist_prepare import ASSETS, SOURCE, prepare, verify_prepared
    root = args.output_dir.expanduser().resolve()
    if args.stage == "plan":
        print(json.dumps({"output_dir": str(root), "digest_source": SOURCE,
            "sources": {name: {"url": item[0], item[1]: item[2]} for name, item in ASSETS.items()},
            "split_counts": {"train": 60000, "test": 10000},
            "labels": "clean cache; fixed synthetic corruption is configured at loading"}, indent=2))
        return
    result = (prepare(root, download=True, progress=lambda **fields: print(json.dumps(fields), flush=True))
              if args.stage == "download" else verify_prepared(root))
    print(json.dumps({"event": "verified", "output_dir": str(root), "manifest_sha256": result[1],
                      **result[0]}, indent=2))


if __name__ == "__main__":
    main()
