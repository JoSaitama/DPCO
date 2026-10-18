"""Foreground CPU sanity checks only; TPU smoke uses launch_tpu.py."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tpu/imagenet100_resnet50_sgd.toml")
    args = parser.parse_args()
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu"}
    for script in ("sanity_core.py", "sanity_recording.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", args.config],
                       cwd=ROOT, env=environment, check=True)
    print("PASS: CPU sanity only. Next launch a real TPU smoke through scripts/launch_tpu.py.")


if __name__ == "__main__":
    main()
