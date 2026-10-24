"""Configuration inspection only; no implicit training or data collection."""

import argparse
from dataclasses import replace
import json

from src.config import METHODS, REFERENCES, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true", required=True,
                        help="Validate/print configuration without starting training")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--reference", choices=REFERENCES)
    parser.add_argument("--flip-probability", type=float)
    parser.add_argument("--policy-mode", choices=("continuous", "alternating", "half"))
    parser.add_argument("--alternating-first", choices=("sgd", "sage"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.method is not None:
        config = replace(config, method=replace(config.method, name=args.method))
    if args.reference is not None:
        config = replace(config, reference=replace(config.reference, name=args.reference))
    if args.policy_mode is not None:
        config = replace(config, method=replace(config.method, policy_mode=args.policy_mode))
    if args.alternating_first is not None:
        config = replace(config, method=replace(config.method, alternating_first=args.alternating_first))
    if args.flip_probability is not None:
        config = replace(config, method=replace(config.method,
                                              flip_probability=args.flip_probability))
    config.validate()
    print(json.dumps(config.as_dict(), indent=2))


if __name__ == "__main__":
    main()
