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
    parser.add_argument("--alternating-first", choices=("sgd", "dpco"))
    parser.add_argument("--dpco-warmup-percent", type=float)
    parser.add_argument("--negative-weight", type=float)
    parser.add_argument("--reference-beta", type=float)
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--adam-epsilon", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.method is not None:
        config = replace(config, method=replace(config.method, name=args.method))
        if args.method in ("adam", "sgd"):
            config = replace(config, method=replace(config.method, negative_weight=-1.),
                reference=replace(config.reference, name="loss"),
                training=replace(config.training, optimizer=args.method,
                                 momentum=0. if args.method == "adam" else config.training.momentum,
                                 nesterov=False if args.method == "adam" else config.training.nesterov))
    if args.reference is not None:
        config = replace(config, reference=replace(config.reference, name=args.reference))
    if args.policy_mode is not None:
        config = replace(config, method=replace(config.method, policy_mode=args.policy_mode))
    if args.alternating_first is not None:
        config = replace(config, method=replace(config.method, alternating_first=args.alternating_first))
    if args.dpco_warmup_percent is not None:
        config = replace(config, method=replace(config.method, dpco_warmup_percent=args.dpco_warmup_percent))
    if args.flip_probability is not None:
        config = replace(config, method=replace(config.method,
                                              flip_probability=args.flip_probability))
    if args.negative_weight is not None:
        config = replace(config, method=replace(config.method, negative_weight=args.negative_weight))
    if args.reference_beta is not None:
        config = replace(config, reference=replace(config.reference, beta=args.reference_beta))
    training = {name: getattr(args, name) for name in ("optimizer", "adam_beta1", "adam_beta2", "adam_epsilon")
                if getattr(args, name) is not None}
    config = replace(config, training=replace(config.training, **training))
    config.validate()
    print(json.dumps(config.as_dict(), indent=2))


if __name__ == "__main__":
    main()
