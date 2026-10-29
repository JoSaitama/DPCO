"""Construct methods once; the trainer does not branch on method names."""


def create_method(name, *, reference_name="loss", flip_probability=0.25, gradient_backend="audit",
                  policy_mode="continuous", alternating_first="sgd", warmup_steps=0,
                  reference_beta=0.9, negative_weight=-1.0):
    if reference_name == "ema":
        if name != "dpco" or warmup_steps or policy_mode != "continuous":
            raise ValueError("EMA reference requires continuous dpco without method warmup")
        from src.methods.dpco_history import HistoricalDPCO
        return HistoricalDPCO(reference_beta, negative_weight, gradient_backend)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if warmup_steps:
        if name != "dpco" and not (name == "dpco_policy" and policy_mode == "continuous"):
            raise ValueError("Warmup requires original or continuous DPCO")
        from src.methods.dpco_warmup import DPCOWarmup
        from src.references import create_reference
        return DPCOWarmup(create_reference(reference_name), warmup_steps, gradient_backend)
    if name == "dpco_policy":
        from src.methods.dpco_policy import DPCOPolicy
        from src.references import create_reference
        return DPCOPolicy(create_reference(reference_name), policy_mode, gradient_backend, alternating_first)
    if name in ("sgd", "adam"):
        from src.methods.sgd import SGD
        return SGD()
    if name == "random_flip":
        from src.methods.random_flip import RandomFlip
        return RandomFlip(flip_probability)
    if name in ("dpco", "dpco_loo", "dtv", "dtv_loo"):
        from src.references import create_reference
        reference = create_reference(reference_name)
        exclude_self = name.endswith("_loo")
        if name.startswith("dpco"):
            from src.methods.dpco import DPCO
            return DPCO(reference, exclude_self, gradient_backend=gradient_backend)
        from src.methods.dtv import DTV
        return DTV(reference, exclude_self, gradient_backend=gradient_backend)
    raise ValueError(f"Unknown method: {name}")
