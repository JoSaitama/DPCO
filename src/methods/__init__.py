"""Construct methods once; the trainer does not branch on method names."""


def create_method(name, *, reference_name="loss", flip_probability=0.25, gradient_backend="audit",
                  policy_mode="continuous", alternating_first="sgd", warmup_steps=0):
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if warmup_steps:
        if name != "sage" and not (name == "sage_policy" and policy_mode == "continuous"):
            raise ValueError("Warmup requires original or continuous SAGE")
        from src.methods.sage_warmup import SAGEWarmup
        from src.references import create_reference
        return SAGEWarmup(create_reference(reference_name), warmup_steps, gradient_backend)
    if name == "sage_policy":
        from src.methods.sage_policy import SAGEPolicy
        from src.references import create_reference
        return SAGEPolicy(create_reference(reference_name), policy_mode, gradient_backend, alternating_first)
    if name == "sgd":
        from src.methods.sgd import SGD
        return SGD()
    if name == "random_flip":
        from src.methods.random_flip import RandomFlip
        return RandomFlip(flip_probability)
    if name in ("sage", "sage_loo", "dtv", "dtv_loo"):
        from src.references import create_reference
        reference = create_reference(reference_name)
        exclude_self = name.endswith("_loo")
        if name.startswith("sage"):
            from src.methods.sage import SAGE
            return SAGE(reference, exclude_self, gradient_backend=gradient_backend)
        from src.methods.dtv import DTV
        return DTV(reference, exclude_self, gradient_backend=gradient_backend)
    raise ValueError(f"Unknown method: {name}")
