"""Construct methods once; the trainer does not branch on method names."""


def create_method(name, *, reference_name="loss", flip_probability=0.25):
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
            return SAGE(reference, exclude_self)
        from src.methods.dtv import DTV
        return DTV(reference, exclude_self)
    raise ValueError(f"Unknown method: {name}")
