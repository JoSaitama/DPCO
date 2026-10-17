"""Reference factories are used at composition time, not by the trainer."""


def create_reference(name):
    if name == "loss":
        from src.references.batch_mean import BatchMeanReference
        return BatchMeanReference()
    if name == "error":
        from src.references.error_subset import ErrorSubsetReference
        return ErrorSubsetReference()
    raise ValueError(f"Unknown or unimplemented reference: {name}")
