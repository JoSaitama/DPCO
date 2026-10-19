"""Data utilities; batch re-exports load JAX only when explicitly requested."""

__all__ = ["ClassificationBatch", "classification_batch"]


def __getattr__(name):
    if name in __all__:
        from src.data import batch
        return getattr(batch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
