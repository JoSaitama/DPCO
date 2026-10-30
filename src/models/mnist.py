"""Small no-BatchNorm models for MNIST mechanism experiments (not a claimed replication)."""

from flax import linen as nn
import jax.numpy as jnp


class MNISTCNN(nn.Module):
    num_classes: int = 10
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        for filters in (16, 32):
            x = nn.relu(nn.Conv(filters, (3, 3), padding="SAME", dtype=self.dtype)(x))
            x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(64, dtype=self.dtype)(x))
        return nn.Dense(self.num_classes, dtype=self.dtype)(x)


class MNISTMLP(nn.Module):
    num_classes: int = 10
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        x = x.reshape((x.shape[0], -1))
        for width in (128, 64):
            x = nn.relu(nn.Dense(width, dtype=self.dtype)(x))
        return nn.Dense(self.num_classes, dtype=self.dtype)(x)
