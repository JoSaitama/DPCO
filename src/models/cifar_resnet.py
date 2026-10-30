"""Basic-block ResNets with a native 32x32 stem; ImageNet models are unchanged."""

from flax import linen as nn
import jax.numpy as jnp


class BasicBlock(nn.Module):
    filters: int
    stride: int = 1
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        residual = x
        y = nn.Conv(self.filters, (3, 3), strides=(self.stride, self.stride), padding="SAME",
                    use_bias=False, dtype=self.dtype, name="conv1")(x)
        y = nn.BatchNorm(use_running_average=not train, momentum=.9, epsilon=1e-5,
                         dtype=self.dtype, name="bn1")(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False,
                    dtype=self.dtype, name="conv2")(y)
        y = nn.BatchNorm(use_running_average=not train, momentum=.9, epsilon=1e-5,
                         dtype=self.dtype, name="bn2")(y)
        if self.stride != 1 or residual.shape[-1] != self.filters:
            residual = nn.Conv(self.filters, (1, 1), strides=(self.stride, self.stride),
                use_bias=False, dtype=self.dtype, name="projection")(residual)
            residual = nn.BatchNorm(use_running_average=not train, momentum=.9, epsilon=1e-5,
                dtype=self.dtype, name="projection_bn")(residual)
        return nn.relu(y + residual)


class CifarResNet(nn.Module):
    stage_sizes: tuple = (3, 4, 6, 3)
    num_classes: int = 100
    dtype: object = jnp.float32
    num_filters: int = 64
    bottleneck: bool = False

    @nn.compact
    def __call__(self, images, train=True):
        x = nn.Conv(self.num_filters, (3, 3), padding="SAME", use_bias=False,
                    dtype=self.dtype, name="conv_init")(images)
        x = nn.BatchNorm(use_running_average=not train, momentum=.9, epsilon=1e-5,
                         dtype=self.dtype, name="bn_init")(x)
        x = nn.relu(x)
        for stage, size in enumerate(self.stage_sizes):
            for block in range(size):
                stride = 2 if stage > 0 and block == 0 else 1
                if self.bottleneck:
                    # Reuse the existing ResNet50 residual block; only the stem differs.
                    from src.models.resnet import Bottleneck
                    x = Bottleneck(self.num_filters * 2**stage, (stride, stride), self.dtype,
                                   name=f"stage_{stage}_block_{block}")(x, train)
                else:
                    x = BasicBlock(self.num_filters * 2**stage, stride, self.dtype,
                                   name=f"stage_{stage}_block_{block}")(x, train)
        return nn.Dense(self.num_classes, dtype=self.dtype, name="head")(jnp.mean(x, axis=(1, 2)))


def create_cifar_resnet(name, num_classes=100, precision="float32"):
    stages = {"cifar_resnet18": (2, 2, 2, 2), "cifar_resnet34": (3, 4, 6, 3),
              "cifar_resnet50": (3, 4, 6, 3)}
    if name not in stages:
        raise ValueError(f"Unknown CIFAR model: {name}")
    return CifarResNet(stages[name], num_classes, {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[precision],
                       bottleneck=name == "cifar_resnet50")
