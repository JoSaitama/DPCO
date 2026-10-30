"""Explicit model dispatch: names, rather than datasets, determine architecture."""

from src.config import CIFAR_MODELS, IMAGENET_MODELS, MNIST_MODELS


def create_model(name, num_classes, precision="float32"):
    if name in IMAGENET_MODELS:
        from src.models.resnet import create_resnet
        return create_resnet(name, num_classes, precision)
    if name in CIFAR_MODELS:
        from src.models.cifar_resnet import create_cifar_resnet
        return create_cifar_resnet(name, num_classes, precision)
    if name in MNIST_MODELS:
        import jax.numpy as jnp
        from src.models.mnist import MNISTCNN, MNISTMLP
        return {"mnist_cnn": MNISTCNN, "mnist_mlp": MNISTMLP}[name](
            num_classes, {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[precision])
    raise ValueError(f"Unsupported model: {name}")


def model_revision(name):
    if name in IMAGENET_MODELS:
        return "81b168d62f151feca9e8e7abc591df80168395b4"
    if name in MNIST_MODELS:
        return "dpco_mnist_no_bn_v1"
    return "dpco_cifar_bottleneck_v1" if name == "cifar_resnet50" else "dpco_cifar_basicblock_v1"
