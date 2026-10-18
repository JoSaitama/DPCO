# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ResNet V1.5 adapted from Google's Flax ImageNet example.

Upstream file revision: 81b168d62f151feca9e8e7abc591df80168395b4.
Local changes: bottleneck models only, configurable head/precision, no pmap axis
name (GSPMD supplies global-batch reductions), and pure train/eval adapters.
Weights are randomly initialized; no pretrained artifact is needed/downloaded.
"""

from functools import partial
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp


class Bottleneck(nn.Module):
    filters: int
    strides: tuple = (1, 1)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm, use_running_average=not train,
                       momentum=.9, epsilon=1e-5, dtype=self.dtype)
        residual = x
        y = nn.relu(norm()(conv(self.filters, (1, 1))(x)))
        y = nn.relu(norm()(conv(self.filters, (3, 3), strides=self.strides)(y)))
        y = norm(scale_init=nn.initializers.zeros_init())(conv(4 * self.filters, (1, 1))(y))
        if residual.shape != y.shape:
            residual = norm(name="norm_proj")(
                conv(4 * self.filters, (1, 1), strides=self.strides, name="conv_proj")(residual))
        return nn.relu(residual + y)


class ResNet(nn.Module):
    stage_sizes: tuple
    num_classes: int
    dtype: Any = jnp.float32
    num_filters: int = 64

    @nn.compact
    def __call__(self, images, train=True):
        x = nn.Conv(self.num_filters, (7, 7), strides=(2, 2),
                    padding=((3, 3), (3, 3)), use_bias=False,
                    dtype=self.dtype, name="conv_init")(images)
        x = nn.BatchNorm(use_running_average=not train, momentum=.9,
                         epsilon=1e-5, dtype=self.dtype, name="bn_init")(x)
        x = nn.max_pool(nn.relu(x), (3, 3), strides=(2, 2), padding="SAME")
        for stage, size in enumerate(self.stage_sizes):
            for block in range(size):
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = Bottleneck(self.num_filters * 2 ** stage, strides, self.dtype,
                               name=f"stage_{stage}_block_{block}")(x, train)
        x = jnp.mean(x, axis=(1, 2))
        return nn.Dense(self.num_classes, dtype=self.dtype, name="head")(x)


def create_resnet(name, num_classes, precision="float32"):
    stages = {"resnet50": (3, 4, 6, 3), "resnet101": (3, 4, 23, 3)}
    if name not in stages:
        raise ValueError(f"Unsupported ResNet: {name}")
    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[precision]
    return ResNet(stages[name], num_classes, dtype)


def model_adapter(model, *, train):
    def apply(params, images, model_state):
        variables = {"params": params, "batch_stats": model_state["batch_stats"]}
        if train:
            logits, changed = model.apply(variables, images, train=True, mutable=["batch_stats"])
            return logits, {"batch_stats": changed["batch_stats"]}
        return model.apply(variables, images, train=False), model_state
    return apply


def initialize(model, seed, image_size):
    variables = jax.jit(lambda key: model.init(key,
        jnp.zeros((1, image_size, image_size, 3), jnp.float32), train=True))(jax.random.key(seed))
    return variables["params"], {"batch_stats": variables["batch_stats"]}
