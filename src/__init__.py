"""FudgeGrad: a small CPU-only reverse-mode autodiff library built on NumPy."""

from .tensor import (
    Tensor,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    cat,
    cross_entropy,
    gradcheck,
    mse_loss,
    stack,
    where,
)
from .nn import (
    AvgPool2d,
    BatchNorm,
    Conv2d,
    Dropout,
    Embedding,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    Module,
    Parameter,
    Sequential,
)
from .functional import avg_pool2d, conv2d, max_pool2d
from .optim import Adam, RMSprop, SGD

__all__ = [
    "Tensor",
    "where",
    "stack",
    "cat",
    "mse_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "gradcheck",
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "Module",
    "Parameter",
    "Linear",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "Sequential",
    "Flatten",
    "Embedding",
    "Dropout",
    "LayerNorm",
    "BatchNorm",
    "SGD",
    "Adam",
    "RMSprop",
]
