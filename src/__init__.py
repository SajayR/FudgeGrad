"""FudgeGrad: a small CPU-only reverse-mode autodiff library built on NumPy."""
from .tensor import Tensor, cat, cross_entropy, gradcheck, mse_loss, stack, where
from .nn import BatchNorm, Dropout, Embedding, Flatten, LayerNorm, Linear, Module, Parameter, Sequential
from .optim import Adam, RMSprop, SGD

__all__ = ["Tensor", "where", "stack", "cat", "mse_loss", "cross_entropy", "gradcheck", "Module", "Parameter", "Linear", "Sequential", "Flatten", "Embedding", "Dropout", "LayerNorm", "BatchNorm", "SGD", "Adam", "RMSprop"]
