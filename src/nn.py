from collections import OrderedDict
import numpy as np
from .tensor import (
    Tensor,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    cross_entropy,
    huber_loss,
    l1_loss,
    mse_loss,
)
from .functional import (
    avg_pool1d,
    avg_pool2d,
    conv1d,
    conv2d,
    max_pool1d,
    max_pool2d,
    scaled_dot_product_attention,
)


class Parameter(Tensor):
    def __init__(self, data, *, dtype=None):
        super().__init__(data, dtype=dtype, requires_grad=True)


class Module:
    def __init__(self):
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_buffers", {})

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _named(self, prefix=""):
        for name, value in self.__dict__.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                yield key, value
            elif isinstance(value, Module):
                yield from value._named(key)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Parameter):
                        yield f"{key}.{i}", item
                    elif isinstance(item, Module):
                        yield from item._named(f"{key}.{i}")

    def named_parameters(self):
        seen = set()
        for name, p in self._named():
            if id(p) not in seen:
                seen.add(id(p))
                yield name, p

    def parameters(self):
        return (p for _, p in self.named_parameters())

    def register_buffer(self, name, value):
        self._buffers[name] = value
        setattr(self, name, value)
        return value

    def named_buffers(self, prefix=""):
        for name, value in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name), value
        for name, value in self.__dict__.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Module):
                yield from value.named_buffers(key)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_buffers(f"{key}.{i}")

    def modules(self):
        yield self
        for value in self.__dict__.values():
            if isinstance(value, Module):
                yield from value.modules()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.modules()

    def train(self, mode=True):
        for module in self.modules():
            module.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self):
        return OrderedDict(
            [(name, p.data.copy()) for name, p in self.named_parameters()]
            + [(name, value.copy()) for name, value in self.named_buffers()]
        )

    def load_state_dict(self, state):
        expected = dict(self.named_parameters())
        expected.update(self.named_buffers())
        if set(state) != set(expected):
            raise ValueError(
                f"state keys differ: missing={set(expected)-set(state)}, unexpected={set(state)-set(expected)}"
            )
        for name, p in expected.items():
            shape = p.shape if isinstance(p, Tensor) else p.shape
            if shape != np.shape(state[name]):
                raise ValueError(
                    f"shape mismatch for {name}: {shape} != {np.shape(state[name])}"
                )
            (p.data if isinstance(p, Tensor) else p)[...] = state[name]
        return self


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        bound = 1 / np.sqrt(in_features)
        self.weight = Parameter(rng.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        self.layers = list(modules)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList only accepts Module instances")
        self.layers.append(module)
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self


class ParameterList(Module):
    def __init__(self, parameters=()):
        super().__init__()
        self.parameters_list = list(parameters)

    def __len__(self):
        return len(self.parameters_list)

    def __iter__(self):
        return iter(self.parameters_list)

    def __getitem__(self, index):
        return self.parameters_list[index]

    def append(self, parameter):
        if not isinstance(parameter, Parameter):
            raise TypeError("ParameterList only accepts Parameter instances")
        self.parameters_list.append(parameter)
        return self


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Identity(Module):
    def forward(self, x):
        return x


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return x.leaky_relu(self.negative_slope)


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.softmax(self.axis)


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, bias=True, seed=None):
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        seeds = np.random.default_rng(seed).integers(2**32, size=4)
        self.q_proj = Linear(embed_dim, embed_dim, bias, int(seeds[0]))
        self.k_proj = Linear(embed_dim, embed_dim, bias, int(seeds[1]))
        self.v_proj = Linear(embed_dim, embed_dim, bias, int(seeds[2]))
        self.out_proj = Linear(embed_dim, embed_dim, bias, int(seeds[3]))

    def forward(self, query, key=None, value=None, mask=None):
        key = query if key is None else key
        value = key if value is None else value
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("MultiheadAttention expects tensors shaped (N, L, E)")
        head_dim = self.embed_dim // self.num_heads

        def heads(x, projection):
            x = projection(x).reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
            return x.permute(0, 2, 1, 3)

        attended = scaled_dot_product_attention(
            heads(query, self.q_proj),
            heads(key, self.k_proj),
            heads(value, self.v_proj),
            mask,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(query.shape)
        return self.out_proj(attended)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, seed=None):
        super().__init__()
        self.weight = Parameter(
            np.random.default_rng(seed).normal(
                0, 1 / np.sqrt(embedding_dim), (num_embeddings, embedding_dim)
            )
        )

    def forward(self, indices):
        return self.weight[
            np.asarray(
                indices.data if isinstance(indices, Tensor) else indices, dtype=int
            )
        ]


class Dropout(Module):
    def __init__(self, p=0.5, seed=None):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        super().__init__()
        self.p, self.rng = p, np.random.default_rng(seed)

    def forward(self, x):
        return (
            x
            if not self.training or self.p == 0
            else x * Tensor(self.rng.random(x.shape) >= self.p) / (1 - self.p)
        )


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.shape, self.eps = shape, eps
        self.weight = Parameter(np.ones(shape)) if elementwise_affine else None
        self.bias = Parameter(np.zeros(shape)) if elementwise_affine else None

    def forward(self, x):
        axes = tuple(range(-len(self.shape), 0))
        y = (x - x.mean(axes, True)) / (x.var(axes, True) + self.eps).sqrt()
        return y * self.weight + self.bias if self.weight is not None else y


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Parameter(np.ones(num_features)) if affine else None
        self.bias = Parameter(np.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", np.zeros(num_features))
        self.register_buffer("running_var", np.ones(num_features))

    def forward(self, x):
        if x.ndim < 2:
            raise ValueError("BatchNorm expects shape (N, C, ...)")
        axes = (0,) + tuple(range(2, x.ndim))
        shape = (1, -1) + (1,) * (x.ndim - 2)
        if self.training:
            mean, var = x.mean(axes, True), x.var(axes, True)
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * mean.data.reshape(-1)
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * var.data.reshape(-1)
        else:
            mean, var = Tensor(self.running_mean.reshape(shape)), Tensor(
                self.running_var.reshape(shape)
            )
        y = (x - mean) / (var + self.eps).sqrt()
        return (
            y * self.weight.reshape(shape) + self.bias.reshape(shape)
            if self.weight is not None
            else y
        )


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        if num_channels % num_groups:
            raise ValueError("num_channels must be divisible by num_groups")
        super().__init__()
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = Parameter(np.ones(num_channels)) if affine else None
        self.bias = Parameter(np.zeros(num_channels)) if affine else None

    def forward(self, x):
        if x.ndim < 2 or x.shape[1] != self.num_channels:
            raise ValueError("GroupNorm expects shape (N, C, ...)")
        shape = x.shape
        grouped = x.reshape(shape[0], self.num_groups, -1)
        y = (grouped - grouped.mean(-1, True)) / (
            grouped.var(-1, True) + self.eps
        ).sqrt()
        y = y.reshape(shape)
        if self.weight is None:
            return y
        affine_shape = (1, -1) + (1,) * (x.ndim - 2)
        return y * self.weight.reshape(affine_shape) + self.bias.reshape(affine_shape)


class InstanceNorm(GroupNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, num_channels, eps, affine)


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        seed=None,
    ):
        super().__init__()
        kh, kw = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        bound = 1 / np.sqrt(in_channels * kh * kw)
        rng = np.random.default_rng(seed)
        self.stride, self.padding = stride, padding
        self.weight = Parameter(
            rng.uniform(-bound, bound, (out_channels, in_channels, kh, kw))
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


class Conv1d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        seed=None,
    ):
        super().__init__()
        bound = 1 / np.sqrt(in_channels * kernel_size)
        self.stride, self.padding = stride, padding
        self.weight = Parameter(
            np.random.default_rng(seed).uniform(
                -bound, bound, (out_channels, in_channels, kernel_size)
            )
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x):
        return conv1d(x, self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool2d(x, self.kernel_size, self.stride)


class MaxPool1d(MaxPool2d):
    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride)


class AvgPool2d(MaxPool2d):
    def forward(self, x):
        return avg_pool2d(x, self.kernel_size, self.stride)


class AvgPool1d(MaxPool2d):
    def forward(self, x):
        return avg_pool1d(x, self.kernel_size, self.stride)


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return mse_loss(input, target, self.reduction)


class L1Loss(MSELoss):
    def forward(self, input, target):
        return l1_loss(input, target, self.reduction)


class HuberLoss(MSELoss):
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, input, target):
        return huber_loss(input, target, self.delta, self.reduction)


class BCELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return binary_cross_entropy(input, target, self.reduction)


class BCEWithLogitsLoss(BCELoss):
    def forward(self, input, target):
        return binary_cross_entropy_with_logits(input, target, self.reduction)


class CrossEntropyLoss(Module):
    def __init__(self, axis=-1, reduction="mean"):
        super().__init__()
        self.axis, self.reduction = axis, reduction

    def forward(self, input, target):
        return cross_entropy(input, target, self.axis, self.reduction)
