from contextlib import contextmanager
from typing import Iterable, Optional, Sequence

import numpy as np

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


@contextmanager
def no_grad():
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


@contextmanager
def enable_grad():
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = previous


def _to_array(data, dtype=None) -> np.ndarray:
    return np.asarray(data, dtype=dtype)


def _grad_dtype(data) -> np.dtype:
    dtype = np.asarray(data).dtype
    return dtype if dtype.kind in "fc" else np.dtype(float)


def _sum_to_shape(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    grad = np.asarray(grad)
    shape = tuple(shape)
    if grad.shape == shape:
        return grad
    if shape == ():
        return np.array(grad).sum()
    pad = grad.ndim - len(shape)
    if pad < 0:
        raise ValueError(f"Cannot reduce gradient from {grad.shape} to {shape}")
    expand_shape = (1,) * pad + shape
    for axis, (g_dim, s_dim) in enumerate(zip(grad.shape, expand_shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _canonicalize_axes(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a if a >= 0 else ndim + a for a in axis))


def _slice(axis, start, stop):
    return (slice(None),) * axis + (slice(start, stop),)


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_backward", "_prev", "_op")
    __array_priority__ = 100

    def __init__(
        self,
        data,
        *,
        dtype=None,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        if isinstance(data, Tensor):
            data = data.data
        if _children and not _grad_enabled:
            requires_grad, _children = False, ()
        self.data = _to_array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = (
            np.zeros_like(self.data, dtype=_grad_dtype(self.data))
            if self.requires_grad
            else None
        )
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def item(self):
        return self.data.item()

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self, *, requires_grad: Optional[bool] = None) -> "Tensor":
        req = self.requires_grad if requires_grad is None else requires_grad
        return Tensor(self.data.copy(), requires_grad=req)

    def astype(self, dtype):
        out = Tensor(
            self.data.astype(dtype),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="astype",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad.astype(_grad_dtype(self.data))

        out._backward = _backward
        return out

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        self.requires_grad = bool(flag)
        self.grad = (
            np.zeros_like(self.data, dtype=_grad_dtype(self.data))
            if self.requires_grad
            else None
        )
        return self

    def zero_grad(self) -> None:
        if self.requires_grad and self.grad is not None:
            self.grad.fill(0)

    def backward(self, grad=None) -> None:
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call backward on a tensor that does not require gradients"
            )
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data, dtype=_grad_dtype(self.data))
        else:
            grad = _to_array(grad, dtype=_grad_dtype(self.data))
        if grad.shape != self.shape:
            raise ValueError(
                f"gradient shape {grad.shape} does not match output shape {self.shape}"
            )
        topo, visited = [], set()

        def build(v: "Tensor"):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in topo:
            if node._prev and node.requires_grad:
                node.grad = np.zeros_like(node.data, dtype=_grad_dtype(node.data))
        if self.requires_grad:
            self.grad = (
                self.grad
                if self.grad is not None
                else np.zeros_like(self.data, dtype=_grad_dtype(self.data))
            )
            self.grad += grad
        for node in reversed(topo):
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data, dtype=_grad_dtype(node.data))
            node._backward()

    def _binary_op(self, other, op, grad_self, grad_other, name):
        other_tensor = other if isinstance(other, Tensor) else None
        other_data = other_tensor.data if other_tensor is not None else other
        data = op(self.data, other_data)
        requires_grad = self.requires_grad or (
            other_tensor is not None and other_tensor.requires_grad
        )
        parents = tuple(p for p in (self, other_tensor) if isinstance(p, Tensor))
        out = Tensor(data, requires_grad=requires_grad, _children=parents, _op=name)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _sum_to_shape(
                    grad_self(out.grad, self.data, other_data), self.shape
                )
            if other_tensor is not None and other_tensor.requires_grad:
                other_tensor.grad += _sum_to_shape(
                    grad_other(out.grad, self.data, other_tensor.data),
                    other_tensor.shape,
                )

        out._backward = _backward
        return out

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        return self._binary_op(other, np.add, lambda g, *_: g, lambda g, *_: g, "add")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary_op(
            other, np.subtract, lambda g, *_: g, lambda g, *_: -g, "sub"
        )

    def __rsub__(self, other):
        return (other if isinstance(other, Tensor) else Tensor(other)) - self

    def __mul__(self, other):
        return self._binary_op(
            other,
            np.multiply,
            lambda g, self_data, other_data: g * other_data,
            lambda g, self_data, other_data: g * self_data,
            "mul",
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._binary_op(
            other,
            np.divide,
            lambda g, self_data, other_data: g / other_data,
            lambda g, self_data, other_data: -g * self_data / (other_data**2),
            "div",
        )

    def __rtruediv__(self, other):
        return (other if isinstance(other, Tensor) else Tensor(other)) / self

    def __pow__(self, power):
        power_tensor = power if isinstance(power, Tensor) else None
        power_data = power_tensor.data if power_tensor is not None else power
        data = np.power(self.data, power_data)
        requires_grad = self.requires_grad or (
            power_tensor is not None and power_tensor.requires_grad
        )
        parents = (self,) + ((power_tensor,) if power_tensor is not None else ())
        out = Tensor(data, requires_grad=requires_grad, _children=parents, _op="pow")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                base_grad = out.grad * power_data * np.power(self.data, power_data - 1)
                self.grad += _sum_to_shape(base_grad, self.shape)
            if power_tensor is not None and power_tensor.requires_grad:
                safe = np.where(self.data > 0, self.data, 1)
                exp_grad = out.grad * data * np.log(safe)
                power_tensor.grad += _sum_to_shape(exp_grad, power_tensor.shape)

        out._backward = _backward
        return out

    def __rpow__(self, base):
        return Tensor(base) ** self

    def __neg__(self):
        out = Tensor(
            -self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad -= _sum_to_shape(out.grad, self.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data @ other_tensor.data
        requires_grad = self.requires_grad or other_tensor.requires_grad
        out = Tensor(
            data,
            requires_grad=requires_grad,
            _children=(self, other_tensor),
            _op="matmul",
        )

        def _backward():
            if out.grad is None:
                return
            a, b, g = self.data, other_tensor.data, out.grad
            a_vector, b_vector = a.ndim == 1, b.ndim == 1
            a = a[None, :] if a_vector else a
            b = b[:, None] if b_vector else b
            if a_vector and b_vector:
                g = g.reshape(g.shape + (1, 1))
            elif a_vector:
                g = np.expand_dims(g, -2)
            elif b_vector:
                g = np.expand_dims(g, -1)
            if self.requires_grad:
                ga = g @ b.swapaxes(-1, -2)
                self.grad += _sum_to_shape(
                    np.squeeze(ga, -2) if a_vector else ga, self.shape
                )
            if other_tensor.requires_grad:
                gb = a.swapaxes(-1, -2) @ g
                other_tensor.grad += _sum_to_shape(
                    np.squeeze(gb, -1) if b_vector else gb, other_tensor.shape
                )

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def transpose(self, *axes):
        data = self.data.transpose(*axes) if axes else self.data.T
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="transpose"
        )
        axes = tuple(range(self.ndim - 1, -1, -1)) if not axes else axes

        def _backward():
            if out.grad is not None and self.requires_grad:
                inv = np.argsort(axes)
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def squeeze(self, axis=None):
        data = np.squeeze(self.data, axis=axis)
        removed = ()
        if axis is None:
            removed = tuple(i for i, dim in enumerate(self.shape) if dim == 1)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            removed = tuple((a if a >= 0 else self.ndim + a) for a in axes)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="squeeze"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            for ax in sorted(removed):
                grad = np.expand_dims(grad, axis=ax)
            self.grad += grad

        out._backward = _backward
        return out

    def unsqueeze(self, axis):
        data = np.expand_dims(self.data, axis=axis)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="unsqueeze"
        )
        axis_norm = axis if axis >= 0 else axis + out.ndim

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += np.squeeze(out.grad, axis=axis_norm)

        out._backward = _backward
        return out

    def flatten(self, start_dim=0, end_dim=-1):
        ndim = self.ndim
        start = start_dim if start_dim >= 0 else ndim + start_dim
        end = end_dim if end_dim >= 0 else ndim + end_dim
        if start < 0 or end >= ndim or start > end:
            raise ValueError("Invalid flatten dimensions")
        prefix = self.shape[:start]
        middle = int(np.prod(self.shape[start : end + 1]))
        suffix = self.shape[end + 1 :]
        return self.reshape(*(prefix + (middle,) + suffix))

    def split(self, sections, axis=0):
        axis = axis if axis >= 0 else self.ndim + axis
        if isinstance(sections, int):
            if sections <= 0:
                raise ValueError("split size must be positive")
            return tuple(
                self[_slice(axis, start, min(start + sections, self.shape[axis]))]
                for start in range(0, self.shape[axis], sections)
            )
        sizes = tuple(sections)
        if sum(sizes) != self.shape[axis]:
            raise ValueError("split sizes must sum to the selected dimension")
        bounds = np.cumsum((0,) + sizes)
        return tuple(
            self[_slice(axis, bounds[i], bounds[i + 1])] for i in range(len(sizes))
        )

    def chunk(self, chunks, axis=0):
        if chunks <= 0:
            raise ValueError("chunks must be positive")
        size = (self.shape[axis] + chunks - 1) // chunks
        return self.split(size, axis)

    def unbind(self, axis=0):
        axis = axis if axis >= 0 else self.ndim + axis
        return tuple(
            self[_slice(axis, i, i + 1)].squeeze(axis) for i in range(self.shape[axis])
        )

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) != self.ndim:
            raise ValueError("permute expects as many axes as tensor dimensions")
        return self.transpose(*axes)

    def broadcast_to(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        data = np.broadcast_to(self.data, shape)
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="broadcast_to",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.shape)

        out._backward = _backward
        return out

    expand = broadcast_to

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="sum"
        )
        axes = _canonicalize_axes(axis, self.ndim)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axes is None:
                grad = np.broadcast_to(
                    np.array(grad, dtype=_grad_dtype(self.data)), self.shape
                )
            else:
                if not keepdims:
                    for ax in axes:
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def cumsum(self, axis=None):
        data = np.cumsum(self.data, axis=axis)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="cumsum"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            if axis is None:
                grad = np.cumsum(out.grad.reshape(-1)[::-1])[::-1].reshape(self.shape)
            else:
                grad = np.flip(
                    np.cumsum(np.flip(out.grad, axis=axis), axis=axis), axis=axis
                )
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        axes = _canonicalize_axes(axis, self.ndim)
        count = (
            self.data.size
            if axes is None
            else np.prod([self.data.shape[a] for a in axes])
        )
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="mean"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad / count
            if axes is None:
                grad = np.broadcast_to(
                    np.array(grad, dtype=_grad_dtype(self.data)), self.shape
                )
            else:
                if not keepdims:
                    for ax in axes:
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False, unbiased=False):
        axes = _canonicalize_axes(axis, self.ndim)
        count = (
            self.data.size
            if axes is None
            else np.prod([self.data.shape[a] for a in axes])
        )
        denom = count - 1 if (unbiased and count > 1) else count
        denom = max(denom, 1)
        mean = self.data.mean(axis=axis, keepdims=True)
        diff = self.data - mean
        sum_axes = None if axes is None else axes
        var_data = (diff**2).sum(axis=sum_axes, keepdims=True) / denom
        if not keepdims:
            if axes is None:
                var_data = np.asarray(var_data).reshape(())
            else:
                var_data = np.squeeze(var_data, axis=axes)
        out = Tensor(
            var_data, requires_grad=self.requires_grad, _children=(self,), _op="var"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad / denom
            if axes is None:
                grad = np.broadcast_to(
                    np.array(grad, dtype=_grad_dtype(self.data)), self.shape
                )
            else:
                if not keepdims:
                    for ax in axes:
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
            centered = self.data - mean
            self.grad += 2 * centered * grad

        out._backward = _backward
        return out

    def std(self, axis=None, keepdims=False, unbiased=False):
        return self.var(axis=axis, keepdims=keepdims, unbiased=unbiased) ** 0.5

    def sqrt(self, eps=0.0):
        data = np.sqrt(self.data + eps)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="sqrt"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                denom = 2 * data
                self.grad += out.grad / (denom + 1e-12)

        out._backward = _backward
        return out

    def abs(self):
        data = np.abs(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="abs"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                grad = np.sign(self.data)
                grad[self.data == 0] = 0
                self.grad += out.grad * grad

        out._backward = _backward
        return out

    def sign(self):
        data = np.sign(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="sign"
        )

        def _backward():
            return

        out._backward = _backward
        return out

    def norm(self, axis=None, keepdims=False, eps=1e-12):
        squared = (self * self).sum(axis=axis, keepdims=keepdims)
        if eps:
            squared = squared + eps
        return squared**0.5

    def logsumexp(self, axis=None, keepdims=False):
        max_keep = self.data.max(axis=axis, keepdims=True)
        shifted = self.data - max_keep
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=axis, keepdims=True)
        logsum = np.log(sum_exp) + max_keep
        if not keepdims and axis is not None:
            axes = _canonicalize_axes(axis, self.ndim)
            logsum = np.squeeze(logsum, axis=axes)
        out = Tensor(
            logsum, requires_grad=self.requires_grad, _children=(self,), _op="logsumexp"
        )
        axes = _canonicalize_axes(axis, self.ndim)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axes is None:
                grad = np.broadcast_to(
                    np.array(grad, dtype=_grad_dtype(self.data)), self.shape
                )
            else:
                if not keepdims:
                    for ax in axes:
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
            self.grad += grad * (exp_shifted / sum_exp)

        out._backward = _backward
        return out

    def exp(self):
        data = np.exp(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="exp"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad * data

        out._backward = _backward
        return out

    def log(self):
        data = np.log(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="log"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def log1p(self):
        return self._unary(np.log1p, lambda x, _: 1 / (1 + x), "log1p")

    def expm1(self):
        return self._unary(np.expm1, lambda x, _: np.exp(x), "expm1")

    def _unary(self, fn, derivative, name):
        data = fn(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op=name
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad * derivative(self.data, data)

        out._backward = _backward
        return out

    def sin(self):
        return self._unary(np.sin, lambda x, _: np.cos(x), "sin")

    def cos(self):
        return self._unary(np.cos, lambda x, _: -np.sin(x), "cos")

    def tan(self):
        return self._unary(np.tan, lambda x, _: 1 / np.cos(x) ** 2, "tan")

    def sinh(self):
        return self._unary(np.sinh, lambda x, _: np.cosh(x), "sinh")

    def cosh(self):
        return self._unary(np.cosh, lambda x, _: np.sinh(x), "cosh")

    def asin(self):
        return self._unary(np.arcsin, lambda x, _: 1 / np.sqrt(1 - x * x), "asin")

    def acos(self):
        return self._unary(np.arccos, lambda x, _: -1 / np.sqrt(1 - x * x), "acos")

    def atan(self):
        return self._unary(np.arctan, lambda x, _: 1 / (1 + x * x), "atan")

    def tanh(self):
        data = np.tanh(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="tanh"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad * (1 - data**2)

        out._backward = _backward
        return out

    def relu(self):
        data = np.maximum(self.data, 0)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="relu"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def leaky_relu(self, negative_slope=0.01):
        return self._unary(
            lambda x: np.where(x > 0, x, negative_slope * x),
            lambda x, _: np.where(x > 0, 1, negative_slope),
            "leaky_relu",
        )

    def elu(self, alpha=1.0):
        return self._unary(
            lambda x: np.where(x > 0, x, alpha * np.expm1(x)),
            lambda x, y: np.where(x > 0, 1, y + alpha),
            "elu",
        )

    def softplus(self):
        return self._unary(
            lambda x: np.logaddexp(0, x),
            lambda x, _: np.exp(-np.logaddexp(0, -x)),
            "softplus",
        )

    def gelu(self):
        c, k = np.sqrt(2 / np.pi), 0.044715
        return self._unary(
            lambda x: 0.5 * x * (1 + np.tanh(c * (x + k * x**3))),
            lambda x, _: 0.5 * (1 + np.tanh(c * (x + k * x**3)))
            + 0.5 * x * (1 - np.tanh(c * (x + k * x**3)) ** 2) * c * (1 + 3 * k * x**2),
            "gelu",
        )

    def swish(self):
        return self * self.sigmoid()

    def sigmoid(self):
        data = np.exp(-np.logaddexp(0, -self.data))
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += out.grad * data * (1 - data)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exps = np.exp(shifted)
        data = exps / exps.sum(axis=axis, keepdims=True)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="softmax"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            dot = (grad * data).sum(axis=axis, keepdims=True)
            self.grad += (grad - dot) * data

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        logsum = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        data = shifted - logsum
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="log_softmax"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            self.grad += grad - np.exp(data) * grad.sum(axis=axis, keepdims=True)

        out._backward = _backward
        return out

    def clip(self, min_value, max_value):
        data = np.clip(self.data, min_value, max_value)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="clip"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                mask = (self.data >= min_value) & (self.data <= max_value)
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        axes = _canonicalize_axes(axis, self.ndim)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="max"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            value, grad = data, out.grad
            if axes is None:
                value, grad = np.asarray(value).reshape((1,) * self.ndim), np.asarray(
                    grad
                ).reshape((1,) * self.ndim)
            elif not keepdims:
                for ax in axes:
                    value, grad = np.expand_dims(value, ax), np.expand_dims(grad, ax)
            mask = self.data == value
            self.grad += mask * grad / mask.sum(axis=axes, keepdims=True)

        out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        return (-self).max(axis=axis, keepdims=keepdims).__neg__()

    def prod(self, axis=None, keepdims=False):
        data = self.data.prod(axis=axis, keepdims=keepdims)
        axes = _canonicalize_axes(axis, self.ndim)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="prod"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            value, grad = data, out.grad
            if axes is None:
                value, grad = np.asarray(value).reshape((1,) * self.ndim), np.asarray(
                    grad
                ).reshape((1,) * self.ndim)
            elif not keepdims:
                for ax in axes:
                    value, grad = np.expand_dims(value, ax), np.expand_dims(grad, ax)
            zeros = self.data == 0
            count = zeros.sum(axis=axes, keepdims=True)
            safe = np.where(zeros, 1, self.data)
            base = safe.prod(axis=axes, keepdims=True)
            quotient = np.divide(
                value,
                self.data,
                out=np.zeros_like(self.data, dtype=_grad_dtype(self.data)),
                where=~zeros,
            )
            self.grad += grad * np.where(
                count == 0, quotient, np.where(count == 1, zeros * base, 0)
            )

        out._backward = _backward
        return out

    def maximum(self, other):
        other_tensor = other if isinstance(other, Tensor) else None
        other_data = (
            other_tensor.data if other_tensor is not None else np.asarray(other)
        )
        data = np.maximum(self.data, other_data)
        requires_grad = self.requires_grad or (
            other_tensor is not None and other_tensor.requires_grad
        )
        parents = tuple(p for p in (self, other_tensor) if isinstance(p, Tensor))
        out = Tensor(
            data, requires_grad=requires_grad, _children=parents, _op="maximum"
        )

        def _backward():
            if out.grad is None:
                return
            other_requires = other_tensor is not None and other_tensor.requires_grad
            grad_dtype = out.grad.dtype
            mask_self = (self.data > other_data).astype(grad_dtype, copy=False)
            mask_other = None
            if other_requires:
                mask_other = (self.data < other_tensor.data).astype(
                    grad_dtype, copy=False
                )
            equal_mask = (self.data == other_data).astype(grad_dtype, copy=False)
            if self.requires_grad:
                coeff = mask_self
                coeff = coeff + (equal_mask * (0.5 if other_requires else 1.0))
                self.grad += _sum_to_shape(out.grad * coeff, self.shape)
            if other_requires:
                coeff = mask_other
                coeff = coeff + (equal_mask * (0.5 if self.requires_grad else 1.0))
                other_tensor.grad += _sum_to_shape(out.grad * coeff, other_tensor.shape)

        out._backward = _backward
        return out

    def minimum(self, other):
        other_tensor = other if isinstance(other, Tensor) else None
        other_data = (
            other_tensor.data if other_tensor is not None else np.asarray(other)
        )
        data = np.minimum(self.data, other_data)
        requires_grad = self.requires_grad or (
            other_tensor is not None and other_tensor.requires_grad
        )
        parents = tuple(p for p in (self, other_tensor) if isinstance(p, Tensor))
        out = Tensor(
            data, requires_grad=requires_grad, _children=parents, _op="minimum"
        )

        def _backward():
            if out.grad is None:
                return
            other_requires = other_tensor is not None and other_tensor.requires_grad
            grad_dtype = out.grad.dtype
            mask_self = (self.data < other_data).astype(grad_dtype, copy=False)
            mask_other = None
            if other_requires:
                mask_other = (self.data > other_tensor.data).astype(
                    grad_dtype, copy=False
                )
            equal_mask = (self.data == other_data).astype(grad_dtype, copy=False)
            if self.requires_grad:
                coeff = mask_self
                coeff = coeff + (equal_mask * (0.5 if other_requires else 1.0))
                self.grad += _sum_to_shape(out.grad * coeff, self.shape)
            if other_requires:
                coeff = mask_other
                coeff = coeff + (equal_mask * (0.5 if self.requires_grad else 1.0))
                other_tensor.grad += _sum_to_shape(out.grad * coeff, other_tensor.shape)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        data = self.data[idx]
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="slice"
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = np.zeros_like(self.data, dtype=_grad_dtype(self.data))
            np.add.at(grad, idx, out.grad)
            self.grad += grad

        out._backward = _backward
        return out

    def flip(self, axis=None):
        data = np.flip(self.data, axis)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="flip"
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self.grad += np.flip(out.grad, axis)

        out._backward = _backward
        return out

    @staticmethod
    def zeros(shape, *, dtype=None, requires_grad=False):
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, *, dtype=None, requires_grad=False):
        return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def full(shape, fill_value, *, dtype=None, requires_grad=False):
        return Tensor(
            np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad
        )

    @staticmethod
    def randn(shape, *, dtype=None, requires_grad=False, seed=None):
        rng = np.random.default_rng(seed)
        return Tensor(
            rng.standard_normal(shape, dtype=dtype), requires_grad=requires_grad
        )

    @staticmethod
    def rand(shape, *, dtype=None, requires_grad=False, seed=None):
        data = np.random.default_rng(seed).random(shape)
        return Tensor(
            data if dtype is None else data.astype(dtype), requires_grad=requires_grad
        )

    @staticmethod
    def arange(start, stop=None, step=1, *, dtype=None, requires_grad=False):
        if stop is None:
            start, stop = 0, start
        return Tensor(
            np.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad
        )

    @staticmethod
    def from_numpy(array: np.ndarray, *, requires_grad=False):
        return Tensor(np.array(array, copy=True), requires_grad=requires_grad)


def where(condition, x, y):
    cond = np.asarray(condition, dtype=bool)
    x_t = x if isinstance(x, Tensor) else Tensor(x)
    y_t = y if isinstance(y, Tensor) else Tensor(y)
    data = np.where(cond, x_t.data, y_t.data)
    requires_grad = x_t.requires_grad or y_t.requires_grad
    parents = tuple(t for t in (x_t, y_t) if isinstance(t, Tensor))
    out = Tensor(data, requires_grad=requires_grad, _children=parents, _op="where")

    def _backward():
        if out.grad is None:
            return
        mask = np.broadcast_to(cond, out.data.shape)
        if x_t.requires_grad:
            x_t.grad += _sum_to_shape(out.grad * mask, x_t.shape)
        if y_t.requires_grad:
            y_t.grad += _sum_to_shape(out.grad * (~mask), y_t.shape)

    out._backward = _backward
    return out


def one_hot(indices, num_classes, dtype=float):
    data = indices.data if isinstance(indices, Tensor) else indices
    return Tensor(np.eye(num_classes, dtype=dtype)[np.asarray(data, dtype=int)])


def stack(tensors: Sequence[Tensor], axis=0):
    data = np.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(
        data, requires_grad=requires_grad, _children=tuple(tensors), _op="stack"
    )

    def _backward():
        if out.grad is None:
            return
        for i, t in enumerate(tensors):
            if not t.requires_grad:
                continue
            index = [slice(None)] * out.grad.ndim
            index[axis] = i
            t.grad += out.grad[tuple(index)]

    out._backward = _backward
    return out


def cat(tensors: Sequence[Tensor], axis=0):
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad, _children=tuple(tensors), _op="cat")
    sizes = np.cumsum([t.data.shape[axis] for t in tensors])

    def _backward():
        if out.grad is None:
            return
        start = 0
        for size, t in zip(sizes, tensors):
            slc = [slice(None)] * out.grad.ndim
            slc[axis] = slice(start, size)
            if t.requires_grad:
                t.grad += out.grad[tuple(slc)]
            start = size

    out._backward = _backward
    return out


def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = pred - target
    loss = diff * diff
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def l1_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _reduce((pred - target).abs(), reduction)


def huber_loss(
    pred: Tensor, target: Tensor, delta=1.0, reduction: str = "mean"
) -> Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive")
    diff = pred - target
    absolute = diff.abs()
    loss = where(
        absolute.data <= delta,
        0.5 * diff * diff / delta,
        absolute - 0.5 * delta,
    )
    return _reduce(loss, reduction)


def _reduce(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def binary_cross_entropy(input: Tensor, target, reduction="mean", eps=1e-12):
    target = target if isinstance(target, Tensor) else Tensor(target)
    p = input.clip(eps, 1 - eps)
    return _reduce(-(target * p.log() + (1 - target) * (1 - p).log()), reduction)


def binary_cross_entropy_with_logits(logits: Tensor, target, reduction="mean"):
    target = target if isinstance(target, Tensor) else Tensor(target)
    return _reduce(logits.softplus() - logits * target, reduction)


def cross_entropy(logits: Tensor, targets, axis=-1, reduction="mean"):
    if not isinstance(targets, Tensor):
        targets = Tensor(np.array(targets), requires_grad=False)
    log_probs = logits.log_softmax(axis=axis)
    axis = axis if axis >= 0 else log_probs.ndim + axis
    if not 0 <= axis < log_probs.ndim:
        raise ValueError("invalid class axis")
    if targets.data.ndim == log_probs.data.ndim:
        return _reduce(-(targets * log_probs).sum(axis=axis), reduction)
    num_classes = log_probs.data.shape[axis]
    one_hot = np.eye(num_classes, dtype=log_probs.data.dtype)[targets.data.astype(int)]
    if axis != log_probs.ndim - 1:
        one_hot = np.moveaxis(one_hot, -1, axis)
    target_tensor = Tensor(one_hot, requires_grad=False)
    return _reduce(-(target_tensor * log_probs).sum(axis=axis), reduction)


def gradcheck(fn, inputs: Sequence[Tensor], eps=1e-4, atol=1e-4, rtol=1e-2):
    for tensor in inputs:
        if not tensor.requires_grad:
            raise ValueError("All inputs to gradcheck must require grad")
    originals = [tensor.data for tensor in inputs]
    for tensor in inputs:
        if tensor.data.dtype.kind not in "fc":
            tensor.data = tensor.data.astype(float)
    out = fn(*inputs)
    if not isinstance(out, Tensor):
        raise TypeError("Function under test must return a Tensor")
    if out.data.size != 1:
        raise ValueError("gradcheck expects scalar output")
    for tensor in inputs:
        tensor.zero_grad()
    out.backward()
    ok = True
    for tensor in inputs:
        analytic = tensor.grad.copy()
        numeric = np.zeros_like(tensor.data, dtype=_grad_dtype(tensor.data))
        it = np.nditer(tensor.data, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = tensor.data[idx]
            tensor.data[idx] = orig + eps
            plus = fn(*inputs).detach().data
            plus_val = plus.item() if plus.size == 1 else plus.reshape(-1).sum()
            tensor.data[idx] = orig - eps
            minus = fn(*inputs).detach().data
            minus_val = minus.item() if minus.size == 1 else minus.reshape(-1).sum()
            tensor.data[idx] = orig
            numeric[idx] = (plus_val - minus_val) / (2 * eps)
            it.iternext()
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            ok = False
            break
    for tensor in inputs:
        tensor.zero_grad()
    for tensor, data in zip(inputs, originals):
        tensor.data = data
    return ok


__all__ = [
    "Tensor",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "where",
    "stack",
    "cat",
    "mse_loss",
    "l1_loss",
    "huber_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "gradcheck",
]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    features = Tensor(rng.normal(size=(128, 2)))
    weight = Tensor.randn((2, 1), seed=1, requires_grad=True)
    bias = Tensor.zeros((1,), requires_grad=True)
    targets = Tensor(
        features.data @ np.array([[1.5], [-2.0]])
        + 0.3
        + rng.normal(scale=0.1, size=(128, 1))
    )
    lr = 0.05
    for step in range(200):
        preds = features @ weight + bias
        loss = mse_loss(preds, targets)
        weight.zero_grad()
        bias.zero_grad()
        loss.backward()
        weight.data -= lr * weight.grad
        bias.data -= lr * bias.grad
        if step % 50 == 0:
            print(f"step {step:03d}  loss={loss.data.item():.6f}")
    print("fitted weight", weight.data.ravel())
    print("fitted bias", bias.data)

    check_tensor = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    print("gradcheck square:", gradcheck(lambda t: (t * t).sum(), (check_tensor,)))
