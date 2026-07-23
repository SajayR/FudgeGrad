import numpy as np
from .tensor import Tensor, _grad_dtype, _sum_to_shape, where


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def pad(x, pad_width, value=0):
    x = x if isinstance(x, Tensor) else Tensor(x)
    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width),) * x.ndim
    else:
        pad_width = tuple(
            (width, width) if isinstance(width, int) else tuple(width)
            for width in pad_width
        )
    if len(pad_width) != x.ndim:
        raise ValueError("pad_width must specify every dimension")
    data = np.pad(x.data, pad_width, constant_values=value)
    out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op="pad")

    def _backward():
        if out.grad is None or not x.requires_grad:
            return
        source = tuple(
            slice(before, before + size)
            for (before, _), size in zip(pad_width, x.shape)
        )
        x.grad += out.grad[source]

    out._backward = _backward
    return out


def scaled_dot_product_attention(query, key, value, mask=None):
    query = query if isinstance(query, Tensor) else Tensor(query)
    key = key if isinstance(key, Tensor) else Tensor(key)
    value = value if isinstance(value, Tensor) else Tensor(value)
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    axes = list(range(key.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    scores = query @ key.transpose(axes) / np.sqrt(query.shape[-1])
    if mask is not None:
        scores = where(mask, scores, -np.inf)
    return scores.softmax(-1) @ value


def conv2d(x, weight, bias=None, stride=1, padding=0):
    x = x if isinstance(x, Tensor) else Tensor(x)
    weight = weight if isinstance(weight, Tensor) else Tensor(weight)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d expects input (N,C,H,W), weight (O,C,kH,kW)")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    n, c, h, w = x.shape
    o, wc, kh, kw = weight.shape
    if c != wc:
        raise ValueError("input and weight channels differ")
    padded = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))[
        :, :, ::sh, ::sw
    ]
    data = np.einsum("nchwkl,ockl->nohw", windows, weight.data) + (
        bias.data.reshape(1, -1, 1, 1)
        if isinstance(bias, Tensor)
        else 0 if bias is None else np.asarray(bias).reshape(1, -1, 1, 1)
    )
    parents = (x, weight) + ((bias,) if isinstance(bias, Tensor) else ())
    out = Tensor(
        data,
        requires_grad=any(p.requires_grad for p in parents),
        _children=parents,
        _op="conv2d",
    )

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            gx = np.zeros_like(padded, dtype=_grad_dtype(x.data))
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    gx[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += np.einsum(
                        "no,ockl->nckl", out.grad[:, :, i, j], weight.data
                    )
            x.grad += gx[:, :, ph : ph + h, pw : pw + w]
        if weight.requires_grad:
            gw = np.zeros_like(weight.data, dtype=_grad_dtype(weight.data))
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    gw += np.einsum(
                        "no,nckl->ockl", out.grad[:, :, i, j], windows[:, :, i, j]
                    )
            weight.grad += gw
        if isinstance(bias, Tensor) and bias.requires_grad:
            bias.grad += _sum_to_shape(out.grad.sum(axis=(0, 2, 3)), bias.shape)

    out._backward = _backward
    return out


def conv1d(x, weight, bias=None, stride=1, padding=0):
    x = x if isinstance(x, Tensor) else Tensor(x)
    weight = weight if isinstance(weight, Tensor) else Tensor(weight)
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("conv1d expects input (N,C,L), weight (O,C,kL)")
    return conv2d(
        x.unsqueeze(-2),
        weight.unsqueeze(-2),
        bias,
        stride=(1, stride),
        padding=(0, padding),
    ).squeeze(-2)


def max_pool2d(x, kernel_size, stride=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(kernel_size if stride is None else stride)
    if x.ndim != 4:
        raise ValueError("max_pool2d expects input (N,C,H,W)")
    windows = np.lib.stride_tricks.sliding_window_view(x.data, (kh, kw), axis=(2, 3))[
        :, :, ::sh, ::sw
    ]
    data = windows.max(axis=(-1, -2))
    out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op="max_pool2d")

    def _backward():
        if out.grad is None or not x.requires_grad:
            return
        gx = np.zeros_like(x.data, dtype=_grad_dtype(x.data))
        mask = windows == data[..., None, None]
        share = (
            out.grad[..., None, None] * mask / mask.sum(axis=(-1, -2), keepdims=True)
        )
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                gx[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += share[
                    :, :, i, j
                ]
        x.grad += gx

    out._backward = _backward
    return out


def max_pool1d(x, kernel_size, stride=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    if x.ndim != 3:
        raise ValueError("max_pool1d expects input (N,C,L)")
    return max_pool2d(
        x.unsqueeze(-2),
        (1, kernel_size),
        None if stride is None else (1, stride),
    ).squeeze(-2)


def avg_pool2d(x, kernel_size, stride=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(kernel_size if stride is None else stride)
    if x.ndim != 4:
        raise ValueError("avg_pool2d expects input (N,C,H,W)")
    windows = np.lib.stride_tricks.sliding_window_view(x.data, (kh, kw), axis=(2, 3))[
        :, :, ::sh, ::sw
    ]
    data = windows.mean(axis=(-1, -2))
    out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op="avg_pool2d")

    def _backward():
        if out.grad is None or not x.requires_grad:
            return
        gx = np.zeros_like(x.data, dtype=_grad_dtype(x.data))
        share = out.grad[..., None, None] / (kh * kw)
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                gx[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += share[
                    :, :, i, j
                ]
        x.grad += gx

    out._backward = _backward
    return out


def avg_pool1d(x, kernel_size, stride=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    if x.ndim != 3:
        raise ValueError("avg_pool1d expects input (N,C,L)")
    return avg_pool2d(
        x.unsqueeze(-2),
        (1, kernel_size),
        None if stride is None else (1, stride),
    ).squeeze(-2)
