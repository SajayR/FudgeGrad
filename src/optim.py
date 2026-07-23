"""In-place NumPy optimizers for FudgeGrad Parameters."""

import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(dict.fromkeys(params))

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay, self.nesterov, self.velocity = (
            lr,
            momentum,
            weight_decay,
            nesterov,
            {},
        )
        if nesterov and momentum <= 0:
            raise ValueError("nesterov requires momentum")

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            if self.momentum:
                v = self.velocity.setdefault(id(p), np.zeros_like(p.data, dtype=float))
                v *= self.momentum
                v += g
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.betas, self.eps, self.weight_decay, self.t, self.state = (
            lr,
            betas,
            eps,
            weight_decay,
            0,
            {},
        )

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            m, v = self.state.setdefault(
                id(p),
                [
                    np.zeros_like(p.data, dtype=float),
                    np.zeros_like(p.data, dtype=float),
                ],
            )
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p.data -= (
                self.lr
                * (m / (1 - b1**self.t))
                / (np.sqrt(v / (1 - b2**self.t)) + self.eps)
            )


class RMSprop(Optimizer):
    def __init__(
        self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0
    ):
        super().__init__(params)
        self.lr, self.alpha, self.eps, self.weight_decay, self.momentum, self.state = (
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            {},
        )

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            avg, buf = self.state.setdefault(
                id(p),
                [
                    np.zeros_like(p.data, dtype=float),
                    np.zeros_like(p.data, dtype=float),
                ],
            )
            avg *= self.alpha
            avg += (1 - self.alpha) * g * g
            step = g / (np.sqrt(avg) + self.eps)
            if self.momentum:
                buf *= self.momentum
                buf += step
                step = buf
            p.data -= self.lr * step
