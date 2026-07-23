import numpy as np


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()
        return self.optimizer.lr


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        super().__init__(optimizer)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma**self.last_epoch


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, t_max, eta_min=0.0):
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        super().__init__(optimizer)
        self.t_max, self.eta_min = t_max, eta_min

    def get_lr(self):
        phase = min(self.last_epoch, self.t_max) / self.t_max
        return (
            self.eta_min
            + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * phase)) / 2
        )


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


class AdamW(Adam):

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            m, v = self.state.setdefault(
                id(p),
                [
                    np.zeros_like(p.data, dtype=float),
                    np.zeros_like(p.data, dtype=float),
                ],
            )
            m *= b1
            m += (1 - b1) * p.grad
            v *= b2
            v += (1 - b2) * p.grad * p.grad
            p.data *= 1 - self.lr * self.weight_decay
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


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay, self.state = lr, eps, weight_decay, {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            total = self.state.setdefault(id(p), np.zeros_like(p.data, dtype=float))
            total += g * g
            p.data -= self.lr * g / (np.sqrt(total) + self.eps)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == np.inf:
        total = max(np.abs(g).max(initial=0) for g in grads)
    else:
        total = sum(np.abs(g.astype(float)) ** norm_type for g in grads).sum() ** (
            1 / norm_type
        )
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for grad in grads:
            grad *= scale
    return total
