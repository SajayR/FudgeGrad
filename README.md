# FudgeGrad

FudgeGrad is a CPU-only reverse-mode autodiff and neural-network library written from scratch with NumPy. It is intentionally small: tensors retain a dynamic computation graph, gradients accumulate at leaves, and models, optimizers, convolutions, attention, and recurrent layers are all built on that core.

## Install

The repository has one runtime dependency:

```bash
pip install numpy
```

Run tests from the repository root:

```bash
python -m unittest discover -v
```

Import directly from the source package:

```python
from src import Tensor, Linear, Sequential, ReLU, Adam, mse_loss
```

## Usage

```python
from src import Adam, Linear, Tensor, mse_loss

x = Tensor([[1.0, 2.0]], requires_grad=False)
y = Tensor([[3.0]])
model, opt = Linear(2, 1, seed=0), Adam(model.parameters(), lr=1e-2)

for _ in range(100):
    opt.zero_grad()
    loss = mse_loss(model(x), y)
    loss.backward()
    opt.step()
```

## Tensor and autograd

`Tensor(data, requires_grad=True)` records operations involving it. Call `backward()` on a scalar result; non-scalar results need an explicit upstream gradient. Leaf gradients accumulate, so call `zero_grad()` before a fresh optimization step.

The core supports broadcasting, slicing and repeated advanced indices, reshape/transpose/permute, reductions, batched matrix multiplication, softmax/log-softmax, normalization, clipping, elementwise math, and gradient checking.

Use `gradcheck(fn, inputs)` to compare analytic and finite-difference gradients, and `no_grad()` for graph-free evaluation.

## Models

Available modules include `Linear`, `Sequential`, `ModuleList`, `ParameterList`, `Embedding`, `Flatten`, `Dropout`, `LayerNorm`, `BatchNorm`, `GroupNorm`, `InstanceNorm`, `Conv1d`, `Conv2d`, pooling layers, activation modules, `MultiheadAttention`, `RNN`, and `LSTM`.

`state_dict()` and `load_state_dict()` handle parameters plus registered buffers such as BatchNorm running statistics. `train()` and `eval()` propagate mode to nested modules.

## Functions and losses

`where`, `stack`, `cat`, `one_hot`, `pad`, 1D/2D convolution and pooling, and `scaled_dot_product_attention` are available as top-level functions.

Losses are available as functions and modules:

- `mse_loss`, `l1_loss`, `huber_loss`
- `binary_cross_entropy`, `binary_cross_entropy_with_logits`
- `cross_entropy`
- `MSELoss`, `L1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`

Most losses accept `reduction="none"`, `"sum"`, or `"mean"`.

## Optimization

`SGD`, `Adam`, `AdamW`, `Adagrad`, and `RMSprop` update parameters in place. `clip_grad_norm_` clips a collection of parameter gradients. `StepLR`, `ExponentialLR`, and `CosineAnnealingLR` update an optimizer's learning rate.

Call `optimizer.zero_grad()`, `loss.backward()`, then `optimizer.step()` in the training loop; call a scheduler's `step()` after its optimizer update.
