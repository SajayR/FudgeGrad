import unittest
import numpy as np
from src import (
    Adam,
    AdamW,
    BatchNorm,
    Embedding,
    Tensor,
    binary_cross_entropy_with_logits,
    cat,
    conv2d,
    clip_grad_norm_,
    gradcheck,
    max_pool2d,
    mse_loss,
    no_grad,
    Linear,
)


class TestAutograd(unittest.TestCase):
    def test_broadcast_and_reused_node(self):
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        ((x * x + x).sum()).backward()
        np.testing.assert_allclose(x.grad, [[3, 5]])

    def test_repeated_backward_accumulates_only_leaves(self):
        x = Tensor(3.0, requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        loss.backward()
        self.assertEqual(x.grad.item(), 12.0)

    def test_batched_matmul_gradcheck(self):
        a, b = Tensor(np.random.randn(2, 3, 4), requires_grad=True), Tensor(
            np.random.randn(4, 5), requires_grad=True
        )
        self.assertTrue(gradcheck(lambda x, y: (x @ y).sum(), (a, b)))

    def test_advanced_index_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[[0, 0, 2]].sum().backward()
        np.testing.assert_allclose(x.grad, [2, 0, 1])

    def test_cat_skipped_parent(self):
        x = Tensor([0.0])
        y = Tensor([1.0, 2.0], requires_grad=True)
        cat((x, y)).sum().backward()
        np.testing.assert_allclose(y.grad, [1, 1])

    def test_embedding_repeated_indices(self):
        e = Embedding(4, 2)
        e(np.array([1, 1, 3])).sum().backward()
        np.testing.assert_allclose(e.weight.grad[:, 0], [0, 2, 0, 1])

    def test_linear_optimizer(self):
        rng = np.random.default_rng(0)
        x = Tensor(rng.normal(size=(32, 2)))
        y = Tensor(x.data @ [[2.0], [-1.0]] + 0.3)
        model = Linear(2, 1, seed=0)
        opt = Adam(model.parameters(), lr=0.08)
        for _ in range(100):
            opt.zero_grad()
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 1e-4)

    def test_batchnorm_eval(self):
        layer = BatchNorm(2)
        layer(Tensor(np.ones((4, 2))))
        layer.eval()
        self.assertEqual(layer(Tensor(np.ones((3, 2)))).shape, (3, 2))

    def test_batchnorm_state_includes_running_statistics(self):
        layer = BatchNorm(2)
        layer(Tensor([[1.0, 2.0], [3.0, 4.0]]))
        state = layer.state_dict()
        layer.running_mean[:] = 99
        layer.load_state_dict(state)
        np.testing.assert_allclose(layer.running_mean, state["running_mean"])

    def test_conv_and_pool_gradchecks(self):
        x, w = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True), Tensor(
            np.random.randn(1, 1, 2, 2), requires_grad=True
        )
        self.assertTrue(gradcheck(lambda a, b: conv2d(a, b).sum(), (x, w)))
        self.assertTrue(
            gradcheck(
                lambda a: max_pool2d(a, 2).sum(),
                (Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True),),
            )
        )

    def test_logits_bce(self):
        x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
        self.assertTrue(
            gradcheck(lambda a: binary_cross_entropy_with_logits(a, [0, 1, 1]), (x,))
        )

    def test_stable_sigmoid(self):
        x = Tensor([-1000.0, 1000.0], requires_grad=True)
        x.sigmoid().sum().backward()
        np.testing.assert_allclose(x.grad, [0, 0])

    def test_integer_reduction_has_fractional_gradient(self):
        x = Tensor([1, 2], requires_grad=True)
        x.mean().backward()
        np.testing.assert_allclose(x.grad, [0.5, 0.5])
        self.assertTrue(gradcheck(lambda a: a.mean(), (x,)))

    def test_adamw_and_gradient_clipping(self):
        weight = Tensor([1.0], requires_grad=True)
        weight.grad[...] = 4
        self.assertEqual(clip_grad_norm_([weight], 1), 4)
        np.testing.assert_allclose(weight.grad, [1])
        AdamW([weight], lr=0.1, weight_decay=0.1).step()
        self.assertLess(weight.item(), 1)

    def test_no_grad_skips_graph_construction(self):
        x = Tensor(2.0, requires_grad=True)
        with no_grad():
            y = x * x
        self.assertFalse(y.requires_grad)
        self.assertEqual(y._prev, ())


if __name__ == "__main__":
    unittest.main()
