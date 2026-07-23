import unittest
import numpy as np
from src import Adam, BatchNorm, Embedding, Linear, Tensor, cat, gradcheck, mse_loss


class TestAutograd(unittest.TestCase):
    def test_broadcast_and_reused_node(self):
        x = Tensor([[1., 2.]], requires_grad=True); ((x * x + x).sum()).backward()
        np.testing.assert_allclose(x.grad, [[3, 5]])

    def test_repeated_backward_accumulates_only_leaves(self):
        x = Tensor(3., requires_grad=True); loss = (x * x).sum(); loss.backward(); loss.backward()
        self.assertEqual(x.grad.item(), 12.)

    def test_batched_matmul_gradcheck(self):
        a, b = Tensor(np.random.randn(2, 3, 4), requires_grad=True), Tensor(np.random.randn(4, 5), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: (x @ y).sum(), (a, b)))

    def test_advanced_index_accumulates(self):
        x = Tensor([1., 2., 3.], requires_grad=True); x[[0, 0, 2]].sum().backward()
        np.testing.assert_allclose(x.grad, [2, 0, 1])

    def test_cat_skipped_parent(self):
        x = Tensor([0.]); y = Tensor([1., 2.], requires_grad=True); cat((x, y)).sum().backward()
        np.testing.assert_allclose(y.grad, [1, 1])

    def test_embedding_repeated_indices(self):
        e = Embedding(4, 2); e(np.array([1, 1, 3])).sum().backward()
        np.testing.assert_allclose(e.weight.grad[:, 0], [0, 2, 0, 1])

    def test_linear_optimizer(self):
        rng = np.random.default_rng(0); x = Tensor(rng.normal(size=(32, 2))); y = Tensor(x.data @ [[2.], [-1.]] + .3)
        model = Linear(2, 1, seed=0); opt = Adam(model.parameters(), lr=.08)
        for _ in range(100): opt.zero_grad(); loss = mse_loss(model(x), y); loss.backward(); opt.step()
        self.assertLess(loss.item(), 1e-4)

    def test_batchnorm_eval(self):
        layer = BatchNorm(2); layer(Tensor(np.ones((4, 2)))); layer.eval()
        self.assertEqual(layer(Tensor(np.ones((3, 2)))).shape, (3, 2))


if __name__ == "__main__": unittest.main()
