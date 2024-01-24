# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNorm(TestCase):
    DIPU = torch.device("dipu")
    CPU = torch.device("cpu")

    def test_norm(self):
        x = torch.randn(3, 3, 2).to(self.DIPU)
        y = torch.norm(x)
        z = torch.norm(x.cpu())
        self.assertEqual(y.to(self.CPU), z)

        y = torch.norm(x, 2)
        z = torch.norm(x.cpu(), 2)
        self.assertEqual(y.to(self.CPU), z)

    def test_linalg_vector_norm(self):
        a = torch.arange(9, dtype=torch.float) - 4
        na = torch.linalg.vector_norm(a, ord=2.0)
        nad = torch.linalg.vector_norm(a.cuda(), ord=2.0)
        self.assertTrue(torch.allclose(na, nad.cpu(), atol=1e-3, rtol=1e-3))

        b = a.reshape((3, 3))
        nb = torch.linalg.vector_norm(b, ord=2.0)
        nbd = torch.linalg.vector_norm(b.cuda(), ord=2.0)
        self.assertTrue(torch.allclose(nb, nbd.cpu(), atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    run_tests()
