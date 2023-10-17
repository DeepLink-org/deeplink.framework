# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestClamp(TestCase):
    def setUp(self):
        self.x = torch.tensor([0.1, 1.2, -0.8, 0.3, 0.7]).cuda()
        self.y = torch.tensor([0.1, 1.2, -0.8, 0.3, 0.7])

    def test_clamp_min_(self):
        self.x.clamp_min_(0.0)
        self.y.clamp_min_(0.0)
        self.assertTrue(torch.allclose(self.x.cpu(), self.y, atol=1e-3, rtol=1e-3))

    def test_clamp_max_(self):
        self.x.clamp_max_(0.5)
        self.y.clamp_max_(0.5)
        self.assertTrue(torch.allclose(self.x.cpu(), self.y, atol=1e-3, rtol=1e-3))

    def test_clamp_min__tensor(self):
        min = torch.tensor([0.2])
        self.x.clamp_min_(min.cuda())
        self.y.clamp_min_(min)
        self.assertTrue(torch.allclose(self.x.cpu(), self.y, atol=1e-3, rtol=1e-3))

    def test_clamp_min_tensor(self):
        min2 = torch.tensor([0.4])
        torch.clamp_min(self.x, min2.cuda())
        torch.clamp_min(self.y, min2)
        self.assertTrue(torch.allclose(self.x.cpu(), self.y, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    run_tests()
