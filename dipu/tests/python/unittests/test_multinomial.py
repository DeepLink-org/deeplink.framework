# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMultinomial(TestCase):
    def test_multinomial(self):
        device = torch.device("dipu")
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).to(device)
        out = torch.ones((300, 300), dtype=torch.long).to(device).mul(-1)
        N = 50000
        samples = torch.multinomial(probs, num_samples=N, replacement=True, out=out)
        t = samples.reshape((-1,))
        self.assertTrue(torch.all(t[:N].ge(0)))
        self.assertTrue(torch.all(t[:N].less(4)))
        self.assertTrue(torch.all(t[N:].eq(-1)))
        for i in range(4):
            cnt = samples.eq(i).sum().item()
            # respect the probability
            self.assertAlmostEqual(cnt / N, probs[i].item(), delta=0.01)


if __name__ == "__main__":
    run_tests()
