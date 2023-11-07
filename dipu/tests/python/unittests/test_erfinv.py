# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestErfinv(TestCase):
    def test_erfinv(self):
        for i in range(100):
            x = torch.randn((i + 1, i + 2))
            self.assertTrue(
                torch.allclose(
                    torch.special.erfinv(x),
                    torch.special.erfinv(x.cuda()).cpu(),
                    atol=1e-3,
                    rtol=1e-3,
                    equal_nan=True,
                )
            )


if __name__ == "__main__":
    run_tests()
