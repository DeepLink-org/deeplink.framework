import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


# class TestArgmax(TestCase):
#     def test_argmax(self):
#         dipu = torch.device("dipu")
#         cpu = torch.device("cpu")
#         input = torch.randn(4, 4)
#         dipu = torch.argmax(input.to(dipu))
#         cpu = torch.argmax(input.to(cpu))
#         self.assertEqual(cpu, dipu.to(cpu))


if __name__ == "__main__":
    run_tests()
