import torch_dipu
import torch 
import unittest

from torch_dipu.testing._internal.common_utils import dipu, TestCase, run_tests


class TestSchema(TestCase):

    def test_arange(self):

        unfold_cuda = torch.nn.Unfold(kernel_size=(2, 3)).cuda()
        input_cuda = torch.randn(2, 5, 3, 4).cuda()

        input_cpu = input_cuda.cpu()
        unfold_cpu = unfold_cuda.cpu()

        output_cuda = unfold_cuda(input_cuda)
        output_cpu = unfold_cpu(input_cpu)

        self.assertTrue(torch.allclose(output_cuda.cpu(), output_cpu))


if __name__ == '__main__':
    run_tests()

