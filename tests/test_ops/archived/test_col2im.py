import torch_dipu
import torch 
import unittest

from torch_dipu.testing._internal.common_utils import dipu, TestCase, run_tests


class TestSchema(TestCase):

    def test_arange(self):
        inp_cuda = torch.randn(1, 3, 10, 12).cuda()
        w_cuda = torch.randn(2, 3, 4, 5).cuda()
        inp_unf_cuda = torch.nn.functional.unfold(inp_cuda, (4, 5))
        out_unf_cuda = inp_unf_cuda.transpose(1, 2).matmul(w_cuda.view(w_cuda.size(0), -1).t()).transpose(1, 2)
        out_cuda = torch.nn.functional.fold(inp_unf_cuda, (7, 8), (1, 1))

        inp = inp_cuda.cpu()
        w = w_cuda.cpu()
        inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        out = torch.nn.functional.fold(inp_unf, (7, 8), (1, 1))

        self.assertTrue(torch.allclose(out_cuda.cpu(), out,atol = 1e-5, rtol = 1e-5))


if __name__ == '__main__':
    run_tests()





