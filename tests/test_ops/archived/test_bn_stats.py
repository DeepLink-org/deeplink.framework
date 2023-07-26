import unittest
import torch


class TestSchema(unittest.TestCase):

    def test_bn_stats(self):
        x_cuda = torch.randn([5,5]).cuda()
        z1_mean, z1_invstd = torch.batch_norm_stats(x_cuda, 1.5)
        import torch_dipu
        x_dipu = x_cuda.cuda()
        z2_mean, z2_invstd = torch.batch_norm_stats(x_dipu, 1.5)
        self.assertTrue(torch.allclose(z1_mean.cpu(),z2_mean.cpu()))
        self.assertTrue(torch.allclose(z1_invstd.cpu(),z2_invstd.cpu()))


if __name__ == '__main__':
    unittest.main()
