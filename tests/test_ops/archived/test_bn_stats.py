import unittest
import torch

# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_stats(self):
        if(torch.cuda.is_available() == False):
            return
        x_cuda = torch.randn([5,5]).cuda()
        z1_mean, z1_invstd = torch.batch_norm_stats(x_cuda, 1e-5)
        import torch_dipu
        x_dipu = x_cuda.cuda()
        z2_mean, z2_invstd = torch.batch_norm_stats(x_dipu, 1e-5)
        self.assertTrue(torch.allclose(z1_mean.cpu(),z2_mean.cpu()))
        self.assertTrue(torch.allclose(z1_invstd.cpu(),z2_invstd.cpu()))
        print(z2_mean)
        print(z2_invstd)


if __name__ == '__main__':
    unittest.main()
