import unittest
import torch

# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_elemt(self):
        if(torch.cuda.is_available() == False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean =  torch.rand(8)
        invstd =  torch.rand(8)
        weight = torch.rand(8)
        bias = torch.rand(8)
        eps = 1e-5
        device_cuda = torch.device("cuda")
        res1 = self._test_bne(input, weight, bias, mean, invstd, eps, device_cuda)
        import torch_dipu
        device_cuda = torch.device("cuda")
        res2 = self._test_bne(input, weight, bias, mean, invstd, eps, device_cuda)
        print(res1)
        print(res2)

    def _test_bne(self, input, weight, bias, mean, invstd, eps, device):
        input_d = input.to(device)
        mean_d = mean.to(device)
        invstd_d = invstd.to(device)
        weight_d = weight.to(device)
        bias_d = bias.to(device)
        out = torch.batch_norm_elemt(input_d, weight_d, bias_d, mean_d, invstd_d, eps)
        return [out]

    def _test_res(self, res1, res2):
        for i in range(len(res1)):
            self.assertTrue(torch.allclose(res1[i].cpu(),res2[i].cpu()))


if __name__ == '__main__':
    unittest.main()
