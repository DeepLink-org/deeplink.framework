import unittest
import torch


# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_backward_reduce(self):
        if(torch.cuda.is_available() == False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean =  torch.rand(8)
        invstd =  torch.rand(8)
        weight = torch.rand(8)
        grad_out = torch.rand(2, 8, 32, 56, 56)
        device_cuda = torch.device("cuda")
        res1 = self._test_bnbr(input, mean, invstd, weight, grad_out, device_cuda)
        import torch_dipu
        device_cuda = torch.device("cuda")
        res2 = self._test_bnbr(input, mean, invstd, weight, grad_out, device_cuda)  
        print(res1)
        print(res2)

    def _test_bnbr(self, input, mean_all, invstd_all, weight, grad_out, device):
        input_d = input.to(device)
        mean_all_d = mean_all.to(device)
        invstd_all_d = invstd_all.to(device)
        weight_d = weight.to(device)
        grad_out_d = grad_out.to(device)
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_out_d,
            input_d,
            mean_all_d,
            invstd_all_d,
            weight_d,
            True,
            True,
            True,
        )
        return [sum_dy, sum_dy_xmu, grad_weight, grad_bias]

    def _test_res(self, res1, res2):
        for i in range(len(res1)):
            self.assertTrue(torch.allclose(res1[i].cpu(),res2[i].cpu()))


if __name__ == '__main__':
    unittest.main()
