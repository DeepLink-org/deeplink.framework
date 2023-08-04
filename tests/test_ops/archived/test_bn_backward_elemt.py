import unittest
import torch

# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_backward_elemt(self):
        if(torch.cuda.is_available() == False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean =  torch.rand(8)
        invstd =  torch.rand(8)
        weight = torch.rand(8)
        grad_out = torch.rand(2, 8, 32, 56, 56)
        sum_dy = torch.rand(8)
        sum_dy_xmu = torch.rand(8)
        count_tensor = torch.tensor([5, 5, 4, 4, 3, 1, 5, 7], dtype=torch.int32)
        device_cuda = torch.device("cuda")
        res1 = self._test_bnbe(input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor, grad_out, device_cuda)
        import torch_dipu
        device_cuda = torch.device("cuda")
        res2 = self._test_bnbe(input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor, grad_out, device_cuda)
        print(res1)
        print(res2)

    def _test_bnbe(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor, grad_out, device):
        input_d = input.to(device)
        mean_d = mean.to(device)
        invstd_d = invstd.to(device)
        weight_d = weight.to(device)
        grad_out_d = grad_out.to(device)
        sum_dy_d = sum_dy.to(device)
        sum_dy_xmu_d = sum_dy_xmu.to(device)
        count_tensor_d = count_tensor.to(device)
        grad_input = torch.batch_norm_backward_elemt(
            grad_out_d,
            input_d,
            mean_d,
            invstd_d,
            weight_d,
            sum_dy_d,
            sum_dy_xmu_d,
            count_tensor_d
        )
        return [grad_input]

    def _test_res(self, res1, res2):
        for i in range(len(res1)):
            self.assertTrue(torch.allclose(res1[i].cpu(),res2[i].cpu()))


if __name__ == '__main__':
    unittest.main()
