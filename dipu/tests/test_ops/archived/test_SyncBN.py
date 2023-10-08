import unittest
import torch
import os
os.environ['DIPU_MOCK_CUDA'] = "False"
import torch_dipu
dipu = torch_dipu.dipu.diputype
device_cuda = torch.device("cuda")
assert device_cuda.type == "cuda"
device_dipu = torch.device(dipu)


# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_stats(self):
        if (torch.cuda.is_available() is False):
            return
        x_cuda = torch.randn([5, 5]).to(device_cuda)
        z1_mean, z1_invstd = torch.batch_norm_stats(x_cuda, 1e-5)
        x_dipu = x_cuda.to(device_dipu)
        z2_mean, z2_invstd = torch.batch_norm_stats(x_dipu, 1e-5)

        self.assertTrue(torch.allclose(z1_mean.cpu(), z2_mean.cpu()))
        self.assertTrue(torch.allclose(z1_invstd.cpu(), z2_invstd.cpu()))

    def test_batch_norm_gather_stats_with_counts(self):
        if (torch.cuda.is_available() is False):
            return
        world_size = 7
        input = torch.rand(2, 8, 32, 56, 56)
        mean_all = torch.rand(world_size, 8)
        invstd_all = torch.rand(world_size, 8)
        running_mean = torch.rand(8)
        running_var = torch.rand(8)
        momentum = 1e-4
        eps = 1e-5
        count_all = torch.rand(world_size)
        res1 = self._test_bng(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all, device_cuda)
        res2 = self._test_bng(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all, device_dipu)
        self._test_res(res1, res2)

    def test_batch_norm_backward_elemt(self):
        if (torch.cuda.is_available() is False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean = torch.rand(8)
        invstd = torch.rand(8)
        weight = torch.rand(8)
        grad_out = torch.rand(2, 8, 32, 56, 56)
        sum_dy = torch.rand(8)
        sum_dy_xmu = torch.rand(8)
        count_tensor = torch.tensor([5, 5, 4, 4, 3, 1, 5, 7], dtype=torch.int32)
        res1 = self._test_bnbe(input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor, grad_out, device_cuda)
        res2 = self._test_bnbe(input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor, grad_out, device_dipu)
        self._test_res(res1, res2)

    def test_batch_norm_elemt(self):
        if (torch.cuda.is_available() is False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean = torch.rand(8)
        invstd = torch.rand(8)
        weight = torch.rand(8)
        bias = torch.rand(8)
        eps = 1e-5
        res1 = self._test_bne(input, weight, bias, mean, invstd, eps, device_cuda)
        res2 = self._test_bne(input, weight, bias, mean, invstd, eps, device_dipu)
        self._test_res(res1, res2)

    def test_batch_norm_backward_reduce(self):
        if (torch.cuda.is_available() is False):
            return
        input = torch.rand(2, 8, 32, 56, 56)
        mean = torch.rand(8)
        invstd = torch.rand(8)
        weight = torch.rand(8)
        grad_out = torch.rand(2, 8, 32, 56, 56)
        res1 = self._test_bnbr(input, mean, invstd, weight, grad_out, device_cuda)
        res2 = self._test_bnbr(input, mean, invstd, weight, grad_out, device_dipu)
        self._test_res(res1, res2)

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

    def _test_bne(self, input, weight, bias, mean, invstd, eps, device):
        input_d = input.to(device)
        mean_d = mean.to(device)
        invstd_d = invstd.to(device)
        weight_d = weight.to(device)
        bias_d = bias.to(device)
        out = torch.batch_norm_elemt(input_d, weight_d, bias_d, mean_d, invstd_d, eps)
        return [out]

    def _test_bng(self, input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all, device):
        input_d = input.to(device)
        mean_all_d = mean_all.to(device)
        invstd_all_d = invstd_all.to(device)
        running_mean_d = running_mean.to(device)
        running_var_d = running_var.to(device)
        momentum_d = momentum
        eps_d = eps
        count_all_d = count_all.to(device)
        mean_d, invstd_d = torch.batch_norm_gather_stats_with_counts(
            input_d,
            mean_all_d,
            invstd_all_d,
            running_mean_d,
            running_var_d,
            momentum_d,
            eps_d,
            count_all_d
        )
        return [mean_d, invstd_d]

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
            self.assertTrue(torch.allclose(res1[i].cpu(), res2[i].cpu()))


if __name__ == '__main__':
    unittest.main()
