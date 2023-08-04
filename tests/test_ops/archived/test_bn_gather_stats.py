import unittest
import torch

# Now the test case only support CUDA, When using other device, the test will skip.
# TODO: save and read baseline data, which will make the test case work in other device.
class TestSchema(unittest.TestCase):

    def test_batch_norm_gather_stats_with_counts(self):
        if(torch.cuda.is_available() == False):
            return
        workpiece = 7
        input = torch.rand(2, 8, 32, 56, 56)
        mean_all =  torch.rand(workpiece,8)
        invstd_all =  torch.rand(workpiece,8)
        running_mean = torch.rand(8)
        running_var = torch.rand(8)
        momentum = 1e-4
        eps = 1e-5
        count_all = torch.rand(workpiece * 8)
        device_cuda = torch.device("cuda")
        res1 = self._test_bng(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all, device_cuda)
        import torch_dipu
        device_cuda = torch.device("cuda")
        res2 = self._test_bng(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all, device_cuda)        
        print(res1)
        print(res2)

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

    def _test_res(self, res1, res2):
        for i in range(len(res1)):
            self.assertTrue(torch.allclose(res1[i].cpu(),res2[i].cpu()))


if __name__ == '__main__':
    unittest.main()
