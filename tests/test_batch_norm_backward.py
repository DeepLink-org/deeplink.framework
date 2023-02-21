
import torch
import torch.nn as nn
import numpy as np

torch.ops.load_library("../build/libtorch_dipu.so")

class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.op1 = nn.Conv2d(in_channels, in_channels, 1)
        self.op2 = nn.BatchNorm2d(in_channels)
        self.op2.running_mean = torch.tensor([i/1000 for i in range(in_channels)])
        self.op2.running_var = torch.tensor([i/1000 for i in range(in_channels)])
        self.op3 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        self.op2.eval()
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x


def test_batchnorm_backward_eval():
    model = Model(in_channels = 16)
    cpu_tensor = torch.randn(2, 16, 1, 1)
    cuda_tensor = cpu_tensor.cuda()
    cpu_tensor.requires_grad = True
    cuda_tensor.requires_grad = True

    for i in range(1):
        out = model(cpu_tensor)
        loss = out.sum()
        loss.backward()
        cpu_grad_list = []
        for _, module in model.named_parameters():
            cpu_grad_list.append(module.grad)
            module.grad = None

        model = model.cuda()
        out = model(cuda_tensor)
        loss = out.sum()
        loss.backward()
        cuda_grad_list = []
        for _, module in model.named_parameters():
            cuda_grad_list.append(module.grad.cpu())

        cpu_grad = cpu_tensor.grad
        cuda_grad = cuda_tensor.grad
        rtol = 1e-5
        atol = 1e-8

        assert np.allclose(cpu_grad.numpy(), cuda_grad.cpu().numpy(), rtol, atol, True)
        print("np allclose success\n", cpu_grad, "\n", cuda_grad)
        for cpu_grad, cuda_grad in zip(cpu_grad_list, cuda_grad_list):
            assert np.allclose(cpu_grad.numpy(), cuda_grad.cpu().numpy(), rtol, atol, True)
            print("np allclose success\n", cpu_grad, "\n", cuda_grad)


if __name__ == "__main__":
    test_batchnorm_backward_eval()
