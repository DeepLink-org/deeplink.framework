import torch
import torch.nn as nn
import numpy as np
import torch_dipu

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.op1 = nn.Embedding(10, 3)

    def forward(self, x):
        x = self.op1(x)
        return x


def test_embedding_backward_eval():
    model = Model()
    cpu_tensor = input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    device = torch.device('dipu')
    dipu_tensor = cpu_tensor.to(device)

    out = model(cpu_tensor)
    loss = out.sum()
    loss.backward()
    cpu_grad_list = []
    for _, module in model.named_parameters():
        cpu_grad_list.append(module.grad)
        module.grad = None

    model = model.to(device)
    out = model(dipu_tensor)
    loss = out.sum()
    loss.backward()
    dipu_grad_list = []
    for _, module in model.named_parameters():
        dipu_grad_list.append(module.grad.cpu())

    cpu_grad = cpu_tensor.grad
    dipu_grad = dipu_tensor.grad
    rtol = 1e-5
    atol = 1e-5
    for cpu_grad, dipu_grad in zip(cpu_grad_list, dipu_grad_list):
        assert np.allclose(cpu_grad.numpy(), dipu_grad.cpu().numpy(), rtol, atol, True)
if __name__ == "__main__":
    test_embedding_backward_eval()