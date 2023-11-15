# Copyright (c) 2023, DeepLink.
import numpy as np
import torch
import torch.nn as nn
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestEmbeddingBackward(TestCase):
    def test_embedding_backward_eval(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.op1 = nn.Embedding(10, 3)

            def forward(self, x):
                x = self.op1(x)
                return x

        model = Model()
        cpu_tensor = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        device = torch.device("dipu")
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
        for cpu_grad, dipu_grad in zip(cpu_grad_list, dipu_grad_list):
            self.assertTrue(
                np.allclose(
                    cpu_grad.numpy(),
                    dipu_grad.cpu().numpy(),
                    rtol=1e-5,
                    atol=1e-5,
                    equal_nan=True,
                )
            )


if __name__ == "__main__":
    run_tests()
