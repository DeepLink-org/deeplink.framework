# Copyright (c) 2023, DeepLink.
import unittest
import torch
import torch_dipu

class TestSchema(unittest.TestCase):

    def test_mse_loss_backward(self):
        loss = torch.nn.MSELoss()
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)
        output = loss(input, target)
        output.backward()

        input2 = input.detach().clone().cuda()
        input2.requires_grad = True
        target2 = target.clone().cuda()
        output2 = loss(input2, target2)
        output2.backward()

        assert torch.allclose(input.grad, input2.grad.cpu(), atol = 1e-3)


if __name__ == '__main__':
    unittest.main()