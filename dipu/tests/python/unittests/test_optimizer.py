# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torchvision.models as models
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestOptimizers(TestCase):
    def test_optimizers(self):
        """won't fail, only detecting errors"""
        model = models.resnet18().cuda()
        input = torch.randn(2, 3, 224, 224).cuda()
        target = torch.zeros(2, 1000).cuda()

        optimizers = [
            torch.optim.Adadelta(model.parameters(), lr=0.01),
            torch.optim.Adadelta(model.parameters(), 0.01, 0.9, 1e-6, 0, False),
            torch.optim.Adadelta(model.parameters(), 0.01, 0.9, 1e-6, 0, True),
            torch.optim.Adagrad(model.parameters(), lr=0.01),
            torch.optim.Adam(model.parameters(), lr=0.01),
            torch.optim.AdamW(model.parameters(), lr=0.01),
            torch.optim.Adamax(model.parameters(), lr=0.01),
            torch.optim.ASGD(model.parameters(), lr=0.01),
            torch.optim.NAdam(model.parameters(), lr=0.01),
            torch.optim.RAdam(model.parameters(), lr=0.01),
            torch.optim.RMSprop(model.parameters(), lr=0.01),
            torch.optim.Rprop(model.parameters(), lr=0.01),
            torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
            torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False),
            torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True),
        ]
        loss_fn = torch.nn.MSELoss()

        for _ in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad()

            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()

            for optimizer in optimizers:
                optimizer.step()


if __name__ == "__main__":
    run_tests()
