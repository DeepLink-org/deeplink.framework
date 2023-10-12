# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestIndex(TestCase):
    def test_index_multidim(self):
        x = torch.randn(3, 4, 5, 6).cuda()
        index = torch.arange(0, 3, 1).cuda()
        self.assertEqual(x[index, :, :, :].cpu(), x.cpu()[index.cpu(), :, :, :])
        self.assertEqual(
            x[index, index, :, :].cpu(),
            x.cpu()[index.cpu(), index.cpu(), :, :],
        )
        self.assertEqual(
            x[index, index, index, :].cpu(),
            x.cpu()[index.cpu(), index.cpu(), index.cpu(), :],
        )
        self.assertEqual(
            x[index, index, index, index].cpu(),
            x.cpu()[index.cpu(), index.cpu(), index.cpu(), index.cpu()],
        )

    def test_index_bool(self):
        input = torch.arange(16).reshape(4, 2, 2).cuda()
        index1 = torch.randint(3, (1,)).cuda()
        index2 = torch.randint(2, (1,)).cuda()
        index3 = torch.tensor([False, True]).cuda()
        self.assertEqual(input[index1].cpu(), input.cpu()[index1.cpu()])
        self.assertEqual(
            input[index1, index2].cpu(),
            input.cpu()[index1.cpu(), index2.cpu()],
        )
        self.assertEqual(
            input[..., index2, ...].cpu(),
            input.cpu()[..., index2.cpu(), ...],
        )

        self.assertEqual(
            input[index1, index2, index3].cpu(),
            input.cpu()[index1.cpu(), index2.cpu(), index3.cpu()],
        )
        self.assertEqual(
            input[index1, index2, ...].cpu(),
            input.cpu()[index1.cpu(), index2.cpu(), ...],
        )
        self.assertEqual(
            input[index1, ..., index3].cpu(),
            input.cpu()[index1.cpu(), ..., index3.cpu()],
        )

    def test_index_empty_index(self):
        input = torch.arange(16).reshape(4, 2, 2).cuda()
        idx = torch.tensor([], dtype=torch.long).reshape(0, 3).cuda()
        idx1 = torch.tensor([], dtype=torch.long).reshape(0, 1).cuda()
        self.assertEqual(input[idx].cpu(), input.cpu()[idx.cpu()])
        self.assertEqual(input[idx, idx1].cpu(), input.cpu()[idx.cpu(), idx1.cpu()])

    def test_index_empty_input(self):
        input = torch.tensor([]).cuda()
        idx = torch.tensor([], dtype=torch.bool).cuda()
        self.assertEqual(input[idx].cpu(), input.cpu()[idx.cpu()])

    def test_index_select(self):
        device = torch.device("dipu")
        x = torch.randn(3, 4)
        indices = torch.tensor([0, 2])
        self.assertEqual(
            torch.index_select(x, 0, indices),
            torch.index_select(x.to(device), 0, indices.to(device)).cpu(),
        )
        self.assertEqual(
            torch.index_select(x, 1, indices),
            torch.index_select(x.to(device), 1, indices.to(device)).cpu(),
        )

    def test_index_put(self):
        x = torch.randn(3, 4, 5).cuda()
        index = torch.arange(0, 3, 1).cuda()

        x[index] = 1
        self.assertEqual(x.cpu(), torch.ones_like(x.cpu()))

        x[index] = 0
        self.assertEqual(x.cpu(), torch.zeros_like(x.cpu()))

        for shape in [(5, 4, 2, 3), (3, 4, 5), (2, 3), (10,)]:
            input = torch.randn(shape).clamp(min=-3, max=3) * 100
            for numel in range(1, input.numel()):
                indices_cpu = []
                indices_device = []
                for _ in range(len(shape)):
                    indice = torch.randint(0, min(shape), (numel,))
                    indices_cpu.append(indice)
                    indices_device.append(indice.cuda())
                values = torch.randn(numel) * 100
                y_cpu = torch.index_put(
                    input.clone(), indices_cpu, values, accumulate=True
                )
                y_device = torch.index_put(
                    input.cuda(), indices_device, values.cuda(), accumulate=True
                )
                self.assertRtolEqual(y_cpu, y_device.cpu())


if __name__ == "__main__":
    run_tests()
