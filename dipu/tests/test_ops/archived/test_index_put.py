import torch
import torch_dipu

x = torch.randn(3,4,5).cuda()
index = torch.arange(0, 3, 1).cuda()

x[index] = 1
assert torch.allclose(x.cpu(), torch.ones_like(x.cpu()))

x[index] = 0
assert torch.allclose(x.cpu(), torch.zeros_like(x.cpu()))


for shape in [(5, 4, 2, 3), (3, 4, 5), (2, 3), (10,)]:
    input =  torch.randn(shape).clamp(min = -3, max = 3) * 100
    for numel in range(1, input.numel()):
        indices_cpu = []
        indices_device = []
        for i in range(len(shape)):
            indice = torch.randint(0, min(shape), (numel,))
            indices_cpu.append(indice)
            indices_device.append(indice.cuda())
            values = torch.randn(numel) * 100
        y_cpu = torch.index_put(input.clone(), indices_cpu, values, accumulate = True)
        y_device = torch.index_put(input.cuda(), indices_device, values.cuda(), accumulate = True).cpu()

        assert torch.allclose(y_cpu, y_device.cpu(), atol = 1e-3, rtol = 1e-3), str((y_cpu - y_device).abs().max())