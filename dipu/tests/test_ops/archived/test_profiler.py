import torch_dipu
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    output = model(inputs)
    output.sum().backward()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=1000))
