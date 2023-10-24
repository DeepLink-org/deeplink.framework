import torch_dipu
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_modules=True,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
) as prof:
    output = model(inputs)
    output.sum().backward()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=1000))
print(prof.key_averages(group_by_stack_n=15).table(sort_by="cuda_time_total", row_limit=1000))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=1000))

prof.export_chrome_trace("./dipu_resnet18_profiler.json")
