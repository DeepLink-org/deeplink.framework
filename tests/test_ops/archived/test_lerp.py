import torch
device = "cuda"
start = torch.arange(1., 5.)
end = torch.empty(4).fill_(10)
torch_result1 = torch.lerp(start, end, 0.5)
torch_result2 = torch.lerp(start, end, torch.full_like(start, 0.5))
torch_result_cuda1 = torch.lerp(start.to(device) , end.to(device), 0.5)
torch_result_cuda2 = torch.lerp(start.to(device) , end.to(device), torch.full_like(start, 0.5).to(device))


import torch_dipu
device = "dipu"
start_dipu = torch.arange(1., 5.)
end_dipu = torch.empty(4).fill_(10)
dipu_result1 = torch.lerp(start_dipu , end_dipu, 0.5)
dipu_result2 = torch.lerp(start_dipu , end_dipu, torch.full_like(start, 0.5))
dipu_result_cuda1 =  torch.lerp(start_dipu.to(device) , end_dipu.to(device), 0.5)
dipu_result_cuda2 =  torch.lerp(start_dipu.to(device) , end_dipu.to(device), torch.full_like(start, 0.5).to(device))


assert torch.equal(torch_result1, dipu_result1), "结果不相等"
assert torch.equal(torch_result1, torch_result_cuda1.to("cpu")), "结果不相等"
assert torch.equal(torch_result1, dipu_result_cuda1.to("cpu")), "结果不相等"
assert torch.equal(torch_result2, dipu_result2), "结果不相等"
assert torch.equal(torch_result2, torch_result_cuda2.to("cpu")), "结果不相等"
assert torch.equal(torch_result2, dipu_result_cuda2.to("cpu")), "结果不相等"

