import torch
import torch_dipu


start = torch.arange(1., 5.)
end = torch.empty(4).fill_(10)


device = "dipu"
start_dipu = torch.arange(1., 5.)
end_dipu = torch.empty(4).fill_(10)
cpu_result1 = torch.lerp(start_dipu , end_dipu, 0.5)
cpu_result2 = torch.lerp(start_dipu , end_dipu, torch.full_like(start, 0.5))
dipu_result1 =  torch.lerp(start_dipu.to(device) , end_dipu.to(device), 0.5)
dipu_result2 =  torch.lerp(start_dipu.to(device) , end_dipu.to(device), torch.full_like(start, 0.5).to(device))

assert torch.equal(dipu_result1, cpu_result1), "lerp is wrong"
assert torch.equal(dipu_result2, cpu_result2), "lerp is wrong"

