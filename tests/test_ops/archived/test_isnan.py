import torch_dipu
import torch

input_tensor = torch.tensor([1.0, float('nan'), 2.0 , 4])
output_tensor = torch.isnan(input_tensor)
expected_output = torch.tensor([False, True, False, False])

assert torch.equal(output_tensor, expected_output),"isnan算子计算不正确"