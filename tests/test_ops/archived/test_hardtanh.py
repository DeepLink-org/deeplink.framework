import torch
import torch.nn.functional as F
import torch_dipu

def test_hardtanh(input, min_val=-1., max_val=1., inplace=False):
    cpu_input = input.cpu()
    device_input = input.cuda()
    if not inplace:
        device_input.requires_grad_(True)
        cpu_input.requires_grad_(True)
    cpu_output = F.hardtanh(cpu_input, min_val, max_val, inplace)
    device_output = F.hardtanh(device_input, min_val, max_val, inplace)


    print(f"device_output = {device_output.cpu()}")
    print(f"cpu_output = {cpu_output}")

    assert torch.allclose(device_output.cpu(), cpu_output)

    if not inplace:
        cpu_output.backward(torch.ones_like(cpu_output))
        device_output.backward(torch.ones_like(device_output))
        print("grad:", device_input.grad.cpu())
        assert torch.allclose(device_input.grad.cpu(), cpu_input.grad)


test_hardtanh(torch.randn(3, 5))
test_hardtanh(torch.randn(34, 51))
test_hardtanh(torch.randn(3, 5), min_val=-3.0)
test_hardtanh(torch.randn(3, 5), max_val=3.0)
test_hardtanh(torch.randn(3, 5) * 20, min_val=-10, max_val=17.0)
test_hardtanh(torch.randn(3, 5) * 20, min_val=-10, max_val=17.0, inplace = True)
