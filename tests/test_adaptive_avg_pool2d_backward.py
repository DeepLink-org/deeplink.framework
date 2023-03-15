import torch
import torch.nn.functional as F
import torch_dipu

def test(devicestr : str):
    device = torch.device(devicestr)
    x = torch.randn(1, 3, 4, 4).to(device)
    x.requires_grad_(True)
    y = F.adaptive_avg_pool2d(x, (2, 2))
    print(f" y = {y}")

    loss_fn = torch.nn.MSELoss().to(device)
    target = torch.zeros_like(y)
    loss = loss_fn(y, target)
    loss.backward()
    print(f"x.grad = {x.grad}")


test('dipu')
test('cpu')