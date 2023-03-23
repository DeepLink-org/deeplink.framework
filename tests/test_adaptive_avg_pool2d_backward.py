import torch
import torch.nn.functional as F
import torch_dipu

def test(x, devicestr : str):
    device = torch.device(devicestr)
    x = x.to(device)
    x.requires_grad_(True)
    y = F.adaptive_avg_pool2d(x, (2, 2))
    print(f" y = {y}")

    loss_fn = torch.nn.MSELoss().to(device)
    target = torch.zeros_like(y)
    loss = loss_fn(y, target)
    loss.backward()
    print(f"x.grad = {x.grad}")


x = torch.randn(1, 3, 4, 4)
test(x, 'dipu')
test(x, 'cpu')