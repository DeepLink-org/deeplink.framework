import torch
import torch_dipu
import torch.nn as nn

def test_linear(x, label, devicestr : str):
    device = torch.device(devicestr)
    x = x.to(device)
    linear_layer = nn.Linear(3, 2).to(device)
    linear_layer.weight = nn.Parameter(torch.ones_like(linear_layer.weight))
    linear_layer.bias = nn.Parameter(torch.ones_like(linear_layer.bias))
    y_pred = linear_layer(x)
    print(f"y_pred = \n{y_pred}")

    label = label.to(device)
    loss_fn = nn.MSELoss().to(device)
    loss = loss_fn(y_pred, label)
    loss.backward()
    print(f"linear_layer.weight.grad = \n{linear_layer.weight.grad}")


# 2D tensor
x = torch.arange(9, dtype=torch.float).reshape(3, 3)
label = torch.randn(3, 2)
test_linear(x, label, "dipu")
test_linear(x, label, "cpu")

# 3D tensor
x = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
label = torch.randn(2, 2, 2)
test_linear(x, label, "dipu")
test_linear(x, label, "cpu")

# 4D tensor
x = torch.arange(24, dtype=torch.float).reshape(2, 2, 2, 3)
label = torch.randn(2, 2, 2, 2)
test_linear(x, label, "dipu")
test_linear(x, label, "cpu")