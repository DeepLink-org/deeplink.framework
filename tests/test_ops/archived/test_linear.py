import torch
import torch_dipu
import torch.nn as nn

def test_linear(devicestr : str):
    device = torch.device(devicestr)
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=torch.float)
    x = x.to(device)
    linear_layer = nn.Linear(3, 2).to(device)
    linear_layer.weight = nn.Parameter(torch.ones_like(linear_layer.weight))
    linear_layer.bias = nn.Parameter(torch.ones_like(linear_layer.bias))
    y_pred = linear_layer(x)
    print(f"y_pred = \n{y_pred}")

    y_true = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float).to(device)
    loss_fn = nn.MSELoss().to(device)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    print(f"linear_layer.weight.grad = \n{linear_layer.weight.grad}")


test_linear("dipu")
test_linear("cpu")