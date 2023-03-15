import torch
import torch_dipu

def test_linear(devicestr : str):
    device = torch.device(devicestr)
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6], 
                      [7, 8, 9]], dtype=torch.float)
    x = x.to(device)
    linear_layer = torch.nn.Linear(3, 2).to(device)
    y_pred = linear_layer(x)
    print(f"y_pred = \n{y_pred}")
    
    y_true = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float)
    loss_fn = torch.nn.MSELoss().to(device)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    print(f"linear_layer.weight.grad = \n{linear_layer.weight.grad}")


test_linear("dipu")
test_linear("cpu")