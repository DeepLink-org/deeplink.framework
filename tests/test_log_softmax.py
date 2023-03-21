import torch
import torch_dipu

def test_log_softmax(x, devicestr : str):
    device = torch.device(devicestr)
    x = x.to(device)
    x.requires_grad_(True)
    log_softmax_x = torch.nn.functional.log_softmax(x, dim=1)
    print(log_softmax_x)

    grad_output = torch.randn(2, 3)
    log_softmax_x.backward(grad_output)
    print(x.grad)


x = torch.randn(2, 3)
test_log_softmax(x, "dipu")
test_log_softmax(x, "cpu")