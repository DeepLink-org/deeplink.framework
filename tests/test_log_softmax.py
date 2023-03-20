import torch
# import torch_dipu

def test_log_softmax(devicestr : str):
    device = torch.device(devicestr)
    x = torch.randn(2, 3, requires_grad=True)
    log_softmax_x = torch.nn.functional.log_softmax(x, dim=1)
    print(log_softmax_x)

    grad_output = torch.randn(2, 3)
    log_softmax_x.backward(grad_output)
    print(x.grad)


# test_log_softmax("dipu")
test_log_softmax("cpu")