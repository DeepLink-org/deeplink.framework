# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def test_max_pool2d_with_indices(x, devicestr : str):
    device = torch.device(devicestr)
    x = x.to(device)
    x.requires_grad_(True)
    kernel_size = (2, 2)
    stride = (2, 2)
    out, indices = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, return_indices=True)
    #print(f"out = {out}\nindices = {indices}")

    grad_output = torch.ones_like(out).to(device)
    out.backward(grad_output)
    #print(f"x.grad = {x.grad}")

def test_max_pool2d(x, devicestr : str):
    device = torch.device(devicestr)
    x = x.to(device)
    x.requires_grad_(True)
    kernel_size = (2, 2)
    stride = (2, 2)
    out = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(x)
    #print(f"out = {out}")

    grad_output = torch.ones_like(out).to(device)
    out.backward(grad_output)
    #print(f"x.grad = {x.grad}")


x = torch.randn((1, 3, 4, 4))
test_max_pool2d_with_indices(x, "dipu")
test_max_pool2d_with_indices(x, "cpu")
test_max_pool2d(x, "dipu")
test_max_pool2d(x, "cpu")