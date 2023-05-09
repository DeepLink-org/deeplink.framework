import torch
import torch.nn as nn
import torch_dipu


def test_dropout(input, p):
    cpu_input = input.cpu().clone()
    cpu_input.requires_grad_(True)
    cpu_out = torch.nn.functional.dropout(cpu_input,p=p)
    cpu_out.backward(torch.ones_like(cpu_input))

    print('cpu_out:',cpu_out)
    print('cpu_grad:', cpu_input.grad)


    device_input = input.cuda()
    device_input.requires_grad_(True)
    device_out = torch.nn.functional.dropout(device_input,p=p)
    device_out.backward(torch.ones_like(device_input))

    print('device_out:', device_out)
    print('device_grad:',device_input.grad)

    print("inplace")
    cpu_input = input.cpu().clone()
    cpu_out = torch.nn.functional.dropout(cpu_input,p=p, inplace = True)

    print('cpu_out:',cpu_out)


    device_input = input.clone().cuda()
    device_out = torch.nn.functional.dropout(device_input,p=p, inplace = True)

    print('device_out:', device_out)


test_dropout(torch.randn(4, 5), 0.8)