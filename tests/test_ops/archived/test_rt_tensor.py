import os
import time
import torch
from torch import nn

import numpy as np


def empty1():
    import torch_dipu
    # dev1 = torch_dipu.diputype
    # device = torch.device(dev1)

    device= torch.device("dipu")
    data1 = torch.empty(2, 4)
    input1 = data1.to(device)

    input2 = torch.empty(2, 4).to(device)
    res1 = input1.add(input2)
    res2 = torch.reshape(res1, (-1,))
    res3 = res2.to("cpu")
    print(res3)


def testdevice():
    import torch_dipu
    device= torch.device(0)
    device= torch.device("cuda:0")
    device= torch.device("cuda", index=0)
    device= torch.device("cuda", 0)
    # device= torch.device(0)
    device= torch.device(index=0, type="cuda")
    ret1 = isinstance(device, torch.device)
    input0 = torch.ones((2, 4))
    input0.to(0)
    input1 = torch.ones((2, 4), device=device)
    print(input1)

if __name__ == '__main__':
    for i in range(1, 2):
        empty1()