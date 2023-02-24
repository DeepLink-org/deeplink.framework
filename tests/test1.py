import os
import time
import torch
from torch import nn
# from torch import autograd
# import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


def debugat():
    # rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    rank = 0
    if rank == 0:
        import os
        import ptvsd
        import socket
        pid1 = os.getpid()

        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(hostname, ip, flush=True)
        host = ip # or "localhost"
        host = "127.0.0.1"
        port = 12345
        print("cwd is:",  os.getcwd(), flush=True)
        ptvsd.enable_attach(address=(host, port), redirect_output=False)
        print("-------------------------print rank,:", rank, "pid1:", pid1, flush=True)
        ptvsd.wait_for_attach()

debugat()

def maxpool2():
    device= torch.device("cpu")
    from  torch.nn.functional import max_pool2d
    input = torch.randn(2, 6, 6, 6).to(device)
    input.requires_grad = True
    m = nn.MaxPool2d((2, 2), stride=(1, 1), return_indices = True).to(device)
    output = m(input)
    asgrad = torch.ones_like(output[0])
    output[0].backward(asgrad)
    print(input.grad)

def eyet1():
    t1 = torch.eye(3)
    print("card num:", t1)

def empty0():
    # input1 = data1.to(device)
    #  maskindex = torch.isfinite(input1)
    #  maskindex = input1.isfinite()
    input2 = torch.empty(2, 4)
    res1 = input2.add(input2)
    res2 = torch.reshape(res1, (-1,))
    print(res2)

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

  # input2 = torch.randn(2, 6, 6, 6).to(device)


def testcpu1():
    t0 = torch.ones((1), dtype=torch.float)
    t01 = int(t0)

    t1 = torch.ones((2,), dtype=torch.float)
    t11 = t1.char()
    torch._cast_Byte(t1)
    data = [[0, 1], [2, 3]]
    t2 = t1.new_tensor(data)
    t3 = t1 + t2

    import torch_dipu
    device = torch.device("dipu")

def testselect1():
    data = [[0, 1], [2, 3]]
    t1 = torch.tensor(data, dtype=torch.float)
    ts1 = t1.select(1, 0)
    print(ts1)

if __name__ == '__main__':
    for i in range(1, 4):
        # maxpool2()
        # testcpu1()
        testselect1()