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

def testDevice1():
    import torch_dipu
    from torch_dipu import dipu
    ret1 = dipu.device_count()
    ret1 = dipu.current_device()
    ret1 = dipu.set_device("cuda:2")
    ret1 = dipu.current_device()
    print(ret1)


def testDevice2():
    import torch_dipu
    from torch_dipu import dipu

    device="cuda"
    in0 = torch.range(1, 12).reshape(1, 3, 2, 2)
    in1 =in0.to(device)
    print(in1)

    with dipu.device(torch.device(2)) as dev1:
        in1 = in0.to(device)
        print(torch.sum(in1))
        dipu.synchronize()
        ret1 = dipu.current_device()
        print(ret1)
        # need implment device guard in op
        # in1 = torch.range(1, 12).reshape(1, 3, 2, 2).to("cuda:0")
        # print(in1)
    in1 = torch.range(1, 12).reshape(1, 3, 2, 2).to(device)
    print(in1)


def testStream():
    import torch_dipu
    from torch_dipu import dipu
    st0 = dipu.Stream(2)
    res1 = st0.priority_range()
    res1 = st0.priority
    res1 = st0.synchronize()
    res1 = st0.query()
    res1 = st0.stream_id
    res1 = st0.dipu_stream
    res1 = st0.device_index
    res1 = st0.device_type
    res1 = st0.device
    res1 = st0._as_parameter_
    dipu.set_stream(st0)
    st1 = dipu.default_stream()
    res2 = st1.query()
    res2 = st1._as_parameter_
    st2 = dipu.current_stream(2)

    with dipu.StreamContext(st2) as s: 
        print("in cur ctx")

    # st1 = dipu.current_stream()
    print(st1)

def testevent():
    import torch_dipu
    from torch_dipu import dipu
    st0 = dipu.Stream(2)
    ev1 = dipu.Event()
    ev1.record(st0)
    time.sleep(1)
    ev2 = dipu.Event()
    ev1.synchronize()
    ev1.wait(st0)
    ev2.record(st0)
    elapsed = ev1.elapsed_time(ev2)
    print(elapsed)



if __name__ == '__main__':
    for i in range(1, 2):
        empty1()
        testdevice()
        testDevice1()
        testDevice2()
        testStream()
        testevent()