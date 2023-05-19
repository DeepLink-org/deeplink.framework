# Copyright (c) 2023, DeepLink.
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
    from torch import cuda
    st0 = cuda.Stream(2)
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
    cuda.set_stream(st0)
    st1 = cuda.default_stream()
    res2 = st1.query()
    res2 = st1._as_parameter_
    st2 = cuda.current_stream(2)

    with cuda.StreamContext(st2) as s: 
        print("in cur ctx")

    # st1 = dipu.current_stream()
    print(st1)

def testevent():
    import torch_dipu
    from torch import cuda

    st0 = cuda.Stream(2)
    ev1 = cuda.Event()
    ev1.record(st0)
    time.sleep(1)
    ev2 = cuda.Event()
    ev1.synchronize()
    ev1.wait(st0)
    ev2.record(st0)
    # sync before call elapse
    ev2.synchronize()
    elapsed = ev1.elapsed_time(ev2)
    print(elapsed)

def testDeviceProperties():
    print("device properties: ", torch.cuda.get_device_properties(1))
    print("device capability: ", torch.cuda.get_device_capability(1))
    print("device name: ", torch.cuda.get_device_name(1))

def test_type():
    import torch_dipu
    dev1 = "cuda"
    template = torch.arange(1, 12, dtype=torch.float32)
    s1 = torch.arange(1, 8, dtype=torch.float32, device=dev1)
    s2 = s1.new(template)
    s3 = s2.new((4, 3))
    s4 = s3.new(size = (4, 3))

    res = isinstance(s4, torch.cuda.FloatTensor)
    assert(res == True)

def test_device_copy():
    import torch_dipu
    dev2 = "cuda:2"
    t1 = torch.randn((2, 2), dtype=torch.float64, device=dev2)
    t1 = torch.zero_(t1)

    tsrc = torch.ones((2, 2), dtype=torch.float64, device = "cuda")
    print(tsrc)

    t1.copy_(tsrc)
    # cpu fallback func not support device guard now! (enhance?)
    print(t1.cpu())

    tc1 = torch.randn((2, 2), dtype=torch.float64)
    tc1.copy_(t1)
    print(tc1)
    
    t0 = torch.tensor([980], dtype=torch.int64).cuda()
    t2 = t0.expand(2)
    t2.to(torch.float)

def test_complex_type():
    import torch_dipu
    from torch_dipu import dipu
    import numpy as np
    dev2 = "cuda:2"

    # manually set device 
    dipu.set_device(2)
    abs = torch.tensor((1, 2), dtype=torch.float64, device=dev2)
    angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64, device=dev2)
    z1 = torch.polar(abs, angle)
    z2 = torch.polar(abs, angle)
    z3 = z1 + z2 
    print(z3)

    dipu.set_device(0)
    # test device change and view on different device.
    zr = torch.view_as_real(z3)
    print(zr.cpu)

if __name__ == '__main__':
    for i in range(1, 2):
        empty1()
        testdevice()
        testDevice1()
        testDevice2()
        test_device_copy()
        testDeviceProperties()
        testStream()
        testevent()
        test_complex_type()