import os
from multiprocessing import Process

def test_op_register(mode):
    os.environ['DIPU_IMMEDIATE_REGISTER_OP'] = str(mode)
    os.environ['DIPU_DUMP_OP_ARGS'] = str(1)
    import torch
    import torch_dipu

    x = torch.randn(3,4).cuda()
    y = x + x


if __name__=='__main__':
    p1 = Process(target = test_op_register, args = (0, ),)
    p1.start()
    p1.join()

    p2 = Process(target = test_op_register, args = (1, ),)
    p2.start()
    p2.join()

    p3 = Process(target = test_op_register, args = ('',),)
    p3.start()
    p3.join()

    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert p3.exitcode == 0


