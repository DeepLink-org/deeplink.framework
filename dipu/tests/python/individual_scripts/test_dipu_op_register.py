# Copyright (c) 2023, DeepLink.
from multiprocessing import Process, set_start_method
from tests.python.utils.local_eviron import local_eviron


def _test_op_register(mode):
    with local_eviron(
        {"DIPU_IMMEDIATE_REGISTER_OP": str(mode), "DIPU_DUMP_OP_ARGS": "1"}
    ):
        import torch
        import torch_dipu

        x = torch.randn(3, 4).cuda()
        _ = x + x


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    p1 = Process(
        target=_test_op_register,
        args=(0,),
    )
    p1.start()
    p1.join()

    p2 = Process(
        target=_test_op_register,
        args=(1,),
    )
    p2.start()
    p2.join()

    p3 = Process(
        target=_test_op_register,
        args=("",),
    )
    p3.start()
    p3.join()

    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert p3.exitcode == 0
