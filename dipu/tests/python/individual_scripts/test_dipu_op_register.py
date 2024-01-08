# Copyright (c) 2023, DeepLink.
import itertools
from typing import Union
from utils.local_eviron import local_eviron
from utils.test_in_subprocess import run_individual_test_cases


def _test_op_register(mode: Union[int, str]) -> None:
    with local_eviron(
        {"DIPU_IMMEDIATE_REGISTER_OP": str(mode), "DIPU_DUMP_OP_ARGS": "1"}
    ):
        import torch
        import torch_dipu

        x = torch.randn(3, 4).cuda()
        _ = x + x


if __name__ == "__main__":
    run_individual_test_cases(
        itertools.product(
            (_test_op_register,),
            (
                {"args": (0,)},
                {"args": (1,)},
                {"args": ("",)},
            ),
        ),
        in_parallel=True,
    )
