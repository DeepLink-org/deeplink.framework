# Copyright (c) 2023, DeepLink.
import io
from torch_dipu.testing._internal.stdout_redirector import stdout_redirector
from torch_dipu.testing._internal.local_eviron import local_eviron


def _test_dipu_fallback():
    captured = io.BytesIO()
    with stdout_redirector(captured):
        with local_eviron(
            {
                "DIPU_FORCE_FALLBACK_OPS_LIST": "add.out,sub.out",
                "DIPU_DUMP_OP_ARGS": "1",
            }
        ):
            import torch
            import torch_dipu

            x = torch.randn(3, 4).cuda()
            _ = x + x
            _ = x - x

    output = captured.getvalue().decode()
    assert "force fallback has been set, add.out will be fallback to cpu" in output
    assert "force fallback has been set, sub.out will be fallback to cpu" in output
    assert "dipu_fallback" in output
    assert "diopiAdd" not in output
    assert "diopiSub" not in output


if __name__ == "__main__":
    _test_dipu_fallback()
