# Copyright (c) 2023, DeepLink.
import io
from stdout_redirector import stdout_redirector
from local_eviron import local_eviron


def _test_copy_dumparg():
    captured = io.BytesIO()
    with stdout_redirector(captured):
        with local_eviron(
            {
                "DIPU_DUMP_OP_ARGS": "2",
            }
        ):
            import torch
            import torch_dipu

            source_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            target_tensor = torch.zeros_like(source_tensor).cuda()
            target_tensor.copy_(source_tensor)

    output = captured.getvalue().decode()
    print(output)
    assert "DIPUCopyInplace.run" in output
    assert "numel: 3, sizes: [3], stride: [1], is_view: 0, dtype: float" in output


if __name__ == "__main__":
    _test_copy_dumparg()
