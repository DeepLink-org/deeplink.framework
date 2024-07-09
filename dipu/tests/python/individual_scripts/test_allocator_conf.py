import io
import itertools
from utils.test_in_subprocess import run_individual_test_cases
from utils.stdout_redirector import stdout_redirector
from utils.local_eviron import local_eviron


def test_set_allocator_settings(allocator: str):
    """currently only detecting errors"""
    with local_eviron(
        {
            "DIPU_DEVICE_MEMCACHING_ALGORITHM": allocator,
            "DIPU_HOST_MEMCACHING_ALGORITHM": allocator,
        }
    ):
        import torch
        import torch_dipu

        captured = io.BytesIO()
        with stdout_redirector(captured):
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        captured_output = captured.getvalue().decode("utf-8")

        is_torch_allocator = allocator == "TORCH"
        failed = (
            "Not using torch allocator, skipping setAllocatorSettings"
            in captured_output
        )
        assert is_torch_allocator == (not failed)


if __name__ == "__main__":
    run_individual_test_cases(
        itertools.product(
            (test_set_allocator_settings,),
            ({"args": (allocator,)} for allocator in ("TORCH", "BF", "BS", "RAW")),
        )
    )
