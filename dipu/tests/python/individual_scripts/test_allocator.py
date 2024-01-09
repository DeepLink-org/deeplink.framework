import itertools
import os
from utils.test_in_subprocess import run_individual_test_cases


def test_allocator(
    max_allocate: int,
    step: int,
    algorithm: str,
    log_mask: int,
    test_pin_memory: bool = True,
):
    os.environ["DIPU_DEVICE_MEMCACHING_ALGORITHM"] = algorithm
    os.environ["DIPU_DEBUG_ALLOCATOR"] = str(log_mask)
    os.environ["DIPU_MEM_CHECK"] = "1"
    os.environ["DIPU_MEM_CHECK_LOG_INTERVAL"] = "100000"
    import torch
    import torch_dipu
    import time

    start = time.time()
    index = 0
    for nbytes in range(0, max_allocate, step):
        if nbytes % 1024 == 0:
            print(f"allocate {nbytes} nbytes use {algorithm}")
        raw = torch.empty(size=(nbytes,), dtype=torch.uint8, device="dipu")
        assert raw.numel() == nbytes

        x1 = raw.to("dipu")
        assert x1.device.index == index
        assert x1.numel() == nbytes
    end = time.time()
    print(
        f"allocate [0, {max_allocate}) bytes on device {index} use {algorithm} success, elasped: {end - start} seconds"
    )

    index = torch.cuda.device_count() - 1
    torch.cuda.set_device(index)
    start = time.time()
    for nbytes in range(0, max_allocate, step):
        if nbytes % 1024 == 0:
            print(f"allocate {nbytes} nbytes use {algorithm}")
        raw = torch.empty(size=(nbytes,), dtype=torch.uint8, device="dipu")
        assert raw.numel() == nbytes

        x1 = raw.to("dipu")
        assert x1.device.index == index
        assert x1.numel() == nbytes
    end = time.time()
    print(
        f"allocate [0, {max_allocate}) bytes on device {index} use {algorithm} success, elasped: {end - start} seconds"
    )

    if test_pin_memory == False:
        return

    start = time.time()
    for nbytes in range(0, max_allocate, step):
        if nbytes % 1024 == 0:
            print(f"allocate {nbytes} nbytes use {algorithm}")
        raw = torch.empty(size=(nbytes,), dtype=torch.uint8)
        pin = raw.pin_memory()
        if nbytes > 0:
            assert pin.is_pinned() == True
            assert raw.is_pinned() == False

        assert raw.numel() == nbytes
        assert pin.numel() == nbytes

    end = time.time()
    print(
        f"allocate [0, {max_allocate}) bytes pin memory use {algorithm} success, elasped: {end - start} seconds"
    )


if __name__ == "__main__":
    MAX_ALLOCATE = 1 << 15
    run_individual_test_cases(
        itertools.product(
            (test_allocator,),
            (
                {"args": (MAX_ALLOCATE, 1, "BF", 0)},
                {"args": (MAX_ALLOCATE, 1, "BS", 0)},
                {"args": (MAX_ALLOCATE, 1, "RAW", 0)},
                {"args": (MAX_ALLOCATE, 17919, "BF", 3, False)},
            ),
        ),
        in_parallel=False,
    )
