import itertools
import os
from utils.test_in_subprocess import run_individual_test_cases


def test_mem_stats(algorithm: str, log_mask: int):
    os.environ["DIPU_DEVICE_MEMCACHING_ALGORITHM"] = algorithm
    os.environ["DIPU_DEBUG_ALLOCATOR"] = str(log_mask)
    print("allocator algorithm:", algorithm)
    import torch
    import torch_dipu
    import random

    ins = []
    pin_ins = []
    real_allocated = 0
    for _ in range(100):
        numel = random.randint(0, 1 << 20)
        x = torch.randn(numel).to(torch.device("cuda:0"))
        y = torch.randn(numel).pin_memory()
        ins.append(x)
        pin_ins.append(y)
        allocated = torch.cuda.memory_allocated(x.device)
        allocated_default = torch.cuda.memory_allocated()
        allocated_by_index = torch.cuda.memory_allocated(0)
        pin_allocated = torch.cuda.memory_allocated(y.device)
        reserved = torch.cuda.memory_reserved(x.device)
        real_allocated += ((numel * 4 - 1) | 511) + 1
        print(
            f"numel:{numel}, allocated:{allocated}, reserved:{reserved}, real_allocated:{real_allocated}"
        )
        assert allocated == real_allocated
        assert allocated == allocated_default
        assert allocated == allocated_by_index
        assert pin_allocated == real_allocated
        assert reserved >= allocated
        del x, y

    real_max_allocate = real_allocated

    for _ in range(len(ins)):
        numel = ins[0].numel()
        real_allocated -= ((numel * 4 - 1) | 511) + 1
        ins.pop(0)
        assert torch.cuda.memory_allocated() == real_allocated

    del pin_ins

    allocated_default = torch.cuda.memory_allocated()
    assert allocated_default == 0
    if algorithm == "RAW":
        assert torch.cuda.memory_reserved() == 0
    else:
        assert torch.cuda.memory_reserved() > 0

    torch.cuda.empty_cache()

    assert torch.cuda.memory_reserved() == 0
    assert torch.cuda.memory_allocated() == 0
    assert torch.cuda.max_memory_allocated() == real_max_allocate
    assert torch.cuda.max_memory_reserved() > 0


if __name__ == "__main__":
    run_individual_test_cases(
        itertools.product(
            (test_mem_stats,),
            (
                {"args": ("BF", 0)},
                {"args": ("BS", 0)},
                {"args": ("RAW", 0)},
            ),
        ),
        in_parallel=False,
    )
